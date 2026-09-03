#!/usr/bin/env python3
"""Generate labelled instructions for the object-extraction stage. No scenes involved.

Extraction is a pure text problem - instruction in, objects out - so it needs no floor
plan, no room graph and no plan. That makes labelled data free: the sentence is built from
slots, so what the answer should be is known by construction rather than annotated. It also
makes it *scene-independent*, which is the point. Training on instructions derived from the
10 benchmark scenes would measure memorisation of those scenes.

Every object the task names falls into exactly one of three classes, by how much the task
says about where it is:

    uncertain   the task names it and says nothing about where it is
    stated      the task names its room       "the office bottom cabinet"
    dependent   the task names its support    "the mug on the counter"

The classes are disjoint and ordered by how much they tell us: a relation to another object
beats a room, and a room beats nothing. That ordering is what lets a location be *resolved*
rather than guessed - follow an object's relation chain to its root, and the root's room
(stated, or the RSN's guess when it is uncertain) is the room for everything hanging off it.
"the mug in the office cabinet" places the mug in the office without ever asking the RSN
about mugs.

The room line is easy to leave out and expensive to lose. An earlier version dropped it:
55% of generated instructions named a room and *none* of the targets kept it, so a model
trained on them learned to delete a fact the task had given it, and the pipeline fell back
to guessing a location it had been told.

**Vocabulary and affordances both come from the dataset, not from a hand-written list.**
The first version typed the fixtures out by hand and a third of them - `drawer`, `cupboard`,
`side_table` - were not BEHAVIOR categories at all, so the model was trained to answer with
names that can never ground. Worse for `drawer`, which is a *link* of a piece of furniture
rather than an object. Now every fixture is a real category, and one whose BDDL properties
match what the sentence asks of it: an instruction that says "closing it each time" gets an
openable, and one that says "switch it on" gets a toggleable.

    python extraction_data.py --n 200  --out data/extraction-dev.json --exclude-benchmark
    python extraction_data.py --n 8000 --out data/extraction-train.json --seed 7 \
                              --exclude-benchmark
"""

import argparse
import csv
import json
import os
import random
import re

from evaluate import same_object
from floor_world import DEFAULT_DATASET

BDDL_DATA = "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/bddl3/bddl/generated_data"

# Slot kinds. The last four are distinguished by *affordance*, not by wording: a template
# that opens something needs an openable, and one that runs something needs a toggleable.
MOVABLE, SURFACE, CONTAINER, MACHINE, HEATER, SWITCH = (
    "movable", "surface", "container", "machine", "heater", "switch")

# Name shapes, applied on top of the affordance so the sentences stay plausible. Being
# openable does not make a `car` somewhere to put the mugs away.
SURFACE_PAT = (r"(table|desk|countertop|counter|shelf|bench|stool|dresser|sofa|chair|bed|"
               r"island|cart|rack|ottoman|sideboard|vanity|nightstand|stand)$")
STORAGE_PAT = (r"(cabinet|bin|basket|crate|chest|hamper|locker|safe|cooler|backpack|"
               r"suitcase|fridge|freezer|wardrobe|closet|briefcase|box|carton|mailbox|"
               r"toolbox)$")
MACHINE_PAT = (r"(oven|microwave|dishwasher|washer|dryer|stove|toaster|blender|kettle|"
               r"cooker|maker|machine|processor|grill|fryer|cauldron)$")

# Rooms an instruction names a fixture by. Measured on the benchmark, this phrasing caused
# 65% of everything the extractor appeared to miss.
ROOMS = ["kitchen", "bathroom", "bedroom", "living room", "dining room", "office",
         "corridor", "pantry", "utility room", "playroom", "closet", "garage",
         "entryway", "storage room", "hallway", "study", "den", "basement"]

# Furniture a container gets attributed to possessively - "the desk's cabinet". The holder
# is a distractor and is *not* part of the answer, which matches the benchmark's own ground
# truth: "the bookcase's cabinet" is `bottom_cabinet`, and the bookcase only locates it.
HOLDERS = ["desk", "bookcase", "dresser", "sideboard", "vanity", "workbench", "nightstand",
           "kitchen island", "wardrobe", "bureau"]

# Two families of relation wording, because English needs both. After a verb the phrase
# reads as a source - "take the mug *out of the cabinet*". After "is" it has to read as a
# position - "the mug *is in the cabinet*", never "is out of". Same relation either way,
# which is the point: the model must not learn that INSIDE means the words "out of".
PHRASES = {
    "INSIDE":  (["out of the {x}", "from the {x}", "from inside the {x}", "in the {x}",
                 "inside the {x}", "from within the {x}"],
                ["in the {x}", "inside the {x}", "stored in the {x}"]),
    "ON_TOP":  (["off the {x}", "from the {x}", "on the {x}", "from on top of the {x}"],
                ["on the {x}", "on top of the {x}", "sitting on the {x}",
                 "resting on the {x}"]),
    "UNDER":   (["from under the {x}", "under the {x}", "from underneath the {x}"],
                ["under the {x}", "underneath the {x}", "beneath the {x}"]),
    "NEXT_TO": (["next to the {x}", "from beside the {x}", "beside the {x}"],
                ["next to the {x}", "beside the {x}", "alongside the {x}"]),
}

# Each shape: several ways of saying one task, the slots it needs, and the relations the
# wording *states* - which is what separates DEPENDENT from UNCERTAIN.
SHAPES = [
    (["take the {a} {a_from_s}, heat it in the {p}, and put it on the {t}",
      "warm the {a} {a_from_s} in the {p}, then leave it on the {t}",
      "grab the {a} {a_from_s}, heat it up in the {p} and set it down on the {t}",
      "the {a} is {a_at_s} - heat it in the {p} and serve it on the {t}",
      "please heat the {a} {a_from_s} in the {p} and put it on the {t}"],
     {"a": MOVABLE, "s": SURFACE, "p": HEATER, "t": SURFACE}, [("a", "ON_TOP", "s")]),

    (["take the {a} {a_from_c}, warm it in the {p}, and leave it on the {t}",
      "get the {a} {a_from_c}, heat it in the {p}, then put it on the {t}",
      "the {a} is {a_at_c}; warm it in the {p} and leave it on the {t}",
      "fetch the {a} {a_from_c} and heat it in the {p}, then set it on the {t}"],
     {"a": MOVABLE, "c": CONTAINER, "p": HEATER, "t": SURFACE}, [("a", "INSIDE", "c")]),

    (["put the {a} and the {b} away in the {c}, closing it each time",
      "store the {a} and the {b} in the {c}, shutting it after each one",
      "tidy the {a} and the {b} into the {c} and close it each time",
      "put away the {a} and the {b} in the {c}, closing it behind you",
      "the {a} and the {b} both belong in the {c} - close it each time"],
     {"a": MOVABLE, "b": MOVABLE, "c": CONTAINER}, []),

    (["load the {a} and the {b} into the {p} and run it",
      "put the {a} and the {b} in the {p}, then switch it on",
      "run the {p} with the {a} and the {b} inside",
      "load the {p} with the {a} and the {b} and start it"],
     {"a": MOVABLE, "b": MOVABLE, "p": MACHINE}, []),

    (["bring the {a}, the {b} and the {d} to the {t}",
      "carry the {a}, the {b} and the {d} over to the {t}",
      "move the {a}, the {b} and the {d} onto the {t}",
      "collect the {a}, the {b} and the {d} and put them on the {t}"],
     {"a": MOVABLE, "b": MOVABLE, "d": MOVABLE, "t": SURFACE}, []),

    (["take the {a} and the {b} {ab_from_c} and put them both on the {t}",
      "get the {a} and the {b} {ab_from_c}, then set them both on the {t}",
      "the {a} and the {b} are {ab_at_c} - move them both to the {t}",
      "unload the {a} and the {b} {ab_from_c} onto the {t}"],
     {"a": MOVABLE, "b": MOVABLE, "c": CONTAINER, "t": SURFACE},
     [("a", "INSIDE", "c"), ("b", "INSIDE", "c")]),

    (["carry the {a} and the {b} to the {t}, then switch the {p} on and off again",
      "move the {a} and the {b} onto the {t}, then turn the {p} on and back off",
      "bring the {a} and the {b} to the {t} and cycle the {p} on and off",
      "put the {a} and the {b} on the {t}, then run the {p} briefly and shut it off"],
     {"a": MOVABLE, "b": MOVABLE, "t": SURFACE, "p": SWITCH}, []),

    (["take the {a} {a_from_c}, put it on the {b}, then store the {b} in the {e} and shut it",
      "get the {a} {a_from_c}, set it on the {b}, and put the {b} away in the {e}, closing it",
      "the {a} is {a_at_c}; stack it on the {b} and store the {b} in the {e}, shut afterwards"],
     {"a": MOVABLE, "c": CONTAINER, "b": MOVABLE, "e": CONTAINER},
     [("a", "INSIDE", "c")]),

    (["swap the {a} {a_from_s} with the {b} {b_from_t}, using the {u} to set one down",
      "exchange the {a} {a_from_s} and the {b} {b_from_t}, parking one on the {u}",
      "the {a} is {a_at_s} and the {b} is {b_at_t} - swap them, using the {u} as space"],
     {"a": MOVABLE, "s": SURFACE, "b": MOVABLE, "t": SURFACE, "u": SURFACE},
     [("a", "ON_TOP", "s"), ("b", "ON_TOP", "t")]),

    (["wash the {a} in the {p}, then dry it in the {q}, leaving both machines off and shut",
      "run the {a} through the {p} and then the {q}, closing and switching off both",
      "clean the {a} in the {p}, dry it in the {q}, and leave both shut and off"],
     {"a": MOVABLE, "p": MACHINE, "q": MACHINE}, []),

    (["move the {a} {a_from_s} to the {t}",
      "take the {a} {a_from_s} and put it on the {t}",
      "shift the {a} {a_from_s} over to the {t}",
      "the {a} is {a_at_s} - move it to the {t}"],
     {"a": MOVABLE, "s": SURFACE, "t": SURFACE}, [("a", "ON_TOP", "s")]),

    (["put the {a} in the {c} and the {b} on the {t}",
      "the {a} goes in the {c}, and the {b} goes on the {t}",
      "place the {a} inside the {c}, then put the {b} on the {t}"],
     {"a": MOVABLE, "c": CONTAINER, "b": MOVABLE, "t": SURFACE}, []),

    (["open the {c}, take out the {a}, and leave it on the {t}, then close the {c}",
      "open the {c} and move the {a} from it onto the {t}, shutting the {c} after",
      "get the {a} out of the {c} onto the {t}, and close the {c} when you are done"],
     {"a": MOVABLE, "c": CONTAINER, "t": SURFACE}, [("a", "INSIDE", "c")]),

    (["turn the {p} on, wait, and turn it off, then put the {a} on the {t}",
      "switch the {p} on and off, then set the {a} down on the {t}",
      "run the {p} briefly, shut it off, and leave the {a} on the {t}"],
     {"p": SWITCH, "a": MOVABLE, "t": SURFACE}, []),

    # UNDER and NEXT_TO. Both are edge types the world graph carries, so a task that states
    # one should be extracted as a relation rather than flattened into "somewhere".
    (["take the {a} {a_from_s} and put it on the {t}",
      "the {a} is {a_at_s} - move it onto the {t}",
      "fetch the {a} {a_from_s}, then leave it on the {t}"],
     {"a": MOVABLE, "s": SURFACE, "t": SURFACE}, [("a", "UNDER", "s")]),

    (["take the {a} {a_from_b} and put it in the {c}, closing it after",
      "the {a} is {a_at_b}; put it away in the {c} and shut it",
      "move the {a} {a_from_b} into the {c} and close it"],
     {"a": MOVABLE, "b": MOVABLE, "c": CONTAINER}, [("a", "NEXT_TO", "b")]),

    (["put the {a} {a_from_s} on the {t}, then switch the {p} on and off",
      "the {a} is {a_at_s} - move it to the {t} and cycle the {p} on and off"],
     {"a": MOVABLE, "s": SURFACE, "t": SURFACE, "p": SWITCH},
     [("a", "NEXT_TO", "s")]),
]

# Categories that are parts of things rather than things. No instruction names them.
_PART = re.compile(r"(_shelf|_back|_side|_top|_baseboard|_door|_lid|_handle|_leg|_drawer|"
                   r"_knob|_panel|_base|_frame|_link|_piece)$|^half_")

# The building, not the things in it. The dataset ships these as categories, and without
# this the generator produced "exchange the firewood grate off the dessert stand and the
# *ceilings* from the stand" - a sentence about carrying a ceiling.
_STRUCTURAL = re.compile(r"^(ceilings?|walls?|floors?|roof|driveway|lawn|sky|ground|"
                         r"building|house)$|_wall$|_floor$|_ceiling$")

_PROPS = None


def with_property(name, dataset_root=DEFAULT_DATASET):
    """Real categories whose BDDL synset carries this property, e.g. `openable`.

    Affordances come from BEHAVIOR's own annotations rather than a hand-written set, so an
    instruction that opens something is asking about an object that opens.
    """
    global _PROPS
    if _PROPS is None:
        props = json.load(open(os.path.join(BDDL_DATA, "properties_to_synsets.json")))
        pairs = csv.DictReader(open(os.path.join(BDDL_DATA, "category_mapping.csv")))
        cat2syn = {r["category"]: r.get("synset") for r in pairs if r.get("category")}
        disk = set(os.listdir(os.path.join(dataset_root, "objects")))
        _PROPS = {p: {c for c, s in cat2syn.items()
                      if s in set(syns) and c in disk and not _PART.search(c)}
                  for p, syns in props.items()}
    return _PROPS.get(name, set())


def movable_pool(dataset_root=DEFAULT_DATASET):
    """Whole objects a robot could pick up, from the categories the dataset ships."""
    cats = os.listdir(os.path.join(dataset_root, "objects"))
    fixed = (with_property("openable", dataset_root) | with_property("toggleable", dataset_root)
             | with_property("heatSource", dataset_root))
    return sorted(c for c in cats if not _PART.search(c) and c not in fixed
                  and not _STRUCTURAL.search(c) and not re.search(SURFACE_PAT, c))


def pools(dataset_root=DEFAULT_DATASET):
    """The five slot vocabularies, each real and each matching its affordance."""
    cats = [c for c in os.listdir(os.path.join(dataset_root, "objects"))
            if not _PART.search(c)]
    openable = with_property("openable", dataset_root)
    toggleable = with_property("toggleable", dataset_root)
    heat = with_property("heatSource", dataset_root)
    return {
        MOVABLE: movable_pool(dataset_root),
        SURFACE: sorted(c for c in cats if re.search(SURFACE_PAT, c)),
        # Closed by the instruction, so it has to open.
        CONTAINER: sorted(c for c in openable if re.search(STORAGE_PAT, c)),
        # Loaded *and* run, so it has to do both.
        MACHINE: sorted(c for c in openable & toggleable if re.search(MACHINE_PAT, c)),
        HEATER: sorted(heat),
        SWITCH: sorted(c for c in toggleable if re.search(MACHINE_PAT, c)),
    }


def say(name, kind, rng, qualify, possessive):
    """Turn a category into the words an instruction would use for it.

    Returns `(words, room)`. A fixture may be named by its room - "the office bottom
    cabinet" - or by the furniture it belongs to. The room is *stated information*, not
    noise: it says where the thing is, which is otherwise something the RSN has to guess,
    so it is returned to be learned rather than dropped. The possessive holder is different
    and is not part of the answer, matching the benchmark's own ground truth.
    """
    words = name.replace("_", " ")
    if kind is MOVABLE:
        return words, None
    if kind is CONTAINER and rng.random() < possessive:
        return f"{rng.choice(HOLDERS)}'s {words}", None
    if rng.random() < qualify:
        room = rng.choice(ROOMS)
        # English states a room two ways - "the office desk" and "the desk in the office" -
        # and a generator that only ever produces the first teaches only the first. The
        # benchmark uses both, and the model trained on prefixes alone attached the room to
        # the wrong object when it met a postfix.
        if rng.random() < 0.5:
            return f"{room} {words}", room.replace(" ", "_")
        return f"{words} in the {room}", room.replace(" ", "_")
    return words, None


def generate(n, seed=0, exclude=(), dataset_root=DEFAULT_DATASET, qualify=0.35,
             possessive=0.15):
    """`n` instructions with their exact answers.

    `exclude` holds categories back, so a validation split can be built from words a model
    was never trained on - the difference between measuring generalisation and measuring
    memorisation. `qualify` and `possessive` set how often a fixture is named indirectly.
    """
    rng = random.Random(seed)
    banned = set(exclude)
    available = {k: [c for c in v if c not in banned]
                 for k, v in pools(dataset_root).items()}
    for kind, pool in available.items():
        if len(pool) < 3:
            raise SystemExit(f"--exclude left only {len(pool)} {kind} categories")

    out = []
    for _ in range(n):
        phrasings, slots, stated = rng.choice(SHAPES)
        # Every slot gets a different object, or the sentence stops making sense: "move
        # the mug from the mug to the mug". Distinct is not enough, though - two names the
        # *scorer* cannot tell apart make the label ambiguous rather than wrong, and the
        # dataset has 47 such pairs (`bar` and `chocolate_bar`, `leaf` and `bay_leaf`). A
        # sentence naming both has no single right answer, so it is never generated.
        chosen, used = {}, []
        for slot, kind in slots.items():
            for _ in range(40):
                pick = rng.choice(available[kind])
                if not any(same_object(pick, u) for u in used):
                    break
            used.append(pick)
            chosen[slot] = pick

        spoken, rooms = {}, {}
        for slot in chosen:
            words, room = say(chosen[slot], slots[slot], rng, qualify, possessive)
            spoken[slot] = words
            if room:
                rooms[chosen[slot]] = room
        for obj, rel, tgt in stated:
            source, at = PHRASES[rel]
            spoken[f"{obj}_from_{tgt}"] = rng.choice(source).format(x=spoken[tgt])
            spoken[f"{obj}_at_{tgt}"] = rng.choice(at).format(x=spoken[tgt])
        # Two objects sharing one stated container get their own slot, since the phrasing
        # names the pair once rather than each in turn.
        if stated and all(r == "INSIDE" for _, r, _ in stated) and "c" in chosen:
            source, at = PHRASES["INSIDE"]
            spoken["ab_from_c"] = rng.choice(source).format(x=spoken["c"])
            spoken["ab_at_c"] = rng.choice(at).format(x=spoken["c"])

        instruction = rng.choice(phrasings).format(**spoken)
        dependent = [{"object": chosen[o], "relation": rel, "target": chosen[t]}
                     for o, rel, t in stated]
        # Exactly one class each, most-informative first. An object whose support the task
        # gives is `dependent` even if its room were also given, because the support pins
        # it and the room only narrows it.
        placed = {d["object"] for d in dependent}
        stated_rooms = {o: r for o, r in rooms.items() if o not in placed}
        uncertain = sorted(set(chosen.values()) - placed - set(stated_rooms))
        out.append({"task": instruction,
                    "extraction": {"uncertain": uncertain,
                                   "stated": stated_rooms,
                                   "dependent": dependent}})
    return out


def benchmark_categories(path="data/tasks.json"):
    """Every category the 100-task benchmark uses, to hold back from training."""
    tasks = json.load(open(path))
    return ({s["name"] for t in tasks for s in t["spawn"]}
            | {a for t in tasks for _, a in t["plan"] if a}
            | {n for t in tasks for n in t["extraction"]["uncertain"]}
            | {d["object"] for t in tasks for d in t["extraction"]["dependent"]}
            | {d["target"] for t in tasks for d in t["extraction"]["dependent"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/extraction-dev.json")
    parser.add_argument("--exclude-benchmark", action="store_true",
                        help="hold back every category the 100-task benchmark uses")
    parser.add_argument("--shapes", help="only these shape indices, e.g. 0-6 (for a "
                                         "held-out-structure split)")
    args = parser.parse_args()

    global SHAPES
    if args.shapes:
        lo, hi = (int(x) for x in args.shapes.split("-"))
        SHAPES = SHAPES[lo:hi + 1]

    exclude = benchmark_categories() if args.exclude_benchmark else set()
    rows = generate(args.n, args.seed, exclude)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    unique = len({r["task"] for r in rows})
    print(f"wrote {len(rows)} instructions ({unique} unique) to {args.out}"
          + (f", holding back {len(exclude)} benchmark categories" if exclude else ""))
    for r in rows[:4]:
        e = r["extraction"]
        print(f"\n  {r['task']}")
        print(f"     uncertain: {e['uncertain']}")
        print(f"     stated:    {e['stated']}")
        print(f"     dependent: {[(d['object'], d['relation'], d['target']) for d in e['dependent']]}")


if __name__ == "__main__":
    raise SystemExit(main())
