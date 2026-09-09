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

    python extraction_data.py --n 200  --out data/extraction-dev.json
    python extraction_data.py --n 8000 --out data/extraction-train.json --seed 7 \
"""

import os as _os, sys as _sys
# Run from anywhere. Find the repo root by marker, put it on the import path, and make
# it the working directory - every path in this file is written 'data/...', so without
# the chdir a build invoked from inside its own folder would quietly write benchmark/data/.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, 'src')):
    _d = _os.path.dirname(_d)
_roots = [_d, _os.path.join(_d, 'omnigibson_runtime')]
_roots += [_f.path for _r in ('src', 'benchmark')
           for _f in _os.scandir(_os.path.join(_d, _r))
           if _f.is_dir() and not _f.name.startswith(('.', '_'))]
for _p in _roots:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)


import argparse
import csv
import json
import os
import random
import re

from object_names import same as same_object
from floor_world import DEFAULT_DATASET
from planner import CONFERS

BDDL_DATA = "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/bddl3/bddl/generated_data"

# Slot kinds. The last four are distinguished by *affordance*, not by wording: a template
# that opens something needs an openable, and one that runs something needs a toggleable.
MOVABLE, SURFACE, CONTAINER, OPEN_CONTAINER, MACHINE, HEATER, SWITCH, WASHER, DRYER = (
    "movable", "surface", "container", "open_container", "machine", "heater", "switch",
    "washer", "dryer")

# A slot that may be either, drawn fresh each time. This is what keeps the model reading the
# sentence instead of the template: a shape whose source is always a SURFACE teaches that
# "swap the X on the Y" means ON_TOP, full stop, and the model then answers `on_top` for a
# cabinet because it has never seen that wording mean anything else. With SUPPORT the same
# wording produces "on the table" and "in the cabinet" in turn, the stated relation follows
# what was drawn, and the preposition is the only thing that distinguishes them - which is
# the signal we want learned.
SUPPORT = "support"

# `AT` in a template's stated relations, and `at` in its goal, mean "whatever the drawn
# category can sensibly host". They are resolved per example, never written down by the
# template - which is the point: a template that always says INSIDE teaches the wording, and
# `UNDER` and `NEXT_TO` living in one template each taught it twice over.
AT = "AT"

# What a thing of each kind can plausibly have something at. A container holds things inside
# it or beside it; you do not put the milk under the fridge. A surface holds them on top,
# underneath, or beside. Anything else is a movable, and a movable can have something on it
# or next to it but not inside unless it opens.
#
# Weighted, not uniform: "on the table" and "in the cupboard" are what instructions mostly
# say, and a dataset where a third of the objects start *under* something is not a dataset of
# household tasks. The rarer prepositions still appear in every template that has a source,
# which is what stops them being a template's signature.
AT_CHOICES = {
    "container": (("INSIDE", 0.80), ("NEXT_TO", 0.20)),
    "surface":   (("ON_TOP", 0.70), ("UNDER", 0.15), ("NEXT_TO", 0.15)),
    "movable":   (("ON_TOP", 0.65), ("NEXT_TO", 0.35)),
}


def _at_relation(category, containers, surfaces, rng):
    """Which relation this drawn target can host, for one example."""
    kind = ("container" if category in containers
            else "surface" if category in surfaces else "movable")
    options, weights = zip(*AT_CHOICES[kind])
    return rng.choices(options, weights=weights)[0]

# What an action can sensibly be done TO. A task is not made harder by being nonsense, only
# less like a task: "wash the bobby pin in the dishwasher" and "heat the volleyball" teach a
# model that the verb says nothing about the object, when in a real instruction it says a
# great deal. From BEHAVIOR's own `cookable` and `cloth` annotations, and for crockery from
# the names, since the dataset has no property for it.
COOKABLE, CLOTH, DISHWARE = "cookable", "cloth", "dishware"
# And the machine has to match the load. A dishwasher does not launder a tarp and an
# electric cauldron does not do the washing-up, so the two washers are separate slots.
LAUNDRY, DISHWASHER = "laundry", "dishwasher"
DISHWARE_PAT = (r"(plate|bowl|cup|mug|glass|dish|saucer|platter|tumbler|fork|knife|spoon|"
                r"ladle|whisk|spatula|pot|pan|tray|jug|pitcher|teapot)$")

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
    (["take the {a} {a_from_s}, heat it {p_at}, and put it on the {t}",
      "warm the {a} {a_from_s} {p_at}, then leave it on the {t}",
      "grab the {a} {a_from_s}, heat it up {p_at} and set it down on the {t}",
      "the {a} is {a_at_s} - heat it {p_at} and serve it on the {t}",
      "please heat the {a} {a_from_s} {p_at} and put it on the {t}",
      "get the {a} {a_from_s}, heat it {p_at}, then put it on the {t}",
      "fetch the {a} {a_from_s} and heat it {p_at}, then set it on the {t}"],
     {"a": COOKABLE, "s": SUPPORT, "p": HEATER, "t": SUPPORT}, [("a", AT, "s")]),

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
     {"a": DISHWARE, "b": DISHWARE, "p": DISHWASHER}, []),

    (["bring the {a}, the {b} and the {d} to the {t}",
      "carry the {a}, the {b} and the {d} over to the {t}",
      "move the {a}, the {b} and the {d} onto the {t}",
      "collect the {a}, the {b} and the {d} and put them on the {t}"],
     {"a": MOVABLE, "b": MOVABLE, "d": MOVABLE, "t": SUPPORT}, []),

    (["take the {a} and the {b} {ab_from_c} and put them both on the {t}",
      "get the {a} and the {b} {ab_from_c}, then set them both on the {t}",
      "the {a} and the {b} are {ab_at_c} - move them both to the {t}",
      "unload the {a} and the {b} {ab_from_c} onto the {t}"],
     {"a": MOVABLE, "b": MOVABLE, "c": SUPPORT, "t": SUPPORT},
     [("a", AT, "c"), ("b", AT, "c")]),

    (["carry the {a} and the {b} to the {t}, then switch the {p} on",
      "move the {a} and the {b} onto the {t}, then turn the {p} on",
      "bring the {a} and the {b} to the {t} and switch the {p} on",
      "put the {a} and the {b} on the {t}, then leave the {p} switched on"],
     {"a": MOVABLE, "b": MOVABLE, "t": SURFACE, "p": SWITCH}, []),

    (["take the {a} {a_from_c}, put it on the {b}, then store the {b} in the {e} and shut it",
      "get the {a} {a_from_c}, set it on the {b}, and put the {b} away in the {e}, closing it",
      "the {a} is {a_at_c}; stack it on the {b} and store the {b} in the {e}, shut afterwards"],
     {"a": MOVABLE, "c": SUPPORT, "b": MOVABLE, "e": CONTAINER},
     [("a", AT, "c")]),

    (["swap the {a} {a_from_s} with the {b} {b_from_t}, using the {u} to set one down",
      "exchange the {a} {a_from_s} and the {b} {b_from_t}, parking one on the {u}",
      "the {a} is {a_at_s} and the {b} is {b_at_t} - swap them, using the {u} as space"],
     {"a": MOVABLE, "s": SUPPORT, "b": MOVABLE, "t": SUPPORT, "u": SURFACE},
     [("a", AT, "s"), ("b", AT, "t")]),

    # The washer washes and the dryer dries. Drawing both from one pool wrote "wash the hot
    # sauce in the clothes dryer, then dry it in the washer" - teaching that the verbs and
    # the machines are unrelated, when the whole point of a state condition is that they
    # are not.
    (["wash the {a} in the {p}, then dry it in the {q}, leaving both machines off and shut",
      "run the {a} through the {p} and then the {q}, closing and switching off both",
      "clean the {a} in the {p}, dry it in the {q}, and leave both shut and off"],
     {"a": CLOTH, "p": LAUNDRY, "q": DRYER}, []),

    (["move the {a} {a_from_s} to the {t}",
      "take the {a} {a_from_s} and put it on the {t}",
      "shift the {a} {a_from_s} over to the {t}",
      "the {a} is {a_at_s} - move it to the {t}"],
     {"a": MOVABLE, "s": SUPPORT, "t": SUPPORT}, [("a", AT, "s")]),

    (["put the {a} in the {c} and the {b} on the {t}",
      "the {a} goes in the {c}, and the {b} goes on the {t}",
      "place the {a} inside the {c}, then put the {b} on the {t}"],
     {"a": MOVABLE, "c": CONTAINER, "b": MOVABLE, "t": SURFACE}, []),

    (["open the {c}, take out the {a}, and leave it on the {t}, then close the {c}",
      "open the {c} and move the {a} from it onto the {t}, shutting the {c} after",
      "get the {a} out of the {c} onto the {t}, and close the {c} when you are done"],
     {"a": MOVABLE, "c": CONTAINER, "t": SURFACE}, [("a", "INSIDE", "c")]),

    (["turn the {p} on, then put the {a} on the {t}",
      "switch the {p} on, then set the {a} down on the {t}",
      "leave the {p} switched on, and the {a} on the {t}"],
     {"p": SWITCH, "a": MOVABLE, "t": SURFACE}, []),

    # UNDER and NEXT_TO. Both are edge types the world graph carries, so a task that states
    # one should be extracted as a relation rather than flattened into "somewhere".
    (["take the {a} {a_from_s} and put it on the {t}",
      "the {a} is {a_at_s} - move it onto the {t}",
      "fetch the {a} {a_from_s}, then leave it on the {t}"],
     {"a": MOVABLE, "s": SUPPORT, "t": SUPPORT}, [("a", AT, "s")]),

    (["take the {a} {a_from_b} and put it in the {c}, closing it after",
      "the {a} is {a_at_b}; put it away in the {c} and shut it",
      "move the {a} {a_from_b} into the {c} and close it"],
     {"a": MOVABLE, "b": MOVABLE, "c": CONTAINER}, [("a", AT, "b")]),

    (["put the {a} {a_from_s} on the {t}, then switch the {p} on",
      "the {a} is {a_at_s} - move it to the {t} and switch the {p} on"],
     {"a": MOVABLE, "s": SUPPORT, "t": SUPPORT, "p": SWITCH},
     [("a", AT, "s")]),
]

# What "done" looks like, per shape, in the language `GraphMachine.unmet` tests. The goal is
# as derivable from the sentence as the objects are - "put it on the breakfast table" is
# `on_top`, and an appliance the task runs must end off and shut - so the generator can emit
# it for nothing, and a model can be trained to read it.
#
# `open` and `toggled` conditions are filtered by affordance when the row is built: a burner
# has no door, and asserting `open(burner, false)` would be a condition nothing can satisfy.
# Two kinds of condition, and no others. WHERE each object ends up, and WHAT was done to
# it - the state a running appliance confers. The safety conditions that used to be here
# (`open(x, false)`, `toggled(x, false)`) are gone: `GraphMachine` derives them from what
# the plan actually disturbed, which is both strictly stronger - it catches a cupboard the
# task never mentioned - and 119 of the benchmark's 318 conditions that a model no longer
# has to reproduce.
#
# `heated` is written against the appliance slot and filtered by affordance when the row is
# built, so a task that runs something which cooks nothing simply has no state condition.
GOALS = [
    [("at", "a", "t"), ("heated", "a", "p")],
    [("object_inside", "a", "c"), ("object_inside", "b", "c")],
    [("object_inside", "a", "p"), ("object_inside", "b", "p"),
     ("heated", "a", "p"), ("heated", "b", "p")],
    [("at", "a", "t"), ("at", "b", "t"), ("at", "d", "t")],
    [("on_top", "a", "t"), ("on_top", "b", "t")],
    [("on_top", "a", "t"), ("on_top", "b", "t"), ("toggled", "p", True)],
    [("on_top", "a", "b"), ("object_inside", "b", "e")],
    [("at", "a", "t"), ("at", "b", "s")],
    [("object_inside", "a", "q"), ("heated", "a", "p"), ("heated", "a", "q")],
    [("at", "a", "t")],
    [("object_inside", "a", "c"), ("on_top", "b", "t")],
    [("on_top", "a", "t")],
    [("on_top", "a", "t"), ("toggled", "p", True)],
    [("on_top", "a", "t")],
    [("object_inside", "a", "c")],
    [("on_top", "a", "t"), ("toggled", "p", True)],
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
    from planner import NOT_GRASPABLE

    # `NOT_GRASPABLE` is the same affordance `GraphMachine` uses to refuse GRASP(bookcase).
    # Without it this pool called a bookcase, a bathtub and a crib "movable", so they were
    # only ever drawn as cargo and never as somewhere to put things - and the goal model
    # then met `bookcase` for the first time at test time, on 22 benchmark tasks.
    fixed = (with_property("openable", dataset_root) | with_property("toggleable", dataset_root)
             | with_property("heatSource", dataset_root) | NOT_GRASPABLE)
    return sorted(c for c in cats if not _PART.search(c) and c not in fixed
                  and not _STRUCTURAL.search(c) and not re.search(SURFACE_PAT, c))


def house_fixtures():
    """Every category a household task may name, derived rather than surveyed by hand."""
    return set(household().get("categories") or ())


def _open_containers():
    """Fillable, no door, and actually installed in houses.

    All three conditions are read rather than listed: `fillable` and `openable` from BDDL,
    and "installed in houses" from what the residential scenes are furnished with. Dropping
    the last one admits beakers and chalices from the chemistry-lab scenes; dropping the
    first admits every surface.
    """
    from planner import FILLABLE, OPENABLE
    from derive_vocab import house_rooms, scene_fixtures

    installed = set(scene_fixtures(indoor=house_rooms(), houses_only=True))
    return sorted(c for c in installed if c in FILLABLE and c not in OPENABLE)


def pools(dataset_root=DEFAULT_DATASET):
    """The slot vocabularies, each real and each matching its affordance."""
    cats = [c for c in os.listdir(os.path.join(dataset_root, "objects"))
            if not _PART.search(c)]
    present = set(cats)
    movable = movable_pool(dataset_root)
    openable = with_property("openable", dataset_root)
    toggleable = with_property("toggleable", dataset_root)
    heat = with_property("heatSource", dataset_root)
    # Fixtures are restricted to what the surveyed houses actually contain. The pools are
    # matched by regex over the whole BEHAVIOR catalogue - 51 scenes, including chemistry
    # labs, restaurants and gyms - so `SURFACE_PAT` accepted a `periodic_table`, a
    # `massage_bed` and a `sauna_bench` as furniture, and instructions came out reading
    # "leave it on the periodic table". A household task set has to be furnished like a
    # house. Movables are left alone: what a task carries is not what the scene is built
    # from, and the survey only records the latter.
    domestic = house_fixtures()

    def domestic_only(pool):
        kept = [c for c in pool if c in domestic]
        return kept or sorted(pool)

    return {
        MOVABLE: domestic_only(movable_pool(dataset_root)),
        SURFACE: domestic_only(sorted(c for c in cats if re.search(SURFACE_PAT, c))),
        # Everywhere a thing can end up: surfaces, storage with a door, and storage
        # without one. The third was missing, so "bring the notebook to the bookcase" was a
        # sentence the generator could not write and the goal model never read.
        SUPPORT: sorted(set(domestic_only(c for c in cats if re.search(SURFACE_PAT, c)))
                        | set(domestic_only(c for c in openable
                                            if re.search(STORAGE_PAT, c)))
                        | {c for c in _open_containers() if not re.search(SURFACE_PAT, c)}),
        # Closed by the instruction, so it has to open.
        CONTAINER: domestic_only(sorted(c for c in openable if re.search(STORAGE_PAT, c))),
        # Things go *in* it but it has no door - a bookcase, a bin, a hamper, a sink. These
        # fell through every pool: not a surface, not an openable container, and (once
        # `movable_pool` stopped calling them cargo) not movable either. 35 of the
        # benchmark's 102 `object_inside` conditions name one.
        OPEN_CONTAINER: sorted(c for c in _open_containers()
                               if not re.search(SURFACE_PAT, c)),
        # Loaded *and* run, so it has to do both.
        MACHINE: sorted(c for c in openable & toggleable if re.search(MACHINE_PAT, c)),
        HEATER: domestic_only(sorted(heat)),
        SWITCH: domestic_only(sorted(c for c in toggleable if re.search(MACHINE_PAT, c))),
        # Kept apart because the verbs are: a washer washes and a dryer dries, and one pool
        # for both wrote "wash it in the clothes dryer, then dry it in the washer".
        WASHER: sorted(CONFERS["washed"] & present),
        DRYER: sorted(CONFERS["dried"] & present),
        COOKABLE: sorted(set(movable) & with_property("cookable", dataset_root)),
        CLOTH: sorted(set(movable) & with_property("cloth", dataset_root)),
        DISHWARE: sorted(c for c in movable if re.search(DISHWARE_PAT, c)),
        LAUNDRY: sorted({"washer", "washing_machine"} & present),
        DISHWASHER: sorted({"dishwasher"} & present),
    }


_VOCAB = None


def household(path="data/household_vocab.json"):
    """The derived household vocabulary: which categories, how common, and where.

    Built by `derive_vocab.py` from BEHAVIOR's own activity definitions and scene contents -
    see that file for what each field answers and why it is derived rather than listed.
    """
    global _VOCAB
    if _VOCAB is None:
        try:
            with open(path) as handle:
                _VOCAB = json.load(handle)
        except OSError:
            _VOCAB = {"categories": [], "weight": {}, "rooms": {}}
    return _VOCAB


_ROOMS_FOR = None


def rooms_for(category, survey_path=None):
    """The room types a category is actually found in, across the surveyed scenes.

    A uniform draw over every room produced "the dishwasher in the bedroom", "the closet
    spatula" and "the soap dish in the basement". Those are not household instructions, and
    a model trained on them learns that the room qualifier carries no information about the
    object - which is the opposite of what it is there to teach. The survey says where each
    kind of thing really lives, so the qualifier is drawn from that.
    """
    rooms = (household().get("rooms") or {}).get(category)
    return rooms or None


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
        # Where this kind of thing is really found, weighted by how often. A category the
        # survey has never seen is not qualified at all rather than qualified at random.
        counts = rooms_for(name)
        if not counts:
            return words, None
        room = rng.choices(list(counts), weights=list(counts.values()))[0].replace("_", " ")
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
    use = household().get("weight") or {}
    banned = set(exclude)
    available = {k: [c for c in v if c not in banned]
                 for k, v in pools(dataset_root).items()}
    # Washers and dryers are a closed class - the dataset ships two and one - so the guard
    # only protects the pools that are meant to be varied.
    for kind, pool in available.items():
        if not pool:
            raise SystemExit(f"holding back the benchmark left no {kind} categories")
        if kind not in (WASHER, DRYER, MACHINE, LAUNDRY, DISHWASHER) and len(pool) < 3:
            raise SystemExit(f"holding back the benchmark left only {len(pool)} "
                             f"{kind} categories")

    movable = movable_pool(dataset_root)
    openable = with_property("openable", dataset_root)
    toggleable = with_property("toggleable", dataset_root)

    out = []
    for _ in range(n):
        index = rng.randrange(len(SHAPES))
        phrasings, slots, stated = SHAPES[index]
        # Every slot gets a different object, or the sentence stops making sense: "move
        # the mug from the mug to the mug". Distinct is not enough, though - two names the
        # *scorer* cannot tell apart make the label ambiguous rather than wrong, and the
        # dataset has 47 such pairs (`bar` and `chocolate_bar`, `leaf` and `bay_leaf`). A
        # sentence naming both has no single right answer, so it is never generated.
        chosen, used = {}, []
        for slot, kind in slots.items():
            # Heat sources are drawn towards the ones with a door. The preposition follows
            # the appliance - "in the oven", "on the grill" - so an unweighted draw over a
            # pool that is mostly doorless made "heat it IN the X" the rare phrasing: 180
            # against 867, where every heating task a person writes says "in". Trained on
            # that, the model met the benchmark's wording as an outlier and lost 11 points.
            # Coherence and frequency both matter, and this keeps both.
            pool = available[kind]
            if kind is HEATER and rng.random() < 0.75:
                openable_heaters = [c for c in pool if c in openable]
                pool = openable_heaters or pool
            # Weighted by how much household use BEHAVIOR records for each category, so an
            # object one activity mentions stays as rare in the data as it is in life. Drawn
            # uniformly, a `graduated_cylinder` was as likely as a `mug` and the instructions
            # read like a stockroom inventory.
            weights = [use.get(c, 1.0) for c in pool]
            for _ in range(40):
                pick = rng.choices(pool, weights=weights)[0]
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
        # "heat it IN the oven" puts the thing inside, so the oven must end shut. "heat it
        # ON the burner" does not, and a burner has no door. Writing "in" for both taught an
        # incoherent rule - 55% of heating tasks ended shut and 45% did not, with nothing in
        # the sentence to tell them apart - and the model then omitted `open(oven, false)`
        # on 13 of the benchmark's 100 tasks. The preposition now follows the appliance.
        if "p" in chosen:
            spoken["p_at"] = ("in the " + spoken["p"] if chosen["p"] in openable
                              else "on the " + spoken["p"])

        # What each stated relation actually is, for this draw. A SUPPORT slot that came
        # back a cabinet states INSIDE and one that came back a table states ON_TOP, and the
        # preposition follows - which is the whole point of drawing it fresh.
        # `fillable` decides in-versus-on. `CONTAINER` is the openable subset, which is the
        # wrong test here: a bin holds things and has no door.
        from planner import FILLABLE

        containers = set(FILLABLE)
        surfaces = set(available[SURFACE]) - containers
        stated = [(o, _at_relation(chosen[t], containers, surfaces, rng) if r == AT else r, t)
                  for o, r, t in stated]

        for obj, rel, tgt in stated:
            source, at = PHRASES[rel]
            spoken[f"{obj}_from_{tgt}"] = rng.choice(source).format(x=spoken[tgt])
            spoken[f"{obj}_at_{tgt}"] = rng.choice(at).format(x=spoken[tgt])
        # Two objects sharing one stated source get their own slot, since the phrasing
        # names the pair once rather than each in turn. Keyed by the slot they share and by
        # the relation actually drawn, so "take the a and the b out of the cabinet" and
        # "off the counter" are the same template.
        shared = {t for _, _, t in stated if sum(1 for _, _, u in stated if u == t) > 1}
        for tgt in shared:
            rel = next(r for _, r, t in stated if t == tgt)
            source, at = PHRASES[rel]
            spoken[f"ab_from_{tgt}"] = rng.choice(source).format(x=spoken[tgt])
            spoken[f"ab_at_{tgt}"] = rng.choice(at).format(x=spoken[tgt])

        instruction = rng.choice(phrasings).format(**spoken)
        dependent = [{"object": chosen[o], "relation": rel, "target": chosen[t]}
                     for o, rel, t in stated]
        # Exactly one class each, most-informative first. An object whose support the task
        # gives is `dependent` even if its room were also given, because the support pins
        # it and the room only narrows it.
        placed = {d["object"] for d in dependent}
        stated_rooms = {o: r for o, r in rooms.items() if o not in placed}
        uncertain = sorted(set(chosen.values()) - placed - set(stated_rooms))
        from build_tasks import stated_preposition

        goal = []
        for kind, left, right in GOALS[index]:
            obj = chosen.get(left)
            if obj is None:
                continue
            if kind == "heated":
                # Which state the appliance confers is a fact about the appliance, so the
                # generator looks it up rather than the template guessing.
                appliance = chosen.get(right)
                state = next((k for k, v in CONFERS.items() if appliance in v), None)
                if state:
                    goal.append([state, obj, True])
                continue
            if kind in ("at", "on_top", "object_inside"):
                # In-versus-on is decided by ONE rule, the same one `build_tasks` applies to
                # the benchmark: the sentence decides when it commits to a preposition, and
                # BEHAVIOR's `fillable` annotation decides when it does not.
                #
                # The shapes used to hardcode `on_top` in the table above and only the `at`
                # relation consulted the annotation, so any fillable category drawn into a
                # surface slot was labelled wrong: 875 rows said things end on top of a
                # fridge or a cabinet. The model learned it, and then read "bring the
                # notebook to the bookcase" as `on_top`, which the benchmark scores wrong.
                # Deriving both from the same function is what keeps the two in step.
                target = chosen[right]
                said = stated_preposition(instruction, target)
                want = said or ("INSIDE" if target in containers else "ON_TOP")
                goal.append(["object_inside" if want == "INSIDE" else "on_top", obj, target])
                continue
            goal.append([kind, obj, right if isinstance(right, bool) else chosen[right]])

        out.append({"task": instruction,
                    "extraction": {"uncertain": uncertain,
                                   "stated": stated_rooms,
                                   "dependent": dependent},
                    "goal": goal})
    return out


def benchmark_categories(path="data/tasks.json", movable_only=False):
    """Every category the 100-task benchmark uses, to hold back from training.

    `movable_only` holds back the 82 *objects* the tasks are about and lets the 28 fixtures
    through - ovens, washers, cabinets, tables. That is the right holdout for a model whose
    job needs to know what an appliance *does*.

    Holding everything back tests two different things at once. "Can it name an object it
    has never seen?" is the question object extraction should answer, and the full holdout
    is right for it. "Does a washer wash?" is not a question about reading a sentence - it
    is background knowledge every household robot has - and excluding `oven`, `microwave`,
    `washer`, `dishwasher` and `clothes_dryer` made it unanswerable: the goal model could
    never learn that running a washer washes, so it could not emit `washed` at all, and
    omitted `open(oven, false)` because it had never met an oven.

    The tasks themselves are still entirely unseen either way. What changes is whether the
    model is allowed to know what a kitchen contains.
    """
    tasks = json.load(open(path))
    used = ({s["name"] for t in tasks for s in t["spawn"]}
            | {a for t in tasks for _, a in t["plan"] if a}
            | {n for t in tasks for n in t["extraction"]["uncertain"]}
            | {d["object"] for t in tasks for d in t["extraction"]["dependent"]}
            | {d["target"] for t in tasks for d in t["extraction"]["dependent"]})
    return used & set(movable_pool()) if movable_only else used


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/extraction-dev.json")
    args = parser.parse_args()

    # One rule, no flag. The benchmark's **movable** objects are always held back, because
    # a model that has seen `apple_pie` in training is being tested on memory rather than
    # reading. Its fixtures are not: knowing that a cabinet opens and a washer washes is
    # background knowledge every instruction assumes, and holding those back left the
    # generator unable to build the very tasks it most needs to teach - drawing a container
    # for a swap became so rare it never happened in 8000 examples.
    #
    # This used to be three flags, and the combination mattered in a way nothing recorded:
    # the two adapters were trained on different ones, so the goal model had never seen a
    # container as a swap source and answered `on_top` for a cabinet.
    held = benchmark_categories(movable_only=True)
    rows = generate(args.n, args.seed, held)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    unique = len({r["task"] for r in rows})
    print(f"wrote {len(rows)} instructions ({unique} unique) to {args.out}"
          + f", holding back {len(held)} of the benchmark's movable categories")
    for r in rows[:4]:
        e = r["extraction"]
        print(f"\n  {r['task']}")
        print(f"     uncertain: {e['uncertain']}")
        print(f"     stated:    {e['stated']}")
        print(f"     dependent: {[(d['object'], d['relation'], d['target']) for d in e['dependent']]}")


if __name__ == "__main__":
    raise SystemExit(main())
