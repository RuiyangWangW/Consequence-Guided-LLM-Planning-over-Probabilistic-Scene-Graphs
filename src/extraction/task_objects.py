"""Extract the objects a task needs, split by whether we know where they are.

Asks the LLM. There is no rule-based path.

Two groups come back:

  uncertain  the task names these but not their location, so the RSN has to place them
  dependent  the task states a relation ("the potato from the fridge"), so the position
             is known and gets a deterministic edge instead of a guess

Extraction is deliberately literal: it pulls out what the task names and infers nothing.
Task descriptions are expected to be explicit - "get a cup and make coffee with the
coffee maker", not "make coffee" - so guessing at unnamed tools is not the extractor's
job. Asking a 7B model to infer them produced junk anyway: "make coffee" yielded `water`,
which is a substance the robot cannot manipulate.

"Get the potato from the fridge and cook it" yields uncertain {fridge, stove} and
dependent {potato INSIDE fridge}. Asking the RSN where the potato is would be strictly
worse than using what the task already told us.

The earlier version carried ~200 lines of hand-written tables - verb->tool, noun->
appliance, stopwords, action verbs, room names, a crude de-pluralizer, and n-gram
matching against the BEHAVIOR vocabulary. All of it existed to work around that
vocabulary being incomplete, and all of it kept failing in new ways: `mug` was dropped
because BEHAVIOR has no such category, `throw` leaked through as an object, `bathroom`
came out as something to fetch. Each fix added another list.

The LLM is already loaded for planning, so asking it costs no extra model load, and the
RSN accepts arbitrary strings by construction - an extracted name never needed to be in
any vocabulary. The tables were re-encoding, worse, knowledge the model already has.

The prompt's examples are as long as the tasks it is asked about, which is not cosmetic.
The earlier version answered every example with two or three objects and Qwen3-8B copied
the habit, returning 2.8 objects where 3.5 were wanted - 81% recall. Adding examples that
answer with five and seven, without dropping any of the ones that teach INSIDE vs ON_TOP,
took it to 96% recall and 82% exact on the benchmark. See `extraction_prompts.py` for the
wordings that lost and `extraction_eval.py` for the measurements.

`--from-bddl` reads ground-truth objects out of a BEHAVIOR task definition, which is
useful for evaluating extraction against a known answer.
"""

import os as _os, sys as _sys
# Runnable as a script from anywhere. The other stages are sibling folders under src/, which
# are not on the path when this file is the one being executed, so find the repo root by
# marker and add every stage. A no-op when an entry point has already done it.
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


import argparse
import os
import re

from object_names import canonical

_CATEGORIES = None

DEFAULT_BDDL = os.environ.get(
    "BEHAVIOR_BDDL_ACTIVITIES",
    "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/bddl3/bddl/activity_definitions",
)

PROMPT = """A household robot has been given a task. Extract the objects it names, split
into two groups.

UNCERTAIN: objects the task names but does not say the location of.

DEPENDENT: objects whose CURRENT location the task states. Write each as
"<object> INSIDE <container>" or "<object> ON_TOP <container>". Only INSIDE and ON_TOP are
allowed, and both words in the relation must be real objects from the task.

This is about where something is NOW, not where the task wants it to end up. "Get the
potato from the fridge" says the potato is in the fridge now, so it is DEPENDENT. "Put
the mug in the dishwasher" describes the goal - the mug is not in the dishwasher yet - so
both are UNCERTAIN and DEPENDENT is empty.

Match the relation to the words used: "inside the cabinet", "from the fridge" and "out of
the fridge" are INSIDE; "on the counter", "on top of the stove" and "off the shelf" are
ON_TOP.

Extract only what the task actually says. Do NOT add objects it does not name, and do not
infer tools, ingredients or containers - if the task does not mention water, do not list
water. Name whole objects, not parts: "fridge", not "refrigerator_door". Skip rooms and
people.

  Task: get the potato from the fridge and cook it on the stove
  UNCERTAIN: fridge, stove
  DEPENDENT: potato INSIDE fridge

  Task: get a cup and make coffee with the coffee maker
  UNCERTAIN: cup, coffee_maker
  DEPENDENT:

  Task: put the mug in the dishwasher
  UNCERTAIN: mug, dishwasher
  DEPENDENT:

  Task: take the book off the shelf and put it on the desk
  UNCERTAIN: shelf, desk
  DEPENDENT: book ON_TOP shelf

  Task: move the plate from the counter into the cabinet
  UNCERTAIN: counter, cabinet
  DEPENDENT: plate ON_TOP counter

  Task: take the plate inside the cabinet and put it on the counter
  UNCERTAIN: cabinet, counter
  DEPENDENT: plate INSIDE cabinet

  Task: take the milk out of the fridge and leave it on the table
  UNCERTAIN: fridge, table
  DEPENDENT: milk INSIDE fridge

  Task: put the mug on top of the stove in the dishwasher
  UNCERTAIN: stove, dishwasher
  DEPENDENT: mug ON_TOP stove

  Task: take the towel and the soap from the shelf, put them in the washer, run it,
        then move them to the dryer
  UNCERTAIN: shelf, washer, dryer
  DEPENDENT: towel ON_TOP shelf, soap ON_TOP shelf

  Task: bring the plate, the bowl and the spoon from the cupboard to the table, then
        wipe the counter with the sponge
  UNCERTAIN: cupboard, table, counter, sponge
  DEPENDENT: plate INSIDE cupboard, bowl INSIDE cupboard, spoon INSIDE cupboard

  Task: put the sketch pad and the ruler in the desk's drawer in the study, then switch
        off the lamp
  UNCERTAIN: sketch_pad, ruler, drawer, lamp
  DEPENDENT:

  Task: take the kettle off the counter, fill it at the sink, put it on the stove and
        turn the stove on
  UNCERTAIN: sink, stove
  DEPENDENT: kettle ON_TOP counter

Some tasks name two objects and some name six. Answer with as many as the task actually
names - stopping at two when the task names six is the most common mistake.

An object named in a DEPENDENT relation must NOT also appear in UNCERTAIN - we already
know where it is. The container or support it refers to DOES belong in UNCERTAIN, since
we still have to find that.

Task: {task}
UNCERTAIN:"""

RELATIONS = ("INSIDE", "ON_TOP", "UNDER", "NEXT_TO")

# The rooms the RSN knows, plus the short forms instructions actually use ("the office
# cabinet", never "the private office cabinet").
ROOM_WORDS = frozenset("""
bar bathroom bedroom biology_lab break_room chemistry_lab childs_room classroom closet
computer_lab conference_hall copy_room corridor dining_room entryway exercise_room garage
garden grocery_store gym hammam infirmary kitchen living_room lobby locker_room
meeting_room pantry_room phone_room playroom private_office shared_office sauna spa
storage_room television_room utility_room
office pantry hall hallway study den porch balcony basement attic room
""".split())


def _known_categories(dataset_root=None):
    """Category names the object dataset ships, cached. Empty if it is not mounted."""
    global _CATEGORIES
    if _CATEGORIES is None:
        try:
            from floor_world import DEFAULT_DATASET

            _CATEGORIES = frozenset(
                os.listdir(os.path.join(dataset_root or DEFAULT_DATASET, "objects")))
        except OSError:
            _CATEGORIES = frozenset()
    return _CATEGORIES


def strip_room(name):
    """Split a room qualifier off the front of an object name.

    Instructions say "the office bottom cabinet" and "the kitchen countertop", and the
    extractor copies the words it is given, so it answers `office_bottom_cabinet`. No such
    BEHAVIOR category exists, and the name then fails to ground: `same_object` matches a
    one-word category inside a longer name (`countertop` is a token of
    `bathroom_countertop`) but not a two-word one, so `bottom_cabinet` never matches
    `office_bottom_cabinet`. Measured on the benchmark this was **65% of every object the
    4B appeared to miss** - names it had read correctly and qualified with a room.

    The qualifier is not noise, though: it says where the thing is, which is exactly what
    the RSN would otherwise have to guess. So return it rather than discarding it.

    A name that is *itself* a category is never split, because 22 real ones begin with a
    room word: `bar_soap` is soap, not soap in a bar, and `gym_shoe`, `garden_chair` and
    `kitchen_analog_scale` are the same mistake. Splitting them corrupted a real object name
    and invented a room the task never mentioned.

    Returns `(bare_name, room)`, with `room` None when there was no qualifier, or
    `(None, room)` when the name was nothing but a room.
    """
    if name in _known_categories():
        return name, None
    parts = name.split("_")
    for size in (2, 1):                    # "dining_room table" before "room table"
        if len(parts) > size and "_".join(parts[:size]) in ROOM_WORDS:
            return "_".join(parts[size:]), "_".join(parts[:size])
    return (None, name) if name in ROOM_WORDS else (name, None)


def _clean(name):
    """Normalize one object name to a lowercase underscored token, or None."""
    name = re.sub(r"^[\s\-\*\d\.\)]+", "", name).strip().lower()
    name = re.sub(r"[^a-z0-9 _]", "", name).strip().replace(" ", "_")
    return name if name and len(name) < 40 else None


def extract(task, model_name="Qwen/Qwen2.5-7B-Instruct", generator=None,
            max_new_tokens=160, temperature=0.0, prompt=None, normalize=True):
    """Split the objects a task needs into uncertain and dependent.

    Returns:
        {
          "uncertain": [name, ...],                     # location unknown - ask the RSN
          "dependent": [{"object": str,                 # location given by the task
                         "relation": "INSIDE"|"ON_TOP",
                         "target": str}, ...],
          "rooms": {name: room},                        # room the task named it in, if any
        }

    The split is what lets the scene graph carry two kinds of edge. An uncertain object
    gets a probabilistic room assignment from the RSN; a dependent object gets a
    deterministic edge to its container or support, because the task stated it. "Get the
    potato from the fridge" tells us exactly where the potato is - guessing at it with a
    prior would be strictly worse than using what we were told.

    `generator` accepts an already-loaded prompt->text function so the pipeline can reuse
    the planner's model.
    """
    if generator is None:
        from planner import get_generator

        generator = get_generator(model_name)

    reply = generator((prompt or PROMPT).format(task=task), max_new_tokens, temperature)

    # Sections are found by label and sliced in the order they appear, not in a fixed one.
    # The prompted wording ends with "UNCERTAIN:" and the trained one leads with
    # "DEPENDENT:", and an order-dependent split silently returned nothing for whichever
    # it was not written for.
    upper = reply.upper()
    marks = []
    for label, key in (("UNCERTAIN:", "uncertain"), ("DEPENDENT:", "dependent"),
                       ("STATED:", "stated"), ("ROOMS:", "stated")):
        at = upper.find(label)
        if at != -1:
            marks.append((at, len(label), key))
    marks.sort()
    if not marks:
        # A prompt ending in "UNCERTAIN:" gets a reply that opens mid-section with no
        # label at all. Everything before the next label is that section.
        marks = [(0, 0, "uncertain")]
        reply = reply
    section = {"uncertain": "", "dependent": "", "stated": ""}
    for i, (at, width, key) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(reply)
        section[key] += reply[at + width:stop]

    # A chatty model keeps going after the answer; stop at the first blank line or at
    # anything that looks like the start of another task.
    def head(block):
        out = []
        for line in block.splitlines():
            if not line.strip():
                if out:
                    break
                continue
            if re.match(r"^\s*(task|uncertain|stated|dependent|rooms|note|explanation)\b",
                        line, re.I):
                break
            out.append(line)
        return " ".join(out)

    uncertain = []
    for part in head(section["uncertain"]).split(","):
        name = _clean(part)
        if name and name not in uncertain:
            uncertain.append(name)

    dependent = []
    for part in re.split(r"[,\n]", head(section["dependent"])):
        m = re.match(r"\s*(.+?)\s+(INSIDE|ON_TOP|ONTOP|NEXT_TO|NEXTTO|NEXT TO|BESIDE|"
                     r"UNDER|UNDERNEATH|BENEATH|IN|ON)\s+(.+?)\s*$", part, re.I)
        if not m:
            continue
        obj, rel, target = _clean(m.group(1)), m.group(2).upper(), _clean(m.group(3))
        if not obj or not target:
            continue
        # Four relations, because the world graph carries four kinematic edge types. A task
        # that says "under the table" states a fact as definite as "on the table", and
        # flattening it to "somewhere" throws it away.
        rel = {"ON_TOP": "ON_TOP", "ONTOP": "ON_TOP", "ON": "ON_TOP",
               "UNDER": "UNDER", "UNDERNEATH": "UNDER", "BENEATH": "UNDER",
               "NEXT_TO": "NEXT_TO", "NEXTTO": "NEXT_TO", "NEXT TO": "NEXT_TO",
               "BESIDE": "NEXT_TO"}.get(rel, "INSIDE")
        if any(d["object"] == obj for d in dependent):
            continue
        dependent.append({"object": obj, "relation": rel, "target": target})

    stated = {}
    for part in re.split(r"[,\n]", head(section["stated"])):
        m = re.match(r"\s*(.+?)\s+(?:IN|INROOM)\s+(.+?)\s*$", part, re.I)
        if m:
            obj, room = _clean(m.group(1)), _clean(m.group(2))
            if obj and room:
                stated[obj] = room

    if normalize:
        # `strip_room` is the backstop for a name the model left fused - "the office bottom
        # cabinet" answered as `office_bottom_cabinet` instead of split across the name and
        # the STATED line. A trained model states the room itself and this finds nothing to
        # do; a prompted one relies on it entirely.
        bare = []
        for name in uncertain:
            short, room = strip_room(name)
            if short is None:                  # the model answered with a room, not an object
                continue
            if room:
                stated.setdefault(short, room)
            if short not in bare:
                bare.append(short)
        uncertain = bare

        kept = []
        for d in dependent:
            obj, obj_room = strip_room(d["object"])
            target, target_room = strip_room(d["target"])
            if obj is None:
                continue                       # "kitchen INSIDE cabinet" is not a fact
            if obj_room:
                stated.setdefault(obj, obj_room)
            if target is None:
                # "mug ON_TOP kitchen" states a room, not a support. The support is wrong
                # but the mug is real, so keep the object and record the room instead.
                if target_room:
                    stated.setdefault(obj, target_room)
                continue
            if target_room:
                stated.setdefault(target, target_room)
            kept.append({"object": obj, "relation": d["relation"], "target": target})
        dependent = kept

        fixed = {}
        for obj, room in stated.items():
            short, fused = strip_room(obj)
            if short is not None:
                fixed[short] = room or fused
        stated = fixed

    # --- the three classes are disjoint, most-informative first -----------------------
    #
    # A support beats a room and a room beats nothing, so each object takes the first class
    # that applies. Enforcing that here rather than trusting the model means `populate` can
    # read the output directly instead of reconciling overlapping lists.

    # A relation is only trustworthy if the model kept its own rule: an object whose
    # location it claims to know must not also be listed as unknown. When both appear, the
    # model has contradicted itself, and in practice it is the relation that is wrong -
    # "put the mug in the dishwasher" yields `UNCERTAIN: mug` and `DEPENDENT: mug INSIDE
    # dishwasher`, restating the goal as a current position. Drop the relation and keep the
    # object, which is the safe direction: a wrong certain edge would tell the planner the
    # mug is somewhere it is not.
    claimed = set(uncertain) | set(stated)
    dropped = [d for d in dependent if d["object"] in claimed]
    dependent = [d for d in dependent if d["object"] not in claimed]

    placed = {d["object"] for d in dependent}

    # Every relation target has to be findable, so it belongs in a class even if the model
    # left it out entirely - otherwise the fridge holding the potato is never in the graph
    # and NAVIGATE_TO(fridge) fails on an object the planner was never given.
    for d in dependent:
        if d["target"] not in placed and d["target"] not in stated \
                and d["target"] not in uncertain:
            uncertain.append(d["target"])

    # An object whose room is stated is not uncertain, and one whose support is stated is
    # neither.
    uncertain = [n for n in uncertain if n not in stated and n not in placed]

    # One vocabulary from here on. Everything downstream compares names with `==`, which
    # it can only do if what leaves this function is the dataset's own word.
    uncertain = [canonical(n) for n in uncertain]
    stated = {canonical(n): room for n, room in stated.items()}
    dependent = [{**d, "object": canonical(d["object"]),
                  "target": canonical(d["target"])} for d in dependent]

    return {"uncertain": uncertain, "stated": stated, "dependent": dependent,
            "dropped_relations": dropped}


def objects_from_bddl(task_name, bddl_root=DEFAULT_BDDL):
    """Ground-truth object categories from a BEHAVIOR task definition.

    BDDL names objects as WordNet synsets ("breakfast_table.n.01_1"); the category is the
    part before the first dot.
    """
    path = os.path.join(bddl_root, task_name, "problem0.bddl")
    if not os.path.exists(path):
        raise SystemExit(f"no BDDL definition for task: {task_name}")
    with open(path) as f:
        text = f.read()

    block = re.search(r"\(:objects(.*?)\n\s*\)", text, re.S)
    if not block:
        return []
    names = []
    for line in block.group(1).splitlines():
        line = line.strip()
        if not line or "-" not in line:
            continue
        category = line.split("-")[-1].strip().split(".")[0]
        if category and category not in names and category not in ("agent", "floor"):
            names.append(category)
    return names


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task", nargs="?", help="natural-language task description")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--from-bddl", help="read ground-truth objects from this task name")
    args = parser.parse_args()

    if args.from_bddl:
        for name in objects_from_bddl(args.from_bddl):
            print(name)
        return

    if not args.task:
        raise SystemExit("provide a task description, or use --from-bddl")

    out = extract(args.task, args.model)
    print("UNCERTAIN (location unknown - RSN will place these):")
    for name in out["uncertain"]:
        print(f"  {name}")
    print("DEPENDENT (location given by the task):")
    for d in out["dependent"] or []:
        print(f"  {d['object']} {d['relation']} {d['target']}")


if __name__ == "__main__":
    main()
