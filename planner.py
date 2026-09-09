"""Turn a task description plus a populated scene graph into a validated action plan.

The LLM proposes a sequence of BEHAVIOR-1K action primitives; a symbolic world model
then replays that sequence and rejects anything the real `StarterSemanticActionPrimitives`
controller would refuse. LLMs reliably produce plausible-looking but infeasible plans -
placing an object never grasped, opening a fridge from another room - so generation
alone is not enough. Validation is what makes the output executable.

The action space is exactly the nine primitives in `controller_functions`. Arity comes
from their real signatures: RELEASE takes no argument (`_execute_release(self)`), every
other primitive takes exactly one object.
"""

import json
import re

from object_names import canonical

# The nine primitives, mirroring StarterSemanticActionPrimitiveSet. `takes_object` is
# read off the controller method signatures, not guessed.
#
# `requires` and `effect` are the specification `graph_machine.GraphMachine` enforces,
# written out for the LLM. They are here rather than only in the machine because a planner
# judged against rules it was never shown will keep breaking them: measured, a 7B model
# given only the one-line docs wrote `PLACE_INSIDE(potato)` - passing the object being
# placed rather than the container - and reproduced it on every retry, because nothing in
# the prompt said the argument is the destination. Keep these in step with
# `GraphMachine.step`; `test_graph_machine.py` pins the machine's half.
PRIMITIVES = {
    "GRASP": {
        "takes_object": True, "doc": "Pick an object up",
        "requires": "the hand is empty; the robot is standing at the object; if the "
                    "object is inside a container, that container is open",
        "effect": "the robot is holding it, and it travels with the robot"},
    "PLACE_ON_TOP": {
        "takes_object": True, "doc": "Put down what is held, on top of something",
        "requires": "the robot is standing at the destination and is holding something",
        "effect": "what was held is on top of the destination; the hand is empty"},
    "PLACE_INSIDE": {
        "takes_object": True, "doc": "Put down what is held, inside something",
        "requires": "the robot is standing at the destination, the destination is "
                    "already open, and the robot is holding something",
        "effect": "what was held is inside the destination; the hand is empty"},
    "OPEN": {
        "takes_object": True, "doc": "Open a door, lid or drawer",
        "requires": "the robot is standing at the object, and the object has a door",
        "effect": "it is open"},
    "CLOSE": {
        "takes_object": True, "doc": "Shut a door, lid or drawer",
        "requires": "the robot is standing at the object, and the object has a door",
        "effect": "it is shut"},
    "NAVIGATE_TO": {
        "takes_object": True, "doc": "Drive to an object",
        "requires": "if the object is inside something with a door, that door is open - "
                    "the robot has to be able to see it to drive to it",
        "effect": "the robot is then standing at that object, and at whatever is on or "
                  "inside it. It is no longer standing at what it drove away from"},
    "RELEASE": {
        "takes_object": False, "doc": "Let go of what is held",
        "requires": "nothing",
        "effect": "the hand is empty; what it held is left on the floor of this room"},
    "TOGGLE_ON": {
        "takes_object": True, "doc": "Switch an object on",
        "requires": "the robot is standing at the object, and the object has a switch",
        "effect": "it is on"},
    "TOGGLE_OFF": {
        "takes_object": True, "doc": "Switch an object off",
        "requires": "the robot is standing at the object, and the object has a switch",
        "effect": "it is off"},
}

# ------------------------------------------------------------------ what objects afford
#
# Read from BEHAVIOR's own annotations, not typed out. These three sets decide every
# affordance the machine refuses on - whether a thing opens, switches on, or can be picked
# up at all - so a hand-written version makes the checker right about the objects somebody
# thought of and guessing about the rest. Ours had eighteen openable categories where BDDL
# annotates thirty-five, and twenty-four switchable where BDDL annotates a hundred and
# eighty-four, and the gap was already live: the task generator can put things in a
# `cedar_chest` and heat them on a `burner`, and the machine called the first doorless and
# the second switchless. Neither word was in the lists because neither is in the benchmark.
#
# One place, so the validator, the graph machine and the 2-D world cannot disagree.

BDDL_DATA = "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/bddl3/bddl/generated_data"

# An instruction says "the cabinet" and "the lamp"; the dataset ships `bottom_cabinet` and
# `table_lamp`. A generic word inherits an affordance when EVERY category it abbreviates
# has it - all three cabinets open, so "cabinet" opens; every lamp switches, so "lamp"
# switches. Where the specific forms disagree, it inherits nothing, which is the honest
# answer: a `feta_box` opens and a `gelatin_box` does not, so "box" says nothing.
#
# This replaces a hand-written list of seventeen words. That list was not just redundant,
# it contradicted the data: it declared a `trash_can` openable when BDDL gives it no lid -
# the very thing the `no_door` fault reports - and declared `briefcase` and `freezer`
# openable when BDDL annotates neither. Four of its entries (`drawer`, `refrigerator`,
# `ceiling_light`, `television`) are not dataset categories at all, so nothing could ever
# check them. The benchmark uses none of the seventeen; it names the specific categories.
def _generic_forms(annotated, universe):
    """The abbreviations every one of whose specific forms carries the property."""
    out = set()
    for category in universe:
        parts = category.split("_")
        for cut in range(1, len(parts)):
            word = "_".join(parts[cut:])
            kin = {c for c in universe if c == word or c.endswith("_" + word)}
            if kin and kin <= annotated:
                out.add(word)
    return out


# What a mobile manipulator can lift. This is the robot's spec, not a number fitted to
# results: BEHAVIOR annotates every category's mass, and the question "can the robot pick
# this up" is answered by the arm's payload against that mass. 15 kg puts a `cedar_chest`
# (5.8) and a `bag_of_rice` (5.0) in the hand and leaves a `breakfast_table` (18) and a
# `bookcase` (36) on the floor, with nothing near the line.
#
# BDDL's `sceneObject` looked like the right annotation and is not: it marks what may
# appear in a scene, so a `water_glass` carries it, and using it refused two benchmark
# tasks that pick one up.
PAYLOAD_KG = 15.0


def _mass_table():
    """Every category's mass in kilograms, as BEHAVIOR annotates it."""
    import csv as _csv
    import os

    out = {}
    try:
        with open(os.path.join(BDDL_DATA, "category_mapping.csv"), newline="") as handle:
            for row in _csv.DictReader(handle):
                category = (row.get("category") or "").strip()
                value = (row.get("mass (auto)") or "").replace(",", "").strip()
                if category and value:
                    try:
                        out[category] = float(value)
                    except ValueError:
                        pass
    except OSError:
        pass
    return out


def _by_property(name):
    """Every object category BEHAVIOR annotates with this property."""
    import collections
    import csv as _csv
    import os

    try:
        with open(os.path.join(BDDL_DATA, "properties_to_synsets.json")) as handle:
            synsets = set(json.load(handle).get(name, ()))
        realises = collections.defaultdict(set)
        with open(os.path.join(BDDL_DATA, "category_mapping.csv"), newline="") as handle:
            for row in _csv.DictReader(handle):
                synset, category = (row.get("synset") or "").strip(), (row.get("category") or "").strip()
                if synset and category:
                    realises[synset].add(category)
    except (OSError, ValueError):
        return set()
    return {c for s in synsets for c in realises.get(s, ())}


_OPENABLE_BASE = _by_property("openable")
OPENABLE = _OPENABLE_BASE | _generic_forms(_OPENABLE_BASE, set(_mass_table()))

# Whether things go *in* a thing, which is a different question from whether it has a door.
# A bin, a hamper and a bookcase are all fillable and none of them opens; a fridge and a
# cabinet are both. Deciding "inside or on top" with `openable` called all three of the
# first group surfaces, so the benchmark demanded rubbish be balanced on top of the bin -
# 22 of its 100 tasks asked for `on_top` into something BEHAVIOR annotates as fillable.
FILLABLE = _by_property("fillable")

# Two conditions, and it takes both. Heavy enough to be beyond the arm, *and* a thing
# scenes are built from. Either alone is wrong: `water_glass` carries `sceneObject` and
# weighs 250 g, and BEHAVIOR annotates a `sugar_sack` at 35 kg because the asset is a
# wholesale sack - each of those refused a benchmark task that picks the thing up. Together
# they agree with every case we can check by hand.
_HEAVY = {c for c, kg in _mass_table().items() if kg >= PAYLOAD_KG}
_FIXED = _HEAVY & _by_property("sceneObject")
NOT_GRASPABLE = _FIXED | _generic_forms(_FIXED, set(_mass_table()))

_TOGGLEABLE_BASE = _by_property("toggleable")
TOGGLEABLE = _TOGGLEABLE_BASE | _generic_forms(_TOGGLEABLE_BASE, set(_mass_table()))


CONFERS = {
    "cooked": {"oven", "microwave", "stove", "burner", "toaster_oven", "toaster",
               "electric_cauldron", "charcoal_grill", "flat_top_grill", "espresso_machine",
               "electric_kettle", "kettle", "pressure_cooker", "rice_cooker", "smoker",
               "deep_fryer", "air_fryer", "slow_cooker"},
    "washed": {"washer", "washing_machine", "dishwasher"},
    "dried": {"clothes_dryer", "dryer"},
}


# What must be switched off before the robot walks away, and what may be left running.
#
# The old rule was "everything you switched on", which cannot express "turn on the lamp":
# the plan met the goal and was then failed for leaving the lamp on. But a lamp is not a
# hazard and an oven is, so the distinction belongs to the object, not to the task.
#
# Three BEHAVIOR annotations answer it together, and none is enough alone. `heatSource` is
# the thing that can start a fire - it catches the coffee maker, which cooks nothing.
# `waterSource` is the tap left running, which floods rather than burns - it catches all
# nine sinks. `CONFERS` is the appliance that runs a cycle on its contents - it catches the
# dishwasher, the washer and the dryer, which BDDL calls neither. Their union is the set
# worth walking back for; a lamp and a television are in none of them.
MUST_SWITCH_OFF = (_by_property("heatSource") | _by_property("waterSource")
                   | set().union(*CONFERS.values()))


OBJECT_STATES = tuple(CONFERS)


class PlanError(Exception):
    """A plan step the controller would refuse."""


def parse_goal(text):
    """Pull the GOAL section out of a reply, as triples the graph machine can test.

    Returns `[(edge_type, object, target_or_bool), ...]`, in the same language
    `GraphMachine.unmet` reads. Anything unrecognised is dropped rather than guessed at:
    a goal the machine cannot test is worse than no goal, because it would reject every
    plan for failing a condition that never had a meaning.
    """
    if "GOAL:" not in text.upper():
        return []
    start = text.upper().index("GOAL:") + len("GOAL:")
    end = text.upper().find("PLAN:", start)
    block = text[start:end if end != -1 else len(text)]

    # The goal language, and nothing outside it. `cooked`/`washed`/`dried` were missing
    # here while the model was being trained to produce them, so every state condition it
    # wrote was silently dropped and the goal looked like placements only - which read as
    # the model failing to learn them.
    kinds = {"on_top": "on_top", "ontop": "on_top", "inside": "object_inside",
             "object_inside": "object_inside"}
    kinds.update({state: state for state in CONFERS})
    # A task that says "turn the lamp on" is asking for an end state, and nothing else
    # checks it: the safety rule deliberately ignores lamps and televisions, so if the goal
    # does not say the lamp ends on, no part of the pipeline does. Omitting it here would
    # drop it silently - exactly what happened to cooked/washed/dried, per the note above.
    kinds["toggled"] = "toggled"
    goal = []
    for match in re.finditer(r"([a-z_]+)\s*\(\s*([^,()]+?)\s*,\s*([^,()]+?)\s*\)",
                             block, re.I):
        kind = kinds.get(match.group(1).strip().lower())
        if kind is None:
            continue
        # The same vocabulary the extractor produces. Two models read the same sentence -
        # one for the objects, one for the goal - and nothing else makes them agree, so a
        # goal saying `dryer` while the graph holds `clothes_dryer` would be unmeetable
        # forever, and the loop would spend every attempt on a condition no plan can reach.
        obj = canonical(_clean_name(match.group(2)))
        rhs = match.group(3).strip().lower()
        if kind in CONFERS or kind == "toggled":
            if rhs not in ("true", "false"):
                continue
            entry = (kind, obj, rhs == "true")
        else:
            entry = (kind, obj, canonical(_clean_name(match.group(3))))
        if obj and entry not in goal:
            goal.append(entry)
    # No `open`/`toggled` here on purpose. "Put back what you disturbed" is not a property
    # of the task, it is a property of every task, and `GraphMachine` derives it from what
    # the plan actually opened - which also catches a cupboard the instruction never
    # mentioned, where a goal condition on a named object cannot.
    return goal


def _clean_name(text):
    """An object name as the graph writes them: lowercase, underscored, no punctuation."""
    name = re.sub(r"[^a-z0-9_ ]", "", text.strip().lower()).strip()
    return re.sub(r"\s+", "_", name)


def parse_plan(text):
    """Pull `PRIMITIVE(object)` steps out of an LLM reply.

    Tolerant of the usual noise: numbering, bullets, markdown fences, prose around the
    list. Anything that is not a recognized primitive call is ignored rather than
    guessed at, so a chatty model does not inject junk steps.
    """
    steps = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "```")):
            continue
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()

        # Canonical form is PRIMITIVE(object). Models also emit near-misses that carry
        # the same unambiguous intent - PLACE_ON_TOP.bed(), GRASP: mug, NAVIGATE_TO bed
        # - and dropping those silently makes a sound plan look incomplete. Rewrite the
        # recognizable shapes rather than discarding a step the model got right.
        m = re.match(r"^([A-Z_]+)\s*[.:]\s*([A-Za-z0-9_]+)\s*\(\s*\)", line)
        if m:
            line = f"{m.group(1)}({m.group(2)})"
        else:
            # `.` belongs in this class as much as `:` does. The line above already accepts
            # `PLACE_ON_TOP.bed()`, so the dotted form was recognised when it happened to
            # carry empty parens and dropped when it did not. That is not a distinction the
            # model is making, and it cost six of eleven multi-task failures in one run: the
            # model wrote `NAVIGATE_TO.armchair / GRASP.newspaper / NAVIGATE_TO.sofa /
            # PLACE_ON_TOP.sofa`, a correct plan, and the parser returned nothing - so a
            # solved errand was recorded as five failed planning attempts. Greedy decoding
            # makes it deterministic, so the same sentence failed every time it appeared.
            # Any punctuation, not a list of the ones seen so far. The model picks a
            # delimiter and sticks to it for the whole reply, and which one it picks varies
            # by sentence: `NAVIGATE_TO.armchair` on one errand, `NAVIGATE_TO/armchair/` on
            # another, both correct plans, both parsed as nothing. Enumerating delimiters
            # meant a solved errand was scored as five failed planning attempts whenever the
            # model chose one that was not on the list - six of eleven multi-task failures in
            # one run, and deterministic under greedy decoding, so the same sentence failed
            # every time it appeared.
            #
            # The gate is `m.group(1) in PRIMITIVES`, not the delimiter: a line is a step
            # only if it opens with a primitive's name and carries exactly one identifier.
            # Prose fails on the first test and an unknown verb on it too, so this stays as
            # unwilling as before to guess at a step the model did not write.
            m = re.match(r"^([A-Z_]+)\s*[^A-Za-z0-9_\s]*\s*([A-Za-z0-9_]+)?"
                         r"\s*[^A-Za-z0-9_\s]*\s*$", line)
            if m and m.group(1) in PRIMITIVES:
                line = f"{m.group(1)}({m.group(2) or ''})"

        m = re.match(r"^([A-Z_]+)\s*\(\s*([^)]*?)\s*\)", line)
        if not m:
            continue
        name, arg = m.group(1).upper(), m.group(2).strip()
        if name not in PRIMITIVES:
            continue
        arg = arg.strip("\"'").strip()
        # Models sometimes write NAVIGATE_TO(kitchen_0, fridge); keep the first term.
        if "," in arg:
            arg = arg.split(",")[0].strip().strip("\"'")
        steps.append({"action": name, "object": arg if arg else None})
    return steps


def validate(steps, graph, strict=True):
    """Replay `steps` against a symbolic world model, collecting errors and warnings.

    Tracks what the robot holds, where it is, and what is open - the state the real
    primitives check. Returns (errors, warnings); `errors` empty means the plan is
    internally consistent and every referenced object is one the scene graph believes
    exists.
    """
    objects = graph.get("objects", {})
    rooms = set(graph.get("rooms", {}))
    unplaced = set(graph.get("unplaced", {}))

    adjacency = {r: set() for r in rooms}
    for a, b in graph.get("edges", []):
        adjacency[a].add(b)
        adjacency[b].add(a)

    errors, warnings = [], []
    held = None
    location = None  # room the robot is in; None until the first navigation
    opened = set()

    def where(obj):
        return objects[obj]["room"] if obj in objects else None

    for i, step in enumerate(steps, 1):
        action, obj = step["action"], step.get("object")
        spec = PRIMITIVES[action]
        tag = f"step {i} {action}({obj or ''})"

        # --- arity, straight from the controller signatures ---
        if spec["takes_object"] and not obj:
            errors.append(f"{tag}: {action} requires an object argument")
            continue
        if not spec["takes_object"] and obj:
            warnings.append(f"{tag}: RELEASE takes no argument; ignoring '{obj}'")
            obj = None

        # --- the object must be something the scene believes in ---
        if obj is not None:
            if obj in unplaced:
                errors.append(f"{tag}: '{obj}' is believed NOT present in this scene")
                continue
            if obj not in objects and obj not in rooms:
                errors.append(f"{tag}: '{obj}' is not in the scene graph")
                continue

        # --- reachability: acting on an object requires being in its room ---
        target_room = obj if obj in rooms else where(obj)
        if action == "NAVIGATE_TO":
            if target_room is not None:
                if location is not None and target_room != location:
                    if target_room not in adjacency.get(location, set()):
                        # Not adjacent is not fatal - the motion planner routes through
                        # intermediate rooms - but a long hop is worth surfacing.
                        warnings.append(
                            f"{tag}: {location} and {target_room} are not directly "
                            "connected; the robot must route through other rooms"
                        )
                location = target_room
        else:
            if target_room is not None and location != target_room:
                if strict:
                    errors.append(
                        f"{tag}: robot is in {location or 'an unknown room'} but "
                        f"'{obj}' is in {target_room}; navigate there first"
                    )
                    continue
                warnings.append(f"{tag}: implicit navigation to {target_room}")
                location = target_room

        # --- per-primitive preconditions, mirroring the controller's own checks ---
        if action == "GRASP":
            if held is not None and held != obj:
                errors.append(f"{tag}: already holding '{held}'; release or place it first")
                continue
            if obj in NOT_GRASPABLE:
                errors.append(
                    f"{tag}: '{obj}' is a fixed appliance or furniture and cannot be "
                    f"picked up; use OPEN/CLOSE for its door or TOGGLE_ON/TOGGLE_OFF "
                    f"to switch it"
                )
                continue
            held = obj

        elif action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
            if held is None:
                errors.append(f"{tag}: nothing in hand to place")
                continue
            if held == obj:
                errors.append(f"{tag}: cannot place '{obj}' onto itself")
                continue
            if action == "PLACE_INSIDE" and obj in OPENABLE and obj not in opened:
                warnings.append(f"{tag}: '{obj}' is usually opened before placing inside")
            held = None

        elif action == "RELEASE":
            if held is None:
                warnings.append(f"{tag}: nothing in hand to release")
            held = None

        elif action == "OPEN":
            if obj in opened:
                warnings.append(f"{tag}: '{obj}' is already open")
            opened.add(obj)

        elif action == "CLOSE":
            if obj not in opened:
                warnings.append(f"{tag}: '{obj}' was not opened by this plan")
            opened.discard(obj)

    # An unfinished plan is an error, not a warning: a sequence that ends mid-carry has
    # not completed the task, and reporting it as EXECUTABLE would be misleading. This
    # is the failure mode LLM planners hit most often - they navigate to the target and
    # forget the final placement - and it is precisely the kind of feedback the retry
    # loop can act on.
    if held is not None:
        errors.append(
            f"plan ends while still holding '{held}': finish by placing or releasing it"
        )
    for obj in sorted(opened):
        warnings.append(f"plan ends with '{obj}' left open")

    return errors, warnings


# What a goal condition may say. The same four the graph machine can test, which is what
# makes a self-declared goal checkable rather than decorative.
GOAL_PREDICATES = """  on_top(object, surface)        the object ends resting on that surface
  inside(object, container)     the object ends inside that container
  open(object, false)           that door or lid ends shut
  toggled(object, false)        that switch ends off"""


def format_goal(goal):
    """The goal in words the planner can act on, or "" if there is none.

    This is the goal *this pipeline predicted from the instruction*, never the benchmark's
    answer key - the key is ground truth and handing it over would be telling the model
    what it is meant to read out of the sentence. The prediction comes from the same source
    the plan does, so showing it leaks nothing.

    It was not shown before, and that was the gap: every plan was validated against this
    goal while the planner had never seen it. A model told only "wash the towels in the
    washer" wrote PLACE_ON_TOP(washer), was refused, and was never once told that what the
    checker wanted was the towels *inside* the machine.
    """
    if not goal:
        return ""
    say = []
    for entry in goal:
        try:
            kind, obj, val = entry
        except (TypeError, ValueError):
            continue
        if kind == "on_top":
            say.append(f"  the {obj} ends on top of the {val}")
        elif kind in ("object_inside", "inside"):
            say.append(f"  the {obj} ends inside the {val}")
        elif kind == "open":
            say.append(f"  the {obj} ends {'open' if val else 'shut'}")
        elif kind == "toggled":
            say.append(f"  the {obj} ends switched {'on' if val else 'off'}")
        elif kind in CONFERS:
            say.append(f"  the {obj} ends {kind}"
                       + ("" if val else " - which it must NOT be"))
        else:
            say.append(f"  {kind}({obj}, {val})")
    if not say:
        return ""
    return ("What the finished house must look like, as read from the task:\n"
            + "\n".join(say)
            + "\n\nYour plan is checked against exactly these conditions. A plan where "
              "every action is legal but one of these does not hold at the end is "
              "rejected.\n")


def build_prompt(task, graph, with_goal=False, goal=()):
    """The planning prompt: action space, scene graph, rules, and output format.

    `with_goal` asks the model to state the finished world before it plans for it.

    The checker can test whether a plan achieves a goal, but nothing upstream produces
    one: the benchmark's goal conditions are *ground truth*, kept for scoring, and handing
    them to the planner would be telling it the answer it is meant to read out of the
    instruction. So the loop was checking only that every action applies and that nothing
    was left open - and a plan that ran flawlessly and did the wrong thing was accepted
    without complaint. Measured, that was 9 of the 8B's 16 goal failures passing on the
    first attempt, including a swap that carefully put both objects on the same table.

    Asking the *planner* for the goal closes that without a leak: the goal comes from the
    instruction, the same place the plan comes from. What it cannot catch is a
    misunderstood task - a model that misreads "swap" writes a wrong goal and then
    satisfies it. What it does catch is the commoner failure by far: understanding the
    task and writing a plan that does not carry it out.
    """
    from scene_graph import format_for_llm

    actions = "\n\n".join(
        f"  {name}({'object' if s['takes_object'] else ''})  - {s['doc']}\n"
        f"      requires: {s['requires']}\n"
        f"      then:     {s['effect']}"
        for name, s in PRIMITIVES.items()
    )
    if with_goal:
        tail = f"""First state the GOAL: the conditions that must hold when the task is
done. Use only these forms, one per line:

{GOAL_PREDICATES}

State every condition the task requires, including putting back what you disturb - if the
task has you open something or switch something on, the goal must say it ends shut and
off. Then write PLAN: and the actions.

GOAL:
  <conditions>
PLAN:
  <actions>

Reply with only those two sections, no prose."""
    else:
        tail = ("Reply with ONLY the action sequence, one action per line, no numbering, "
                "no prose.")

    prompt = f"""You are a task planner for a household robot in a simulated home.

Produce a sequence of atomic actions that completes the task. You may ONLY use these
actions, exactly as written:

{actions}

{format_for_llm(graph)}

{format_goal(goal)}
Every action above is checked against its `requires` before it runs. If one fails, the
whole plan is rejected.

Rules:
- Use only the objects listed above. Do not invent objects.
- The number after each object is how confident we are that it is really there. A low
  number means the object may not exist in this house; plan for it only if the task
  requires it.
- **NAVIGATE_TO takes an object, never a room.** `NAVIGATE_TO(kitchen)` is not a step;
  name the thing in the kitchen you are going to touch.
- **NAVIGATE_TO the object you are about to act on, every time.** Being in the same room
  is not enough, and having driven there earlier is not enough - if the robot has driven
  somewhere else since, drive back.
- For PLACE_ON_TOP and PLACE_INSIDE the argument is the **destination**: the surface or
  container being put onto or into. What is being put down is whatever the robot is
  holding, and is never named.
- The robot has one hand. GRASP before any PLACE, and after a PLACE the hand is empty
  again - to move something a second time, GRASP it again.
- OPEN and CLOSE act on the object directly. Do NOT grasp a door, appliance or cabinet
  in order to open it: write OPEN(fridge), never GRASP(fridge).
- To get something back out of a container, NAVIGATE_TO the container, OPEN it, then
  GRASP the object.
- RELEASE takes no argument: write exactly RELEASE().

Examples of correct sequences:

  Task: open the microwave
  NAVIGATE_TO(microwave)
  OPEN(microwave)

  Task: turn on the light, then turn it off
  NAVIGATE_TO(light)
  TOGGLE_ON(light)
  TOGGLE_OFF(light)

  Task: put the book on the table
  NAVIGATE_TO(book)
  GRASP(book)
  NAVIGATE_TO(table)
  PLACE_ON_TOP(table)

  Task: heat the pie in the oven, then put it on the counter
  NAVIGATE_TO(pie)
  GRASP(pie)
  NAVIGATE_TO(oven)
  OPEN(oven)
  PLACE_INSIDE(oven)
  CLOSE(oven)
  TOGGLE_ON(oven)
  TOGGLE_OFF(oven)
  OPEN(oven)
  GRASP(pie)
  CLOSE(oven)
  NAVIGATE_TO(counter)
  PLACE_ON_TOP(counter)

Task: {task}

{tail}"""

    return prompt


def generate(task, graph, model_name="Qwen/Qwen2.5-7B-Instruct",
             max_new_tokens=512, strict=True, verbose=True):
    """Ask the LLM for a plan once, and report what the validator makes of it.

    Deliberately single-shot. Repairing a rejected plan - feeding validator errors back,
    replanning, or recovering mid-execution - is the open research question this project
    exists to study, so the pipeline surfaces the raw failure rather than papering over
    it here.
    """
    generator = get_generator(model_name)
    reply = generator(build_prompt(task, graph), max_new_tokens)
    steps = parse_plan(reply)
    errors, warnings = validate(steps, graph, strict)
    if verbose:
        print(f"  {len(steps)} steps, {len(errors)} errors, {len(warnings)} warnings")
    return {"steps": steps, "errors": errors, "warnings": warnings, "raw": reply}


# One loaded model per name. Object extraction and planning both call the LLM, and a 7B
# load costs ~30s of GPU time, so the second caller reuses the first one's weights.
_GENERATORS = {}
# Loaded base models, keyed by name, so several LoRA adapters can share one.
_BASES = {}


def get_generator(model_name="Qwen/Qwen2.5-7B-Instruct", adapter=None):
    """Return a cached prompt->text function for this model."""
    key = (model_name, adapter)
    if key not in _GENERATORS:
        _GENERATORS[key] = _local_generator(model_name, adapter)
    return _GENERATORS[key]


def release_generator(model_name=None):
    """Drop a loaded model and give the GPU memory back.

    Comparing models means loading several in one process, and an 8B in fp16 is ~16 GB -
    two of them will not sit on one card. Dropping the closure is not enough on its own,
    because the allocator keeps the freed blocks reserved; `empty_cache` returns them.
    """
    import gc

    # Keys are (model_name, adapter), so a bare model name drops every adapter loaded on
    # top of it too - which is what a caller freeing the card wants.
    for key in [k for k in _GENERATORS if model_name in (None, k[0])]:
        _GENERATORS.pop(key, None)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _slug(path):
    """A PEFT adapter name: no dots or slashes, which `add_module` refuses."""
    return path.strip("/").replace("/", "_").replace(".", "_").replace("-", "_")


def _local_generator(model_name, adapter=None):
    """Lazily load a local instruct model and return a prompt->text function.

    `adapter` points at a LoRA directory from `finetune_extraction.py`, whose base model
    it names, so a fine-tuned extractor is loaded by adapter path alone.

    **Adapters on the same base share it.** The pipeline runs three models at once - an 8B
    planner and two fine-tuned 1.7B heads, one reading objects and one reading the goal -
    and loading Qwen3-1.7B twice put 23.5 GB on a 24 GB card and killed the run on the
    second task. A LoRA adapter is 78 MB against a 3.4 GB base, so the base is loaded once
    and each adapter is attached to it and selected per call.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter:
        import json

        model_name = json.load(open(f"{adapter}/training.json"))["base"]
        shared = _BASES.get(model_name)
        if shared is not None:
            name = _slug(adapter)
            if name not in getattr(shared, "peft_config", {}):
                shared.load_adapter(adapter, adapter_name=name)
            return _bind(shared, AutoTokenizer.from_pretrained(model_name), name)
    tok = AutoTokenizer.from_pretrained(model_name)
    # `device_map="auto"` needs `accelerate`, which is not in the `behavior` env and is
    # not worth installing there - the env has a verified torch/CUDA/OmniGibson stack.
    # A 7B model in fp16 is ~15GB and fits on one A5000, so load it and move it whole,
    # falling back to sharding only if accelerate happens to be available.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter, adapter_name=_slug(adapter))
        _BASES[model_name] = model
    model.eval()

    return _bind(model, tok, _slug(adapter) if adapter else None)


def _bind(model, tok, adapter=None):
    """The prompt->text function for one model, selecting `adapter` before each call."""
    import torch

    def run(prompt, max_new_tokens, temperature=0.0):
        if adapter is not None:
            model.set_adapter(adapter)
        """`temperature` 0 is greedy, which is the right default: one question, one answer.

        Anything above it samples, which is what a *retry* needs. Greedy decoding makes a
        repair loop pointless past the second attempt - measured, a rejected plan came back
        byte-identical five times running, because the model had already given its best
        answer to a prompt it was not persuaded by.
        """
        messages = [{"role": "user", "content": prompt}]
        # Depending on the transformers version this returns either a bare tensor or a
        # BatchEncoding; normalize to a tensor of ids so both work.
        #
        # `enable_thinking=False` is for the Qwen3 family, whose chat template turns on a
        # `<think>...</think>` preamble by default. Left on, the model spends the whole
        # token budget reasoning and the reply that reaches `parse_plan` has no actions in
        # it at all - the plan is not wrong, it never arrives. Templates that do not know
        # the argument reject it, so it is offered and withdrawn.
        try:
            encoded = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            encoded = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}),
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        # Belt and braces: if a thinking block comes back anyway, the answer is what
        # follows it.
        if "</think>" in reply:
            reply = reply.rsplit("</think>", 1)[1]
        return reply

    return run


def format_plan(result):
    """Human-readable plan report."""
    lines = []
    for i, s in enumerate(result["steps"], 1):
        arg = s["object"] or ""
        lines.append(f"  {i:2d}. {s['action']}({arg})")
    if not result["steps"]:
        lines.append("  (no valid actions parsed)")
    if result["errors"]:
        lines += ["", "ERRORS (plan is not executable):"]
        lines += [f"  ! {e}" for e in result["errors"]]
    if result["warnings"]:
        lines += ["", "warnings:"]
        lines += [f"  - {w}" for w in result["warnings"]]
    return "\n".join(lines)
