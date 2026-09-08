#!/usr/bin/env python3
"""Checks for the mechanical repair pass, over hand-built graphs and over real failures.

The hand-built cases pin each rule and each guard; the replay at the end runs the pass over
every plan the 4B actually had refused, which is the only way to see whether the rules
compose on plans nobody wrote for them.
"""

import json
import sys

from graph_machine import GraphMachine
from repair import repair
from world_graph import WorldGraph

passed = failed = 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


def world():
    """A kitchen: a fridge with an apple in it, a table, and a lidless bin."""
    g = WorldGraph()
    g.rooms["kitchen_0"] = {"room_type": "kitchen"}
    g.see_object("kitchen_table", "breakfast_table", "kitchen_0")
    g.see_object("fridge", "fridge", "kitchen_0")
    g.see_object("apple", "apple", "kitchen_0")
    g.see_object("bin", "public_trash_can", "kitchen_0")
    g.add_edge("object_inside", "apple", "fridge")
    return g


def outcome(g, plan, goal=()):
    return GraphMachine(g.copy()).run(plan, goal)


# --- rule 1: the missing drive ------------------------------------------------------
g = world()
plan = [("NAVIGATE_TO", "fridge"), ("OPEN", "fridge"), ("GRASP", "apple"),
        ("PLACE_ON_TOP", "kitchen_table")]
fixed, notes = repair(g, plan)
check("not_near: a NAVIGATE_TO is inserted before the place",
      ("NAVIGATE_TO", "kitchen_table") in fixed, True)
check("not_near: the repaired plan is applicable", outcome(g, fixed).failed_at, None)
check("not_near: the original steps survive in order",
      [s for s in plan if s in fixed] == plan, True)

# --- rule 1 guard: never synthesise a drive to a room -------------------------------
g = world()
fixed, notes = repair(g, [("GRASP", "kitchen_0")])
check("not_near guard: a room is not repaired into a NAVIGATE_TO",
      any(a == "NAVIGATE_TO" and o == "kitchen_0" for a, o in fixed), False)

# --- a name the pipeline never produced is refused, and never repaired ---------------
# One vocabulary. The graph is built from the extractor's names and the plan is asked for
# them, so an unresolvable name is one the model invented - a fact only the model can fix.
g = world()
out = outcome(g, [("NAVIGATE_TO", "unicorn"), ("GRASP", "unicorn")])
check("invented name: refused at the first step", out.failed_at, 0)
check("invented name: reported as unknown", out.steps[0].fault, ("unknown", "unicorn"))
fixed, notes = repair(g, [("NAVIGATE_TO", "unicorn"), ("GRASP", "unicorn")])
check("invented name: the repair leaves it to the LLM", notes, [])

# --- rule 2: the shut container -----------------------------------------------------
g = world()
plan = [("NAVIGATE_TO", "apple"), ("GRASP", "apple"),
        ("NAVIGATE_TO", "kitchen_table"), ("PLACE_ON_TOP", "kitchen_table")]
fixed, notes = repair(g, plan)
check("closed: an OPEN is inserted for the blocking container",
      ("OPEN", "fridge") in fixed, True)
check("closed: rule 1 supplies the drive to the container on the next round",
      fixed[fixed.index(("OPEN", "fridge")) - 1], ("NAVIGATE_TO", "fridge"))
check("closed: the repaired plan is applicable", outcome(g, fixed).failed_at, None)

# --- rule 2 + discharge: the inserted OPEN is closed again --------------------------
g = world()
goal = (("on_top", "apple", "kitchen_table"), ("open", "fridge", False))
fixed, notes = repair(g, plan, goal)
out = outcome(g, fixed, goal)
check("discharge: the fridge the repair opened is shut again", ("CLOSE", "fridge") in fixed, True)
check("discharge: goal met after repair", out.goal_met, True)
check("discharge: safe after repair", out.safe, True)

# --- a room argument is dropped, and the next step names what the plan meant ---------
g = world()
plan = [("NAVIGATE_TO", "kitchen_0"), ("NAVIGATE_TO", "kitchen_table")]
fixed, notes = repair(g, plan)
check("room: the drive to a room is dropped", ("NAVIGATE_TO", "kitchen_0") in fixed, False)
check("room: the drive to the object survives", ("NAVIGATE_TO", "kitchen_table") in fixed, True)

# --- a doorless container loses every OPEN and CLOSE in one edit ---------------------
# The bin has no lid, so both steps are doomed; taking them one per round would need six
# rounds for the three-item plans the models actually write.
g = world()
plan = [("NAVIGATE_TO", "apple"), ("NAVIGATE_TO", "fridge"), ("OPEN", "fridge"),
        ("NAVIGATE_TO", "apple"), ("GRASP", "apple"),
        ("NAVIGATE_TO", "bin"), ("OPEN", "bin"), ("PLACE_INSIDE", "bin"), ("CLOSE", "bin")]
fixed, notes = repair(g, plan)
check("no_door: both the OPEN and its CLOSE go in one edit",
      [(a, o) for a, o in fixed if o == "bin" and a in ("OPEN", "CLOSE")], [])
check("no_door: one round removed the pair, not one round per step",
      len([n for n in notes if n.startswith("no_door")]), 1)
check("no_door: the placement into the bin survives", ("PLACE_INSIDE", "bin") in fixed, True)

# --- a grasp of fixed furniture is dropped ------------------------------------------
# Nearly always the model running go-grasp-place over the destination it had just driven
# to, so removing the grasp leaves a plan that says what it meant.
g = world()
plan = [("NAVIGATE_TO", "kitchen_table"), ("GRASP", "kitchen_table"),
        ("PLACE_ON_TOP", "kitchen_table")]
fixed, notes = repair(g, plan)
check("not_graspable: the GRASP on fixed furniture is dropped",
      ("GRASP", "kitchen_table") in fixed, False)

# --- a toggle of something with no switch loses the pair -----------------------------
g = world()
plan = [("NAVIGATE_TO", "kitchen_table"), ("TOGGLE_ON", "kitchen_table"),
        ("TOGGLE_OFF", "kitchen_table")]
fixed, notes = repair(g, plan)
check("no_switch: both toggles go in one edit",
      [(a, o) for a, o in fixed if a.startswith("TOGGLE")], [])

# --- the two hand-state rules, which read the goal for intent -----------------------
# The two-hands mistake: grab both, then place both. Nothing in the graph says where the
# first one was going; the goal does.
g = world()
g.see_object("mug", "mug", "kitchen_0")
plan = [("NAVIGATE_TO", "apple"), ("GRASP", "apple"), ("GRASP", "mug"),
        ("NAVIGATE_TO", "kitchen_table"), ("PLACE_ON_TOP", "kitchen_table")]
goal = (("on_top", "apple", "kitchen_table"), ("on_top", "mug", "kitchen_table"))
g.remove_edge("object_inside", "apple", "fridge")
fixed, notes = repair(g, plan, goal)
out = outcome(g, fixed, goal)
check("holding: the held object's errand is finished before the next grasp",
      out.failed_at, None)
check("holding: and the goal is then met", out.goal_met, True)
check("holding: it places rather than dropping",
      any(a == "RELEASE" for a, o in fixed), False)

# The mirror image: placed twice having grasped once.
g = world()
g.remove_edge("object_inside", "apple", "fridge")
plan = [("NAVIGATE_TO", "apple"), ("GRASP", "apple"),
        ("NAVIGATE_TO", "kitchen_table"), ("PLACE_ON_TOP", "kitchen_table"),
        ("PLACE_ON_TOP", "kitchen_table")]
goal = (("on_top", "apple", "kitchen_table"),)
fixed, notes = repair(g, plan, goal)
check("empty_hand: declines when the goal wants nothing more there", notes, [])

g = world()
g.see_object("mug", "mug", "kitchen_0")
g.remove_edge("object_inside", "apple", "fridge")
plan = [("NAVIGATE_TO", "apple"), ("GRASP", "apple"),
        ("NAVIGATE_TO", "kitchen_table"), ("PLACE_ON_TOP", "kitchen_table"),
        ("PLACE_ON_TOP", "kitchen_table")]
goal = (("on_top", "apple", "kitchen_table"), ("on_top", "mug", "kitchen_table"))
fixed, notes = repair(g, plan, goal)
out = outcome(g, fixed, goal)
check("empty_hand: the missing GRASP is the object the goal wants there",
      ("GRASP", "mug") in fixed, True)
check("empty_hand: and the plan then runs", out.failed_at, None)
check("empty_hand: and meets the goal", out.goal_met, True)

# --- what must still NOT be repaired ------------------------------------------------
g = world()
plan = [("NAVIGATE_TO", "fridge"), ("OPEN", "fridge"), ("NAVIGATE_TO", "apple"),
        ("GRASP", "apple"), ("GRASP", "apple")]
fixed, notes = repair(g, plan)
check("holding: with no goal there is no intent to read, so no repair", notes, [])
check("holding: the plan comes back untouched", fixed, plan)

g = world()
plan = [("NAVIGATE_TO", "kitchen_table"), ("PLACE_ON_TOP", "kitchen_table")]
fixed, notes = repair(g, plan)
check("empty_hand: with no goal there is nothing to grasp towards", notes, [])

g = world()
plan = [("NAVIGATE_TO", "fridge"), ("OPEN", "fridge"), ("NAVIGATE_TO", "apple"),
        ("GRASP", "apple"), ("GRASP", "apple")]
fixed, notes = repair(g, plan, (("on_top", "banana", "kitchen_table"),))
check("holding: a goal that never mentions the held object is no help", notes, [])

# --- a repair never makes a plan worse ----------------------------------------------
g = world()
for plan in ([("RELEASE", "apple")], [("FLY", "apple")], [], [("OPEN", "apple")]):
    before = outcome(g, plan)
    fixed, notes = repair(g, plan)
    after = outcome(g, fixed)
    check(f"no regression on {plan}",
          (after.failed_at is None) >= (before.failed_at is None), True)

# --- replay: every plan the 4B actually had refused ----------------------------------
try:
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
    rows = [r for r in json.load(open("data/v8-4b.json"))["rows"]
            if r["checked_cause"] == "planning"]
except (OSError, ImportError):
    rows = []

if "--replay" in sys.argv and rows:
    print(f"\n  replaying {len(rows)} refused plans")
    fixed_n = worse = 0
    for r in rows:
        ex = r["extracted"]
        seed = WorldGraph.from_scene_graph(
            populate(r["scene"], ex["uncertain"], ex["dependent"], stated=ex["stated"],
                     model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD))
        goal = tuple(tuple(x) for x in r["predicted_goal"])
        plan = [(a, o) for a, o in r["final_plan"]]
        before = outcome(seed, plan, goal)
        after_plan, notes = repair(seed, plan, goal)
        after = outcome(seed, after_plan, goal)
        if before.failed_at is not None and after.failed_at is None:
            fixed_n += 1
        if before.failed_at is None and after.failed_at is not None:
            worse += 1
    print(f"  made applicable: {fixed_n}/{len(rows)}   made worse: {worse}")
    check("replay: nothing made worse", worse, 0)

print(f"\n{passed}/{passed + failed} checks passed")
sys.exit(1 if failed else 0)
