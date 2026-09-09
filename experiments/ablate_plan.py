#!/usr/bin/env python3
"""Drop one action from a working plan and see which model notices.

The pipeline has two things that judge a plan before it runs, and they judge different
things. `graph_machine.check` replays the plan as graph edits in about a millisecond, so a
bad plan is rejected before a long run proves it wrong. `sim2d.Sim2D` drives the robot
across a grid and adds what a symbolic model has to assume: whether the robot can get
there, and whether it is close enough to touch what it is acting on.

This asks how much each one actually catches. Start from a plan that works, drop exactly
one action, and run both. A plan missing a step fails in one of three ways, and telling
them apart is the point:

    inapplicable   some action's preconditions no longer hold - PLACE_INSIDE into an oven
                   nobody opened. The plan is wrong and both models should say so.
    goal not met   every action still applies, and the required edges are absent at the
                   end. The plan is executable and does not do the task.
    harmless       the step was not load-bearing for the goal as stated, and dropping it
                   changes nothing. That is a fact about the *goal*, not about the plan.

    python ablate_plan.py                       # leave-one-out over the whole plan
    python ablate_plan.py --plan plan.json      # against a plan from pipeline.py
    python ablate_plan.py --fuzz 400            # random plans: where do the two differ?
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import copy
import json

from floor_world import FloorWorld
from graph_machine import GraphMachine
from sim2d import Sim2D, categories_for, stage_plan
from world_graph import WorldGraph

SCENE = "Beechwood_0_int"

# The task, its plan and its goal, written out rather than read from the LLM so the
# ablation is against a plan known to work. This is the corrected version of what
# Qwen2.5-7B produced for "take the potato from the countertop, heat it in the oven, then
# put it on the breakfast table" - the LLM's own plan is rejected before it gets here.
TASK = ("take the potato from the countertop, heat it in the oven, "
        "then put it on the breakfast table")

PLAN = [
    ("NAVIGATE_TO", "countertop"), ("GRASP", "potato"),
    ("NAVIGATE_TO", "oven"), ("OPEN", "oven"),
    ("PLACE_INSIDE", "oven"), ("CLOSE", "oven"),
    ("TOGGLE_ON", "oven"), ("TOGGLE_OFF", "oven"),
    ("OPEN", "oven"), ("GRASP", "potato"), ("CLOSE", "oven"),
    ("NAVIGATE_TO", "breakfast_table"), ("PLACE_ON_TOP", "breakfast_table"),
]

GOAL = [("on_top", "potato", "breakfast_table")]

# The RSN's output for this task, so the ablation runs against the same world the pipeline
# would have built. `potato ON_TOP countertop` is stated by the task, not guessed.
GRAPH = {
    "objects": {
        "countertop": {"room": "kitchen_0", "probability": 0.99},
        "oven": {"room": "kitchen_0", "probability": 0.77},
        "breakfast_table": {"room": "kitchen_0", "probability": 0.56},
        "potato": {"room": "kitchen_0", "probability": 1.0},
    },
    "relations": [{"from": "potato", "relation": "ON_TOP", "to": "countertop"}],
}

# Corruptions that are not a missing step: the same actions in the wrong order, or on the
# wrong object. A plan can be complete and still be nonsense. Each takes the *grounded*
# plan and a name binding, because a corruption that names a category the world does not
# hold tests the grounding rather than the models.
REORDERINGS = [
    ("place before grasping",
     lambda p, b: p[:1] + p[2:5] + [p[1]] + p[5:]),
    ("close the oven before filling it",
     lambda p, b: p[:4] + [p[5], p[4]] + p[6:]),
    ("grasp the oven instead of the potato",
     lambda p, b: [("GRASP", b["oven"]) if s[0] == "GRASP" else s for s in p]),
    ("put the potato back on the counter",
     lambda p, b: p[:-1] + [("PLACE_ON_TOP", b["countertop"])]),
    ("navigate to the table but never place",
     lambda p, b: p[:-1]),
    ("never navigate anywhere",
     lambda p, b: [s for s in p if s[0] != "NAVIGATE_TO"]),
    ("put the potato inside the countertop",
     lambda p, b: [(a, b["countertop"]) if a == "PLACE_INSIDE" else (a, o)
                   for a, o in p]),
    ("open the oven and never close it",
     lambda p, b: [s for s in p if s[0] != "CLOSE"]),
    ("heat the oven before the potato is in it",
     lambda p, b: p[:4] + [p[6], p[7]] + p[4:6] + p[8:]),
]


def build(saved):
    """A staged world and a grounded plan, fresh each time - the run mutates both."""
    world = FloorWorld.load(saved["scene"],
                            categories=categories_for(saved["plan"]["steps"]))
    steps, hints, _, focus = stage_plan(world, saved, verbose=False)
    return world, [(s["action"], s.get("object")) for s in steps], hints, focus


def ground_goal(goal, binding):
    return [(edge, binding.get(a, a), binding.get(b, b)) for edge, a, b in goal]


def offline(world, plan, goal, scene_graph):
    """What the graph model says, seeded the way this experiment needs.

    In the pipeline the seed is `WorldGraph.from_scene_graph` over the RSN's output - the
    same beliefs the LLM was shown and the same ones `planner.validate` checked, one graph
    and one piece of code for both. Here it is **ground truth**, deliberately: the question
    this file asks is which broken plans the precondition model catches, and seeding from
    the RSN would fold the RSN's own uncertainty into the answer. The object locations are
    given so that what is left being tested is the machine.

    Open and toggled state is seeded too, for the same reason: a shut oven looks shut, and
    without it the machine cannot tell "closed" from "never mentioned".
    """
    machine = GraphMachine(world.truth.copy())
    machine.open = dict(world.open)
    machine.toggled = dict(world.toggled)
    outcome = machine.run(plan, goal)
    if outcome.failed_at is not None:
        return "inapplicable", outcome.failed_at + 1, outcome.steps[outcome.failed_at].reason
    if not outcome.goal_met:
        missing = ", ".join(f"{t}({a}, {b})" for t, a, b in outcome.missing)
        return "goal not met", None, f"missing {missing}"
    return "ok", None, "goal met"


def simulated(world, plan, hints):
    """What the 2-D simulator says, driving the robot and acting only within reach."""
    sim = Sim2D(world, room_hints=hints, verbose=False)
    results = sim.run(plan)
    for index, result in enumerate(results):
        if not result.ok:
            return "inapplicable", index + 1, result.reason, sim
    return "ok", None, "ran to the end", sim


def verdict(world, plan, hints, goal, scene_graph):
    """Both models on one plan, plus whether the goal actually holds in the world."""
    off = offline(world, plan, goal, scene_graph)
    status, step, reason, sim = simulated(world, plan, hints)
    if status == "ok":
        met = all(world.truth.has_edge(*edge) for edge in goal)
        status, reason = ("ok", "goal met") if met else ("goal not met", "goal edges absent")
    return off, (status, step, reason)


def row(label, off, sim):
    def cell(result):
        status, step, _ = result
        return f"{status}" + (f" @{step}" if step else "")
    mark = "  " if off[0] == sim[0] else " !"
    return (f"{mark} {label:38s} {cell(off):16s} {cell(sim):18s} "
            f"{(sim[2] if sim[0] != 'ok' else off[2])[:64]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", help="a pipeline.py --json plan to ablate instead")
    parser.add_argument("--scene", default=SCENE)
    parser.add_argument("--fuzz", type=int, metavar="N",
                        help="run N random plans through both models instead, and report "
                             "where they disagree")
    args = parser.parse_args()

    if args.plan:
        with open(args.plan) as f:
            saved = json.load(f)
    else:
        saved = {"scene": args.scene, "task": TASK, "graph": GRAPH,
                 "plan": {"steps": [{"action": a, "object": o} for a, o in PLAN]}}

    if args.fuzz:
        agree, disagree, lengths, only_graph = fuzz(saved, trials=args.fuzz)
        total = agree + sum(disagree.values())
        mean = sum(lengths) / len(lengths) if lengths else 0
        print(f"{total} plans the graph model accepts, on {saved['scene']}, "
              f"{mean:.1f} actions each\n")
        print(f"  the simulator accepts {agree}/{total} of them ({agree / total:.0%})")
        for (_, why), count in sorted(disagree.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  only the simulator refuses: {why}")
        print(f"\n  and the other way round, on plans the simulator accepts:")
        if not only_graph:
            print("       nothing - the graph model refused none of them")
        for why, count in sorted(only_graph.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4d}  only the graph model refuses: {why}")
        return

    base_steps = [(s["action"], s.get("object")) for s in saved["plan"]["steps"]]
    world, plan, hints, _ = build(saved)
    binding = dict(zip([a for _, a in base_steps if a],
                       [a for _, a in plan if a]))
    goal = ground_goal(GOAL, binding)

    print(f"task:  {saved.get('task', '')}")
    print(f"scene: {saved['scene']}")
    print(f"goal:  {', '.join(f'{t}({a}, {b})' for t, a, b in goal)}")
    print(f"plan:  {len(plan)} actions\n")

    header = f"   {'change':38s} {'graph model':16s} {'2-D simulator':18s} why"
    print(header)
    print("   " + "-" * (len(header) - 3))

    grounded_graph = _reground(saved.get("graph", {}), binding)
    off, sim = verdict(world, plan, hints, goal, grounded_graph)
    print(row("(the plan as written)", off, sim))
    print()

    for index in range(len(plan)):
        action, arg = plan[index]
        dropped = plan[:index] + plan[index + 1:]
        world, staged, hints, _ = build(saved)
        off, sim = verdict(world, dropped, hints, goal, grounded_graph)
        label = f"drop {index + 1}. {action}({_short(arg)})"
        print(row(label, off, sim))

    print()
    for label, corrupt in REORDERINGS:
        world, staged, hints, _ = build(saved)
        off, sim = verdict(world, corrupt(staged, binding), hints, goal, grounded_graph)
        print(row(label, off, sim))

    print("\n  ! marks a plan the two models disagree about.")


ACTIONS = ["NAVIGATE_TO", "GRASP", "PLACE_ON_TOP", "PLACE_INSIDE", "OPEN", "CLOSE",
           "TOGGLE_ON", "TOGGLE_OFF", "RELEASE"]


def fuzz(saved, trials=200, length=8, seed=0):
    """Generate plans the graph model accepts, then see whether the simulator does too.

    The ablation above asks about 22 plans chosen to be interesting. This asks about plans
    chosen by nobody, which is the only way to find a difference nobody thought to look
    for.

    Plans are built by walking the graph model forward: at each step, random actions are
    tried until one is *accepted*, so what comes out is a plan the graph model runs end to
    end. Sampling actions uniformly instead is close to useless - almost every random plan
    dies on its first step in both models, for the same reason, and the run reports 100%
    agreement having tested nothing. What is wanted is deep, valid states.

    Both directions are asked, because the difference is not one-sided. The simulator is
    stricter about distance - two things in one kitchen are in the same room and four
    metres apart. But it is *looser* about rooms: it grants "near" on geometry, so an
    appliance set into the wall between two rooms can be worked at from either side, where
    the graph model insists on the annotated room.
    """
    import random

    rng = random.Random(seed)
    world, _, _, _ = build(saved)
    names = world.truth.object_names()
    hints = {name: world.room_of(name) for name in names if world.room_of(name)}
    baseline = (world.truth.copy(), dict(world.open), dict(world.toggled))

    def reset():
        truth, opened, toggled = baseline
        world.truth = truth.copy()
        world.open, world.toggled = dict(opened), dict(toggled)

    # --- direction 2: plans the simulator accepts, checked by the graph model ---------
    # Slower, because building one means actually driving the robot, so fewer of them.
    only_graph = {}
    for _ in range(max(1, trials // 8)):
        reset()
        sim = Sim2D(world, room_hints=hints, verbose=False)
        steps = []
        for _ in range(rng.randint(2, length)):
            for _ in range(12):
                action = rng.choice(ACTIONS)
                arg = None if action == "RELEASE" else rng.choice(names)
                if sim.step(action, arg).ok:
                    steps.append((action, arg))
                    break
        if not steps:
            continue
        reset()
        machine = GraphMachine(world.truth.copy())
        machine.open, machine.toggled = dict(world.open), dict(world.toggled)
        outcome = machine.run(steps)
        if outcome.failed_at is not None:
            reason = outcome.steps[outcome.failed_at].reason
            key = _reason_class(reason)
            only_graph[key] = only_graph.get(key, 0) + 1

    agree, disagree, lengths = 0, {}, []
    for _ in range(trials):
        # --- a plan the graph model accepts, built one accepted step at a time ---
        reset()
        machine = GraphMachine(world.truth.copy())
        machine.open, machine.toggled = dict(world.open), dict(world.toggled)
        steps = []
        for index in range(rng.randint(2, length)):
            for _ in range(40):
                action = rng.choice(ACTIONS)
                arg = None if action == "RELEASE" else rng.choice(names)
                if machine.step(index, action, arg).ok:
                    steps.append((action, arg))
                    break
        if not steps:
            continue
        lengths.append(len(steps))

        # --- the same plan in the simulator ---
        reset()
        results = Sim2D(world, room_hints=hints, verbose=False).run(steps)
        refused = next((r for r in results if not r.ok), None)
        if refused is None:
            agree += 1
            continue
        key = ("only the simulator", _reason_class(refused.reason))
        disagree[key] = disagree.get(key, 0) + 1

    return agree, disagree, lengths, only_graph


def _reason_class(reason):
    """Bucket a refusal by what it was about, not by which object it named."""
    for probe, label in (("beyond the", "not within reach"),
                         ("no stance near", "no stance the robot can route to"),
                         ("is not in the world", "argument names nothing"),
                         ("never been seen", "unseen and no room given"),
                         ("is not in", "unreachable room / not there"),
                         ("NAVIGATE_TO it first", "not standing at it"),
                         ("already holding", "hand already full"),
                         ("nothing in the hand", "hand empty"),
                         ("fixed furniture", "not graspable"),
                         ("has no switch", "no switch"),
                         ("does not open", "no door"),
                         ("OPEN it before", "container not open"),
                         ("which is closed", "reaching into a closed container"),
                         ("on itself", "placing an object on itself")):
        if probe in (reason or ""):
            return label
    return (reason or "")[:40]


def _reground(graph, binding):
    """The RSN's scene graph with its category names replaced by the grounded instances.

    The planner names `countertop`; the world holds `countertop_jveutp_0`. The plan the
    models are handed has been grounded, so the graph they are checked against has to be.
    """
    rename = lambda name: binding.get(name, name)
    return {
        **graph,
        "objects": {rename(n): {**i, "category": i.get("category", n)}
                    for n, i in (graph.get("objects") or {}).items()},
        "relations": [{**r, "from": rename(r["from"]), "to": rename(r["to"])}
                      for r in (graph.get("relations") or [])],
    }


def _short(name):
    if not name:
        return ""
    parts = name.split("_")
    return "_".join(parts[:-2]) if len(parts) >= 3 and len(parts[-2]) == 6 else name


if __name__ == "__main__":
    main()
