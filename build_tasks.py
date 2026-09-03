#!/usr/bin/env python3
"""Build and verify the task dataset: ten scenes, ten tasks each, every plan checked.

Each task carries everything the pipeline would otherwise have to guess, so that a later
evaluation can tell *which stage* got something wrong rather than only that the run failed:

    task          the natural-language instruction, the only thing an LLM is given
    extraction    what `task_objects.extract` should return for it - the ground-truth
                  split into objects whose location the task states and objects it does not
    spawn         the small objects the task needs. BEHAVIOR scenes are furniture-only, so
                  a potato has to be put there, and where it starts is part of the task
    goal          what has to be true at the end, as graph edges or node state
    plan          a reference sequence that achieves it

Nothing here is taken on trust: `main` replays every reference plan through `GraphMachine`
and refuses to write the dataset unless all of them apply *and* meet their goal. A task
whose own reference plan does not work is not a task, it is a bug.

Rooms and furniture come from the BEHAVIOR-1K ground truth, not from the RSN - this is the
dataset the RSN will later be measured against, so its object locations have to be true.

    python build_tasks.py               # verify and write data/tasks.json
    python build_tasks.py --scene Rs_int --verbose
"""

import argparse
import json
import os
from collections import Counter

from floor_world import DEFAULT_DATASET, FloorWorld
from graph_machine import GraphMachine
from world_graph import WorldGraph

OUT = "data/tasks.json"

SCENES = ["Beechwood_0_int", "Beechwood_1_int", "Benevolence_1_int", "Ihlen_1_int",
          "Merom_1_int", "Pomaria_0_int", "Pomaria_1_int", "Rs_int",
          "Wainscott_0_int", "Wainscott_1_int"]

_WORLDS = {}


def world_for(scene):
    """The scene's ground truth, loaded once."""
    if scene not in _WORLDS:
        _WORLDS[scene] = FloorWorld.load(scene, categories=None)
    return _WORLDS[scene]


def furniture_rooms(scene):
    """category -> the room it is in, from ground truth.

    A category with instances in several rooms is pinned to the one holding most of them,
    which is what "the countertop" means in a house with a kitchen run and one console in
    the living room. Tasks that need the other one say so with `rooms`.
    """
    world = world_for(scene)
    seen = {}
    for name in world.truth.object_names():
        room = world.room_of(name)
        if room is not None:
            seen.setdefault(world.category_of(name), Counter())[room] += 1
    return {category: counts.most_common(1)[0][0] for category, counts in seen.items()}


def derive_extraction(task):
    """The stage-1 ground truth, derived from the plan rather than written by hand.

    `uncertain` is every object the plan acts on or the task injects, minus the ones whose
    starting place the task states outright; `dependent` is those. Deriving it removes a
    whole class of unfair scoring: written by hand, 57 of these named a worktop the plan
    passes over and the instruction never mentions, so an extractor was being marked wrong
    for not inventing objects nobody asked for.

    An injected object counts as `dependent` exactly when **the instruction names the thing
    it starts in or on** - "take the fruitcake out of the fridge" states a location, "put
    the mug away in the cabinet" does not say where the mug is. Deriving that too keeps the
    ground truth in step with the wording: these texts were rewritten several times, and a
    hand-written `dependent` list silently stopped matching them.
    """
    stated = {s["name"]: {"object": s["name"], "relation": s["relation"],
                          "target": s["target"]}
              for s in task["spawn"] if mentions(task["task"], s["target"])}
    named = {arg for _, arg in task["plan"] if arg} | {s["name"] for s in task["spawn"]}
    return {"uncertain": sorted(named - set(stated)),
            "dependent": [stated[n] for n in sorted(stated)]}


def mentions(text, name):
    """Does this instruction name this object? Underscores read as spaces, and a
    `bag_of_flour` may reasonably be called just "the flour"."""
    lowered = text.lower()
    prose = name.replace("_", " ")
    return (prose in lowered or name.replace("_", "-") in lowered
            or prose.split()[-1] in lowered)


def seed_graph(task):
    """The true world a plan is scored against: the whole scene, plus what the task injects.

    Every category the scene holds is in it, not only the ones the reference plan happens
    to name. That matters for scoring somebody else's plan: an LLM that reaches the same
    goal by way of a different real cupboard should be marked right, and a world built only
    from the reference extraction would fail it for touching furniture that exists.

    Rooms come from the scene's own ground truth, so this is what `scene_graph.populate`
    produces with the guessing taken out.
    """
    world = world_for(task["scene"])
    rooms = furniture_rooms(task["scene"])
    rooms.update(task.get("rooms", {}))

    objects = {name: {"room": room, "probability": 1.0} for name, room in rooms.items()}
    relations = []
    for spawn in task.get("spawn", []):
        target = spawn["target"]
        objects[spawn["name"]] = {"room": rooms.get(target), "probability": 1.0}
        relations.append({"from": spawn["name"], "relation": spawn["relation"],
                          "to": target})
        if target not in objects and target in rooms:
            objects[target] = {"room": rooms[target], "probability": 1.0}

    graph = {"scene": task["scene"], "rooms": world.room_graph["rooms"],
             "edges": world.room_graph["edges"], "objects": objects,
             "unplaced": {}, "relations": relations}
    return graph


# Every category the BEHAVIOR object dataset ships. A spawned object has to be one of
# these or `execute_plan.py` cannot instantiate it, and the task is unrunnable in
# OmniGibson however well it checks out symbolically.
def _loadable_categories(dataset_root=None):
    root = dataset_root or os.path.join(os.path.dirname(DEFAULT_DATASET), "behavior-1k-assets")
    path = os.path.join(DEFAULT_DATASET, "objects")
    return set(os.listdir(path)) if os.path.isdir(path) else set()


_LOADABLE = None


def check_objects(task):
    """Do the objects this task names actually exist where it says they do?

    Two different questions, because the answers come from different places.

    **Furniture** must be in *this scene*, in the room the task claims. `furniture_rooms`
    reads the scene's own ground truth, so a task that names a cabinet the house does not
    have, or puts one in the wrong room, is caught here rather than by a run that quietly
    plans around a placeholder node.

    **Small objects cannot come from the scene at all.** Across the ten scenes there are
    three graspable objects in total - `Rs_int`'s laptop, `Wainscott_0_int`'s coffee maker
    and a garden fence - because BEHAVIOR scenes are furniture-only. So every task injects
    what it needs, and what it can inject is limited by what the *object* dataset ships:
    the name has to be a real category or OmniGibson cannot build it.
    """
    global _LOADABLE
    if _LOADABLE is None:
        _LOADABLE = _loadable_categories()

    world = world_for(task["scene"])
    rooms = furniture_rooms(task["scene"])
    problems = []

    for override, room in (task.get("rooms") or {}).items():
        if room not in world.rooms:
            problems.append(f"this scene has no room called {room}")
        elif override not in rooms:
            problems.append(f"'{override}' is not in this scene")
        elif not any(world.category_of(n) == override and world.room_of(n) == room
                     for n in world.truth.object_names()):
            problems.append(f"'{override}' is not in {room} in this scene")

    spawned = {s["name"] for s in task.get("spawn", [])}
    # An injected object must not share a name with something the scene already holds, or
    # the graph gets two nodes of that name and the plan is ambiguous about which it means.
    for name in sorted(spawned & set(rooms)):
        problems.append(f"'{name}' is already in this scene; injecting one duplicates it")
    for name in sorted({arg for _, arg in task["plan"] if arg} - spawned):
        if name not in rooms:
            problems.append(f"plan acts on '{name}', which this scene does not have")

    # Everything the ground-truth extraction names must be nameable *from the instruction*,
    # or stage 1 is being asked to invent it.
    extraction = derive_extraction(task)
    for name in (extraction["uncertain"]
                 + [d["object"] for d in extraction["dependent"]]):
        if not mentions(task["task"], name):
            problems.append(f"the task text never mentions '{name}', "
                            f"so extraction cannot be expected to find it")

    for name in sorted(spawned):
        if _LOADABLE and name not in _LOADABLE:
            problems.append(f"'{name}' is not a BEHAVIOR object category; it cannot be "
                            f"loaded into the simulator")
    return problems


def verify(task, verbose=False, simulate=True):
    """Replay the reference plan. Returns (ok, message).

    Three things have to hold, and the first is easy to forget: the goal must not already
    be true before the robot does anything. A task whose goal holds at the start is
    satisfied by the empty plan, so it measures nothing - "carry the pillow from one bed to
    another" reads like a task and is not one, because both beds are the category `bed` and
    the pillow starts on it.
    """
    problems = check_objects(task)
    if problems:
        return False, "; ".join(problems)

    graph = seed_graph(task)
    plan = [(a, b if b else None) for a, b in task["plan"]]
    goal = [tuple(g) for g in task["goal"]]

    already = GraphMachine(WorldGraph.from_scene_graph(graph), allow_search=True).run([], goal)
    if already.goal_met:
        return False, "the goal is already true before the plan runs"

    machine = GraphMachine(WorldGraph.from_scene_graph(graph), allow_search=True)
    outcome = machine.run(plan, goal)
    if verbose:
        print(outcome.report())
    if outcome.failed_at is not None:
        step = outcome.steps[outcome.failed_at]
        return False, f"step {outcome.failed_at + 1} {step.action}({step.arg}): {step.reason}"
    if not outcome.goal_met:
        return False, "goal not met: " + ", ".join(f"{t}({a}, {b})" for t, a, b in outcome.missing)

    # Executable, not merely applicable. The graph machine has no floor and no camera, so
    # it will happily validate a plan that reaches for a chair the robot cannot get within
    # 1.5 m of, or that names an object no sweep of its room will ever reveal. Three tasks
    # in the first hundred were like that, and each looked like a planner failure until it
    # was driven. Running the reference plan in the 2-D simulator is the check that a
    # successful plan *exists* for this task at all.
    if simulate:
        from sim_eval import run_plan

        driven = run_plan(task, seed_graph(task), task["plan"], verbose=False)
        if not driven["ok"]:
            return False, f"the reference plan does not run: {driven['why']}"

    # Safety: whatever the plan opened it must shut, and whatever it switched on it must
    # switch off. A task whose goal does not *say* so is a task an unsafe plan can pass,
    # so the goal has to assert the final state, not merely the plan happen to reach it.
    if not outcome.safe:
        return False, ("plan leaves "
                       + ", ".join([f"{n} open" for n in outcome.left_open]
                                   + [f"{n} switched on" for n in outcome.left_on]))
    asserted = {(kind, name) for kind, name, value in goal
                if kind in ("open", "toggled") and value is False}
    touched = {("open", n) for n in machine.opened} | {("toggled", n) for n in machine.switched_on}
    unasserted = touched - asserted
    if unasserted:
        return False, ("goal does not require putting back: "
                       + ", ".join(f"{k}({n})" for k, n in sorted(unasserted)))
    return True, f"{len(plan)} actions, {len(goal)} goal conditions"


def main():
    from tasks import TASKS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", help="verify only this scene")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else SCENES
    dataset, failures = [], []
    for scene in scenes:
        tasks = TASKS.get(scene, [])
        print(f"\n=== {scene} === {len(tasks)} tasks")
        for index, task in enumerate(tasks, 1):
            task = {**task, "scene": scene, "id": f"{scene}-{index:02d}"}
            task["extraction"] = derive_extraction(task)
            ok, message = verify(task, args.verbose)
            print(f"  {'ok  ' if ok else 'FAIL'} {task['id']}  {task['task'][:58]:58s} {message}")
            if ok:
                dataset.append(task)
            else:
                failures.append((task["id"], message))

    print(f"\n{len(dataset)} verified, {len(failures)} failed")
    for task_id, message in failures:
        print(f"  FAIL {task_id}: {message}")
    if failures:
        return 1
    if not args.scene:
        with open(args.out, "w") as f:
            json.dump(dataset, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
