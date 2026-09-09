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
import re

from scene_graph import ROOM_SYNONYMS
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


EXTRACTION_TRUTH = "data/extraction_truth.json"
_TRUTH = None


def extraction_truth(task, path=EXTRACTION_TRUTH):
    """The stage-1 ground truth for one task: what the INSTRUCTION says, read by a reader.

    This used to be computed. The rule took the last word of an object's name and looked for
    it anywhere in the sentence, so "put the tablespoon away in the top cabinet" was recorded
    as saying where the *breakfast table* is - the letters "table" occur inside "tablespoon" -
    and "the top cabinet in the kitchen" as saying where the *bottom* cabinet is. Five of the
    hundred tasks were wrong in that direction, each of them marking a correct extraction as
    a miss.

    Whether a sentence states a location is a question about English, and no substring test
    answers it. So the answers are written down instead: `data/extraction_truth.json`, one
    entry per task, authored by reading each instruction and independently checked by a
    second reader. It is data, versioned and inspectable, rather than a rule nobody re-reads.
    """
    global _TRUTH
    # Keyed by path. It used to be a single dict filled by whichever caller arrived first,
    # so a second answer file - the subtasks have their own - would silently be served the
    # first one's contents and every id would come back missing.
    if _TRUTH is None:
        _TRUTH = {}
    if path not in _TRUTH:
        with open(path) as handle:
            _TRUTH[path] = json.load(handle)
    answer = _TRUTH[path].get(task["id"])
    if answer is None:
        raise SystemExit(f"no extraction ground truth for {task['id']} - add it to {path}")
    return {"uncertain": list(answer.get("uncertain") or []),
            "stated": dict(answer.get("stated") or {}),
            "dependent": [dict(d) for d in (answer.get("dependent") or [])]}


_PREPOSITIONS = {"on", "onto", "in", "into", "inside", "to", "from", "beside", "under",
                 "over", "at", "near", "with", "and", "then"}
_ON = re.compile(r"\b(?:on top of|onto|on)\s+(?:the\s+)?([a-z' ]+)")
_IN = re.compile(r"\b(?:inside(?: of)?|into|in)\s+(?:the\s+)?([a-z' ]+)")


def stated_preposition(text, target):
    """ON_TOP or INSIDE if the instruction says which, else None.

    Whether a thing goes in something or on it is usually settled by what the thing is -
    rubbish goes *in* a bin - and BEHAVIOR's `fillable` annotation answers that. But it does
    not always: a tray is annotated fillable and nine instructions say "put it **on** the
    tray", which is what a person means and what the tray is for. The sentence is the better
    authority when it commits, so it is asked first and the annotation is the fallback.
    """
    prose = target.replace("_", " ")
    words = prose.split()
    lowered = text.lower()
    for pattern, relation in ((_IN, "INSIDE"), (_ON, "ON_TOP")):
        for match in pattern.finditer(lowered):
            tail = match.group(1).split()
            # The preposition governs this destination if its name starts within the next
            # few words. A room qualifier sits between the article and the name often
            # enough to matter - "in the *bathroom* furniture sink", "in the *utility room*
            # bottom cabinet" - and requiring the name to start immediately missed all of
            # them, silently falling back to the annotation on 28 destinations.
            for skip in range(4):
                head = tail[skip:]
                # Never skip across another preposition. "out of the fridge in the kitchen
                # onto the countertop" would otherwise let the `in` reach `countertop` and
                # claim it for INSIDE - 40 rows said things go inside a breakfast table.
                if skip and (skip > len(tail) or tail[skip - 1] in _PREPOSITIONS):
                    break
                if head[:len(words)] == words or head[:1] == words[-1:]:
                    return relation
    return None


def reconcile_relations(task):
    """Make the goal and the plan agree with how the instruction words each destination.

    The shapes decide in-versus-on from the category alone. Where the sentence itself says,
    the sentence wins - so "on the tray" stays `on_top` even though a tray is fillable, and
    "to the trash can", which commits to nothing, falls back to the annotation and becomes
    `object_inside`.
    """
    from planner import FILLABLE, OPENABLE

    goal, plan = [list(g) for g in task["goal"]], [list(p) for p in task["plan"]]
    destinations = {g[2] for g in goal if g[0] in ("on_top", "object_inside")
                    and isinstance(g[2], str)}
    for target in destinations:
        said = stated_preposition(task["task"], target)
        want = said or ("INSIDE" if target in FILLABLE else "ON_TOP")
        # Opening and closing is sequenced by the shape; rewriting a destination with a door
        # would leave that sequencing wrong, so those are left to the shape.
        if target in OPENABLE:
            continue
        edge = "object_inside" if want == "INSIDE" else "on_top"
        action = "PLACE_INSIDE" if want == "INSIDE" else "PLACE_ON_TOP"
        for g in goal:
            if g[0] in ("on_top", "object_inside") and g[2] == target:
                g[0] = edge
        for step in plan:
            if step[0] in ("PLACE_ON_TOP", "PLACE_INSIDE") and step[1] == target:
                step[0] = action
    task["goal"] = [list(g) for g in goal]
    task["plan"] = [list(p) for p in plan]
    return task


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
    # The ground truth is now read rather than derived, so the old "is this name in the
    # sentence" check has nothing to catch - a reader does not put an object in the answer
    # that the sentence never named. What is still worth checking is that it names things
    # the simulator can actually load, which is the loop below.

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

    already = GraphMachine(WorldGraph.from_scene_graph(graph)).run([], goal)
    if already.goal_met:
        return False, "the goal is already true before the plan runs"

    machine = GraphMachine(WorldGraph.from_scene_graph(graph))
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

    # Safety: whatever the plan opened it must shut, and whatever it switched on that is
    # worth walking back for it must switch off. `GraphMachine` decides that on its own -
    # `left_open` over everything opened, `left_on` filtered by `planner.MUST_SWITCH_OFF` -
    # so the task only has to *be* safe, not to restate it.
    #
    # The goal used to be required to assert every `open(x, False)` and `toggled(x, False)`
    # as well. That was dropped: it restated a check the machine already makes, and it
    # restated it in a form that enforces nothing, because `unmet` reads an untouched door
    # as shut and an untouched switch as off - so the condition was satisfied by a plan
    # that never went near the object, and only ever bit when `outcome.safe` had already
    # caught it. Worse, it made "leave the lamp on" inexpressible: the goal had to demand
    # the opposite of the instruction.
    if not outcome.safe:
        return False, ("plan leaves "
                       + ", ".join([f"{n} open" for n in outcome.left_open]
                                   + [f"{n} switched on" for n in outcome.left_on]))
    return True, f"{len(plan)} actions, {len(goal)} goal conditions"


def main():
    from tasks import TASKS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", help="verify only this scene")
    parser.add_argument("--verbose", action="store_true")
    # NOT `data/tasks.json`. The corrections that were once applied to that JSON by hand -
    # conferred cooked/washed/dried conditions, the stripped safety conditions the machine
    # now enforces itself, and three goals whose lamp must end on - have since been folded
    # back into `tasks.py` and `task_shapes.py`, so this script now reproduces the committed
    # benchmark byte for byte. The guard stays anyway: it once wiped 22 conferred states,
    # 111 strips and 3 goal corrections without failing, because a regenerated file is
    # perfectly valid whether or not it is the benchmark. Writing to `--out` and diffing
    # turns a silent divergence into a visible one.
    parser.add_argument("--out", default="data/tasks.check.json")
    args = parser.parse_args()

    scenes = [args.scene] if args.scene else SCENES
    dataset, failures = [], []
    for scene in scenes:
        tasks = TASKS.get(scene, [])
        print(f"\n=== {scene} === {len(tasks)} tasks")
        for index, task in enumerate(tasks, 1):
            task = {**task, "scene": scene, "id": f"{scene}-{index:02d}"}
            task = reconcile_relations(task)
            task["extraction"] = extraction_truth(task)
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
    if os.path.abspath(args.out) == os.path.abspath(OUT):
        raise SystemExit(f"refusing to overwrite {OUT} - it holds hand-applied corrections "
                         f"that regenerating discards. Write elsewhere and diff.")
    if not args.scene:
        with open(args.out, "w") as f:
            json.dump(dataset, f, indent=1)
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    # Only here, not at import: this module is also imported at run time for `seed_graph`,
    # and a chdir on that path would move the working directory out from under a caller.
    # As a build it writes relative `data/...`, so it has to run from the repo root.
    _os.chdir(_d)
    raise SystemExit(main())
