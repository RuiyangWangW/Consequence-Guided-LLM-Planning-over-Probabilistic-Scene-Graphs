#!/usr/bin/env python3
"""Run a pipeline plan in the 2-D simulator and see whether the robot actually finishes.

`evaluate.py` scores a plan by replaying it against a `GraphMachine` seeded from ground
truth. That answers "is this plan valid and does it reach the goal", which is the question
the safety filter exists to ask - but it is not the question "would the robot do it". The
machine has no camera, no floor, and no notion of a room that has to be searched: a
`NAVIGATE_TO` always succeeds, so a plan that depends on finding a mug in the wrong room
scores the same as one that does not.

This module closes that gap. The plan runs in `sim2d`: the robot drives A* routes, sweeps
rooms with a simulated wedge camera, and only knows what it has seen.

**The world is ground truth and the belief is not.** That separation is the whole point and
it is easy to lose - `sim2d.stage_plan` injects each object into the room the *belief*
names, which is right for a demo and fatal for a measurement, because a wrong belief then
places the object where the belief expects it and can never be caught. Here the objects go
where `task["spawn"]` says, the belief comes from extraction and the RSN, and where they
disagree the robot has to discover it the way it would in the house.
"""

import json

from floor_world import FloorWorld
from sim2d import Sim2D


def categories_for(task, graph):
    """Every category this run could touch: what the task spawns, what the plan names, and
    what the belief thinks exists. Loading the whole scene is slower and loading less risks
    a plan failing on furniture that is really there."""
    wanted = {s["target"] for s in task.get("spawn", [])}
    wanted |= {a for _, a in task.get("plan", []) if a}
    wanted |= set((graph.get("objects") or {}))
    return sorted(n for n in wanted if n)


def build_world(task, graph, plan_objects=(), rng=None):
    """The true house: scene furniture, plus the task's objects where the *task* puts them.

    **Which instance a spawn lands on is decided by the task's own ground truth, never by
    the belief under test.** A task says "the wrapping paper is on a bottom cabinet" and
    Merom has three; picking the first one put the paper in a child's room while the
    reference ground truth had it in the kitchen, and the reference plan - a plan known to
    be correct - then failed to find it. Worse, if the choice followed the belief, a wrong
    belief would place the object where it expected and could never be caught, which is the
    one thing this harness exists to detect.

    `build_tasks.seed_graph` is that single source: it is the world with the guessing taken
    out, so every belief is measured against the same house.
    """
    from build_tasks import seed_graph

    truth = seed_graph(task)
    categories = sorted(set(categories_for(task, graph)) | set(categories_for(task, truth))
                        | {n for n in plan_objects if n})
    world = FloorWorld.load(task["scene"], categories=categories)

    # Which instance, among the several a category has, must be decided once and the same
    # way every time - and it has to be one the robot can get to. Beechwood has nine
    # countertops and Pomaria three armchairs, one of them wedged so deep in a furniture
    # cluster that the nearest floor is 2.6 m away, beyond the arm's 1.5 m. A task that
    # spawns there is a task no plan can finish, which measures nothing.
    from sim2d import REACH, Sim2D

    probe = Sim2D(world, verbose=False)

    def reachable(instance):
        stances = probe.stances_for(instance)
        return any(probe.route_to(s) is not None
                   and world.distance_to(instance, *world.to_world(*s)) <= REACH
                   for s in stances)

    placed = []
    for spawn in task.get("spawn", []):
        target = spawn["target"]
        instances = world.truth.by_category(target)
        if not instances:
            continue                       # the scene has no such support; skip the object
        want = ((truth.get("objects") or {}).get(target) or {}).get("room")
        here = [i for i in instances if world.room_of(i) == want]
        support = (next((i for i in here if reachable(i)), None)
                   or next((i for i in instances if reachable(i)), None)
                   or (here[0] if here else instances[0]))
        key = "inside" if spawn["relation"].upper() == "INSIDE" else "on_top"
        world.add_object(spawn["name"], spawn["name"], **{key: support})
        placed.append(spawn["name"])
    return world, placed


def ground(name, world, graph):
    """A category name from the plan onto an instance in this world.

    The planner names categories because that is what the RSN predicts; the house holds
    instances, and Beechwood has nine countertops. Prefer the instance in the room the
    belief expects - picking the wrong one sends the robot to a counter in another room and
    the failure looks like navigation rather than grounding.
    """
    if name is None or name in world.truth.objects:
        return name
    instances = world.truth.by_category(name)
    if not instances:
        return name
    believed = ((graph.get("objects") or {}).get(name) or {}).get("room")
    return next((i for i in instances if world.room_of(i) == believed), instances[0])


def run_plan(task, graph, steps, start_room=None, verbose=False):
    """Execute `steps` in the simulator. Returns a verdict dict.

        {"ok": bool, "why": str, "driven": float, "steps_run": int,
         "goal_met": bool, "unsafe": [...], "failed_at": int|None}

    A run is a success only if every action applied, the goal holds in the *true* world,
    and nothing was left open or switched on - the same three conditions the symbolic
    scorer uses, so the two are comparable and the difference between them is exactly the
    cost of perception.
    """
    plan = [(s["action"], s.get("object")) if isinstance(s, dict) else (s[0], s[1])
            for s in steps]
    world, _ = build_world(task, graph, [a for _, a in plan])

    bound = [(action, ground(arg, world, graph)) for action, arg in plan]
    # What the robot believes, and where it should look first. The ranking behind each
    # first choice is what makes a wrong belief survivable.
    hints = {}
    for name, info in (graph.get("objects") or {}).items():
        instance = ground(name, world, graph)
        order = info.get("candidates") or ([info["room"]] if info.get("room") else [])
        if order:
            hints[instance] = list(order)

    sim = Sim2D(world, start_room=start_room, room_hints=hints,
                focus={a for _, a in bound if a}, verbose=verbose)
    results = sim.run(bound, stop_on_failure=True)

    failed = next((i for i, r in enumerate(results) if not r.ok), None)
    # `left_unsafe` returns {"open": [...], "on": [...]}, and an empty one of those is
    # still a truthy dict - testing the dict itself marks every run unsafe.
    left = sim.left_unsafe()
    unsafe = sorted(left["open"]) + sorted(left["on"])
    # The goal is checked against the TRUE world the robot has been driving in, using the
    # same definition the symbolic scorer uses - and grounded the same way the plan was.
    # The goal names categories (`breakfast_table`) and the house holds instances
    # (`breakfast_table_skczfi_1`); grounding the plan but not the goal marks a run that
    # put the pie exactly where it was asked as having missed.
    goal = []
    for entry in task.get("goal", ()):
        edge_type, src, dst = entry
        dst = ground(dst, world, graph) if isinstance(dst, str) else dst
        goal.append((edge_type, ground(src, world, graph), dst))
    missing = sim.truth_machine.unmet(goal)
    goal_met = not missing
    driven = sim.distance

    ok = failed is None and not unsafe and goal_met
    if ok:
        why = ""
    elif failed is not None:
        why = f"step {failed + 1} {results[failed].action}: {results[failed].reason}"
    elif unsafe:
        why = f"left unsafe: {unsafe}"
    else:
        why = f"goal not met in the true world: {missing}"
    return {"ok": ok, "why": why, "driven": round(driven, 1),
            "steps_run": len(results), "goal_met": goal_met,
            "missing": [list(m) for m in missing], "unsafe": list(unsafe),
            "failed_at": failed}


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/tasks.json")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from build_tasks import seed_graph

    task = json.load(open(args.tasks))[args.index]
    print(f"{task['id']}  {task['task']}\n")
    verdict = run_plan(task, seed_graph(task), task["plan"], verbose=not args.quiet)
    print(f"\n{json.dumps(verdict, indent=1)}")
    return 0 if verdict["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
