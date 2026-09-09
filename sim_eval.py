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
import math

from floor_world import FloorWorld
from object_names import match
from sim2d import Sim2D

#: One simulator control step - one waypoint driven, or one heading turned to - in seconds.
#: Wall-clock for a run is this times the number of steps, plus the algorithm's own compute.
SIM_STEP_SECONDS = 0.1


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

    memo = {}

    def reachable(instance):
        if instance not in memo:
            stances = probe.stances_for(instance)
            memo[instance] = any(probe.route_to(s) is not None
                                 and world.distance_to(instance, *world.to_world(*s)) <= REACH
                                 for s in stances)
        return memo[instance]

    # `ground` has to answer the same question about the plan's names and the goal's that
    # this answers about the spawn's supports, and it was answering it differently: the
    # spawn preferred an instance the robot can get to, and the grounding fell back to
    # `instances[0]`. So a task could spawn its object on a reachable countertop and then
    # score the plan against an unreachable one. Sharing the predicate - memoised, because
    # routing is not cheap and `ground` is called for every step - makes it one rule.
    world.reachable_instance = reachable

    # Where the *task* says each piece of furniture is. This is the benchmark's own statement
    # about which physical object an instruction is about, and it is the same source the spawn
    # below uses to choose a support. `ground` needs it for the same reason: the belief answers
    # "where does the robot think it is", which is a different question from "which one is this
    # task about", and only the first should cost search.
    world.declared_room = lambda n: ((truth.get("objects") or {}).get(n) or {}).get("room")

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
        # Where *on* the support. `reachable_point_on` defaults to the edge cell nearest
        # the support's centre, which keeps the object on the furniture but says nothing
        # about the side the robot will approach from. On a bed that is the difference
        # between a task and an impossible one: the robot drove to the bed, stopped at a
        # routable stance, and the cardstock sat 2.20 m away across the mattress - past the
        # 1.50 m arm - so `NAVIGATE_TO(bed)` could never make `GRASP(cardstock)` work.
        # Anchoring the placement at a stance the robot can actually reach the support from
        # makes "drive to the support, then take the thing off it" hold for large furniture
        # as well as for a countertop.
        stance = next((world.to_world(*st) for st in probe.stances_for(support)
                       if probe.route_to(st) is not None), None)
        where = world.reachable_point_on(support, near=stance) if stance else None
        world.add_object(spawn["name"], spawn["name"], position=where, **{key: support})
        placed.append(spawn["name"])
    return world, placed


def _same_name(a, b):
    """Two spellings of one name, for reading the belief's own relation records."""
    return a is not None and b is not None and (a == b or match(a, [b]) is not None)


def ground(name, world, graph):
    """A category name from the plan onto an instance in this world.

    Two questions, and the belief answers both. *Which category* did the plan mean -
    `object_names.match` decides that, and the world's own categories are the pool, so the
    simulator is no stricter about names than the scorer is. Then *which instance* - the
    house holds nine countertops and three top cabinets.

    The believed room settles both. Where several categories fit the name equally well
    (`cabinet` is as much a `top_cabinet` as a `bottom_cabinet`), the one with an instance
    in the room the belief names wins; picking the shorter string instead is a coin flip
    dressed up as a rule. And within a category the instance in that room wins, because
    going to a counter in another room is a failure that reads as navigation rather than as
    grounding.
    """
    if name is None or name in world.truth.objects:
        return name

    # The belief may hold this object under a different spelling of the same name - the
    # task says "the bedroom cabinet", so the belief has `cabinet`, while the goal it is
    # being scored against says `bottom_cabinet`. Looking the room up by exact key misses
    # that, and a missing room drops the preference below and grounds the goal to an
    # arbitrary instance: all four "put it away in the <room> cabinet" tasks were scored
    # against a cabinet in a different room entirely - one of them in the utility room, for
    # a task about a bedroom - which no plan could have satisfied. The same matching rule
    # the plan is grounded with settles it.
    objects = graph.get("objects") or {}
    believed = (objects.get(name) or {}).get("room")
    if believed is None:
        key = match(name, list(objects))
        if key:
            believed = (objects.get(key) or {}).get("room")

    def in_believed_room(category):
        return any(world.room_of(i) == believed for i in world.truth.by_category(category))

    pool = {world.truth.objects[o]["category"] for o in world.truth.object_names()}
    category = match(name, pool, prefer=in_believed_room if believed else None)
    instances = world.truth.by_category(category) if category else []
    if not instances:
        return name

    # Which instance. The room is only the RSN's guess and it is wrong often enough to
    # matter - it put a top cabinet in the bathroom when the mug's was in the child's room,
    # and the plan then opened a cabinet in the wrong house-half. The *relations* are not a
    # guess: the task says the toothbrush is in the bathroom cabinet, and the setup put it
    # in one particular cabinet, so the cabinet holding the toothbrush is the one the task
    # is about. Ask that first, and fall back to the room only when no relation names it.
    held = [rel for rel in (graph.get("relations") or [])
            if rel.get("to") == name or _same_name(rel.get("to"), name)]
    for rel in held:
        moved = rel.get("from")
        edge = "object_inside" if str(rel.get("relation", "")).upper() == "INSIDE" else "on_top"
        for _, support in world.truth.edges_of(edge, src=moved):
            if support in instances:
                return support

    # Which instance, when no relation names one. The believed room comes first, but the
    # belief is the RSN's guess and can name a room no instance is in - and then this used to
    # take `instances[0]`, which is an arbitrary pick that can land in a part of the house the
    # robot cannot walk to. On `Wainscott_0_int-09` it did exactly that: six coffee tables,
    # the belief guessing `bedroom_0` where there is none, and the first instance sitting in
    # `living_room_2`, across the gap that splits that scene in two. The reference plan itself
    # then fails, so the task is unsolvable before any planning happens - and the benchmark's
    # own verification never saw it, because `build_tasks.verify` grounds against the truth,
    # where the right instance is found by room.
    #
    # An unreachable instance cannot be the one the task meant: every task here is verified
    # achievable. So prefer reachable, exactly as the spawn does above.
    # Which instance, when no relation names one. Four preferences, weakest assumption last.
    #
    # The room the *task declares* comes first. `Wainscott_0_int` has four console tables and the
    # subtask says the plate goes on the one in `living_room_1`; the RSN guessed `corridor_0`,
    # where there is none, and this used to fall through to `instances[0]` - the console table in
    # `bedroom_0`. The robot then swept all twelve rooms without finding it, because that room's
    # centroid is on the far side of the gap that splits the scene's room graph, so the sweep can
    # never enter it. One subtask broken that way took eighteen multi-task instructions with it,
    # and the benchmark's own verification could not see any of them: it grounds against the
    # truth, where the room lookup finds the right table immediately.
    #
    # Then the believed room, which is what the plan was written against. Then merely somewhere
    # the robot can get to. A wrong belief should cost the robot a search, not make the task
    # impossible, and binding the name to a table the task never meant is what made it impossible.
    declared = getattr(world, "declared_room", None)
    want = declared(name) if declared else None
    can_reach = getattr(world, "reachable_instance", None)
    stated = [i for i in instances if want and world.room_of(i) == want]
    here = [i for i in instances if world.room_of(i) == believed]
    if can_reach is None:
        return (stated or here or instances)[0]
    return (next((i for i in stated if can_reach(i)), None)
            or next((i for i in here if can_reach(i)), None)
            or next((i for i in instances if can_reach(i)), None)
            or (stated or here or instances)[0])


def route_matrix(task, graph, steps, start_room=None):
    """Exact pairwise drive distances between everything a plan navigates to.

    Oracle knows where every object is, so its route is fully determined by the ordering and
    it never searches. That means the route can be *computed* rather than driven - and it can
    be computed exactly, not estimated, because this uses the same A* on the same eroded grid
    the simulator itself drives on, between the same stance cells.

    What makes a single matrix enough is a property `build_multitask.combine` enforces: the
    errands in one instruction **share no movable object**. So every `NAVIGATE_TO` target is
    either a source object still at its spawn position or a fixed piece of furniture; nothing
    a later errand drives to has been moved by an earlier one, and every target's position is
    known before any ordering is chosen. Any ordering's cost is then a sum of consecutive legs
    out of this matrix, and 120 permutations cost 120 additions instead of 120 simulations.

    Returns `(legs, start)` where `legs[(a, b)]` is metres from target `a` to target `b` and
    `start[a]` is metres from where the robot begins. Unreachable pairs are absent.
    """
    from floor_world import astar

    plan = [(s["action"], s.get("object")) if isinstance(s, dict) else (s[0], s[1])
            for s in steps]
    world, _ = build_world(task, graph, [a for _, a in plan])
    # Keyed by the name the PLAN uses, not the instance it grounds to: `route_cost` walks a
    # plan, and grounding is a deterministic function of (name, world, belief), so the two
    # agree by construction while the caller never has to ground anything itself.
    names = list(dict.fromkeys(arg for action, arg in plan
                               if action == "NAVIGATE_TO" and arg))
    probe = Sim2D(world, start_room=start_room, room_hints={}, focus=set(), verbose=False)

    mask, _labels = world.traversable(probe.radius)
    cells = {}
    for name in names:
        instance = ground(name, world, graph)
        if not instance:
            continue
        stance = next((st for st in probe.stances_for(instance)
                       if probe.route_to(st) is not None), None)
        if stance is not None:
            cells[name] = stance
    begin = world.nearest_free_cell(probe.x, probe.y, probe.radius)

    def metres(route):
        if not route:
            return None
        total, (row, col) = 0.0, route[0]
        x, y = world.to_world(row, col)
        for nrow, ncol in route[1:]:
            nx, ny = world.to_world(nrow, ncol)
            total += math.hypot(nx - x, ny - y)
            x, y = nx, ny
        return total

    legs, start = {}, {}
    for name, cell in cells.items():
        if begin is not None:
            got = metres(astar(mask, begin, cell))
            if got is not None:
                start[name] = got
    for a, cell_a in cells.items():
        for b, cell_b in cells.items():
            if a == b:
                legs[(a, b)] = 0.0
                continue
            got = metres(astar(mask, cell_a, cell_b))
            if got is not None:
                legs[(a, b)] = got
    return legs, start


def route_cost(plan, legs, start, world=None, graph=None):
    """What driving `plan` costs, out of a `route_matrix`. None if any leg is unreachable."""
    total, here = 0.0, None
    for action, name in plan:
        if action != "NAVIGATE_TO" or not name:
            continue
        if here is None:
            if name not in start:
                return None
            total += start[name]
        else:
            if (here, name) not in legs:
                return None
            total += legs[(here, name)]
        here = name
    return total


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
    # The goal is grounded **here**, before a single action runs, for the same reason the
    # plan is. `ground` picks between several instances of a category by asking which one
    # the task's relations name - the coffee table the vase is on - and running the plan
    # moves the vase off it. Grounding the goal afterwards therefore asked a question whose
    # answer the plan had already changed: on all four swap tasks the plan bound
    # `coffee_table` to the table the vase started on and the goal bound it to a different
    # one, so a run that did exactly what was asked was marked as missing it.
    goal = []
    for entry in task.get("goal", ()):
        edge_type, src, dst = entry
        dst = ground(dst, world, graph) if isinstance(dst, str) else dst
        goal.append((edge_type, ground(src, world, graph), dst))
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
            # `steps_run` counts the plan's own actions; `sim_steps` counts the simulator's
            # control steps, which is what wall-clock is charged against at SIM_STEP_SECONDS
            # apiece. A plan of ten actions can cost hundreds of steps if most of them are
            # searching.
            "sim_steps": sim.steps,
            "sim_seconds": round(sim.steps * SIM_STEP_SECONDS, 1),
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
