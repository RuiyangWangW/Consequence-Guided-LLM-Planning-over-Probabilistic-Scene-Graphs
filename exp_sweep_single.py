#!/usr/bin/env python3
"""Every one of the 100 single tasks, driven against its own BELIEF as well as the truth.

The benchmark verifies itself with `build_tasks.verify` / `run_reference_sim.py`, which
drive each task's reference plan against `build_tasks.seed_graph(task)` - ground truth.
But nothing is ever *evaluated* that way. A real run gets a belief from
`scene_graph.populate`, and `sim_eval.ground` binds the plan's category names to instances
using that belief. So a task can be verified achievable and still be impossible in every
run that matters:

    `Wainscott_0_int-09` has six coffee tables. The RSN guessed `bedroom_0`, which has
    none, and `ground` fell through to `instances[0]` - the table in `living_room_2`,
    across the gap that splits that scene into two disconnected halves. The reference plan
    failed, so no planner could have solved the task, and verification never saw it because
    it grounds against the truth, where the right table is found by room.

That defect is fixed (`ground` now prefers instances the robot can reach, sharing the
memoised `world.reachable_instance` predicate `build_world` already used for spawn
supports). This sweep asks what else is in that class, with the fix in place:

    for every task: belief = populate(task's own extraction answer key)
                    truth  = seed_graph(task)
                    run the task's OWN reference plan against each

A task where **truth passes and belief fails** is a benchmark defect of exactly that shape:
the known-good plan cannot be executed, so no planner can be marked down for failing it.
A task where both fail is a defect verification should already have caught; a task where
the belief costs distance but still finishes is the system working as intended - a wrong
guess should cost something.

Each defect gets a grounding table: which name bound to which instance, in which room,
under the belief and under the truth, whether that instance is reachable at all
(`world.reachable_instance`) and whether its room is reachable from the robot's start
(`cost_matrix`, where `inf` means a different half of the house).

    python exp_sweep_single.py --out data/exp-sweep-single.json
    python exp_sweep_single.py --explain Wainscott_0_int-09
"""

import argparse
import json
import math
import time
import traceback

from build_tasks import seed_graph, world_for
from object_names import match
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import build_world, ground, run_plan

TASKS = "data/tasks.json"

# ---------------------------------------------------------------- caches
# populate() reloads the RSN checkpoint and the frozen text encoder on every call, once
# per object name. Nothing about that changes the answer, so it is cached in-process; the
# beliefs themselves are cached per task id because the diagnostics re-ask for them.
_BELIEFS = {}
_COSTS = {}


def belief_for(task):
    """The belief a real evaluation would run this task against: populate() from the
    task's OWN stage-1 answer key, exactly as `evaluate.py` does."""
    if task["id"] not in _BELIEFS:
        extraction = task["extraction"]
        _BELIEFS[task["id"]] = populate(
            task["scene"], extraction["uncertain"], extraction["dependent"],
            stated=extraction.get("stated") or {},
            model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    return _BELIEFS[task["id"]]


def costs_for(scene):
    """A* room-to-room distances for one scene; `inf` for pairs in different halves."""
    if scene not in _COSTS:
        import cost_matrix
        _COSTS[scene] = cost_matrix.build(scene)
    return _COSTS[scene]


def stamp():
    """Which version of the code under test these numbers describe.

    `sim_eval.py` is being edited by other work while this runs - the grounding rule changed
    once mid-sweep - and a pass/fail table is worthless if nobody can tell which `ground` it
    was measuring. The digests go in the output and are re-read at the end, so a run that
    straddles an edit says so instead of averaging two different systems.
    """
    import hashlib

    out = {}
    for path in ("sim_eval.py", "scene_graph.py", "sim2d.py", "build_tasks.py", TASKS):
        try:
            with open(path, "rb") as handle:
                out[path] = hashlib.sha256(handle.read()).hexdigest()[:12]
        except OSError:
            out[path] = None
    return out


def plan_names(task):
    """Every category name the run has to bind: the plan's arguments and the goal's."""
    names = [a for _, a in task["plan"] if a]
    for edge_type, src, dst in task.get("goal", ()):
        names.append(src)
        if isinstance(dst, str):
            names.append(dst)
    return list(dict.fromkeys(n for n in names if n))


# ---------------------------------------------------------------- context: bad guesses


def wrong_rooms(task, belief):
    """Where the belief names a room the object is not in - the RSN guessing badly.

    Two flavours, because they are different mistakes:

      `no_instance`   the believed room holds no instance of that category at all. The
                      robot will sweep a room where the thing cannot be, which is the
                      guess that actually costs distance.
      `wrong_pinned`  the believed room differs from the room the truth graph pins the
                      category to (`furniture_rooms` majority vote). Softer: a category
                      with instances in three rooms is only "wrong" by a convention.
    """
    world = world_for(task["scene"])
    truth = seed_graph(task)
    truth_objects = truth.get("objects") or {}
    pool = {world.category_of(n) for n in world.truth.object_names()}
    pool.discard(None)

    rows = []
    for name, info in (belief.get("objects") or {}).items():
        believed = info.get("room")
        category = name if name in pool else match(name, sorted(pool))
        truth_room = (truth_objects.get(name) or {}).get("room")
        if truth_room is None and category:
            truth_room = (truth_objects.get(category) or {}).get("room")
        instances = world.truth.by_category(category) if category else []
        in_room = [i for i in instances if world.room_of(i) == believed]
        rows.append({
            "name": name, "category": category, "believed": believed,
            "truth_room": truth_room, "instances": len(instances),
            "in_believed_room": len(in_room),
            "no_instance": bool(category and instances and not in_room),
            "wrong_pinned": bool(truth_room and believed and truth_room != believed),
            "stated": bool(info.get("stated")),
            "via": info.get("via") is not None,
        })
    return rows


# ---------------------------------------------------------------- the grounding table


def grounding(task, belief, want_reach=True):
    """How every name the run binds resolves, under the belief and under the truth.

    One world serves both: `build_world` places the task's objects from `seed_graph` no
    matter which graph it is handed, and loads the union of both graphs' categories, so
    the house the belief run drives in and the house the truth run drives in are the same
    house. Only the *binding* differs, which is the whole question.
    """
    truth = seed_graph(task)
    names = plan_names(task)
    world, _ = build_world(task, belief, names)

    reach = getattr(world, "reachable_instance", None)
    rows = []
    for name in names:
        b = ground(name, world, belief)
        t = ground(name, world, truth)
        row = {"name": name, "belief_instance": b, "truth_instance": t,
               "belief_room": world.room_of(b) if b in world.truth.objects else None,
               "truth_room": world.room_of(t) if t in world.truth.objects else None,
               "same": b == t,
               "in_goal": any(name in (g[1], g[2]) for g in task.get("goal", ()))}
        if want_reach and reach is not None:
            for key, inst in (("belief_reachable", b), ("truth_reachable", t)):
                row[key] = bool(reach(inst)) if inst in world.truth.objects else None
        rows.append(row)
    return world, rows


def start_room(world):
    """Where `run_plan` parks the robot when no start room is given - the middle of the
    largest connected patch of standable floor - as (room, (x, y)).

    That patch's centre can land on floor no room polygon covers (a doorway seam, an open
    plan edge), and then `room_at` is None and every room-to-room distance keyed off it is
    missing. So fall back to the nearest room by centroid: the room-level question is
    "which half of the house is the robot in", and the nearest room answers it.
    """
    import numpy as np

    from sim2d import Sim2D

    sim = Sim2D(world, verbose=False)
    room = sim.room
    if room is None:
        best = None
        for candidate in world.rooms:
            cells = np.argwhere(world.room_mask(candidate))
            if not len(cells):
                continue
            x, y = world.to_world(*cells.mean(axis=0).round().astype(int))
            d = math.hypot(x - sim.x, y - sim.y)
            if best is None or d < best[0]:
                best = (d, candidate)
        room = best[1] if best else None
    return room, (sim.x, sim.y)


def explain(task, belief=None, out=print):
    """Everything needed to say WHY a task fails against its belief and not the truth."""
    belief = belief or belief_for(task)
    truth = seed_graph(task)
    world, rows = grounding(task, belief)
    here, pose = start_room(world)
    table = costs_for(task["scene"])
    distance = table["distance"]

    out(f"\n{task['id']}  {task['task']}")
    out(f"  scene {task['scene']}, robot starts at ({pose[0]:.1f}, {pose[1]:.1f}) in {here}")
    out(f"  belief rooms: " + ", ".join(
        f"{n}->{(i or {}).get('room')}" for n, i in (belief.get("objects") or {}).items()))
    out(f"  {'name':22s} {'belief instance':34s} {'room':16s} reach  D(start)   truth instance")
    for row in rows:
        d = distance.get(f"{here}|{row['belief_room']}", float('nan'))
        flag = "" if row["same"] else "   <-- DIFFERS"
        # `--no-fix` takes the reachability predicate off the world, so the column is
        # simply not there; a missing answer must read as "not asked", not crash the
        # explanation of the very failure that mode exists to produce.
        out(f"  {row['name']:22s} {str(row['belief_instance']):34s} "
            f"{str(row['belief_room']):16s} {str(row.get('belief_reachable', '?')):5s} "
            f"{d:8.1f}   {row['truth_instance']}{flag}")

    for label, graph in (("truth", truth), ("belief", belief)):
        verdict = run_plan(task, graph, task["plan"], verbose=False)
        out(f"  {label:6s} {'ok' if verdict['ok'] else 'FAIL'}  {verdict['driven']:6.1f} m  "
            f"{verdict['why'][:110]}")
    return rows


def diagnose(rows, verdict):
    """Name the failure's shape from the grounding table and the simulator's complaint."""
    why = verdict.get("why") or ""
    if any(r.get("belief_reachable") is False for r in rows):
        return "unreachable-instance"
    if any(not r["same"] and r["in_goal"] for r in rows):
        return "goal-instance-mismatch"
    if any(not r["same"] for r in rows):
        return "instance-mismatch"
    if "is in none of" in why or "has never been seen" in why:
        return "search-exhausted"
    if "beyond the" in why:
        return "out-of-reach"
    if "goal not met" in why:
        return "goal-not-met"
    return "other"


# ---------------------------------------------------------------- goal, believed


def believed_done(tasks, out=print):
    """Tasks whose goal is already true in the world the robot *believes* in.

    Same shape of gap as the grounding one, one stage earlier. `build_tasks.verify`
    refuses any task whose goal holds before the robot moves - but it tests the *seed*
    graph, and nothing tests the belief. A task that reads as finished in the belief is one
    the symbolic checker will accept an empty plan for, so the planner is being asked a
    question whose answer is "do nothing"; only the drive against the true world catches
    it. `evaluate.py` records this per run; this counts it over the whole benchmark, from
    each task's own answer-key extraction.
    """
    from graph_machine import GraphMachine
    from world_graph import WorldGraph

    hits = []
    for task in tasks:
        belief = belief_for(task)
        goal = [tuple(g) for g in task["goal"]]
        machine = GraphMachine(WorldGraph.from_scene_graph(belief))
        missing = machine.run([], goal).missing
        truth_missing = GraphMachine(
            WorldGraph.from_scene_graph(seed_graph(task))).run([], goal).missing
        if not missing:
            hits.append(task["id"])
            out(f"  {task['id']:22s} goal already true in the belief; "
                f"truth still needs {[list(m) for m in truth_missing]}")
    out(f"\n{len(hits)}/{len(tasks)} tasks read as already finished in their own belief")
    return hits


# ---------------------------------------------------------------- house halves


def scene_components(scene):
    """The scene's rooms grouped into what the robot can actually drive between.

    `Wainscott_0_int` and `Beechwood_0_int` are known to come in two pieces, but which
    scenes split - and which side the robot starts on - is a fact to compute, not to
    assume. Union-find over the finite entries of `cost_matrix`'s A* distance table.
    """
    table = costs_for(scene)
    rooms, distance = table["rooms"], table["distance"]
    parent = {r: r for r in rooms}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a in rooms:
        for b in rooms:
            if a != b and distance[f"{a}|{b}"] != float("inf"):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    groups = {}
    for room in rooms:
        groups.setdefault(find(room), []).append(room)
    return [sorted(v) for v in groups.values()]


def components_report(tasks, sweep_path=None, out=print):
    """Which scenes are in pieces, which piece the robot starts in, and whether any task
    binds a name to an instance on the far side - the `Wainscott_0_int-09` geometry, named
    before any plan runs."""
    first = {}
    for task in tasks:
        first.setdefault(task["scene"], task)
    rows = []
    if sweep_path:
        rows = json.load(open(sweep_path)).get("rows", [])

    for scene, task in sorted(first.items()):
        groups = scene_components(scene)
        world, _ = build_world(task, seed_graph(task), plan_names(task))
        here, _pose = start_room(world)
        home = next((g for g in groups if here in g), None)
        out(f"\n{scene}: {len(groups)} component(s), robot starts in {here}")
        for group in groups:
            mark = "robot" if group is home else " far "
            out(f"   [{mark}] {', '.join(group)}")
        if home is None:
            continue
        for row in rows:
            if row.get("scene") != scene:
                continue
            far = [b for b in row.get("bound", [])
                   if b.get("belief_room") and b["belief_room"] not in home]
            if far:
                out(f"   {row['id']} binds into the far half: "
                    + ", ".join(f"{b['name']}->{b['belief_instance']} in {b['belief_room']}"
                                for b in far)
                    + f"   (belief run ok={row['belief']['ok']})")


# ---------------------------------------------------------------- sensitivity

def disable_fix():
    """Put the pre-fix `ground` back, to check this sweep can see the defect it hunts.

    A sweep that finds nothing is only worth reading if it would have found something.
    The fix was to have `ground` prefer an instance the task's own graph declares the room
    of (`world.declared_room`), then one the robot can reach (`world.reachable_instance`).
    `build_world` attaches both to the world; taking both attributes off it - and nothing
    else - restores the old `instances[0]` fallback exactly, while leaving the spawn
    placement (which has its own local copy of the reachability test) alone. The tasks that
    fail under `--no-fix` and pass without it are the ones the fix rescued.

    Stripping only one is worth knowing too, and is what this did when `ground` had only
    the reachability preference: with `declared_room` still in place the old defect no
    longer reproduces, because the declared room now answers first.
    """
    import sim_eval

    original = sim_eval.build_world

    def patched(task, graph, plan_objects=(), rng=None):
        world, placed = original(task, graph, plan_objects, rng)
        for attribute in ("reachable_instance", "declared_room"):
            if hasattr(world, attribute):
                delattr(world, attribute)
        return world, placed

    sim_eval.build_world = patched
    globals()["build_world"] = patched      # grounding() holds its own reference
    return patched


# ---------------------------------------------------------------- the sweep


def sweep(tasks, out_path=None, verbose=True):
    rows, started, before = [], time.time(), stamp()
    for i, task in enumerate(tasks, 1):
        began = time.time()
        row = {"id": task["id"], "scene": task["scene"], "task": task["task"]}
        try:
            belief = belief_for(task)
            row["truth"] = run_plan(task, seed_graph(task), task["plan"], verbose=False)
            row["belief"] = run_plan(task, belief, task["plan"], verbose=False)
            row["rooms"] = wrong_rooms(task, belief)
            row["no_instance"] = sum(r["no_instance"] for r in row["rooms"])
            row["wrong_pinned"] = sum(r["wrong_pinned"] for r in row["rooms"])
            _, bound = grounding(task, belief, want_reach=False)
            row["bound"] = bound
            row["mismatched"] = [r["name"] for r in bound if not r["same"]]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["trace"] = traceback.format_exc()[-800:]
        row["seconds"] = round(time.time() - began, 1)
        rows.append(row)
        if verbose:
            t = row.get("truth", {}).get("ok")
            b = row.get("belief", {}).get("ok")
            mark = {(True, True): "..", (True, False): "DEFECT", (False, True): "??",
                    (False, False): "both-fail"}.get((t, b), "ERROR")
            print(f"{i:3d}/{len(tasks)} {task['id']:22s} truth={t} belief={b} "
                  f"{mark:9s} bad-rooms={row.get('no_instance', '?')} "
                  f"({row['seconds']:4.1f}s) {row.get('belief', {}).get('why', row.get('error',''))[:60]}",
                  flush=True)
        if out_path:
            with open(out_path, "w") as handle:
                json.dump({"complete": i == len(tasks), "code": before, "rows": rows},
                          handle, indent=1)
    after = stamp()
    if verbose:
        print(f"\nswept {len(rows)} tasks in {time.time() - started:.0f}s")
        print("code under test: " + ", ".join(f"{k}={v}" for k, v in before.items()))
        if after != before:
            changed = [k for k in before if after[k] != before[k]]
            print(f"  WARNING: {', '.join(changed)} changed while this ran - "
                  f"these numbers straddle an edit and should be re-run")
    if out_path:
        with open(out_path, "w") as handle:
            json.dump({"complete": True, "code": before, "code_after": after,
                       "rows": rows}, handle, indent=1)
    return rows


def report(rows, tasks_by_id):
    ok = lambda row, key: bool(row.get(key, {}).get("ok"))
    good = [r for r in rows if "error" not in r]
    defects = [r for r in good if ok(r, "truth") and not ok(r, "belief")]
    both = [r for r in good if not ok(r, "truth") and not ok(r, "belief")]
    inverted = [r for r in good if not ok(r, "truth") and ok(r, "belief")]
    errors = [r for r in rows if "error" in r]

    print("\n" + "=" * 78)
    print(f"{len(rows)} tasks; {len(errors)} harness errors")
    print(f"  truth-grounded pass : {sum(ok(r, 'truth') for r in good)}/{len(good)}")
    print(f"  belief-grounded pass: {sum(ok(r, 'belief') for r in good)}/{len(good)}")
    print(f"  DEFECTS (truth ok, belief fails): {len(defects)}")
    print(f"  both fail (verification should have caught): {len(both)}")
    print(f"  belief ok but truth fails: {len(inverted)}")

    for label, group in (("DEFECT", defects), ("BOTH FAIL", both), ("INVERTED", inverted)):
        for row in group:
            print(f"\n  [{label}] {row['id']}  {row['task'][:70]}")
            print(f"     truth : {row['truth']['ok']}  {row['truth']['why'][:90]}")
            print(f"     belief: {row['belief']['ok']}  {row['belief']['why'][:90]}")
            print(f"     names bound differently under the belief: {row['mismatched'] or 'none'}")
            print(f"     shape: {diagnose(row.get('bound') or [], row['belief'])}")

    for row in errors:
        print(f"\n  [ERROR] {row['id']}: {row['error']}")

    # Context: how often the RSN guesses a room the object is not in, and what it costs.
    bad = [r for r in good if r["no_instance"]]
    pinned = [r for r in good if r["wrong_pinned"]]
    clean = [r for r in good if not r["no_instance"]]
    print("\n" + "-" * 78)
    print("context - a wrong guess should cost distance, not the task:")
    print(f"  tasks whose belief names a room with NO instance of some object: {len(bad)}/{len(good)}"
          f"  -> {sum(ok(r, 'belief') for r in bad)} still pass")
    print(f"  tasks whose belief differs from the truth graph's pinned room  : {len(pinned)}/{len(good)}"
          f"  -> {sum(ok(r, 'belief') for r in pinned)} still pass")
    print(f"  tasks with a clean belief                                     : {len(clean)}/{len(good)}"
          f"  -> {sum(ok(r, 'belief') for r in clean)} pass")
    for label, group in (("bad-room", bad), ("clean", clean)):
        d = [r["belief"]["driven"] - r["truth"]["driven"] for r in group
             if ok(r, "belief") and ok(r, "truth")]
        if d:
            print(f"  extra metres driven vs the truth run, {label:8s}: "
                  f"mean {sum(d)/len(d):+6.1f} m over {len(d)} passing tasks")
    names = [n for r in good for n in r["rooms"]]
    print(f"  per-name: {sum(n['no_instance'] for n in names)}/{len(names)} believed rooms hold "
          f"no instance of the object; {sum(n['wrong_pinned'] for n in names)} differ from the pin")
    mism = [r for r in good if r["mismatched"]]
    print(f"  tasks where some name binds to a DIFFERENT instance under the belief: {len(mism)}"
          f"  -> {sum(ok(r, 'belief') for r in mism)} still pass")
    return defects


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--out", default="data/exp-sweep-single.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", nargs="*", help="task ids to run instead of all")
    parser.add_argument("--explain", help="one task id: print its grounding table and stop")
    parser.add_argument("--believed-done", action="store_true",
                        help="count tasks whose goal already holds in their own belief - "
                             "the check build_tasks.verify only makes against the truth")
    parser.add_argument("--components", action="store_true",
                        help="print each scene's connected pieces, the piece the robot "
                             "starts in, and any task that binds a name into another one")
    parser.add_argument("--no-fix", action="store_true",
                        help="restore the pre-fix instances[0] grounding, to show this "
                             "sweep can see the defect class it is hunting")
    args = parser.parse_args()

    if args.no_fix:
        disable_fix()
        print("running with the reachability-preferring ground() DISABLED\n")

    tasks = json.load(open(args.tasks))
    by_id = {t["id"]: t for t in tasks}

    if args.explain:
        explain(by_id[args.explain])
        return 0

    if args.believed_done:
        believed_done(tasks)
        return 0

    if args.components:
        components_report(tasks, args.out if __import__("os").path.exists(args.out) else None)
        return 0

    if args.only:
        tasks = [by_id[i] for i in args.only]
    if args.limit:
        tasks = tasks[:args.limit]

    rows = sweep(tasks, args.out)
    defects = report(rows, by_id)
    for row in defects:
        explain(by_id[row["id"]])
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
