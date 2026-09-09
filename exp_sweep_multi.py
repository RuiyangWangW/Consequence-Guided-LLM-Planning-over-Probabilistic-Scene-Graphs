#!/usr/bin/env python3
"""Which multi-task instructions and errands are impossible against a *belief*?

THE DEFECT CLASS THIS HUNTS

`build_tasks.verify` - the function that gates every task in `data/multitask.json` and
`data/subtasks.json` - drives each reference plan against `build_tasks.seed_graph(task)`,
the TRUTH. Every real evaluation drives plans against a BELIEF from `scene_graph.populate`,
and `sim_eval.ground` binds a category name in the plan (and in the goal) to a concrete
INSTANCE using that belief. The two can disagree, and where they do the benchmark's own
verification is blind: a task can be verified achievable and be impossible for every
planner that ever sees it.

That is not hypothetical. `Wainscott_0_int` has six coffee tables; the RSN guessed
`bedroom_0`, where there is none; `ground` fell through to `instances[0]`, which sits in
`living_room_2` on the far side of the gap that splits the scene in two; the *reference*
plan then failed. A fix is now in `sim_eval.ground`: it prefers instances the robot can
reach, sharing the memoised `world.reachable_instance` predicate `build_world` already
used to pick spawn supports. This sweep runs WITH that fix in place.

WHAT IT DOES

For every task in both files:

    truth   = run_plan(task, seed_graph(task), task["plan"])        what verify checks
    belief  = run_plan(task, populate(...),    task["plan"])        what evaluation drives

and reports every task where TRUTH PASSES and BELIEF FAILS. That gap is the ceiling on
anything measured against a belief: a task whose own known-good plan cannot be executed is
not a task a planner can be marked down for failing.

TELLING A DEFECT FROM A COST

A wrong RSN guess is *supposed* to cost something - the robot sweeps the wrong room first
and drives further. The defect is when it makes the task IMPOSSIBLE. The discriminator is
that `ground` is a function of (name, world, belief) and NOT of the plan: every plan that
names the same category gets the same instance. So when the belief binds a plan name to a
different instance than the truth does, no other plan can rescue it by being cleverer -
only by naming a different category. Hence the classification in `classify`:

    binding-divergence   some plan or goal name binds to a different instance under the
                         belief than under the truth. Sub-classified by whether that
                         instance is reachable at all - `world.reachable_instance`, the
                         same predicate the fix uses - and by which room it is in.
    search-exhausted     bindings agree; the run died searching. The belief only supplied
                         the room ordering, so this is a wrong guess costing what a wrong
                         guess should cost - unless the target is unreachable, in which
                         case no ordering would have found it.
    other                anything else, printed verbatim so it can be read.

RUNNING

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python exp_sweep_multi.py --shards 8 --shard 0
    ...                                                              --shards 8 --shard 7
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python exp_sweep_multi.py --merge

A task costs about 3-4 s (populate ~1 s, two driven runs ~1 s and ~2 s), so 618 tasks is
roughly 40 minutes in one process and five in eight. Sharding is by index modulo, so every
shard sees all ten scenes and a partial sweep is still a fair sample. Nothing is modified
in place: every module is imported and used exactly as the pipeline uses it.
"""

import argparse
import json
import os
import sys
import time
import traceback

MULTI = "data/multitask.json"
SUBS = "data/subtasks.json"
STAMP = "data/multitask-stamp.json"
OUT = "data/exp-sweep-multi.json"


# --------------------------------------------------------------------------- the world

def load_tasks(which):
    """The two task files, tagged with which set they came from."""
    out = []
    if which in ("multi", "both"):
        for t in json.load(open(MULTI)):
            out.append(("multi", t))
    if which in ("subs", "both"):
        for t in json.load(open(SUBS)):
            out.append(("subs", t))
    return out


def stamp():
    """The content hash of the multi-task set actually on disk, so a run says what it ran
    against rather than what it was told to expect."""
    try:
        return json.load(open(STAMP))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def belief_for(task):
    """The belief a real evaluation drives against: extraction ground truth through the RSN."""
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate

    extraction = task["extraction"]
    return populate(task["scene"], extraction["uncertain"], extraction["dependent"],
                    stated=extraction.get("stated") or {},
                    model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)


# ------------------------------------------------------------------------ the diagnosis

def _names_of(task):
    """Every category name the plan and the goal ask `ground` about."""
    names = [b for _, b in task["plan"] if b]
    for entry in task.get("goal", ()):
        _, src, dst = entry
        for n in (src, dst):
            if isinstance(n, str):
                names.append(n)
    return list(dict.fromkeys(names))


def diagnose(task, belief):
    """How the belief and the truth bind this task's names, in ONE shared world.

    The world is built from the task's own ground truth either way - `build_world` places
    spawns by `seed_graph`, never by the belief - so a single world answers both questions
    and the two `ground` calls are comparable object for object.
    """
    from build_tasks import seed_graph
    from sim_eval import build_world, ground

    truth = seed_graph(task)
    world, placed = build_world(task, belief, [b for _, b in task["plan"] if b])
    reach = getattr(world, "reachable_instance", None)

    rows = {}
    for name in _names_of(task):
        b = ground(name, world, belief)
        t = ground(name, world, truth)
        rows[name] = {
            "belief_instance": b, "truth_instance": t, "same": b == t,
            "belief_room": world.room_of(b) if b in world.truth.objects else None,
            "truth_room": world.room_of(t) if t in world.truth.objects else None,
            "belief_reachable": bool(reach(b)) if (reach and b in world.truth.objects) else None,
            "truth_reachable": bool(reach(t)) if (reach and t in world.truth.objects) else None,
            "believed_room": ((belief.get("objects") or {}).get(name) or {}).get("room"),
            "instances": len(world.truth.by_category(world.category_of(b))
                             if b in world.truth.objects else []),
        }
    return {"bindings": rows, "spawned": placed}


def classify(task, verdict, diag):
    """Name the shape of one belief-side failure. See the module docstring.

    Divergence on its own is NOT a defect and must not be reported as one. `run_plan`
    grounds the plan and the goal through the same belief, so a name that binds to a
    different-but-workable instance stays self-consistent: the robot puts the plate on the
    console table the belief means and the goal asks about that same table, and the run
    passes. `Wainscott_0_int-m01` does exactly this - `console_table` binds to
    `console_table_emeeke_0` under the belief and `..._2` under the truth - and it passes
    both ways. What matters is whether the divergence lands on the name the run actually
    died on, which is why the failing step's own argument is looked up here rather than
    the bindings being counted in bulk.
    """
    diverged = {n: r for n, r in diag["bindings"].items() if not r["same"]}
    why = verdict.get("why") or ""
    at = verdict.get("failed_at")
    culprit = None
    if at is not None and at < len(task["plan"]):
        culprit = task["plan"][at][1] or None
    row = diverged.get(culprit) if culprit else None

    if row is not None:
        tail = "unreachable" if row["belief_reachable"] is False else "other-room"
        return f"binding-divergence/at-failure/{tail}", {culprit: row}
    if at is not None:
        # The failing name binds the same way the truth binds it, so the belief only chose
        # where to look. Unreachable-but-agreed is a different animal: no search order
        # would have helped, and the truth run passing means the truth run never had to
        # touch it.
        bound = diag["bindings"].get(culprit) or {}
        if bound.get("belief_reachable") is False:
            return "agreed-binding/unreachable", {culprit: bound}
        if "is in none of" in why or "is not in" in why or "never been seen" in why:
            return "search-exhausted", ({culprit: bound} if culprit else {})
        return "step-failed/other", ({culprit: bound} if culprit else {})
    if not verdict.get("goal_met", True):
        return ("goal-not-met/divergence" if diverged else "goal-not-met/other"), diverged
    if verdict.get("unsafe"):
        return "left-unsafe", {}
    return "other", diverged


# ----------------------------------------------------------------------------- the sweep

def one(kind, task):
    """Both verdicts and, when they disagree, the diagnosis. Never raises."""
    from build_tasks import seed_graph
    from sim_eval import run_plan

    row = {"id": task["id"], "scene": task["scene"], "set": kind,
           "subgoals": len(task.get("subgoals") or []),
           "rooms": len(set((task.get("rooms") or {}).values()))}
    began = time.time()
    try:
        belief = belief_for(task)
        row["truth"] = run_plan(task, seed_graph(task), task["plan"], verbose=False)
        row["belief"] = run_plan(task, belief, task["plan"], verbose=False)
        if row["truth"]["ok"] and not row["belief"]["ok"]:
            diag = diagnose(task, belief)
            row["diagnosis"] = diag
            row["class"], row["diverged"] = classify(task, row["belief"], diag)
    except Exception as exc:                       # a harness fault, not a task fault
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-2000:]
    row["seconds"] = round(time.time() - began, 2)
    return row


def sweep(args):
    if args.without:
        without(args.without)
    tasks = load_tasks(args.set)
    if args.scene:
        tasks = [(k, t) for k, t in tasks if t["scene"] == args.scene]
    if args.ids:
        want = set(args.ids.split(","))
        tasks = [(k, t) for k, t in tasks if t["id"] in want]
    if args.stride > 1:
        tasks = tasks[::args.stride]
    if args.limit:
        tasks = tasks[:args.limit]
    mine = [(i, k, t) for i, (k, t) in enumerate(tasks) if i % args.shards == args.shard]
    out = args.out or (OUT if args.shards == 1
                       else OUT.replace(".json", f".shard{args.shard}.json"))
    rows, started = [], time.time()
    for n, (i, kind, task) in enumerate(mine, 1):
        row = one(kind, task)
        rows.append(row)
        flag = ("ERR " if "error" in row else
                "ok  " if row["belief"]["ok"] else
                "GAP " if row["truth"]["ok"] else "both")
        print(f"[{args.shard}] {n:3d}/{len(mine)} {row['id']:24s} {flag} "
              f"{row.get('class','')} {(row.get('belief') or {}).get('why','')[:70]}",
              flush=True)
        with open(out, "w") as f:
            json.dump({"complete": n == len(mine), "stamp": stamp(),
                       "shard": args.shard, "shards": args.shards,
                       "without": args.without,
                       "set": args.set, "stride": args.stride,
                       "seconds": round(time.time() - started, 1), "rows": rows}, f, indent=1)
    print(f"[{args.shard}] {len(rows)} tasks in {time.time() - started:.0f}s -> {out}")
    return 0


# ---------------------------------------------------------------------------- the report

def components(scene):
    """Connected groups of rooms, by A* on the floor the robot drives on.

    `cost_matrix.build` gives `inf` for a pair no route joins, which is the only honest
    answer: a plan that needs such a pair is not expensive, it is impossible.
    """
    import cost_matrix

    table = cost_matrix.build(scene)
    rooms = table["rooms"]
    parent = {r: r for r in rooms}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for a in rooms:
        for b in rooms:
            if table["distance"][f"{a}|{b}"] != float("inf"):
                parent[find(a)] = find(b)
    groups = {}
    for r in rooms:
        groups.setdefault(find(r), []).append(r)
    return sorted(groups.values(), key=len, reverse=True)


def merge(args):
    rows, stamps = [], []
    paths = args.inputs or sorted(
        os.path.join("data", f) for f in os.listdir("data")
        if f.startswith("exp-sweep-multi.shard") and f.endswith(".json"))
    if not paths and os.path.exists(OUT):
        paths = [OUT]
    for path in paths:
        blob = json.load(open(path))
        rows += blob["rows"]
        stamps.append((path, blob.get("stamp", {}).get("content"), blob.get("complete")))
    rows.sort(key=lambda r: (r["set"], r["id"]))

    print("=" * 78)
    print("STAMP the sweep actually ran against")
    for path, content, complete in stamps:
        print(f"   {os.path.basename(path):34s} content={content}  complete={complete}")
    print(f"   data/multitask-stamp.json on disk now: {stamp().get('content')}")

    for kind in ("multi", "subs"):
        sub = [r for r in rows if r["set"] == kind]
        if not sub:
            continue
        errors = [r for r in sub if "error" in r]
        good = [r for r in sub if "error" not in r]
        truth_ok = [r for r in good if r["truth"]["ok"]]
        belief_ok = [r for r in good if r["belief"]["ok"]]
        gap = [r for r in truth_ok if not r["belief"]["ok"]]
        both_bad = [r for r in good if not r["truth"]["ok"]]
        print("\n" + "=" * 78)
        print(f"{kind}: {len(sub)} tasks swept  ({len(errors)} harness errors)")
        print(f"   truth  passes {len(truth_ok):4d}/{len(good)}"
              f"   -> {len(both_bad)} reference plans fail against the TRUTH")
        print(f"   belief passes {len(belief_ok):4d}/{len(good)}")
        print(f"   GAP (truth ok, belief FAILS): {len(gap)}")
        by_class = {}
        for r in gap:
            by_class.setdefault(r.get("class", "?"), []).append(r)
        for name, group in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
            scenes = {}
            for r in group:
                scenes[r["scene"]] = scenes.get(r["scene"], 0) + 1
            print(f"      {name:34s} {len(group):3d}   "
                  + ", ".join(f"{s}:{c}" for s, c in sorted(scenes.items())))
        if both_bad:
            print("\n   reference plans that fail against the TRUTH:")
            for r in both_bad:
                print(f"      {r['id']:26s} {r['truth']['why'][:78]}")
        if gap:
            print("\n   every task in the gap:")
            for r in gap:
                print(f"      {r['id']:26s} {r.get('class','?'):32s} {r['belief']['why'][:60]}")
                for n, b in sorted((r.get("diverged") or {}).items()):
                    print(f"          {n:18s} belief->{b['belief_instance']} "
                          f"({b['belief_room']}, reachable={b['belief_reachable']})  "
                          f"truth->{b['truth_instance']} ({b['truth_room']})"
                          f"  believed_room={b['believed_room']}")
        if errors:
            print("\n   harness errors:")
            for r in errors:
                print(f"      {r['id']:26s} {r['error'][:78]}")

    if args.components:
        print("\n" + "=" * 78)
        print("scene connectivity (A* on the driven floor, via cost_matrix)")
        for scene in sorted({r["scene"] for r in rows}):
            groups = components(scene)
            note = "one piece" if len(groups) == 1 else f"{len(groups)} PIECES"
            print(f"   {scene:20s} {note}")
            if len(groups) > 1:
                for g in groups:
                    print(f"        {sorted(g)}")
    return 0


def probe(args):
    """Everything about ONE task, for a reproduction somebody else can run.

    Four questions, in the order that turns "the belief run failed" into "the task is
    impossible":

      1. WHAT BINDS.        `ground` under the belief and under the truth, side by side.
      2. IS IT REACHABLE.   Two answers, and they disagree, which is the point.
         `world.reachable_instance` - the predicate the fix installed - asks whether any
         cell the robot can route to lies within `REACH` of the object. That is a straight
         line and it goes through walls. `Sim2D.go_to_room` asks whether any standable
         floor *inside the object's room* is in the robot's own connected region, which is
         the question `navigate_to` actually has to answer for an object never seen, since
         finding one means entering its room and sweeping it.
      3. DID THE FIX MOVE IT. `ground` consults `reachable_instance` through `getattr`, so
         deleting the attribute reproduces the pre-fix binding exactly, on the same world.
      4. COULD ANY PLAN WIN. `ground` is a function of (name, world, belief) and not of the
         plan, so every plan naming the category gets the same instance - and so does the
         GOAL. The oracle plan below names the truth instance directly, bypassing grounding
         for the plan while the goal is still grounded through the belief. If that run also
         misses the goal, no plan over these names can pass and the task is impossible.
    """
    from build_tasks import seed_graph
    from sim2d import REACH, Sim2D
    from sim_eval import build_world, ground, run_plan

    tasks = {t["id"]: (k, t) for k, t in load_tasks("both")}
    kind, task = tasks[args.probe]
    belief = belief_for(task)
    truth = seed_graph(task)
    world, placed = build_world(task, belief, [b for _, b in task["plan"] if b])
    sim = Sim2D(world, verbose=False)

    print(f"{task['id']}  [{kind}]  {task['scene']}")
    print(f"   {task['task']}")
    print(f"   robot starts at ({sim.x:.1f}, {sim.y:.1f}) in room {sim.room}")
    print(f"   spawned {placed}")

    print("\n1-3. bindings")
    for name in _names_of(task):
        b = ground(name, world, belief)
        t = ground(name, world, truth)
        if b not in world.truth.objects:
            print(f"   {name:18s} (a spawned object, not furniture)")
            continue
        room = world.room_of(b)
        enterable = sim.go_to_room(room)[0] is not None if room else None
        stances = sim.stances_for(b)
        near = [c for c in stances
                if sim.route_to(c) is not None
                and world.distance_to(b, *world.to_world(*c)) <= REACH]
        # The pre-fix binding: `ground` reads the predicate off the world with getattr.
        saved = world.reachable_instance
        del world.reachable_instance
        before = ground(name, world, belief)
        world.reachable_instance = saved
        print(f"   {name:18s} believed_room={((belief.get('objects') or {}).get(name) or {}).get('room')}")
        print(f"      belief -> {b:26s} room={room:16s} reachable_instance={world.reachable_instance(b)} "
              f"routable-stances-within-{REACH}m={len(near)}/{len(stances)}  room enterable={enterable}")
        print(f"      truth  -> {t:26s} room={world.room_of(t)}")
        print(f"      pre-fix binding was {before}  ({'the fix moved it' if before != b else 'unchanged by the fix'})")

    print("\n4. could any plan win?")
    verdict = run_plan(task, belief, task["plan"], verbose=False)
    print(f"   reference plan vs belief: ok={verdict['ok']}  {verdict['why'][:100]}")
    print(f"   truth run:                ok={run_plan(task, truth, task['plan'])['ok']}")
    oracle = [[a, (ground(b, world, truth) if b else b)] for a, b in task["plan"]]
    verdict = run_plan(task, belief, oracle, verbose=False)
    print(f"   ORACLE plan (every name replaced by the TRUTH instance, so the plan is not")
    print(f"   grounded at all and drives to exactly the right furniture):")
    print(f"      ok={verdict['ok']}  {verdict['why'][:100]}")
    print(f"      missing={verdict['missing']}")
    print("   If that misses the goal, the goal itself is bound to the wrong instance and")
    print("   no plan expressible in these category names can satisfy it.")
    return 0


def without(which):
    """Undo one or both of today's grounding fixes AT RUNTIME, to attribute what each buys.

    `sim_eval.ground` reads both of its instance preferences off the world with `getattr`,
    so removing the attribute after `build_world` has returned reproduces the older
    behaviour exactly, on the same world and the same belief - no editing, no branch:

        --without reach       drop `world.reachable_instance` - the fix landed this morning
                              that makes `ground` prefer an instance the robot can get to
        --without declared    drop `world.declared_room`      - the fix that makes `ground`
                              prefer the room the TASK declares over the room the RSN guessed
        --without reach,declared                              - the benchmark as it stood
                              before either, which is the baseline the 22 known failures
                              were counted against

    Only `ground` is affected. `build_world` chooses its spawn supports with a closure it
    captured before returning, so the objects still start exactly where the task puts them
    and the world under test does not move.
    """
    import sim_eval

    drop = {w.strip() for w in which.split(",") if w.strip()}
    original = sim_eval.build_world

    def build_world(task, graph, plan_objects=(), rng=None):
        world, placed = original(task, graph, plan_objects, rng)
        if "reach" in drop and hasattr(world, "reachable_instance"):
            del world.reachable_instance
        if "declared" in drop and hasattr(world, "declared_room"):
            del world.declared_room
        return world, placed

    sim_eval.build_world = build_world
    return original


def tighten():
    """Install the candidate fix AT RUNTIME, without touching `sim_eval` on disk.

    `build_world` decides that an instance is reachable when *some cell the robot can
    route to* lies within `REACH` of it. `REACH` is a straight line to the object's near
    edge and it does not care what is between: on `Wainscott_0_int` the eight best stances
    for `console_table_emeeke_0` are cells whose `room_ids` is 0 - unlabelled threshold
    floor - sitting in the robot's connected region 1, while every standable cell of
    `bedroom_0` itself is in region 2. The predicate says "reachable" for a table on the
    far side of the wall that splits the house.

    The conjunct added here is the question `navigate_to` actually asks of an object it
    has never seen: it drives into the object's ROOM and sweeps it, so the room must hold
    standable floor in the robot's own region. That is `Sim2D.go_to_room`'s test, reused
    rather than restated.

    Note `room_mask`, not `room_at`. `room_at` answers with the majority room within 0.6 m
    when a cell carries no segment id, which is exactly how a threshold cell outside the
    bedroom comes back as `bedroom_0` and is the reason the two disagree at all.
    """
    import sim_eval
    from sim2d import Sim2D

    original = sim_eval.build_world

    def build_world(task, graph, plan_objects=(), rng=None):
        world, placed = original(task, graph, plan_objects, rng)
        base = world.reachable_instance
        probe = Sim2D(world, verbose=False)
        mask, labels = world.traversable(probe.radius)
        region = world.region_of(probe.x, probe.y, probe.radius)
        enterable = {}

        def can_enter(room):
            if room not in enterable:
                enterable[room] = bool((world.room_mask(room) & mask
                                        & (labels == region)).any())
            return enterable[room]

        def reachable(instance):
            room = world.room_of(instance)
            return bool(base(instance)) and (room is None or can_enter(room))

        world.reachable_instance = reachable
        return world, placed

    sim_eval.build_world = build_world
    return original


def counterfactual(args):
    """Re-drive the tasks in the gap with the tightened predicate, and count the scope.

    Two numbers. How many of the failing tasks pass once `ground` stops choosing an
    instance in a room the robot cannot enter - that is what the fix is worth. And how
    many furniture instances in each scene the current predicate calls reachable while
    their room is unenterable - that is how much unexploded ordnance is left in the other
    nine scenes even though today's task set does not step on it.
    """
    from build_tasks import seed_graph
    from sim2d import Sim2D
    from sim_eval import run_plan

    rows = []
    for path in sorted(os.path.join("data", f) for f in os.listdir("data")
                       if f.startswith("exp-sweep-multi.shard") and f.endswith(".json")):
        rows += json.load(open(path))["rows"]
    gap = [r for r in rows
           if "error" not in r and r["truth"]["ok"] and not r["belief"]["ok"]]
    tasks = {t["id"]: t for _, t in load_tasks("both")}

    print(f"re-driving the {len(gap)} tasks in the gap with the tightened predicate\n")
    tighten()
    fixed = 0
    for r in sorted(gap, key=lambda r: r["id"]):
        task = tasks[r["id"]]
        verdict = run_plan(task, belief_for(task), task["plan"], verbose=False)
        fixed += verdict["ok"]
        print(f"   {r['id']:26s} was FAIL -> {'PASS' if verdict['ok'] else 'still FAIL: ' + verdict['why'][:60]}",
              flush=True)
    print(f"\n{fixed}/{len(gap)} of the gap closes with the room-enterability conjunct")

    print("\nscope: instances the CURRENT predicate calls reachable whose room the robot"
          "\ncannot enter (one world per scene, all furniture loaded)")
    from build_tasks import world_for
    for scene in sorted({r["scene"] for r in rows}):
        example = next(t for t in tasks.values() if t["scene"] == scene)
        original = sim_eval_build_world_original[0]
        world, _ = original(example, seed_graph(example), [])
        probe = Sim2D(world, verbose=False)
        mask, labels = world.traversable(probe.radius)
        region = world.region_of(probe.x, probe.y, probe.radius)
        stuck = []
        for name in world.truth.object_names():
            room = world.room_of(name)
            if room is None or not world.reachable_instance(name):
                continue
            if not (world.room_mask(room) & mask & (labels == region)).any():
                stuck.append((name, room))
        n_rooms = len({r for _, r in stuck})
        print(f"   {scene:20s} {len(stuck):4d} instances in {n_rooms} unenterable rooms "
              + (", ".join(sorted({r for _, r in stuck})) if stuck else ""))
    return 0


sim_eval_build_world_original = []


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", default="both", choices=["multi", "subs", "both"])
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1,
                        help="subsample tasks[::stride]; every scene stays represented")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scene", help="sweep one scene only")
    parser.add_argument("--ids", help="comma-separated task ids to sweep")
    parser.add_argument("--out")
    parser.add_argument("--without", default="",
                        help="drop grounding fixes before sweeping: reach, declared, or both")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--inputs", nargs="*")
    parser.add_argument("--probe", help="dissect one task id and stop")
    parser.add_argument("--counterfactual", action="store_true",
                        help="re-drive the gap with the candidate fix installed")
    parser.add_argument("--components", action="store_true",
                        help="also report which scenes come in disconnected pieces")
    args = parser.parse_args()
    if args.probe:
        return probe(args)
    if args.counterfactual:
        import sim_eval
        sim_eval_build_world_original.append(sim_eval.build_world)
        return counterfactual(args)
    return merge(args) if args.merge else sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
