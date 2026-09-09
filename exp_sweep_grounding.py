#!/usr/bin/env python3
"""Grounding consistency: does a category name mean the same instance everywhere?

The benchmark is verified against the TRUTH - `build_tasks.verify` drives each reference
plan through `build_tasks.seed_graph(task)`, where every category's room is read off the
scene. Every real evaluation drives the same plan against a BELIEF from
`scene_graph.populate`, and `sim_eval.ground` turns a category name (`countertop`) into one
instance (`countertop_tpuwys_3`) using that belief. Nothing in the benchmark checks that
those two answers agree, so a task can be verified achievable and be impossible in the run.

That is not hypothetical. `Wainscott_0_int-09` has six coffee tables; the RSN guessed
`bedroom_0`, which holds none; `ground` fell through to `instances[0]`, which sits in
`living_room_2` on the far side of the gap that splits that scene into two disconnected
halves. The reference plan itself then failed - the task was unsolvable before any planner
saw it - and verification never noticed, because it grounds against the truth where the
right table is found by room.

This sweep measures the mechanism rather than that one symptom, over all three benchmarks
(`data/tasks.json`, `data/multitask.json`, `data/subtasks.json`):

    1. POPULATION AT RISK. How many names in a plan or a goal have more than one instance
       in the scene, broken down by scene and category. A name with one instance cannot be
       mis-grounded; every count below is a fraction of this.

    2. PLAN vs GOAL. `run_plan` grounds the plan and the goal separately. If they disagree
       the plan is driven to one countertop and scored against another, and a correct run is
       marked wrong. Both bindings are computed and compared - at the moment the run starts,
       which is what the code does, and again after the plan has run, which is what the code
       used to do, so the size of that fix is a number rather than a story.

    3. TRUTH vs BELIEF. The same name grounded against `seed_graph(task)` (what verification
       uses) and against the belief (what evaluation uses). Where these differ the benchmark
       is checking a different object from the one the robot goes to. Where the belief's
       choice is also unreachable, the task is impossible and no plan can fix it.

    4. DECLARED ROOMS. How often the chosen instance is in a room other than the one
       `task["rooms"]` (or the scene's majority room) declares. Not a defect on its own -
       deciding is what the belief is for - but if it were most of them the declarations
       would be decorative.

    5. THE RELATION BRANCH. `ground` prefers an instance named by a task relation, and that
       branch is deliberately not reachability-guarded, because a stated relation is not a
       guess. Whether it ever picks an instance the robot cannot get to.

    6. THE FIX. Cases where the pre-fix rule (`next((i for i in instances if
       world.room_of(i) == believed), instances[0])`) and the post-fix rule disagree, with
       the reachability of both choices, and - with `--verify` - the reference plan run both
       ways so the claim "this task was impossible and now is not" is a reproduction.

Nothing here is modified in place: `ground_prefix` is a copy of the pre-fix function, and it
is monkey-patched over `sim_eval.ground` only inside `--verify`, only for the pre-fix run.

    python exp_sweep_grounding.py --which all
    python exp_sweep_grounding.py --which single --verify
    python exp_sweep_grounding.py --scene Wainscott_0_int --verify --verbose
"""

import argparse
import collections
import json
import os
import time

import sim_eval
from build_tasks import furniture_rooms, seed_graph, world_for
from object_names import match
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import build_world, ground, run_plan

BENCHMARKS = {"single": "data/tasks.json",
              "multi": "data/multitask.json",
              "sub": "data/subtasks.json"}

OUT = "logs/grounding_sweep.json"


# ------------------------------------------------------------------ speed, not semantics

def accelerate():
    """Memoise the frozen text encoder and the RSN checkpoint for the length of the sweep.

    `embed_categories.embed_names` builds a `SentenceTransformer` on **every call**, and
    `query_rsn.predict_rooms` calls it once per object name, and `query_rsn.load` re-reads
    the checkpoint on every `populate`. On a network-mounted dataset that was fifteen of the
    twenty seconds a task took, which is the difference between sweeping a hundred tasks and
    sweeping all seven hundred and eighteen.

    Nothing about an answer changes. The encoder is frozen and the model is in `eval()`, so
    the same name yields the same vector and the same distribution; this is memoisation, not
    an approximation, and `--check-cache` proves it by populating one task both ways and
    comparing the beliefs. The patch lives here and is applied at runtime - no module on
    disk is touched, because other agents are working in this repo.
    """
    import numpy as np

    import embed_categories
    import query_rsn

    encoders, vectors = {}, {}

    def embed_names(names, model_name=embed_categories.DEFAULT_MODEL, batch_size=64):
        missing = [n for n in names if (n, model_name) not in vectors]
        if missing:
            from sentence_transformers import SentenceTransformer

            if model_name not in encoders:
                encoders[model_name] = SentenceTransformer(model_name)
            fresh = encoders[model_name].encode(
                [embed_categories.name_to_text(n) for n in missing],
                batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False)
            for name, vec in zip(missing, fresh):
                vectors[(name, model_name)] = np.asarray(vec, dtype=np.float32)
        return np.stack([vectors[(n, model_name)] for n in names]).astype(np.float32)

    checkpoints, original_load = {}, query_rsn.load

    def load(path, device):
        key = (path, str(device))
        if key not in checkpoints:
            checkpoints[key] = original_load(path, device)
        return checkpoints[key]

    embed_categories.embed_names = embed_names
    query_rsn.load = load


def check_cache(task):
    """Populate one task with the memoisation off and on; the beliefs must be identical."""
    plain = populate(task["scene"], task["extraction"]["uncertain"],
                     task["extraction"]["dependent"],
                     stated=task["extraction"].get("stated") or {},
                     model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    accelerate()
    cached = populate(task["scene"], task["extraction"]["uncertain"],
                      task["extraction"]["dependent"],
                      stated=task["extraction"].get("stated") or {},
                      model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    same = json.dumps(plain, sort_keys=True) == json.dumps(cached, sort_keys=True)
    print(f"cache check on {task['id']}: beliefs {'identical' if same else 'DIFFER'}")
    return same


# ------------------------------------------------------------- reachability, once per scene

# Whether the robot can get to an instance is a fact about the SCENE, not about the task:
# the traversable map is eroded from the floor-plan raster, the robot's default start is the
# middle of the largest region of it, and neither changes when a task spawns a mug. But
# `build_world` builds its memo per world, so a sweep that grounds 718 tasks re-runs the same
# A* thousands of times - and a census over every instance of `countertop` in Beechwood is
# sixty-four A* searches on a 750 000-cell map. Sharing the answers across worlds turns the
# sweep from three quarters of an hour into a few minutes. `--check-reach` proves the two
# agree by asking two independently built worlds of one scene about every instance.
_REACH = {}


def share_reachability(world, scene):
    """Point this world's `reachable_instance` at the scene-wide cache, filling it as it goes."""
    cache = _REACH.setdefault(scene, {})
    inner = world.reachable_instance

    def reachable(instance):
        if instance not in cache:
            cache[instance] = bool(inner(instance))
        return cache[instance]

    world.reachable_instance = reachable
    return world


def check_reach(task, names=None):
    """Two separately built worlds of one scene must answer reachability identically."""
    truth = seed_graph(task)
    first, _ = build_world(task, truth, plan_names(task))
    second, _ = build_world(task, belief_for(task), plan_names(task))
    instances = names or sorted(first.truth.object_names())[:40]
    disagree = [i for i in instances
                if bool(first.reachable_instance(i)) != bool(second.reachable_instance(i))]
    print(f"reachability check on {task['id']}: {len(instances)} instances, "
          f"{len(disagree)} disagreements {disagree[:5]}")
    return not disagree


# ------------------------------------------------------------------ the pre-fix function

def ground_prefix(name, world, graph):
    """`sim_eval.ground` as it was before the reachability preference was added.

    Character-for-character the same but for the last two lines: the believed room first,
    then `instances[0]` - an arbitrary pick that can land in a part of the house the robot
    cannot walk to. Kept here so the fix can be measured against it rather than described.
    """
    if name is None or name in world.truth.objects:
        return name

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

    held = [rel for rel in (graph.get("relations") or [])
            if rel.get("to") == name or sim_eval._same_name(rel.get("to"), name)]
    for rel in held:
        moved = rel.get("from")
        edge = "object_inside" if str(rel.get("relation", "")).upper() == "INSIDE" else "on_top"
        for _, support in world.truth.edges_of(edge, src=moved):
            if support in instances:
                return support

    return next((i for i in instances if world.room_of(i) == believed), instances[0])


# ------------------------------------------------------------------ one name, taken apart

def analyse_name(name, world, graph):
    """Everything `ground` decides about one name, with the branch it took written down.

    Replays `sim_eval.ground` step by step - and then asserts its own answer equals
    `ground`'s, so this cannot quietly drift away from the function it is measuring.
    """
    out = {"name": name, "chosen": None, "branch": None, "category": None,
           "n_instances": 0, "n_reachable": None, "believed": None, "here": 0,
           "room": None, "reachable": None,
           "prefix": None, "prefix_room": None, "prefix_reachable": None}
    if name is None or name in world.truth.objects:
        out["branch"] = "node"          # a spawned object, or already an instance id
        out["chosen"] = name
        return out

    objects = graph.get("objects") or {}
    believed = (objects.get(name) or {}).get("room")
    if believed is None:
        key = match(name, list(objects))
        if key:
            believed = (objects.get(key) or {}).get("room")
    out["believed"] = believed

    def in_believed_room(category):
        return any(world.room_of(i) == believed for i in world.truth.by_category(category))

    pool = {world.truth.objects[o]["category"] for o in world.truth.object_names()}
    category = match(name, pool, prefer=in_believed_room if believed else None)
    instances = world.truth.by_category(category) if category else []
    out["category"] = category
    out["n_instances"] = len(instances)
    if not instances:
        out["branch"] = "unmatched"     # nothing in the world answers to this name
        out["chosen"] = name
        return out

    can_reach = getattr(world, "reachable_instance", None)
    here = [i for i in instances if world.room_of(i) == believed]
    out["here"] = len(here)

    held = [rel for rel in (graph.get("relations") or [])
            if rel.get("to") == name or sim_eval._same_name(rel.get("to"), name)]
    chosen = None
    for rel in held:
        moved = rel.get("from")
        edge = "object_inside" if str(rel.get("relation", "")).upper() == "INSIDE" else "on_top"
        for _, support in world.truth.edges_of(edge, src=moved):
            if support in instances:
                chosen, out["branch"] = support, "relation"
                break
        if chosen is not None:
            break

    if chosen is None:
        if can_reach is None:
            chosen = here[0] if here else instances[0]
            out["branch"] = "unguarded"
        elif next((i for i in here if can_reach(i)), None) is not None:
            chosen = next(i for i in here if can_reach(i))
            out["branch"] = "believed_room"
        elif next((i for i in instances if can_reach(i)), None) is not None:
            chosen = next(i for i in instances if can_reach(i))
            # The fix's own branch: the believed room held nothing the robot can get to.
            out["branch"] = "reachable_fallback"
        else:
            chosen = here[0] if here else instances[0]
            out["branch"] = "nothing_reachable"

    out["chosen"] = chosen
    out["room"] = world.room_of(chosen)
    out["reachable"] = bool(can_reach(chosen)) if can_reach else None
    # What the rule before the fix would have picked, and whether the robot could get there.
    prefix = chosen if out["branch"] == "relation" else (here[0] if here else instances[0])
    out["prefix"] = prefix
    out["prefix_room"] = world.room_of(prefix)
    out["prefix_reachable"] = bool(can_reach(prefix)) if can_reach else None
    out["n_reachable"] = sum(1 for i in instances if can_reach(i)) if can_reach else None

    assert chosen == ground(name, world, graph), (
        f"analyse_name disagrees with sim_eval.ground on {name}: {chosen} vs "
        f"{ground(name, world, graph)}")
    assert prefix == ground_prefix(name, world, graph), (
        f"ground_prefix disagrees with the replay on {name}")
    return out


def plan_names(task):
    return [a for _, a in [(s["action"], s.get("object")) if isinstance(s, dict) else (s[0], s[1])
                           for s in task.get("plan", [])] if a]


def goal_names(task):
    names = []
    for entry in task.get("goal", ()):
        _, src, dst = entry
        if isinstance(src, str):
            names.append(src)
        if isinstance(dst, str):
            names.append(dst)
    return names


# ------------------------------------------------------------------ scene-level facts

_SCENE = {}


def scene_facts(scene):
    """Category multiplicity and the connectivity of the standable floor, per scene.

    The robot starts in the largest connected region of eroded floor (`Sim2D._starting_pose`)
    and `stances_for` will not route outside it, so an instance in another region is not
    merely far, it is unreachable for the whole run. Counting instances per region tells you
    which categories can be mis-grounded into a part of the house that no plan can reach.
    """
    if scene in _SCENE:
        return _SCENE[scene]
    import numpy as np

    from floor_world import DEFAULT_ROBOT_RADIUS

    world = world_for(scene)
    mask, labels = world.traversable(DEFAULT_ROBOT_RADIUS)
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    home = int(np.argmax(sizes))
    regions = [int(s) for s in sizes if s]

    counts = collections.Counter()
    split = {}
    for name in world.truth.object_names():
        category = world.category_of(name)
        counts[category] += 1
        position = world.truth.position_of(name)
        region = world.region_of(position[0], position[1]) if position else 0
        split.setdefault(category, collections.Counter())[region] += 1

    facts = {"scene": scene,
             "rooms": len(world.rooms),
             "regions": sorted(regions, reverse=True),
             "home_region": home,
             "categories": len(counts),
             "multi": {c: n for c, n in sorted(counts.items()) if n > 1},
             "outside_home": {c: sum(n for r, n in rs.items() if r != home)
                              for c, rs in split.items()
                              if any(r != home for r in rs)}}
    _SCENE[scene] = facts
    return facts


# ------------------------------------------------------------------ one task

_BELIEFS = {}


def belief_for(task):
    """`populate` for this task, cached by its extraction so the RSN is asked once."""
    extraction = task["extraction"]
    key = (task["scene"], tuple(extraction["uncertain"]),
           tuple(sorted((extraction.get("stated") or {}).items())),
           tuple(sorted((d["object"], d["relation"], d["target"])
                        for d in (extraction.get("dependent") or []))))
    if key not in _BELIEFS:
        _BELIEFS[key] = populate(task["scene"], extraction["uncertain"],
                                 extraction["dependent"],
                                 stated=extraction.get("stated") or {},
                                 model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    return _BELIEFS[key]


def sweep_task(task, post=False):
    """Ground every name this task uses, both ways, and return one row per name.

    `post=True` additionally runs the plan and re-grounds the goal afterwards, which is what
    `run_plan` used to do - the relation branch reads `world.truth`'s edges and the plan
    moves the object those edges are about, so the answer can change under it.
    """
    belief = belief_for(task)
    truth = seed_graph(task)
    names_plan, names_goal = set(plan_names(task)), set(goal_names(task))
    every = sorted(names_plan | names_goal)

    world_b = share_reachability(build_world(task, belief, list(names_plan))[0], task["scene"])
    world_t = share_reachability(build_world(task, truth, list(names_plan))[0], task["scene"])

    declared = dict(furniture_rooms(task["scene"]))
    explicit = dict(task.get("rooms") or {})
    declared.update(explicit)

    rows = []
    for name in every:
        b = analyse_name(name, world_b, belief)
        t = analyse_name(name, world_t, truth)
        category = b["category"]
        # The task's declaration for this name: its own `rooms` override first, then the
        # scene's majority room, matched the same way the plan's names are matched.
        key = name if name in declared else match(name, list(declared))
        rows.append({
            "task": task["id"], "scene": task["scene"], "name": name,
            "in_plan": name in names_plan, "in_goal": name in names_goal,
            "category": category, "n_instances": b["n_instances"],
            "n_reachable": b["n_reachable"],
            "branch": b["branch"], "believed": b["believed"], "here": b["here"],
            "chosen": b["chosen"], "room": b.get("room"), "reachable": b["reachable"],
            "prefix": b["prefix"], "prefix_room": b.get("prefix_room"),
            "prefix_reachable": b["prefix_reachable"],
            "truth_chosen": t["chosen"], "truth_room": t.get("room"),
            "truth_branch": t["branch"], "truth_reachable": t["reachable"],
            "truth_n_instances": t["n_instances"],
            "declared_room": declared.get(key), "declared_explicit": key in explicit,
        })

    if post:
        for row in rows:
            row["post_chosen"] = None
        after = _ground_after_plan(task, belief, world_b)
        for row in rows:
            if row["name"] in after:
                row["post_chosen"] = after[row["name"]]
    return rows


def _ground_after_plan(task, belief, world):
    """Re-ground every name once the plan has run, on a fresh world.

    `run_plan` grounds the goal before the first action for a reason; this measures what the
    answers would be if it did not. The world is rebuilt because running a plan mutates it.
    """
    from sim2d import Sim2D

    fresh = share_reachability(build_world(task, belief, plan_names(task))[0], task["scene"])
    plan = [(s["action"], s.get("object")) if isinstance(s, dict) else (s[0], s[1])
            for s in task["plan"]]
    bound = [(action, ground(arg, fresh, belief)) for action, arg in plan]
    sim = Sim2D(fresh, focus={a for _, a in bound if a}, verbose=False)
    sim.run(bound, stop_on_failure=True)
    return {name: ground(name, fresh, belief)
            for name in set(plan_names(task)) | set(goal_names(task))}


# ------------------------------------------------------------------ the sweep

def run(tasks, post=False, verbose=False):
    rows, errors = [], []
    started = time.time()
    for index, task in enumerate(tasks, 1):
        try:
            rows.extend(sweep_task(task, post=post))
        except Exception as exc:                       # a task that cannot even be built
            errors.append({"task": task["id"], "error": f"{type(exc).__name__}: {exc}"})
            if verbose:
                print(f"  ERROR {task['id']}: {type(exc).__name__}: {exc}")
        if verbose and index % 25 == 0:
            print(f"  {index}/{len(tasks)} tasks, {len(rows)} names, "
                  f"{time.time() - started:.0f}s")
    return rows, errors


def empty_plan_check(tasks, verbose=False):
    """Is the goal already true before the robot moves - under the BELIEF's grounding?

    `build_tasks.verify` asks this, and asks it of the truth: it refuses a task whose goal
    holds in `seed_graph` before the plan runs, because such a task is satisfied by the empty
    plan and measures nothing. The run does not use that grounding. If the belief binds a
    goal's destination to the very instance the task spawned the object on - which the
    relation branch will do whenever the sentence names the same category for source and
    destination - then the goal holds at step zero and any plan, including no plan at all,
    scores. That is the same defect shape as an impossible task, pointing the other way.

    Driving the empty plan is the whole test: `run_plan` grounds the goal exactly as a real
    run does and then asks the true world whether it is met.
    """
    trivial = []
    for index, task in enumerate(tasks, 1):
        belief = belief_for(task)
        empty = run_plan(task, belief, [], verbose=False)
        truth_empty = run_plan(task, seed_graph(task), [], verbose=False)
        if empty["goal_met"] or truth_empty["goal_met"]:
            trivial.append({"task": task["id"], "belief_goal_met": empty["goal_met"],
                            "truth_goal_met": truth_empty["goal_met"],
                            "missing": empty["missing"]})
            if verbose:
                print(f"    TRIVIAL {task['id']}: belief={empty['goal_met']} "
                      f"truth={truth_empty['goal_met']}")
        if verbose and index % 50 == 0:
            print(f"    empty-plan check {index}/{len(tasks)}")
    return trivial


def verify_pairs(tasks, ids, verbose=False):
    """Run the reference plan against the belief twice: post-fix `ground`, then pre-fix.

    This is the reproduction for every claim about the fix. `sim_eval.ground` is replaced
    for the second run only, and put back afterwards.
    """
    out = []
    by_id = {t["id"]: t for t in tasks}
    for task_id in ids:
        task = by_id.get(task_id)
        if task is None:
            continue
        belief = belief_for(task)
        record = {"task": task_id}
        record["belief_post"] = run_plan(task, belief, task["plan"], verbose=False)
        original = sim_eval.ground
        try:
            sim_eval.ground = ground_prefix
            record["belief_pre"] = run_plan(task, belief, task["plan"], verbose=False)
        finally:
            sim_eval.ground = original
        record["truth"] = run_plan(task, seed_graph(task), task["plan"], verbose=False)
        out.append(record)
        if verbose:
            print(f"  {task_id}: truth={record['truth']['ok']} "
                  f"post={record['belief_post']['ok']} pre={record['belief_pre']['ok']}"
                  f"  {'' if record['belief_post']['ok'] else record['belief_post']['why'][:80]}")
    return out


# ------------------------------------------------------------------ reporting

def summarise(rows, label, out=print):
    at_risk = [r for r in rows if r["n_instances"] > 1]
    # A name the world resolved to at least one instance. The other two kinds - a spawned
    # object, which is already a node, and a name no category answers to - have no instance
    # to choose between, so counting them among the "disagreements" would inflate every
    # number below with rows where there was nothing to decide.
    named = [r for r in rows if r["branch"] not in ("node", "unmatched")]
    ghosts = [r for r in rows if r["branch"] == "unmatched"]
    out(f"\n=== {label} ===")
    out(f"  {len({r['task'] for r in rows})} tasks, {len(rows)} name occurrences, "
        f"{len(named)} of them category names the world has to resolve")
    out(f"  population at risk: {len(at_risk)} occurrences "
        f"({len({(r['task'], r['name']) for r in at_risk})} distinct task/name) have "
        f">1 instance in the scene")
    if ghosts:
        out(f"  {len(ghosts)} occurrences name nothing the world holds "
            f"(they ground to themselves): {sorted({r['name'] for r in ghosts})[:8]}")

    # 2. plan vs goal, same task, same name
    both = [r for r in at_risk if r["in_plan"] and r["in_goal"]]
    out(f"\n  [2] plan and goal occurrences of an at-risk name: {len(both)}; "
        f"bound differently at run start: 0 by construction "
        f"(`ground` is a function of name+world+graph and `run_plan` calls it once each "
        f"before the first action)")
    post = [r for r in rows if r.get("post_chosen") is not None]
    if post:
        moved = [r for r in post if r["post_chosen"] != r["chosen"]]
        out(f"      re-grounded AFTER the plan runs: {len(moved)}/{len(post)} names move "
            f"({len({r['task'] for r in moved})} tasks) - the size of the "
            f"ground-the-goal-first fix")
        for r in moved[:6]:
            out(f"        {r['task']:24s} {r['name']:16s} {r['chosen']} -> {r['post_chosen']}")

    # 3. truth vs belief
    differ = [r for r in at_risk if r["chosen"] != r["truth_chosen"]]
    out(f"\n  [3] truth-grounded vs belief-grounded: {len(differ)}/{len(at_risk)} at-risk "
        f"occurrences bind to a different instance ({len({r['task'] for r in differ})} tasks)")
    goal_differ = [r for r in differ if r["in_goal"]]
    out(f"      of those, {len(goal_differ)} are GOAL names - the benchmark checks a "
        f"different object from the one the robot is sent to")
    bad = [r for r in at_risk if r["reachable"] is False]
    out(f"      belief-grounded to an UNREACHABLE instance: {len(bad)} "
        f"({len({r['task'] for r in bad})} tasks){' - ' + str(sorted({r['task'] for r in bad})) if bad else ''}")
    worse = [r for r in bad if r["truth_reachable"]]
    out(f"      ...of which the truth grounding was reachable: {len(worse)} "
        f"(these are tasks verification passes and evaluation cannot)")
    for r in worse[:10]:
        out(f"        {r['task']:24s} {r['name']:16s} belief {r['chosen']} in "
            f"{r['room']} / truth {r['truth_chosen']} in {r['truth_room']}")

    # 4. declared rooms
    declared = [r for r in at_risk if r["declared_room"]]
    off = [r for r in declared if r["room"] and r["room"] != r["declared_room"]]
    explicit = [r for r in declared if r["declared_explicit"]]
    explicit_off = [r for r in off if r["declared_explicit"]]
    out(f"\n  [4] declared rooms: {len(off)}/{len(declared)} at-risk occurrences land in a "
        f"room other than the declaration "
        f"({len(explicit_off)}/{len(explicit)} where the task states it in `rooms`)")
    by_task = collections.Counter(r["task"] for r in explicit_off)
    for task_id, n in by_task.most_common(6):
        example = next(r for r in explicit_off if r["task"] == task_id)
        out(f"        {task_id:24s} {example['name']:16s} declared {example['declared_room']}"
            f" -> chose {example['room']} (believed {example['believed']})")

    # 5. the relation branch
    relation = [r for r in named if r["branch"] == "relation"]
    unreachable = [r for r in relation if r["reachable"] is False]
    out(f"\n  [5] relation branch fired {len(relation)} times "
        f"({len([r for r in relation if r['n_instances'] > 1])} of them on at-risk names); "
        f"picked an unreachable instance {len(unreachable)} times"
        + (f": {sorted({(r['task'], r['name']) for r in unreachable})}" if unreachable else ""))

    # 6. the fix
    disagree = [r for r in named if r["prefix"] != r["chosen"]]
    fixed = [r for r in disagree if r["reachable"] and r["prefix_reachable"] is False]
    other = [r for r in disagree if not (r["reachable"] and r["prefix_reachable"] is False)]
    out(f"\n  [6] pre-fix vs post-fix rule: {len(disagree)} occurrences disagree "
        f"({len({r['task'] for r in disagree})} tasks)")
    out(f"      post-fix reachable AND pre-fix unreachable: {len(fixed)} "
        f"({len({r['task'] for r in fixed})} tasks) - the fix doing exactly what it claims")
    if other:
        out(f"      disagreements NOT of that shape: {len(other)} "
            f"{sorted({(r['task'], r['name']) for r in other})[:6]}")
    for r in fixed[:12]:
        out(f"        {r['task']:24s} {r['name']:16s} pre {r['prefix']} in {r['prefix_room']}"
            f" (unreachable) -> post {r['chosen']} in {r['room']}")
    nothing = [r for r in named if r["branch"] == "nothing_reachable"]
    if nothing:
        out(f"      no instance of the category is reachable at all: {len(nothing)} "
            f"{sorted({(r['task'], r['name']) for r in nothing})[:6]}")
    return {"tasks": len({r["task"] for r in rows}), "occurrences": len(rows),
            "resolved": len(named), "at_risk": len(at_risk),
            "truth_belief_differ": len(differ), "goal_differ": len(goal_differ),
            "unreachable": len(bad), "unreachable_truth_ok": len(worse),
            "declared_off": len(off), "declared_total": len(declared),
            "explicit_off": len(explicit_off), "explicit_total": len(explicit),
            "relation": len(relation), "relation_unreachable": len(unreachable),
            "prefix_disagree": len(disagree), "prefix_fixed": len(fixed),
            "nothing_reachable": len(nothing)}


def scene_table(scenes, out=print):
    out("\n=== scenes: instances per category, and how the floor is split ===")
    out(f"  {'scene':20s} {'rooms':>5s} {'regions':>26s} {'multi-instance categories':>26s}")
    for scene in scenes:
        facts = scene_facts(scene)
        regions = ",".join(str(n) for n in facts["regions"][:5])
        out(f"  {scene:20s} {facts['rooms']:5d} {regions:>26s} "
            f"{len(facts['multi']):5d} of {facts['categories']}")
    out("\n  the fattest categories (scene: category x count), and how many sit outside the")
    out("  region the robot starts in - an instance out there is unreachable for the run")
    for scene in scenes:
        facts = scene_facts(scene)
        worst = sorted(facts["multi"].items(), key=lambda kv: -kv[1])[:6]
        outside = facts["outside_home"]
        pieces = ", ".join(f"{c} x{n}" + (f" ({outside[c]} outside)" if outside.get(c) else "")
                           for c, n in worst)
        out(f"    {scene:20s} {pieces}")


def analyse(path, out=print):
    """Re-read a finished sweep and answer the questions that need the whole table.

    Two things are only visible across tasks. **Which categories carry the risk** - a name
    with six instances is not a defect, but it is where every defect will come from, so the
    breakdown is by category and scene rather than by task. And **whether truth and belief
    disagreed about the CATEGORY** rather than the instance: `cabinet` is as much a
    `top_cabinet` as a `bottom_cabinet` and `match` breaks that tie with the believed room,
    which the two graphs answer differently. The rows store instances, so the category is
    recovered from the scene's own ground truth here.
    """
    data = json.load(open(path))
    rows = data["rows"]
    named = [r for r in rows if r["branch"] not in ("node", "unmatched")]
    at_risk = [r for r in named if r["n_instances"] > 1]

    category_of = {}
    for scene in sorted({r["scene"] for r in rows}):
        world = world_for(scene)
        for name in world.truth.object_names():
            category_of[(scene, name)] = world.category_of(name)

    out(f"\n=== {path}: {len(rows)} name occurrences over "
        f"{len({r['task'] for r in rows})} tasks ===")

    out("\n  the population at risk, by category "
        "(occurrences of a name with >1 instance, and how many instances)")
    counts = collections.Counter()
    sizes = {}
    for r in at_risk:
        counts[r["category"]] += 1
        sizes.setdefault(r["category"], set()).add((r["scene"], r["n_instances"]))
    out(f"  {'category':24s} {'occurrences':>11s}  instances per scene")
    for category, n in counts.most_common(20):
        spread = ", ".join(f"{s.split('_int')[0]}:{k}"
                           for s, k in sorted(sizes[category], key=lambda p: -p[1])[:6])
        out(f"  {category:24s} {n:11d}  {spread}")

    divergent = [r for r in at_risk
                 if category_of.get((r["scene"], r["truth_chosen"])) != r["category"]]
    out(f"\n  truth and belief resolved the NAME to a different CATEGORY: "
        f"{len(divergent)} occurrences ({len({r['task'] for r in divergent})} tasks)")
    for r in divergent[:10]:
        out(f"    {r['task']:24s} {r['name']:16s} belief {r['category']} -> "
            f"truth {category_of.get((r['scene'], r['truth_chosen']))}")

    out(f"\n  branch taken, over the {len(named)} names the world resolved:")
    for branch, n in collections.Counter(r["branch"] for r in named).most_common():
        risky = sum(1 for r in named if r["branch"] == branch and r["n_instances"] > 1)
        out(f"    {branch:20s} {n:6d}  ({risky} of them on a name with >1 instance)")

    out(f"\n  reachability of the instance `ground` chose, over at-risk names:")
    out(f"    reachable       {sum(1 for r in at_risk if r['reachable'])}")
    out(f"    unreachable     {sum(1 for r in at_risk if r['reachable'] is False)}")
    census = [r for r in at_risk if r["n_reachable"] is not None]
    partial = [r for r in census if r["n_reachable"] < r["n_instances"]]
    out(f"    names whose category has at least one UNREACHABLE instance: "
        f"{len(partial)}/{len(census)} - the picks the fallback has to avoid")
    none_reachable = [r for r in census if r["n_reachable"] == 0]
    out(f"    names with NO reachable instance at all: {len(none_reachable)} "
        f"{sorted({(r['task'], r['name']) for r in none_reachable})[:6]}")

    # Did the sentence itself name the room, and did grounding go somewhere else? A task
    # that says "the bedroom cabinet" has that room in `extraction.stated`, and the belief
    # carries it as a fact rather than a guess. If the instance chosen is in a room of a
    # different TYPE, the run is being scored against furniture the instruction ruled out -
    # and it is scored consistently, so it passes. That is not an impossible task; it is a
    # task whose answer key stopped matching its sentence.
    from scene_graph import ROOM_SYNONYMS

    stated_room = {}
    for kind, path in BENCHMARKS.items():
        for task in json.load(open(path)):
            for name, room in (task["extraction"].get("stated") or {}).items():
                stated_room[(task["id"], name)] = room
    rooms_of = {}
    for scene in sorted({r["scene"] for r in rows}):
        rooms_of[scene] = {rid: info["room_type"]
                           for rid, info in world_for(scene).room_graph["rooms"].items()}

    def same_type(scene, room_id, said):
        if room_id is None or said is None:
            return True
        actual = rooms_of[scene].get(room_id)
        return actual == said or ROOM_SYNONYMS.get(said, said) == actual

    said_rows = [r for r in named if (r["task"], r["name"]) in stated_room]
    elsewhere = [r for r in said_rows
                 if not same_type(r["scene"], r["room"], stated_room[(r["task"], r["name"])])]
    out(f"\n  names whose ROOM the instruction states: {len(said_rows)}; "
        f"grounded into a room of another type anyway: {len(elsewhere)}")
    for r in elsewhere[:10]:
        out(f"    {r['task']:24s} {r['name']:16s} said {stated_room[(r['task'], r['name'])]}"
            f" -> chose {r['room']} ({rooms_of[r['scene']].get(r['room'])}), "
            f"branch {r['branch']}")

    verifications = data.get("verifications") or []
    if verifications:
        broken = [v for v in verifications if v["truth"]["ok"] and not v["belief_post"]["ok"]]
        rescued = [v for v in verifications
                   if v["belief_post"]["ok"] and not v["belief_pre"]["ok"]]
        regressed = [v for v in verifications
                     if v["belief_pre"]["ok"] and not v["belief_post"]["ok"]]
        both = [v for v in verifications
                if not v["belief_post"]["ok"] and not v["belief_pre"]["ok"]]
        out(f"\n  reference plans re-run: {len(verifications)}")
        out(f"    ok on truth, fails on the belief (post-fix): {len(broken)}")
        for v in broken:
            out(f"      {v['task']:24s} {v['belief_post']['why'][:90]}")
        out(f"    rescued by the fix (pre fails, post passes): {len(rescued)} "
            f"{[v['task'] for v in rescued]}")
        out(f"    regressed by the fix: {len(regressed)} {[v['task'] for v in regressed]}")
        out(f"    failing both ways: {len(both)} {[v['task'] for v in both][:10]}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--which", default="all", choices=["single", "multi", "sub", "all"])
    parser.add_argument("--scene", help="only tasks in this scene")
    parser.add_argument("--limit", type=int, help="first N tasks of each benchmark")
    parser.add_argument("--post", action="store_true",
                        help="also re-ground after the plan runs (slow: runs every plan)")
    parser.add_argument("--verify", action="store_true",
                        help="re-run the reference plan pre-fix and post-fix on flagged tasks")
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-cache", action="store_true",
                        help="do not memoise the encoder/checkpoint (slow, identical answers)")
    parser.add_argument("--check-cache", action="store_true",
                        help="prove the memoisation changes no belief, then carry on")
    parser.add_argument("--empty", action="store_true",
                        help="drive the EMPTY plan: is the goal already met under the belief?")
    parser.add_argument("--check-reach", action="store_true",
                        help="prove reachability is a scene fact, not a per-world one")
    parser.add_argument("--analyse", metavar="JSON",
                        help="re-read a finished sweep and print the cross-task breakdowns")
    args = parser.parse_args()

    if args.analyse:
        analyse(args.analyse)
        return 0

    first = json.load(open(BENCHMARKS["single"]))[0]
    if args.check_cache:
        if not check_cache(first):
            raise SystemExit("the memoisation changed a belief - rerun with --no-cache")
    elif not args.no_cache:
        accelerate()
    if args.check_reach and not check_reach(first):
        raise SystemExit("reachability is not a scene fact here - do not share the cache")

    which = list(BENCHMARKS) if args.which == "all" else [args.which]
    report, all_rows, all_verifications = {}, [], []
    scenes = set()
    for kind in which:
        tasks = json.load(open(BENCHMARKS[kind]))
        if args.scene:
            tasks = [t for t in tasks if t["scene"] == args.scene]
        if args.limit:
            tasks = tasks[:args.limit]
        scenes |= {t["scene"] for t in tasks}
        print(f"\n### {kind}: {len(tasks)} tasks from {BENCHMARKS[kind]}")
        rows, errors = run(tasks, post=args.post, verbose=not args.quiet)
        for row in rows:
            row["benchmark"] = kind
        all_rows.extend(rows)
        report[kind] = summarise(rows, f"{kind}  ({len(tasks)} tasks)")
        report[kind]["errors"] = errors
        if errors:
            print(f"  {len(errors)} tasks errored: {[e['task'] for e in errors][:6]}")

        if args.empty:
            trivial = empty_plan_check(tasks, verbose=not args.quiet)
            print(f"  [7] goal already met before the robot moves: {len(trivial)} tasks "
                  f"{[t['task'] for t in trivial][:10]}")
            report[kind]["trivial"] = trivial

        if args.verify:
            flagged = sorted({r["task"] for r in rows
                              if r["branch"] not in ("node", "unmatched")
                              and (r["reachable"] is False
                                   or r["prefix"] != r["chosen"]
                                   or (r["n_instances"] > 1
                                       and r["chosen"] != r["truth_chosen"]))})
            print(f"\n  verifying {len(flagged)} flagged tasks (reference plan, "
                  f"truth / post-fix belief / pre-fix belief)")
            checks = verify_pairs(tasks, flagged, verbose=not args.quiet)
            for check in checks:
                check["benchmark"] = kind
            all_verifications.extend(checks)
            broken = [c for c in checks if c["truth"]["ok"] and not c["belief_post"]["ok"]]
            rescued = [c for c in checks
                       if c["belief_post"]["ok"] and not c["belief_pre"]["ok"]]
            regressed = [c for c in checks
                         if c["belief_pre"]["ok"] and not c["belief_post"]["ok"]]
            print(f"  reference plan ok on truth but NOT on the belief: {len(broken)} "
                  f"{[c['task'] for c in broken]}")
            print(f"  fixed by the reachability preference (pre fails, post passes): "
                  f"{len(rescued)} {[c['task'] for c in rescued]}")
            print(f"  broken by it (pre passes, post fails): {len(regressed)} "
                  f"{[c['task'] for c in regressed]}")
            report[kind]["verify"] = {"checked": len(checks), "broken": len(broken),
                                      "rescued": [c["task"] for c in rescued],
                                      "regressed": [c["task"] for c in regressed],
                                      "broken_ids": [c["task"] for c in broken]}

    scene_table(sorted(scenes))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"report": report, "rows": all_rows,
                   "verifications": all_verifications,
                   "scenes": {s: scene_facts(s) for s in sorted(scenes)}}, handle, indent=1)
    print(f"\nwrote {args.out}  ({len(all_rows)} name rows)")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("errors",)}
                      for k, v in report.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
