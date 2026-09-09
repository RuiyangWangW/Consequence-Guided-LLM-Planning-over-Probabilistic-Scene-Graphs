#!/usr/bin/env python3
"""Scene geometry as a source of impossible tasks.

`build_tasks.verify` drives every reference plan against `seed_graph(task)` - the truth -
and refuses to write a task whose plan does not finish. That check is real, and it still
cannot see one whole class of defect, because no evaluation ever scores a plan against the
truth: `sim_eval.ground` binds a category name to an *instance* using the BELIEF, and the
belief is the RSN's guess. When the guess names a room the category is not in, grounding
falls through to whatever instance is left - and in a scene that comes in two disconnected
pieces, "whatever is left" can sit on the far side of the gap. The reference plan then
fails, so no plan exists, and verification never notices.

The fix in `sim_eval` makes `ground` prefer an instance the robot can reach, sharing the
predicate `build_world` already used for spawn supports. This script asks what the geometry
still makes impossible with that fix in place.

**What "connected" means here, and why it is asked twice.** The robot drives
`floor_world.astar` on the eroded floor: 8-connected, and forbidden to cut a diagonal
corner. Two other notions are used around it, and neither is the same relation:

    ndimage.label(structure=ones(3,3))   joins cells that touch only at a corner, which
                                         A* cannot cross. `Sim2D.stances_for` filters on
                                         this, so it is optimistic, and `route_to` is what
                                         actually settles it.
    cost_matrix.build(scene)["distance"] A* between ROOM CENTROIDS. Honest about corners,
                                         but it stands or falls on one cell per room, and
                                         in `Pomaria_0_int` that cell lands in a 2.3 m2
                                         pocket behind the furniture, so the living room is
                                         reported unreachable from all eight other rooms.

So this script computes the ground truth itself - a breadth-first flood fill from the
robot's start under exactly the rule `astar` moves by - and reports `cost_matrix`'s answer
beside it. Where the two disagree, one of them is a defect, and the flood fill is not the
one with a single sample point per room.

The sweeps, in order:

  1. **Components.** Per scene: the flood fill's components, which one the robot starts in,
     what furniture is in the others, and whether `cost_matrix` agrees.
  2. **Tasks that need decoration.** How many single tasks, multi-task instructions and
     errands name furniture whose every instance is outside the robot's component.
  3. **Every instance unreachable.** The same question asked with the predicate `ground` and
     `build_world` actually use - a stance in the robot's floor region, routable, within
     `REACH` of the object's near edge - because a room the robot can enter still holds
     furniture wedged beyond the arm, and because that predicate reaches through walls
     (`FloorWorld.distance_to` is a straight line) and so calls some stranded furniture
     reachable. A name is *usable* only if some instance passes both.
  4. **The category fork.** `ground` picks a category before it picks an instance, and only
     the instance choice was made reachability-aware. `cabinet` fits `top_cabinet` and
     `bottom_cabinet` equally; if the believed room holds an unusable one and the other is
     usable, the fix never fires.
  5. **The `rooms` claims.** A task may pin a category to a room, and `seed_graph` writes
     that into the truth a plan is scored against. `build_tasks.check_objects` verifies it
     for the hundred single tasks and nothing verifies it for the 500 instructions or the
     118 errands - and `ground` now prefers the declared room over the believed one, so a
     claim the scene contradicts is a preference that matches nothing.
  6. **The spawn path.** `build_world` puts the task's object on a support chosen the same
     way and falling back the same way. An object spawned on an unreachable support can
     never be picked up.

`--ground` then asks the real `sim_eval.ground`, under the real belief, what every plan and
goal name binds to - restricted to the scenes that hold an unusable instance, because in the
others no binding can be unusable and the restriction is a proof rather than a sample.
`--sim` drives the reference plan of every flagged task twice - against the truth, the way
verification does, and against the belief `scene_graph.populate` produces, the way every
evaluation does - so "impossible" comes with the run that shows it.

    python exp_sweep_scenes.py                          # geometry, all ten scenes
    python exp_sweep_scenes.py --ground --sim           # plus grounding and the reproductions
    python exp_sweep_scenes.py --scene Pomaria_0_int --sim
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import time
from collections import defaultdict, deque

import numpy as np

import cost_matrix
from build_tasks import SCENES, furniture_rooms
from floor_world import DEFAULT_ROBOT_RADIUS, FloorWorld
from object_names import candidates as name_candidates
from sim2d import STANCE_CANDIDATES, REACH, Sim2D

SINGLE = "data/tasks.json"
MULTI = "data/multitask.json"
SUBTASKS = "data/subtasks.json"

# The eight moves `floor_world.astar` makes, and the corner rule it makes them under.
_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


def flood(mask, start):
    """Every cell A* could reach from `start`, under exactly the rule `astar` moves by.

    A* is 8-connected and refuses to slip diagonally between two blocked cells, because the
    base would clip both. `ndimage.label` has no such scruple, so the components it finds
    are a superset of the ones the robot can actually drive between - `Beechwood_0_int`'s
    utility room is joined to the rest of the house by a single diagonal pinch and shares a
    label with it while being unreachable. This is the relation that decides whether a task
    is possible, so it is computed rather than borrowed.
    """
    height, width = mask.shape
    seen = np.zeros_like(mask)
    if start is None or not mask[start]:
        return seen
    seen[start] = True
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        for dr, dc in _STEPS:
            nr, nc = row + dr, col + dc
            if not (0 <= nr < height and 0 <= nc < width) or not mask[nr, nc] or seen[nr, nc]:
                continue
            if dr and dc and not (mask[row + dr, col] and mask[row, col + dc]):
                continue
            seen[nr, nc] = True
            queue.append((nr, nc))
    return seen


def distance_components(table):
    """`cost_matrix`'s answer: union-find over the room pairs it gives a finite distance."""
    rooms = list(table["rooms"])
    parent = {r: r for r in rooms}

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for key, value in table["distance"].items():
        if value == float("inf"):
            continue
        a, b = key.split("|")
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups = defaultdict(list)
    for r in rooms:
        groups[find(r)].append(r)
    return sorted((sorted(v) for v in groups.values()), key=lambda g: (-len(g), g[0]))


# --------------------------------------------------------------------------- geometry


def scene_geometry(scene, verbose=True):
    """Everything about one scene that decides whether a task in it can be done.

    Loaded with every category, because "which instances does this category have" is the
    question, and the `Sim2D` probe is built exactly as `sim_eval.build_world` builds its
    own - default start, which is the middle of the largest labelled region of standable
    floor. That default is the start every `run_plan` in the repo uses.
    """
    began = time.time()
    world = FloorWorld.load(scene, categories=None)
    probe = Sim2D(world, verbose=False)
    mask, labels = world.traversable(DEFAULT_ROBOT_RADIUS)
    resolution2 = world.resolution ** 2

    start_cell = world.nearest_free_cell(probe.x, probe.y, DEFAULT_ROBOT_RADIUS)
    drivable = flood(mask, start_cell)

    # The true components, by repeated flood fill: the robot's first, then whatever floor is
    # left, until every standable cell belongs to one. Rooms are assigned to the component
    # holding most of their standable floor, which is the question `go_to_room` asks - not
    # the one `cost_matrix` asks of a single centroid cell.
    covered = drivable.copy()
    fills = [drivable]
    while True:
        rest = np.argwhere(mask & ~covered)
        if not len(rest):
            break
        piece = flood(mask, tuple(rest[0]))
        fills.append(piece)
        covered |= piece

    room_cells, room_fill = {}, {}
    for room in sorted(world.rooms):
        here = world.room_mask(room) & mask
        counts = [int((here & f).sum()) for f in fills]
        room_cells[room] = {"standable": int(here.sum()),
                            "drivable_by_robot": counts[0],
                            "per_component": counts}
        room_fill[room] = int(np.argmax(counts)) if here.sum() else None

    # Keyed by the fill's own index, not renumbered: `component_of` below indexes the same
    # list, and squeezing out the empty fills silently shifted every id by one - the report
    # then put Beechwood's utility room in a component it said held no furniture.
    # A fill with no room to its name is furniture-locked floor, not a part of the house.
    components = {index: sorted(r for r, i in room_fill.items() if i == index)
                  for index in range(len(fills))}
    components = {i: c for i, c in components.items() if c}
    robot_component = components.get(0, [])

    # `go_to_room` is the operative test for "can the robot get into this room": the nearest
    # standable cells of the room that share its LABEL, the first eight of them, and a route.
    # Replicated rather than approximated, because the eight-candidate truncation is real.
    region = world.region_of(probe.x, probe.y, DEFAULT_ROBOT_RADIUS)
    enterable = {}
    for room in sorted(world.rooms):
        target = mask & world.room_mask(room) & (labels == region)
        rows, cols = np.nonzero(target)
        if not len(rows):
            enterable[room] = False
            continue
        row, col = world.to_cell(probe.x, probe.y)
        order = np.argsort((rows - row) ** 2 + (cols - col) ** 2)
        enterable[room] = any(probe.route_to((int(rows[i]), int(cols[i]))) is not None
                              for i in order[:STANCE_CANDIDATES])

    # The predicate `sim_eval.build_world` installs as `world.reachable_instance` and the
    # fixed `ground` consults, verbatim and memoised.
    memo = {}

    def reachable(instance):
        if instance not in memo:
            stances = probe.stances_for(instance)
            memo[instance] = any(probe.route_to(s) is not None
                                 and world.distance_to(instance, *world.to_world(*s)) <= REACH
                                 for s in stances)
        return memo[instance]

    instances = sorted(world.truth.object_names())
    for name in instances:
        reachable(name)

    by_category = defaultdict(list)
    for name in instances:
        by_category[world.category_of(name)].append(name)

    component_of = {}
    for name in instances:
        room = world.room_of(name)
        if room in room_fill and room_fill[room] is not None:
            component_of[name] = room_fill[room]
            continue
        # Set into a wall or a counter run, so it carries no room. Ask the floor instead.
        position = world.truth.position_of(name)
        cell = (None if position is None
                else world.nearest_free_cell(position[0], position[1], DEFAULT_ROBOT_RADIUS))
        component_of[name] = next((i for i, f in enumerate(fills) if cell and f[cell]), None)

    # `reachable` is optimistic in one direction that matters: `FloorWorld.distance_to` is a
    # straight line to the object's near edge and knows nothing about walls, while
    # `stances_for` only insists the stance share the robot's LABEL. A basin mounted on the
    # party wall of a house-half the robot can never drive into is therefore "reachable" -
    # fifteen of `Wainscott_0_int`'s instances are exactly that. `usable` is the conjunction:
    # the code's own predicate AND an instance whose room the robot can actually get into.
    usable = {}
    for name in instances:
        room = world.room_of(name)
        here = enterable.get(room, component_of[name] == 0)
        usable[name] = bool(memo[name]) and component_of[name] == 0 and bool(here)

    table = cost_matrix.build(scene)
    cm_components = distance_components(table)
    cm_room = {r: next(i for i, c in enumerate(cm_components) if r in c)
               for r in table["rooms"]}
    truth_room = {r: room_fill.get(r) for r in table["rooms"]}
    # Two partitions of the same rooms. They disagree iff some pair is joined by one and
    # split by the other, which is the comparison that survives the components being
    # numbered differently.
    disagree = []
    rooms = sorted(table["rooms"])
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if (cm_room[a] == cm_room[b]) != (truth_room[a] == truth_room[b]):
                disagree.append((a, b, cm_room[a] == cm_room[b]))

    geo = {
        "scene": scene, "world": world, "probe": probe,
        "rooms": sorted(world.rooms),
        "resolution2": resolution2,
        "robot_xy": (round(probe.x, 2), round(probe.y, 2)),
        "robot_room": probe.room,
        "robot_region_label": region,
        "drivable_m2": round(float(drivable.sum()) * resolution2, 1),
        "standable_m2": round(float(mask.sum()) * resolution2, 1),
        "fill_sizes": [round(float(f.sum()) * resolution2, 1) for f in fills],
        "components": components,
        "robot_component": robot_component,
        "room_component": room_fill,
        "room_cells": room_cells,
        "enterable": enterable,
        "reachable": dict(memo),
        "usable": usable,
        "component_of": component_of,
        "cross_wall": sorted(n for n in instances if memo[n] and not usable[n]),
        "instances": instances,
        "by_category": {k: list(v) for k, v in by_category.items()},
        "cost_matrix_components": cm_components,
        "cost_matrix_disagreements": disagree,
        "seconds": round(time.time() - began, 1),
    }
    if verbose:
        print(f"  {scene}: {len(instances)} instances, {len(components)} components, "
              f"{len(disagree)} room pairs where cost_matrix disagrees, "
              f"{geo['seconds']}s", flush=True)
    return geo


# ------------------------------------------------------------------------- task names


def spawned_names(task):
    return {s["name"] for s in task.get("spawn", [])}


def needed_furniture(task):
    """The category names a task's plan and goal act on that the scene has to supply.

    The task's own injected objects are excluded: a potato is spawned, not found, and its
    reachability is a question about the support it is spawned on - which `sweep_spawns`
    asks separately.
    """
    spawned = spawned_names(task)
    names = {arg for _, arg in task.get("plan", []) if arg}
    names |= {s["target"] for s in task.get("spawn", [])}
    for entry in task.get("goal", ()):
        names.add(entry[1])
        if isinstance(entry[2], str):
            names.add(entry[2])
    for sub in task.get("subgoals", ()):
        names |= {arg for _, arg in sub.get("plan", []) if arg}
        for entry in sub.get("goal", ()):
            names.add(entry[1])
            if isinstance(entry[2], str):
                names.add(entry[2])
    return sorted(n for n in names - spawned if n)


def resolve(geo, name):
    """Every category the name could mean, and every instance those categories have.

    `sim_eval.ground` narrows this to one category with `object_names.match`, breaking ties
    on the believed room. The whole set is kept here, because the question is whether ANY
    reading of the name gives the robot something it can act on. If none does, no belief can
    save the task and the defect belongs to the benchmark.
    """
    pool = list(geo["by_category"])
    cats = name_candidates(name, pool)
    return cats, [i for c in cats for i in geo["by_category"][c]]


# ------------------------------------------------------------------------- the sweeps


def _detail(geo, name, cats, instances):
    return {"name": name, "categories": cats, "instances": instances,
            "rooms": sorted({str(geo["world"].room_of(i)) for i in instances})}


def sweep_tasks(tasks, geos, label):
    """Which tasks name furniture no plan could act on, and in which of four ways.

        no_instance  the scene holds no category the name could mean
        stranded     every instance is in a room outside the robot's component
        unreachable  no instance passes the predicate `ground` and `build_world` use
        unusable     no instance passes both that predicate and the component test
    """
    rows = []
    for task in tasks:
        geo = geos[task["scene"]]
        stranded, unreachable, unusable, missing, fork = [], [], [], [], []
        for name in needed_furniture(task):
            cats, instances = resolve(geo, name)
            if not instances:
                missing.append(name)
                continue
            if all(geo["component_of"][i] != 0 for i in instances):
                stranded.append(_detail(geo, name, cats, instances))
            if not any(geo["reachable"][i] for i in instances):
                unreachable.append(_detail(geo, name, cats, instances))
            if not any(geo["usable"][i] for i in instances):
                unusable.append(_detail(geo, name, cats, instances))
            elif len(cats) > 1:
                # The fork `ground` can still take wrongly. Two categories fit the name, one
                # of them has nothing usable, and the believed room - not reachability -
                # decides which is used, because by the time the instance preference runs the
                # category is already chosen.
                dead = [c for c in cats
                        if not any(geo["usable"][i] for i in geo["by_category"][c])]
                if dead and len(dead) < len(cats):
                    fork.append({"name": name, "categories": cats,
                                 "unusable_categories": dead})
        if stranded or unreachable or unusable or missing or fork:
            rows.append({"id": task["id"], "scene": task["scene"], "kind": label,
                         "stranded": stranded, "unreachable": unreachable,
                         "unusable": unusable, "no_instance": missing,
                         "category_fork": fork})
    return rows


def sweep_rooms(tasks, geos, label):
    """Does each task's own `rooms` claim hold in the scene, and is what it names usable?

    A task may pin a category to a room - "the dining room coffee table" - and `seed_graph`
    writes that straight into the truth the plan is scored against. `build_tasks.check_objects`
    verifies it for the hundred single tasks and nothing verifies it for the multi-task
    instructions or the errands. It matters more now than it did: `ground` prefers the
    DECLARED room before the believed one, so a claim the scene contradicts is a preference
    that silently matches nothing and drops through to a weaker rung.
    """
    rows = []
    for task in tasks:
        geo = geos[task["scene"]]
        world = geo["world"]
        bad = []
        for category, room in (task.get("rooms") or {}).items():
            instances = geo["by_category"].get(category) or []
            if room not in world.rooms:
                bad.append({"category": category, "room": room,
                            "why": "this scene has no such room"})
            elif not instances:
                bad.append({"category": category, "room": room,
                            "why": "this scene has no instance of the category"})
            else:
                here = [i for i in instances if world.room_of(i) == room]
                if not here:
                    bad.append({"category": category, "room": room,
                                "why": "no instance of it is in that room",
                                "actually_in": sorted({str(world.room_of(i))
                                                       for i in instances})})
                elif not any(geo["usable"][i] for i in here):
                    bad.append({"category": category, "room": room,
                                "why": "every instance in that room is unusable",
                                "instances": here})
        if bad:
            rows.append({"id": task["id"], "scene": task["scene"], "kind": label,
                         "claims": bad})
    return rows


def sweep_spawns(tasks, geos, label):
    """Where `build_world` would put each task's objects, and whether the robot can get there.

    A verbatim replay of the support choice in `sim_eval.build_world`: the instances of the
    spawn's target category, the room `seed_graph` puts that category in, reachable-in-that
    room first, then reachable anywhere, then the room, then `instances[0]`. The last two
    rungs are the ones that can strand an object where nothing can pick it up.
    """
    rows = []
    for task in tasks:
        geo = geos[task["scene"]]
        rooms = dict(furniture_rooms(task["scene"]))
        rooms.update(task.get("rooms") or {})
        for spawn in task.get("spawn", []):
            target = spawn["target"]
            instances = geo["by_category"].get(target) or []
            if not instances:
                rows.append({"id": task["id"], "scene": task["scene"], "kind": label,
                             "object": spawn["name"], "target": target, "support": None,
                             "why": "the scene has no instance of this category"})
                continue
            want = rooms.get(target)
            here = [i for i in instances if geo["world"].room_of(i) == want]
            reach = geo["reachable"].get
            support = (next((i for i in here if reach(i)), None)
                       or next((i for i in instances if reach(i)), None)
                       or (here[0] if here else instances[0]))
            if geo["usable"][support]:
                continue
            rows.append({"id": task["id"], "scene": task["scene"], "kind": label,
                         "object": spawn["name"], "target": target, "support": support,
                         "support_room": geo["world"].room_of(support),
                         "wanted_room": want,
                         "reachable_predicate": bool(reach(support)),
                         "component": geo["component_of"][support],
                         "instances": instances,
                         "why": ("no instance of the support category is reachable"
                                 if not any(reach(i) for i in instances)
                                 else "the chosen support is reachable only through a wall")})
    return rows


# ------------------------------------------------------------- grounding under a belief


def sweep_grounding(tasks, geos, beliefs, label):
    """What `sim_eval.ground` binds each plan and goal name to under the real belief, and
    whether the robot can act on it.

    This is the residual the reachability fix leaves. The fix chooses among the instances of
    ONE category; the category itself is still chosen by the belief, and the fallback rung at
    the bottom of `ground` still returns something when nothing usable exists.

    The real `ground` is called, through a real `build_world`, rather than re-implemented -
    it is the function under test, and a copy of it would only ever prove things about the
    copy. That costs a `FloorWorld` and a `Sim2D` per task, which is why the caller restricts
    this to the scenes that HAVE an unusable instance: in a scene where every instance is
    usable, no binding can be unusable, so there is provably nothing to find.
    """
    from sim_eval import build_world, ground

    rows = []
    for task in tasks:
        belief = beliefs.get(task["id"])
        if belief is None:
            continue
        geo = geos[task["scene"]]
        names = {a for _, a in task.get("plan", []) if a}
        for entry in task.get("goal", ()):
            names.add(entry[1])
            if isinstance(entry[2], str):
                names.add(entry[2])
        try:
            world, _ = build_world(task, belief, [a for _, a in task.get("plan", [])])
        except Exception as exc:
            rows.append({"id": task["id"], "scene": task["scene"], "kind": label,
                         "error": f"{type(exc).__name__}: {exc}", "bindings": []})
            continue
        bad = []
        for name in sorted(n for n in names if n):
            bound = ground(name, world, belief)
            if geo["usable"].get(bound, True):
                continue
            category = geo["world"].category_of(bound)
            bad.append({"name": name, "bound": bound,
                        "room": str(geo["world"].room_of(bound)),
                        "reachable_predicate": bool(geo["reachable"][bound]),
                        "component": geo["component_of"][bound],
                        "usable_alternatives": [i for i in geo["by_category"].get(category, [])
                                                if geo["usable"][i]]})
        if bad:
            rows.append({"id": task["id"], "scene": task["scene"], "kind": label,
                         "bindings": bad})
    return rows


# ---------------------------------------------------------------------- reproductions


def belief_for(task):
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate

    extraction = task["extraction"]
    return populate(task["scene"], extraction["uncertain"], extraction["dependent"],
                    stated=extraction.get("stated") or {},
                    model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)


def reproduce(task, belief=None):
    """The reference plan, twice: as verification drives it, and as evaluation drives it."""
    from build_tasks import seed_graph
    from sim_eval import run_plan

    out = {"id": task["id"], "scene": task["scene"]}
    try:
        out["truth"] = run_plan(task, seed_graph(task), task["plan"], verbose=False)
    except Exception as exc:
        out["truth"] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    try:
        out["belief"] = run_plan(task, belief if belief is not None else belief_for(task),
                                 task["plan"], verbose=False)
    except Exception as exc:
        out["belief"] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- reporting


def report_scene(geo, used_categories):
    world = geo["world"]
    print(f"\n=== {geo['scene']} === {len(geo['rooms'])} rooms, "
          f"{len(geo['instances'])} instances, {geo['standable_m2']} m2 standable floor")
    for index, comp in sorted(geo["components"].items()):
        here = [n for n in geo["instances"] if geo["component_of"][n] == index]
        used = [n for n in here if world.category_of(n) in used_categories]
        mark = "   <- robot starts here" if index == 0 else ""
        print(f"  component {index}: {len(comp):2d} rooms, {geo['fill_sizes'][index]:6.1f} m2, "
              f"{len(here):3d} instances ({len(used)} of categories the benchmark names){mark}")
        print(f"      {', '.join(comp)}")
    print(f"  robot start: {geo['robot_xy']} in room {geo['robot_room']}, "
          f"{geo['drivable_m2']} m2 of the {geo['standable_m2']} m2 drivable")
    shut = sorted(r for r in geo["rooms"] if not geo["enterable"][r])
    if shut:
        print(f"  rooms `go_to_room` cannot enter: {', '.join(shut)}")
    unreachable = [n for n in geo["instances"] if not geo["reachable"][n]]
    unusable = [n for n in geo["instances"] if not geo["usable"][n]]
    wedged = [n for n in unreachable if geo["component_of"][n] == 0]
    print(f"  the code's predicate calls {len(unreachable)}/{len(geo['instances'])} "
          f"instances unreachable, {len(wedged)} of them inside the robot's own component "
          f"(wedged past the {REACH} m arm): {', '.join(wedged) if wedged else '-'}")
    print(f"  no plan can act on {len(unusable)}/{len(geo['instances'])} instances; "
          f"{len(geo['cross_wall'])} of those the predicate calls reachable anyway "
          f"(straight-line reach through a party wall)")
    dead = sorted(c for c, names in geo["by_category"].items()
                  if c in used_categories and not any(geo["usable"][n] for n in names))
    if dead:
        print(f"  benchmark categories with NO usable instance at all: {', '.join(dead)}")
    if geo["cost_matrix_disagreements"]:
        joined = [(a, b) for a, b, cm in geo["cost_matrix_disagreements"] if cm]
        split = [(a, b) for a, b, cm in geo["cost_matrix_disagreements"] if not cm]
        print(f"  ** cost_matrix disagrees on {len(geo['cost_matrix_disagreements'])} room "
              f"pairs: {len(split)} it calls unreachable that A* reaches, "
              f"{len(joined)} the other way")
        for a, b in split[:12]:
            print(f"       cost_matrix says inf, the floor says reachable: {a} <-> {b}")
    else:
        print(f"  cost_matrix agrees with the flood fill on all "
              f"{len(geo['rooms']) * (len(geo['rooms']) - 1) // 2} room pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", action="append", help="restrict to these scenes")
    parser.add_argument("--sim", action="store_true",
                        help="drive the reference plan of every flagged task, truth and belief")
    parser.add_argument("--sim-limit", type=int, default=80)
    parser.add_argument("--ground", action="store_true",
                        help="also ask what `ground` binds each name to under the real belief")
    parser.add_argument("--out", help="write the findings as JSON")
    args = parser.parse_args()

    scenes = args.scene or SCENES
    single = [t for t in json.load(open(SINGLE)) if t["scene"] in scenes]
    multi = [t for t in json.load(open(MULTI)) if t["scene"] in scenes]
    subs = [t for t in json.load(open(SUBTASKS)) if t["scene"] in scenes]
    groups = ((single, "single"), (multi, "multi"), (subs, "errand"))
    print(f"{len(single)} single tasks, {len(multi)} multi-task instructions, "
          f"{len(subs)} errands, over {len(scenes)} scenes")

    used = defaultdict(set)
    for group, _ in groups:
        for task in group:
            used[task["scene"]] |= set(needed_furniture(task))

    print("\nscene geometry: flood fill on the eroded floor, per-instance reachability, "
          "and cost_matrix for comparison")
    geos = {scene: scene_geometry(scene) for scene in scenes}

    used_categories = {}
    for scene in scenes:
        pool = set()
        for name in used[scene]:
            pool |= set(name_candidates(name, list(geos[scene]["by_category"])))
        used_categories[scene] = pool

    for scene in scenes:
        report_scene(geos[scene], used_categories[scene])

    print("\n\n=== tasks that name furniture no plan can act on ===")
    findings = []
    for group, label in groups:
        rows = sweep_tasks(group, geos, label)
        findings.extend(rows)
        counts = {k: len([r for r in rows if r[k]])
                  for k in ("stranded", "unreachable", "unusable", "no_instance",
                            "category_fork")}
        print(f"\n{label}: {len(group)} tasks")
        print(f"   {counts['stranded']:3d} name furniture whose every instance is OUTSIDE "
              f"the robot's component")
        print(f"   {counts['unreachable']:3d} name furniture with NO reachable instance "
              f"(the code's own predicate)")
        print(f"   {counts['unusable']:3d} name furniture with NO usable instance "
              f"(reachable AND in a room the robot can enter)")
        print(f"   {counts['no_instance']:3d} name a category the scene does not have")
        print(f"   {counts['category_fork']:3d} name a category fork one reading of which "
              f"is unusable")
        for row in [r for r in rows if r["unusable"]][:40]:
            for item in row["unusable"]:
                print(f"      {row['id']:24s} {item['name']:20s} -> "
                      f"{len(item['instances'])} instance(s) in {item['rooms']}")
        for row in [r for r in rows if r["no_instance"]][:20]:
            print(f"      {row['id']:24s} MISSING {row['no_instance']}")
        for row in [r for r in rows if r["category_fork"]][:20]:
            for item in row["category_fork"]:
                print(f"      {row['id']:24s} fork {item['name']:16s} {item['categories']} "
                      f"dead: {item['unusable_categories']}")

    print("\n\n=== `rooms` claims the scene does not bear out ===")
    room_rows = []
    for group, label in groups:
        rows = sweep_rooms(group, geos, label)
        room_rows.extend(rows)
        print(f"{label}: {len(rows)} task(s) with a room claim the scene contradicts")
        for row in rows[:40]:
            for c in row["claims"]:
                print(f"   {row['id']:24s} {c['category']:18s} declared in {c['room']:18s} "
                      f"- {c['why']} {c.get('actually_in') or c.get('instances') or ''}")

    print("\n\n=== spawn supports the robot cannot act on ===")
    spawn_rows = []
    for group, label in groups:
        rows = sweep_spawns(group, geos, label)
        spawn_rows.extend(rows)
        print(f"{label}: {len(rows)} spawn(s) land on a support no plan can act on")
        for row in rows[:40]:
            print(f"   {row['id']:24s} {row['object']:16s} on {row['target']:16s} -> "
                  f"{row['support']} in {row.get('support_room')}  ({row['why']})")

    result = {"scenes": {s: {k: v for k, v in geos[s].items()
                             if k not in ("world", "probe")} for s in scenes},
              "tasks": findings, "spawns": spawn_rows, "room_claims": room_rows}

    beliefs = {}
    ground_rows = []
    if args.ground:
        # Only the scenes that hold an unusable instance can produce an unusable binding, so
        # the rest need no run at all - the restriction is a proof, not a sample.
        risky = {s for s in scenes if any(not u for u in geos[s]["usable"].values())}
        want = [t for group, _ in groups for t in group if t["scene"] in risky]
        print(f"\n\n=== what `ground` binds under the real belief: {len(want)} tasks in "
              f"{len(risky)} scene(s) that hold an unusable instance "
              f"({', '.join(sorted(risky))}) ===")
        print(f"    the other {len(scenes) - len(risky)} scene(s) cannot produce one: "
              f"every instance in them is usable")
        began = time.time()
        for i, task in enumerate(want, 1):
            try:
                beliefs[task["id"]] = belief_for(task)
            except Exception as exc:
                print(f"   {task['id']}: {type(exc).__name__}: {exc}")
            if i % 100 == 0:
                print(f"   {i}/{len(want)} beliefs ({time.time() - began:.0f}s)", flush=True)
        for group, label in groups:
            rows = sweep_grounding([t for t in group if t["scene"] in risky],
                                   geos, beliefs, label)
            ground_rows.extend(rows)
            print(f"{label}: {len(rows)} task(s) where `ground` binds a name to something "
                  f"no plan can act on", flush=True)
            for row in rows[:40]:
                for b in row.get("bindings", ()):
                    print(f"   {row['id']:24s} {b['name']:18s} -> {b['bound']} "
                          f"in {b['room']} (component {b['component']}); "
                          f"usable alternatives: {b['usable_alternatives'] or 'none'}")
        result["grounding"] = ground_rows

    if args.sim:
        flagged = {}
        for group, _ in groups:
            for task in group:
                tid = task["id"]
                if (any(r["id"] == tid and (r["unusable"] or r["no_instance"] or
                                            r["category_fork"]) for r in findings)
                        or any(r["id"] == tid for r in spawn_rows)
                        or any(r["id"] == tid for r in ground_rows)):
                    flagged[tid] = task
        order = list(flagged.values())[:args.sim_limit]
        print(f"\n\n=== driving the reference plan of {len(order)} flagged task(s) "
              f"of {len(flagged)} ===")
        runs = []
        for i, task in enumerate(order, 1):
            began = time.time()
            row = reproduce(task, beliefs.get(task["id"]))
            runs.append(row)
            print(f"{i:3d}/{len(order)} {task['id']:24s} "
                  f"truth {'ok  ' if row['truth']['ok'] else 'FAIL'} | "
                  f"belief {'ok  ' if row['belief']['ok'] else 'FAIL'} "
                  f"({time.time() - began:4.1f}s)", flush=True)
            if not row["truth"]["ok"]:
                print(f"        truth : {row['truth']['why'][:110]}")
            if not row["belief"]["ok"]:
                print(f"        belief: {row['belief']['why'][:110]}")
        result["runs"] = runs
        both = [r for r in runs if not r["truth"]["ok"] and not r["belief"]["ok"]]
        only = [r for r in runs if r["truth"]["ok"] and not r["belief"]["ok"]]
        print(f"\n{len(both)} fail against BOTH truth and belief - impossible outright")
        print(f"{len(only)} fail against the belief only")

    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1, default=str)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
