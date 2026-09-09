#!/usr/bin/env python3
"""Why four plans reached for something the robot had never driven to.

Four failures, three of them the same shape. In `Beechwood_0_int-08` (both models),
`Wainscott_0_int-07` (both models) and `Ihlen_1_int-07` (8B) the validation loop ACCEPTED
the plan - `accepted_at` is 1 or 2, so `GraphMachine` found every precondition satisfied -
and the simulator then refused the sixth step because the object was 5.8-8.7 m away. A
symbolic checker and a driven world disagreeing about `near` is the one disagreement this
harness exists to detect, so it is worth knowing exactly which of the two is wrong.

The fourth, `Wainscott_0_int-02` (4B), is a search that swept seven rooms and found
nothing. The scene is Wainscott, which the room-graph work flagged as one of the houses
whose floor plan comes in pieces, so "the scene is broken" is the cheap answer and has to
be ruled out before it is believed.

WHAT THIS SCRIPT MEASURES

  A. the mechanism   Rebuild the belief the pipeline planned against, replay the accepted
                     plan through `GraphMachine` step by step, and print the `nearby` set
                     at the refused GRASP together with the belief edge that put the object
                     there. No inference: the reach set is read out of the graph.

  B. counterfactual  Rebuild the belief from the CORRECT extraction in `data/tasks.json`,
                     re-check the same plan, re-run `repair` on the model's own first plan,
                     and DRIVE what comes out. Only a driven pass shows the gap cost the
                     task.

  C. the search      For `Wainscott_0_int-02`: where the cheese really is, which rooms the
                     belief ranked, whether the floor grid connects them, and two probe
                     plans that differ only in whether the fridge is opened first.

Nothing here is modified in place; every module is imported and used as the pipeline uses
it. Run:

    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 python exp_fail_reach.py [task-id]

The only thing wanting a card is the RSN, and `scene_graph.populate` is the only call that
takes a device - `query_rsn.predict_rooms` reaches `embed_categories.embed_names`, which
builds its own `SentenceTransformer` on whatever CUDA it finds and ignores the argument.
So when the card is busy with a benchmark run, take it away from the process entirely:

    HF_HUB_OFFLINE=1 SF_DEVICE=cpu CUDA_VISIBLE_DEVICES= python exp_fail_reach.py

Verified to give bit-identical verdicts either way; the RSN is an argmax.
"""

import json
import os
import sys

from build_tasks import seed_graph
from graph_machine import GraphMachine
from object_names import same
from repair import MAX_ROUNDS, REPAIRABLE, _discharge, _edit, repair
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import build_world, run_plan
from world_graph import WorldGraph

TASKS = "data/tasks.json"
RESULTS = {"4B": "data/v8-4b.json", "8B": "data/v8-8b.json"}

# The failures in this group, and which model saw each.
GROUP = [
    ("Beechwood_0_int-08", ("4B", "8B")),
    ("Wainscott_0_int-07", ("4B", "8B")),
    ("Ihlen_1_int-07", ("4B", "8B")),
    ("Wainscott_0_int-02", ("4B",)),
]

# A plan recorded as `final_plan` has already been mended. Where the row records exactly
# one mend, the model's own words are recoverable by undoing it - and for a task accepted
# on attempt 2 that written plan, not attempt 1's, is what the counterfactual must repair.
# Ihlen_1_int-07/8B: mended = ["holding(plate) at step 4: 12 -> 14 actions"], so the two
# actions the repair spliced in at index 3 come back out.
EXTRA_PROBES = {
    ("Ihlen_1_int-07", "8B"): (
        "attempt-2's WRITTEN plan (final_plan with the one recorded mend undone), "
        "repaired under the CORRECT belief",
        [("NAVIGATE_TO", "plate"), ("GRASP", "plate"), ("NAVIGATE_TO", "mug"),
         ("GRASP", "mug"), ("NAVIGATE_TO", "coffee_table"),
         ("PLACE_ON_TOP", "coffee_table"), ("NAVIGATE_TO", "plate"),
         ("GRASP", "plate"), ("NAVIGATE_TO", "coffee_table"),
         ("PLACE_ON_TOP", "coffee_table"), ("NAVIGATE_TO", "floor_lamp"),
         ("TOGGLE_ON", "floor_lamp")]),
}

_BELIEF_CACHE = {}


def rule(char="=", width=100):
    print(char * width)


def load_rows():
    tasks = {t["id"]: t for t in json.load(open(TASKS))}
    rows = {}
    for tag, path in RESULTS.items():
        blob = json.load(open(path))
        rows[tag] = {r["id"]: r for r in blob["rows"]}
    return tasks, rows


def belief_from(scene, extraction, key):
    """The graph the planner was shown, built the way `evaluate.py` builds it."""
    if key in _BELIEF_CACHE:
        return _BELIEF_CACHE[key]
    # The RSN is the only thing here that wants a GPU, and the card is often busy with a
    # benchmark run. `SF_DEVICE=cpu` moves just that; everything downstream is numpy.
    graph = populate(scene, extraction.get("uncertain") or [],
                     extraction.get("dependent") or [],
                     stated=extraction.get("stated") or {},
                     model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD,
                     device=os.environ.get("SF_DEVICE"))
    _BELIEF_CACHE[key] = graph
    return graph


def as_plan(steps):
    return [(a, o) for a, o in steps]


def as_goal(goal):
    return [tuple(g) for g in goal]


def show_belief(graph, names):
    """Where the belief puts each of these, and on what."""
    objects = graph.get("objects") or {}
    for name in names:
        info = objects.get(name)
        if info is None:
            print(f"      {name:16s} ABSENT from the belief")
            continue
        rels = [f"{r['relation']} {r['to']}" for r in (graph.get("relations") or [])
                if r.get("from") == name]
        print(f"      {name:16s} room={info.get('room'):20s} p={info.get('probability'):.2f}"
              f"  relations={rels or '[]'}  candidates={(info.get('candidates') or [])[:4]}")


def replay(graph_dict, plan, goal, upto):
    """Run the plan a step at a time and hand back the machine just before step `upto`."""
    machine = GraphMachine(WorldGraph.from_scene_graph(graph_dict).copy())
    for i, (action, arg) in enumerate(plan):
        if i == upto:
            return machine
        machine.step(i, action, arg)
    return machine


def goal_name_check(graph_dict, goal):
    """Does the known `unmet` == defect bite here? Ask it out loud rather than assume."""
    objects = list((graph_dict.get("objects") or {}))
    bad = []
    for entry in goal:
        for name in (entry[1], entry[2]):
            if not isinstance(name, str) or name in objects:
                continue
            hit = [o for o in objects if same(name, o)]
            bad.append((name, hit))
    return bad


# ---------------------------------------------------------------- A. the three reach cases

def reach_case(task, row, tag):
    ident = task["id"]
    rule()
    print(f"A.  {ident}   [{tag}]   accepted_at={row['accepted_at']}  attempts={row['attempts']}")
    print(f"    task: {task['task']}")
    print(f"    stage-1 error recorded: {row['extraction']}")
    print(f"    simulator refused: {row['simulated']['why']}")
    rule("-")

    heard = row["extracted"]
    truth_ext = task["extraction"]
    print("    extraction, as heard vs as it should have been")
    print(f"      heard.dependent = {heard['dependent']}")
    print(f"      truth.dependent = {truth_ext['dependent']}")
    print(f"      heard.stated    = {heard['stated']}")
    print(f"      truth.stated    = {truth_ext['stated']}")

    heard_graph = belief_from(task["scene"], heard, (ident, "heard"))
    truth_graph = belief_from(task["scene"], truth_ext, (ident, "truth"))

    plan = as_plan(row["final_plan"])
    goal = as_goal(row["predicted_goal"])
    failed_at = row["simulated"]["failed_at"]          # 0-based index of the refused step
    action, arg = plan[failed_at]

    names = sorted({n for _, n in plan if n} | {g[1] for g in goal} | {g[2] for g in goal
                                                                      if isinstance(g[2], str)})
    print("\n    the BELIEF the planner was shown (heard extraction)")
    show_belief(heard_graph, names)
    print("    the BELIEF it would have been shown (correct extraction)")
    show_belief(truth_graph, names)

    # --- what the checker thought -------------------------------------------------------
    out = GraphMachine(WorldGraph.from_scene_graph(heard_graph).copy()).run(plan, goal)
    print(f"\n    GraphMachine on the accepted plan, HEARD belief:"
          f"  failed_at={out.failed_at}  goal_met={out.goal_met}  safe={out.safe}")
    print(f"    goal-name resolution check (the known `unmet` == defect): "
          f"{goal_name_check(heard_graph, goal) or 'every goal name is a graph node - defect does not bite'}")

    # Does the wrong belief already satisfy the goal before the robot moves? That is the
    # sharpest statement of what the invented relation did.
    for label, gdict in (("heard", heard_graph), ("correct", truth_graph)):
        pre = GraphMachine(WorldGraph.from_scene_graph(gdict).copy()).run([], goal)
        held = [g for g in goal if tuple(g) not in {tuple(m) for m in pre.missing}]
        print(f"    goal edges ALREADY TRUE in the {label} belief, before any action: "
              f"{len(held)}/{len(goal)}  {held}")

    machine = replay(heard_graph, plan, goal, failed_at)
    nearby = sorted(machine.graph.near_objects())
    print(f"\n    at step {failed_at + 1} the plan says {action}({arg}).")
    print(f"      robot believes it is standing at: {nearby}")
    print(f"      graph.is_near({arg!r}) = {machine.graph.is_near(arg)}")
    print(f"      the true distance the simulator measured: "
          f"{row['simulated']['why'].split(' is ')[-1].split(',')[0]}")
    prev = plan[failed_at - 1] if failed_at else None
    if prev:
        reach = sorted(machine._within_reach_of(prev[1])) if prev[1] else []
        print(f"      previous step was {prev[0]}({prev[1]}); "
              f"_within_reach_of({prev[1]!r}) = {reach}")
    edges = [(e, a, b) for e in ("on_top", "object_inside")
             for a, b in machine.graph.edges_of(e, src=arg)]
    print(f"      belief edges out of {arg!r} at this moment: {edges}")

    # --- the same plan, correct belief ---------------------------------------------------
    out2 = GraphMachine(WorldGraph.from_scene_graph(truth_graph).copy()).run(plan, goal)
    why = (out2.steps[out2.failed_at].reason if out2.failed_at is not None else "")
    print(f"\n    B1. same plan re-checked against the CORRECT belief: "
          f"failed_at={out2.failed_at}  goal_met={out2.goal_met}")
    if out2.failed_at is not None:
        print(f"        step {out2.failed_at + 1} {plan[out2.failed_at][0]}"
              f"({plan[out2.failed_at][1]}): {why}")
        print(f"        fault = {out2.steps[out2.failed_at].fault}")

    # --- repair the model's own first plan under the correct belief ----------------------
    first = as_plan(row["first_plan"])
    seed_true = WorldGraph.from_scene_graph(truth_graph)
    fixed, notes = repair(seed_true, first, goal)
    out3 = GraphMachine(seed_true.copy()).run(fixed, goal)
    print(f"\n    B2. `repair` re-run on the model's FIRST plan under the CORRECT belief")
    print(f"        notes: {notes or '(no edit improved it)'}")
    print(f"        would the loop accept it? failed_at={out3.failed_at} "
          f"goal_met={out3.goal_met} safe={out3.safe}  -> "
          f"{'ACCEPT' if out3.failed_at is None and out3.goal_met and out3.safe else 'REFUSE'}")
    for i, (a, o) in enumerate(fixed, 1):
        print(f"          {i:2d}. {a}({o})")

    # A plan the loop mended is not the plan the model wrote. Where the row records a
    # single mend, the written plan can be recovered by undoing it, and that is the plan
    # the counterfactual should really be run on for a task accepted on a later attempt.
    extra = EXTRA_PROBES.get((ident, tag))
    if extra:
        label, written = extra
        wfixed, wnotes = repair(seed_true, written, goal)
        wout = GraphMachine(seed_true.copy()).run(wfixed, goal)
        print(f"\n    B2b. {label}")
        print(f"        notes: {wnotes or '(no edit improved it)'}")
        print(f"        would the loop accept it? failed_at={wout.failed_at} "
              f"goal_met={wout.goal_met} safe={wout.safe}  -> "
              f"{'ACCEPT' if wout.failed_at is None and wout.goal_met and wout.safe else 'REFUSE'}")
        for i, (a, o) in enumerate(wfixed, 1):
            print(f"          {i:2d}. {a}({o})")
        v = run_plan(task, truth_graph, wfixed, verbose=False)
        print(f"        driven: ok={v['ok']} driven={v['driven']}m steps={v['steps_run']} "
              f"goal_met={v['goal_met']}  {v['why'][:80]}")
        # The same written plan under the relation-stripped belief, so B2b and B2c can be
        # compared on equal terms.
        sg = json.loads(json.dumps(heard_graph))
        sg["relations"] = []
        sseed = WorldGraph.from_scene_graph(sg)
        sfix, snote = repair(sseed, written, goal)
        so = GraphMachine(sseed.copy()).run(sfix, goal)
        sv = run_plan(task, sg, sfix, verbose=False)
        print(f"        same written plan, RELATION-STRIPPED belief: {snote or '(none)'}")
        print(f"          accept? failed_at={so.failed_at} goal_met={so.goal_met} "
              f"safe={so.safe}  driven: ok={sv['ok']} {sv['driven']}m "
              f"goal_met={sv['goal_met']}  {sv['why'][:70]}")

    # B2c separates the two things the invented relation did. It put the object in the
    # WRONG ROOM (private_office_0 for a notebook that is in the kitchen) and it granted
    # FALSE REACH (`_within_reach_of(desk)` returns the notebook, because the belief says
    # the notebook is on the desk). Strip the relation and keep the wrong room, and only
    # the second is undone: if that alone rescues the task, the fatal half is the reach
    # rule and the wrong room is survivable, because the search layer works down the
    # ranking. If it does not, the room was fatal too and only stage 1 can fix it.
    stripped = json.loads(json.dumps(heard_graph))
    stripped["relations"] = []
    seed_strip = WorldGraph.from_scene_graph(stripped)
    sout = GraphMachine(seed_strip.copy()).run(plan, goal)
    sfixed, snotes = repair(seed_strip, first, goal)
    sout2 = GraphMachine(seed_strip.copy()).run(sfixed, goal)
    print("\n    B2c. heard belief with the invented RELATION stripped, wrong ROOMS kept")
    print(f"        accepted plan re-checked: failed_at={sout.failed_at}"
          + (f"  ({sout.steps[sout.failed_at].fault})" if sout.failed_at is not None else ""))
    print(f"        repair on the first plan: {snotes or '(none)'}")
    print(f"        would the loop accept it? failed_at={sout2.failed_at} "
          f"goal_met={sout2.goal_met} safe={sout2.safe}  -> "
          f"{'ACCEPT' if sout2.failed_at is None and sout2.goal_met and sout2.safe else 'REFUSE'}")
    v = run_plan(task, stripped, sfixed, verbose=False)
    print(f"        driven: ok={v['ok']} driven={v['driven']}m steps={v['steps_run']} "
          f"goal_met={v['goal_met']}  {v['why'][:90]}")

    # --- and drive it --------------------------------------------------------------------
    print("\n    B3. driven in the simulator (the TRUE world, always)")
    for label, gdict, steps in (
        ("accepted plan  / heard belief   (the recorded failure)", heard_graph, plan),
        ("accepted plan  / correct belief", truth_graph, plan),
        ("repaired plan  / correct belief", truth_graph, fixed),
        ("reference plan / correct belief (control)", truth_graph, as_plan(task["plan"])),
    ):
        try:
            v = run_plan(task, gdict, steps, verbose=False)
            print(f"        {label:48s} ok={str(v['ok']):5s} driven={v['driven']:6.1f}m "
                  f"steps={v['steps_run']:2d} goal_met={v['goal_met']}  {v['why'][:80]}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"        {label:48s} EXC {type(exc).__name__}: {exc}")


def repair_progress_first(graph, plan, goal=(), rounds=MAX_ROUNDS):
    """`repair.repair` with one line changed: how two outcomes are ranked.

    `repair._rank` is `(applies, goal_met, safe, progress, -length)`. Safety sits *above*
    progress, and a plan refused at step 1 never opened anything, so it is vacuously safe -
    which makes it beat any repair that actually runs and leaves a fridge open. The final
    guard in `repair` then rolls the repair back and returns the plan untouched. That is
    the same vacuum the docstring of `_rank` says it fixed; it fixed it only for the pair
    where `applies` differs.

    This variant ranks progress before safety and is otherwise `repair` verbatim, so the
    difference between the two is exactly that ordering. It is a counterfactual, not a
    proposal - `repair.py` is not modified.
    """
    def rank(outcome, length):
        return (outcome.failed_at is None, outcome.goal_met,
                outcome.failed_at if outcome.failed_at is not None else len(outcome.steps),
                outcome.safe, -length)

    def run(candidate):
        return GraphMachine(graph.copy()).run(candidate, goal)

    start_out = run(plan)
    best, best_out = list(plan), start_out
    current, notes, seen = list(plan), [], {tuple(plan)}
    for _ in range(rounds):
        out = run(current)
        if rank(out, len(current)) > rank(best_out, len(best)):
            best, best_out = list(current), out
        if out.failed_at is None:
            break
        kind, subject = out.steps[out.failed_at].fault or (None, None)
        if kind not in REPAIRABLE:
            break
        candidate = _edit(out.graph, current, out.failed_at, kind, subject, goal)
        if candidate is None or tuple(candidate) in seen:
            break
        seen.add(tuple(candidate))
        notes.append(f"{kind}({subject}) at step {out.failed_at + 1}: "
                     f"{len(current)} -> {len(candidate)} actions")
        current = candidate
    if best_out.failed_at is None and not best_out.safe:
        candidate = _discharge(best, best_out)
        out = run(candidate)
        if rank(out, len(candidate)) > rank(best_out, len(best)):
            notes.append(f"discharged {len(candidate) - len(best)} left-open/left-on")
            best, best_out = candidate, out
    if rank(best_out, len(best)) <= rank(start_out, len(plan)):
        return list(plan), []
    return best, notes


# ---------------------------------------------------------------- C. the missing object

def search_case(task, rows):
    ident = task["id"]
    rule()
    print(f"C.  {ident}   the object the robot never found")
    print(f"    task: {task['task']}")
    for tag in ("4B", "8B"):
        r = rows[tag].get(ident)
        if r:
            print(f"    [{tag}] accepted_at={r['accepted_at']} checked={r['checked']} "
                  f"-> {r['checked_why'][:110]}")
    rule("-")

    row = rows["4B"][ident]
    heard = row["extracted"]
    graph = belief_from(task["scene"], heard, (ident, "heard"))

    print("    what the belief says")
    show_belief(graph, ["swiss_cheese", "fridge", "microwave", "breakfast_table"])
    print(f"    relations: {graph.get('relations')}")

    # Where the cheese REALLY is, and whether the robot could stand next to it.
    truth = seed_graph(task)
    print(f"\n    seed_graph (ground truth belief-free world) says:")
    for name in ("swiss_cheese", "fridge"):
        info = (truth.get("objects") or {}).get(name)
        print(f"      {name:16s} {info}")
    print(f"      relations: {truth.get('relations')}")

    world, placed = build_world(task, graph, [a for _, a in as_plan(row["final_plan"])])
    print(f"\n    the TRUE house `sim_eval.build_world` actually built (spawned {placed})")
    for cat in ("swiss_cheese", "fridge", "microwave", "breakfast_table"):
        for inst in world.truth.by_category(cat):
            print(f"      {inst:32s} room={world.room_of(inst)}"
                  f"  open={world.open.get(inst)}")
    holders = [(a, b) for a, b in world.truth.edges_of("object_inside", src="swiss_cheese")]
    print(f"      swiss_cheese is INSIDE {holders}")
    for _, container in holders:
        print(f"      is {container} open at spawn? {world.open.get(container)}  "
              f"-> `Sim2D.can_see` gates on exactly this")

    # Which rooms the belief told the robot to sweep, in order, and whether they connect.
    from sim2d import Sim2D
    hints = (graph["objects"].get("swiss_cheese") or {}).get("candidates") or []
    print(f"\n    room ranking handed to the search: {hints}")
    rooms_ranked = list(hints)
    ruled = row["checked_why"]
    print(f"    rooms the run actually ruled out: {ruled[ruled.find('none of'):][:160]}")

    probe = Sim2D(world, verbose=False)
    scene_rooms = sorted(world.rooms)
    print(f"\n    scene has {len(scene_rooms)} rooms: {scene_rooms}")
    print("    can the robot DRIVE to each from its start? (A* on the floor grid)")
    unreachable = []
    for room in scene_rooms:
        route, problem = probe.go_to_room(room)
        okr = route is not None
        if not okr:
            unreachable.append(room)
        print(f"      {room:18s} {'reachable' if okr else 'UNREACHABLE  ' + str(problem)}")
    print(f"    -> {len(scene_rooms) - len(unreachable)}/{len(scene_rooms)} rooms drivable; "
          f"disconnected pieces: {unreachable or 'none'}")

    # Two probe plans differing only in whether the fridge is opened first.
    print("\n    probe plans (same world, same belief, only the ordering differs)")
    for label, steps in (
        ("NAVIGATE_TO swiss_cheese first (what the 4B's final plan did)",
         [("NAVIGATE_TO", "swiss_cheese"), ("GRASP", "swiss_cheese")]),
        ("stand AT the fridge but do not open it, then go to the cheese",
         [("NAVIGATE_TO", "fridge"), ("NAVIGATE_TO", "swiss_cheese")]),
        ("OPEN the fridge first, then go to the cheese",
         [("NAVIGATE_TO", "fridge"), ("OPEN", "fridge"),
          ("NAVIGATE_TO", "swiss_cheese"), ("GRASP", "swiss_cheese")]),
    ):
        v = run_plan(task, graph, steps, verbose=False)
        print(f"      {label:62s} steps_run={v['steps_run']} driven={v['driven']:6.1f}m")
        print(f"        why: {v['why'] or 'every action applied'}")

    print("\n    the 4B's two plans, side by side")
    print(f"      first_plan[0:4] = {row['first_plan'][:4]}   -> sim_first: "
          f"{rows['4B'][ident]['sim_first']['why'][:80]}")
    print(f"      final_plan[0:4] = {row['final_plan'][:4]}   -> simulated: "
          f"{row['simulated']['why'][:80]}")
    r8 = rows["8B"].get(ident)
    if r8:
        print(f"      8B final_plan[0:4] = {r8['final_plan'][:4]}  -> {r8['checked']} "
              f"({r8['simulated']['driven']} m, goal_met={r8['simulated']['goal_met']})")

    # What the SYMBOLIC checker made of the same plan. The loop never accepted it
    # (accepted_at is None), so this is not a checker/simulator disagreement at all - both
    # refused it, for different reasons, and the search failure is only what the simulator
    # noticed first.
    for label, steps in (("first_plan", row["first_plan"]), ("final_plan", row["final_plan"])):
        o = GraphMachine(WorldGraph.from_scene_graph(graph).copy()).run(
            as_plan(steps), as_goal(row["predicted_goal"]))
        why = o.steps[o.failed_at].reason if o.failed_at is not None else ""
        print(f"\n    GraphMachine on the 4B's {label}: failed_at={o.failed_at} "
              f"goal_met={o.goal_met} safe={o.safe} left_open={o.left_open}")
        if o.failed_at is not None:
            print(f"      step {o.failed_at + 1} {steps[o.failed_at]}: {why[:100]}")
            print(f"      fault = {o.steps[o.failed_at].fault}")

    # `closed` is in `repair.REPAIRABLE`, so ask why the loop did not mend it - the row
    # records `mended: null`, meaning `repair` handed the plan back untouched.
    seed = WorldGraph.from_scene_graph(graph)
    for label, steps in (("first_plan", row["first_plan"]), ("final_plan", row["final_plan"])):
        fixed, notes = repair(seed, as_plan(steps), as_goal(row["predicted_goal"]))
        o = GraphMachine(seed.copy()).run(fixed, as_goal(row["predicted_goal"]))
        print(f"\n    `repair` on the 4B's {label}: {notes or '(handed back untouched)'}")
        print(f"      failed_at={o.failed_at} goal_met={o.goal_met} safe={o.safe}")
        if fixed != as_plan(steps):
            v = run_plan(task, graph, fixed, verbose=False)
            print(f"      driven: ok={v['ok']} {v['driven']}m steps={v['steps_run']} "
                  f"goal_met={v['goal_met']}  {v['why'][:90]}")

    print("\n    the same two plans through a repair that ranks PROGRESS above SAFETY")
    for label, steps in (("first_plan", row["first_plan"]), ("final_plan", row["final_plan"])):
        fixed, notes = repair_progress_first(seed, as_plan(steps),
                                             as_goal(row["predicted_goal"]))
        o = GraphMachine(seed.copy()).run(fixed, as_goal(row["predicted_goal"]))
        print(f"      {label}: {notes or '(handed back untouched)'}")
        print(f"        failed_at={o.failed_at} goal_met={o.goal_met} safe={o.safe} -> "
              f"{'ACCEPT' if o.failed_at is None and o.goal_met and o.safe else 'REFUSE'}")
        if fixed != as_plan(steps):
            v = run_plan(task, graph, fixed, verbose=False)
            print(f"        driven: ok={v['ok']} {v['driven']}m steps={v['steps_run']} "
                  f"goal_met={v['goal_met']}  {v['why'][:90]}")

    # And the control: does the reference plan drive under this same belief?
    v = run_plan(task, graph, as_plan(task["plan"]), verbose=False)
    print(f"\n    control: the reference plan under the 4B's own belief -> ok={v['ok']} "
          f"driven={v['driven']}m  {v['why'][:80]}")


def main():
    tasks, rows = load_rows()
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for ident, tags in GROUP:
        if only and only != ident:
            continue
        task = tasks[ident]
        if ident == "Wainscott_0_int-02":
            search_case(task, rows)
            continue
        for tag in tags:
            reach_case(task, rows[tag][ident], tag)
    rule()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
