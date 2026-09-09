#!/usr/bin/env python3
"""Why the five SWAP tasks fail, measured rather than asserted.

A "swap" wants A where B is and B where A is, with a named third surface to park one of
them on. The five tasks below all end with `checked == "goal"` and `accepted_at is None`:
no attempt was ever accepted, and the plan that was driven did not do the task.

The README calls those plans "no-ops". Measured, three of the five driven plans are exact
no-ops - every object back on the support it started on - one is a no-op plus a stranded
blanket on the parking surface, and one is a half swap that never brings the second object
back. So the claim is right about the shape and worth stating precisely.

The script separates the explanations that would each look the same from the outside:

  1. the goal model misread the sentence      -> compare `predicted_goal` with tasks.json
  2. the checker refused a correct plan       -> replay the reference plan against the
                                                 PREDICTED goal on the SAME belief graph
  3. the planner cannot write a swap          -> replay the plan in the TRUE world and read
                                                 off where the two objects actually end up

and the harness defect the brief names: `GraphMachine.unmet` resolves goal names through
`_resolve_goal_name`, which in the committed version is `return name if name in
self.graph.objects else name` - a no-op. If the predicted goal says `tv` and the graph says
`standing_tv`, every plan is refused for a reason that has nothing to do with the plan.
Section D rewrites the predicted goal through `object_names.match` against the graph's own
names and re-runs the accept test, so the claim "the name gap cost this task" is either
measured or dropped. (It is dropped: on all five, every goal term is already a literal key
of the belief graph, so the rewrite is the identity and the verdict does not move.)

Sections:
  0  is the belief graph rebuilt today still the one the recorded run planned against
  A  the predicted goal against the answer key
  B  what the plan does in the TRUE world - the net effect, object by object
  C  the reference plan, replayed and driven, so we know the task is possible at all
  D  the name-gap counterfactual on the in-loop accept test
  E  Pomaria_1_int-10 only: the stage-1 "error" against what the task actually spawns
  H  the same task under the other model, from the recorded runs
  G  --mend: a goal-directed mend derived from the checker's own `missing` list, driven
  I  --benchmark-mend: that mend over every "plan ran, goal unmet" row in both runs
  F  --replan: re-run the five-attempt loop and read every attempt

  Section F is a counterfactual, not a reproduction. `scene_graph.populate` has been
  edited since these results were written, so the planner sees a different scene-graph
  block; section 0 establishes what the rebuild does still reproduce, which is the
  grounding and the verdict, and F is read as "the same model on the same task today".

    python exp_fail_swaps.py --device cpu                        # 0,A-E,H
    python exp_fail_swaps.py --device cpu --mend --benchmark-mend # + G,I
    python exp_fail_swaps.py --replan                             # + F (needs a GPU)
"""

import argparse
import json
import sys

from build_tasks import seed_graph
from graph_machine import GraphMachine
from object_names import match, same
from world_graph import WorldGraph

CASES = [
    ("Ihlen_1_int-09", "Qwen/Qwen3-4B", "data/v8-4b.json"),
    ("Merom_1_int-09", "Qwen/Qwen3-4B", "data/v8-4b.json"),
    ("Pomaria_0_int-06", "Qwen/Qwen3-8B", "data/v8-8b.json"),
    ("Rs_int-10", "Qwen/Qwen3-8B", "data/v8-8b.json"),
    ("Pomaria_1_int-10", "Qwen/Qwen3-4B", "data/v8-4b.json"),
]

SUPPORT_EDGES = ("on_top", "object_inside")


def load():
    tasks = {t["id"]: t for t in json.load(open("data/tasks.json"))}
    rows = {}
    for path in {p for _, _, p in CASES}:
        for r in json.load(open(path))["rows"]:
            rows[(path, r["id"])] = r
    return tasks, rows


def support_of(graph, name):
    """Where `name` rests, as (relation, support), or None if it rests on nothing."""
    for edge in SUPPORT_EDGES:
        for _, dst in graph.edges_of(edge, src=name):
            return edge, dst
    return None


def replay(task, plan, graph=None):
    """Run `plan` in a world and hand back the machine's outcome and the final graph."""
    seed = WorldGraph.from_scene_graph(graph if graph is not None else seed_graph(task))
    machine = GraphMachine(seed.copy())
    out = machine.run([(a, o) for a, o in plan], [tuple(g) for g in task["goal"]])
    return out, machine.graph


def moved_objects(task):
    return [s["name"] for s in task.get("spawn", [])]


def where_everything_is(task, graph):
    return {name: support_of(graph, name) for name in moved_objects(task)}


def section_zero(tasks, rows):
    """Is the belief graph rebuilt today the one the recorded run planned against?

    `scene_graph.populate` has been edited since these results were written - the RSN's
    ranking now covers every room instance rather than the largest of each type - so a
    counterfactual built on a graph rebuilt now is only worth reading if the rebuild still
    drives the recorded plans to the recorded verdict. This checks that before anything
    else uses it.
    """
    from sim_eval import run_plan
    print("=" * 78)
    print("0. is the rebuilt belief faithful? re-drive the RECORDED final_plan")
    print("=" * 78)
    same_all = True
    for tid, model, path in CASES:
        task, row = tasks[tid], rows[(path, tid)]
        v = run_plan(task, believed_graph(row, task), row["final_plan"], verbose=False)
        rec = row["simulated"]
        agree = rec["ok"] == v["ok"] and rec["missing"] == v["missing"]
        same_all &= agree
        print(f"  {tid:22s} recorded ok={rec['ok']} missing={rec['missing']}")
        print(f"  {'':22s} rebuilt  ok={v['ok']} missing={v['missing']}   agree={agree}")
    print(f"\n  -> the rebuild reproduces the recorded verdict on "
          f"{'all' if same_all else 'NOT all'} five")
    return same_all


def section_a(tasks, rows):
    print("=" * 78)
    print("A. the predicted goal against the answer key (tasks.json['goal'])")
    print("=" * 78)
    verdicts = {}
    for tid, model, path in CASES:
        task, row = tasks[tid], rows[(path, tid)]
        truth = [tuple(g) for g in task["goal"]]
        pred = [tuple(g) for g in row["predicted_goal"]]
        exact = sorted(truth) == sorted(pred)
        fuzzy = len(truth) == len(pred) and all(
            any(t[0] == p[0] and same(t[1], p[1]) and same(str(t[2]), str(p[2]))
                for p in pred) for t in truth)
        verdicts[tid] = exact
        print(f"\n{tid}  ({model})")
        print(f"  truth     {truth}")
        print(f"  predicted {pred}")
        print(f"  identical strings: {exact}   equal under object_names.same: {fuzzy}")
    print(f"\n  -> goal model correct on {sum(verdicts.values())}/{len(CASES)}")
    return verdicts


def section_b(tasks, rows):
    print()
    print("=" * 78)
    print("B. what the plans DO - replayed against the TRUE world (build_tasks.seed_graph)")
    print("=" * 78)
    effects = {}
    for tid, model, path in CASES:
        task, row = tasks[tid], rows[(path, tid)]
        seed = WorldGraph.from_scene_graph(seed_graph(task))
        start = where_everything_is(task, seed)
        print(f"\n{tid}  ({model})   {task['task']}")
        print(f"  start:  " + ", ".join(f"{k} {v[0]} {v[1]}" if v else f"{k} nowhere"
                                        for k, v in start.items()))
        print(f"  goal:   " + ", ".join(f"{t}({a}, {b})" for t, a, b in task["goal"]))
        per = {}
        for label in ("first_plan", "final_plan"):
            plan = [tuple(s) for s in row[label]]
            out, graph = replay(task, plan)
            end = where_everything_is(task, graph)
            unchanged = end == start
            per[label] = {"end": end, "noop": unchanged,
                          "failed_at": out.failed_at, "missing": out.missing}
            print(f"  {label:11s} ({len(plan):2d} actions) "
                  f"{'REFUSED at step %d' % (out.failed_at + 1) if out.failed_at is not None else 'ran to the end'}")
            print(f"    end:  " + ", ".join(f"{k} {v[0]} {v[1]}" if v else f"{k} nowhere"
                                            for k, v in end.items()))
            print(f"    net effect vs start: "
                  + ("NO-OP - the world is exactly as it began" if unchanged
                     else "changed: " + ", ".join(
                         f"{k}: {start[k]} -> {end[k]}" for k in end if end[k] != start[k])))
            print(f"    true goal missing: {out.missing}")
        effects[tid] = per
    return effects


def section_c(tasks, rows, drive=True):
    print()
    print("=" * 78)
    print("C. is the task possible at all? the reference plan, replayed and driven")
    print("=" * 78)
    from sim_eval import run_plan
    results = {}
    for tid, model, path in CASES:
        task, row = tasks[tid], rows[(path, tid)]
        out, graph = replay(task, [tuple(s) for s in task["plan"]])
        end = where_everything_is(task, graph)
        print(f"\n{tid}")
        print(f"  reference plan, {len(task['plan'])} actions, replayed symbolically:")
        print(f"    applies: {out.failed_at is None}   goal met: {out.goal_met}   "
              f"safe: {out.safe}   end: {end}")
        entry = {"symbolic_ok": out.failed_at is None and out.goal_met and out.safe}
        if drive:
            truth_drive = run_plan(task, seed_graph(task), task["plan"], verbose=False)
            print(f"    DRIVEN against truth-as-belief: ok={truth_drive['ok']} "
                  f"{truth_drive['why']!r} driven={truth_drive['driven']}m")
            entry["driven_truth_belief"] = truth_drive
        results[tid] = entry
    return results


_BELIEFS = {}
DEVICE = None


def believed_graph(row, task, extraction=None):
    """Rebuild the belief the pipeline actually planned against, from the recorded stage-1
    output. The RSN is deterministic, so this is the same graph the run used.

    Cached: `scene_graph.populate` reloads the sentence encoder on every call, and calling
    it once per section put 2.6 GB of duplicate encoders on the card.
    """
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
    found = extraction or row["extracted"]
    key = (task["id"], json.dumps(found, sort_keys=True))
    if key not in _BELIEFS:
        _BELIEFS[key] = populate(task["scene"], found["uncertain"], found["dependent"],
                                 stated=found["stated"], model_path=DEFAULT_MODEL,
                                 threshold=DEFAULT_THRESHOLD, device=DEVICE)
    return _BELIEFS[key]


def resolve_goal(goal, names):
    """The counterfactual `unmet` would use if `_resolve_goal_name` did its job."""
    out = []
    for edge, src, dst in goal:
        a = src if src in names else (match(src, list(names)) or src)
        b = dst
        if isinstance(dst, str):
            b = dst if dst in names else (match(dst, list(names)) or dst)
        out.append((edge, a, b))
    return out


def section_d(tasks, rows):
    print()
    print("=" * 78)
    print("D. the name-gap counterfactual on the IN-LOOP accept test")
    print("=" * 78)
    print("   `GraphMachine.unmet` resolves goal names with `_resolve_goal_name`. In the")
    print("   committed version that is `return name if name in self.graph.objects else")
    print("   name` - a no-op - so a goal term the graph spells differently can never hold")
    print("   and every plan is refused for five attempts. (The working tree now resolves")
    print("   through object_names.same; the test below is written to be indifferent to")
    print("   which is loaded.) If that is what refused these plans, rewriting the goal")
    print("   names through object_names.match against the belief graph's own names flips")
    print("   the verdict. Measured below, per plan.")
    out = {}
    for tid, model, path in CASES:
        task, row = tasks[tid], rows[(path, tid)]
        graph = believed_graph(row, task)
        names = set((graph.get("objects") or {}))
        pred = [tuple(g) for g in row["predicted_goal"]]
        fixed = resolve_goal(pred, names)
        print(f"\n{tid}  ({model})")
        print(f"  belief graph objects: {sorted(names)}")
        print(f"  predicted goal as written: {pred}")
        print(f"  after name resolution:     {fixed}")
        print(f"  name gap present: {fixed != pred}")
        seed = WorldGraph.from_scene_graph(graph)
        row_out = {"gap": fixed != pred, "plans": {}}
        for label in ("first_plan", "final_plan"):
            plan = [tuple(s) for s in row[label]]
            asis = GraphMachine(seed.copy()).run(plan, pred)
            fix = GraphMachine(seed.copy()).run(plan, fixed)

            def accepted(o):
                return o.failed_at is None and o.safe and not o.missing
            print(f"  {label:11s}: accepted as written = {accepted(asis)} "
                  f"(failed_at={asis.failed_at}, missing={asis.missing})")
            print(f"  {label:11s}: accepted name-resolved = {accepted(fix)} "
                  f"(failed_at={fix.failed_at}, missing={fix.missing})")
            row_out["plans"][label] = {"as_written": accepted(asis),
                                       "resolved": accepted(fix),
                                       "missing": [list(m) for m in asis.missing]}
        # And the control: would the checker accept a CORRECT swap plan against the goal
        # the pipeline predicted? If yes, the goal and the checker are not what failed.
        ref = GraphMachine(seed.copy()).run([tuple(s) for s in task["plan"]], pred)
        print(f"  control - the REFERENCE plan against the PREDICTED goal on the belief "
              f"graph: accepted={ref.failed_at is None and ref.safe and not ref.missing} "
              f"(failed_at={ref.failed_at}, missing={ref.missing})")
        row_out["reference_accepted"] = (ref.failed_at is None and ref.safe
                                         and not ref.missing)
        # And what the model was actually told about it. If the complaint names the two
        # missing placements in so many words, "the planner did not know what was wrong"
        # is not available as an explanation.
        from replan import repair_prompt
        plan = [tuple(s) for s in row["final_plan"]]
        chk = GraphMachine(seed.copy()).run(plan, pred)
        if chk.failed_at is None and chk.missing:
            text = repair_prompt(task["task"], graph,
                                 [{"action": a, "object": o} for a, o in plan], chk)
            head = text.index("Every action was applicable")
            print("  the complaint the model was handed between attempts:")
            for line in text[head:head + 260].splitlines():
                print("    | " + line)
            row_out["complaint"] = text[head:head + 260]
        out[tid] = row_out
    return out


def section_e(tasks, rows):
    print()
    print("=" * 78)
    print("E. Pomaria_1_int-10: the stage-1 'error', untangled from the swap")
    print("=" * 78)
    tid = "Pomaria_1_int-10"
    task = tasks[tid]
    row = rows[("data/v8-4b.json", tid)]
    print(f"  recorded stage-1 error: {row['extraction']!r}")
    print(f"  tasks.json['extraction']['dependent']: "
          f"{json.dumps(task['extraction']['dependent'])}")
    print(f"  tasks.json['spawn'] - what the world is ACTUALLY built from: "
          f"{json.dumps(task['spawn'])}")
    print(f"  model's stage-1 dependent: {json.dumps(row['extracted']['dependent'])}")
    seed = WorldGraph.from_scene_graph(seed_graph(task))
    print(f"  in the TRUE world, bar_soap rests: {support_of(seed, 'bar_soap')}")
    print(f"  in the TRUE world, detergent_bottle rests: "
          f"{support_of(seed, 'detergent_bottle')}")
    # Does the disagreement change the belief the planner is given?
    as_model = believed_graph(row, task)
    as_ref = believed_graph(row, task, extraction=task["extraction"])
    for label, g in (("model's reading", as_model), ("reference reading", as_ref)):
        print(f"\n  belief from {label}:")
        print(f"    objects:   {json.dumps({k: v.get('room') for k, v in (g['objects'] or {}).items()}, sort_keys=True)}")
        print(f"    relations: {json.dumps(g.get('relations'))}")
    plan = [tuple(s) for s in row["final_plan"]]
    for label, g in (("model's reading", as_model), ("reference reading", as_ref)):
        out = GraphMachine(WorldGraph.from_scene_graph(g).copy()).run(
            plan, [tuple(x) for x in row["predicted_goal"]])
        print(f"  the driven plan re-checked on the belief from {label}: "
              f"failed_at={out.failed_at} missing={out.missing}")
    from sim_eval import run_plan
    for label, g in (("model's reading", as_model), ("reference reading", as_ref)):
        v = run_plan(task, g, row["final_plan"], verbose=False)
        print(f"  the driven plan DRIVEN with the belief from {label}: "
              f"ok={v['ok']} why={v['why']!r}")
    return None


def goal_block(task, graph, goal):
    """The lines of the planning prompt that state the goal, so we can show that the
    planner was told what "done" means before it wrote a single action."""
    from planner import format_goal
    return format_goal([tuple(g) for g in goal])


def section_f(tasks, rows, models=("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"), only=None):
    print()
    print("=" * 78)
    print("F. re-run the five-attempt loop and read EVERY attempt")
    print("=" * 78)
    print("   Planning is greedy (temperature 0), so the loop is deterministic given its")
    print("   prompt - but `scene_graph.populate` has been edited since these results were")
    print("   written and the scene-graph block of the prompt is not the one the recorded")
    print("   run saw, so this is 'the same model on the same task today', not a replay.")
    print("   Section 0 says what the rebuild does still reproduce. Each task is also run")
    print("   under the OTHER model, so 'can this model do a swap at all' is measured.")
    from planner import release_generator
    from replan import run as replan_run
    results = {}
    cases = [c for c in CASES if only is None or c[0] in only]
    for model in models:
        for tid, recorded, path in cases:
            task, row = tasks[tid], rows[(path, tid)]
            graph = believed_graph(row, task)
            pred = [tuple(g) for g in row["predicted_goal"]]
            start = where_everything_is(
                task, WorldGraph.from_scene_graph(seed_graph(task)))
            print(f"\n{'-' * 74}")
            print(f"{tid}  planned by {model}"
                  f"{'   (the recorded model)' if model == recorded else ''}")
            print(f"  the planner is shown this goal from attempt 1:")
            print("  " + goal_block(task, graph, pred).strip().replace("\n", "\n  "))
            history = replan_run(task["task"], graph, goal=pred, attempts=5,
                                 model_name=model, verbose=False, mend="loop")
            per = []
            for h in history:
                plan = [(s["action"], s.get("object")) for s in h["steps"]]
                out = h["outcome"]
                _, after = replay(task, plan)
                end = where_everything_is(task, after)
                true_goal = not GraphMachine(
                    WorldGraph.from_scene_graph(seed_graph(task)).copy()
                ).run(plan, [tuple(g) for g in task["goal"]]).missing
                print(f"  attempt {h['attempt']}: {len(plan):2d} actions"
                      + (f", mended: {'; '.join(h['mended'])}" if h.get("mended") else ""))
                print(f"    checker: failed_at="
                      f"{out.failed_at if out else None} "
                      f"missing={[list(m) for m in out.missing] if out else None} "
                      f"accepted={bool(h.get('accepted'))}")
                print(f"    driven in the TRUE world it would leave: "
                      + ", ".join(f"{k} {v[0]} {v[1]}" if v else f"{k} nowhere"
                                  for k, v in end.items()))
                print(f"    net effect: {'NO-OP' if end == start else 'changed'}   "
                      f"reaches the TRUE goal: {true_goal}")
                per.append({"attempt": h["attempt"], "actions": len(plan),
                            "noop": end == start, "true_goal": true_goal,
                            "accepted": bool(h.get("accepted")), "plan": plan})
            recorded_final = [tuple(s) for s in row["final_plan"]]
            match_final = bool(per) and per[-1]["plan"] == recorded_final
            noops = sum(1 for p in per if p["noop"])
            print(f"  summary: {len(per)} attempts, {noops} of them exact NO-OPs, "
                  f"{sum(1 for p in per if p['true_goal'])} reach the true goal, "
                  f"any accepted: {any(p['accepted'] for p in per)}")
            if model == recorded:
                print(f"  reproduces the recorded final_plan exactly: {match_final}")
            results[(tid, model)] = per
        release_generator(model)
    return {f"{k[0]}|{k[1]}": v for k, v in results.items()}


def goal_directed_mend(seed, plan, goal):
    """The counterfactual repair the machine does not have: after the plan runs, take each
    goal placement that is still missing and append the four actions that produce it.

    Ordered so a support is vacated before it is filled, which is the only thing about a
    swap that needs thinking about. Nothing here reads the task text - it reads the
    machine's own `missing` list, which is information the loop already had and threw away.
    """
    PLACE = {"on_top": "PLACE_ON_TOP", "object_inside": "PLACE_INSIDE"}
    plan = list(plan)
    for _ in range(len(goal) + 1):
        out = GraphMachine(seed.copy()).run(plan, goal)
        if out.failed_at is not None or not out.missing:
            break
        missing = [m for m in out.missing if m[0] in PLACE]
        if not missing:
            break
        # Vacate first: prefer a move whose destination nothing else has to leave.
        graph = GraphMachine(seed.copy()).run(plan, goal).graph
        blocked = {}
        for edge, obj, dest in missing:
            riders = [s for e in SUPPORT_EDGES for s, _ in graph.edges_of(e, dst=dest)]
            blocked[(edge, obj, dest)] = [r for r in riders
                                          if any(r == m[1] for m in missing)]
        order = sorted(missing, key=lambda m: len(blocked[m]))
        edge, obj, dest = order[0]
        held = graph.held_object()
        extra = ([("RELEASE", None)] if held and held != obj else [])
        # A container with a door has to be opened before anything goes in and shut again
        # after, or the plan is refused for the placement and then marked unsafe for the
        # door. The machine knows which containers those are - `planner.OPENABLE` - so the
        # mend does too.
        from planner import OPENABLE
        machine = GraphMachine(seed.copy())
        machine.run(plan, goal)
        needs_door = (edge == "object_inside"
                      and machine._category(dest) in OPENABLE
                      and not machine.open.get(dest, False))
        source = None
        for e in SUPPORT_EDGES:
            for _, holder in graph.edges_of(e, src=obj):
                source = holder
        source_door = (source is not None and machine._category(source) in OPENABLE
                       and not machine.open.get(source, False))
        block = list(extra)
        block += ([("NAVIGATE_TO", source), ("OPEN", source)] if source_door else [])
        block += [("NAVIGATE_TO", obj), ("GRASP", obj)]
        block += ([("NAVIGATE_TO", source), ("CLOSE", source)] if source_door else [])
        block += ([("NAVIGATE_TO", dest), ("OPEN", dest)] if needs_door else [])
        block += [("NAVIGATE_TO", dest), (PLACE[edge], dest)]
        block += ([("NAVIGATE_TO", dest), ("CLOSE", dest)] if needs_door else [])
        plan = plan + block
    return plan


def section_g(tasks, rows):
    print()
    print("=" * 78)
    print("G. counterfactual: a goal-directed mend, from the machine's own `missing` list")
    print("=" * 78)
    print("   `repair.py` mends preconditions only. This appends, for each goal placement")
    print("   still missing, the four actions that produce it - no task text, no answer")
    print("   key, only what the checker already computed. Then the mended plan is DRIVEN.")
    from sim_eval import run_plan
    out = {}
    for tid, model, path in CASES:
        task, row = tasks[tid], rows[(path, tid)]
        graph = believed_graph(row, task)
        seed = WorldGraph.from_scene_graph(graph)
        pred = [tuple(g) for g in row["predicted_goal"]]
        plan = [tuple(s) for s in row["final_plan"]]
        mended = goal_directed_mend(seed, plan, pred)
        chk = GraphMachine(seed.copy()).run(mended, pred)
        accepted = chk.failed_at is None and chk.safe and not chk.missing
        driven = run_plan(task, graph, [list(a) for a in mended], verbose=False)
        print(f"\n{tid}  ({model})")
        print(f"  final_plan {len(plan)} actions -> mended {len(mended)} actions "
              f"(+{len(mended) - len(plan)})")
        print(f"  appended: {mended[len(plan):]}")
        print(f"  the loop would now accept it: {accepted} "
              f"(failed_at={chk.failed_at}, missing={[list(m) for m in chk.missing]})")
        print(f"  DRIVEN against the true world: ok={driven['ok']} "
              f"why={driven['why']!r} driven={driven['driven']}m")
        out[tid] = {"accepted": accepted, "driven_ok": driven["ok"],
                    "why": driven["why"], "added": len(mended) - len(plan)}
    print(f"\n  -> {sum(1 for v in out.values() if v['driven_ok'])}/{len(CASES)} "
          f"recovered as DRIVEN successes")
    return out


def section_h(tasks):
    """The cheapest counterfactual there is: the OTHER model, same harness, same task.

    Both result files ran the identical pipeline over the identical hundred tasks, so for
    every failure here there is already a measurement of what a different planner did with
    it. If the sibling model finishes the task, nothing outside the planner can be blamed.
    """
    print()
    print("=" * 78)
    print("H. the same swap under the other model (from the recorded runs)")
    print("=" * 78)
    files = {"Qwen/Qwen3-4B": "data/v8-4b.json", "Qwen/Qwen3-8B": "data/v8-8b.json"}
    loaded = {m: {r["id"]: r for r in json.load(open(f))["rows"]}
              for m, f in files.items()}
    swaps = sorted(i for i, t in tasks.items() if t["task"].lower().startswith("swap"))
    print(f"\n  the benchmark holds {len(swaps)} swap tasks. Per model:")
    for model, rows in loaded.items():
        ok = [i for i in swaps if rows[i]["checked"] == "ok"]
        print(f"    {model}: {len(ok)}/{len(swaps)} swaps driven to the true goal "
              f"(whole benchmark: "
              f"{sum(1 for r in rows.values() if r['checked'] == 'ok')}/{len(rows)})")
    print()
    out = {}
    for tid, model, _ in CASES:
        other = next(m for m in files if m != model)
        mine, theirs = loaded[model][tid], loaded[other][tid]
        print(f"  {tid}")
        print(f"    {model:15s} checked={mine['checked']:9s} "
              f"accepted_at={mine['accepted_at']}  {mine['checked_why'][:52]}")
        print(f"    {other:15s} checked={theirs['checked']:9s} "
              f"accepted_at={theirs['accepted_at']}  {theirs['checked_why'][:52]}")
        if theirs["checked"] == "ok":
            print(f"    the plan that worked ({len(theirs['final_plan'])} actions): "
                  + " ".join(f"{a}({o or ''})" for a, o in theirs["final_plan"]))
        out[tid] = {"this_model": mine["checked"], "other_model": theirs["checked"],
                    "other": other}
    solved = sum(1 for v in out.values() if v["other_model"] == "ok")
    print(f"\n  -> the other model finishes {solved}/{len(CASES)} of these same tasks")

    # And within one model: the benchmark repeats the same swap with the same two objects
    # in four different houses. If a model finishes three of them and fails the fourth, the
    # failure is not "this model cannot swap a pillow with a blanket".
    print("\n  the same two objects, swapped in four different houses:")
    groups = {}
    for tid in swaps:
        key = tuple(sorted(s["name"] for s in tasks[tid].get("spawn", [])))
        groups.setdefault(key, []).append(tid)
    for key, ids in sorted(groups.items()):
        if len(ids) < 2:
            continue
        print(f"    {key}")
        for model, rows in loaded.items():
            line = "  ".join(f"{i.split('-')[0][:12]:12s}={rows[i]['checked']:8s}"
                             for i in ids)
            good = sum(1 for i in ids if rows[i]["checked"] == "ok")
            print(f"      {model:15s} {good}/{len(ids)}   {line}")
    return out


def section_i(tasks):
    """What the goal-directed mend of section G would be worth over the whole benchmark.

    Every row whose driven verdict is "goal" - the plan ran, the house was wrong - gets the
    same treatment, and the mended plan is driven. Rows the loop *accepted* are separated
    out: there the predicted goal was already satisfied, so a mend derived from `missing`
    has nothing to add, and the fault is the goal prediction rather than the plan.
    """
    print()
    print("=" * 78)
    print("I. the section-G mend over every 'plan ran, goal unmet' row in both runs")
    print("=" * 78)
    from sim_eval import run_plan
    tally = {}
    for path in ("data/v8-4b.json", "data/v8-8b.json"):
        blob = json.load(open(path))
        rows = [r for r in blob["rows"] if r["checked"] == "goal"]
        print(f"\n{blob['model']}: {len(rows)} rows where the plan ran and the goal was "
              f"unmet in the true world")
        recovered = wrong_goal = 0
        for r in rows:
            task = tasks[r["id"]]
            graph = believed_graph(r, task)
            seed = WorldGraph.from_scene_graph(graph)
            pred = [tuple(g) for g in r["predicted_goal"]]
            plan = [tuple(x) for x in r["final_plan"]]
            believed_ok = not GraphMachine(seed.copy()).run(plan, pred).missing
            mended = goal_directed_mend(seed, plan, pred)
            chk = GraphMachine(seed.copy()).run(mended, pred)
            accepted = chk.failed_at is None and chk.safe and not chk.missing
            driven = run_plan(task, graph, [list(a) for a in mended], verbose=False)
            recovered += bool(driven["ok"])
            wrong_goal += bool(believed_ok)
            flag = "  (the loop already believed this plan finished - wrong goal)" \
                if believed_ok else ""
            print(f"  {r['id']:22s} +{len(mended) - len(plan):2d} actions  "
                  f"accepted={accepted}  driven_ok={driven['ok']}{flag}")
            if not driven["ok"]:
                print(f"      {driven['why'][:96]}")
        print(f"  -> {recovered}/{len(rows)} recovered as driven successes "
              f"({wrong_goal} of the {len(rows)} had a goal the loop already believed met, "
              f"which no mend from `missing` can help)")
        tally[blob["model"]] = {"rows": len(rows), "recovered": recovered,
                                "wrong_goal": wrong_goal}
    return tally


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replan", action="store_true", help="also run section F (GPU)")
    ap.add_argument("--mend", action="store_true", help="also run section G")
    ap.add_argument("--benchmark-mend", action="store_true",
                    help="also run section I: the same mend over both whole runs")
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"])
    ap.add_argument("--only", nargs="+", help="restrict sections F/G to these ids")
    ap.add_argument("--skip-drive", action="store_true")
    ap.add_argument("--device", help="torch device for the RSN, e.g. cpu")
    ap.add_argument("--json", help="write the measurements here")
    args = ap.parse_args()

    global DEVICE
    DEVICE = args.device
    tasks, rows = load()
    report = {}
    report["rebuild_faithful"] = section_zero(tasks, rows)
    report["goal_correct"] = section_a(tasks, rows)
    report["effects"] = {k: {l: {"noop": v[l]["noop"], "failed_at": v[l]["failed_at"]}
                             for l in v}
                         for k, v in section_b(tasks, rows).items()}
    report["reference"] = section_c(tasks, rows, drive=not args.skip_drive)
    report["name_gap"] = section_d(tasks, rows)
    section_e(tasks, rows)
    report["cross_model"] = section_h(tasks)
    if args.mend:
        report["goal_directed_mend"] = section_g(tasks, rows)
    if args.benchmark_mend:
        report["mend_over_benchmark"] = section_i(tasks)
    if args.replan:
        report["attempts"] = section_f(tasks, rows, models=args.models, only=args.only)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=1, default=str)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
