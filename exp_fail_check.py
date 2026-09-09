#!/usr/bin/env python3
"""Independent re-check of the 20 single-task failures in data/v8-4b.json / v8-8b.json.

Nothing here trusts an earlier agent's attribution. Every counterfactual is run.

Two fidelity choices, both deliberate:

  * The belief graph is rebuilt with the HEAD copy of `scene_graph.populate` (loaded out
    of git, not out of the working tree), because the working tree's `populate` was
    changed after the v8 runs - it now expands every room instance instead of the largest
    per type, so a rebuild with it is a different belief from the one the run had.
  * `GraphMachine._resolve_goal_name` is monkeypatched back to the HEAD no-op for every
    "as it ran" check, because the working tree already carries another agent's
    object_names.same() fix. `--resolver same` restores it, which is the names
    counterfactual.

Sections are selected on the command line; run with no arguments for the cheap ones.
"""

import json
import os
import subprocess
import sys
import types

REPO = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.environ.get("SCRATCH", "/tmp/exp_fail_check")
sys.path.insert(0, REPO)

import graph_machine
from graph_machine import GraphMachine
from world_graph import WorldGraph
from object_names import same, match
import repair as repair_mod
import planner


# ---------------------------------------------------------------- HEAD modules

def head_module(name):
    """Import `name` as it stood at HEAD, without touching the working tree."""
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, f"{name}_head.py")
    if not os.path.exists(path):
        src = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{name}.py"],
                             capture_output=True, text=True, check=True).stdout
        open(path, "w").write(src)
    mod = types.ModuleType(f"{name}_head")
    mod.__file__ = path
    sys.modules[f"{name}_head"] = mod
    exec(compile(open(path).read(), path, "exec"), mod.__dict__)
    return mod


STRICT = lambda self, name: name if name in self.graph.objects else name


def SAME(self, name):
    if name in self.graph.objects:
        return name
    hits = [n for n in self.graph.objects if same(name, n)]
    return hits[0] if len(hits) == 1 else name


def set_resolver(kind):
    graph_machine.GraphMachine._resolve_goal_name = STRICT if kind == "strict" else SAME


# ---------------------------------------------------------------- data

def load():
    tasks = {t["id"]: t for t in json.load(open(os.path.join(REPO, "data/tasks.json")))}
    runs = {}
    for arm in ("4b", "8b"):
        runs[arm] = json.load(open(os.path.join(REPO, f"data/v8-{arm}.json")))["rows"]
    return tasks, runs


def failures(rows):
    return [r for r in rows if r["checked"] != "ok"]


_BELIEF = {}


def belief(row, head=True, drop_dependent=False):
    """The belief graph for a row, rebuilt from the row's own recorded extraction."""
    key = (row["id"], row["scene"], json.dumps(row["extracted"], sort_keys=True),
           head, drop_dependent)
    if key in _BELIEF:
        return _BELIEF[key]
    mod = head_module("scene_graph") if head else __import__("scene_graph")
    ex = row["extracted"]
    dep = [] if drop_dependent else ex["dependent"]
    uncertain = list(ex["uncertain"])
    if drop_dependent:
        # keep the objects, lose only the invented support relation
        uncertain += [d["object"] for d in ex["dependent"]
                      if d["object"] not in uncertain and d["object"] not in ex["stated"]]
    g = mod.populate(row["scene"], uncertain, dep, stated=ex["stated"],
                     model_path=mod.DEFAULT_MODEL, threshold=mod.DEFAULT_THRESHOLD)
    _BELIEF[key] = g
    return g


def plan_of(row, which):
    return [(a, o) for a, o in (row[which] or [])]


def goal_of(row):
    return [tuple(g) for g in (row["predicted_goal"] or ())]


def check(row, plan, goal=None, graph=None):
    g = graph if graph is not None else belief(row)
    seed = WorldGraph.from_scene_graph(g)
    out = GraphMachine(seed.copy()).run(list(plan), goal if goal is not None else goal_of(row))
    accepted = out.failed_at is None and out.safe and (not goal_of(row) or out.goal_met)
    return out, accepted


def drive(row, tasks, plan, graph=None):
    from sim_eval import run_plan
    g = graph if graph is not None else belief(row)
    try:
        return run_plan(tasks[row["id"]], g, [(a, o) for a, o in plan], verbose=False)
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}", "error": True}


# ================================================================ A. name gap
def section_a(tasks, runs):
    """Which predicted-goal terms miss the belief graph under ==, and does same() catch them?"""
    print("\n=== A. goal-name gap over all 200 rows ===")
    total = {}
    for arm, rows in runs.items():
        gaps = []
        for row in rows:
            g = belief(row)
            names = set(g["objects"])
            for rel, src, dst in goal_of(row):
                for term in (src, dst):
                    if not isinstance(term, str) or term in names:
                        continue
                    hits = [n for n in names if same(term, n)]
                    gaps.append((row["id"], row["checked"], term, hits))
        total[arm] = gaps
        print(f"  {arm}: {len(gaps)} goal terms miss ==")
        for gid, checked, term, hits in gaps:
            print(f"     {gid:24s} checked={checked:9s} '{term}' -> same() {hits}")
    return total


# ================================================================ B. reproduce
def section_b(tasks, runs, do_drive=False):
    """Does today's rebuild reproduce the recorded verdict for each failure row?"""
    print("\n=== B. reproduction of the 20 recorded failures (strict resolver) ===")
    for arm, rows in runs.items():
        for row in failures(rows):
            out, acc = check(row, plan_of(row, "final_plan"))
            fa = out.failed_at
            reason = out.steps[fa].reason if fa is not None else ""
            print(f"  {arm} {row['id']:22s} recorded={row['checked']:9s} "
                  f"acc@{row['accepted_at']}  machine(final): failed_at={fa} "
                  f"missing={[list(m) for m in out.missing]} safe={out.safe}")
            if fa is not None:
                print(f"      refusal: step {fa+1} {reason[:100]}")
            if do_drive:
                v = drive(row, tasks, plan_of(row, "final_plan"))
                rec = row["simulated"] or {}
                agree = (v.get("ok") == rec.get("ok"))
                print(f"      driven now: ok={v.get('ok')} why={str(v.get('why'))[:80]!r} "
                      f"{v.get('driven')}m | recorded ok={rec.get('ok')} {rec.get('driven')}m "
                      f"| verdict reproduces: {agree}")


# ================================================================ C. repair variants
def repair_with(rule):
    """A copy of repair._edit with one extra rule spliced in, and repair() rebound to it."""
    orig_edit = repair_mod._edit

    def edit(graph, plan, index, kind, subject, goal=()):
        out = orig_edit(graph, plan, index, kind, subject, goal)
        if out is not None:
            return out
        return rule(graph, plan, index, kind, subject, goal)
    return edit


def delete_stranded_place(graph, plan, index, kind, subject, goal=()):
    """emptyhand's proposed rule: drop a PLACE that has nothing left to place."""
    action, arg = plan[index]
    if kind != "empty_hand" or action not in repair_mod.PLACES:
        return None
    want = repair_mod.PLACES[action]
    for relation, moved, destination in goal:
        if relation != want or not repair_mod._same(destination, arg):
            continue
        if moved in graph.objects and not graph.has_edge(want, moved, arg):
            return None                      # the original rule already fired
    return plan[:index] + plan[index + 1:]


def goal_directed_mend(seed, plan, goal, machine=None, compose_repair=True):
    """swaps' proposed post-pass: append the actions the checker's own `missing` implies."""
    current = list(plan)
    added, start = 0, len(plan)
    for _ in range(8):
        if compose_repair:
            # the appended tail can itself need mending - the plan may end with a full
            # hand, and repair's `holding` rule is what discharges it
            current, _n = run_repair(seed, current, goal)
        out = GraphMachine(seed.copy()).run(current, goal)
        if out.failed_at is not None:
            break
        placements = [(r, m, d) for r, m, d in out.missing if r in repair_mod.PLACE_FOR]
        if not placements:
            break
        rel, moved, dest = placements[0]
        verb = repair_mod.PLACE_FOR[rel]
        tail = [("NAVIGATE_TO", moved), ("GRASP", moved)]
        openable = any(same(dest, o) for o in planner.OPENABLE)
        if openable:
            tail += [("NAVIGATE_TO", dest), ("OPEN", dest)]
        tail += [("NAVIGATE_TO", dest), (verb, dest)]
        if openable:
            tail += [("NAVIGATE_TO", dest), ("CLOSE", dest)]
        probe = current + tail
        if compose_repair:
            probe, _n = run_repair(seed, probe, goal)
        out2 = GraphMachine(seed.copy()).run(probe, goal)
        if out2.failed_at is not None or len(out2.missing) >= len(out.missing):
            break
        current = probe
        added = len(current) - start
    return current, added


def run_repair(seed, plan, goal, edit=None):
    """repair.repair with an optionally patched _edit, restored afterwards."""
    old = repair_mod._edit
    if edit is not None:
        repair_mod._edit = edit
    try:
        return repair_mod.repair(seed, list(plan), goal)
    finally:
        repair_mod._edit = old


def accept(seed, plan, goal):
    out = GraphMachine(seed.copy()).run(list(plan), goal)
    return out, (out.failed_at is None and out.safe and (not goal or out.goal_met))


# ============================================== C. the empty_hand delete rule
def section_c(tasks, runs, do_drive=False, scope="fail"):
    print("\n=== C. emptyhand's guarded-delete rule, on each row's RAW attempt-1 plan ===")
    edit = repair_with(delete_stranded_place)
    flips = {"4b": [], "8b": []}
    for arm, rows in runs.items():
        pool = failures(rows) if scope == "fail" else rows
        for row in pool:
            plan, goal = plan_of(row, "first_plan"), goal_of(row)
            if not plan:
                continue
            seed = WorldGraph.from_scene_graph(belief(row))
            base_plan, base_notes = run_repair(seed, plan, goal)
            _, base_acc = accept(seed, base_plan, goal)
            new_plan, new_notes = run_repair(seed, plan, goal, edit=edit)
            _, new_acc = accept(seed, new_plan, goal)
            if base_acc == new_acc and base_plan == new_plan:
                continue
            print(f"  {arm} {row['id']:22s} checked={row['checked']:9s} "
                  f"base_accept={base_acc} new_accept={new_acc}")
            if new_acc and not base_acc:
                print(f"      notes: {new_notes}")
                flips[arm].append(row["id"])
                if do_drive:
                    v = drive(row, tasks, new_plan)
                    print(f"      DRIVEN: ok={v.get('ok')} why={str(v.get('why'))[:90]!r} "
                          f"{v.get('driven')}m")
    print(f"  flips to accepted: {json.dumps(flips)}")
    return flips


# ============================================== D. the goal-directed mend
def section_d(tasks, runs, do_drive=False, scope="fail"):
    print("\n=== D. swaps' goal-directed mend, applied after normal repair ===")
    got = {"4b": [], "8b": []}
    for arm, rows in runs.items():
        pool = failures(rows) if scope == "fail" else rows
        for row in pool:
            goal = goal_of(row)
            if not goal:
                continue
            seed = WorldGraph.from_scene_graph(belief(row))
            best = None
            for which in ("final_plan", "first_plan"):
                plan = plan_of(row, which)
                if not plan:
                    continue
                mended, _ = run_repair(seed, plan, goal)
                out, acc = accept(seed, mended, goal)
                if acc:
                    continue
                grown, added = goal_directed_mend(seed, mended, goal)
                if added == 0:
                    continue
                out2, acc2 = accept(seed, grown, goal)
                if acc2:
                    best = (which, grown, added)
                    break
            if best is None:
                continue
            which, grown, added = best
            print(f"  {arm} {row['id']:22s} checked={row['checked']:9s} "
                  f"{which} +{added} actions -> ACCEPTED")
            got[arm].append(row["id"])
            if do_drive:
                v = drive(row, tasks, grown)
                print(f"      DRIVEN: ok={v.get('ok')} why={str(v.get('why'))[:90]!r} "
                      f"{v.get('driven')}m")
    print(f"  recovered: {json.dumps(got)}")
    return got


# ============================================== E. the reach / invented relation
def section_e(tasks, runs, do_drive=False):
    print("\n=== E. reach rows: invented support relation, and the B2c counterfactual ===")
    ids = ("Beechwood_0_int-08", "Wainscott_0_int-07", "Ihlen_1_int-07")
    for arm, rows in runs.items():
        for row in rows:
            if row["id"] not in ids or row["checked"] == "ok":
                continue
            truth = tasks[row["id"]].get("extraction") or {}
            print(f"\n  {arm} {row['id']} checked={row['checked']} acc@{row['accepted_at']}")
            print(f"      heard dependent : {row['extracted']['dependent']}")
            print(f"      truth dependent : {truth.get('dependent')}")
            print(f"      truth stated    : {truth.get('stated')}")
            g_heard = belief(row)
            g_strip = belief(row, drop_dependent=True)
            print(f"      heard rooms : { {k: v.get('room') for k, v in g_heard['objects'].items()} }")
            print(f"      strip rooms : { {k: v.get('room') for k, v in g_strip['objects'].items()} }")
            # is the goal already satisfied in the heard belief?
            for tag, g in (("heard", g_heard), ("stripped", g_strip)):
                seed = WorldGraph.from_scene_graph(g)
                out = GraphMachine(seed.copy()).run([], [tuple(x) for x in tasks[row["id"]]["goal"]])
                pre = len(tasks[row["id"]]["goal"]) - len(out.missing)
                print(f"      {tag:9s}: goal edges already true before any action: "
                      f"{pre}/{len(tasks[row['id']]['goal'])}")
            # B2c: strip the relation, keep the wrong room, re-repair the model's own plan
            for which in ("first_plan", "final_plan"):
                plan = plan_of(row, which)
                if not plan:
                    continue
                for tag, g in (("heard", g_heard), ("stripped", g_strip)):
                    seed = WorldGraph.from_scene_graph(g)
                    mended, notes = run_repair(seed, plan, goal_of(row))
                    out, acc = accept(seed, mended, goal_of(row))
                    line = (f"      {which:11s} on {tag:9s}: accept={acc} "
                            f"failed_at={out.failed_at} notes={notes}")
                    print(line)
                    if acc and do_drive and tag == "stripped":
                        v = drive(row, tasks, mended, graph=g)
                        print(f"          DRIVEN: ok={v.get('ok')} "
                              f"why={str(v.get('why'))[:90]!r} {v.get('driven')}m")


# ============================================== F. Wainscott_0_int-02
def section_f(tasks, runs, do_drive=False):
    print("\n=== F. 4B Wainscott_0_int-02: closed fridge, and repair._rank ===")
    row = next(r for r in runs["4b"] if r["id"] == "Wainscott_0_int-02")
    g = belief(row)
    print(f"      belief swiss_cheese: {g['objects'].get('swiss_cheese')}")
    print(f"      relations: {g.get('relations')}")
    print(f"      extraction error recorded: {row['extraction']}")
    seed = WorldGraph.from_scene_graph(g)
    for which in ("first_plan", "final_plan"):
        plan = plan_of(row, which)
        out, acc = accept(seed, plan, goal_of(row))
        fa = out.failed_at
        print(f"      {which}: len={len(plan)} failed_at={fa} "
              f"fault={out.steps[fa].fault if fa is not None else None}")
        if fa is not None:
            print(f"        reason: {out.steps[fa].reason[:120]}")
        mended, notes = run_repair(seed, plan, goal_of(row))
        print(f"        repair as-is: notes={notes}")
        # progress-above-safety rank
        old_rank = repair_mod._rank
        repair_mod._rank = lambda outcome, length: (
            outcome.failed_at is None, outcome.goal_met,
            outcome.failed_at if outcome.failed_at is not None else len(outcome.steps),
            outcome.safe, -length)
        try:
            mended2, notes2 = repair_mod.repair(seed, list(plan), goal_of(row))
        finally:
            repair_mod._rank = old_rank
        print(f"        repair progress-first: notes={notes2} len={len(mended2)}")
        if do_drive and notes2:
            v = drive(row, tasks, mended2)
            print(f"        DRIVEN progress-first: ok={v.get('ok')} "
                  f"why={str(v.get('why'))[:110]!r} {v.get('driven')}m steps={v.get('steps_run')}")


# ============================================== G. names counterfactual
def section_g(tasks, runs, do_drive=False):
    print("\n=== G. names: the resolver counterfactual on every row that carries a gap ===")
    for arm, rows in runs.items():
        for row in rows:
            g = belief(row)
            names = set(g["objects"])
            gap = [t for _, s, d in goal_of(row) for t in (s, d)
                   if isinstance(t, str) and t not in names and [n for n in names if same(t, n)]]
            if not gap:
                continue
            seed = WorldGraph.from_scene_graph(g)
            print(f"\n  {arm} {row['id']} checked={row['checked']} acc@{row['accepted_at']} gap={gap}")
            for which in ("first_plan", "final_plan"):
                plan = plan_of(row, which)
                if not plan:
                    continue
                for kind in ("strict", "same"):
                    set_resolver(kind)
                    mended, notes = run_repair(seed, plan, goal_of(row))
                    out, acc = accept(seed, mended, goal_of(row))
                    print(f"      {which:11s} resolver={kind:6s}: accept={acc} "
                          f"failed_at={out.failed_at} missing={[list(m) for m in out.missing]}")
                set_resolver("strict")
            if do_drive:
                v = drive(row, tasks, plan_of(row, "first_plan"))
                print(f"      first_plan DRIVEN: ok={v.get('ok')} "
                      f"why={str(v.get('why'))[:90]!r} {v.get('driven')}m "
                      f"(recorded sim_first ok={(row['sim_first'] or {}).get('ok')} "
                      f"{(row['sim_first'] or {}).get('driven')}m)")


def main():
    args = set(sys.argv[1:])
    do_drive = "--drive" in args
    scope = "all" if "--all" in args else "fail"
    set_resolver("strict")
    tasks, runs = load()
    if "a" in args:
        section_a(tasks, runs)
    if "b" in args:
        section_b(tasks, runs, do_drive)
    if "c" in args:
        section_c(tasks, runs, do_drive, scope)
    if "d" in args:
        section_d(tasks, runs, do_drive, scope)
    if "e" in args:
        section_e(tasks, runs, do_drive)
    if "f" in args:
        section_f(tasks, runs, do_drive)
    if "g" in args:
        section_g(tasks, runs, do_drive)
    if "h" in args:
        section_h(tasks, runs, do_drive)
    if "i" in args:
        section_i(tasks, runs, do_drive)
    if "j" in args:
        section_j(tasks, runs, do_drive)
    if "k" in args:
        section_k(tasks, runs, do_drive)
    if "l" in args:
        section_l(tasks, runs, do_drive)
    if "m" in args:
        section_m(tasks, runs, do_drive)




# ============================================== H. what the names fix actually drives
def section_h(tasks, runs, do_drive=True):
    """The fix accepts attempt 1 - so attempt 1's MENDED plan is what gets driven.

    Section G drove the raw first_plan, which is not what the loop would hand the
    simulator. This drives exactly what `replan.run` would have accepted.
    """
    print("\n=== H. names fix: drive the plan the fixed loop would actually accept ===")
    for arm, rows in runs.items():
        for row in rows:
            g = belief(row)
            names = set(g["objects"])
            gap = [t for _, s, d in goal_of(row) for t in (s, d)
                   if isinstance(t, str) and t not in names and [n for n in names if same(t, n)]]
            if not gap:
                continue
            seed = WorldGraph.from_scene_graph(g)
            set_resolver("same")
            mended, notes = run_repair(seed, plan_of(row, "first_plan"), goal_of(row))
            out, acc = accept(seed, mended, goal_of(row))
            set_resolver("strict")
            v = drive(row, tasks, mended) if (acc and do_drive) else None
            print(f"  {arm} {row['id']:20s} recorded checked={row['checked']:4s} "
                  f"({(row['simulated'] or {}).get('driven')}m)")
            print(f"      fixed loop accepts attempt 1: {acc}  mended={notes} "
                  f"len={len(mended)} (raw first_plan len={len(plan_of(row,'first_plan'))})")
            if v:
                print(f"      that accepted plan DRIVEN: ok={v.get('ok')} "
                      f"why={str(v.get('why'))[:100]!r} {v.get('driven')}m")
                print(f"      => task after fix: {'ok' if v.get('ok') else 'FAIL'} "
                      f"(was {row['checked']})")




# ============================================== I. mend disagreements
def section_i(tasks, runs, do_drive=False):
    print("\n=== I. why the goal-directed mend does or does not fire on the disputed rows ===")
    probes = [("4b", "Pomaria_0_int-04"), ("8b", "Pomaria_0_int-10"),
              ("4b", "Wainscott_1_int-01"), ("4b", "Wainscott_1_int-02"),
              ("8b", "Pomaria_1_int-03")]
    for arm, rid in probes:
        row = next(r for r in runs[arm] if r["id"] == rid)
        seed = WorldGraph.from_scene_graph(belief(row))
        goal = goal_of(row)
        for kind in ("strict", "same"):
            set_resolver(kind)
            for which in ("first_plan", "final_plan"):
                plan = plan_of(row, which)
                if not plan:
                    continue
                mended, _ = run_repair(seed, plan, goal)
                out, acc = accept(seed, mended, goal)
                grown, added = goal_directed_mend(seed, mended, goal)
                out2, acc2 = accept(seed, grown, goal)
                print(f"  {arm} {rid} {which:11s} resolver={kind:6s}: "
                      f"base acc={acc} missing={[list(m) for m in out.missing]} "
                      f"-> +{added} acc={acc2} missing={[list(m) for m in out2.missing]} "
                      f"failed_at={out2.failed_at}")
                if acc2 and do_drive:
                    v = drive(row, tasks, grown)
                    print(f"      DRIVEN: ok={v.get('ok')} why={str(v.get('why'))[:90]!r} "
                          f"{v.get('driven')}m")
        set_resolver("strict")




# ============================================== J. all harness fixes together
def section_j(tasks, runs, do_drive=True):
    """Every proposed harness fix at once, on each failing row's own attempt-1 plan.

    Fixes stacked: (1) same()-resolver for goal names, (2) the guarded empty_hand delete,
    (3) the goal-directed mend, (4) the reach guard, modelled as stripping the unverified
    `dependent` relation out of the belief, (5) progress-above-safety in repair._rank.
    """
    print("\n=== J. union of all four harness fixes, per failing row ===")
    edit = repair_with(delete_stranded_place)
    old_rank = repair_mod._rank
    progress_first = lambda o, n: (o.failed_at is None, o.goal_met,
                                   o.failed_at if o.failed_at is not None else len(o.steps),
                                   o.safe, -n)
    tally = {"4b": [], "8b": []}
    for arm, rows in runs.items():
        for row in failures(rows):
            goal = goal_of(row)
            best = None
            for guard in (False, True):          # reach guard off / on
                g = belief(row, drop_dependent=guard)
                seed = WorldGraph.from_scene_graph(g)
                for which in ("first_plan", "final_plan"):
                    plan = plan_of(row, which)
                    if not plan:
                        continue
                    set_resolver("same")
                    repair_mod._rank = progress_first
                    try:
                        mended, notes = run_repair(seed, plan, goal, edit=edit)
                        out, acc = accept(seed, mended, goal)
                        if not acc:
                            mended, _n = goal_directed_mend(seed, mended, goal)
                            out, acc = accept(seed, mended, goal)
                    finally:
                        repair_mod._rank = old_rank
                        set_resolver("strict")
                    if acc:
                        best = (guard, which, mended)
                        break
                if best:
                    break
            if best is None:
                print(f"  {arm} {row['id']:22s} still refused by every plan on file")
                continue
            guard, which, mended = best
            v = drive(row, tasks, mended,
                      graph=belief(row, drop_dependent=guard)) if do_drive else None
            ok = bool(v and v.get("ok"))
            tally[arm].append((row["id"], ok, guard, which))
            print(f"  {arm} {row['id']:22s} accepted via {which} "
                  f"(reach-guard={guard}) -> DRIVEN ok={ok} "
                  f"why={str((v or {}).get('why'))[:70]!r} {(v or {}).get('driven')}m")
    for arm in tally:
        won = [t for t in tally[arm] if t[1]]
        print(f"  {arm}: {len(won)} of {len(failures(runs[arm]))} failures recovered: "
              f"{[t[0] for t in won]}")




# ============================================== K. reach guard as a checker rule
def reach_guard(seed_relations):
    """`_within_reach_of` that refuses to walk an unverified belief edge.

    The edge stays in the belief - the searcher still uses it as a hint - but standing at
    a support no longer grants reach to something the pipeline only *believes* is on it.
    Edges the plan itself created are still walked, because those the machine watched happen.
    """
    unverified = set(seed_relations)

    def within(self, name):
        moving, stack = {name}, [name]
        while stack:
            here = stack.pop()
            for edge_type in ("on_top", "object_inside"):
                for rider, _ in self.graph.edges_of(edge_type, dst=here):
                    if (edge_type, rider, here) in unverified or rider in moving:
                        continue
                    moving.add(rider)
                    stack.append(rider)
        return moving
    return within


def seed_edges_of(g):
    out = set()
    for rel in (g.get("relations") or []):
        edge = ("object_inside" if str(rel.get("relation", "")).upper() == "INSIDE"
                else "on_top")
        out.add((edge, rel["from"], rel["to"]))
    return out


def section_k(tasks, runs, do_drive=True):
    print("\n=== K. union of the four harness fixes, reach guard as a CHECKER rule ===")
    edit = repair_with(delete_stranded_place)
    old_rank, old_reach = repair_mod._rank, GraphMachine._within_reach_of
    old_carry = GraphMachine._carried_with
    progress_first = lambda o, n: (o.failed_at is None, o.goal_met,
                                   o.failed_at if o.failed_at is not None else len(o.steps),
                                   o.safe, -n)
    tally = {"4b": [], "8b": []}
    for arm, rows in runs.items():
        for row in failures(rows):
            goal, g = goal_of(row), belief(row)
            seed = WorldGraph.from_scene_graph(g)
            found = None
            for which in ("first_plan", "final_plan"):
                plan = plan_of(row, which)
                if not plan:
                    continue
                set_resolver("same")
                repair_mod._rank = progress_first
                GraphMachine._within_reach_of = reach_guard(seed_edges_of(g))
                if "--carry" in sys.argv:
                    GraphMachine._carried_with = reach_guard(seed_edges_of(g))
                try:
                    mended, notes = run_repair(seed, plan, goal, edit=edit)
                    out, acc = accept(seed, mended, goal)
                    if not acc:
                        mended, _n = goal_directed_mend(seed, mended, goal)
                        out, acc = accept(seed, mended, goal)
                finally:
                    repair_mod._rank = old_rank
                    GraphMachine._within_reach_of = old_reach
                    GraphMachine._carried_with = old_carry
                    set_resolver("strict")
                if acc:
                    found = (which, mended, notes)
                    break
            if found is None:
                print(f"  {arm} {row['id']:22s} REFUSED on every plan on file "
                      f"(the loop would go back to the model)")
                tally[arm].append((row["id"], False, None))
                continue
            which, mended, notes = found
            v = drive(row, tasks, mended) if do_drive else None
            ok = bool(v and v.get("ok"))
            tally[arm].append((row["id"], ok, which))
            print(f"  {arm} {row['id']:22s} accepted via {which} -> DRIVEN ok={ok} "
                  f"why={str((v or {}).get('why'))[:70]!r} {(v or {}).get('driven')}m")
    for arm in tally:
        won = [t[0] for t in tally[arm] if t[1]]
        print(f"  {arm}: {len(won)} of {len(failures(runs[arm]))} recovered {won}")




# ============================================== L. the plan-level claims
def section_l(tasks, runs, do_drive=False):
    print("\n=== L1. self-placement: does GraphMachine accept PLACE_ON_TOP(x) while holding x? ===")
    row = next(r for r in runs["4b"] if r["id"] == "Ihlen_1_int-07")
    seed = WorldGraph.from_scene_graph(belief(row))
    probe = [("NAVIGATE_TO", "plate"), ("GRASP", "plate"), ("PLACE_ON_TOP", "plate")]
    out = GraphMachine(seed.copy()).run(probe, [])
    last = out.steps[-1]
    print(f"  PLACE_ON_TOP('plate') while holding 'plate': ok={last.ok} "
          f"reason={last.reason!r} edits={getattr(last, 'edits', None)}")
    print(f"  graph.has_edge('on_top','plate','plate') afterwards: "
          f"{out.graph.has_edge('on_top', 'plate', 'plate')}")
    print(f"  held afterwards: {out.graph.held_object()}")

    print("\n=== L2. census of attempt-1 plans ===")
    for arm, rows in runs.items():
        faults, self_place, extra_place = {}, 0, 0
        for r in rows:
            plan = plan_of(r, "first_plan")
            if not plan:
                continue
            seed = WorldGraph.from_scene_graph(belief(r))
            out = GraphMachine(seed.copy()).run(plan, goal_of(r))
            if out.failed_at is None:
                faults["ran to the end"] = faults.get("ran to the end", 0) + 1
            else:
                k = (out.steps[out.failed_at].fault or ("unknown", None))[0]
                faults[k] = faults.get(k, 0) + 1
            held = None
            for a, o in plan:
                if a == "GRASP":
                    held = o
                elif a in ("PLACE_ON_TOP", "PLACE_INSIDE"):
                    if held is not None and o == held:
                        self_place += 1
                        break
            g = sum(1 for a, _ in plan if a == "GRASP")
            p = sum(1 for a, _ in plan if a in ("PLACE_ON_TOP", "PLACE_INSIDE"))
            extra_place += p > g
        print(f"  {arm}: first faults {dict(sorted(faults.items(), key=lambda kv: -kv[1]))}")
        print(f"       plans placing onto the carried object: {self_place}; "
              f"plans with more placements than grasps: {extra_place}")

    print("\n=== L3. the appliance-step claims, read out of the stored plans ===")
    for arm, rid, wanted in (("4b", "Wainscott_1_int-01", ("TOGGLE_ON", "clothes_dryer")),
                             ("4b", "Wainscott_1_int-02", ("PLACE_INSIDE", "washer")),
                             ("8b", "Pomaria_1_int-03", ("TOGGLE_ON", "clothes_dryer")),
                             ("8b", "Pomaria_1_int-03", ("PLACE_INSIDE", "washer"))):
        r = next(x for x in runs[arm] if x["id"] == rid)
        plan = plan_of(r, "final_plan")
        hit = [s for s in plan if s[0] == wanted[0]]
        print(f"  {arm} {rid}: {wanted[0]} steps in final_plan = "
              f"{[s[1] for s in hit]}   contains {wanted}: {list(wanted) in [list(s) for s in plan]}")
    r = next(x for x in runs["8b"] if x["id"] == "Pomaria_0_int-10")
    for which in ("first_plan", "final_plan"):
        p = plan_of(r, which)
        print(f"  8b Pomaria_0_int-10 {which}: {len(p)} actions, "
              f"{len(set(p))} distinct, breakfast_table named: "
              f"{any(o == 'breakfast_table' for _, o in p)}")

    print("\n=== L4. exact no-op? replay each swap final plan on the TRUE seed graph ===")
    from build_tasks import seed_graph
    for arm, rid in (("4b", "Ihlen_1_int-09"), ("4b", "Merom_1_int-09"),
                     ("8b", "Pomaria_0_int-06"), ("8b", "Rs_int-10")):
        r = next(x for x in runs[arm] if x["id"] == rid)
        truth = WorldGraph.from_scene_graph(seed_graph(tasks[rid]))
        before = {(t, s, d) for t in ("on_top", "object_inside")
                  for s, d in truth.edges_of(t)}
        out = GraphMachine(truth.copy()).run(plan_of(r, "final_plan"), [])
        after = {(t, s, d) for t in ("on_top", "object_inside")
                 for s, d in out.graph.edges_of(t)}
        print(f"  {arm} {rid}: failed_at={out.failed_at} "
              f"world changed: {before != after} added={sorted(after - before)} "
              f"removed={sorted(before - after)}")

    print("\n=== L5. swap tasks: how many does each model finish? ===")
    swap = [t["id"] for t in tasks.values() if t["task"].strip().lower().startswith("swap")]
    for arm, rows in runs.items():
        got = [r["id"] for r in rows if r["id"] in swap and r["checked"] == "ok"]
        print(f"  {arm}: {len(got)}/{len(swap)} swap tasks pass -> {sorted(got)}")




# ============================================== M. was the harness ready for a right plan?
def section_m(tasks, runs, do_drive=True):
    """The control every attribution needs: the reference plan, this row's own belief and
    its own predicted goal, under the resolver the v8 run used."""
    print("\n=== M. reference plan vs each failing row's belief and predicted goal ===")
    set_resolver("strict")
    for arm, rows in runs.items():
        for row in failures(rows):
            g = belief(row)
            seed = WorldGraph.from_scene_graph(g)
            ref = [(a, o) for a, o in tasks[row["id"]]["plan"]]
            out, acc = accept(seed, ref, goal_of(row))
            v = drive(row, tasks, ref) if do_drive else None
            print(f"  {arm} {row['id']:22s} reference plan accepted={acc} "
                  f"failed_at={out.failed_at} missing={[list(m) for m in out.missing]} "
                  f"| driven ok={(v or {}).get('ok')} {(v or {}).get('driven')}m")


if __name__ == "__main__":
    main()
