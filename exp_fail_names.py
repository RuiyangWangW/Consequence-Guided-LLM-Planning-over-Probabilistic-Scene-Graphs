#!/usr/bin/env python3
"""Did the goal-name gap refuse these plans, or did the model just get them wrong?

`GraphMachine.unmet` reads a goal term with `==` (`_resolve_goal_name` is a documented
no-op: `return name if name in self.graph.objects else name`). The belief graph is keyed by
whatever `task_objects.extract` wrote and the goal by whatever the goal adapter wrote, and
those two are *different models reading the same sentence*. When they disagree on a string
- `tv` against `standing_tv` - a condition can never be satisfied, every attempt is refused,
and `accepted_at` is None with a degenerate last plan reported.

That is a hypothesis, not a finding, and this file tests it. For each suspected row:

    1. rebuild the belief graph the run used - `populate` over the row's own stored
       `extracted`, which is deterministic and reproduces it exactly;
    2. name the goal terms that fail `==` against that graph but resolve under
       `object_names.match`, which is the harness gap made concrete;
    3. re-run `GraphMachine` over the row's stored plans with the goal AS WRITTEN and with
       the goal CANONICALISED, and compare the accept decision `replan.run` would have
       taken (`failed_at is None and safe and goal_met`);
    4. drive the plan that the fix would have accepted through `sim_eval.run_plan`, which
       scores against the TRUE goal in the TRUE world.

Step 4 is what stops the counterfactual being wishful. A name fix that makes the loop
accept a plan the robot then fails to drive has cost the benchmark nothing.

`--scan` asks the same question of all 200 rows: how many carry a goal name that `==`
misses and `object_names.same` catches, and how many of those rows actually failed.

    python exp_fail_names.py                # the five suspected rows, with drives
    python exp_fail_names.py --scan         # every row in both files, no drives
    python exp_fail_names.py --all          # both
"""

import argparse
import contextlib
import json
import sys

from graph_machine import GraphMachine
from object_names import candidates, match, same
from world_graph import WorldGraph

# ------------------------------------------------------------------ the two resolvers
#
# `graph_machine.py` is under active edit by other agents, so the behaviour the v8 rows
# were produced with cannot be read off the file - it has to be pinned here. Both policies
# are patched in at run time; nothing in the repo is modified.
#
#   strict  what HEAD (ac85b59, the commit the v8 runs used) does: `_resolve_goal_name` is
#           the no-op `return name if name in self.graph.objects else name`, so a goal term
#           is compared to graph nodes with `==` and nothing else.
#   same    ask `object_names.same`, and take the answer only when exactly one node fits.
#
# Verified against the file: HEAD line 723 is the no-op, and the v8 result files predate
# today's edit to it (v8-4b.json 2026-09-07 23:58, graph_machine.py 2026-09-08 10:37), so
# `strict` is what refused these plans. `scene_graph.py` is *older* than the result files
# (2026-09-07 23:35), so the working-tree `populate` is the one that built their beliefs.


def _strict(self, name):
    return name if name in self.graph.objects else name


def _same(self, name):
    if name in self.graph.objects:
        return name
    hits = [node for node in self.graph.objects if same(name, node)]
    return hits[0] if len(hits) == 1 else name


POLICIES = {"strict": _strict, "same": _same}


@contextlib.contextmanager
def resolver(policy):
    """Run the machine with one goal-name policy, whatever the file currently says."""
    keep = GraphMachine._resolve_goal_name
    GraphMachine._resolve_goal_name = POLICIES[policy]
    try:
        yield
    finally:
        GraphMachine._resolve_goal_name = keep

FILES = {"4b": "data/v8-4b.json", "8b": "data/v8-8b.json"}

# The rows this file was written to settle.
FOCUS = [("4b", "Wainscott_1_int-01"), ("4b", "Wainscott_1_int-02"),
         ("8b", "Pomaria_1_int-03"), ("4b", "Pomaria_0_int-04"),
         ("8b", "Pomaria_0_int-10")]


def load_rows(path):
    return {r["id"]: r for r in json.load(open(path))["rows"]}


def load_tasks(path="data/tasks.json"):
    return {t["id"]: t for t in json.load(open(path))}


def belief_graph(row):
    """The belief the run actually planned against.

    `evaluate.evaluate` calls `populate` on the extraction it stored in the row, and both
    the RSN forward pass and the ranking that follows it are deterministic, so this is a
    reproduction rather than a re-roll. Checked: two calls produce byte-identical dicts.
    """
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate

    found = row["extracted"]
    return populate(row["scene"], found["uncertain"], found["dependent"],
                    stated=found["stated"], model_path=DEFAULT_MODEL,
                    threshold=DEFAULT_THRESHOLD)


def goal_terms(goal):
    """Every *name* a goal mentions, with where it sits. Booleans are not names."""
    for i, (edge_type, src, dst) in enumerate(goal):
        yield i, "src", edge_type, src
        if isinstance(dst, str):
            yield i, "dst", edge_type, dst


def name_gaps(goal, pool):
    """Goal names `==` misses. Each is one of three kinds, and they mean different things.

        matched      `==` fails, `object_names.match` finds exactly one node - the harness
                     gap this file exists to measure
        ambiguous    several nodes fit equally; a fix has to choose, and choosing is not
                     this module's business
        unresolvable nothing in the belief is this name. The goal model invented a term,
                     or extraction never surfaced the object - not a matching bug
    """
    gaps = []
    for index, side, edge_type, name in goal_terms(goal):
        if name in pool:
            continue
        # `same` and not `match`, because `same` is the symmetric question and `match` is
        # not: `candidates("bath_towel", ["bath_towels"])` is empty while the reverse is
        # not, so asking one-way calls a singular-against-plural pair unresolvable. `same`
        # is also the rule the fix in `graph_machine` uses, so this measures that fix.
        options = [c for c in pool if same(name, c)]
        kind = ("unresolvable" if not options
                else "matched" if len(options) == 1 else "ambiguous")
        gaps.append({"index": index, "side": side, "edge_type": edge_type,
                     "name": name, "kind": kind, "options": options,
                     "resolves_to": options[0] if len(options) == 1 else None,
                     "one_way_match": match(name, pool),
                     "one_way_candidates": candidates(name, pool)})
    return gaps


def resolve_one(name, pool):
    """The single node this name could be, or the name unchanged.

    Unambiguous or nothing: where two nodes fit, choosing between them is a question about
    the world and guessing it is how the loose matching this replaced went wrong.
    """
    if name in pool:
        return name
    hits = [c for c in pool if same(name, c)]
    return hits[0] if len(hits) == 1 else name


def canonicalise(goal, pool):
    """The goal rewritten in the belief graph's own words, where that is unambiguous."""
    out, changed = [], []
    for edge_type, src, dst in goal:
        new_src = resolve_one(src, pool)
        new_dst = resolve_one(dst, pool) if isinstance(dst, str) else dst
        if new_src != src:
            changed.append(f"{src} -> {new_src}")
        if new_dst != dst:
            changed.append(f"{dst} -> {new_dst}")
        out.append((edge_type, new_src, new_dst))
    return out, changed


def machine_verdict(graph, plan, goal, policy="strict"):
    """What `replan.run` would decide about this plan, on this belief, for this goal.

    The accept test is copied from `replan.run` verbatim - applies, tidies up, and reaches
    the goal - so that a difference here is a difference the loop would have acted on.
    """
    seed = WorldGraph.from_scene_graph(graph)
    steps = [(a, o) for a, o in plan]
    goal = [tuple(g) for g in goal]
    with resolver(policy):
        out = GraphMachine(seed.copy()).run(steps, goal)
        accepted = out.failed_at is None and out.safe and (not goal or out.goal_met)
        return {"accepted": bool(accepted), "failed_at": out.failed_at,
                "failed_why": (out.steps[out.failed_at].reason
                               if out.failed_at is not None else None),
                "missing": [list(m) for m in out.missing],
                "left_open": list(out.left_open), "left_on": list(out.left_on)}


def drive(task, graph, plan):
    """The TRUE verdict: run it in the 2-D simulator against `task['goal']`."""
    from sim_eval import run_plan

    try:
        return run_plan(task, graph, [(a, o) for a, o in plan], verbose=False)
    except Exception as exc:                    # a drive that crashes is not a pass
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}", "error": True}


def study(arm, task_id, rows, tasks, do_drive=True):
    """One row, from the belief the run used to what the fix would be worth."""
    row = rows[arm][task_id]
    task = tasks[task_id]
    graph = belief_graph(row)
    pool = sorted(graph["objects"])
    predicted = [tuple(g) for g in row["predicted_goal"]]
    truth = [tuple(g) for g in task["goal"]]

    gaps = name_gaps(predicted, pool)
    fixed_goal, changed = canonicalise(predicted, pool)

    report = {"arm": arm, "id": task_id, "task": row["task"],
              "belief_objects": pool,
              "belief_rooms": {n: (i.get("room")) for n, i in graph["objects"].items()},
              "predicted_goal": [list(g) for g in predicted],
              "true_goal": [list(g) for g in truth],
              "goal_name_gaps": gaps,
              "canonicalised_goal": [list(g) for g in fixed_goal],
              "renames": changed,
              "accepted_at": row["accepted_at"], "attempts": row["attempts"],
              "checked": row["checked"], "checked_why": row["checked_why"],
              "plans": {}}

    for which in ("first_plan", "final_plan"):
        plan = row[which]
        if not plan:
            continue
        # Four readings of one plan, differing only in how a goal *name* is read.
        #   as_written     the goal as the adapter wrote it, `==` only - reproduces v8
        #   same_resolver  the same goal, with the machine asking `object_names.same`
        #   canonicalised  the goal rewritten into the belief's words before the machine
        #                  ever sees it - the fix implemented outside graph_machine, so the
        #                  two agreeing is evidence and not a tautology
        #   against_true_goal  the ANSWER KEY, which no stage sees. Separates "wrote the
        #                  wrong words" from "wrote the wrong goal".
        entry = {"steps": len(plan),
                 "as_written": machine_verdict(graph, plan, predicted, "strict"),
                 "same_resolver": machine_verdict(graph, plan, predicted, "same"),
                 "canonicalised": machine_verdict(graph, plan, fixed_goal, "strict"),
                 "against_true_goal": machine_verdict(
                     graph, plan, canonicalise(truth, pool)[0], "strict")}
        if do_drive:
            entry["driven"] = drive(task, graph, plan)
        report["plans"][which] = entry

    order = ("first_plan", "final_plan")
    got = lambda w, k: report["plans"].get(w, {}).get(k, {}).get("accepted")
    did_accept = [w for w in order if got(w, "as_written")]
    would_accept = [w for w in order if got(w, "canonicalised")]
    same_accept = [w for w in order if got(w, "same_resolver")]
    # Would the answer key itself have been reached? If the plan misses the TRUE goal too,
    # no amount of name-fixing saves the task and the model simply planned the wrong thing.
    truth_accept = [w for w in order if got(w, "against_true_goal")]
    # A rescue only counts if the plan the fix accepts drives clean in the true world.
    rescued = next((w for w in would_accept
                    if report["plans"][w].get("driven", {}).get("ok")), None)
    report["verdict"] = {
        "name_gap_present": any(g["kind"] == "matched" for g in gaps),
        "accepted_as_written": did_accept,
        "accepted_if_canonicalised": would_accept,
        "accepted_with_same_resolver": same_accept,
        "reaches_true_goal_symbolically": truth_accept,
        "gap_changed_the_decision": bool(set(would_accept) - set(did_accept)),
        "fix_rescues_via": rescued,
        "fix_would_pass": bool(rescued),
    }
    return report


# --------------------------------------------------------------- the other counterfactual
#
# "What would have to change for this to pass?" has two possible answers, and only measuring
# tells them apart. Either the *checker* was wrong about a correct plan - the name gap - or
# the *plan* was wrong and the checker said so. For the second, the test is a plan that is
# right: hand-written, run against the same belief and the same predicted goal the row used,
# and driven in the same house. If that is accepted and drives clean, nothing in the harness
# was standing between this task and a pass, and the refusal was the model's own mistake.
#
# These are written in the belief's own vocabulary - `tshirt`, `bath_towels` - because that
# is what the row's belief graph holds; using the dataset's spelling would test the name gap
# again instead of the plan.
REFERENCE = {
    "Wainscott_1_int-01": [
        ("NAVIGATE_TO", "tshirt"), ("GRASP", "tshirt"),
        ("NAVIGATE_TO", "washer"), ("OPEN", "washer"), ("PLACE_INSIDE", "washer"),
        ("CLOSE", "washer"), ("TOGGLE_ON", "washer"), ("TOGGLE_OFF", "washer"),
        ("OPEN", "washer"), ("GRASP", "tshirt"), ("CLOSE", "washer"),
        ("NAVIGATE_TO", "clothes_dryer"), ("OPEN", "clothes_dryer"),
        ("PLACE_INSIDE", "clothes_dryer"), ("CLOSE", "clothes_dryer"),
        ("TOGGLE_ON", "clothes_dryer"), ("TOGGLE_OFF", "clothes_dryer"),
    ],
    "Wainscott_1_int-02": [
        ("NAVIGATE_TO", "bath_towels"), ("GRASP", "bath_towels"),
        ("NAVIGATE_TO", "washer"), ("OPEN", "washer"), ("PLACE_INSIDE", "washer"),
        ("CLOSE", "washer"), ("TOGGLE_ON", "washer"), ("TOGGLE_OFF", "washer"),
        ("OPEN", "washer"), ("GRASP", "bath_towels"), ("CLOSE", "washer"),
        ("NAVIGATE_TO", "clothes_dryer"), ("OPEN", "clothes_dryer"),
        ("PLACE_INSIDE", "clothes_dryer"), ("CLOSE", "clothes_dryer"),
        ("TOGGLE_ON", "clothes_dryer"), ("TOGGLE_OFF", "clothes_dryer"),
    ],
    "Pomaria_1_int-03": [
        ("NAVIGATE_TO", "bath_towels"), ("GRASP", "bath_towels"),
        ("NAVIGATE_TO", "washer"), ("OPEN", "washer"), ("PLACE_INSIDE", "washer"),
        ("CLOSE", "washer"), ("TOGGLE_ON", "washer"), ("TOGGLE_OFF", "washer"),
        ("OPEN", "washer"), ("GRASP", "bath_towels"), ("CLOSE", "washer"),
        ("NAVIGATE_TO", "clothes_dryer"), ("OPEN", "clothes_dryer"),
        ("PLACE_INSIDE", "clothes_dryer"), ("CLOSE", "clothes_dryer"),
        ("TOGGLE_ON", "clothes_dryer"), ("TOGGLE_OFF", "clothes_dryer"),
    ],
    "Pomaria_0_int-10": [
        ("NAVIGATE_TO", "bottom_cabinet"), ("OPEN", "bottom_cabinet"),
        ("GRASP", "folder"), ("NAVIGATE_TO", "breakfast_table"),
        ("PLACE_ON_TOP", "breakfast_table"),
        ("NAVIGATE_TO", "bottom_cabinet"), ("GRASP", "envelope"),
        ("NAVIGATE_TO", "breakfast_table"), ("PLACE_ON_TOP", "breakfast_table"),
        ("NAVIGATE_TO", "bottom_cabinet"), ("CLOSE", "bottom_cabinet"),
    ],
}


def reference_check(arm, task_id, rows, tasks, do_drive=True):
    """Is a *correct* plan for this task accepted, on this row's own belief and goal?"""
    row, task = rows[arm][task_id], tasks[task_id]
    graph = belief_graph(row)
    plan = REFERENCE[task_id]
    predicted = [tuple(g) for g in row["predicted_goal"]]
    out = {"arm": arm, "id": task_id, "steps": len(plan),
           "plan": [list(s) for s in plan],
           "as_written": machine_verdict(graph, plan, predicted, "strict"),
           "against_true_goal": machine_verdict(
               graph, plan, canonicalise([tuple(g) for g in task["goal"]],
                                         sorted(graph["objects"]))[0], "strict")}
    if do_drive:
        out["driven"] = drive(task, graph, plan)
    return out


def repair_text(arm, task_id, rows, tasks, which="first_plan"):
    """The complaint the loop actually sent back to the model after refusing this plan.

    Reading it is the difference between inferring the mechanism and seeing it: the prompt
    names the condition the model is being asked to fix, and on `Pomaria_0_int-04` that
    condition is one the plan already satisfies under any other spelling.
    """
    from replan import repair_prompt

    row, task = rows[arm][task_id], tasks[task_id]
    graph = belief_graph(row)
    plan = row[which]
    predicted = [tuple(g) for g in row["predicted_goal"]]
    seed = WorldGraph.from_scene_graph(graph)
    with resolver("strict"):
        outcome = GraphMachine(seed.copy()).run([(a, o) for a, o in plan], predicted)
        steps = [{"action": a, "object": o} for a, o in plan]
        return repair_prompt(row["task"], graph, steps, outcome)


def scan(rows, tasks, verbose=True):
    """Every row of both files: which carry a goal name `==` misses and `same` catches.

    The belief graph is keyed by extraction's own words, so the pool a goal term is
    compared against is exactly what `populate` placed - rebuilt per row, not guessed.
    """
    hits, truth_gap = [], []
    for arm, table in rows.items():
        for task_id, row in table.items():
            predicted = [tuple(g) for g in (row["predicted_goal"] or ())]
            if not predicted:
                continue
            graph = belief_graph(row)
            pool = sorted(graph["objects"])

            # The OTHER naming gap, at a different interface: the belief and the goal agree
            # with each other and both disagree with the dataset - `tshirt` where the answer
            # key says `t_shirt`. That one crosses into the simulator, where `sim_eval.ground`
            # resolves it with the same `object_names` rules, so it should cost nothing. Rows
            # carrying it that still pass are the evidence that it does not.
            true_names = {n for _, _, _, n in goal_terms([tuple(g) for g in tasks[task_id]["goal"]])}
            crossings = sorted({(n, t) for _, _, _, n in goal_terms(predicted)
                                for t in true_names if n != t and same(n, t)})
            if crossings:
                truth_gap.append({"arm": arm, "id": task_id, "pairs": crossings,
                                  "checked": row["checked"], "failed": row["checked"] != "ok",
                                  "accepted_at": row["accepted_at"]})

            gaps = name_gaps(predicted, pool)
            if not gaps:
                continue
            hits.append({"arm": arm, "id": task_id, "task": row["task"],
                         "pool": pool,
                         "predicted_goal": [list(g) for g in predicted],
                         "gaps": gaps,
                         "kinds": sorted({g["kind"] for g in gaps}),
                         "accepted_at": row["accepted_at"],
                         "checked": row["checked"], "checked_why": row["checked_why"],
                         "failed": row["checked"] != "ok"})
            if verbose:
                print(f"  {arm} {task_id:22s} {'/'.join(sorted({g['kind'] for g in gaps}))}"
                      f"  {[g['name'] + '->' + str(g['resolves_to']) for g in gaps]}"
                      f"  checked={row['checked']}")
    return hits, truth_gap


def confirm(hit, rows, tasks):
    """For a scanned row, does the gap actually change the accept decision?"""
    return study(hit["arm"], hit["id"], rows, tasks, do_drive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", action="store_true", help="all 200 rows, no drives")
    parser.add_argument("--reference", action="store_true",
                        help="is a correct plan accepted on the same belief and goal?")
    parser.add_argument("--prompt", metavar="ARM:ID",
                        help="print the complaint the loop sent back for one row")
    parser.add_argument("--all", action="store_true", help="focus rows and the scan")
    parser.add_argument("--no-drive", action="store_true")
    parser.add_argument("--out", default="data/exp-fail-names.json")
    args = parser.parse_args()

    rows = {arm: load_rows(path) for arm, path in FILES.items()}
    tasks = load_tasks()
    result = {}

    if not args.scan or args.all:
        result["focus"] = []
        for arm, task_id in FOCUS:
            print(f"\n{'=' * 92}\n{arm}  {task_id}")
            report = study(arm, task_id, rows, tasks, do_drive=not args.no_drive)
            result["focus"].append(report)
            print(f"  task            {report['task']}")
            print(f"  belief objects  {report['belief_objects']}")
            print(f"  predicted goal  {report['predicted_goal']}")
            print(f"  true goal       {report['true_goal']}")
            print(f"  name gaps       {[(g['name'], g['kind'], g['resolves_to']) for g in report['goal_name_gaps']] or 'NONE - every goal name is a belief node'}")
            for which, entry in report["plans"].items():
                aw, cn, tg = entry["as_written"], entry["canonicalised"], entry["against_true_goal"]
                sr = entry["same_resolver"]
                print(f"  {which} ({entry['steps']} steps)")
                print(f"     as written [v8] accepted={aw['accepted']}  failed_at={aw['failed_at']}"
                      f"  missing={aw['missing']}  open={aw['left_open']} on={aw['left_on']}")
                print(f"     same resolver   accepted={sr['accepted']}  missing={sr['missing']}")
                print(f"     canonicalised   accepted={cn['accepted']}  failed_at={cn['failed_at']}"
                      f"  missing={cn['missing']}")
                print(f"     vs TRUE goal    accepted={tg['accepted']}  missing={tg['missing']}")
                if "driven" in entry:
                    d = entry["driven"]
                    print(f"     driven         ok={d['ok']}  {d.get('driven')}m  {d.get('why', '')[:90]}")
            print(f"  VERDICT         {report['verdict']}")

    if args.prompt:
        arm, task_id = args.prompt.split(":")
        print(repair_text(arm, task_id, rows, tasks))
        return 0

    if args.reference or args.all:
        print(f"\n{'=' * 92}\na CORRECT plan, on the row's own belief and its own predicted goal\n")
        result["reference"] = []
        for arm, task_id in FOCUS:
            if task_id not in REFERENCE:
                continue
            out = reference_check(arm, task_id, rows, tasks, do_drive=not args.no_drive)
            result["reference"].append(out)
            aw, tg = out["as_written"], out["against_true_goal"]
            print(f"  {arm} {task_id:22s} {out['steps']:2d} steps"
                  f"  accepted={aw['accepted']}  missing={aw['missing']}"
                  f"  failed_at={aw['failed_at']} {aw['failed_why'] or ''}")
            print(f"      vs TRUE goal accepted={tg['accepted']} missing={tg['missing']}")
            if "driven" in out:
                d = out["driven"]
                print(f"      driven ok={d['ok']} {d.get('driven')}m {d.get('why', '')[:80]}")

    if args.scan or args.all:
        print(f"\n{'=' * 92}\nscanning all rows for goal names `==` misses\n")
        hits, truth_gap = scan(rows, tasks)
        result["scan"] = hits
        result["belief_vs_truth_gap"] = truth_gap
        matched = [h for h in hits if "matched" in h["kinds"]]
        failed = [h for h in matched if h["failed"]]
        print(f"\n  rows with any goal-name gap        {len(hits)} / "
              f"{sum(len(t) for t in rows.values())}")
        print(f"  ... resolvable by object_names     {len(matched)}")
        print(f"  ... of those, the row FAILED       {len(failed)}")
        print(f"  ... of those, accepted_at is None  {sum(1 for h in failed if h['accepted_at'] is None)}")
        by_kind = {}
        for h in hits:
            for k in h["kinds"]:
                by_kind.setdefault(k, []).append(h["id"])
        for kind, ids in sorted(by_kind.items()):
            print(f"  kind {kind:14s} {len(ids)}  {ids}")
        tg_failed = [h for h in truth_gap if h["failed"]]
        print(f"\n  rows whose goal AND belief spell it differently from the answer key: "
              f"{len(truth_gap)}")
        print(f"  ... of those the row FAILED        {len(tg_failed)}  "
              f"{[h['id'] for h in tg_failed]}")
        print(f"  ... so this second gap costs        {len(tg_failed)} of {len(truth_gap)} rows"
              f" - it is resolved by sim_eval.ground")
        for h in truth_gap:
            print(f"      {h['arm']} {h['id']:22s} {h['pairs']}  checked={h['checked']}")
        # A gap only costs a task if it flips the decision. Confirm each resolvable one.
        print("\n  confirming which resolvable gaps flip the accept decision:")
        result["confirmed"] = []
        for h in matched:
            report = confirm(h, rows, tasks)
            result["confirmed"].append(report)
            v = report["verdict"]
            print(f"    {h['arm']} {h['id']:22s} flips={v['gap_changed_the_decision']}"
                  f"  as_written={v['accepted_as_written']}"
                  f"  canonical={v['accepted_if_canonicalised']}  checked={h['checked']}")

    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
