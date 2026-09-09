#!/usr/bin/env python3
"""The five ordering methods, measured on the benchmark's own reference subplans.

Running them through the pipeline measures the ordering *plus* whatever the planner did that
day. Running them on the benchmark's reference subplans - identical for every method, and known
good - leaves the ordering as the only thing that varies. That is what this does.

    SayPlan       no ordering stage; the errands in the order they were written
    GAVEL-MAP     reorders online, carrying one best guess per object
    GAVEL Static  orders once against the prior, never revises
    GAVEL         reorders online, carrying the full distribution
    Oracle        knows every object's room, and picks the ordering that is genuinely
                  shortest by driving all of them

See `baselines.py` for what each is and why. Two distances are reported throughout: `walked`,
what the cost model charged, and `driven`, what the 2-D simulator really drove.

    python oracle_order.py --tasks data/multitask.json --out data/order-oracle.json
"""

import os as _os, sys as _sys
# Walk up to the repo root - the directory holding the library modules - so this file
# runs from wherever it is filed. Anchored on a marker rather than a fixed number of
# parents, so moving it a level deeper does not silently break the import.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.exists(_os.path.join(_d, 'graph_machine.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)


import argparse
import json

from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import route_cost, route_matrix, run_plan
from build_tasks import seed_graph
from world_graph import WorldGraph

import baselines
import gavel
from graph_machine import GraphMachine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/multitask.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-sim", action="store_true",
                        help="skip the simulator and report the cost model only")
    parser.add_argument("--out")
    parser.add_argument("--arms", default=",".join(baselines.ALL),
                        help="which methods to run; ORACLE is by far the most expensive "
                             "because it drives every permutation to choose between them")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))[::args.stride]
    if args.limit:
        tasks = tasks[:args.limit]
    if args.shards > 1:
        tasks = tasks[args.shard::args.shards]
    arms = tuple(a for a in baselines.ALL if a in set(args.arms.split(",")))

    # Which benchmark this ran against. Results carrying different stamps are not a
    # comparison, however similar their task ids look.
    from build_multitask import stamp_of
    stamp = stamp_of(json.load(open(args.tasks)), None, None)

    rows = []
    for index, task in enumerate(tasks, 1):
        ex = task["extraction"]
        # All three classes, as the pipeline passes them. Dropping `stated` - which this did -
        # throws away rooms the instruction names outright and sets the RSN guessing instead,
        # so the beliefs the ordering is computed over are worse than the task warrants.
        graph = populate(task["scene"], ex["uncertain"], ex["dependent"],
                         stated=ex.get("stated") or {},
                         model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
        seed = WorldGraph.from_scene_graph(graph)
        plans = [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]]
        goal = [tuple(g) for g in task["goal"]]
        row = {"id": task["id"], "scene": task["scene"], "subgoals": len(plans)}

        def drive(ordered):
            steps, _ = gavel.compose(seed, ordered, goal)
            try:
                return run_plan(task, graph, steps, verbose=False)["driven"]
            except Exception:
                return None

        # Oracle gets ground truth throughout - see `baselines.oracle`.
        truth_graph = seed_graph(task)
        cached = {}

        def oracle_drive(ordered):
            steps, _ = gavel.compose(seed, ordered, goal)
            try:
                return run_plan(task, truth_graph, steps, verbose=False)["driven"]
            except Exception:
                return None

        def oracle_measure(order):
            steps, _ = gavel.compose(seed, [plans[i] for i in order], goal)
            if "legs" not in cached:
                cached["legs"], cached["start"] = route_matrix(task, truth_graph, steps)
            return route_cost(steps, cached["legs"], cached["start"])

        for method in arms:
            if method == baselines.ORACLE and args.no_sim:
                continue
            is_oracle = method == baselines.ORACLE
            result = baselines.run(
                method, task, plans, truth_graph if is_oracle else graph, seed,
                drive=oracle_drive if is_oracle else drive,
                measure=oracle_measure if is_oracle else None, goal=goal)
            if result.get("steps") is not None:
                steps = result["steps"]                       # EPoG wrote its own plan
                composed = GraphMachine(seed.copy()).run(steps, goal)
            else:
                ordered = [plans[i] for i in result["order"]]
                steps, composed = gavel.compose(seed, ordered, goal)
            entry = {"order": result["order"], "walked": round(result["walked"], 1),
                     "reorders": result.get("reorders", 0),
                     "blocked": result.get("blocked", 0),
                     "order_seconds": result.get("order_seconds"),
                     "composed_ok": composed.failed_at is None and not composed.missing
                                    and composed.safe}
            if not args.no_sim:
                if method == baselines.ORACLE and result.get("driven") is not None:
                    # Re-drive the ordering it chose, so its step count and success are
                    # measured the same way as everyone else's - against the truth graph,
                    # which is what Oracle is given.
                    sim = run_plan(task, truth_graph, steps, verbose=False)
                    entry.update({"ok": sim["ok"], "driven": sim["driven"],
                                  "sim_steps": sim["sim_steps"],
                                  "sim_seconds": sim["sim_seconds"], "why": sim["why"][:200]})
                else:
                    try:
                        sim = run_plan(task, graph, steps, verbose=False)
                        entry.update({"ok": sim["ok"], "driven": sim["driven"],
                                      "sim_steps": sim["sim_steps"],
                                      "sim_seconds": sim["sim_seconds"],
                                      "why": sim["why"][:200]})
                    except Exception as exc:
                        entry.update({"ok": False, "driven": None,
                                      "why": f"{type(exc).__name__}: {exc}"})
            row[method] = entry
        rows.append(row)
        print(f"[{index}/{len(tasks)}] {row['id']:24s} " + "  ".join(
            f"{baselines.LABELS[m]} {row[m].get('driven')}m" for m in baselines.ALL
            if m in row), flush=True)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(rows, handle, indent=1)
            with open(args.out.replace(".json", "-stamp.json"), "w") as handle:
                json.dump(stamp, handle, indent=1)

    summarise(rows)


def summarise(rows):
    """Success rate, travel distance and running time, per method.

    Running time is the simulator's control steps at `sim_eval.SIM_STEP_SECONDS` apiece plus
    the method's own compute. They are reported separately as well as together, because they
    are different kinds of cost: the first is the robot's time and scales with how far it has
    to go, the second is the planner's and scales with how hard it thinks.
    """
    n = len(rows)
    print(f"\n{n} instructions\n")
    print(f"  {'method':14s} {'success':>9s} {'driven':>10s} "
          f"{'sim time':>10s} {'compute':>9s} {'total':>9s}")
    for method in baselines.ALL:
        got = [r[method] for r in rows if method in r]
        if not got:
            continue
        ok = sum(1 for g in got if g.get("ok"))
        driven = [g["driven"] for g in got if g.get("driven") is not None]
        simt = [g.get("sim_seconds") or 0.0 for g in got]
        comp = [g.get("order_seconds") or 0.0 for g in got]
        mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
        print(f"  {baselines.LABELS[method]:14s} {ok:4d}/{len(got):<4d} "
              f"{mean(driven):9.1f}m "
              f"{mean(simt):9.1f}s {mean(comp):8.2f}s {mean(simt)+mean(comp):8.1f}s")
    print("\n  (means per instruction; success is the 2-D simulator's verdict against the"
          " true world.\n   Distance is what the robot DROVE. Each method's own cost estimate"
          " is kept in the rows but not\n   reported: it is internal to the method, and the"
          " methods do not all estimate the same\n   quantity - EPoG's C_MAP charges only"
          " navigation, GAVEL's also charges searching.)")

    print("\n  paired against GAVEL, on driven distance:")
    base = baselines.GAVEL
    for method in baselines.ALL:
        if method == base:
            continue
        pairs = [(r[method], r[base]) for r in rows if method in r and base in r
                 and r[method].get("driven") is not None and r[base].get("driven") is not None]
        if not pairs:
            continue
        a = sum(x["driven"] for x, _ in pairs)
        b = sum(y["driven"] for _, y in pairs)
        print(f"    {baselines.LABELS[method]:14s} {a:8.0f} m  ->  GAVEL {b:8.0f} m   "
              f"{100*(a-b)/max(a,1e-9):+6.1f}%   over {len(pairs)} tasks")

    reorders = {m: sum(r.get(m, {}).get("reorders") or 0 for r in rows)
                for m in (baselines.GAVEL_MAP, baselines.GAVEL)}
    print("\n  orderings revised online: " +
          ", ".join(f"{baselines.LABELS[m]} {reorders[m]}" for m in reorders))
    print(f"  orderings rejected by the composition check: "
          f"{sum(r.get(baselines.GAVEL, {}).get('blocked') or 0 for r in rows)}")

    by_n = {}
    for r in rows:
        acc = by_n.setdefault(r["subgoals"], {m: [0.0, 0] for m in baselines.ALL})
        for m in baselines.ALL:
            d = r.get(m, {}).get("driven")
            if d is not None:
                acc[m][0] += d; acc[m][1] += 1
    print("\n  mean driven distance by number of errands:")
    print("    errands  " + "".join(f"{baselines.LABELS[m]:>14s}" for m in baselines.ALL))
    for k in sorted(by_n):
        cells = "".join(f"{(by_n[k][m][0]/by_n[k][m][1]):>13.1f}m" if by_n[k][m][1]
                        else f"{'-':>14s}" for m in baselines.ALL)
        print(f"    {k:>7d}  {cells}")


if __name__ == "__main__":
    raise SystemExit(main())
