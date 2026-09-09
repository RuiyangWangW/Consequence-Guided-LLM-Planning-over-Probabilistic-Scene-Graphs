#!/usr/bin/env python3
"""What can the EPoG baseline actually do, and what does collapsing the belief cost it?

EPoG writes its plan by subtracting the believed graph from the goal graph. That gives it a
real advantage over ordering somebody else's per-errand subplans - it sees all the goals at
once and never emits a redundant drive - and a real weakness that the headline table hides:
**the structure of its plan is read off the belief, not only the positions.** `_support_of`
decides whether the object it wants is inside a closed container, and that decides whether
the plan contains an `OPEN`. Believe the textbook is loose on a table when the truth has it
shut inside a cabinet and the plan has no `OPEN` in it; the simulator refuses `GRASP` and the
whole instruction fails. No ordering can repair that, because the missing action was never
written.

Three arms, same tasks, same simulator, same true goal, so the only thing that varies is what
the planner saw:

    reference    the benchmark's own subplans, concatenated in the order the instruction
                 names them. Known-good. The control: anything it fails, nobody can do.
    epog-truth   EPoG planning against `seed_graph(task)` - the true world. Its symbolic
                 ceiling: every failure here is the planner's own coverage, not perception.
    epog-belief  EPoG planning against `scene_graph.populate` - the MAP belief, which is what
                 the baseline is actually given. The gap to `epog-truth` is the price of
                 collapsing a distribution to its argmax.

The true goal is used rather than the pipeline's predicted one deliberately: a goal-adapter
error would show up as an EPoG failure and it is not one. This measures the planner.
"""

import argparse
import json
import sys
import time

import epog
from build_tasks import seed_graph
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import run_plan


def reference_plan(task):
    """The benchmark's subplans, concatenated in the order the instruction names them."""
    return [(a, o) for sub in task["subgoals"] for a, o in sub["plan"]]


def drive(task, belief, steps):
    """Run `steps` in the simulator against the true world. Grounding always uses `belief`,
    so the arms differ in the plan they wrote and in nothing else."""
    if not steps:
        return {"ok": False, "why": "empty plan", "driven": None, "failed_at": None}
    try:
        return run_plan(task, belief, steps, verbose=False)
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}", "driven": None,
                "failed_at": None, "error": True}


def one(task):
    row = {"id": task["id"], "scene": task["scene"], "errands": len(task["subgoals"]),
           "arms": {}}
    goal = [tuple(g) for g in task["goal"]]

    at = time.perf_counter()
    ex = task["extraction"]
    belief = populate(task["scene"], ex["uncertain"], ex["dependent"],
                      stated=ex.get("stated") or {},
                      model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    row["populate_seconds"] = round(time.perf_counter() - at, 2)
    truth = seed_graph(task)          # the same scene-graph dict shape as the belief

    plans = {"reference": (reference_plan(task), 0, 0.0)}
    for name, graph in (("epog-truth", truth), ("epog-belief", belief)):
        at = time.perf_counter()
        steps, _cost, scored = epog.plan(task, graph, goal)
        plans[name] = (steps, scored, _cost)
        row.setdefault("plan_seconds", {})[name] = round(time.perf_counter() - at, 2)

    for name, (steps, scored, cost) in plans.items():
        sim = drive(task, belief, steps)
        row["arms"][name] = {
            "ok": bool(sim.get("ok")), "why": sim.get("why", ""),
            "driven": sim.get("driven"), "steps": len(steps),
            "failed_at": sim.get("failed_at"),
            "failed_action": (list(steps[sim["failed_at"]])
                              if sim.get("failed_at") is not None
                              and sim["failed_at"] < len(steps) else None),
            "orders_scored": scored, "c_map": round(cost, 1) if cost else 0.0,
            "plan": [list(s) for s in steps],
        }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))
    if args.limit:
        tasks = tasks[:args.limit]
    mine = [t for i, t in enumerate(tasks) if i % args.shards == args.shard]

    rows = []
    for i, task in enumerate(mine):
        try:
            rows.append(one(task))
        except Exception as exc:
            rows.append({"id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
        if (i + 1) % 10 == 0 or i + 1 == len(mine):
            json.dump(rows, open(args.out, "w"), indent=1)
            done = [r for r in rows if "arms" in r]
            line = "  ".join(
                f"{a}={sum(1 for r in done if r['arms'][a]['ok'])}/{len(done)}"
                for a in ("reference", "epog-truth", "epog-belief") if done)
            print(f"[{args.shard}] {i+1}/{len(mine)}  {line}", flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
