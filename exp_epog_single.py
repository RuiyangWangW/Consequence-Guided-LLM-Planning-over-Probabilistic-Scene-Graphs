#!/usr/bin/env python3
"""Does the EPoG baseline hold up on the single-task long-horizon benchmark?

`data/multitask.json` turned out to be an easy world for a graph-diff planner: its goals are
`on_top`, `object_inside` and `toggled` and nothing else, so the difference between the belief
and the goal *is* the task, and EPoG scored 48/50 against the pipeline's 44/50.

`data/tasks.json` is not that world. Twenty-two of its hundred tasks ask for a conferred state
- `cooked`, `washed`, `dried` - which corresponds to no primitive at all and has to be expanded
into an appliance macro: put the thing in, shut the door, run it, switch it off, open up, take
it back out, shut the door again. Nine steps where the goal graph names one edge, and every one
of them a chance for the expansion to be wrong. This is the part of EPoG that `multitask.json`
never exercised.

Three arms, one simulator, one belief:

    reference     the benchmark's own plan. Known-good, the control: what it fails, nobody can.
    epog-true     EPoG given the true goal. Its ceiling - every failure here is the planner's
                  own coverage, with perception and the goal adapter taken out of it.
    epog-pred     EPoG given the goal the adapter reads off the instruction, which is what the
                  baseline actually gets. The gap to `epog-true` is the goal adapter's cost,
                  and for this planner that cost is total: the goal *is* the plan, so a goal
                  that comes back wrong is not degraded, it is fatal.

All three are grounded against the same belief graph and judged by the 2-D simulator - every
action applied, the goal true in the real house, nothing left open or switched on.
"""

import argparse
import json
import sys
import time

import epog
from finetune_extraction import GOAL_INSTRUCTION, INSTRUCTION
from planner import get_generator, parse_goal
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import run_plan

DEFAULT_GOAL_MODEL = "models/state-1.7b-v5"
DEFAULT_EXTRACTOR = "models/h1-1.7b-v5"


def drive(task, belief, steps):
    if not steps:
        return {"ok": False, "why": "empty plan", "driven": None, "failed_at": None}
    try:
        return run_plan(task, belief, steps, verbose=False)
    except Exception as exc:
        return {"ok": False, "why": f"{type(exc).__name__}: {exc}", "driven": None,
                "failed_at": None, "error": True}


def one(task, goal_gen, extract_gen=None):
    row = {"id": task["id"], "scene": task["scene"], "task": task["task"], "arms": {}}
    truth_goal = [tuple(g) for g in task["goal"]]
    row["confers"] = sorted({g[0] for g in truth_goal} & {"cooked", "washed", "dried"})

    # The belief the baseline is actually given. With `extract_gen` this is the pipeline's
    # own stage 1, the same adapter and the same prompt every other method runs, so the
    # comparison is like for like; without it the benchmark's ground-truth extraction is used
    # and the number is a ceiling rather than a score.
    ex = task["extraction"]
    if extract_gen is not None:
        from task_objects import extract
        got = extract(task["task"], generator=extract_gen, prompt=INSTRUCTION + "\n")
        ex = {"uncertain": got["uncertain"], "stated": got.get("stated") or {},
              "dependent": got["dependent"]}
    row["extracted"] = ex
    belief = populate(task["scene"], ex["uncertain"], ex["dependent"],
                      stated=ex.get("stated") or {},
                      model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)

    at = time.perf_counter()
    predicted = [tuple(g) for g in
                 parse_goal("GOAL:\n" + goal_gen(GOAL_INSTRUCTION.format(task=task["task"]),
                                                 200))]
    row["goal_seconds"] = round(time.perf_counter() - at, 2)
    row["goal_true"] = [list(g) for g in truth_goal]
    row["goal_predicted"] = [list(g) for g in predicted]
    row["goal_exact"] = set(predicted) == set(truth_goal)

    plans = {"reference": [(a, o) for a, o in task["plan"]]}
    for name, goal in (("epog-true", truth_goal), ("epog-pred", predicted)):
        at = time.perf_counter()
        steps, cost, scored = epog.plan(task, belief, goal)
        plans[name] = steps
        row.setdefault("plan_seconds", {})[name] = round(time.perf_counter() - at, 2)
        row.setdefault("c_map", {})[name] = round(cost, 1)
        row.setdefault("scored", {})[name] = scored

    for name, steps in plans.items():
        sim = drive(task, belief, steps)
        row["arms"][name] = {
            "ok": bool(sim.get("ok")), "why": sim.get("why", ""),
            "driven": sim.get("driven"), "steps": len(steps),
            "sim_steps": sim.get("sim_steps"), "sim_seconds": sim.get("sim_seconds"),
            "failed_at": sim.get("failed_at"),
            "missing": sim.get("missing") or [], "unsafe": sim.get("unsafe") or [],
            "plan": [list(s) for s in steps],
        }
    return row


def report(rows):
    good = [r for r in rows if "arms" in r]
    arms = ("reference", "epog-true", "epog-pred")
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    conf = [r for r in good if r["confers"]]
    print(f"\n{len(good)} single-horizon tasks, {len(conf)} of them asking for a "
          f"conferred state\n")
    print(f"  {'arm':12s}{'success':>10s}{'on confer':>12s}{'driven(ok)':>13s}{'steps':>8s}")
    for a in arms:
        ok = [r for r in good if r["arms"][a]["ok"]]
        okc = [r for r in conf if r["arms"][a]["ok"]]
        print(f"  {a:12s}{len(ok):5d}/{len(good):<4d}{len(okc):7d}/{len(conf):<4d}"
              f"{mean([r['arms'][a]['driven'] for r in ok if r['arms'][a]['driven'] is not None]):12.1f}m"
              f"{mean([r['arms'][a]['steps'] for r in good]):8.1f}")
    exact = sum(1 for r in good if r["goal_exact"])
    print(f"\n  goal adapter exact on {exact}/{len(good)} tasks")
    for a in ("epog-true", "epog-pred"):
        bad = [r for r in good if not r["arms"][a]["ok"]]
        print(f"\n  {a} failures ({len(bad)}):")
        for r in bad[:14]:
            tag = "+".join(r["confers"]) or "-"
            print(f"    {r['id']:22s} [{tag:6s}] {r['arms'][a]['why'][:72]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/tasks.json")
    ap.add_argument("--goal-model", dest="goal_model", default=DEFAULT_GOAL_MODEL)
    ap.add_argument("--extractor", default=None,
                    help=f"run stage 1 with this adapter (e.g. {DEFAULT_EXTRACTOR}) instead "
                         f"of using the benchmark's ground-truth extraction")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--analyse", help="re-report from a saved JSON and exit")
    args = ap.parse_args()

    if args.analyse:
        report(json.load(open(args.analyse)))
        return 0

    tasks = json.load(open(args.tasks))
    mine = [t for i, t in enumerate(tasks) if i % args.shards == args.shard]
    goal_gen = get_generator(adapter=args.goal_model)
    extract_gen = get_generator(adapter=args.extractor) if args.extractor else None

    rows = []
    for i, task in enumerate(mine):
        try:
            rows.append(one(task, goal_gen, extract_gen))
        except Exception as exc:
            rows.append({"id": task["id"], "error": f"{type(exc).__name__}: {exc}"})
        json.dump(rows, open(args.out, "w"), indent=1)
        done = [r for r in rows if "arms" in r]
        if done and ((i + 1) % 5 == 0 or i + 1 == len(mine)):
            line = "  ".join(f"{a}={sum(1 for r in done if r['arms'][a]['ok'])}/{len(done)}"
                             for a in ("reference", "epog-true", "epog-pred"))
            print(f"[{args.shard}] {i+1}/{len(mine)}  {line}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
