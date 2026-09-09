#!/usr/bin/env python3
"""Does reordering pay when the robot is CONFIDENTLY WRONG, rather than merely unsure?

Six earlier experiments concluded that online reordering adds little because 83% of task
objects already have a point-mass belief - "nothing left to learn". The adversarial critique
showed that reading is the wrong one. A point mass is not a *correct* belief: 18.5% of them
name a room the object is not in, and 64% of the entire benchmark's reordering saving comes
from thirteen tasks that have no prior ambiguity at all. What moves an ordering is not
uncertainty, it is **discovering that a confident belief is wrong**.

That predicts a sharp response along an axis nobody swept. `exp_uncertainty` deleted stated
rooms, which adds *ambiguity* - and ambiguity is the low-yield axis, worth +0.24%. This sweeps
the other one: it takes objects whose belief is a confident point mass and moves that point mass
to a room the object is genuinely not in, at rates from none to all of them, leaving the amount
of ambiguity untouched.

If the critique is right, the static-to-gavel gap should climb steeply with the corruption rate
and the ambiguity sweep should stay flat. If the gap stays flat here too, then neither reading is
right and something else is going on - which would be the more valuable result.

    python exp_corrupt.py --stride 5 --out data/exp-corrupt.json
"""

import argparse
import copy
import json
import random
import statistics as st

import gavel
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from search_cost import Beliefs, _room_of
from world_graph import WorldGraph

RATES = (0.0, 0.25, 0.5, 0.75, 1.0)


def confident(graph):
    """Objects the belief is sure about: a single candidate room carrying all the mass."""
    out = []
    for name, record in (graph.get("objects") or {}).items():
        belief = record.get("belief") or {}
        if len(belief) == 1 or (belief and max(belief.values()) > 0.99):
            out.append(name)
    return out


def corrupt(graph, truth, rate, rng, rooms):
    """Move a fraction of the confident beliefs to a room the object is not in.

    Only the belief moves. The world does not change, so the robot is not being asked to do
    anything different - it is being asked to do the same thing while believing something false,
    which is the situation a wrong RSN guess or a misread instruction actually produces.
    """
    out = copy.deepcopy(graph)
    names = confident(out)
    rng.shuffle(names)
    spoiled = 0
    for name in names[:round(rate * len(names))]:
        true_room = _room_of(truth, name)
        wrong = [r for r in rooms if r != true_room]
        if not wrong:
            continue
        pick = rng.choice(wrong)
        record = out["objects"][name]
        record["belief"] = {pick: 1.0}
        record["room"] = pick
        record["candidates"] = [pick] + [r for r in (record.get("candidates") or []) if r != pick]
        spoiled += 1
    return out, spoiled, len(names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--out", default="data/exp-corrupt.json")
    args = parser.parse_args()

    tasks = json.load(open(args.tasks)) if False else json.load(open("data/multitask.json"))
    tasks = tasks[::args.stride][:args.limit]
    rows = []
    for index, task in enumerate(tasks, 1):
        ex = task["extraction"]
        base = populate(task["scene"], ex["uncertain"], ex["dependent"],
                        stated=ex.get("stated") or {},
                        model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
        truth = WorldGraph.from_scene_graph(gavel.seed_graph(task))
        table = gavel._table(task["scene"])
        plans = [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]]
        row = {"id": task["id"], "scene": task["scene"], "n": len(plans), "levels": {}}
        for rate in RATES:
            rng = random.Random(args.seed + index)
            sg, spoiled, total = corrupt(base, truth, rate, rng, table["rooms"])
            g = WorldGraph.from_scene_graph(sg)
            try:
                a = gavel.solve(task, plans, sg, g, reorder=False)
                b = gavel.solve(task, plans, sg, g, reorder=True)
            except Exception as exc:
                row["levels"][str(rate)] = {"error": f"{type(exc).__name__}: {exc}"}
                continue
            row["levels"][str(rate)] = {
                "spoiled": spoiled, "confident": total,
                "static": round(a["walked"], 2), "gavel": round(b["walked"], 2),
                "differ": list(a["order"]) != list(b["order"]),
                "reorders": b["reorders"],
            }
        rows.append(row)
        if index % 10 == 0:
            print(f"  {index}/{len(tasks)}", flush=True)
        json.dump(rows, open(args.out, "w"), indent=1)

    print(f"\n{len(rows)} instructions, corruption sweep\n")
    print(f"  {'corrupted':>10s} {'spoiled':>8s} {'static':>10s} {'gavel':>10s} "
          f"{'gain':>8s} {'orders differ':>14s}")
    for rate in RATES:
        got = [r["levels"].get(str(rate)) for r in rows]
        got = [g for g in got if g and "error" not in g]
        if not got:
            continue
        s = sum(g["static"] for g in got)
        v = sum(g["gavel"] for g in got)
        differ = sum(1 for g in got if g["differ"])
        spoiled = sum(g["spoiled"] for g in got)
        print(f"  {rate*100:9.0f}% {spoiled:8d} {s:10.0f} {v:10.0f} "
              f"{100*(s-v)/max(s,1e-9):+7.2f}% {differ:8d}/{len(got):<5d}")
    print("\n  (`gain` is the paired static->gavel saving on the cost model's meter; "
          "`orders differ` counts tasks where the two arms chose a different order at all)")


if __name__ == "__main__":
    raise SystemExit(main())
