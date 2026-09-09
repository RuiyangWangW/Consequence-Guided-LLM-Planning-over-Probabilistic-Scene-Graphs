#!/usr/bin/env python3
"""Does spreading the errands across rooms give the ordering stage more to win?

The ordering objective is J(order) = sum(internal) + sum(transitions). Only the transition
term depends on the order, so the most an ordering stage can ever save is bounded by how big
the transitions are. If every errand is in the same corner of the house, transitions are tiny
and no ordering policy can beat any other.

This measures, per instruction, three things and correlates them:

    spread_rooms   how many distinct rooms the errands touch
    spread_metres  mean pairwise A* distance between the rooms the errands act in
    headroom       (worst ordering - best ordering) / worst, over all n! orderings

If headroom rises with spread, then building the benchmark to spread errands around the house
is the right fix and the current +0.6% is a property of the task mix rather than the method.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import itertools
import json
import statistics as st

import cost_matrix
import gavel
import order as ordering
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from search_cost import Beliefs, _room_of
from world_graph import WorldGraph


def rooms_acted_in(task, truth):
    """The rooms each errand actually acts in, from the truth graph."""
    out = []
    for sub in task["subgoals"]:
        rs = set()
        for _, obj in sub["plan"]:
            if obj:
                room = _room_of(truth, obj)
                if room:
                    rs.add(room)
        out.append(rs)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--limit", type=int, default=170)
    parser.add_argument("--out", default="data/spread-headroom.json")
    args = parser.parse_args()

    tasks = json.load(open("data/multitask.json"))[::args.stride][:args.limit]
    rows = []
    for index, task in enumerate(tasks, 1):
        ex = task["extraction"]
        sg = populate(task["scene"], ex["uncertain"], ex["dependent"],
                      stated=ex.get("stated") or {},
                      model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
        g = WorldGraph.from_scene_graph(sg)
        truth = WorldGraph.from_scene_graph(gavel.seed_graph(task))
        table = gavel._table(task["scene"])
        D = lambda a, b: table["distance"].get(f"{a}|{b}", float("inf"))
        S = lambda r: table["search"].get(r, 0.0)
        plans = [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]]
        n = len(plans)

        A, head = ordering.pairwise(plans, g, lambda: Beliefs(sg, table["rooms"]), D, S)
        costs = [ordering.score(p, A, head) for p in itertools.permutations(range(n))]
        finite = [c for c in costs if c != float("inf")]
        if len(finite) < 2:
            continue
        best, worst = min(finite), max(finite)
        headroom = (worst - best) / worst if worst > 0 else 0.0

        rs = rooms_acted_in(task, truth)
        allrooms = set().union(*rs) if rs else set()
        pairs = [(a, b) for i, a in enumerate(rs) for b in rs[i + 1:]]
        gaps = []
        for a, b in pairs:
            ds = [D(x, y) for x in a for y in b if D(x, y) != float("inf")]
            if ds:
                gaps.append(min(ds))          # how far apart the two errands are at closest
        rows.append({
            "id": task["id"], "scene": task["scene"], "n": n,
            "rooms": len(allrooms),
            "metres": st.mean(gaps) if gaps else 0.0,
            "best": best, "worst": worst, "headroom": headroom,
            "infinite_orderings": len(costs) - len(finite),
        })
        if index % 25 == 0:
            print(f"  {index}/{len(tasks)}", flush=True)

    json.dump(rows, open(args.out, "w"), indent=1)

    def corr(xs, ys):
        mx, my = st.mean(xs), st.mean(ys)
        sx = sum((x - mx) ** 2 for x in xs) ** 0.5
        sy = sum((y - my) ** 2 for y in ys) ** 0.5
        return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy) if sx and sy else float("nan")

    print(f"\n{len(rows)} instructions\n")
    print(f"  headroom (worst->best, % of worst): mean {st.mean(r['headroom'] for r in rows)*100:5.1f}%  "
          f"median {st.median(r['headroom'] for r in rows)*100:5.1f}%")
    print(f"  distinct rooms touched : mean {st.mean(r['rooms'] for r in rows):.2f}")
    print(f"  mean errand separation : {st.mean(r['metres'] for r in rows):.1f} m")
    print(f"\n  correlation headroom vs distinct rooms   r = {corr([r['rooms'] for r in rows], [r['headroom'] for r in rows]):+.3f}")
    print(f"  correlation headroom vs errand separation r = {corr([r['metres'] for r in rows], [r['headroom'] for r in rows]):+.3f}")

    print("\n  headroom by how far apart the errands are:")
    rows.sort(key=lambda r: r["metres"])
    q = max(len(rows) // 4, 1)
    for name, chunk in (("closest 25%", rows[:q]), ("2nd", rows[q:2*q]),
                        ("3rd", rows[2*q:3*q]), ("furthest 25%", rows[3*q:])):
        if chunk:
            print(f"    {name:14s} separation {st.mean(r['metres'] for r in chunk):5.1f} m  "
                  f"rooms {st.mean(r['rooms'] for r in chunk):4.2f}  "
                  f"headroom {st.mean(r['headroom'] for r in chunk)*100:5.1f}%")

    print("\n  headroom by distinct rooms touched:")
    by = {}
    for r in rows:
        by.setdefault(r["rooms"], []).append(r["headroom"])
    for k in sorted(by):
        print(f"    {k} rooms ({len(by[k]):3d} tasks): headroom {st.mean(by[k])*100:5.1f}%")


if __name__ == "__main__":
    raise SystemExit(main())
