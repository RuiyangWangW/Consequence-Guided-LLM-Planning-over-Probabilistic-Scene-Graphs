#!/usr/bin/env python3
"""How often is the RSN right, and is that why the full pipeline does not beat MAP?

GAVEL and GAVEL-MAP differ in exactly one thing: GAVEL's cost model carries the whole room
distribution and GAVEL-MAP carries only its argmax. If the argmax is nearly always correct,
the distribution behind it has nothing to say - the two arms plan the same order, drive the
same route, and the ablation measures nothing. That is the hypothesis this tests.

For every object the RSN actually guesses about, it reports:

    top-1        the MAP room is the true room. This is the number that decides whether the
                 distribution can matter at all.
    rank         where the true room sits in the ranked candidates. Rank 2 is a near miss the
                 robot recovers from cheaply; rank 5 is a belief that is confidently wrong,
                 and that is the case ordering is supposed to pay for.
    mass         the probability the belief puts on its first choice. A wrong guess held at
                 0.9 costs more than a wrong guess held at 0.3, because nothing warns the
                 planner to hedge.

**Only guessed objects count.** An object whose room the instruction states outright
("the kitchen countertop") is not a guess, and scoring those would report the extractor's
reading comprehension as if it were the RSN's accuracy. The three-class extraction truth is
what separates them: `uncertain` is guessed, `stated` is told, and `dependent` inherits its
room from whatever it rests on.
"""

import argparse
import collections
import json
import sys

from build_tasks import furniture_rooms
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate


def one(task, truth_rooms):
    ex = task["extraction"]
    graph = populate(task["scene"], ex["uncertain"], ex["dependent"],
                     stated=ex.get("stated") or {},
                     model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    guessed = set(ex["uncertain"])          # the objects the RSN is actually asked about
    out = []
    for name in guessed:
        info = (graph.get("objects") or {}).get(name)
        if not info:
            continue
        true_room = truth_rooms.get(name)
        if not true_room:
            continue
        candidates = list(info.get("candidates") or ([info["room"]] if info.get("room") else []))
        rank = candidates.index(true_room) + 1 if true_room in candidates else 0
        out.append({"object": name, "believed": info.get("room"), "true": true_room,
                    "top1": info.get("room") == true_room, "rank": rank,
                    "rooms": len(candidates),
                    "mass": round(float(info.get("probability") or 0.0), 3)})
    return out


def report(rows):
    seen = [r for row in rows for r in row["objects"]]
    if not seen:
        print("nothing guessed")
        return
    top1 = sum(1 for r in seen if r["top1"])
    print(f"\n{len(rows)} tasks, {len(seen)} guessed objects\n")
    print(f"  RSN top-1 accuracy      {top1}/{len(seen)}  ({100*top1/len(seen):.1f}%)")
    ranks = collections.Counter(r["rank"] for r in seen)
    print(f"  rank of the true room   " +
          "  ".join(f"{k if k else 'unranked'}:{v}" for k, v in sorted(ranks.items())))
    wrong = [r for r in seen if not r["top1"]]
    print(f"  rooms per scene (mean candidates) {sum(r['rooms'] for r in seen)/len(seen):.1f}")
    if wrong:
        print(f"\n  when it is wrong ({len(wrong)}):")
        print(f"    mean confidence in the wrong room {sum(r['mass'] for r in wrong)/len(wrong):.2f}")
        print(f"    true room at rank 2 on {sum(1 for r in wrong if r['rank']==2)}, "
              f"rank>=3 on {sum(1 for r in wrong if r['rank']>=3)}, "
              f"unranked on {sum(1 for r in wrong if r['rank']==0)}")
        by = collections.Counter((r["object"], r["believed"], r["true"]) for r in wrong)
        print("    most common misses (object: believed -> true):")
        for (o, b, t), n in by.most_common(12):
            print(f"      {n:3d}x  {o:22s} {b}  ->  {t}")
    right = collections.Counter(r["object"] for r in seen if r["top1"])
    print("\n  objects it always gets right (top 12 by count):")
    for o, n in right.most_common(12):
        print(f"      {n:3d}x  {o}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out")
    ap.add_argument("--analyse")
    args = ap.parse_args()

    if args.analyse:
        report(json.load(open(args.analyse)))
        return 0

    tasks = json.load(open(args.tasks))[::args.stride]
    if args.limit:
        tasks = tasks[:args.limit]
    rooms_cache = {}
    rows = []
    for i, task in enumerate(tasks, 1):
        scene = task["scene"]
        if scene not in rooms_cache:
            rooms_cache[scene] = furniture_rooms(scene)
        rows.append({"id": task["id"], "scene": scene,
                     "objects": one(task, rooms_cache[scene])})
        if i % 25 == 0 or i == len(tasks):
            seen = [r for row in rows for r in row["objects"]]
            got = sum(1 for r in seen if r["top1"])
            print(f"  {i}/{len(tasks)}  top-1 {got}/{len(seen)}", flush=True)
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
