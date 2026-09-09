#!/usr/bin/env python3
"""Merge sharded `evaluate_multi` results and summarise them as one run.

The work is split across GPUs by dealing the selected tasks round-robin, so each shard holds
a different slice of the same sample. Merging is a concatenation; the summary is the one
`evaluate_multi` would have printed had it run in a single process.
"""

import argparse
import glob
import json

import evaluate_multi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="data/multi-4b-shard*.json")
    p.add_argument("--out", default="data/multi-4b.json")
    args = p.parse_args()

    # `evaluate_multi` writes a `-stamp.json` beside each output, and `shard*.json` matches
    # it too. Skip those - and check them: they are a content hash of the task file each
    # shard actually read, so two shards run against different generations of the benchmark
    # would otherwise merge silently into one table. That has happened.
    paths = [p for p in sorted(glob.glob(args.pattern)) if not p.endswith("-stamp.json")]
    stamps = {}
    for path in paths:
        side = path.replace(".json", "-stamp.json")
        try:
            stamps[path] = json.load(open(side))
        except FileNotFoundError:
            pass
    distinct = {json.dumps(v, sort_keys=True) for v in stamps.values()}
    if len(distinct) > 1:
        raise SystemExit(f"shards disagree about the benchmark they ran against:\n" +
                         "\n".join(f"  {p}  {json.dumps(v, sort_keys=True)[:120]}"
                                   for p, v in stamps.items()))

    rows, seen = [], set()
    for path in paths:
        for r in json.load(open(path)):
            if r["id"] in seen:
                continue          # a shard re-run overlapping another; keep the first
            seen.add(r["id"])
            rows.append(r)
    rows.sort(key=lambda r: r["id"])
    json.dump(rows, open(args.out, "w"), indent=1)
    if stamps:
        json.dump(next(iter(stamps.values())),
                  open(args.out.replace(".json", "-stamp.json"), "w"), indent=1)
    print(f"merged {len(rows)} instructions from {len(paths)} shards -> {args.out}"
          f"   benchmark stamp {'ok, all shards agree' if len(distinct) == 1 else 'missing'}")
    evaluate_multi.summarise(rows)


if __name__ == "__main__":
    raise SystemExit(main())
