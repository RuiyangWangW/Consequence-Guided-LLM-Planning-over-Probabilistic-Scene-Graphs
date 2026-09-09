#!/usr/bin/env python3
"""Re-run the belief-grounded check over both benchmarks after the grounding fix.

The benchmark verifies itself by driving each reference plan against the *truth*
(`build_tasks.verify`, `run_reference_sim.py`). Every real evaluation drives against a
*belief*, and `sim_eval.ground` binds a category name to an instance using that belief. A task
can therefore be verified and still be impossible: the grounding picks a physical object the
task never meant, and no plan can succeed. This drives every reference plan both ways and
reports the gap.
"""

import argparse
import json

from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from build_tasks import seed_graph
from sim_eval import run_plan


def check(task):
    ex = task["extraction"]
    belief = populate(task["scene"], ex["uncertain"], ex["dependent"],
                      stated=ex.get("stated") or {},
                      model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    out = {}
    for label, graph in (("truth", seed_graph(task)), ("belief", belief)):
        try:
            r = run_plan(task, graph, task["plan"], verbose=False)
            out[label] = {"ok": r["ok"], "why": r["why"][:120]}
        except Exception as exc:
            out[label] = {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sets", default="single,subs,multi")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", default="data/resweep.json")
    args = p.parse_args()

    sources = {"single": ("data/tasks.json", None), "subs": ("data/subtasks.json", None),
               "multi": ("data/multitask.json", None)}
    rows = []
    for name in args.sets.split(","):
        path, _ = sources[name]
        tasks = json.load(open(path))[::args.stride]
        for i, t in enumerate(tasks, 1):
            r = check(t)
            rows.append({"id": t["id"], "set": name, "scene": t["scene"], **r})
            if i % 50 == 0:
                print(f"  {name} {i}/{len(tasks)}", flush=True)
            json.dump(rows, open(args.out, "w"), indent=1)

    print(f"\n{len(rows)} reference plans driven both ways\n")
    for name in args.sets.split(","):
        got = [r for r in rows if r["set"] == name]
        bad = [r for r in got if r["truth"]["ok"] and not r["belief"]["ok"]]
        broke = [r for r in got if not r["truth"]["ok"]]
        print(f"  {name:7s} {len(got):4d} tasks | truth-ok but belief-FAILS: {len(bad)}"
              f" | truth itself fails: {len(broke)}")
        for r in bad[:10]:
            print(f"      {r['id']:24s} {r['belief']['why'][:88]}")
        for r in broke[:5]:
            print(f"      BROKEN {r['id']:24s} {r['truth']['why'][:80]}")


if __name__ == "__main__":
    raise SystemExit(main())
