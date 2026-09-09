#!/usr/bin/env python3
"""How many of the benchmark's own reference plans finish in the 2-D simulator?

Every plan in `data/tasks.json` was validated by the graph machine when the benchmark was
built, so symbolically all 100 succeed. Running them with a camera and a floor is a
different question, and the gap between the two numbers is the **ceiling** for anything
measured in simulation: a task whose known-good plan cannot be executed is not a task an
LLM can be marked down for failing.

    python run_reference_sim.py --out data/reference-sim.json
"""

import os as _os, sys as _sys
# Run from anywhere. Find the repo root by marker, put it on the import path, and make
# it the working directory - every path in this file is written 'data/...', so without
# the chdir a build invoked from inside its own folder would quietly write benchmark/data/.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, 'src')):
    _d = _os.path.dirname(_d)
_roots = [_d, _os.path.join(_d, 'omnigibson_runtime')]
_roots += [_f.path for _r in ('src', 'benchmark')
           for _f in _os.scandir(_os.path.join(_d, _r))
           if _f.is_dir() and not _f.name.startswith(('.', '_'))]
for _p in _roots:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
_os.chdir(_d)


import argparse
import json
import time

from build_tasks import seed_graph
from sim_eval import run_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/tasks.json")
    parser.add_argument("--out", default="data/reference-sim.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))[:args.limit]
    rows, started = [], time.time()
    for i, task in enumerate(tasks, 1):
        began = time.time()
        try:
            verdict = run_plan(task, seed_graph(task), task["plan"], verbose=False)
        except Exception as exc:                       # a harness fault, not a plan fault
            verdict = {"ok": False, "why": f"{type(exc).__name__}: {exc}",
                       "driven": 0.0, "steps_run": 0, "error": True}
        verdict["id"] = task["id"]
        verdict["scene"] = task["scene"]
        rows.append(verdict)
        print(f"{i:3d}/{len(tasks)} {task['id']:22s} "
              f"{'ok  ' if verdict['ok'] else 'FAIL'} {verdict['driven']:6.1f} m "
              f"({time.time() - began:4.1f}s)  {verdict['why'][:60]}", flush=True)
        with open(args.out, "w") as f:
            json.dump({"complete": i == len(tasks), "rows": rows}, f, indent=1)

    ok = sum(r["ok"] for r in rows)
    print(f"\n{ok}/{len(rows)} reference plans finish in simulation "
          f"({time.time() - started:.0f}s)")
    print("\nwhy the rest do not:")
    for row in rows:
        if not row["ok"]:
            print(f"   {row['id']:22s} {row['why'][:88]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
