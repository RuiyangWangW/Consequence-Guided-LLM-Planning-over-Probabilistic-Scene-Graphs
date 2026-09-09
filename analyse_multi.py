#!/usr/bin/env python3
"""Read the multi-task results and say what they mean.

Two questions the raw rows do not answer on their own:

  * **Does reordering help, and by how much?** Paired per task, on both meters - the cost
    model the ordering stage optimised, and the distance the simulator actually drove.
  * **Is the cost model measuring the right thing?** The ordering minimises expected cost;
    if that number is uncorrelated with what the robot really drives, the stage is
    optimising a quantity nobody pays, and any saving it reports is luck.

    python analyse_multi.py --oracle data/order-oracle.json --run data/multi-4b.json
"""

import argparse
import json
import math


def correlation(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def _errands(row):
    """How many errands this row's instruction had, whichever shape the file uses."""
    n = row.get("subgoals")
    return n if isinstance(n, int) else len(n or [])


def paired(rows, get, key):
    out = []
    for r in rows:
        a, b = get(r, "static"), get(r, "gavel")
        if a and b and a.get(key) is not None and b.get(key) is not None:
            out.append((r, a, b))
    return out


def report(label, rows, get):
    print(f"\n=== {label}  ({len(rows)} tasks)")
    for key, meter in (("walked", "cost model"), ("driven", "simulator")):
        pairs = paired(rows, get, key)
        if not pairs:
            continue
        s = sum(a[key] for _, a, _ in pairs)
        g = sum(b[key] for _, _, b in pairs)
        better = sum(1 for _, a, b in pairs if b[key] < a[key] - 1e-6)
        worse = sum(1 for _, a, b in pairs if b[key] > a[key] + 1e-6)
        print(f"  by {meter:11s} {len(pairs):3d} tasks: static {s:8.0f} m -> gavel {g:8.0f} m"
              f"   {100*(s-g)/max(s,1e-9):+5.1f}%"
              f"   shorter {better}, longer {worse}, same {len(pairs)-better-worse}")

    # Two-errand instructions are a structural zero for this comparison: there are only two
    # orderings and the executor commits to the first before it can learn anything, so the arms
    # are provably identical on them. Leaving them in is not wrong, but they are a fifth of the
    # benchmark contributing a guaranteed nothing, so the split is printed rather than buried.
    for key, meter in (("walked", "cost model"), ("driven", "simulator")):
        big = [(a, b) for r, a, b in
               ((r, get(r, "static"), get(r, "gavel")) for r in rows)
               if a and b and a.get(key) is not None and b.get(key) is not None
               and _errands(r) > 2]
        if not big:
            continue
        sb = sum(a[key] for a, _ in big)
        gb = sum(b[key] for _, b in big)
        moved = sum(1 for a, b in big if abs(a[key] - b[key]) > 1e-6)
        print(f"  {meter:11s} excluding 2-errand tasks ({len(big):3d}): "
              f"{sb:8.0f} m -> {gb:8.0f} m   {100*(sb-gb)/max(sb,1e-9):+5.1f}%"
              f"   ({moved} not tied)")

    # Does the meter the ordering optimises track the one the robot pays?
    both = [(a, b) for _, a, b in paired(rows, get, "driven")
            if a.get("walked") is not None and b.get("walked") is not None]
    if both:
        xs = [a["walked"] for a, _ in both] + [b["walked"] for _, b in both]
        ys = [a["driven"] for a, _ in both] + [b["driven"] for _, b in both]
        print(f"  cost model vs simulator distance: r = {correlation(xs, ys):+.3f} "
              f"over {len(xs)} runs")
        # And the sharper question: when the model says one ordering is cheaper, is it?
        agree = sum(1 for a, b in both
                    if (b["walked"] - a["walked"]) * (b["driven"] - a["driven"]) > 0)
        moved = sum(1 for a, b in both if abs(b["walked"] - a["walked"]) > 1e-6
                    and abs(b["driven"] - a["driven"]) > 1e-6)
        if moved:
            print(f"  when the model prefers one ordering, the simulator agrees on "
                  f"{agree}/{moved} of the tasks where both meters moved")

    by_n = {}
    for r, a, b in paired(rows, get, "driven"):
        acc = by_n.setdefault(r.get("subgoals") or len(r.get("subgoals", [])), [0.0, 0.0, 0])
        acc[0] += a["driven"]; acc[1] += b["driven"]; acc[2] += 1
    if by_n:
        print("  by number of errands:")
        for n in sorted(by_n):
            x, y, c = by_n[n]
            print(f"    {n} errands ({c:3d} tasks): {x/c:7.1f} m -> {y/c:7.1f} m  "
                  f"{100*(x-y)/max(x,1e-9):+5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle")
    parser.add_argument("--run")
    args = parser.parse_args()

    if args.oracle:
        rows = json.load(open(args.oracle))
        report("ordering on reference subplans (no model)", rows, lambda r, a: r.get(a))
        ok = {a: sum(1 for r in rows if r.get(a, {}).get("ok")) for a in ("static", "gavel")}
        print(f"  reference plans reaching the goal: static {ok['static']}/{len(rows)}, "
              f"gavel {ok['gavel']}/{len(rows)}")
        blocked = sum(r.get("gavel", {}).get("blocked") or 0 for r in rows)
        print(f"  orderings rejected by the composition check: {blocked}")

    if args.run:
        rows = json.load(open(args.run))
        report("the 4B pipeline", rows, lambda r, a: r["arms"].get(a))
        print("\n  goal reached, by arm:")
        for arm in ("monolithic", "static", "gavel"):
            got = [r["arms"][arm] for r in rows if arm in r["arms"]]
            if got:
                print(f"    {arm:11s} {sum(1 for g in got if g.get('ok')):3d}/{len(got)}")
        bad1 = sum(1 for r in rows if r.get("extraction"))
        bad3 = sum(1 for r in rows if r.get("grounding"))
        print(f"\n  stage-1 extraction errors {bad1}/{len(rows)}, "
              f"stage-3 grounding errors {bad3}/{len(rows)}")
        # Which errands failed, and why - the question "are these failures reasonable?"
        stuck = [(r["id"], e) for r in rows
                 for e in r["arms"].get("static", {}).get("errands", [])
                 if isinstance(e, dict) and not e.get("accepted")]
        print(f"  errands the validation loop never accepted: {len(stuck)}")
        for tid, e in stuck[:12]:
            print(f"    {tid:22s} {e['errand'][:52]:52s} {e.get('why','')[:60]}")


if __name__ == "__main__":
    raise SystemExit(main())
