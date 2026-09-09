#!/usr/bin/env python3
"""Does uncertainty restore GAVEL's gain?  A causal test of the "nothing to learn" story.

GAVEL re-optimises the errand order at every subplan boundary; the `static` arm optimises
once against the prior and then follows that order to the end.  On the 500-task benchmark
the two differ by well under a percent, and 429/500 tasks are bit-identical.  The leading
explanation is that there is nothing to re-optimise *against*: the instructions say where
things are ("the kitchen countertop", "the mug in the office cabinet"), so 83.4% of task
objects already have <=1 candidate room and sweeping a room teaches the robot nothing it
did not already believe.

That is a hypothesis about the benchmark, and it is testable by *removing the knowledge*
and re-measuring.  Four belief levels, same tasks, same reference subplans, same executor:

    full      the extraction as the benchmark ships it
    half      half the `stated` object->room facts deleted
    none      every `stated` fact deleted - the RSN has to guess every room
    none+dep  that, and every stated *relation* deleted too - no anchors at all

If the static-vs-gavel gap grows monotonically as knowledge is removed, the small gain is
a property of this benchmark's instructions and not a defect of the method.  If the gap
stays flat even at `none+dep`, the hypothesis is wrong and the cause is elsewhere.

**Deleting a fact means demoting it, not dropping the object.**  `populate` builds its
object list from `uncertain + stated + dependent-roots`, so passing `stated={}` without
doing anything else does not make the robot *uncertain* about the floor lamp - it makes
the floor lamp vanish from the belief graph entirely, `Beliefs.prior` has no entry for it,
`walk` finds no candidate rooms to sweep and drives straight to it for free.  That is
*less* uncertainty, not more.  So every fact this script removes has its object moved into
the `uncertain` list, where the RSN must rank it over every room in the house.

    python exp_uncertainty.py --stride 4          # 125 tasks, four levels, three arms
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import math
import random
import statistics
import sys
import time

# ---------------------------------------------------------------- speed, not semantics
# `scene_graph.populate` asks the RSN about every object, `query_rsn.predict_rooms` calls
# `embed_categories.embed_names`, and that constructs a fresh SentenceTransformer *per
# object*.  Four belief levels over a hundred-odd tasks is several thousand encoder loads
# of the same few hundred names.  Both functions are deterministic in their arguments
# (`model.eval()`, no sampling), so memoising them changes nothing but the clock.  Done by
# rebinding the module attributes, because `populate` imports them inside its own body -
# no file in the repo is edited.
import query_rsn

_RSN = {}
_PROBS = {}
_load = query_rsn.load
_predict = query_rsn.predict_rooms


def _cached_load(path, device):
    key = (path, str(device))
    if key not in _RSN:
        _RSN[key] = _load(path, device)
    return _RSN[key]


def _cached_predict(model, ckpt, name, device):
    if name not in _PROBS:
        _PROBS[name] = _predict(model, ckpt, name, device)
    return _PROBS[name]


query_rsn.load = _cached_load
query_rsn.predict_rooms = _cached_predict

import gavel                                    # noqa: E402
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate   # noqa: E402
from search_cost import Beliefs                 # noqa: E402
from world_graph import WorldGraph              # noqa: E402

LEVELS = ("full", "half", "none", "none+dep")
SEED = 20250908


def degrade(extraction, level, seed):
    """The extraction as it would read if the instruction had said less.

    Returns the `(uncertain, dependent, stated)` triple `populate` wants.  Whatever is
    deleted is added to `uncertain`, so the object is still in the graph and still needed -
    the robot simply has to guess where it is instead of being told.

    `half` deletes ceil(n/2) of the stated facts, so a task that states exactly one room
    genuinely loses it; with floor(n/2) the single-fact tasks - a third of them - would sit
    at `full` under a label saying `half`.
    """
    uncertain = list(extraction.get("uncertain") or [])
    dependent = [dict(d) for d in (extraction.get("dependent") or [])]
    stated = dict(extraction.get("stated") or {})

    if level == "full":
        return uncertain, dependent, stated

    if level == "half":
        keys = sorted(stated)
        rng = random.Random(seed)
        dropped = set(rng.sample(keys, (len(keys) + 1) // 2)) if keys else set()
        kept = {k: v for k, v in stated.items() if k not in dropped}
        return uncertain + sorted(dropped), dependent, kept

    # `none` and `none+dep`: every stated room is gone.
    uncertain = uncertain + sorted(stated)
    if level == "none":
        return uncertain, dependent, {}

    # `none+dep`: the relations go too, so a dependent object is no longer anchored to its
    # support and has to be found on its own RSN ranking.  Both ends of every relation are
    # promoted to `uncertain`, because a support that was only in the graph as somebody
    # else's root would otherwise disappear along with the relation.
    for d in dependent:
        uncertain.append(d["object"])
        uncertain.append(d["target"])
    return list(dict.fromkeys(uncertain)), [], {}


def known_fraction(scene_graph_dict, rooms):
    """Fraction of task objects the belief is already certain about: <=1 candidate room.

    This is the same quantity as the 83.4% the benchmark reports, read off the object the
    ordering stage actually consumes - `search_cost.Beliefs.prior` - rather than off the
    extraction, so a stated room the scene does not have is counted as the guess it becomes.
    """
    prior = Beliefs(scene_graph_dict, rooms).prior
    if not prior:
        return None, 0
    certain = sum(1 for d in prior.values() if len(d) <= 1)
    return certain / len(prior), len(prior)


def run_level(task, plans, goal, level, seed):
    table = gavel._table(task["scene"])
    uncertain, dependent, stated = degrade(task["extraction"], level, seed)
    sg = populate(task["scene"], uncertain, dependent, stated=stated,
                  model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    g = WorldGraph.from_scene_graph(sg)
    known, n_objects = known_fraction(sg, table["rooms"])

    n = len(plans)
    arms = {
        "instr": gavel.solve(task, plans, sg, g, reorder=False, force=tuple(range(n))),
        "static": gavel.solve(task, plans, sg, g, reorder=False),
        "gavel": gavel.solve(task, plans, sg, g, reorder=True),
    }
    return {"level": level, "known": known, "objects": n_objects,
            "arms": {k: {"walked": v["walked"], "order": list(v["order"]),
                         "estimated": v["estimated"], "reorders": v["reorders"],
                         "blocked": v["blocked"]}
                     for k, v in arms.items()}}


def run_oracle_level(task, plans, level, seed):
    """Every ordering of this task, walked, at this belief level.

    The static-vs-gavel gap is bounded above by how much room there is left: if the order
    chosen against the prior is already the one that turns out cheapest in hindsight, no
    online information can improve on it, whatever the belief knows.  `force` makes the
    executor follow a permutation and optimise nothing, so walking all n! of them gives the
    best and worst any ordering policy could have done on this task at this level, and
    locates `static` and `gavel` inside that band.
    """
    import itertools

    gavel._table(task["scene"])
    uncertain, dependent, stated = degrade(task["extraction"], level, seed)
    sg = populate(task["scene"], uncertain, dependent, stated=stated,
                  model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    g = WorldGraph.from_scene_graph(sg)
    n = len(plans)
    walks = {}
    for perm in itertools.permutations(range(n)):
        walks[perm] = gavel.solve(task, plans, sg, g, reorder=False, force=perm)["walked"]
    st = gavel.solve(task, plans, sg, g, reorder=False)
    gv = gavel.solve(task, plans, sg, g, reorder=True)
    return {"level": level,
            "walks": {",".join(map(str, k)): v for k, v in walks.items()},
            "static": {"walked": st["walked"], "order": list(st["order"])},
            "gavel": {"walked": gv["walked"], "order": list(gv["order"])}}


def oracle(tasks, levels, out, seed0):
    rows, started = [], time.time()
    for i, t in enumerate(tasks):
        plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
        row = {"id": t["id"], "scene": t["scene"], "n": len(plans), "levels": {}}
        for level in levels:
            row["levels"][level] = run_oracle_level(t, plans, level, seed0 + i)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  oracle {i+1}/{len(tasks)}  {time.time()-started:.0f}s", flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}", flush=True)
    oracle_report(rows, levels)
    return rows


def oracle_report(rows, levels):
    print()
    print("=" * 96)
    print("How much room is there to improve?  every ordering walked, per belief level "
          "(n>=3 tasks only)")
    print("=" * 96)
    head = (f"{'level':9s} {'tasks':>5s} {'spread%':>8s} {'static gap%':>11s} "
            f"{'gavel gap%':>10s} {'static=best':>11s} {'gavel=best':>10s} "
            f"{'gavel<static':>12s}")
    print(head)
    print("-" * len(head))
    for level in levels:
        spread, sgap, ggap, sbest, gbest, better, used = [], [], [], 0, 0, 0, 0
        for r in rows:
            if r["n"] < 3:
                continue
            cell = r["levels"][level]
            vals = [v for v in cell["walks"].values() if finite(v)]
            if not vals or min(vals) <= 0 or len(vals) != math.factorial(r["n"]):
                continue
            lo, hi = min(vals), max(vals)
            st, gv = cell["static"]["walked"], cell["gavel"]["walked"]
            if not (finite(st) and finite(gv)):
                continue
            used += 1
            spread.append(100.0 * (hi - lo) / lo)
            sgap.append(100.0 * (st - lo) / lo)
            ggap.append(100.0 * (gv - lo) / lo)
            sbest += st <= lo + 1e-6
            gbest += gv <= lo + 1e-6
            better += gv < st - 1e-9
        if not used:
            continue
        print(f"{level:9s} {used:5d} {statistics.median(spread):7.1f}% "
              f"{statistics.median(sgap):10.1f}% {statistics.median(ggap):9.1f}% "
              f"{100.0*sbest/used:10.1f}% {100.0*gbest/used:9.1f}% "
              f"{100.0*better/used:11.1f}%")
    print("   spread% = (worst ordering - best ordering) / best ordering, per task")
    print("   gap%    = how far above the best ordering that arm landed, per task")


def sim_phase(tasks, levels, out, seed0):
    """The same comparison, judged by the 2-D simulator instead of the cost model.

    `walked` is what the executor's own cost model charges - A* between room centroids plus
    a modelled sweep.  The headline number this experiment is testing ("+0.6%") is the
    simulator's `driven`, which is what the robot's wheels actually turn through while a
    wedge camera finds things.  The two can disagree, so the trend across belief levels is
    re-measured on the metric the headline uses, on the same tasks, paired.
    """
    from sim_eval import run_plan

    rows, started = [], time.time()
    for i, t in enumerate(tasks):
        plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
        goal = [tuple(x) for x in t["goal"]]
        row = {"id": t["id"], "scene": t["scene"], "n": len(plans), "levels": {}}
        for level in levels:
            gavel._table(t["scene"])
            uncertain, dependent, stated = degrade(t["extraction"], level, seed0 + i)
            sg = populate(t["scene"], uncertain, dependent, stated=stated,
                          model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
            g = WorldGraph.from_scene_graph(sg)
            cell = {}
            for arm, kw in (("static", dict(reorder=False)), ("gavel", dict(reorder=True))):
                r = gavel.solve(t, plans, sg, g, **kw)
                steps, _ = gavel.compose(g, [plans[k] for k in r["order"]], goal)
                sim = run_plan(t, sg, steps, verbose=False)
                cell[arm] = {"walked": r["walked"], "order": list(r["order"]),
                             "ok": bool(sim.get("ok")), "driven": float(sim.get("driven") or 0.0),
                             "why": sim.get("why")}
            row["levels"][level] = cell
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  sim {i+1}/{len(tasks)}  {time.time()-started:.0f}s", flush=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {out}", flush=True)
    sim_report(rows, levels)
    return rows


def sim_report(rows, levels):
    print()
    print("=" * 100)
    print("SIMULATOR-DRIVEN distance, paired, by belief level")
    print("=" * 100)
    head = (f"{'level':9s} {'both ok':>8s} {'dropped':>8s} {'sum static':>10s} "
            f"{'sum gavel':>10s} {'sum delta%':>10s} {'med delta%':>10s} "
            f"{'wins':>5s} {'loss':>5s} {'ties':>5s}")
    print(head)
    print("-" * len(head))
    for level in levels:
        usable = [r for r in rows
                  if r["levels"][level]["static"]["ok"] and r["levels"][level]["gavel"]["ok"]]
        s = [r["levels"][level]["static"]["driven"] for r in usable]
        gv = [r["levels"][level]["gavel"]["driven"] for r in usable]
        pct = [100.0 * (a - b) / a for a, b in zip(s, gv) if a > 0]
        wins = sum(1 for a, b in zip(s, gv) if b < a - 1e-9)
        loss = sum(1 for a, b in zip(s, gv) if b > a + 1e-9)
        print(f"{level:9s} {len(usable):8d} {len(rows)-len(usable):8d} {sum(s):10.0f} "
              f"{sum(gv):10.0f} {100.0*(sum(s)-sum(gv))/sum(s):9.2f}% "
              f"{statistics.median(pct):9.2f}% {wins:5d} {loss:5d} "
              f"{len(usable)-wins-loss:5d}")
    print()
    print("Plans the simulator could not finish (dropped above), by scene:")
    for level in levels:
        bad = {}
        for r in rows:
            for arm in ("static", "gavel"):
                if not r["levels"][level][arm]["ok"]:
                    bad[r["scene"]] = bad.get(r["scene"], 0) + 1
        print(f"   {level:9s} {sorted(bad.items()) or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=4, help="use tasks[::stride]")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/exp-uncertainty.json")
    ap.add_argument("--levels", nargs="+", default=list(LEVELS))
    ap.add_argument("--report", action="store_true",
                    help="re-print the tables from --out without re-running anything")
    ap.add_argument("--oracle", action="store_true",
                    help="walk every ordering of every task, to bound what any policy "
                         "could have gained at each belief level")
    ap.add_argument("--oracle-out", default="data/exp-uncertainty-oracle.json")
    ap.add_argument("--sim", action="store_true",
                    help="judge the same comparison by the 2-D simulator's driven distance")
    ap.add_argument("--sim-out", default="data/exp-uncertainty-sim.json")
    args = ap.parse_args()

    if args.report:
        return report(json.load(open(args.out)), args.levels)

    tasks = json.load(open(args.tasks))[:: args.stride]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(tasks)} tasks (stride {args.stride}), levels {args.levels}", flush=True)

    if args.oracle:
        return oracle(tasks, args.levels, args.oracle_out, SEED)

    if args.sim:
        return sim_phase(tasks, args.levels, args.sim_out, SEED)

    rows, started = [], time.time()
    for i, t in enumerate(tasks):
        plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
        goal = [tuple(x) for x in t["goal"]]
        row = {"id": t["id"], "scene": t["scene"], "n": len(plans), "levels": {}}
        for level in args.levels:
            row["levels"][level] = run_level(t, plans, goal, level, SEED + i)
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(tasks)}  {time.time()-started:.0f}s", flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"wrote {args.out}", flush=True)
    report(rows, args.levels)


# ------------------------------------------------------------------------ reporting

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def report(rows, levels):
    print()
    print("=" * 92)
    print("PAIRED static-vs-gavel by belief level   (walked = distance the executor pays "
          "in the truth graph)")
    print("=" * 92)
    header = (f"{'level':9s} {'tasks':>5s} {'known%':>7s} {'obj/task':>8s} "
              f"{'med static':>10s} {'med gavel':>10s} {'sum delta%':>10s} "
              f"{'med delta%':>10s} {'!=order':>7s} {'wins':>5s} {'loss':>5s} {'reord':>6s}")
    print(header)
    print("-" * len(header))
    summary = {}
    for level in levels:
        usable = [r for r in rows
                  if finite(r["levels"][level]["arms"]["static"]["walked"])
                  and finite(r["levels"][level]["arms"]["gavel"]["walked"])]
        s = [r["levels"][level]["arms"]["static"]["walked"] for r in usable]
        gv = [r["levels"][level]["arms"]["gavel"]["walked"] for r in usable]
        known = [r["levels"][level]["known"] for r in usable if r["levels"][level]["known"] is not None]
        objs = [r["levels"][level]["objects"] for r in usable]
        deltas = [a - b for a, b in zip(s, gv)]                     # >0 means gavel cheaper
        pct = [100.0 * (a - b) / a for a, b in zip(s, gv) if a > 0]
        diff_order = sum(1 for r in usable
                         if r["levels"][level]["arms"]["static"]["order"]
                         != r["levels"][level]["arms"]["gavel"]["order"])
        wins = sum(1 for d in deltas if d > 1e-9)
        loss = sum(1 for d in deltas if d < -1e-9)
        reord = sum(r["levels"][level]["arms"]["gavel"]["reorders"] for r in usable)
        summary[level] = {
            "tasks": len(usable), "known": statistics.mean(known) if known else float("nan"),
            "objects": statistics.mean(objs) if objs else float("nan"),
            "med_static": statistics.median(s) if s else float("nan"),
            "med_gavel": statistics.median(gv) if gv else float("nan"),
            "sum_pct": 100.0 * (sum(s) - sum(gv)) / sum(s) if sum(s) else float("nan"),
            "med_pct": statistics.median(pct) if pct else float("nan"),
            "diff_order": diff_order, "wins": wins, "loss": loss, "reorders": reord,
        }
        d = summary[level]
        print(f"{level:9s} {d['tasks']:5d} {100*d['known']:6.1f}% {d['objects']:8.1f} "
              f"{d['med_static']:10.1f} {d['med_gavel']:10.1f} {d['sum_pct']:9.2f}% "
              f"{d['med_pct']:9.2f}% {d['diff_order']:7d} {d['wins']:5d} {d['loss']:5d} "
              f"{d['reorders']:6d}")

    print()
    print("Does ordering matter at all?  instruction-order baseline vs the two optimisers")
    header2 = (f"{'level':9s} {'med instr':>10s} {'instr->static':>14s} "
               f"{'instr->gavel':>13s}")
    print(header2)
    print("-" * len(header2))
    for level in levels:
        usable = [r for r in rows
                  if all(finite(r["levels"][level]["arms"][a]["walked"])
                         for a in ("instr", "static", "gavel"))]
        it = [r["levels"][level]["arms"]["instr"]["walked"] for r in usable]
        s = [r["levels"][level]["arms"]["static"]["walked"] for r in usable]
        gv = [r["levels"][level]["arms"]["gavel"]["walked"] for r in usable]
        f1 = 100.0 * (sum(it) - sum(s)) / sum(it) if sum(it) else float("nan")
        f2 = 100.0 * (sum(it) - sum(gv)) / sum(it) if sum(it) else float("nan")
        print(f"{level:9s} {statistics.median(it):10.1f} {f1:13.2f}% {f2:12.2f}%")

    # Wainscott_0_int is in two disconnected pieces; its numbers are not comparable to the
    # rest and are reported apart rather than averaged in.
    print()
    for tag, keep in (("excluding Wainscott_0_int", lambda r: r["scene"] != "Wainscott_0_int"),
                      ("Wainscott_0_int only", lambda r: r["scene"] == "Wainscott_0_int")):
        subset = [r for r in rows if keep(r)]
        if not subset:
            continue
        print(f"{tag}  ({len(subset)} tasks)")
        for level in levels:
            usable = [r for r in subset
                      if finite(r["levels"][level]["arms"]["static"]["walked"])
                      and finite(r["levels"][level]["arms"]["gavel"]["walked"])]
            if not usable:
                continue
            s = [r["levels"][level]["arms"]["static"]["walked"] for r in usable]
            gv = [r["levels"][level]["arms"]["gavel"]["walked"] for r in usable]
            diff = sum(1 for r in usable
                       if r["levels"][level]["arms"]["static"]["order"]
                       != r["levels"][level]["arms"]["gavel"]["order"])
            print(f"   {level:9s} n={len(usable):4d}  sum delta "
                  f"{100.0*(sum(s)-sum(gv))/sum(s):6.2f}%   orders differ {diff:3d}")
    print()
    nonfinite = [(r["id"], lvl, arm)
                 for r in rows for lvl in levels for arm in ("instr", "static", "gavel")
                 if not finite(r["levels"][lvl]["arms"][arm]["walked"])]
    print(f"non-finite walked cells: {len(nonfinite)}"
          + (f"  e.g. {nonfinite[:3]}" if nonfinite else ""))

    # How many errands there are is not a nuisance variable, it is a hard ceiling. With two
    # errands there is exactly one ordering decision and both arms make it at the same
    # boundary from the same prior, so `static` and `gavel` are identical *by construction* -
    # no amount of removed knowledge can separate them. Anything the levels do must happen
    # in the n>=3 tasks, so the gap is reported per errand count as well as overall.
    print()
    print("By errand count - only n>=3 can ever differ (n=2 has one decision, made "
          "identically by both arms)")
    counts = sorted({r["n"] for r in rows})
    head3 = f"{'level':9s} " + " ".join(f"{'n='+str(c):>18s}" for c in counts)
    print(head3)
    print("-" * len(head3))
    for level in levels:
        cells = []
        for c in counts:
            usable = [r for r in rows if r["n"] == c
                      and finite(r["levels"][level]["arms"]["static"]["walked"])
                      and finite(r["levels"][level]["arms"]["gavel"]["walked"])]
            s = [r["levels"][level]["arms"]["static"]["walked"] for r in usable]
            gv = [r["levels"][level]["arms"]["gavel"]["walked"] for r in usable]
            diff = sum(1 for r in usable
                       if r["levels"][level]["arms"]["static"]["order"]
                       != r["levels"][level]["arms"]["gavel"]["order"])
            pct = 100.0 * (sum(s) - sum(gv)) / sum(s) if sum(s) else float("nan")
            cells.append(f"{pct:6.2f}% {diff:3d}/{len(usable):<3d}")
        print(f"{level:9s} " + " ".join(f"{c:>18s}" for c in cells))
    print("   (cell = sum delta%, then tasks whose static/gavel orders differ / tasks)")

    print()
    print("Where does the belief actually change?  reorders fired per task, by level")
    for level in levels:
        fired = [r["levels"][level]["arms"]["gavel"]["reorders"] for r in rows if r["n"] >= 3]
        blocked = sum(r["levels"][level]["arms"]["gavel"]["blocked"] for r in rows)
        nz = sum(1 for x in fired if x)
        print(f"   {level:9s} n>=3 tasks {len(fired):4d}   with >=1 reorder {nz:4d} "
              f"({100.0*nz/max(len(fired),1):5.1f}%)   total reorders {sum(fired):4d}   "
              f"compose-blocked {blocked}")


if __name__ == "__main__":
    sys.exit(main())
