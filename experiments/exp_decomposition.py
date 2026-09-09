#!/usr/bin/env python3
"""How much of a multi-errand journey can ordering possibly move?

GAVEL's ordering objective decomposes.  `order.pairwise` returns `(A, head)` where
`head[j]` is what subplan `j` costs run first from the robot's start and `A[i][j]` is
what `j` costs run from wherever `i` left the robot.  Every ordering pays exactly one
term per subplan - `head[j]` if `j` is first, `A[pred(j)][j]` otherwise - so

    J(sigma) = sum_j c_j(pred(j)),     c_j(None) = head[j],  c_j(i) = A[i][j]

**The separation of "internal" from "transition" is exact, not a heuristic.**  Read
`search_cost.rollout`: the running position `where` is overwritten by the belief of the
first `NAVIGATE_TO` target (`where = {known: 1.0} if known else dict(dist)`), and that
belief comes from a factory-fresh `Beliefs`, never from `start`.  The `GraphMachine`
steps do not see `start` either.  So everything after the first `NAVIGATE_TO` is a
constant of the subplan alone:

    rollout(j, start=s) = expected_nav(s, first_nav_target(j)) + R_j

with `R_j` independent of `s`.  Therefore

    internal   R_j        = head[j] - expected_nav(start, first_nav_target(j))
    approach   T(i -> j)  = A[i][j] - R_j = expected_nav(end_i, first_nav_target(j))
    J(sigma)   = sum_j R_j            <- order-INVARIANT, identical for all n! orderings
               + sum_j T(pred(j), j)  <- order-DEPENDENT, the only part in play

The script recomputes `expected_nav` itself and then *verifies* the identity against
every one of the n^2 - n entries of `A`, reporting the worst residual.  If that residual
is zero the split is arithmetic fact rather than modelling choice.

Note what `R_j` contains: it is NOT manipulation time.  It is all the walking and
sweeping *inside* an errand - the leg from the mug to the dishwasher, the sweep for the
second object - which the robot must cover in every ordering.  And note what the
approach term contains: the drive to the first candidate room *plus* the expected sweep
of it, so even the "movable" part is not fully movable.

Three cost scales are measured, deliberately, because they answer different questions:

    estimated   J from the prior-time pairwise matrix - what the ordering stage optimises
    walked      gavel's executor run against ground truth with `force=<permutation>`
    driven      the 2-D simulator's true distance, on a subsample

Usage:
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 python exp_decomposition.py \
        --stride 1 --sim-stride 5 --out logs/decomposition.json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import itertools
import json
import math
import os
import statistics
import sys
import time

import numpy as np

INF = float("inf")


# --------------------------------------------------------------------------------------
# Speed only.  `embed_categories.embed_names` rebuilds a SentenceTransformer on every
# call and `query_rsn.load` re-reads the checkpoint on every call, which costs ~6 s per
# task and dominates a 500-task pass.  Both are pure functions of their arguments, so
# memoising them in *this process* changes no number anywhere.  No module file is edited.
# --------------------------------------------------------------------------------------
def _memoise_rsn():
    import embed_categories
    import query_rsn

    encoders = {}

    def embed_names(names, model_name=embed_categories.DEFAULT_MODEL, batch_size=64):
        from sentence_transformers import SentenceTransformer

        encoder = encoders.get(model_name)
        if encoder is None:
            encoder = encoders[model_name] = SentenceTransformer(model_name)
        texts = [embed_categories.name_to_text(n) for n in names]
        vecs = encoder.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    embed_categories.embed_names = embed_names

    loaded, real_load = {}, query_rsn.load

    def load(path, device):
        key = (path, str(device))
        if key not in loaded:
            loaded[key] = real_load(path, device)
        return loaded[key]

    query_rsn.load = load

    cache, real_predict = {}, query_rsn.predict_rooms

    def predict_rooms(model, ckpt, name, device):
        key = (id(model), name, str(device))
        if key not in cache:
            cache[key] = real_predict(model, ckpt, name, device)
        return cache[key]

    query_rsn.predict_rooms = predict_rooms


# --------------------------------------------------------------------------------------
# Per-task measurement
# --------------------------------------------------------------------------------------
def first_nav(plan):
    """The target of the first `NAVIGATE_TO` - the only step whose cost sees `start`."""
    for action, arg in plan:
        if action == "NAVIGATE_TO" and arg:
            return arg
    return None


def score(order, A, head):
    total = head[order[0]]
    for a, b in zip(order, order[1:]):
        total += A[a][b]
    return total


def scene_size(table):
    """Room count and the spread of the house, in metres, over reachable pairs only."""
    finite = [v for v in table["distance"].values() if v != INF and v > 0]
    return {"rooms": len(table["rooms"]),
            "median_room_distance": statistics.median(finite) if finite else None,
            "max_room_distance": max(finite) if finite else None,
            "disconnected_pairs": sum(1 for v in table["distance"].values() if v == INF)}


def measure(task, tasks_index, want_sim, sim_mode="all"):
    import gavel
    import order as ordering
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
    from search_cost import Beliefs, _room_of, expected_nav, rollout
    from world_graph import WorldGraph

    table, distance, search = gavel._costs(task["scene"])
    ex = task["extraction"]
    sg = populate(task["scene"], ex["uncertain"], ex["dependent"],
                  stated=ex.get("stated") or {}, model_path=DEFAULT_MODEL,
                  threshold=DEFAULT_THRESHOLD)
    graph = WorldGraph.from_scene_graph(sg)
    plans = [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]]
    goal = [tuple(x) for x in task["goal"]]
    n = len(plans)
    make = lambda: Beliefs(sg, table["rooms"])

    row = {"id": task.get("id", tasks_index), "index": tasks_index, "scene": task["scene"],
           "n": n, **scene_size(table)}

    # ---- the prior-time pairwise matrix, exactly as the ordering stage builds it -------
    A, head = ordering.pairwise(plans, graph, make, distance, search, start=None)
    ends = []
    for plan in plans:
        _, end, _ = rollout(plan, graph, make(), distance, search, start=None)
        ends.append(end)

    room = _room_of(graph, "robot")
    origin = ({room: 1.0} if room
              else {r: 1.0 / len(table["rooms"]) for r in table["rooms"]})
    firsts = [first_nav(p) for p in plans]

    def approach(where, j):
        if firsts[j] is None:
            return 0.0
        return expected_nav(where, firsts[j], make(), distance, search)

    head_approach = [approach(origin, j) for j in range(n)]
    internal = [head[j] - head_approach[j] for j in range(n)]

    # ---- verify the split against every off-diagonal entry of A -----------------------
    residual, bad = 0.0, False
    trans = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            trans[i][j] = approach(ends[i], j)
            predicted = trans[i][j] + internal[j]
            if predicted == INF and A[i][j] == INF:
                continue
            if math.isnan(predicted) or math.isinf(predicted) != math.isinf(A[i][j]):
                bad = True
                continue
            residual = max(residual, abs(predicted - A[i][j]))
    row["residual"] = residual
    row["degenerate"] = bool(bad or any(math.isnan(x) or x < -1e-9 for x in internal))
    row["internal_total"] = None if row["degenerate"] else sum(internal)

    # ---- every ordering, scored on the estimated objective ---------------------------
    perms = list(itertools.permutations(range(n)))
    est = {p: score(p, A, head) for p in perms}
    finite = {p: v for p, v in est.items() if v != INF and not math.isnan(v)}
    row["n_orders"] = len(perms)
    row["n_orders_finite_est"] = len(finite)
    if finite:
        best_p = min(finite, key=finite.get)
        worst_p = max(finite, key=finite.get)
        row["est_best"] = finite[best_p]
        row["est_worst"] = finite[worst_p]
        row["est_mean"] = statistics.fmean(finite.values())
        row["est_best_order"] = list(best_p)
        row["est_worst_order"] = list(worst_p)
        identity = tuple(range(n))
        row["est_identity"] = finite.get(identity, est.get(identity))
        # The floor: pay every errand from its cheapest possible predecessor.  A lower
        # bound on J that no permutation need attain (the argmins can conflict).
        floor = 0.0
        for j in range(n):
            options = [head_approach[j]] + [trans[i][j] for i in range(n) if i != j]
            options = [v for v in options if v != INF]
            floor += (min(options) if options else INF) + (internal[j] if not bad else 0.0)
        row["est_floor"] = floor
    else:
        row["est_best"] = row["est_worst"] = row["est_mean"] = None
        row["est_identity"] = row["est_floor"] = None
        row["est_best_order"] = row["est_worst_order"] = None

    # ---- what the executor really walks, for every ordering --------------------------
    walked = {}
    for perm in perms:
        try:
            walked[perm] = gavel.solve(task, plans, sg, graph, reorder=False,
                                       force=perm)["walked"]
        except Exception as exc:                      # noqa: BLE001 - recorded, not hidden
            row.setdefault("errors", []).append(f"force {perm}: {exc}")
    row["walked_all"] = {",".join(map(str, p)): v for p, v in walked.items()}
    row["est_all"] = {",".join(map(str, p)): v for p, v in est.items() if v != INF}
    wfinite = {p: v for p, v in walked.items() if v != INF and not math.isnan(v)}
    row["n_orders_finite_walk"] = len(wfinite)
    if wfinite:
        wbest = min(wfinite, key=wfinite.get)
        wworst = max(wfinite, key=wfinite.get)
        row["walk_best"] = wfinite[wbest]
        row["walk_worst"] = wfinite[wworst]
        row["walk_mean"] = statistics.fmean(wfinite.values())
        row["walk_best_order"] = list(wbest)
        row["walk_identity"] = walked.get(tuple(range(n)))
    else:
        row["walk_best"] = row["walk_worst"] = row["walk_mean"] = None
        row["walk_best_order"] = row["walk_identity"] = None

    # ---- where the two shipped arms land inside that envelope ------------------------
    static = gavel.solve(task, plans, sg, graph, reorder=False)
    online = gavel.solve(task, plans, sg, graph, reorder=True)
    row["static_order"] = list(static["order"])
    row["gavel_order"] = list(online["order"])
    row["static_walked"] = static["walked"]
    row["gavel_walked"] = online["walked"]
    row["gavel_reorders"] = online["reorders"]
    if wfinite:
        ranked = sorted(wfinite.values())
        for tag, value in (("static", static["walked"]), ("gavel", online["walked"])):
            if value == INF or math.isnan(value):
                row[f"{tag}_percentile"] = None
                continue
            below = sum(1 for v in ranked if v < value - 1e-9)
            row[f"{tag}_percentile"] = below / max(len(ranked) - 1, 1)

    # ---- the simulator ---------------------------------------------------------------
    # `walked` is the executor's own cost model.  The simulator is the arbiter, and the two
    # need not rank orderings the same way - so on the subsample every ordering is
    # simulated and the driven-distance envelope is measured directly rather than inherited
    # from `walked`.
    if want_sim:
        from sim_eval import run_plan

        if sim_mode == "all":
            todo = list(perms)
        else:
            todo = sorted({tuple(range(n)), tuple(static["order"]), tuple(online["order"])}
                          | ({wbest, wworst} if wfinite else set()))
        sims = {}
        for perm in todo:
            steps, _ = gavel.compose(graph, [plans[i] for i in perm], goal)
            try:
                out = run_plan(task, sg, steps, verbose=False)
                got = {"ok": bool(out["ok"]), "driven": out["driven"], "why": out.get("why")}
            except Exception as exc:                  # noqa: BLE001
                got = {"ok": False, "driven": None, "why": f"exception: {type(exc).__name__}"}
            sims[",".join(map(str, perm))] = got
        row["sims"] = sims
        row["sim_mode"] = sim_mode
        row["sim_identity"] = sims.get(",".join(map(str, range(n))))
        row["sim_static"] = sims.get(",".join(map(str, static["order"])))
        row["sim_gavel"] = sims.get(",".join(map(str, online["order"])))
        if wfinite:
            row["sim_walk_best"] = sims.get(",".join(map(str, wbest)))
            row["sim_walk_worst"] = sims.get(",".join(map(str, wworst)))
    return row


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------
def quantiles(values):
    values = sorted(v for v in values if v is not None and not math.isnan(v) and v != INF)
    if not values:
        return None
    def q(p):
        if len(values) == 1:
            return values[0]
        pos = p * (len(values) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (pos - lo)
    return {"n": len(values), "min": values[0], "p25": q(0.25), "median": q(0.50),
            "p75": q(0.75), "p90": q(0.90), "max": values[-1],
            "mean": statistics.fmean(values)}


def line(label, stats, unit=""):
    if not stats:
        return f"  {label:34s}  (no data)"
    return (f"  {label:34s} n={stats['n']:4d}  med {stats['median']:8.2f}{unit}"
            f"  p25 {stats['p25']:7.2f}  p75 {stats['p75']:7.2f}"
            f"  p90 {stats['p90']:7.2f}  max {stats['max']:8.2f}  mean {stats['mean']:7.2f}")


def ratio(num, den):
    if num is None or den is None or den in (0, INF) or math.isnan(den) or num == INF:
        return None
    return num / den


def report(rows, stride, sim_stride, elapsed):
    out = []
    P = out.append

    P("=" * 96)
    P("HOW MUCH OF THE JOURNEY CAN ORDERING MOVE?  cost decomposition over multitask.json")
    P("=" * 96)
    P(f"tasks measured: {len(rows)}   stride={stride}   sim-stride={sim_stride}"
      f"   wall {elapsed/60:.1f} min")

    # --- exactness of the split ---
    res = [r["residual"] for r in rows if r["residual"] is not None]
    degen = [r for r in rows if r["degenerate"]]
    P("")
    P("-- 1. IS THE SPLIT EXACT? ------------------------------------------------------")
    P("   claim: A[i][j] == expected_nav(end_i, first_nav_target(j)) + R_j, R_j constant")
    P(f"   worst residual over all {sum(r['n']*(r['n']-1) for r in rows)} off-diagonal"
      f" entries: {max(res) if res else float('nan'):.3e} m")
    P(f"   tasks where the split could not be formed (inf/NaN): {len(degen)}")

    ok = [r for r in rows if not r["degenerate"] and r["est_best"] not in (None, INF)]
    dropped_inf = [r for r in rows if r["est_best"] in (None, INF)]
    P("")
    P("-- 2. THE ORDER-INVARIANT SHARE (estimated objective, prior-time matrix) --------")
    P(f"   tasks usable: {len(ok)}   dropped for infinite/absent best ordering:"
      f" {len(dropped_inf)}"
      + (f"  ({', '.join(sorted({r['scene'] for r in dropped_inf}))})" if dropped_inf else ""))
    P(line("internal / J(best order)",
           quantiles([ratio(r["internal_total"], r["est_best"]) for r in ok])))
    P(line("internal / J(worst order)",
           quantiles([ratio(r["internal_total"], r["est_worst"]) for r in ok])))
    P(line("approach share of J(best)  [in play]",
           quantiles([ratio(r["est_best"] - r["internal_total"], r["est_best"]) for r in ok])))
    P(line("J(best) internal, m", quantiles([r["internal_total"] for r in ok]), " m"))
    P(line("J(best) approach, m",
           quantiles([r["est_best"] - r["internal_total"] for r in ok]), " m"))

    P("")
    P("-- 3. BEST vs WORST ORDERING (estimated objective) ------------------------------")
    P(line("spread, absolute m",
           quantiles([r["est_worst"] - r["est_best"] for r in ok]), " m"))
    P(line("spread / J(worst)   [max saving %]",
           quantiles([ratio(r["est_worst"] - r["est_best"], r["est_worst"]) for r in ok])))
    P(line("(J_mean - J_best)/J_mean  [vs random]",
           quantiles([ratio(r["est_mean"] - r["est_best"], r["est_mean"]) for r in ok])))
    P(line("(J_id - J_best)/J_id  [vs instruction]",
           quantiles([ratio((r["est_identity"] or INF) - r["est_best"], r["est_identity"])
                      for r in ok if r["est_identity"] not in (None, INF)])))

    # --- executed cost ---
    wok = [r for r in rows if r["walk_best"] not in (None, INF)
           and r["walk_worst"] not in (None, INF)]
    P("")
    P("-- 4. TRUE EXECUTED COST (gavel executor vs ground truth, every ordering) -------")
    P(f"   tasks with a finite walked cost for at least one ordering: {len(wok)}")
    P(line("walk_best, m", quantiles([r["walk_best"] for r in wok]), " m"))
    P(line("worst-best spread, m",
           quantiles([r["walk_worst"] - r["walk_best"] for r in wok]), " m"))
    P(line("spread / walk_worst [max saving %]",
           quantiles([ratio(r["walk_worst"] - r["walk_best"], r["walk_worst"]) for r in wok])))
    P(line("(id - best)/id   [vs instruction order]",
           quantiles([ratio((r["walk_identity"] or INF) - r["walk_best"], r["walk_identity"])
                      for r in wok if r["walk_identity"] not in (None, INF)])))
    P(line("(mean - best)/mean [vs random order]",
           quantiles([ratio(r["walk_mean"] - r["walk_best"], r["walk_mean"]) for r in wok])))

    P("")
    P("-- 5. WHERE THE SHIPPED ARMS LAND INSIDE THAT ENVELOPE -------------------------")
    P("   headroom = what a hindsight-perfect ordering would still save over the arm")
    P(line("(static - best)/static  [headroom]",
           quantiles([ratio(r["static_walked"] - r["walk_best"], r["static_walked"])
                      for r in wok if r["static_walked"] not in (None, INF)])))
    P(line("(gavel  - best)/gavel   [headroom]",
           quantiles([ratio(r["gavel_walked"] - r["walk_best"], r["gavel_walked"])
                      for r in wok if r["gavel_walked"] not in (None, INF)])))
    P(line("(static - gavel)/static [measured gain]",
           quantiles([ratio(r["static_walked"] - r["gavel_walked"], r["static_walked"])
                      for r in wok if r["static_walked"] not in (None, INF)])))
    exact = [r for r in wok if r["static_walked"] is not None
             and r["walk_best"] is not None
             and abs(r["static_walked"] - r["walk_best"]) < 1e-6]
    P(f"   static ordering is ALREADY the hindsight-optimal one: {len(exact)}/{len(wok)}"
      f" = {100*len(exact)/max(len(wok),1):.1f}%")
    exg = [r for r in wok if r["gavel_walked"] is not None and r["walk_best"] is not None
           and abs(r["gavel_walked"] - r["walk_best"]) < 1e-6]
    P(f"   gavel  ordering is ALREADY the hindsight-optimal one: {len(exg)}/{len(wok)}"
      f" = {100*len(exg)/max(len(wok),1):.1f}%")
    pct = quantiles([r.get("static_percentile") for r in wok])
    P(line("static's percentile among all orderings", pct))

    # --- totals, the paired aggregate ---
    P("")
    P("-- 6. PAIRED TOTALS OVER THE USABLE TASKS (sum of metres) ----------------------")
    def total(key):
        return sum(r[key] for r in wok if r.get(key) not in (None, INF))
    complete = [r for r in wok if all(r.get(k) not in (None, INF) for k in
                ("walk_best", "walk_worst", "walk_identity", "static_walked", "gavel_walked",
                 "walk_mean"))]
    P(f"   tasks with all five quantities finite: {len(complete)}")
    if complete:
        s = {k: sum(r[k] for r in complete) for k in
             ("walk_best", "walk_worst", "walk_identity", "walk_mean", "static_walked",
              "gavel_walked")}
        for k, v in s.items():
            P(f"     {k:16s} {v:10.1f} m")
        P(f"     perfect ordering vs instruction order : "
          f"{100*(s['walk_identity']-s['walk_best'])/s['walk_identity']:5.2f} %")
        P(f"     perfect ordering vs random ordering   : "
          f"{100*(s['walk_mean']-s['walk_best'])/s['walk_mean']:5.2f} %")
        P(f"     perfect ordering vs static (prior-opt): "
          f"{100*(s['static_walked']-s['walk_best'])/s['static_walked']:5.2f} %")
        P(f"     gavel vs static (the measured gain)   : "
          f"{100*(s['static_walked']-s['gavel_walked'])/s['static_walked']:5.2f} %")
        P(f"     worst ordering vs best (the envelope) : "
          f"{100*(s['walk_worst']-s['walk_best'])/s['walk_worst']:5.2f} %")

    # --- scaling ---
    P("")
    P("-- 7. SCALING WITH NUMBER OF ERRANDS -------------------------------------------")
    P(f"   {'n':>2}  {'tasks':>5}  {'orders':>6}  {'internal/J_best':>15}"
      f"  {'spread/J_worst':>14}  {'walkspread/worst':>16}  {'static headroom':>15}")
    for n in sorted({r["n"] for r in rows}):
        grp = [r for r in ok if r["n"] == n]
        wgrp = [r for r in wok if r["n"] == n]
        a = quantiles([ratio(r["internal_total"], r["est_best"]) for r in grp])
        b = quantiles([ratio(r["est_worst"] - r["est_best"], r["est_worst"]) for r in grp])
        c = quantiles([ratio(r["walk_worst"] - r["walk_best"], r["walk_worst"]) for r in wgrp])
        d = quantiles([ratio(r["static_walked"] - r["walk_best"], r["static_walked"])
                       for r in wgrp if r["static_walked"] not in (None, INF)])
        fmt = lambda s: f"{100*s['median']:13.1f} %" if s else "        n/a  "
        P(f"   {n:>2}  {len(grp):>5}  {math.factorial(n):>6}  {fmt(a):>15}"
          f"  {fmt(b):>14}  {fmt(c):>16}  {fmt(d):>15}")

    P("")
    P("-- 8. SCALING WITH SCENE SIZE ---------------------------------------------------")
    P(f"   {'scene':22s} {'rooms':>5} {'medD':>6} {'tasks':>5} {'internal/J_best':>15}"
      f" {'spread/J_worst':>14} {'walkspread/worst':>16}")
    for scene in sorted({r["scene"] for r in rows}):
        grp = [r for r in ok if r["scene"] == scene]
        wgrp = [r for r in wok if r["scene"] == scene]
        any_row = next(r for r in rows if r["scene"] == scene)
        a = quantiles([ratio(r["internal_total"], r["est_best"]) for r in grp])
        b = quantiles([ratio(r["est_worst"] - r["est_best"], r["est_worst"]) for r in grp])
        c = quantiles([ratio(r["walk_worst"] - r["walk_best"], r["walk_worst"]) for r in wgrp])
        fmt = lambda s: f"{100*s['median']:13.1f} %" if s else "        n/a  "
        md = any_row["median_room_distance"]
        P(f"   {scene:22s} {any_row['rooms']:>5} {md if md is None else round(md,1):>6}"
          f" {len(grp):>5} {fmt(a):>15} {fmt(b):>14} {fmt(c):>16}")

    # --- simulator ---
    sims = [r for r in rows if r.get("sims")]
    if sims:
        P("")
        P("-- 9. SIMULATOR-DRIVEN DISTANCE, EVERY ORDERING ENUMERATED ---------------------")
        P(f"   tasks simulated: {len(sims)}   mode={sims[0].get('sim_mode')}"
          f"   sim runs: {sum(len(r['sims']) for r in sims)}")
        allfail = [r for r in sims if not any(v["ok"] for v in r["sims"].values())]
        partial = [r for r in sims
                   if any(v["ok"] for v in r["sims"].values())
                   and not all(v["ok"] for v in r["sims"].values())]
        P(f"   tasks where NO ordering finished in the simulator: {len(allfail)}"
          + (f"  ({', '.join(sorted({r['scene'] for r in allfail}))})" if allfail else ""))
        P(f"   tasks where some but not all orderings finished:   {len(partial)}"
          + (f"  ({', '.join(sorted({r['scene'] for r in partial}))})" if partial else ""))

        # Only tasks where EVERY ordering finished give an honest envelope: if a subset
        # failed, the surviving min is not the minimum over orderings.
        full = [r for r in sims if r["sims"] and all(v["ok"] and v["driven"] is not None
                                                     for v in r["sims"].values())]
        P(f"   tasks where EVERY ordering finished (used below): {len(full)}")
        if full:
            for r in full:
                dv = {k: v["driven"] for k, v in r["sims"].items()}
                r["_dbest"] = min(dv.values())
                r["_dworst"] = max(dv.values())
                r["_dmean"] = statistics.fmean(dv.values())
                r["_did"] = dv.get(",".join(map(str, range(r["n"]))))
                r["_dstatic"] = dv.get(",".join(map(str, r["static_order"])))
                r["_dgavel"] = dv.get(",".join(map(str, r["gavel_order"])))
                ranked = sorted(dv.values())
                r["_dstatic_pct"] = (sum(1 for v in ranked if v < r["_dstatic"] - 1e-9)
                                     / max(len(ranked) - 1, 1))
            P(line("driven best, m", quantiles([r["_dbest"] for r in full]), " m"))
            P(line("(worst-best)/worst driven [envelope]",
                   quantiles([ratio(r["_dworst"] - r["_dbest"], r["_dworst"]) for r in full])))
            P(line("(identity-best)/identity driven",
                   quantiles([ratio(r["_did"] - r["_dbest"], r["_did"]) for r in full])))
            P(line("(mean-best)/mean driven [vs random]",
                   quantiles([ratio(r["_dmean"] - r["_dbest"], r["_dmean"]) for r in full])))
            P(line("(static-best)/static driven [headroom]",
                   quantiles([ratio(r["_dstatic"] - r["_dbest"], r["_dstatic"]) for r in full])))
            P(line("(gavel-best)/gavel driven [headroom]",
                   quantiles([ratio(r["_dgavel"] - r["_dbest"], r["_dgavel"]) for r in full])))
            P(line("(static-gavel)/static driven [gain]",
                   quantiles([ratio(r["_dstatic"] - r["_dgavel"], r["_dstatic"]) for r in full])))
            P(line("static's percentile among driven orderings",
                   quantiles([r["_dstatic_pct"] for r in full])))
            tot = {k: sum(r[f"_d{k}"] for r in full)
                   for k in ("best", "worst", "mean", "id", "static", "gavel")}
            P("   paired totals over those tasks, metres driven:")
            for k, v in tot.items():
                P(f"     {k:8s} {v:9.1f} m")
            P(f"     perfect vs instruction order : "
              f"{100*(tot['id']-tot['best'])/tot['id']:5.2f} %")
            P(f"     perfect vs random ordering   : "
              f"{100*(tot['mean']-tot['best'])/tot['mean']:5.2f} %")
            P(f"     perfect vs static            : "
              f"{100*(tot['static']-tot['best'])/tot['static']:5.2f} %")
            P(f"     gavel vs static              : "
              f"{100*(tot['static']-tot['gavel'])/tot['static']:5.2f} %")
            P(f"     worst vs best (the envelope) : "
              f"{100*(tot['worst']-tot['best'])/tot['worst']:5.2f} %")
            hit = sum(1 for r in full if abs(r["_dstatic"] - r["_dbest"]) < 1e-6)
            P(f"   static ordering IS the driven-optimal ordering: {hit}/{len(full)}"
              f" = {100*hit/len(full):.1f}%")

            # Does the estimator rank orderings the way the simulator does?  If it does
            # not, the ordering stage is optimising a proxy and even a perfect search over
            # its own objective cannot reach the driven optimum.
            def kendall(xs, ys):
                conc = disc = 0
                for a in range(len(xs)):
                    for b in range(a + 1, len(xs)):
                        dx, dy = xs[a] - xs[b], ys[a] - ys[b]
                        if dx == 0 or dy == 0:
                            continue
                        conc += (dx > 0) == (dy > 0)
                        disc += (dx > 0) != (dy > 0)
                return (conc - disc) / (conc + disc) if conc + disc else None

            for label, key in (("estimated J", "est_all"), ("executor walked", "walked_all")):
                taus = []
                for r in full:
                    src = r.get(key) or {}
                    keys = [k for k in r["sims"] if k in src
                            and src[k] not in (None, INF)]
                    if len(keys) < 2:
                        continue
                    t = kendall([src[k] for k in keys],
                                [r["sims"][k]["driven"] for k in keys])
                    if t is not None:
                        taus.append(t)
                P(line(f"Kendall tau({label} , driven)", quantiles(taus)))
    out.extend(representability(rows))
    out.extend(extras(rows))
    P("=" * 96)
    return "\n".join(out)


def representability(rows):
    """Can the pairwise FORM represent the true executed cost at all?

    `J(sigma) = head[s0] + sum A[s_k-1][s_k]` has `n + n(n-1)` free parameters against
    `n!` orderings.  At n=4 that is 16 unknowns for 24 equations and at n=5, 25 for 120,
    so a least-squares fit is a real test rather than an interpolation.  A high residual
    would mean no pairwise matrix, however well estimated, could rank orderings correctly;
    a low one puts the blame on the *numbers* in the matrix instead of its shape.

    The estimated objective is fitted too, as a control: by construction it must come back
    at exactly zero residual, which is what proves the fit is measuring what it claims.
    """
    out = ["", "-- 10. IS THE PAIRWISE FORM ADEQUATE?  (least squares over all n! orders) --"]
    designs = {}
    for n in (4, 5):
        perms = list(itertools.permutations(range(n)))
        cols = [("h", j) for j in range(n)] + [("a", i, j) for i in range(n)
                                               for j in range(n) if i != j]
        index = {c: k for k, c in enumerate(cols)}
        X = np.zeros((len(perms), len(cols)))
        for r, p in enumerate(perms):
            X[r, index[("h", p[0])]] = 1
            for a, b in zip(p, p[1:]):
                X[r, index[("a", a, b)]] = 1
        designs[n] = (perms, X)

    for key, label in (("est_all", "estimated J   (control, must be 0)"),
                       ("walked_all", "executor walked (true cost)")):
        for n in (4, 5):
            perms, X = designs[n]
            rel, r2 = [], []
            for row in rows:
                if row["n"] != n:
                    continue
                src = row.get(key) or {}
                y = []
                for p in perms:
                    v = src.get(",".join(map(str, p)))
                    if v is None or v == INF or (isinstance(v, float) and math.isnan(v)):
                        y = None
                        break
                    y.append(v)
                if not y:
                    continue
                y = np.asarray(y, dtype=float)
                if y.max() - y.min() < 1e-9:
                    continue
                beta, *_ = np.linalg.lstsq(X, y, rcond=None)
                res = y - X @ beta
                rel.append(float(np.sqrt((res ** 2).mean())) / (y.max() - y.min()))
                ss = float(((y - y.mean()) ** 2).sum())
                r2.append(1 - float((res ** 2).sum()) / ss if ss > 0 else 1.0)
            if rel:
                out.append(f"   {label:36s} n={n}  tasks={len(rel):3d}"
                           f"  median RMS residual / spread = "
                           f"{100*statistics.median(rel):5.2f} %"
                           f"   median R^2 = {statistics.median(r2):.4f}")
    return out


def extras(rows):
    """Diagnostics that need the saved rows rather than the per-task loop."""
    out = ["", "-- 11. DIAGNOSTICS -------------------------------------------------------------"]
    ok = [r for r in rows if not r["degenerate"] and r.get("est_best") not in (None, INF)]
    tight = [ratio(r["est_best"] - r["est_floor"], r["est_best"]) for r in ok
             if r.get("est_floor") not in (None, INF)]
    s = quantiles(tight)
    if s:
        out.append(f"   (J_best - per-errand floor)/J_best: median {100*s['median']:.2f} %"
                   f"  p90 {100*s['p90']:.2f} %   (how close the best real ordering gets"
                   f" to paying every errand from its cheapest possible predecessor)")
    pool = [r for r in ok if r.get("walk_best_order") is not None]
    agree = [r for r in pool if r["est_best_order"] == r["walk_best_order"]]
    out.append(f"   argmin(estimated J) == argmin(executed walked): {len(agree)}/{len(pool)}"
               f" = {100*len(agree)/max(len(pool),1):.1f} %")
    both = [r for r in rows if r.get("static_order") and r.get("gavel_order")]
    same = sum(1 for r in both if r["static_order"] == r["gavel_order"])
    out.append(f"   static order == gavel order: {same}/{len(both)}"
               f" = {100*same/max(len(both),1):.1f} %")
    never = sum(1 for r in rows if not r.get("gavel_reorders"))
    out.append(f"   tasks where gavel never revised the schedule: {never}/{len(rows)}"
               f" = {100*never/max(len(rows),1):.1f} %")
    for tag, key in (("static headroom", "static_walked"), ("gavel headroom", "gavel_walked")):
        vals = [r[key] - r["walk_best"] for r in rows
                if r.get(key) not in (None, INF) and r.get("walk_best") not in (None, INF)]
        s = quantiles(vals)
        if s:
            out.append(f"   {tag} in metres walked: median {s['median']:.2f}"
                       f"  p90 {s['p90']:.2f}  mean {s['mean']:.2f}")
    gain = [r["static_walked"] - r["gavel_walked"] for r in rows
            if r.get("static_walked") not in (None, INF)
            and r.get("gavel_walked") not in (None, INF)]
    s = quantiles(gain)
    if s:
        out.append(f"   gavel's actual gain in metres walked: median {s['median']:.2f}"
                   f"  p90 {s['p90']:.2f}  mean {s['mean']:.2f}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyse", help="re-report from a saved results JSON and exit")
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=1, help="use tasks[::stride]")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sim-stride", type=int, default=5,
                    help="simulate every k-th measured task; 0 disables the simulator")
    ap.add_argument("--sim-mode", choices=("all", "picks"), default="all",
                    help="'all' simulates every ordering (true driven envelope); "
                         "'picks' only the five orderings of interest")
    ap.add_argument("--out", default="logs/decomposition.json")
    ap.add_argument("--report", default="logs/decomposition.txt")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    if args.analyse:
        saved = json.load(open(args.analyse))
        text = report(saved["rows"], saved.get("stride", 1), saved.get("sim_stride", 0), 0.0)
        print(text)
        if args.report:
            with open(args.report, "w") as handle:
                handle.write(text + "\n")
            print(f"wrote {args.report}")
        return 0

    _memoise_rsn()

    tasks = json.load(open(args.tasks))
    chosen = list(enumerate(tasks))[::args.stride]
    if args.limit:
        chosen = chosen[:args.limit]

    started, rows = time.time(), []
    for k, (index, task) in enumerate(chosen):
        want_sim = bool(args.sim_stride) and (k % args.sim_stride == 0)
        try:
            rows.append(measure(task, index, want_sim, args.sim_mode))
        except Exception as exc:                       # noqa: BLE001
            print(f"[{k}] task {index} FAILED: {type(exc).__name__}: {exc}", flush=True)
            continue
        if k % 20 == 0 or k == len(chosen) - 1:
            print(f"[{k+1}/{len(chosen)}] task {index} {task['scene']} n={rows[-1]['n']}"
                  f"  {time.time()-started:.0f}s", flush=True)

    text = report(rows, args.stride, args.sim_stride, time.time() - started)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump({"stride": args.stride, "sim_stride": args.sim_stride,
                       "rows": rows}, handle, indent=1, default=str)
        print(f"\nwrote {args.out}")
    if args.report:
        os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
        with open(args.report, "w") as handle:
            handle.write(text + "\n")
        print(f"wrote {args.report}")


if __name__ == "__main__":
    raise SystemExit(main())
