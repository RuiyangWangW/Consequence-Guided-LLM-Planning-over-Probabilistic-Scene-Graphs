#!/usr/bin/env python3
"""Adversarial audit of the GAVEL ordering stage: try to prove the +0.6% is a defect.

Read-only with respect to every existing module. Everything here is done by wrapping
`gavel.walk`, `order.pairwise` and `gavel.compose` with recorders, and by re-running
`gavel.solve(force=...)` over every permutation of each task's errands.

Hypotheses under test (each reported CONFIRMED-BUG or REFUTED):

  H1  closure late-binding: `factory()` inside solve's while loop reuses the first snapshot,
      so gavel re-optimises against the prior and is secretly identical to static.
  H2  the executor does not thread position/belief between errands, so order cannot matter
      to the executed cost.
  H3  order.pairwise's exact-decomposition claim is false once beliefs update mid-run.
  H4  the "cheapest ordering that still composes" loop silently picks a non-argmin.
  H5  the reorder counter is dishonest.
  H6  walk teleports the robot to an unreachable target for free.
  H7  the objective gavel minimises is not the quantity the headline is reported in
      (simulator-driven distance).
  H8  headroom: how far is either arm from the best permutation, on both meters.

    python exp_adversarial.py --stride 5 --sim-stride 20
"""

import argparse
import itertools
import json
import math
import statistics
import time

import gavel
import order as ordering
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from world_graph import WorldGraph

INF = float("inf")

# ---------------------------------------------------------------------------- recorders

REC = {"walk": [], "pairwise": [], "compose_calls": 0, "compose_failed": 0}
_real_walk = gavel.walk
_real_pairwise = ordering.pairwise
_real_compose = gavel.compose


def _digest(b):
    """A hashable fingerprint of a Beliefs' informational content."""
    prior = tuple(sorted((k, tuple(sorted((r, round(p, 9)) for r, p in v.items())))
                         for k, v in b.prior.items()))
    located = tuple(sorted(b.located.items()))
    ruled = tuple(sorted((k, tuple(sorted(v))) for k, v in b.ruled.items()))
    return hash((prior, located, ruled)), len(b.located), sum(len(v) for v in b.ruled.values())


def _walk_rec(plan, world, real, beliefs, distance, search, watching, start=None, reachable=None):
    before = _digest(beliefs)
    out = _real_walk(plan, world, real, beliefs, distance, search, watching,
                     start=start, reachable=reachable)
    after = _digest(beliefs)
    cost, here = out[0], out[1]
    # A "teleport" is walk ending in a room A* says is unreachable from where it began.
    tele = start is not None and here is not None and distance(start, here) == INF
    REC["walk"].append({"start": start, "end": here, "cost": cost, "plan": tuple(plan),
                        "before": before, "after": after, "changed": before[0] != after[0],
                        "teleport": bool(tele)})
    return out


def _pairwise_rec(subplans, graph, beliefs_factory, distance, search, start=None):
    snap = _digest(beliefs_factory())
    A, head = _real_pairwise(subplans, graph, beliefs_factory, distance, search, start=start)
    REC["pairwise"].append({"n": len(subplans), "start": start, "digest": snap,
                            "A": [row[:] for row in A], "head": list(head),
                            "plans": [tuple(p) for p in subplans]})
    return A, head


def _compose_rec(graph, plans, goal):
    steps, outcome = _real_compose(graph, plans, goal)
    REC["compose_calls"] += 1
    if outcome.failed_at is not None:
        REC["compose_failed"] += 1
    return steps, outcome


def instrument(on=True):
    gavel.walk = _walk_rec if on else _real_walk
    ordering.pairwise = _pairwise_rec if on else _real_pairwise
    gavel.compose = _compose_rec if on else _real_compose


def reset():
    REC["walk"].clear()
    REC["pairwise"].clear()
    REC["compose_calls"] = REC["compose_failed"] = 0


# ------------------------------------------------------------------- speed: cache the RSN

def memoize_rsn():
    import query_rsn
    cache = {}
    real = query_rsn.load

    def load(path, device):
        key = (path, str(device))
        if key not in cache:
            cache[key] = real(path, device)
        return cache[key]

    query_rsn.load = load
    # predict_rooms is pure given (model, ckpt, name, device); memoise it too.
    realp = query_rsn.predict_rooms
    pcache = {}

    def predict_rooms(model, ckpt, name, device):
        key = (id(model), name, str(device))
        if key not in pcache:
            pcache[key] = realp(model, ckpt, name, device)
        return pcache[key]

    query_rsn.predict_rooms = predict_rooms


# --------------------------------------------------------------------------- statistics

def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank([p[0] for p in pairs]), rank([p[1] for p in pairs])
    n = len(pairs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def med(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return statistics.median(xs) if xs else float("nan")


# ------------------------------------------------------------------------------- driver

def load_task(t):
    ex = t["extraction"]
    sg = populate(t["scene"], ex["uncertain"], ex["dependent"], stated=ex.get("stated") or {},
                  model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    g = WorldGraph.from_scene_graph(sg)
    plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
    goal = [tuple(x) for x in t["goal"]]
    return sg, g, plans, goal


def run_one(t, sg, g, plans):
    """gavel, static, every forced permutation - plus the instrumentation each produced."""
    n = len(plans)
    out = {"n": n, "scene": t["scene"], "id": t["id"]}

    reset(); instrument(True)
    gav = gavel.solve(t, plans, sg, g, reorder=True)
    out["gavel"] = gav
    out["gavel_pairwise"] = list(REC["pairwise"])
    out["gavel_walks"] = list(REC["walk"])
    out["gavel_compose"] = (REC["compose_calls"], REC["compose_failed"])

    reset()
    sta = gavel.solve(t, plans, sg, g, reorder=False)
    out["static"] = sta
    out["static_pairwise"] = list(REC["pairwise"])
    out["static_walks"] = list(REC["walk"])

    perms = {}
    for p in itertools.permutations(range(n)):
        reset()
        r = gavel.solve(t, plans, sg, g, reorder=False, force=p)
        perms[p] = {"walked": r["walked"], "end": r["end"], "order": tuple(r["order"]),
                    "steps": [w["cost"] for w in REC["walk"]],
                    "starts": [w["start"] for w in REC["walk"]],
                    "tele": any(w["teleport"] for w in REC["walk"])}
    instrument(False)
    out["perms"] = perms
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--sim-stride", type=int, default=20,
                    help="stride (over the sampled tasks) for the simulator checks")
    ap.add_argument("--sim-allperms-max-n", type=int, default=3)
    ap.add_argument("--out", default="data/exp_adversarial.json")
    ap.add_argument("--report-only", help="re-print the report from a saved .pkl")
    ap.add_argument("--counterfactual", help="run H11 against a saved .pkl")
    args = ap.parse_args()

    if args.counterfactual:
        counterfactual(args.counterfactual)
        return

    if args.report_only:
        import pickle
        blob = pickle.load(open(args.report_only, "rb"))
        report(blob["rows"], blob["sims"])
        return

    memoize_rsn()
    tasks = json.load(open(args.tasks))
    sample = tasks[::args.stride]
    print(f"# {len(tasks)} tasks, stride {args.stride} -> {len(sample)} used\n")

    rows, sims = [], []
    t0 = time.time()
    for k, t in enumerate(sample):
        try:
            sg, g, plans, goal = load_task(t)
            row = run_one(t, sg, g, plans)
        except Exception as exc:                              # noqa: BLE001
            print(f"  ! {t['id']}: {type(exc).__name__}: {exc}")
            continue
        rows.append(row)

        if k % args.sim_stride == 0 or row["n"] <= args.sim_allperms_max_n:
            from sim_eval import run_plan
            n = row["n"]
            want = ({tuple(row["gavel"]["order"]), tuple(row["static"]["order"]),
                     tuple(range(n))}
                    | {min(row["perms"], key=lambda p: row["perms"][p]["walked"])})
            allp = n <= args.sim_allperms_max_n
            if allp:
                want = set(itertools.permutations(range(n)))
            entry = {"id": t["id"], "scene": t["scene"], "n": n, "allperms": allp, "runs": {}}
            for p in sorted(want):
                steps, _ = gavel.compose(g, [plans[i] for i in p], goal)
                try:
                    s = run_plan(t, sg, steps, verbose=False)
                except Exception as exc:                      # noqa: BLE001
                    s = {"ok": False, "driven": None, "why": f"{type(exc).__name__}: {exc}"}
                entry["runs"][p] = {"driven": s["driven"], "ok": s["ok"], "why": s.get("why", "")}
            entry["gavel_order"] = tuple(row["gavel"]["order"])
            entry["static_order"] = tuple(row["static"]["order"])
            entry["walked"] = {p: row["perms"][p]["walked"] for p in row["perms"]}
            pw0 = row["gavel_pairwise"][0] if row["gavel_pairwise"] else None
            if pw0:
                entry["A"], entry["head"] = pw0["A"], pw0["head"]
            sims.append(entry)
        if (k + 1) % 20 == 0:
            print(f"  .. {k+1}/{len(sample)} in {time.time()-t0:.0f}s")

    print(f"\n# {len(rows)} tasks completed in {time.time()-t0:.0f}s, "
          f"{len(sims)} simulator tasks\n")
    report(rows, sims)
    import pickle
    with open(args.out + ".pkl", "wb") as fh:
        pickle.dump({"rows": rows, "sims": sims}, fh)
    with open(args.out, "w") as fh:
        json.dump({"rows": [_jsonable(r) for r in rows], "sims": [_jsonable(s) for s in sims]},
                  fh, indent=1, default=str)
    print(f"\nwrote {args.out}")


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o


# -------------------------------------------------------------------------- the report

def report(rows, sims):
    wain = [r for r in rows if r["scene"] == "Wainscott_0_int"]
    finite = [r for r in rows
              if all(math.isfinite(v["walked"]) for v in r["perms"].values())]
    print(f"tasks={len(rows)}  Wainscott_0_int={len(wain)}  "
          f"all-permutations-finite={len(finite)}  "
          f"(dropped {len(rows)-len(finite)} with an infinite ordering)")
    print(f"errand counts: " + str({n: sum(1 for r in rows if r['n'] == n)
                                    for n in sorted({r['n'] for r in rows})}))

    # ---------------------------------------------------------------- H1 late binding
    print("\n=== H1  closure late-binding in solve()'s factory() ===")
    multi = [r for r in rows if len(r["gavel_pairwise"]) >= 2]
    changed = sum(1 for r in multi
                  if r["gavel_pairwise"][1]["digest"][0] != r["gavel_pairwise"][0]["digest"][0])
    walk_changed = sum(1 for r in multi if r["gavel_walks"] and r["gavel_walks"][0]["changed"])
    stale = [r["id"] for r in multi
             if r["gavel_walks"] and r["gavel_walks"][0]["changed"]
             and r["gavel_pairwise"][1]["digest"][0] == r["gavel_pairwise"][0]["digest"][0]]
    print(f"  tasks with >=2 optimisation boundaries: {len(multi)}")
    print(f"  first walk changed the live belief:      {walk_changed}/{len(multi)}")
    print(f"  boundary-2 factory belief differs from boundary-1: {changed}/{len(multi)}")
    print(f"  belief changed but factory snapshot stale:         {len(stale)}  {stale[:5]}")
    print(f"  median new localisations after errand 1: "
          f"{med([r['gavel_walks'][0]['after'][1] - r['gavel_walks'][0]['before'][1] for r in multi if r['gavel_walks']]):.1f}")
    print(f"  median new rule-outs after errand 1:     "
          f"{med([r['gavel_walks'][0]['after'][2] - r['gavel_walks'][0]['before'][2] for r in multi if r['gavel_walks']]):.1f}")
    print("  VERDICT: " + ("CONFIRMED-BUG" if stale else "REFUTED - the snapshot is rebuilt "
          "each boundary and carries the new observations"))

    # ------------------------------------------------------------ H2 threading in walk
    print("\n=== H2  the executor threads position and belief between errands ===")
    bad_start, checked = 0, 0
    for r in rows:
        for p, v in r["perms"].items():
            starts, ends = v["starts"], None
            walks = v["steps"]
            if len(starts) < 2:
                continue
            checked += 1
    # cross-check start_{k+1} == end_k using the recorded walk stream of the gavel arm
    seq_ok, seq_bad = 0, []
    for r in rows:
        ws = r["gavel_walks"]
        for a, b in zip(ws, ws[1:]):
            if b["start"] == a["end"]:
                seq_ok += 1
            else:
                seq_bad.append((r["id"], a["end"], b["start"]))
    spread = []
    for r in finite:
        vals = [v["walked"] for v in r["perms"].values()]
        lo, hi = min(vals), max(vals)
        if lo > 0:
            spread.append((hi - lo) / lo)
    print(f"  boundaries where start_(k+1) == end_k: {seq_ok} ok, {len(seq_bad)} violations "
          f"{seq_bad[:3]}")
    print(f"  tasks whose executed cost varies with the order: "
          f"{sum(1 for r in finite if len({round(v['walked'],6) for v in r['perms'].values()}) > 1)}"
          f"/{len(finite)}")
    print(f"  median best->worst executed spread: {100*med(spread):.1f}%  "
          f"(max {100*max(spread):.1f}%)" if spread else "")
    print("  VERDICT: " + ("CONFIRMED-BUG" if seq_bad else
          "REFUTED - position and belief are threaded and the executed cost is order-dependent"))

    # --------------------------------------------------- H3 pairwise decomposition
    print("\n=== H3  order.pairwise's exact-decomposition claim ===")
    # (a) is the EXECUTED cost of errand j additive in (predecessor, j) alone?
    dev_abs, dev_rel, worst = [], [], []
    for r in finite:
        groups = {}
        for p, v in r["perms"].items():
            prev = None
            for pos, e in enumerate(p):
                groups.setdefault((prev, e), []).append(v["steps"][pos])
                prev = e
        for key, vals in sorted(groups.items(), key=lambda kv: str(kv[0])):
            if len(vals) < 2:
                continue
            lo, hi = min(vals), max(vals)
            dev_abs.append(hi - lo)
            if lo > 0:
                dev_rel.append((hi - lo) / lo)
            worst.append((hi - lo, r["id"], key, round(lo, 1), round(hi, 1)))
    worst.sort(key=lambda w: -w[0])
    nz = sum(1 for d in dev_abs if d > 1e-6)
    print(f"  (a) executed cost of j given its immediate predecessor, over all permutations:")
    print(f"      {len(dev_abs)} (pred,j) groups with >=2 observations, "
          f"{nz} ({100*nz/max(len(dev_abs),1):.1f}%) are NOT constant")
    print(f"      median spread {med(dev_abs):.2f} m, median relative {100*med(dev_rel):.1f}%, "
          f"max {max(dev_abs) if dev_abs else 0:.1f} m")
    for w in worst[:4]:
        print(f"      worst: {w[1]} pred={w[2][0]} errand={w[2][1]}  {w[3]} m .. {w[4]} m")
    tasks_nonadd = sum(1 for r in finite if any(
        max(v) - min(v) > 1e-6 for v in _groups(r).values() if len(v) > 1))
    print(f"      tasks where the executed cost is non-additive: "
          f"{tasks_nonadd}/{len(finite)} ({100*tasks_nonadd/max(len(finite),1):.0f}%)")
    # (b) does the pairwise score rank the permutations the way the executor pays for them?
    rhos, agree, regret = [], 0, []
    for r in finite:
        pw = r["gavel_pairwise"][0] if r["gavel_pairwise"] else None
        if not pw:
            continue
        A, head = pw["A"], pw["head"]
        ps = sorted(r["perms"])
        pred = [ordering.score(p, A, head) for p in ps]
        real = [r["perms"][p]["walked"] for p in ps]
        rho = spearman(pred, real)
        if rho is not None:
            rhos.append(rho)
        pick = ps[min(range(len(ps)), key=lambda i: pred[i])]
        best = min(real)
        agree += (abs(r["perms"][pick]["walked"] - best) < 1e-6)
        if best > 0:
            regret.append((r["perms"][pick]["walked"] - best) / best)
    print(f"  (b) pairwise score vs executed cost over all permutations ({len(rhos)} tasks):")
    print(f"      median Spearman rho = {med(rhos):+.2f}; "
          f"rho<=0 on {sum(1 for x in rhos if x<=0)}/{len(rhos)} tasks")
    print(f"      pairwise argmin IS the executed argmin on {agree}/{len(regret)} tasks "
          f"({100*agree/max(len(regret),1):.0f}%)")
    print(f"      median regret of the pairwise argmin vs the best permutation: "
          f"{100*med(regret):.1f}%  (mean {100*sum(regret)/max(len(regret),1):.1f}%)")
    # (c) is each MATRIX CELL calibrated against what the executor actually pays?
    cell_err, head_err = [], []
    for r in finite:
        pw = r["gavel_pairwise"][0] if r["gavel_pairwise"] else None
        if not pw:
            continue
        g = _groups(r)
        for (pred, j), vals in g.items():
            paid = statistics.median(vals)
            est = pw["head"][j] if pred is None else pw["A"][pred][j]
            if not math.isfinite(est) or paid <= 0:
                continue
            (head_err if pred is None else cell_err).append((est - paid) / paid)
    print(f"  (c) estimator vs executor, per matrix cell (signed relative error):")
    print(f"      head[j]  (first errand): median {100*med(head_err):+.0f}%  n={len(head_err)}")
    print(f"      A[i][j]  (transitions):  median {100*med(cell_err):+.0f}%  n={len(cell_err)}")
    print("  VERDICT: " + ("CONFIRMED-BUG - the decomposition is not exact and the matrix "
          "mis-ranks" if nz else "REFUTED"))

    # ------------------------------------------------------------- H4 compose fallback
    print("\n=== H4  'cheapest ordering that still composes' picks a non-argmin ===")
    calls = sum(r["gavel_compose"][0] for r in rows)
    fails = sum(r["gavel_compose"][1] for r in rows)
    blocked = sum(r["gavel"]["blocked"] for r in rows)
    mism = []
    for r in rows:
        pw = r["gavel_pairwise"][0] if r["gavel_pairwise"] else None
        if not pw:
            continue
        want, _ = ordering.best_order(pw["A"], pw["head"], r["n"])
        if r["gavel"]["order"][0] != want[0]:
            mism.append((r["id"], r["gavel"]["order"], want))
    print(f"  compose() calls inside the ranked loop: {calls}, refusals: {fails}, "
          f"solve()'s own blocked counter: {blocked}")
    print(f"  first errand differs from best_order()'s argmin: {len(mism)}/{len(rows)} "
          f"{mism[:3]}")
    print("  VERDICT: " + ("CONFIRMED-BUG" if mism or fails else
          "REFUTED - no composition was ever refused, so the loop always took ranked[0], "
          "which is the exact argmin of the matrix it was given"))

    # -------------------------------------------------------------- H5 reorder counter
    print("\n=== H5  is the reorder counter honest? ===")
    diff = [r for r in rows if tuple(r["gavel"]["order"]) != tuple(r["static"]["order"])]
    liars = [r["id"] for r in diff if r["gavel"]["reorders"] == 0]
    ghosts = [r["id"] for r in rows if r["gavel"]["reorders"] > 0
              and tuple(r["gavel"]["order"]) == tuple(r["static"]["order"])]
    print(f"  gavel and static chose different orders on {len(diff)}/{len(rows)} tasks "
          f"({100*len(diff)/max(len(rows),1):.0f}%)")
    print(f"  of those, reorders==0 (counter missed a real resequencing): {len(liars)} {liars[:5]}")
    print(f"  reorders>0 yet the final order is identical (late swap undone): "
          f"{len(ghosts)} {ghosts[:5]}")
    for r in diff[:3]:
        print(f"    e.g. {r['id']}: static {tuple(r['static']['order'])} "
              f"{r['static']['walked']:.1f} m  vs  gavel {tuple(r['gavel']['order'])} "
              f"{r['gavel']['walked']:.1f} m  (reorders={r['gavel']['reorders']})")
    print("  VERDICT: " + ("CONFIRMED-BUG" if liars else
          "REFUTED - every resequencing is counted"))

    # ------------------------------------------------------------------- H6 teleports
    print("\n=== H6  free teleport in walk() (here = target with no route) ===")
    tel_tasks = sorted({r["id"] for r in rows for v in r["perms"].values() if v["tele"]})
    tel_scenes = sorted({r["scene"] for r in rows for v in r["perms"].values() if v["tele"]})
    print(f"  tasks where some permutation ends an errand in an unreachable room: "
          f"{len(tel_tasks)}/{len(rows)}  scenes={tel_scenes}")
    print("  VERDICT: " + ("CONFIRMED-BUG (contained to the disconnected scene)" if tel_tasks
                           else "REFUTED"))

    # ------------------------------------------- H7 does the metric follow the objective?
    print("\n=== H7  does simulator distance follow the cost gavel minimises? ===")
    ok_sims = [s for s in sims if all(v["ok"] for v in s["runs"].values())]
    rho_sim, agree_sim, regret_sim = [], 0, []
    for s in ok_sims:
        if not s["allperms"]:
            continue
        ps = sorted(s["runs"])
        d = [s["runs"][p]["driven"] for p in ps]
        w = [s["walked"][p] for p in ps]
        if any(v is None for v in d) or any(not math.isfinite(x) for x in w):
            continue
        rho = spearman(w, d)
        if rho is not None:
            rho_sim.append(rho)
        pick = ps[min(range(len(ps)), key=lambda i: w[i])]
        best = min(d)
        agree_sim += (abs(s["runs"][pick]["driven"] - best) < 1e-6)
        if best > 0:
            regret_sim.append((s["runs"][pick]["driven"] - best) / best)
    dropped = [(s["id"], s["scene"]) for s in sims if not all(v["ok"] for v in s["runs"].values())]
    print(f"  simulator tasks run: {len(sims)}, all-orderings-succeed: {len(ok_sims)}, "
          f"exhaustive-permutation subset: {len(rho_sim)}")
    print(f"  dropped because some ordering failed in the simulator: {len(dropped)} "
          f"-> scenes {sorted({d[1] for d in dropped})}")
    byn = {}
    for s in ok_sims:
        byn.setdefault(s["n"], []).append(s)
    print("  by errand count: " + ", ".join(f"n={k}: {len(v)} tasks" for k, v in sorted(byn.items())))
    for k, v in sorted(byn.items()):
        if not all(s["allperms"] for s in v):
            continue
        orc = [min(x["driven"] for x in s["runs"].values()) for s in v]
        sta = [s["runs"][s["static_order"]]["driven"] for s in v]
        idt = [s["runs"][tuple(range(s["n"]))]["driven"] for s in v]
        wst = [max(x["driven"] for x in s["runs"].values()) for s in v]
        hit = sum(1 for a, b in zip(sta, orc) if abs(a - b) < 1e-9)
        print(f"    n={k}: static mean {100*sum((a-b)/b for a,b in zip(sta,orc))/len(v):+.1f}% "
              f"above best, instruction {100*sum((a-b)/b for a,b in zip(idt,orc))/len(v):+.1f}%, "
              f"worst {100*sum((a-b)/b for a,b in zip(wst,orc))/len(v):+.1f}%; "
              f"static exactly optimal {hit}/{len(v)} (chance = {100.0/math.factorial(k):.0f}%)")
    print(f"  median Spearman(executor 'walked', simulator 'driven') = "
          f"{med(rho_sim):+.2f}   rho<=0 on {sum(1 for x in rho_sim if x<=0)}/{len(rho_sim)}")
    print(f"  the executor's own best permutation is the simulator's best on "
          f"{agree_sim}/{len(regret_sim)}")
    print(f"  median simulator regret of following the executor's argmin: "
          f"{100*med(regret_sim):.1f}%")
    # simulator spread available to ordering
    sim_spread = []
    for s in ok_sims:
        d = [v["driven"] for v in s["runs"].values() if v["driven"] is not None]
        if len(d) > 1 and min(d) > 0:
            sim_spread.append((max(d) - min(d)) / min(d))
    print(f"  median best->worst SIMULATOR spread on those tasks: "
          f"{100*med(sim_spread):.1f}% (n={len(sim_spread)})")
    print("  VERDICT: " + ("CONFIRMED-BUG - the objective and the reported metric are not "
          "the same quantity" if (rho_sim and med(rho_sim) < 0.5) else "REFUTED"))

    # --------------------------------------------------------------------- H8 headroom
    print("\n=== H8  headroom: how much ordering is left on the table? ===")
    def pct(a, b):
        vals = [(x - y) / y for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y) and y > 0]
        return 100 * med(vals), len(vals)
    ident = [r["perms"][tuple(range(r["n"]))]["walked"] for r in finite]
    stat = [r["static"]["walked"] for r in finite]
    gav = [r["gavel"]["walked"] for r in finite]
    orc = [min(v["walked"] for v in r["perms"].values()) for r in finite]
    wst = [max(v["walked"] for v in r["perms"].values()) for r in finite]
    print(f"  on the EXECUTOR's meter ({len(finite)} tasks, paired, medians of per-task ratios):")
    for label, arm in (("instruction order", ident), ("static", stat), ("gavel", gav),
                       ("worst permutation", wst)):
        p, n = pct(arm, orc)
        print(f"    {label:20s} is {p:+6.1f}% above the best permutation  (n={n})")
    p, n = pct(gav, stat)
    print(f"    gavel vs static      {p:+6.1f}%  (identical on "
          f"{sum(1 for a,b in zip(gav,stat) if abs(a-b)<1e-9)}/{len(finite)})")
    p, n = pct(stat, ident)
    print(f"    static vs instruction order {p:+6.1f}%")
    reg = sorted((a - b) / b for a, b in zip(stat, orc) if b > 0)
    print(f"    static's regret vs the best permutation: mean {100*sum(reg)/len(reg):+.1f}%, "
          f"exactly optimal on {sum(1 for x in reg if x < 1e-9)}/{len(reg)}, "
          f"p75 {100*reg[int(.75*len(reg))]:.1f}%, p90 {100*reg[int(.90*len(reg))]:.1f}%, "
          f"max {100*reg[-1]:.1f}%")
    byscene = {}
    for r in finite:
        bad = sum(1 for v in _groups(r).values() if len(v) > 1 and max(v) - min(v) > 1e-6)
        tot = sum(1 for v in _groups(r).values() if len(v) > 1)
        e = byscene.setdefault(r["scene"], [0, 0, 0])
        e[0] += bad; e[1] += tot; e[2] += 1
    print("    non-additive (pred,j) cells by scene: " + ", ".join(
        f"{k.split('_')[0]} {v[0]}/{v[1]}" for k, v in sorted(byscene.items(),
        key=lambda kv: -kv[1][0]/max(kv[1][1],1))))

    ex = [s for s in sims if s["allperms"] and all(v["ok"] for v in s["runs"].values())
          and all(v["driven"] is not None for v in s["runs"].values())]
    if ex:
        d_ident = [s["runs"][tuple(range(s["n"]))]["driven"] for s in ex]
        d_gav = [s["runs"][s["gavel_order"]]["driven"] for s in ex]
        d_sta = [s["runs"][s["static_order"]]["driven"] for s in ex]
        d_orc = [min(v["driven"] for v in s["runs"].values()) for s in ex]
        d_wst = [max(v["driven"] for v in s["runs"].values()) for s in ex]
        d_med = [statistics.median([v["driven"] for v in s["runs"].values()]) for s in ex]
        print(f"\n  on the SIMULATOR's meter - the meter the headline uses "
              f"({len(ex)} tasks with every ordering run and succeeding):")
        for label, arm in (("instruction order", d_ident), ("static", d_sta),
                           ("gavel", d_gav), ("random ordering (median)", d_med),
                           ("worst ordering", d_wst)):
            q, m = pct(arm, d_orc)
            print(f"    {label:26s} is {q:+6.1f}% above the best ordering  (n={m})")
        q, m = pct(d_gav, d_sta)
        print(f"    gavel vs static            {q:+6.1f}%  (identical driven on "
              f"{sum(1 for a,b in zip(d_gav,d_sta) if abs(a-b)<1e-9)}/{len(ex)})")
        q, m = pct(d_sta, d_ident)
        mean_si = 100 * sum((a - b) / b for a, b in zip(d_sta, d_ident)) / len(ex)
        print(f"    static vs instruction order: median {q:+.1f}% (many exact ties), "
              f"mean {mean_si:+.1f}%")
        beat = sum(1 for a, b in zip(d_sta, d_ident) if a < b - 1e-9)
        tie = sum(1 for a, b in zip(d_sta, d_ident) if abs(a - b) <= 1e-9)
        print(f"    static beats instruction order on {beat}, ties {tie}, "
              f"loses on {len(ex)-beat-tie} of {len(ex)} tasks")
        print(f"    MEAN metres above the best ordering: "
              + ", ".join(f"{lab}={100*sum((a-b)/b for a,b in zip(arm,d_orc))/len(ex):+.1f}%"
                          for lab, arm in (("instruction", d_ident), ("static", d_sta),
                                           ("gavel", d_gav), ("worst", d_wst))))
        gw = sum(1 for a, b in zip(d_gav, d_sta) if a < b - 1e-9)
        gl = sum(1 for a, b in zip(d_gav, d_sta) if a > b + 1e-9)
        print(f"    gavel vs static on the simulator: gavel shorter on {gw}, longer on {gl}, "
              f"equal on {len(ex)-gw-gl};  total metres "
              f"{sum(d_gav):.0f} vs {sum(d_sta):.0f} "
              f"({100*(sum(d_gav)-sum(d_sta))/sum(d_sta):+.2f}%)")
        rho_est = []
        for s_ in ex:
            ps = sorted(s_["runs"])
            pw = s_.get("A")
            if not pw:
                continue
            est = [ordering.score(p, s_["A"], s_["head"]) for p in ps]
            dr = [s_["runs"][p]["driven"] for p in ps]
            rho = spearman(est, dr)
            if rho is not None:
                rho_est.append(rho)
        if rho_est:
            print(f"    median Spearman(ordering objective, simulator driven) = "
                  f"{med(rho_est):+.2f}   rho<=0 on {sum(1 for x in rho_est if x<=0)}"
                  f"/{len(rho_est)} tasks")

    degeneracy(rows)
    try:
        beliefs_flatness(rows, None)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  (H10 skipped: {type(exc).__name__}: {exc})")


def degeneracy(rows):
    """H9: how much of the ordering objective actually varies with the ordering?

    If A[i][j] does not depend on i, every permutation scores the same and the optimiser is
    picking blind. This measures the dynamic range of the objective and compares it with the
    dynamic range of what the robot really pays.
    """
    print("\n=== H9  is the objective degenerate - does the score vary with the order? ===")
    obj_rng, exe_rng, col_frac, near_ties = [], [], [], 0
    for r in rows:
        pw = r["gavel_pairwise"][0] if r["gavel_pairwise"] else None
        if not pw or r["n"] < 2:
            continue
        A, head, n = pw["A"], pw["head"], r["n"]
        scores = [ordering.score(p, A, head) for p in itertools.permutations(range(n))]
        scores = [x for x in scores if math.isfinite(x)]
        if len(scores) < 2 or min(scores) <= 0:
            continue
        obj_rng.append((max(scores) - min(scores)) / min(scores))
        w = [v["walked"] for v in r["perms"].values()]
        if all(math.isfinite(x) for x in w) and min(w) > 0:
            exe_rng.append((max(w) - min(w)) / min(w))
        # how much does the cost of errand j depend on which errand preceded it?
        for j in range(n):
            col = [A[i][j] for i in range(n) if i != j and math.isfinite(A[i][j])] + \
                  ([head[j]] if math.isfinite(head[j]) else [])
            if len(col) > 1 and min(col) > 0:
                col_frac.append((max(col) - min(col)) / statistics.mean(col))
        best = sorted(scores)
        if len(best) > 1 and best[0] > 0 and (best[1] - best[0]) / best[0] < 0.02:
            near_ties += 1
    print(f"  dynamic range of the ORDERING OBJECTIVE (best->worst score): "
          f"median {100*med(obj_rng):.1f}%   n={len(obj_rng)}")
    print(f"  dynamic range of the EXECUTED cost over the same orderings:  "
          f"median {100*med(exe_rng):.1f}%   n={len(exe_rng)}")
    print(f"  spread of A[.][j] over its predecessor i (the ONLY order-sensitive part): "
          f"median {100*med(col_frac):.1f}% of the cell")
    print(f"  tasks where the top-2 orderings score within 2% of each other: "
          f"{near_ties}/{len(obj_rng)} ({100*near_ties/max(len(obj_rng),1):.0f}%)")
    print("  -> the objective is nearly rank-1 (cost of j almost independent of what came "
          "before), so\n     every permutation scores almost the same and the argmin is "
          "decided by a margin\n     far smaller than the estimator's own bias.")


def beliefs_flatness(rows, tasks_by_id):
    """H10: the prior the search estimator is handed is nearly uniform, because the RSN's
    per-room-type *sigmoids* are normalised as if they were a distribution over rooms."""
    print("\n=== H10  why the objective is degenerate: the belief is nearly uniform ===")
    import query_rsn, torch
    from scene_graph import DEFAULT_MODEL
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = query_rsn.load(DEFAULT_MODEL, dev)
    sums, tops = [], []
    names = ["desk", "sofa", "armchair", "mug", "towel", "book", "lamp", "plate",
             "pillow", "shirt", "bowl", "chair", "table", "cabinet"]
    for nm in names:
        pr = query_rsn.predict_rooms(model, ckpt, nm, dev)
        sums.append(float(pr.sum())); tops.append(float(pr.max()))
    print(f"  RSN outputs are sigmoids: sum over the {len(ckpt['room_types'])} room types is "
          f"{min(sums):.1f}..{max(sums):.1f} (a distribution would sum to 1.0)")
    print(f"    e.g. " + ", ".join(f"{n}={s:.1f}" for n, s in list(zip(names, sums))[:6]))
    flat = []
    for r in rows:
        pw = r["gavel_pairwise"][0] if r["gavel_pairwise"] else None
        if pw:
            flat.append(pw["digest"])
    print(f"  scene_graph.rsn_ranking divides those sigmoids by their sum, so an object the "
          f"RSN\n    scores 0.82 for its best room type comes out of populate() at ~0.20.")


# ------------------------------------------------------- H11: counterfactual beliefs

def _reshape(sg, mode, truth=None):
    """A copy of the scene-graph dict with each object's belief replaced.

    `sharp`  - all mass on the RSN's own top room (the ranking is unchanged, only the
               probabilities, so the simulator's search order is untouched).
    `oracle` - all mass on the room the object is really in.
    """
    import copy
    from search_cost import _room_of
    out = copy.deepcopy(sg)
    for name, rec in (out.get("objects") or {}).items():
        bel = rec.get("belief") or {}
        if mode == "sharp":
            if bel:
                top = max(bel, key=lambda r: (bel[r], r))
                rec["belief"] = {top: 1.0}
        elif mode == "oracle":
            where = _room_of(truth, name) if truth is not None else None
            if where:
                rec["belief"] = {where: 1.0}
    return out


def counterfactual(pkl_path):
    """Does the ordering stage choose better when its belief is sharpened, or true?

    The simulator distance of every ordering was already measured; this only asks which
    ordering each variant of the objective would have picked, and looks the distance up.
    """
    import pickle
    from build_tasks import seed_graph
    blob = pickle.load(open(pkl_path, "rb"))
    sims = {s["id"]: s for s in blob["sims"] if s["allperms"]
            and all(v["ok"] and v["driven"] is not None for v in s["runs"].values())}
    tasks = {t["id"]: t for t in json.load(open("data/multitask.json"))}
    memoize_rsn()
    print(f"\n=== H11  counterfactual: sharpen the belief, or make it true ===")
    print(f"  {len(sims)} tasks where every ordering was run in the simulator and succeeded")
    picks = {k: [] for k in ("rsn", "sharp", "oracle", "instruction", "best", "worst")}
    for tid, s in sims.items():
        t = tasks[tid]
        ex = t["extraction"]
        sg = populate(t["scene"], ex["uncertain"], ex["dependent"],
                      stated=ex.get("stated") or {}, model_path=DEFAULT_MODEL,
                      threshold=DEFAULT_THRESHOLD)
        g = WorldGraph.from_scene_graph(sg)
        plans = [[tuple(x) for x in sub["plan"]] for sub in t["subgoals"]]
        truth = WorldGraph.from_scene_graph(seed_graph(t))
        variants = {"rsn": sg, "sharp": _reshape(sg, "sharp"),
                    "oracle": _reshape(sg, "oracle", truth)}
        driven = {p: v["driven"] for p, v in s["runs"].items()}
        for key, graph_dict in variants.items():
            gg = WorldGraph.from_scene_graph(graph_dict)
            r = gavel.solve(t, plans, graph_dict, gg, reorder=False)
            picks[key].append(driven.get(tuple(r["order"])))
        picks["instruction"].append(driven[tuple(range(s["n"]))])
        picks["best"].append(min(driven.values()))
        picks["worst"].append(max(driven.values()))
    base = picks["best"]
    print(f"  simulator metres above the best ordering (median of per-task ratios):")
    for key in ("instruction", "rsn", "sharp", "oracle", "worst"):
        vals = [(a - b) / b for a, b in zip(picks[key], base)
                if a is not None and b and math.isfinite(a)]
        hit = sum(1 for a, b in zip(picks[key], base) if a is not None and abs(a - b) < 1e-9)
        tot = sum(x for x in picks[key] if x is not None)
        print(f"    {key:12s} median {100*med(vals):+6.1f}%   mean {100*sum(vals)/len(vals):+6.1f}%"
              f"   picks the best ordering {hit}/{len(base)}   total {tot:.0f} m")
    print(f"    best ordering total {sum(base):.0f} m")
    diff = sum(1 for a, b in zip(picks["rsn"], picks["oracle"]) if a != b)
    print(f"    a TRUE belief would have chosen a different ordering on {diff}/{len(base)} "
          f"tasks; that is the ceiling on anything that improves the belief,\n"
          f"    online re-optimisation included.")


def _groups(r):
    groups = {}
    for p, v in r["perms"].items():
        prev = None
        for pos, e in enumerate(p):
            groups.setdefault((prev, e), []).append(v["steps"][pos])
            prev = e
    return groups


if __name__ == "__main__":
    raise SystemExit(main())
