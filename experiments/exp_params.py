#!/usr/bin/env python3
"""Is the cost model's *shape* hiding GAVEL's gain?

The ordering stage prices an errand as navigation plus search. Search is only paid where a
room must be swept, so if sweeping is cheap next to driving, information about where things
are is worth little and re-optimising the order online cannot pay for itself. Two constants
set that balance:

    search_cost.ETA    = 0.5    fraction of a room swept before the object turns up
    cost_matrix.SWATHE_M = 1.2  metres driven per square metre swept  (s(r) = area / SWATHE)

so *smaller* SWATHE_M means *more expensive* sweeping, and larger ETA means a longer sweep
once the right room is reached.

This script does three things, on the benchmark's own reference subplans (no planner, so the
only thing that varies is the cost model):

  1. splits the predicted and the realised cost into navigation and sweeping, and splits the
     sweeping further into the part that depends on the order (rooms swept in vain, because
     the belief sent the robot to the wrong room first) and the part that does not (the
     eta*s(r) charged on arrival at the room the object is really in, which every ordering
     pays for every errand);
  2. re-runs three arms - `none` (instruction order, no optimisation), `static` (optimised
     once against the prior) and `gavel` (re-optimised at every boundary) - over a 3x3 grid
     of (ETA, SWATHE_M);
  3. checks the direction: making sweeping dear should make information dearer and the
     static->gavel gap wider.

Nothing in the repo is modified. The constants are monkey-patched in this process only, and
`gavel.walk` is swapped for a line-for-line copy that keeps the three sub-totals apart; the
copy is verified against the original by re-running a slice of tasks with the real `walk`
and asserting the walked distances match to 1e-9.

    python exp_params.py --stride 2 --out data/exp-params.json
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import functools
import inspect
import json
import os
import pickle
import statistics
import sys
import time

import cost_matrix
import gavel
import order as ordering
import search_cost
from graph_machine import GraphMachine
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from search_cost import Beliefs, _room_of
from world_graph import WorldGraph

INF = float("inf")
ETAS = (0.1, 0.5, 0.9)
SWATHES = (0.3, 1.2, 4.0)
BASE = (0.5, 1.2)


# ----------------------------------------------------------------------------- parameters
def _eta_slot(fn):
    """Which entry of `fn.__defaults__` is `eta`.

    The three estimator entry points bind `eta=ETA` at def time, so rebinding the module
    global alone leaves every default at 0.5 and the patch silently does nothing.
    """
    spec = inspect.getfullargspec(fn)
    return spec.args.index("eta") - (len(spec.args) - len(spec.defaults))


_ETA_SLOTS = {fn: _eta_slot(fn) for fn in
              (search_cost.expected_search, search_cost.expected_nav, search_cost.rollout)}


def set_params(eta, swathe, rebuild=True):
    search_cost.ETA = eta
    gavel.ETA_SWEEP = eta            # gavel bound its own copy at import
    for fn, slot in _ETA_SLOTS.items():
        d = list(fn.__defaults__)
        d[slot] = eta
        fn.__defaults__ = tuple(d)
    cost_matrix.SWATHE_M = swathe
    if rebuild:
        gavel._table.cache_clear()   # s(r) = area / SWATHE_M lives in the cached table


# ------------------------------------------------------------------- instrumented executor
_SPLIT = []


def walk_split(plan, world, real, beliefs, distance, search, watching, start=None,
               reachable=None):
    """`gavel.walk`, with the running total kept in three pieces.

    nav          driving between rooms
    sweep_home   eta*s(r) paid on arriving at the room the object is really in. Charged on
                 every NAVIGATE_TO, certain or not, and the target room does not depend on
                 the order - so this is an order-invariant additive constant.
    sweep_lost   s(r) paid in full for a room entered and searched without finding it. This
                 is the only sweeping any ordering can avoid, and the only place information
                 can pay.
    """
    eta = gavel.ETA_SWEEP
    believed = GraphMachine(world.copy(), copy=False)
    actual = GraphMachine(real.copy(), copy=False)
    here = start or _room_of(actual.graph, "robot") or (beliefs.rooms[0] if beliefs.rooms else None)
    total, seen = 0.0, set()
    nav = home = lost = 0.0

    for index, (action, arg) in enumerate(plan):
        if action == "NAVIGATE_TO" and arg:
            target = _room_of(actual.graph, arg)
            dist, ranked, known = beliefs.belief(arg)
            if known is not None:
                route = [known]
            else:
                route = sorted((r for r in ranked if distance(here, r) != INF),
                               key=lambda r: (-dist.get(r, 0.0), distance(here, r), r))
            if target and target not in route:
                route.append(target)
            for room in route:
                step = distance(here, room)
                if step == INF:
                    continue
                sweep = search(room) if room != target else eta * search(room)
                total += step + sweep
                nav += step
                if room == target:
                    home += sweep
                else:
                    lost += sweep
                here = room
                seen |= gavel.observe(beliefs, actual.graph, room, watching, reachable)
                if room == target:
                    break
            if target:
                beliefs.place(arg, target)
                here = target
        outcome = actual.step(index, action, arg)
        believed.step(index, action, arg)
        if not outcome.ok:
            break
        if action in ("PLACE_ON_TOP", "PLACE_INSIDE") and arg:
            room = _room_of(actual.graph, arg)
            for edge in ("on_top", "object_inside"):
                for held, _ in actual.graph.edges_of(edge, dst=arg):
                    if room:
                        beliefs.place(held, room)
    _SPLIT.append((nav, home, lost))
    return total, here, believed.graph, actual.graph, seen


def run_arm(task, plans, sg, graph, **kw):
    """One `gavel.solve`, with the executed cost split three ways."""
    del _SPLIT[:]
    result = gavel.solve(task, plans, sg, graph, **kw)
    nav = sum(a for a, _, _ in _SPLIT)
    home = sum(b for _, b, _ in _SPLIT)
    lost = sum(c for _, _, c in _SPLIT)
    result = dict(result)
    result.update({"nav": nav, "sweep_home": home, "sweep_lost": lost})
    return result


# --------------------------------------------------------------------------- predicted cost
def predicted_split(task, plans, sg, graph):
    """What the *estimator* thinks the chosen ordering costs, and how much of that is sweep.

    `expected_search` sorts candidates by belief then distance - never by s(r) - and every
    term is linear in s, so rolling out with s == 0 gives the navigation part exactly and
    the remainder is the sweeping part. No approximation.
    """
    table, distance, search = gavel._costs(task["scene"])
    mk = lambda: Beliefs(sg, table["rooms"])
    zero = lambda r: 0.0
    A1, h1 = ordering.pairwise(plans, graph, mk, distance, search, start=None)
    A0, h0 = ordering.pairwise(plans, graph, mk, distance, zero, start=None)
    pick = None
    for candidate in ordering.ranked(A1, h1, len(plans)):        # same filter solve() uses
        if gavel.compose(graph, [plans[i] for i in candidate], ())[1].failed_at is None:
            pick = candidate
            break
    if pick is None:
        pick = ordering.best_order(A1, h1, len(plans))[0]
    return ordering.score(pick, A1, h1), ordering.score(pick, A0, h0)


def uncertainty(task, plans, sg):
    """How many NAVIGATE_TO targets the prior leaves genuinely ambiguous (>1 candidate room)."""
    table, _, _ = gavel._costs(task["scene"])
    b = Beliefs(sg, table["rooms"])
    targets = [a for plan in plans for act, a in plan if act == "NAVIGATE_TO" and a]
    multi = sum(1 for t in targets if len(b.prior.get(t) or {}) > 1)
    return len(targets), multi


# ------------------------------------------------------------------------------ scene graphs
def scene_graphs(tasks, cache_path):
    """`populate` once per task, reused by all 27 arm-runs. The RSN is the slow part."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, "rb") as handle:
            cache = pickle.load(handle)
    fresh = 0
    for i, task in enumerate(tasks):
        if task["id"] in cache:
            continue
        ex = task["extraction"]
        cache[task["id"]] = populate(task["scene"], ex["uncertain"], ex["dependent"],
                                     stated=ex.get("stated") or {},
                                     model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
        fresh += 1
        if fresh % 25 == 0:
            print(f"  populated {i+1}/{len(tasks)}", flush=True)
            if cache_path:
                with open(cache_path, "wb") as handle:
                    pickle.dump(cache, handle)
    if cache_path and fresh:
        with open(cache_path, "wb") as handle:
            pickle.dump(cache, handle)
    return cache


# ---------------------------------------------------------------------------------- summary
def pct(a, b):
    return 100.0 * (a - b) / a if a else 0.0


def paired(rows, arm_a, arm_b, key):
    """(n, sum_a, sum_b, better, worse, median relative gap %) over tasks finite in both."""
    pairs = [(r[arm_a][key], r[arm_b][key]) for r in rows
             if r.get(arm_a) and r.get(arm_b)
             and r[arm_a].get(key) is not None and r[arm_b].get(key) is not None
             and r[arm_a][key] < INF and r[arm_b][key] < INF]
    if not pairs:
        return 0, 0.0, 0.0, 0, 0, 0.0
    sa = sum(a for a, _ in pairs)
    sb = sum(b for _, b in pairs)
    better = sum(1 for a, b in pairs if b < a - 1e-6)
    worse = sum(1 for a, b in pairs if b > a + 1e-6)
    rel = statistics.median([100.0 * (a - b) / a if a > 0 else 0.0 for a, b in pairs])
    return len(pairs), sa, sb, better, worse, rel


def permutation_sweep(task, plans, sg, graph):
    """Every ordering of this task, executed. Gives the achievable headroom and, with it,
    a direct test of whether `sweep_home` really is an order-invariant constant."""
    import itertools
    out = []
    n = len(plans)
    for perm in itertools.permutations(range(n)):
        r = run_arm(task, plans, sg, graph, force=perm)
        out.append({"order": list(r["order"]), "walked": r["walked"], "nav": r["nav"],
                    "sweep_home": r["sweep_home"], "sweep_lost": r["sweep_lost"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--no-perms", action="store_true")
    ap.add_argument("--cache", default=os.environ.get("SG_CACHE", "/tmp/exp_params_sg.pkl"))
    ap.add_argument("--out", default="data/exp-params.json")
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))[::args.stride]
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"{len(tasks)} tasks (stride {args.stride})", flush=True)

    t0 = time.time()
    graphs = scene_graphs(tasks, args.cache)
    print(f"scene graphs ready in {time.time()-t0:.0f}s", flush=True)

    prepared = []
    for task in tasks:
        sg = graphs[task["id"]]
        prepared.append((task, [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]],
                         sg, WorldGraph.from_scene_graph(sg), [tuple(x) for x in task["goal"]]))

    # ------------------------------------------------------------ verify the instrumented walk
    set_params(*BASE)
    check = []
    for task, plans, sg, g, _ in prepared[:25]:
        real = gavel.solve(task, plans, sg, g, reorder=True)["walked"]
        gavel.walk, keep = walk_split, gavel.walk
        mine = run_arm(task, plans, sg, g, reorder=True)
        gavel.walk = keep
        check.append(abs(real - mine["walked"]))
        assert abs(real - mine["walked"]) < 1e-9, (task["id"], real, mine["walked"])
        assert abs(mine["nav"] + mine["sweep_home"] + mine["sweep_lost"] - mine["walked"]) < 1e-9
    print(f"instrumented walk matches gavel.walk on {len(check)} tasks "
          f"(max |diff| {max(check):.2e})", flush=True)
    gavel.walk = walk_split

    # ------------------------------------------------------------------- prior uncertainty
    tot_nav, tot_multi, per_task = 0, 0, []
    for task, plans, sg, g, _ in prepared:
        a, b = uncertainty(task, plans, sg)
        tot_nav += a
        tot_multi += b
        per_task.append(b)
    print(f"NAVIGATE_TO targets: {tot_nav}, of which >1 candidate room: {tot_multi} "
          f"({100*tot_multi/max(tot_nav,1):.1f}%); tasks with none ambiguous: "
          f"{sum(1 for x in per_task if x == 0)}/{len(per_task)}", flush=True)

    # --------------------------------------------------------------------------- the grid
    sim_cache = {}
    run_plan = None
    if not args.no_sim:
        from sim_eval import run_plan as _rp
        run_plan = _rp

    def simulate(idx, task, plans, sg, order, goal):
        key = (idx, tuple(order))
        if key not in sim_cache:
            steps, _ = gavel.compose(WorldGraph.from_scene_graph(sg),
                                     [plans[i] for i in order], goal)
            try:
                sim = run_plan(task, sg, steps, verbose=False)
                sim_cache[key] = {"ok": bool(sim["ok"]), "driven": sim["driven"],
                                  "why": (sim["why"] or "")[:120]}
            except Exception as exc:
                sim_cache[key] = {"ok": False, "driven": None,
                                  "why": f"{type(exc).__name__}: {exc}"[:120]}
        return dict(sim_cache[key])

    out = {"tasks": len(tasks), "stride": args.stride, "settings": [],
           "nav_targets": tot_nav, "nav_ambiguous": tot_multi}
    for swathe in SWATHES:
        for eta in ETAS:
            set_params(eta, swathe, rebuild=(eta == ETAS[0]))
            began = time.time()
            rows = []
            for idx, (task, plans, sg, g, goal) in enumerate(prepared):
                n = len(plans)
                row = {"id": task["id"], "scene": task["scene"], "n": n}
                arms = {"none": {"force": tuple(range(n))},
                        "static": {"reorder": False},
                        "gavel": {"reorder": True}}
                for name, kw in arms.items():
                    r = run_arm(task, plans, sg, g, **kw)
                    row[name] = {"order": list(r["order"]), "walked": r["walked"],
                                 "nav": r["nav"], "sweep_home": r["sweep_home"],
                                 "sweep_lost": r["sweep_lost"],
                                 "reorders": r["reorders"], "blocked": r["blocked"]}
                if not args.no_perms:
                    perms = permutation_sweep(task, plans, sg, g)
                    best = min(perms, key=lambda p: p["walked"])
                    worst = max(perms, key=lambda p: p["walked"])
                    row["oracle"] = dict(best)
                    row["worst"] = dict(worst)
                    homes = [p["sweep_home"] for p in perms]
                    row["home_spread"] = max(homes) - min(homes)
                    row["perm_spread"] = ((worst["walked"] - best["walked"]) / worst["walked"]
                                          if worst["walked"] > 0 else 0.0)
                pred, pred_nav = predicted_split(task, plans, sg, g)
                row["pred"] = pred
                row["pred_nav"] = pred_nav
                if run_plan is not None:
                    for name in ("none", "static", "gavel") + (() if args.no_perms else ("oracle",)):
                        row[name].update(simulate(idx, task, plans, sg,
                                                  row[name]["order"], goal))
                rows.append(row)
                if (idx + 1) % 50 == 0:
                    print(f"    {idx+1}/{len(prepared)} ({time.time()-began:.0f}s, "
                          f"{len(sim_cache)} sims cached)", flush=True)
            entry = {"eta": eta, "swathe": swathe, "rows": rows,
                     "seconds": round(time.time() - began, 1)}
            out["settings"].append(entry)
            report(entry)
            with open(args.out, "w") as handle:
                json.dump(out, handle)
    print(f"\nwrote {args.out}", flush=True)
    grid(out)
    failures(out)


def report(entry):
    rows = entry["rows"]
    tag = f"ETA={entry['eta']}  SWATHE_M={entry['swathe']}"
    print(f"\n=== {tag} ({entry['seconds']}s) ===", flush=True)
    for arm in ("none", "static", "gavel", "oracle"):
        if arm not in rows[0]:
            continue
        nav = sum(r[arm]["nav"] for r in rows)
        home = sum(r[arm]["sweep_home"] for r in rows)
        lost = sum(r[arm]["sweep_lost"] for r in rows)
        tot = nav + home + lost
        print(f"  {arm:6s} walked {tot:9.0f} m = nav {100*nav/tot:5.1f}%  "
              f"sweep-on-arrival {100*home/tot:5.1f}%  sweep-in-vain {100*lost/tot:5.1f}%")
    pr = [(r["pred"], r["pred_nav"]) for r in rows if r["pred"] < INF]
    if pr:
        p, pn = sum(a for a, _ in pr), sum(b for _, b in pr)
        print(f"  predicted (first-pass estimate, {len(pr)}/{len(rows)} finite): {p:.0f} m = "
              f"nav {100*pn/p:.1f}%  sweep {100*(p-pn)/p:.1f}%")
    if "home_spread" in rows[0]:
        hs = [r["home_spread"] for r in rows]
        flat = sum(1 for h in hs if h < 1e-9)
        med = statistics.median([100 * r["perm_spread"] for r in rows])
        print(f"  sweep-on-arrival identical across ALL orderings on {flat}/{len(rows)} tasks "
              f"(median |max-min| {statistics.median(hs):.2f} m); "
              f"median best-vs-worst ordering spread {med:.1f}%")
    for a, b in (("none", "static"), ("static", "gavel"), ("static", "oracle"),
                 ("gavel", "oracle"), ("none", "gavel")):
        if b not in rows[0] or a not in rows[0]:
            continue
        n, sa, sb, better, worse, rel = paired(rows, a, b, "walked")
        print(f"  cost model {a:6s}->{b:6s}: {sa:8.0f} -> {sb:8.0f} m  {pct(sa,sb):+6.2f}%  "
              f"{(sa-sb)/max(n,1):+6.2f} m/task  median {rel:+5.2f}%  "
              f"better {better:3d} worse {worse:3d} same {n-better-worse:3d}")
    if any("driven" in r["static"] for r in rows):
        for a, b in (("none", "static"), ("static", "gavel"), ("static", "oracle"),
                     ("none", "gavel")):
            if b not in rows[0] or a not in rows[0]:
                continue
            n, sa, sb, better, worse, rel = paired(rows, a, b, "driven")
            print(f"  simulator  {a:6s}->{b:6s}: {sa:8.0f} -> {sb:8.0f} m  {pct(sa,sb):+6.2f}%  "
                  f"{(sa-sb)/max(n,1):+6.2f} m/task  median {rel:+5.2f}%  "
                  f"better {better:3d} worse {worse:3d} same {n-better-worse:3d}  (n={n})")
    diff = [r for r in rows if r["static"]["order"] != r["gavel"]["order"]]
    print(f"  orders differing static vs gavel: {len(diff)}/{len(rows)}; "
          f"reorders fired: {sum(r['gavel']['reorders'] for r in rows)}")
    if diff:
        dn = sum(r["static"]["nav"] - r["gavel"]["nav"] for r in diff)
        dh = sum(r["static"]["sweep_home"] - r["gavel"]["sweep_home"] for r in diff)
        dl = sum(r["static"]["sweep_lost"] - r["gavel"]["sweep_lost"] for r in diff)
        print(f"  on those {len(diff)}: gavel saves {dn+dh+dl:+.1f} m total = "
              f"nav {dn:+.1f} + sweep-on-arrival {dh:+.1f} + sweep-in-vain {dl:+.1f}")


def grid(out):
    print("\n\n================ GRID ================")
    print("  SWATHE  ETA | sweep% of walked (avoidable) | static->gavel cost model | "
          "static->gavel simulator | static->oracle | differ")
    for e in out["settings"]:
        rows = e["rows"]
        nav = sum(r["gavel"]["nav"] for r in rows)
        home = sum(r["gavel"]["sweep_home"] for r in rows)
        lost = sum(r["gavel"]["sweep_lost"] for r in rows)
        tot = nav + home + lost
        n1, sa, sb, _, _, _ = paired(rows, "static", "gavel", "walked")
        n2, da, db, _, _, _ = paired(rows, "static", "gavel", "driven")
        n3, oa, ob, _, _, _ = paired(rows, "static", "oracle", "walked")
        diff = sum(1 for r in rows if r["static"]["order"] != r["gavel"]["order"])
        print(f"  {e['swathe']:5.1f}  {e['eta']:.1f} | {100*(home+lost)/tot:5.1f}% "
              f"({100*lost/tot:4.1f}%) | {pct(sa,sb):+6.2f}% "
              f"({(sa-sb)/max(n1,1):+5.2f} m/task) | {pct(da,db):+6.2f}% "
              f"({(da-db)/max(n2,1):+5.2f} m/task, n={n2}) | {pct(oa,ob):+6.2f}% | "
              f"{diff}/{len(rows)}")


def failures(out):
    rows = out["settings"][0]["rows"]
    if "driven" not in rows[0]["static"]:
        return
    print("\nsimulator outcomes at the first setting (orders are identical across settings "
          "for most tasks, so this is representative):")
    for arm in ("none", "static", "gavel"):
        bad = [r for r in rows if not r[arm].get("ok")]
        scenes = {}
        for r in bad:
            scenes[r["scene"]] = scenes.get(r["scene"], 0) + 1
        print(f"  {arm:6s} failed {len(bad)}/{len(rows)}: {scenes}")


if __name__ == "__main__":
    raise SystemExit(main())


RESULTS = """Measured on 250 tasks (data/multitask.json[::2]), reference subplans, no planner.
250/250 usable for the cost model; simulator distance available on all 250 (13 tasks fail to
reach the goal, all Wainscott_0_int, identically in every arm, and their driven distance is
still a real driven distance so they stay in the paired comparison).

  SWATHE  ETA | sweep% of walked (avoidable) | static->gavel cost model | static->gavel sim | static->cost-model-oracle | differ
    0.3  0.1 |  50.2% (28.7%) | +1.37% (+0.96 m/task) | +0.39% (+0.23 m/task) | +19.80% cm / +1.65% sim | 44/250
    0.3  0.5 |  73.3% (15.3%) | +0.93% (+1.20 m/task) | +0.41% (+0.24 m/task) | +11.03% cm / +1.30% sim | 42/250
    0.3  0.9 |  81.7% (10.6%) | +0.62% (+1.17 m/task) | +0.58% (+0.34 m/task) |  +7.99% cm / +1.10% sim | 41/250
    1.2  0.1 |  19.9% (11.1%) | +2.58% (+1.12 m/task) | +0.57% (+0.33 m/task) | +15.81% cm / +1.61% sim | 38/250
    1.2  0.5 |  40.7% ( 8.4%) | +2.11% (+1.24 m/task) | +0.66% (+0.38 m/task) | +12.72% cm / +1.55% sim | 41/250   <- shipped
    1.2  0.9 |  52.8% ( 6.7%) | +1.68% (+1.24 m/task) | +0.84% (+0.49 m/task) | +10.42% cm / +1.42% sim | 40/250
    4.0  0.1 |   7.0% ( 3.9%) | +2.95% (+1.12 m/task) | +0.85% (+0.50 m/task) | +16.18% cm / +2.13% sim | 41/250
    4.0  0.5 |  17.1% ( 3.5%) | +2.51% (+1.06 m/task) | +0.74% (+0.43 m/task) | +14.46% cm / +1.79% sim | 40/250
    4.0  0.9 |  25.2% ( 3.2%) | +2.44% (+1.14 m/task) | +0.56% (+0.32 m/task) | +13.17% cm / +1.58% sim | 42/250

The sweep share of the charged cost moves from 7.0% to 81.7% - a 13.3x change in the price of
sweeping and a 9x change in eta - and the absolute static->gavel gain does not move: 0.96 to
1.24 m/task in the cost model, 0.23 to 0.50 m/task in the simulator, with 38-44 of 250 orders
changing at every setting. The direction check fails outright in the simulator: the gain is
largest where sweeping is *cheapest* (4.0/0.1, +0.85%) and smallest where it is dearest
(0.3/0.1, +0.39%).

Why: `walk` charges eta*s(target) on arrival at the room the object is really in, on EVERY
NAVIGATE_TO, whether the belief was certain or not. That room does not depend on the order, and
the term is identical across all n! orderings on 212/250 tasks (the 38 exceptions are
Wainscott_0_int and Pomaria_0_int, where disconnection makes a target unreachable from some
start rooms). At the shipped constants it is 32.3% of charged cost and at 0.3/0.9 it is 68%. It
is an additive constant: it cannot change any argmin, it only inflates the denominator of every
percentage. On the tasks where gavel does pick a different order, its saving decomposes as
86.2% navigation + 0.0% sweep-on-arrival + 13.8% sweep-in-vain at the shipped constants
(95.4/0.0/4.6 at 4.0/0.1). Reordering here is winning a routing prize, not an information prize.

And the ceiling is low regardless. Exhausting all n! orderings, the cost model believes 12.7%
is available over `static`; the simulator delivers 1.55% of it, shorter on 71 tasks and longer
on 67 - a coin flip. walked-vs-driven correlates r=+0.72 in levels but only +0.50 on the
differences ordering actually decides, and the sign of a cost-model preference is right on
53-60% of the tasks where both meters move.

What actually triggers a reorder (shipped constants, 250 tasks):
    sharp and right belief for every target      35 tasks    0 reorder ( 0.0%)
    sharp but >=1 wrong                          46 tasks    7 reorder (15.2%)
    >=1 ambiguous, none wrong                    76 tasks    8 reorder (10.5%)
    >=1 ambiguous AND >=1 wrong                  93 tasks   26 reorder (28.0%)
Of 1542 NAVIGATE_TO targets: 1066 sharp-and-right, 211 sharp-but-wrong, 265 ambiguous (17.2%).
"""
