#!/usr/bin/env python3
"""Audit: is GAVEL's online re-ordering loop actually learning anything?

The +0.6% gap between `static` (order optimised once against the prior) and `gavel`
(order re-optimised at every errand boundary) has two very different explanations:

    (a) the robot learns almost nothing new while executing, so there is nothing to
        re-optimise against - a real property of a benchmark whose instructions state
        where most objects are;
    (b) the information *is* gathered but never reaches the estimator - a defect that
        would make the online path inert no matter how much there was to learn.

This script separates them.  It re-implements `gavel.solve` / `gavel.walk` /
`gavel.observe` **verbatim** (copied, not imported, so gavel.py is untouched) with a
probe on every place information enters or leaves the belief, and it asserts on every
task that the traced copy reproduces `gavel.solve`'s order, distance and reorder count
exactly - so the instrumentation is measuring the shipped policy and not a fork of it.

What it measures, per task:

    * objects that are genuinely uncertain at t=0 (belief spread over >1 room);
    * every observation event during execution, split into
          localized      an object found by a sweep,
          ruled-effective a room actually deleted from an object's belief,
          ruled-noop      a room the object's belief never contained (no effect at all),
          refuted         a belief emptied and rebuilt from the fallback;
      each tagged with whether it concerned an object of a *still-remaining* errand,
      i.e. whether it could possibly change a future decision;
    * at every boundary, whether the argmin disagreed with the schedule already in hand,
      the predicted saving of that disagreement, and a **counterfactual**: the same
      argmin recomputed from the *prior* belief at the same position.  A reorder the
      counterfactual also makes was not caused by learning - it was caused by the robot
      now knowing where it is standing rather than a distribution over where it might be.
    * the realised saving, paired, gavel vs static, in executor metres and (where the
      orders differ) in simulator-driven metres.

Usage:  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python exp_infogain.py --stride 2
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import json
import statistics
import sys
import time
from collections import Counter

import gavel
import order as ordering
from build_tasks import seed_graph
from graph_machine import GraphMachine
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from search_cost import ETA as ETA_SWEEP
from search_cost import Beliefs, _room_of
from world_graph import WorldGraph

INF = float("inf")


# ---------------------------------------------------------------------------
# Speed only: `scene_graph.populate` reloads the RSN checkpoint and re-embeds the
# object name once per object, which dominates the run.  Memoising is semantics
# preserving - both are pure functions of their arguments - and touches no repo file.
# ---------------------------------------------------------------------------
def _memoise_rsn():
    import query_rsn

    real_load, real_predict = query_rsn.load, query_rsn.predict_rooms
    cache_l, cache_p = {}, {}

    def load(path, device):
        key = (path, str(device))
        if key not in cache_l:
            cache_l[key] = real_load(path, device)
        return cache_l[key]

    def predict_rooms(model, ckpt, name, device):
        key = (id(model), name, str(device))
        if key not in cache_p:
            cache_p[key] = real_predict(model, ckpt, name, device)
        return cache_p[key]

    query_rsn.load, query_rsn.predict_rooms = load, predict_rooms


# ---------------------------------------------------------------------------
# gavel.observe, copied verbatim, with every belief-changing event logged.
# ---------------------------------------------------------------------------
def observe(beliefs, truth, room, names, reachable, log, remaining_names):
    seen = set()
    for name in names:
        if name in beliefs.located:
            continue
        where = _room_of(truth, name)
        if where is None:
            continue
        relevant = name in remaining_names
        if where == room:
            was_certain = len(beliefs.prior.get(name) or {}) <= 1
            beliefs.place(name, room)
            seen.add(name)
            log["localized"].append((name, room, relevant, was_certain))
            continue
        if name not in beliefs.prior or room not in beliefs.prior[name]:
            beliefs.ruled.setdefault(name, set()).add(room)
            log["ruled_noop"].append((name, room, relevant))
            continue
        beliefs.ruled.setdefault(name, set()).add(room)
        del beliefs.prior[name][room]
        log["ruled_effective"].append((name, room, relevant))
        left = beliefs.prior[name]
        if reachable is not None and not (set(left) & reachable):
            log["refuted"].append((name, room, relevant))
            spare = {r: p for r, p in (beliefs.fallback.get(name) or {}).items()
                     if r in reachable and r not in beliefs.ruled.get(name, set())}
            if spare:
                bulk = sum(spare.values())
                beliefs.prior[name] = {r: p / bulk for r, p in spare.items()}
            else:
                fresh = sorted(reachable - beliefs.ruled.get(name, set())) or sorted(reachable)
                beliefs.prior[name] = {r: 1.0 / len(fresh) for r in fresh}
        elif sum(left.values()) > 0:
            total = sum(left.values())
            beliefs.prior[name] = {r: p / total for r, p in left.items()}
        else:
            fresh = sorted(reachable or beliefs.rooms)
            beliefs.prior[name] = {r: 1.0 / len(fresh) for r in fresh}
    return seen


# ---------------------------------------------------------------------------
# gavel.walk, copied verbatim, threading the log through to `observe`.
# ---------------------------------------------------------------------------
def walk(plan, world, real, beliefs, distance, search, watching, start, reachable,
         log, remaining_names):
    believed = GraphMachine(world.copy(), copy=False)
    actual = GraphMachine(real.copy(), copy=False)
    here = start or _room_of(actual.graph, "robot") or (beliefs.rooms[0] if beliefs.rooms else None)
    total, seen = 0.0, set()

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
                total += step + (search(room) if room != target else ETA_SWEEP * search(room))
                here = room
                log["sweeps"] += 1
                if room != target:
                    log["wrong_guesses"] += 1
                seen |= observe(beliefs, actual.graph, room, watching, reachable, log,
                                remaining_names)
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
    return total, here, believed.graph, actual.graph, seen


# ---------------------------------------------------------------------------
# Belief bookkeeping helpers.
# ---------------------------------------------------------------------------
def snapshot_of(beliefs):
    return {"prior": {k: dict(v) for k, v in beliefs.prior.items()},
            "located": dict(beliefs.located), "found": set(beliefs.found),
            "ruled": {k: set(v) for k, v in beliefs.ruled.items()},
            "fallback": {k: dict(v) for k, v in beliefs.fallback.items()}}


def fingerprint(snap, names):
    """A hashable summary of what the estimator would read for `names`."""
    out = []
    for n in sorted(names):
        if n in snap["located"]:
            out.append((n, "LOC", snap["located"][n]))
        else:
            d = snap["prior"].get(n) or {}
            out.append((n, "P", tuple(sorted((r, round(p, 9)) for r, p in d.items()))))
    return tuple(out)


def uncertain_names(snap, names):
    """Objects with a genuine spread: not localized and >1 candidate room."""
    return {n for n in names
            if n not in snap["located"] and len(snap["prior"].get(n) or {}) > 1}


def choose(plans, world, factory, distance, search, here):
    """gavel.solve's ordering decision, copied verbatim: cheapest composable order."""
    A, head = ordering.pairwise(plans, world, factory, distance, search,
                                start={here: 1.0} if here else None)
    pick, estimate, blocked = None, INF, 0
    for candidate in ordering.ranked(A, head, len(plans)):
        trial = [plans[i] for i in candidate]
        if gavel.compose(world, trial, ())[1].failed_at is None:
            pick, estimate = candidate, ordering.score(candidate, A, head)
            break
    if pick is None:
        pick, estimate = ordering.best_order(A, head, len(plans))
        blocked = 1
    return pick, estimate, A, head, blocked


# ---------------------------------------------------------------------------
# gavel.solve(reorder=True), copied verbatim, instrumented.
# ---------------------------------------------------------------------------
def trace_solve(task, subplans, scene_graph_dict, graph, stale=False):
    """`stale=True` is the decisive ablation: re-optimise at every boundary exactly as
    gavel does, from the robot's *actual* position, but score the orderings against the
    belief as it stood at t=0.  The executor still learns - it still sweeps and still
    pays what the truth costs - the estimator just never hears about it.  The gap
    static -> stale is what re-planning from a known position buys; the gap stale ->
    gavel is what the online *information* buys, and nothing else."""
    table, distance, search = gavel._costs(task["scene"])
    truth = WorldGraph.from_scene_graph(seed_graph(task))
    watching = sorted({o for plan in subplans for a, o in plan if o})
    mk = lambda: Beliefs(scene_graph_dict, table["rooms"])

    origin = _room_of(truth, "robot") or (table["rooms"][0] if table["rooms"] else None)
    reachable = {r for r in table["rooms"] if distance(origin, r) != INF}

    beliefs = mk()
    remaining = list(range(len(subplans)))
    world, real, here = graph.copy(), truth.copy(), None
    sequence, walked, estimates, reorders, blocked = [], 0.0, [], 0, 0
    trace, snapshot0, previous = [], None, None

    while remaining:
        plans = [subplans[i] for i in remaining]
        snapshot = snapshot_of(beliefs)
        if snapshot0 is None:
            snapshot0 = snapshot

        def make_factory(snap):
            def factory():
                fresh = mk()
                fresh.prior = {k: dict(v) for k, v in snap["prior"].items()}
                fresh.located = dict(snap["located"])
                fresh.found = set(snap["found"])
                fresh.ruled = {k: set(v) for k, v in snap["ruled"].items()}
                fresh.fallback = {k: dict(v) for k, v in snap["fallback"].items()}
                return fresh
            return factory

        pick, estimate, A, head, was_blocked = choose(
            plans, world, make_factory(snapshot0 if stale else snapshot),
            distance, search, here)
        blocked += was_blocked
        estimates.append(estimate)

        n = len(plans)
        identity = tuple(range(n))
        changed = bool(trace) and pick != identity
        if changed:
            reorders += 1

        # What the same machinery would have picked from this same position with the
        # *prior* belief: isolates "learned something" from "now knows where it stands".
        cf_pick, cf_est_identity = identity, None
        delta_mean = delta_spread = margin_prior = None
        if bool(trace) and n > 1 and not stale:
            cf_pick, _, cfA, cfhead, _ = choose(plans, world, make_factory(snapshot0),
                                                distance, search, here)
            cf_est_identity = ordering.score(identity, cfA, cfhead)
            # Split what learning did to the objective into the part that cancels in an
            # argmin and the part that does not.  J_updated(s) = J_prior(s) + delta(s);
            # a delta that is the same for every ordering moves no decision at all, and
            # the argmin provably cannot change while spread(delta) < margin(prior).
            perms = list(ordering.enumerate_orders(n))
            up = [ordering.score(o, A, head) for o in perms]
            pr = [ordering.score(o, cfA, cfhead) for o in perms]
            if all(x != INF for x in up + pr):
                delta = [a - b for a, b in zip(up, pr)]
                delta_mean = sum(delta) / len(delta)
                delta_spread = max(delta) - min(delta)
                srt = sorted(pr)
                margin_prior = srt[1] - srt[0] if len(srt) > 1 else 0.0

        rem_names = {o for i in remaining for a, o in subplans[i] if o}

        # Has the belief the estimator is about to read - about the errands still to do -
        # actually moved since the previous boundary?  Compared over the SAME name set, so
        # a shrinking remaining-set cannot fake a change.
        rem_changed = rem_learned = None
        if previous is not None:
            rem_changed = fingerprint(snapshot, rem_names) != fingerprint(previous, rem_names)
            was_open = uncertain_names(previous, rem_names)
            rem_learned = any(
                fingerprint(snapshot, {nm}) != fingerprint(previous, {nm}) for nm in was_open)

        # How flat is the objective?  Gap between the cheapest ordering and the next one.
        scores = sorted(ordering.score(o, A, head) for o in ordering.enumerate_orders(n))
        margin = (scores[1] - scores[0]) if len(scores) > 1 else None
        record = {
            "boundary": len(trace),
            "remaining": list(remaining),
            "n_remaining": n,
            "here": here,
            "pick": list(pick),
            "changed": changed,
            "changed_head": bool(trace) and pick[0] != 0,
            "est_pick": estimate,
            "est_identity": ordering.score(identity, A, head),
            "cf_pick": list(cf_pick),
            "cf_changed": bool(trace) and tuple(cf_pick) != identity,
            "n_located": len(snapshot["located"]),
            "n_uncertain_remaining": len(uncertain_names(snapshot, rem_names)),
            "fp_all": fingerprint(snapshot, watching),
            "rem_changed": rem_changed,
            "rem_learned": rem_learned,
            "margin": margin,
            "best_score": scores[0],
            "cf_est_identity": cf_est_identity,
            "delta_mean": delta_mean,
            "delta_spread": delta_spread,
            "margin_prior": margin_prior,
        }

        chosen = remaining[pick[0]]
        after = [remaining[i] for i in pick[1:]]
        rem_after = {o for i in after for a, o in subplans[i] if o}
        log = {"localized": [], "ruled_effective": [], "ruled_noop": [], "refuted": [],
               "sweeps": 0, "wrong_guesses": 0}
        cost, here, world, real, seen = walk(subplans[chosen], world, real, beliefs,
                                             distance, search, watching, here, reachable,
                                             log, rem_after)
        walked += cost
        sequence.append(chosen)
        remaining = after
        record["log"] = log
        trace.append(record)
        previous = snapshot

    return ({"order": sequence, "walked": walked,
             "estimated": estimates[0] if estimates else 0.0,
             "reorders": reorders, "blocked": blocked, "end": here}, trace, watching,
            snapshot0)


# ---------------------------------------------------------------------------
def med(xs):
    return statistics.median(xs) if xs else float("nan")


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sim", action="store_true",
                    help="also run the 2-D simulator on tasks whose orders differ")
    ap.add_argument("--simall", action="store_true",
                    help="also run the simulator on the static arm of EVERY task, to get the "
                         "denominator the headline +0.6%% is a fraction of")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    _memoise_rsn()
    tasks = json.load(open(args.tasks))
    chosen = tasks[:: args.stride]
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"# {len(chosen)} tasks (of {len(tasks)}, stride {args.stride})", flush=True)

    rows, started = [], time.time()
    for k, t in enumerate(chosen):
        ex = t["extraction"]
        sg = populate(t["scene"], ex["uncertain"], ex["dependent"],
                      stated=ex.get("stated") or {}, model_path=DEFAULT_MODEL,
                      threshold=DEFAULT_THRESHOLD)
        g = WorldGraph.from_scene_graph(sg)
        plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]

        gres, trace, watching, snap0 = trace_solve(t, plans, sg, g)
        stale, _, _, _ = trace_solve(t, plans, sg, g, stale=True)
        ref = gavel.solve(t, plans, sg, g, reorder=True)
        stat = gavel.solve(t, plans, sg, g, reorder=False)
        naive = gavel.solve(t, plans, sg, g, force=tuple(range(len(plans))))

        fidelity = (ref["order"] == gres["order"] and ref["reorders"] == gres["reorders"]
                    and (ref["walked"] == gres["walked"]
                         or abs(ref["walked"] - gres["walked"]) < 1e-6))

        loc = [e for r in trace for e in r["log"]["localized"]]
        eff = [e for r in trace for e in r["log"]["ruled_effective"]]
        noop = [e for r in trace for e in r["log"]["ruled_noop"]]
        refuted = [e for r in trace for e in r["log"]["refuted"]]

        row = {
            "id": t["id"], "scene": t["scene"], "n": len(plans),
            "fidelity": fidelity,
            "objects": len(watching),
            "uncertain_start": len(uncertain_names(snap0, set(watching))),
            "order_gavel": gres["order"], "order_static": stat["order"],
            "walked_gavel": gres["walked"], "walked_static": stat["walked"],
            "walked_stale": stale["walked"], "walked_naive": naive["walked"],
            "order_stale": stale["order"], "order_naive": naive["order"],
            "reorders": gres["reorders"],
            "boundaries": len(trace),
            "decisions": sum(1 for r in trace if r["boundary"] > 0 and r["n_remaining"] > 1),
            "changed_boundaries": sum(1 for r in trace if r["changed"]),
            "cf_changed_boundaries": sum(1 for r in trace if r["cf_changed"]),
            "learned_changed": sum(1 for r in trace if r["changed"] and not r["cf_changed"]),
            "position_changed": sum(1 for r in trace if r["changed"] and r["cf_changed"]),
            "sweeps": sum(r["log"]["sweeps"] for r in trace),
            "wrong_guesses": sum(r["log"]["wrong_guesses"] for r in trace),
            "localized": len(loc),
            "localized_new": sum(1 for e in loc if not e[3]),
            "localized_relevant": sum(1 for e in loc if e[2]),
            "localized_relevant_new": sum(1 for e in loc if e[2] and not e[3]),
            "ruled_effective": len(eff),
            "ruled_effective_relevant": sum(1 for e in eff if e[2]),
            "ruled_noop": len(noop),
            "refuted": len(refuted),
            "fp_changed": [bool(trace[i]["fp_all"] != trace[i - 1]["fp_all"])
                           for i in range(1, len(trace))],
            "rem_changed": [bool(r["rem_changed"]) for r in trace if r["rem_changed"] is not None],
            "rem_learned": [bool(r["rem_learned"]) for r in trace if r["rem_learned"] is not None],
            "margins": [r["margin"] / r["best_score"] for r in trace
                        if r["margin"] is not None and r["n_remaining"] > 1
                        and 0 < r["best_score"] < INF and r["margin"] != INF],
            "objective_shift": [abs(r["est_identity"] - r["cf_est_identity"])
                                for r in trace if r["cf_est_identity"] is not None
                                and r["est_identity"] != INF and r["cf_est_identity"] != INF],
            "objective_base": [r["cf_est_identity"] for r in trace
                               if r["cf_est_identity"] is not None and r["cf_est_identity"] != INF],
            "split": [(r["delta_mean"], r["delta_spread"], r["margin_prior"], r["best_score"])
                      for r in trace if r["delta_spread"] is not None],
            "located_series": [r["n_located"] for r in trace],
            "uncertain_series": [r["n_uncertain_remaining"] for r in trace],
            "predicted_saving": sum(max(0.0, r["est_identity"] - r["est_pick"])
                                    for r in trace if r["changed"]
                                    and r["est_identity"] != INF and r["est_pick"] != INF),
            "predicted_saving_first": next(
                (r["est_identity"] - r["est_pick"] for r in trace if r["changed"]
                 and r["est_identity"] != INF and r["est_pick"] != INF), 0.0),
        }
        row["orders_differ"] = row["order_gavel"] != row["order_static"]
        rows.append(row)

        if k % 25 == 0:
            print(f"  {k}/{len(chosen)}  {time.time()-started:.0f}s", flush=True)

    # ---- simulator, only where the composed step sequence actually differs ----
    if args.sim:
        from sim_eval import run_plan
        for row, t in zip(rows, chosen):
            if args.simall and not row["orders_differ"]:
                ex = t["extraction"]
                sg = populate(t["scene"], ex["uncertain"], ex["dependent"],
                              stated=ex.get("stated") or {}, model_path=DEFAULT_MODEL,
                              threshold=DEFAULT_THRESHOLD)
                g = WorldGraph.from_scene_graph(sg)
                plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
                goal = [tuple(x) for x in t["goal"]]
                steps, _ = gavel.compose(g, [plans[i] for i in row["order_static"]], goal)
                sim = run_plan(t, sg, steps, verbose=False)
                row["sim_same"] = sim["driven"] if sim.get("ok") else None
            row["sim_gavel"] = row["sim_static"] = None
            if not row["orders_differ"]:
                continue
            ex = t["extraction"]
            sg = populate(t["scene"], ex["uncertain"], ex["dependent"],
                          stated=ex.get("stated") or {}, model_path=DEFAULT_MODEL,
                          threshold=DEFAULT_THRESHOLD)
            g = WorldGraph.from_scene_graph(sg)
            plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
            goal = [tuple(x) for x in t["goal"]]
            for tag, order in (("gavel", row["order_gavel"]), ("static", row["order_static"])):
                steps, _ = gavel.compose(g, [plans[i] for i in order], goal)
                sim = run_plan(t, sg, steps, verbose=False)
                row[f"sim_{tag}"] = sim["driven"] if sim.get("ok") else None
                row[f"simok_{tag}"] = bool(sim.get("ok"))

    report(rows, args)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1, default=str)
        print(f"\nwrote {args.out}")


def report(rows, args):
    N = len(rows)
    P = lambda a: f"{a:3d}/{N} ({pct(a, N):.1f}%)"
    print("\n" + "=" * 78)
    print(f"EXP_INFOGAIN  -  {N} tasks, stride {args.stride}")
    print("=" * 78)

    bad = [r["id"] for r in rows if not r["fidelity"]]
    print(f"\n[0] fidelity of the traced copy vs gavel.solve: "
          f"{N - len(bad)}/{N} identical" + (f"  MISMATCH: {bad[:5]}" if bad else ""))

    inf_rows = [r for r in rows if r["walked_gavel"] == INF or r["walked_static"] == INF]
    print(f"    tasks with an infinite executor distance (disconnected scene): {len(inf_rows)}"
          + (f"  scenes: {sorted({r['scene'] for r in inf_rows})}" if inf_rows else ""))

    # ---------------- 1. how much is uncertain to begin with ----------------
    objs = sum(r["objects"] for r in rows)
    unc = sum(r["uncertain_start"] for r in rows)
    print(f"\n[1] PRIOR UNCERTAINTY")
    print(f"    task objects (in subplans)        {objs}")
    print(f"    of which >1 candidate room        {unc} ({pct(unc, objs):.1f}%)"
          f"   -> effectively known {pct(objs - unc, objs):.1f}%")
    print(f"    tasks with 0 uncertain objects    {P(sum(1 for r in rows if r['uncertain_start'] == 0))}")
    print(f"    median uncertain objects / task   {med([r['uncertain_start'] for r in rows]):.1f}"
          f"   (median objects/task {med([r['objects'] for r in rows]):.1f})")

    # ---------------- 2. what the run actually observes ----------------
    print(f"\n[2] OBSERVATION EVENTS DURING EXECUTION  (totals over {N} tasks)")
    for key, label in (("sweeps", "room visits (each triggers observe)"),
                       ("wrong_guesses", "  of which a wrong guess (full sweep paid)"),
                       ("localized", "objects localized by a sweep"),
                       ("localized_new", "  ...that were NOT already single-candidate"),
                       ("ruled_effective", "rooms deleted from a live belief"),
                       ("ruled_noop", "  rule-outs with no effect (room not in belief)"),
                       ("refuted", "beliefs emptied and rebuilt from fallback")):
        tot = sum(r[key] for r in rows)
        hit = sum(1 for r in rows if r[key])
        print(f"    {label:46s} {tot:6d}   on {P(hit)}")

    print(f"\n[3] DID THE ROBOT LEARN ANYTHING IT DID NOT ALREADY KNOW?")
    any_event = [r for r in rows if r["localized"] or r["ruled_effective"]]
    genuinely = [r for r in rows if r["localized_new"] or r["ruled_effective"]]
    actionable = [r for r in rows if r["localized_relevant_new"] or r["ruled_effective_relevant"]]
    confirm_only = [r for r in rows if (r["localized"] or r["ruled_effective"])
                    and not (r["localized_new"] or r["ruled_effective"])]
    print(f"    L0 any observation event at all              {P(len(any_event))}")
    print(f"    L1 genuinely new (belief actually changed)   {P(len(genuinely))}")
    print(f"    L2 ...and about a STILL-REMAINING errand     {P(len(actionable))}   <-- learnable")
    print(f"    confirmation only (already single-candidate) {P(len(confirm_only))}")

    dec = [r for r in rows if r["decisions"] > 0]
    print(f"    tasks with >=1 real decision left (n>1 at a later boundary) {P(len(dec))}")
    two = sum(1 for r in rows if r["n"] == 2)
    print(f"    (a 2-errand task has NO decision after the first: {P(two)} of the sample)")
    both = [r for r in actionable if r["decisions"] > 0]
    print(f"    L2 AND a decision left                       {P(len(both))}")

    print(f"\n[4] DID WHAT IT LEARNED CHANGE THE ORDER?")
    changed = [r for r in rows if r["changed_boundaries"]]
    print(f"    tasks where the argmin ever disagreed        {P(len(changed))}")
    print(f"    tasks where gavel's final order != static's  {P(sum(1 for r in rows if r['orders_differ']))}")
    if both:
        c = sum(1 for r in both if r["changed_boundaries"])
        print(f"    OF the {len(both)} 'learned something actionable' tasks, order changed on "
              f"{c} ({pct(c, len(both)):.1f}%)")
    if actionable:
        c = sum(1 for r in actionable if r["changed_boundaries"])
        print(f"    OF the {len(actionable)} L2 tasks (decision or not), order changed on "
              f"{c} ({pct(c, len(actionable)):.1f}%)")
    ids = {id(r) for r in actionable}
    nolearn = [r for r in rows if id(r) not in ids]
    c = sum(1 for r in nolearn if r["changed_boundaries"])
    print(f"    OF the {len(nolearn)} tasks that learned nothing actionable, order changed on "
          f"{c} ({pct(c, max(len(nolearn),1)):.1f}%)")
    print(f"    by prior uncertainty:")
    for lo, hi, lab in ((0, 0, "0 uncertain objects"), (1, 1, "1 uncertain object"),
                        (2, 99, ">=2 uncertain objects")):
        sub = [r for r in rows if lo <= r["uncertain_start"] <= hi]
        ch = sum(1 for r in sub if r["orders_differ"])
        print(f"      {lab:24s} {len(sub):3d} tasks, order changed on {ch:3d}"
              f" ({pct(ch, max(len(sub),1)):.1f}%)")

    # per-boundary breakdown
    tot_dec = sum(r["decisions"] for r in rows)
    tot_ch = sum(r["changed_boundaries"] for r in rows)
    tot_cf = sum(r["cf_changed_boundaries"] for r in rows)
    tot_learn = sum(r["learned_changed"] for r in rows)
    tot_pos = sum(r["position_changed"] for r in rows)
    print(f"\n[5] PER-BOUNDARY BREAKDOWN (boundaries after the first with >1 errand left)")
    print(f"    decision points                              {tot_dec}")
    print(f"    argmin disagreed with schedule in hand       {tot_ch} ({pct(tot_ch, tot_dec):.1f}%)")
    print(f"    ...also disagreed under the PRIOR belief     {tot_pos}  (caused by knowing"
          f" where the robot stands, not by learning)")
    print(f"    ...disagreed ONLY under the updated belief   {tot_learn}  (caused by learning)")
    print(f"    counterfactual (prior-belief) disagreements  {tot_cf}")

    # ---------------- 6. belief really moves ----------------
    fp = [x for r in rows for x in r["fp_changed"]]
    fpr = [x for r in rows for x in r["rem_changed"]]
    fpl = [x for r in rows for x in r["rem_learned"]]
    grew = [r for r in rows if len(r["located_series"]) > 1
            and r["located_series"][-1] > r["located_series"][0]]
    print(f"\n[6] DEFECT CHECKS - is the update actually reaching the estimator?")
    print(f"    boundary pairs compared                      {len(fp)}")
    print(f"    belief fingerprint (all watched objects) changed between consecutive"
          f" boundaries  {sum(fp)}/{len(fp)} ({pct(sum(fp), len(fp)):.1f}%)")
    print(f"    belief about the errands STILL TO DO changed (same name set both sides)"
          f"   {sum(fpr)}/{len(fpr)} ({pct(sum(fpr), len(fpr)):.1f}%)")
    print(f"    ...and the change touched an object that was still UNCERTAIN         "
          f"   {sum(fpl)}/{len(fpl)} ({pct(sum(fpl), len(fpl)):.1f}%)")
    print(f"    Beliefs.located grew during the run          "
          f"{len(grew)}/{sum(1 for r in rows if len(r['located_series'])>1)} multi-boundary tasks")
    ser = [r["located_series"] for r in rows if len(r["located_series"]) > 1][:5]
    print(f"    example located-count series                 {ser}")
    us = [r["uncertain_series"] for r in rows if len(r["uncertain_series"]) > 1
          and r["uncertain_series"][0] > 0][:5]
    print(f"    example uncertain-remaining series           {us}")
    shift = [x for r in rows for x in r["objective_shift"]]
    base = [x for r in rows for x in r["objective_base"]]
    print(f"\n    HOW FAR DOES LEARNING MOVE THE OBJECTIVE?  (same ordering, same position,")
    print(f"    scored under the updated belief vs under the prior belief)")
    print(f"      decision boundaries scored both ways       {len(shift)}")
    if shift:
        rel = [a / b for a, b in zip(shift, base) if b > 0]
        print(f"      |updated - prior| cost of the schedule in hand:"
              f" median {med(shift):.2f} m, mean {sum(shift)/len(shift):.2f} m,"
              f" max {max(shift):.2f} m")
        print(f"      as a fraction of that schedule's cost:      median {med(rel)*100:.1f}%,"
              f" max {max(rel)*100:.1f}%")
        print(f"      boundaries where it moved by <0.5 m:        "
              f"{sum(1 for x in shift if x < 0.5)}/{len(shift)}")
    marg = [x for r in rows for x in r["margins"]]
    if marg:
        print(f"      flatness of the objective: median gap best-vs-second"
              f" {med(marg)*100:.1f}% of best (n={len(marg)} decision points)")

    split = [x for r in rows for x in r["split"]]
    if split:
        dm = [abs(a) for a, b, c, d in split]
        ds = [b for a, b, c, d in split]
        mp = [c for a, b, c, d in split]
        print(f"\n[6b] WHY THE UPDATE DOES NOT MOVE THE DECISION")
        print(f"     J_updated(order) = J_prior(order) + delta(order), over all n! orders at")
        print(f"     each decision boundary.  A delta identical for every order cancels in the")
        print(f"     argmin; only its SPREAD can flip one.  n = {len(split)} decision boundaries.")
        print(f"     |common-mode shift|  (mean of delta)     median {med(dm):7.2f} m"
              f"   mean {sum(dm)/len(dm):7.2f} m")
        print(f"     differential shift   (spread of delta)   median {med(ds):7.2f} m"
              f"   mean {sum(ds)/len(ds):7.2f} m")
        print(f"     prior margin         (best vs second)    median {med(mp):7.2f} m"
              f"   mean {sum(mp)/len(mp):7.2f} m")
        ratio = [b / abs(a) for a, b, c, d in split if abs(a) > 1e-9]
        if ratio:
            print(f"     spread / |common|                       median {med(ratio)*100:6.1f}%"
                  f"   (the rest of what is learned is common-mode and cancels)")
        safe = sum(1 for a, b, c, d in split if b < c)
        print(f"     boundaries where spread < margin, so the argmin PROVABLY cannot flip:"
              f" {safe}/{len(split)} ({pct(safe, len(split)):.1f}%)")
        zero = sum(1 for a, b, c, d in split if b < 1e-9)
        print(f"     boundaries where the spread is exactly zero (delta identical for every"
              f" order): {zero}/{len(split)} ({pct(zero, len(split)):.1f}%)")

    # ---------------- 7. predicted vs realised ----------------
    finite = [r for r in rows if r["walked_gavel"] != INF and r["walked_static"] != INF]
    diff = [r for r in finite if r["orders_differ"]]
    print(f"\n[7] PREDICTED VS REALISED SAVING  (paired, executor metres)")
    print(f"    finite-distance tasks                        {len(finite)}/{N}"
          f"   (dropped {N-len(finite)} with inf)")
    tot_g = sum(r["walked_gavel"] for r in finite)
    tot_s = sum(r["walked_static"] for r in finite)
    print(f"    total walked  gavel {tot_g:9.1f} m   static {tot_s:9.1f} m"
          f"   -> {pct(tot_s - tot_g, tot_s):+.2f}%")
    print(f"    tasks with different orders                  {len(diff)}/{len(finite)}"
          f" ({pct(len(diff), len(finite)):.1f}%)")
    if diff:
        pr = [r["predicted_saving"] for r in diff]
        re_ = [r["walked_static"] - r["walked_gavel"] for r in diff]
        print(f"    on those, predicted saving  median {med(pr):6.2f} m   mean {sum(pr)/len(pr):6.2f} m")
        print(f"              realised  saving  median {med(re_):6.2f} m   mean {sum(re_)/len(re_):6.2f} m")
        worse = sum(1 for x in re_ if x < 0)
        print(f"              realised saving <= 0 on {worse}/{len(diff)} of them"
              f"  (min {min(re_):+.1f} m, max {max(re_):+.1f} m)")
        pairs = [(p, q) for p, q in zip(pr, re_) if p > 0]
        if pairs:
            ratio = [q / p for p, q in pairs]
            print(f"              realised/predicted ratio median {med(ratio):+.2f}"
                  f"  (n={len(pairs)})")
    fin = [r for r in rows if all(r[k] != INF for k in
                                  ("walked_gavel", "walked_static", "walked_stale", "walked_naive"))]
    if fin:
        print(f"\n[8] WHAT EACH INCREMENT IS WORTH  (paired, {len(fin)} finite tasks)")
        tots = {k: sum(r["walked_" + k] for r in fin)
                for k in ("naive", "static", "stale", "gavel")}
        print(f"    instruction order, no optimisation at all   {tots['naive']:9.1f} m")
        print(f"    static  (optimised once, never revised)     {tots['static']:9.1f} m"
              f"   {pct(tots['naive']-tots['static'], tots['naive']):+.2f}% vs naive")
        print(f"    stale   (re-optimised, PRIOR belief only)   {tots['stale']:9.1f} m"
              f"   {pct(tots['static']-tots['stale'], tots['static']):+.2f}% vs static")
        print(f"    gavel   (re-optimised, updated belief)      {tots['gavel']:9.1f} m"
              f"   {pct(tots['stale']-tots['gavel'], tots['stale']):+.2f}% vs stale,"
              f" {pct(tots['static']-tots['gavel'], tots['static']):+.2f}% vs static")
        d1 = [r["walked_static"] - r["walked_stale"] for r in fin]
        d2 = [r["walked_stale"] - r["walked_gavel"] for r in fin]
        print(f"    per-task median  static-stale {med(d1):+.2f} m   stale-gavel {med(d2):+.2f} m")
        print(f"    tasks where stale != static order: "
              f"{sum(1 for r in fin if r['order_stale'] != r['order_static'])},"
              f"  gavel != stale order: "
              f"{sum(1 for r in fin if r['order_gavel'] != r['order_stale'])}")

    if args.sim and any(r.get("sim_gavel") is not None for r in rows):
        s = [r for r in rows if r.get("sim_gavel") is not None and r.get("sim_static") is not None]
        print(f"\n    SIMULATOR (only the {len(s)} order-differing tasks that both ran):")
        dg, ds = sum(r["sim_gavel"] for r in s), sum(r["sim_static"] for r in s)
        print(f"      driven gavel {dg:8.1f} m  static {ds:8.1f} m  -> {pct(ds-dg, ds):+.2f}%"
              f" on this subset")
        d = [r["sim_static"] - r["sim_gavel"] for r in s]
        print(f"      per-task median saving {med(d):+.2f} m, "
              f"gavel worse on {sum(1 for x in d if x < 0)}/{len(s)}")
        d.sort()
        print(f"      biggest wins {[round(x,1) for x in d[-5:]]}  "
              f"biggest losses {[round(x,1) for x in d[:5]]}")
        print(f"      sum of savings {sum(d):+.1f} m; without the single largest win"
              f" {sum(d)-d[-1]:+.1f} m ({pct(sum(d)-d[-1], ds-d[-1]):+.2f}%)")
        same = [r["sim_same"] for r in rows if r.get("sim_same") is not None]
        if same:
            denom = sum(same) + sum(r["sim_static"] for r in s)
            saved = sum(r["sim_static"] - r["sim_gavel"] for r in s)
            print(f"      GLOBAL, all tasks that ran: {len(same)} identical-order tasks drove"
                  f" {sum(same):.1f} m (saving exactly 0 by construction)")
            print(f"      total static-arm driven distance {denom:.1f} m over"
                  f" {len(same)+len(s)} tasks -> gavel saves {saved:+.1f} m = {pct(saved, denom):+.2f}%")
        fail = [r["id"] for r in rows if r.get("orders_differ")
                and (r.get("sim_gavel") is None or r.get("sim_static") is None)]
        print(f"      order-differing tasks where a sim run failed: {len(fail)} {fail[:6]}")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    sys.exit(main())
