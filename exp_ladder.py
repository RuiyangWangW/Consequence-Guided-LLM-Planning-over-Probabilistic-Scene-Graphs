#!/usr/bin/env python3
"""The ladder: what is ORDERING worth at all, not just what is *revising* the order worth?

The published static-vs-gavel comparison holds the ordering stage fixed and toggles only
whether it is re-run online. Both arms optimise. So that comparison measures the online
increment on top of an already-optimal order and nothing else - it cannot say whether the
ordering stage as a whole earns its place.

This script measures the whole ladder, per task, paired, in both meters:

    written   force=(0,1,...,n-1)          the errands in the order the instruction names
    random    force=<seeded permutation>   a coin flip, reproducible from --seed
    static    reorder=False                prior-optimal, never revised
    gavel     reorder=True                 re-optimised at every errand boundary
    oracle    argmin over all n! of walked the best order in the cost model's own meter

Because a composed plan depends on the permutation and nothing else, every one of those
policies is a *lookup* into the same table: for each task we enumerate all n! permutations,
score each with `gavel.solve(force=...)` (the cost model's `walked`) and drive each through
`sim_eval.run_plan` (the simulator's `driven`). 4,496 simulator runs for tasks[::3]. That
buys two things the ladder alone cannot give: a true oracle in the *driven* meter, not only
in the cost model's, and the percentile rank of each policy inside the full spread of
orderings.

Nothing in the repo is modified. Everything here is read-only against gavel/sim_eval.

    python exp_ladder.py --stride 3 --jobs 10
    python exp_ladder.py --aggregate-only          # re-print the report from the shards
"""

import argparse
import itertools
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
POLICIES = ["written", "random", "static", "gavel", "oracle"]
INF = float("inf")


# ---------------------------------------------------------------- one task

def key(perm):
    return ",".join(str(i) for i in perm)


def run_task(task, seed):
    """Every permutation of one task, in both meters, plus what each policy picked."""
    import gavel
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
    from sim_eval import run_plan
    from world_graph import WorldGraph

    ex = task["extraction"]
    # All three classes, or the beliefs are wrong.
    sg = populate(task["scene"], ex["uncertain"], ex["dependent"],
                  stated=ex.get("stated") or {},
                  model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    g = WorldGraph.from_scene_graph(sg)
    plans = [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]]
    goal = [tuple(x) for x in task["goal"]]
    n = len(plans)

    perms = {}
    for perm in itertools.permutations(range(n)):
        r = gavel.solve(task, plans, sg, g, force=perm)
        assert tuple(r["order"]) == perm, (r["order"], perm)
        steps, outcome = gavel.compose(g, [plans[i] for i in perm], goal)
        sim = run_plan(task, sg, steps, verbose=False)
        perms[key(perm)] = {"walked": r["walked"], "driven": sim["driven"],
                            "sim_ok": bool(sim["ok"]), "why": sim.get("why") or "",
                            "compose_failed_at": outcome.failed_at,
                            "end": r["end"]}

    # The two arms that actually optimise. Their `walked` must equal the forced replay of
    # the order they chose - the executor is the same function - and that is asserted, so a
    # silent divergence between "what the arm did" and "what the table says that order
    # costs" cannot slip through and make the ladder incomparable.
    static = gavel.solve(task, plans, sg, g, reorder=False)
    online = gavel.solve(task, plans, sg, g, reorder=True)
    for arm in (static, online):
        replay = perms[key(tuple(arm["order"]))]["walked"]
        assert abs(replay - arm["walked"]) < 1e-6, (arm["order"], arm["walked"], replay)

    # Why a task has headroom or has none. Two things decide it: how much the robot does
    # not know (an object with one candidate room is a drive, not a search, and ordering
    # buys little), and how spread out the errands are (errands that all land in one room
    # cost the same in every order).
    from build_tasks import seed_graph
    from search_cost import _room_of

    truth = WorldGraph.from_scene_graph(seed_graph(task))
    objects = sg.get("objects") or {}
    watching = sorted({o for plan in plans for _, o in plan if o})
    counts, known, wrong_top = [], 0, 0
    for name in watching:
        record = objects.get(name) or {}
        # "Known" means the *belief* is a point mass, not that `candidates` is short.
        # `candidates` is the search order and always ends with the RSN's full ranking of
        # the scene, so it is never length 1 for anything - counting it would report that
        # no object in the benchmark is ever known, which is the opposite of the truth.
        # A stated room and a dependent object inheriting its support's room both give
        # `belief == {room: 1.0}`: one sweep of one room, not a search.
        dist = {r: p for r, p in (record.get("belief") or {}).items() if p > 0}
        cands = list(record.get("candidates")
                     or ([record["room"]] if record.get("room") else []))
        counts.append(len(dist))
        if len(dist) <= 1:
            known += 1
        where = _room_of(truth, name)
        if cands and where is not None and cands[0] != where:
            wrong_top += 1
    per_errand = []
    for plan in plans:
        per_errand.append(sorted({r for r in (_room_of(truth, o) for a, o in plan
                                              if a == "NAVIGATE_TO" and o) if r}))
    belief = {"objects": len(watching), "known": known,
              "uncertain": len(watching) - known,
              "belief_rooms_mean": (sum(counts) / len(counts)) if counts else 0.0,
              "candidates_mean": (sum(len(list((objects.get(m) or {}).get("candidates") or [])) for m in watching) / max(len(watching), 1)),
              "wrong_top": wrong_top,
              "rooms_touched": len({r for rs in per_errand for r in rs}),
              "errand_rooms": per_errand}

    rng = random.Random(f"{seed}-{task['id']}")
    chosen = {
        "written": tuple(range(n)),
        "random": tuple(rng.sample(range(n), n)),
        "static": tuple(static["order"]),
        "gavel": tuple(online["order"]),
        "oracle": min(itertools.permutations(range(n)),
                      key=lambda p: (perms[key(p)]["walked"], p)),
    }
    return {"id": task["id"], "scene": task["scene"], "n": n,
            "perms": perms, "belief": belief,
            "order": {name: list(p) for name, p in chosen.items()},
            "reorders": online["reorders"], "blocked": online["blocked"] + static["blocked"],
            "estimated": {"static": static["estimated"], "gavel": online["estimated"]}}


def worker(tasks, seed, out_path):
    rows = []
    started = time.time()
    for i, task in enumerate(tasks):
        rows.append(run_task(task, seed))
        with open(out_path, "w") as handle:
            json.dump(rows, handle)
        print(f"[{os.path.basename(out_path)}] {i+1}/{len(tasks)} {task['id']} "
              f"n={len(task['subgoals'])} {time.time()-started:.0f}s", flush=True)


# ---------------------------------------------------------------- statistics

def med(xs):
    return statistics.median(xs) if xs else float("nan")


def pct_change(xs, base):
    """Paired per-task percent change. Only defined where the baseline is positive."""
    return [100.0 * (x - b) / b for x, b in zip(xs, base) if b > 0 and math.isfinite(x) and math.isfinite(b)]


def capture(rows, policy, meter, oracle_key):
    """Share of the written->oracle headroom this policy captures, per task and in total.

    Per task only where there IS headroom: a task whose written order is already optimal
    has a zero denominator and captures nothing and everything at once. Those are counted
    separately rather than folded in as 0 or 1.
    """
    each, num, den, none = [], 0.0, 0.0, 0
    for row in rows:
        w, p, o = row["written"][meter], row[policy][meter], row[oracle_key][meter]
        if not all(math.isfinite(v) for v in (w, p, o)):
            continue
        num += w - p
        den += w - o
        if w - o > 1e-9:
            each.append(100.0 * (w - p) / (w - o))
        else:
            none += 1
    return each, (100.0 * num / den if den > 1e-9 else float("nan")), none


def table(rows, meter, oracle_key, out):
    base = [r["written"][meter] for r in rows]
    out.append(f"  {'policy':<9} {'total':>10} {'median':>9} {'mean':>9} "
               f"{'tot %vs written':>16} {'median paired %':>16} {'headroom captured':>18}")
    for name in POLICIES + ([oracle_key] if oracle_key not in POLICIES else []):
        xs = [r[name][meter] for r in rows]
        ch = pct_change(xs, base)
        each, agg, flat = capture(rows, name, meter, oracle_key)
        tot_pct = 100.0 * (sum(xs) - sum(base)) / sum(base)
        out.append(f"  {name:<9} {sum(xs):10.1f} {med(xs):9.2f} {sum(xs)/len(xs):9.2f} "
                   f"{tot_pct:+15.2f}% {med(ch):+15.2f}% "
                   f"{agg:9.1f}% tot / {med(each):5.1f}% med")
    flat = capture(rows, "written", meter, oracle_key)[2]
    out.append(f"  (the per-task headroom median is over the {len(rows)-flat} tasks that "
               f"HAVE headroom; {flat} tasks have none - the written order is already "
               f"optimal in this meter)")


def report(rows, meta):
    out = []
    add = out.append
    add("=" * 100)
    add("THE LADDER - what is ordering worth at all?   exp_ladder.py")
    add("=" * 100)
    add(f"tasks: {meta['used']} of 500 (tasks[::{meta['stride']}]), seed={meta['seed']}, "
        f"scenes={meta['scenes']}, subgoal counts {meta['ncount']}")
    add(f"permutations enumerated / simulator runs: {meta['sims']}   "
        f"(every n! ordering of every task, both meters)")
    add("")

    # ---- what had to be dropped, and why
    bad = [r for r in rows if not r["all_sim_ok"]]
    wain = [r for r in bad if r["scene"] == "Wainscott_0_int"]
    add(f"EXCLUSIONS")
    add(f"  tasks where at least one of the n! orderings fails in the simulator: "
        f"{len(bad)}/{len(rows)}  ({len(wain)} of them Wainscott_0_int, "
        f"{len(bad)-len(wain)} elsewhere)")
    add(f"  tasks where EVERY ordering fails: "
        f"{sum(1 for r in rows if not any(p['sim_ok'] for p in r['perms'].values()))}")
    add(f"  non-finite `walked` anywhere: {meta['inf_walked']}   "
        f"non-finite `driven` anywhere: {meta['inf_driven']}")
    add(f"  the driven table below uses the {len(rows)-len(bad)} tasks in which ALL "
        f"orderings run clean, so every policy is compared on the same task.")
    add(f"  the walked table uses all {len(rows)} tasks - the cost model does not depend "
        f"on the simulator succeeding.")
    add("")

    add(f"WALKED  (the cost model's own meter, {len(rows)} tasks, paired)")
    table(rows, "walked", "oracle", out)
    add("")

    clean = [r for r in rows if r["all_sim_ok"]]
    add(f"DRIVEN  (what the 2-D simulator really drives, {len(clean)} tasks, paired)")
    add("  `oracle` is the argmin of walked, as specified; `driven_oracle` is the argmin of")
    add("  driven - the true floor in this meter, which no policy can see.")
    table(clean, "driven", "driven_oracle", out)
    add("")

    # ---- the looser exclusion, as a robustness check on the one above
    loose = [r for r in rows if all(r["perms"][key(r["order"][p])]["sim_ok"]
                                    for p in POLICIES + ["driven_oracle"])]
    add("ROBUSTNESS OF THE EXCLUSION  (driven totals under the looser filter: only the")
    add(f"  orderings the six policies actually pick must run clean - {len(loose)} tasks)")
    table(loose, "driven", "driven_oracle", out)
    add("")

    # ---- the published comparison, in this harness's numbers
    def boot(rs, a, b, meter, draws=10000):
        """Paired bootstrap over tasks of the total-distance % difference a vs b."""
        r = random.Random(12345)
        pairs = [(x[a][meter], x[b][meter]) for x in rs
                 if math.isfinite(x[a][meter]) and math.isfinite(x[b][meter])]
        out = []
        for _ in range(draws):
            sample = [pairs[r.randrange(len(pairs))] for _ in range(len(pairs))]
            top = sum(p[0] for p in sample)
            bot = sum(p[1] for p in sample)
            out.append(100.0 * (top - bot) / bot)
        out.sort()
        return out[int(0.025 * draws)], out[int(0.975 * draws)]

    add("THE PUBLISHED COMPARISON, RE-MEASURED HERE  (gavel against static)")
    for label, rs, meter in (("walked", rows, "walked"), ("driven", clean, "driven")):
        s_, g_ = [r["static"][meter] for r in rs], [r["gavel"][meter] for r in rs]
        ch = pct_change(g_, s_)
        add(f"  {label:<7} total {100.0*(sum(g_)-sum(s_))/sum(s_):+.2f}%   "
            f"median paired {med(ch):+.2f}%   "
            f"tasks where gavel is strictly cheaper: "
            f"{sum(1 for a, b in zip(g_, s_) if a < b - 1e-9)}/{len(rs)}   "
            f"strictly worse: {sum(1 for a, b in zip(g_, s_) if a > b + 1e-9)}/{len(rs)}")
        lo, hi = boot(rs, "gavel", "static", meter)
        add(f"          paired bootstrap 95% CI on that total: [{lo:+.2f}%, {hi:+.2f}%]")
    add("  for scale, the same statistic for the rungs below (driven, clean tasks):")
    for a, b in (("static", "written"), ("oracle", "written"),
                 ("driven_oracle", "written"), ("random", "written")):
        xs = [r[a]["driven"] for r in clean]
        ys = [r[b]["driven"] for r in clean]
        lo, hi = boot(clean, a, b, "driven")
        add(f"    {a:<14} vs {b:<8} total {100.0*(sum(xs)-sum(ys))/sum(ys):+.2f}%   "
            f"95% CI [{lo:+.2f}%, {hi:+.2f}%]")
    add("")

    # ---- how much of the spread the policies live in
    add("WHERE EACH POLICY SITS IN THE FULL SPREAD OF ORDERINGS")
    add(f"  {'policy':<13} {'walked: %ile':>13} {'= best':>8} {'driven: %ile':>14} {'= best':>8}")
    for name in POLICIES + ["driven_oracle"]:
        wp = [r["rank"]["walked"][name] for r in rows]
        dp = [r["rank"]["driven"][name] for r in clean]
        wb = 100.0 * sum(1 for r in rows if r["is_best"]["walked"][name]) / len(rows)
        db = 100.0 * sum(1 for r in clean if r["is_best"]["driven"][name]) / len(clean)
        add(f"  {name:<13} {med(wp):12.1f}% {wb:7.1f}% {med(dp):13.1f}% {db:7.1f}%")
    add("  (percentile = share of this task's orderings that are strictly cheaper; 0 = best)")
    add("")

    # ---- spread, to show ordering CAN matter
    spread_w = [100.0 * (r["worst"]["walked"] - r["best"]["walked"]) / r["best"]["walked"]
                for r in rows if r["best"]["walked"] > 0]
    spread_d = [100.0 * (r["worst"]["driven"] - r["best"]["driven"]) / r["best"]["driven"]
                for r in clean if r["best"]["driven"] > 0]
    add("SPREAD BETWEEN BEST AND WORST ORDERING OF THE SAME TASK")
    add(f"  walked: median {med(spread_w):.1f}%  max {max(spread_w):.1f}%  "
        f"(tasks with zero spread: {sum(1 for s in spread_w if s < 1e-9)})")
    add(f"  driven: median {med(spread_d):.1f}%  max {max(spread_d):.1f}%  "
        f"(tasks with zero spread: {sum(1 for s in spread_d if s < 1e-9)})")
    add("")

    # ---- agreement between the policies
    add("HOW OFTEN THE POLICIES AGREE ON THE ORDER")
    pairs = [("written", "static"), ("written", "gavel"), ("static", "gavel"),
             ("static", "oracle"), ("gavel", "oracle"), ("random", "written")]
    for a, b in pairs:
        same = sum(1 for r in rows if r["order"][a] == r["order"][b])
        add(f"  {a:<8} == {b:<8} {same:4d}/{len(rows)}  {100.0*same/len(rows):5.1f}%")
    add(f"  gavel reordered at least once: "
        f"{sum(1 for r in rows if r['reorders'] > 0)}/{len(rows)}   "
        f"blocked compositions: {sum(r['blocked'] for r in rows)}")
    add("")

    # ---- does the cost model pick the right order?
    add("IS THE COST MODEL PICKING THE ORDER THE SIMULATOR WOULD PICK?")
    hit = sum(1 for r in clean if r["is_best"]["driven"]["oracle"])
    add(f"  walked-optimal order is also driven-optimal: {hit}/{len(clean)}  "
        f"{100.0*hit/len(clean):.1f}%")
    for name in ("static", "gavel"):
        h = sum(1 for r in clean if r["is_best"]["driven"][name])
        add(f"  {name}'s order is driven-optimal: {h}/{len(clean)}  {100.0*h/len(clean):.1f}%")
    ratios = [p["walked"] / p["driven"] for r in clean for p in r["perms"].values()
              if p["driven"] > 0 and math.isfinite(p["walked"])]
    add(f"  cost model / simulator distance ratio, over all {len(ratios)} clean orderings: "
        f"median {med(ratios):.2f}x  (the model charges a whole-room sweep the simulator's "
        f"camera often short-circuits)")
    taus = [r["tau"] for r in clean if r["tau"] is not None]
    add(f"  Kendall tau between walked and driven across a task's orderings: "
        f"median {med(taus):.3f}  (n={len(taus)} tasks with >1 distinct value)")
    add("")

    # ---- what predicts headroom
    add("WHAT DECIDES WHETHER A TASK HAS ANY HEADROOM AT ALL")
    add(f"  task objects whose belief is a point mass (\"known\"): "
        f"{sum(r['belief']['known'] for r in rows)}/"
        f"{sum(r['belief']['objects'] for r in rows)}  "
        f"{100.0*sum(r['belief']['known'] for r in rows)/sum(r['belief']['objects'] for r in rows):.1f}%")
    add(f"  tasks in which EVERY task object is known: "
        f"{sum(1 for r in rows if r['belief']['uncertain'] == 0)}/{len(rows)}")
    add(f"  belief's first guess is the wrong room: "
        f"{sum(r['belief']['wrong_top'] for r in rows)}/"
        f"{sum(r['belief']['objects'] for r in rows)} objects")
    add("")
    add("  by number of UNCERTAIN task objects (driven meter, clean tasks, median paired % vs written)")
    add(f"  {'uncertain':>9} {'tasks':>6} " + " ".join(f"{p:>9}" for p in POLICIES + ["driven_oracle"])
        + f" {'spread':>8}")
    buckets = [(0, 0), (1, 1), (2, 2), (3, 99)]
    for lo, hi in buckets:
        sub = [r for r in clean if lo <= r["belief"]["uncertain"] <= hi]
        if not sub:
            continue
        base = [r["written"]["driven"] for r in sub]
        cells = [f"{med(pct_change([r[name]['driven'] for r in sub], base)):+8.2f}%"
                 for name in POLICIES + ["driven_oracle"]]
        sp = med([100.0 * (r["worst"]["driven"] - r["best"]["driven"]) / r["best"]["driven"]
                  for r in sub if r["best"]["driven"] > 0])
        label = f"{lo}" if lo == hi else f"{lo}+"
        add(f"  {label:>9} {len(sub):>6} " + " ".join(cells) + f" {sp:7.1f}%")
    add("")
    add("  by number of DISTINCT ROOMS the errands touch (same meter)")
    add(f"  {'rooms':>9} {'tasks':>6} " + " ".join(f"{p:>9}" for p in POLICIES + ["driven_oracle"])
        + f" {'spread':>8}")
    for lo, hi in [(1, 1), (2, 2), (3, 3), (4, 99)]:
        sub = [r for r in clean if lo <= r["belief"]["rooms_touched"] <= hi]
        if not sub:
            continue
        base = [r["written"]["driven"] for r in sub]
        cells = [f"{med(pct_change([r[name]['driven'] for r in sub], base)):+8.2f}%"
                 for name in POLICIES + ["driven_oracle"]]
        sp = med([100.0 * (r["worst"]["driven"] - r["best"]["driven"]) / r["best"]["driven"]
                  for r in sub if r["best"]["driven"] > 0])
        label = f"{lo}" if lo == hi else f"{lo}+"
        add(f"  {label:>9} {len(sub):>6} " + " ".join(cells) + f" {sp:7.1f}%")
    add("")

    # ---- by subgoal count
    add("BY NUMBER OF ERRANDS  (median paired % vs written, driven meter, clean tasks)")
    add(f"  {'n':>2} {'tasks':>6} " + " ".join(f"{p:>9}" for p in POLICIES + ["driven_oracle"]))
    for n in sorted({r["n"] for r in clean}):
        sub = [r for r in clean if r["n"] == n]
        base = [r["written"]["driven"] for r in sub]
        cells = []
        for name in POLICIES + ["driven_oracle"]:
            cells.append(f"{med(pct_change([r[name]['driven'] for r in sub], base)):+8.2f}%")
        add(f"  {n:>2} {len(sub):>6} " + " ".join(cells))
    # ---- the one summary that answers the question
    add("THE LADDER, AS A DECOMPOSITION OF THE WRITTEN->TRUE-OPTIMUM GAP")
    add("  Percentage points of the written order's simulator distance, clean tasks. Each")
    add("  rung is what that step of the pipeline is worth on top of the rung below it.")
    base = sum(r["written"]["driven"] for r in clean)
    rungs = [("ordering at all (written -> static)", "written", "static"),
             ("re-optimising online (static -> gavel)", "static", "gavel"),
             ("optimising `walked` exactly instead (gavel -> oracle)", "gavel", "oracle"),
             ("a cost model that matched the simulator (oracle -> driven_oracle)",
              "oracle", "driven_oracle")]
    for label, a, b in rungs:
        gain = sum(r[a]["driven"] - r[b]["driven"] for r in clean)
        add(f"  {label:<60} {100.0*gain/base:6.2f} pts  "
            f"{100.0*gain/(base-sum(r['driven_oracle']['driven'] for r in clean)):5.1f}% of the gap")
    total = base - sum(r["driven_oracle"]["driven"] for r in clean)
    add(f"  {'TOTAL written -> driven-optimal':<60} {100.0*total/base:6.2f} pts  100.0% of the gap")
    add("")
    add("=" * 100)
    return "\n".join(out)


def enrich(row):
    """Attach every derived per-task quantity the report needs."""
    perms = row["perms"]
    row["all_sim_ok"] = all(p["sim_ok"] for p in perms.values())
    walked = {k: v["walked"] for k, v in perms.items()}
    driven = {k: v["driven"] for k, v in perms.items()}
    row["order"] = {k: tuple(v) for k, v in row["order"].items()}
    row["order"]["driven_oracle"] = tuple(
        int(i) for i in min(driven, key=lambda k: (driven[k], k)).split(","))
    for name, perm in row["order"].items():
        row[name] = {"walked": walked[key(perm)], "driven": driven[key(perm)]}
    row["best"] = {"walked": min(walked.values()), "driven": min(driven.values())}
    row["worst"] = {"walked": max(walked.values()), "driven": max(driven.values())}
    row["rank"], row["is_best"] = {}, {}
    for meter, vals in (("walked", walked), ("driven", driven)):
        row["rank"][meter], row["is_best"][meter] = {}, {}
        for name, perm in row["order"].items():
            mine = vals[key(perm)]
            cheaper = sum(1 for v in vals.values() if v < mine - 1e-9)
            row["rank"][meter][name] = 100.0 * cheaper / len(vals)
            row["is_best"][meter][name] = mine <= min(vals.values()) + 1e-9
    ks = sorted(perms)
    a = [walked[k] for k in ks]
    b = [driven[k] for k in ks]
    con = dis = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            s = (a[i] - a[j]) * (b[i] - b[j])
            if s > 0:
                con += 1
            elif s < 0:
                dis += 1
    row["tau"] = (con - dis) / (con + dis) if con + dis else None
    return row


# ---------------------------------------------------------------- driver

def aggregate(shard_dir, stride, seed, quiet=False):
    rows = []
    for name in sorted(os.listdir(shard_dir)):
        if name.startswith("shard-") and name.endswith(".json"):
            with open(os.path.join(shard_dir, name)) as handle:
                rows.extend(json.load(handle))
    rows = [enrich(r) for r in rows]
    rows.sort(key=lambda r: r["id"])
    meta = {"used": len(rows), "stride": stride, "seed": seed,
            "scenes": len({r["scene"] for r in rows}),
            "ncount": sorted({n: sum(1 for r in rows if r["n"] == n)
                              for n in {r["n"] for r in rows}}.items()),
            "sims": sum(len(r["perms"]) for r in rows),
            "inf_walked": sum(1 for r in rows for p in r["perms"].values()
                              if not math.isfinite(p["walked"])),
            "inf_driven": sum(1 for r in rows for p in r["perms"].values()
                              if not math.isfinite(p["driven"]))}
    text = report(rows, meta)
    if not quiet:
        print(text)
    return rows, meta, text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--dir", default="data/ladder")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--shards", type=int)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()

    os.chdir(HERE)
    tasks = json.load(open(args.tasks))[::args.stride]
    os.makedirs(args.dir, exist_ok=True)

    if args.shard is not None:
        # Contiguous blocks, so a worker rebuilds few A* room tables.
        size = math.ceil(len(tasks) / args.shards)
        mine = tasks[args.shard * size:(args.shard + 1) * size]
        worker(mine, args.seed, os.path.join(args.dir, f"shard-{args.shard:02d}.json"))
        return 0

    if not args.aggregate_only:
        for stale in os.listdir(args.dir):
            if stale.startswith("shard-"):
                os.remove(os.path.join(args.dir, stale))
        # The workers run on the CPU, deliberately. The RSN is a small MLP and its frozen
        # BGE-small encoder is rebuilt per object, so ten workers each holding one on the
        # GPU is ~19 GB for arithmetic that takes the same 2 s per task on a core - and it
        # OOMs whatever else is sharing the card. Checked against the GPU path on three
        # tasks: identical `walked` to the last bit, so nothing about the measurement
        # depends on this.
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2",
                   MKL_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        procs = []
        for i in range(args.jobs):
            log = open(os.path.join(args.dir, f"log-{i:02d}.txt"), "w")
            procs.append(subprocess.Popen(
                [sys.executable, os.path.abspath(__file__), "--tasks", args.tasks,
                 "--stride", str(args.stride), "--seed", str(args.seed),
                 "--dir", args.dir, "--shard", str(i), "--shards", str(args.jobs)],
                cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT))
        codes = [p.wait() for p in procs]
        if any(codes):
            print(f"WARNING: shard exit codes {codes}", file=sys.stderr)

    rows, meta, text = aggregate(args.dir, args.stride, args.seed)
    with open(os.path.join(args.dir, "ladder.json"), "w") as handle:
        json.dump({"meta": meta, "rows": rows}, handle, default=str)
    with open(os.path.join(args.dir, "ladder.txt"), "w") as handle:
        handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
