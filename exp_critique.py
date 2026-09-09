#!/usr/bin/env python3
"""exp_critique.py - adversarial cross-check of six GAVEL post-hoc reports.

Settles, on the FULL 500-task benchmark wherever possible:
  1. the paired gavel-vs-static gap on BOTH meters (executor `walked` and simulator `driven`)
     -- the six reports quote -1.67%, -0.03%, +0.59%, +0.66%, +1.21%, +1.72%, +2.11%;
  2. the scope and effect of the `gavel.walk` unreachable-leg behaviour (adversarial H6);
  3. whether n=2 tasks really have "literally zero headroom" (exp_decomposition) or leave
     +9.8% on the table (exp_adversarial);
  4. walked-vs-driven rank agreement, measured the SAME way for every n.

Writes ONE new file. No existing module is modified; the two monkeypatches are an
instrumented copy of gavel.walk (asserted bit-identical to the shipped one) and lru_cache
on three pure loader functions.
"""
import argparse, functools, itertools, json, math, os, sys, time
import statistics as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------- memoise pure loaders
import query_rsn, embed_categories
query_rsn.load = functools.lru_cache(maxsize=None)(query_rsn.load)
_raw_predict = query_rsn.predict_rooms
_pr_cache = {}
def _predict_rooms(model, ckpt, name, device):
    key = (id(model), name, str(device))
    if key not in _pr_cache:
        _pr_cache[key] = _raw_predict(model, ckpt, name, device)
    return _pr_cache[key]
query_rsn.predict_rooms = _predict_rooms
_raw_embed = embed_categories.embed_names
_emb_cache = {}
def _embed_names(names, model_name=embed_categories.DEFAULT_MODEL, batch_size=64):
    key = (tuple(names), model_name)
    if key not in _emb_cache:
        _emb_cache[key] = _raw_embed(list(names), model_name, batch_size)
    return _emb_cache[key]
embed_categories.embed_names = _embed_names

import gavel, cost_matrix, order as ordering
from graph_machine import GraphMachine
from search_cost import ETA as ETA_SWEEP
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from world_graph import WorldGraph
from sim_eval import run_plan

INF = float("inf")
TRACE = {"teleport": 0, "skip": 0, "legs": 0, "teleport_rooms": []}


def walk_instrumented(plan, world, real, beliefs, distance, search, watching, start=None,
                      reachable=None):
    """Byte-for-byte gavel.walk with three counters added. Asserted equal to the original."""
    from search_cost import _room_of
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
            reached = False
            TRACE["legs"] += 1
            for room in route:
                step = distance(here, room)
                if step == INF:
                    TRACE["skip"] += 1
                    continue
                total += step + (search(room) if room != target else ETA_SWEEP * search(room))
                here = room
                seen |= gavel.observe(beliefs, actual.graph, room, watching, reachable)
                if room == target:
                    reached = True
                    break
            if target:
                if not reached:
                    # `here = target` below runs anyway: the robot is teleported, free.
                    TRACE["teleport"] += 1
                    TRACE["teleport_rooms"].append((here, target))
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


def reset_trace():
    TRACE["teleport"] = TRACE["skip"] = TRACE["legs"] = 0
    TRACE["teleport_rooms"] = []


def snap():
    return dict(teleport=TRACE["teleport"], skip=TRACE["skip"], legs=TRACE["legs"],
                rooms=list(TRACE["teleport_rooms"]))


def load_task(t):
    ex = t["extraction"]
    sg = populate(t["scene"], ex["uncertain"], ex["dependent"], stated=ex.get("stated") or {},
                  model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    g = WorldGraph.from_scene_graph(sg)
    plans = [[tuple(s) for s in sub["plan"]] for sub in t["subgoals"]]
    goal = [tuple(x) for x in t["goal"]]
    return sg, g, plans, goal


def arm(t, plans, sg, g, goal, *, reorder=None, force=None, sim=True):
    reset_trace()
    kw = {"force": force} if force is not None else {"reorder": reorder}
    r = gavel.solve(t, plans, sg, g, **kw)
    tr = snap()
    out = {"order": list(r["order"]), "walked": r["walked"], "estimated": r["estimated"],
           "reorders": r["reorders"], "blocked": r["blocked"],
           "teleport": tr["teleport"], "skip": tr["skip"], "legs": tr["legs"]}
    if sim:
        steps, outcome = gavel.compose(g, [plans[i] for i in r["order"]], goal)
        s = run_plan(t, sg, steps, verbose=False)
        out["driven"] = s["driven"]
        out["ok"] = bool(s["ok"])
        out["why"] = s.get("why", "")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-enum-max-n", type=int, default=3,
                    help="simulate every permutation for tasks with at most this many errands")
    a = ap.parse_args()

    tasks = json.load(open("data/multitask.json"))
    idx = list(range(0, len(tasks), a.stride))
    idx = [i for k, i in enumerate(idx) if k % a.nshards == a.shard]

    # verify the instrumented walk reproduces the shipped one before patching anything
    check = []
    for i in idx[:3]:
        t = tasks[i]
        sg, g, plans, goal = load_task(t)
        base = gavel.solve(t, plans, sg, g, reorder=True)
        gavel.walk = walk_instrumented
        mine = gavel.solve(t, plans, sg, g, reorder=True)
        gavel.walk = _shipped_walk
        check.append((base["walked"] - mine["walked"], base["order"] == mine["order"]))
    assert all(abs(d) < 1e-12 and same for d, same in check), check
    gavel.walk = walk_instrumented

    rows = []
    t0 = time.time()
    for k, i in enumerate(idx):
        t = tasks[i]
        try:
            sg, g, plans, goal = load_task(t)
        except Exception as e:
            rows.append({"i": i, "error": f"populate:{e}"})
            continue
        n = len(plans)
        row = {"i": i, "n": n, "scene": t["scene"], "id": t.get("id", "")}
        # belief sharpness
        try:
            from search_cost import Beliefs
            tbl = gavel._table(t["scene"])
            b = Beliefs(sg, tbl["rooms"])
            objs = [o for p in plans for act, o in p if o]
            objs = sorted(set(objs))
            unc = sum(1 for o in objs if len(b.prior.get(o, {})) > 1)
            row["objs"], row["uncertain"] = len(objs), unc
        except Exception as e:
            row["objs"] = row["uncertain"] = None
        try:
            row["static"] = arm(t, plans, sg, g, goal, reorder=False)
            row["gavel"] = arm(t, plans, sg, g, goal, reorder=True)
            row["ident"] = arm(t, plans, sg, g, goal, force=tuple(range(n)))
        except Exception as e:
            row["error"] = f"arm:{e}"
            rows.append(row)
            continue
        # every permutation on the executor meter; and in the simulator for small n
        perms = {}
        do_sim = n <= a.sim_enum_max_n
        for p in itertools.permutations(range(n)):
            try:
                e = arm(t, plans, sg, g, goal, force=p, sim=do_sim)
            except Exception as ex:
                e = {"error": str(ex)}
            perms["".join(map(str, p))] = e
        row["perms"] = perms
        row["perm_sim"] = do_sim
        rows.append(row)
        if k % 10 == 0:
            print(f"[{a.shard}] {k+1}/{len(idx)} task {i} n={n} {time.time()-t0:.0f}s", flush=True)
    json.dump(rows, open(a.out, "w"))
    print(f"[{a.shard}] wrote {a.out}: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)


_shipped_walk = gavel.walk
if __name__ == "__main__":
    main()
