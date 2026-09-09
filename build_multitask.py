#!/usr/bin/env python3
"""Verify the short subtasks, then combine them into the multi-task benchmark.

Verification is `build_tasks.verify` - the same function that verifies the single-task
benchmark, unchanged. It replays every reference plan through `GraphMachine`, drives it in
the simulator, checks the furniture is really in the room the sentence names, checks the
movable is a category OmniGibson can build, and applies the shared in-versus-on rule. This
module adds no checks of its own except the one the builder cannot know about: the
subtasks inside one long instruction must use **distinct movables**, which is what makes
them independent and what lets the expected-cost objective decompose.

    python build_multitask.py --subtasks data/subtasks.json --out data/multitask.json

Neither output is `data/tasks.json`; the single-task benchmark is never written here.
"""

import argparse
import itertools
import json
import random

SUBTASK_TRUTH = "data/subtask_extraction.json"
BENCHMARK = "data/tasks.json"


def verified_subtasks(verbose=False, simulate=True):
    """Every subtask, with the same verification the single-task benchmark gets."""
    from build_tasks import extraction_truth, reconcile_relations, verify

    from subtasks import SUBTASKS

    out, failures = [], []
    for scene, entries in SUBTASKS.items():
        for index, entry in enumerate(entries, 1):
            task = {**entry, "scene": scene, "id": f"{scene}-s{index:02d}"}
            task = reconcile_relations(task)
            # Exactly what `build_tasks` does for the single-task set, from a file in the
            # same format, read by the same function.
            task["extraction"] = extraction_truth(task, path=SUBTASK_TRUTH)
            ok, message = verify(task, verbose=verbose, simulate=simulate)
            print(f"  {'ok  ' if ok else 'FAIL'} {task['id']:22s} "
                  f"{task['task'][:56]:56s} {message}")
            (out if ok else failures).append(task if ok else (task["id"], message))
    return out, failures


def rooms_touched(picked):
    """Every room the errands in this combination act in."""
    return {room for task in picked for room in (task.get("rooms") or {}).values()}


def spread_cap(pool, size):
    """The most rooms any admissible combination of this size can reach in this scene.

    Enumerated rather than guessed, because it varies a lot: `Beechwood_1_int` can reach seven
    rooms with four errands and `Wainscott_0_int` can reach three with any number, since all
    twelve of its subtasks live in three rooms. A fixed target would silently exclude whole
    scenes; a target relative to what the scene can actually do does not.
    """
    best = 0
    for combo in itertools.combinations(pool, size):
        if independent(combo) and one_room_each(combo):
            best = max(best, len(rooms_touched(combo)))
    return best


def independent(picked):
    """No two errands here touch the same object.

    Two errands that never share an object cannot constrain each other, so every ordering is
    legal, there is no precedence to respect, and the expected-cost objective decomposes into a
    pairwise matrix. "No shared object" rather than "distinct movables": a switch task injects
    nothing, and two of them on the same lamp would still collide, so this counts everything a
    subtask acts on - what it spawns and what its goal names.
    """
    touched = [frozenset({s["name"] for s in t["spawn"]} | {g[1] for g in t["goal"]})
               for t in picked]
    return len(set().union(*touched)) == sum(len(x) for x in touched)


def one_room_each(picked):
    """No furniture category may be in two different rooms across these errands.

    The belief is keyed by category: `stated` maps `bookcase` to a room and `populate`
    creates one `bookcase` node. So an instruction that says "the office bookcase" in one
    clause and "the kitchen bookcase" in another cannot be represented - one of the two is
    wrong whatever the planner does, and `NAVIGATE_TO bookcase` has no single referent. It is
    not a hard task, it is an unanswerable one, and 174 of the first 500 instructions were
    built that way.

    The single-task benchmark never hits this because one task names one bookcase. Enforcing
    the same property here is what keeps the two sets comparable. Every scene still admits
    200 to 1,500 combinations of 2 to 5 errands, so nothing is lost but the ambiguity.

    Sharing a destination is still fine - two things can go in the *same* bookcase - because
    that leaves the referent unique and every ordering legal.
    """
    rooms = {}
    for task in picked:
        for name, room in (task.get("rooms") or {}).items():
            if rooms.setdefault(name, room) != room:
                return False
    return True


def merge_extraction(picked):
    """One instruction's stage-1 truth, from the errands it is made of.

    The three classes are disjoint, so merging is not a union of three lists. An object can
    be `uncertain` in one errand and `stated` in another - "the breakfast table" in one
    sentence and "the kitchen breakfast table" in the next - and in the combined instruction
    it *is* stated, because one of the clauses says so. So `stated` and `dependent` win, and
    whatever they claim is removed from `uncertain`.
    """
    stated, dependent, seen = {}, [], set()
    for task in picked:
        stated.update(task["extraction"].get("stated") or {})
        for d in task["extraction"]["dependent"]:
            key = (d["object"], d["relation"], d["target"])
            if key not in seen:
                seen.add(key)
                dependent.append(dict(d))
    spoken = set(stated) | {d["object"] for d in dependent}
    uncertain = sorted({o for t in picked for o in t["extraction"]["uncertain"]} - spoken)
    return {"uncertain": uncertain, "stated": stated, "dependent": dependent}


def combine(subtasks, per_scene=50, seed=0, sizes=(2, 3, 4, 5), live=None,
            live_share=0.7):
    """Long instructions, each several subtasks of one scene done in any order.

    Three properties are enforced here that `build_tasks` cannot: the errands share no object,
    no furniture category appears in two rooms, and the errands are spread across the house.
    The first makes every ordering legal, the second makes every reference unambiguous, and the
    third is what gives the ordering stage anything to win at all.

    The first: the subtasks in a combination **share no object**. Two errands that never touch the same object
    cannot constrain each other, so every ordering is legal, there is no precedence to
    respect, and the expected-cost objective decomposes into a pairwise matrix. Sharing a
    destination is fine - two things can go in the same bookcase - because that still
    leaves the order free.

    Sizes and subtask lengths both vary - 2 to 5 subgoals, each 2, 4 or 6 primitives - so
    a long instruction runs from about 6 to 26 actions. A benchmark where every task had
    the same shape would let a planner learn the shape instead of the task, and would say
    nothing about how ordering scales with the number of errands.

    The reference plan is the subplans concatenated in the order drawn. It is one valid
    ordering, not the cheapest: which ordering is cheapest is the question GAVEL answers,
    so the benchmark must not presuppose it.

    `live`, if given, is a predicate on a finished combination and is the strongest form of
    the spread rule above. `live_share` is the fraction of each scene's instructions to fill
    with it before topping the scene up without it.

    **It is a quota, not a filter.** A benchmark where every instruction is live would say
    nothing about how often the question arises, and it would drop whole scenes:
    `Pomaria_1_int` produced 0 live instructions out of 50, because its subtasks name the room
    outright and leave the RSN almost nothing to guess. So each scene fills its quota if it
    can, tops up with ordinary draws if it cannot, and keeps its place in the benchmark either
    way. A scene that yields nothing live after a fair trial is abandoned early rather than
    spending its whole attempt budget on a question it cannot pose. Spread asks that the errands be far enough apart for *some*
    ordering to beat another; `live` asks the sharper question of whether the arms being
    compared actually decide differently on this instruction - whether carrying the whole room
    distribution picks a different order than carrying its argmax, or whether anything the
    robot sees on the way makes it revise. Measured over the first 500 instructions, 299 were
    dead by that test: every arm chose the identical order, so the instruction contributed
    exactly nothing to the comparison, diluting the reported effect without even adding noise.

    **It selects on whether the decision differs, never on which arm wins.** Selecting on the
    outcome would manufacture the result; selecting on disagreement only concentrates
    instructions where the question is live. It does mean the benchmark is built using the
    estimator under test, and it skews towards more errands and towards scenes with more
    uncertain objects - `Pomaria_1_int` offers only one distinct room guess in total, so
    nothing there can be live. Both are the price of a benchmark that is a diagnostic for
    ordering rather than a representative sample of household chores.
    """
    rng = random.Random(seed)
    by_scene = {}
    for task in subtasks:
        by_scene.setdefault(task["scene"], []).append(task)

    out = []
    for scene, pool in sorted(by_scene.items()):
        caps = {size: spread_cap(pool, size) for size in sizes}
        seen = set()
        # Aim for as much spread as the scene will give and settle for less only when the pool
        # runs out. A single fixed target cannot serve ten houses: asking for two rooms more
        # than there are errands yields 4.6 rooms an instruction in the roomy scenes but leaves
        # `Pomaria_0_int` and `Pomaria_1_int` able to fill only 22 and 26 of their 50, which
        # would quietly unbalance the benchmark. Trying the ambitious target first and dropping
        # to the next only on exhaustion keeps all ten scenes at 50 and still takes whatever
        # spread each one can actually offer.
        # Two passes over the same draw: the first insists on live instructions until the
        # quota is met, the second fills whatever is left with ordinary ones.
        quota = int(round(per_scene * live_share)) if live is not None else 0
        tested, misses = 0, 0
        budget = max(40, 3 * quota)
        for want_live, target in ((quota, quota), (0, per_scene)):
            for bump in (2, 1, 0):
                if len([t for t in out if t["scene"] == scene]) >= target:
                    break
                attempts = 0
                while len([t for t in out if t["scene"] == scene]) < target:
                    if want_live and live is not None and (tested >= budget or misses >= 25):
                        break      # this scene cannot pose the question; stop paying to ask
                    attempts += 1
                    # Bounded: the draw rejects on independence, on room consistency, on
                    # spread and on repeats, so an exhausted pool would otherwise spin here
                    # forever rather than fall back to a looser target.
                    if attempts > 60000:
                        break
                    size = rng.choice(sizes)
                    picked = rng.sample(pool, size)
                    if not independent(picked):
                        continue
                    if not one_room_each(picked):
                        continue
                    # **Spread the errands around the house.** Only the transition term of the
                    # objective depends on the order, so an instruction whose errands all sit
                    # in one corner gives the ordering stage nothing to win however good it is
                    # - and drawing uniformly produces exactly that, because a scene's
                    # manipulable furniture clusters into a few rooms. Measured over 155
                    # instructions, the gap between the best and the worst ordering rises
                    # monotonically with the number of rooms touched: 11.2% at two rooms,
                    # 13.0% at three, 14.3% at four, 18.5% at five, 36.6% at six. Uniform
                    # draws averaged 3.4 rooms.
                    #
                    # The target is capped at what the scene can actually reach, because some
                    # scenes cannot spread at all - every one of `Wainscott_0_int`'s twelve
                    # subtasks is in one of three rooms - and a fixed target would drop such a
                    # scene from the benchmark rather than give it the most spread it has.
                    if len(rooms_touched(picked)) < min(size + bump, caps[size]):
                        continue
                    key = tuple(sorted(t["id"] for t in picked))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidate = {
                        "id": f"{scene}-m00", "scene": scene,
                        "task": ", ".join(t["task"] for t in picked[:-1])
                                + ", and " + picked[-1]["task"],
                        "subgoals": [{"id": t["id"], "task": t["task"], "goal": t["goal"],
                                      "plan": t["plan"]} for t in picked],
                        "spawn": [s for t in picked for s in t["spawn"]],
                        "goal": [g for t in picked for g in t["goal"]],
                        "plan": [step for t in picked for step in t["plan"]],
                        "rooms": {k: v for t in picked
                                  for k, v in (t.get("rooms") or {}).items()},
                        "extraction": merge_extraction(picked),
                    }
                    if want_live and live is not None:
                        tested += 1
                        if not live(candidate):
                            misses += 1
                            continue           # every arm would choose the same order
                        misses = 0
                    out.append(candidate)
        for position, task in enumerate([t for t in out if t["scene"] == scene], 1):
            task["id"] = f"{scene}-m{position:02d}"   # ids must be dense after rejections
        got = len([t for t in out if t["scene"] == scene])
        if got < per_scene:
            print(f"  only {got} of {per_scene} combinations for {scene} - "
                  f"the pool admits no more")
    return out


def ordering_live(task):
    """Do the arms under comparison actually decide differently on this instruction?

    Three cheap estimator runs, no simulator and no language model: order once from the
    argmax, order once from the full distribution, and order online. The instruction is live
    if the first two disagree about the order, or if the online arm ever revises. Anything
    else is an instruction on which GAVEL, GAVEL Static and GAVEL-MAP do literally the same
    thing, and which therefore cannot distinguish them however it is scored.
    """
    import gavel
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
    from world_graph import WorldGraph

    extraction = task["extraction"]
    graph = populate(task["scene"], extraction["uncertain"], extraction["dependent"],
                     stated=extraction.get("stated") or {},
                     model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)
    seed = WorldGraph.from_scene_graph(graph)
    plans = [[tuple(step) for step in sub["plan"]] for sub in task["subgoals"]]
    argmax = gavel.solve(task, plans, graph, seed, reorder=False, collapse=True)
    static = gavel.solve(task, plans, graph, seed, reorder=False)
    online = gavel.solve(task, plans, graph, seed, reorder=True)
    return argmax["order"] != static["order"] or online.get("reorders", 0) > 0


def stamp_of(tasks, seed, per_scene):
    """A fingerprint of exactly which benchmark this is.

    Results measured against different generations of this file are not comparable, and the
    file is regenerated whenever a combination rule changes. That happened mid-analysis once,
    and it silently invalidated a set of numbers that still looked like they belonged together:
    three quarters of the ids kept their name while changing how many errands they held. A
    content hash makes the mismatch loud instead - anything reporting a result can record the
    stamp it ran against, and two results carrying different stamps are known not to be a
    comparison.
    """
    import hashlib

    body = json.dumps([[t["id"], t["task"]] for t in tasks], sort_keys=True).encode()
    rooms = [len({r for sub in t["subgoals"] for r in ()} |
                 set((t.get("rooms") or {}).values())) for t in tasks]
    sizes = {}
    for task in tasks:
        sizes[len(task["subgoals"])] = sizes.get(len(task["subgoals"]), 0) + 1
    return {
        "content": hashlib.sha256(body).hexdigest()[:16],
        "seed": seed, "per_scene": per_scene, "tasks": len(tasks),
        "sizes": {str(k): v for k, v in sorted(sizes.items())},
        "mean_rooms": round(sum(rooms) / max(len(rooms), 1), 2),
        "scenes": sorted({t["scene"] for t in tasks}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtasks", default="data/subtasks.json")
    parser.add_argument("--out", default="data/multitask.json")
    parser.add_argument("--per-scene", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-simulate", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-live", action="store_true",
                        help="keep only instructions on which ordering the errands from the "
                             "full distribution, from its argmax, and online do not all give "
                             "the same answer - see `ordering_live`")
    args = parser.parse_args()
    for path in (args.subtasks, args.out):
        if path == BENCHMARK:
            raise SystemExit(f"refusing to write {BENCHMARK} - that is the single-task set")

    good, bad = verified_subtasks(simulate=not args.no_simulate)
    print(f"\n{len(good)} subtasks verified, {len(bad)} failed")
    for task_id, message in bad:
        print(f"  FAIL {task_id}: {message}")
    if bad:
        return 1
    with open(args.subtasks, "w") as handle:
        json.dump(good, handle, indent=1)
    print(f"wrote {args.subtasks}")
    if args.verify_only:
        return 0

    longs = combine(good, per_scene=args.per_scene, seed=args.seed,
                    live=ordering_live if args.require_live else None)
    print(f"\n{len(longs)} long instructions; verifying each composition")
    from build_tasks import verify
    bad = []
    for task in longs:
        ok, message = verify(task, simulate=not args.no_simulate)
        if not ok:
            bad.append((task["id"], message))
    print(f"  {len(longs) - len(bad)}/{len(longs)} verified")
    for task_id, message in bad[:10]:
        print(f"  FAIL {task_id}: {message}")
    if bad:
        return 1
    stamp = stamp_of(longs, args.seed, args.per_scene)
    with open(args.out, "w") as handle:
        json.dump(longs, handle, indent=1)
    with open(args.out.replace(".json", "-stamp.json"), "w") as handle:
        json.dump(stamp, handle, indent=1)
    print(f"wrote {args.out}")
    print(f"  stamp {stamp['content']}  seed {stamp['seed']}  "
          f"{stamp['tasks']} tasks, mean {stamp['mean_rooms']} rooms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
