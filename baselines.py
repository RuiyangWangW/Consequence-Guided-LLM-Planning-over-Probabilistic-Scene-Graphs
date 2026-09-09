#!/usr/bin/env python3
"""The five methods the multi-task experiment compares, defined once.

Every one of them is handed the same instruction, the same scene and the same subplans, and
differs only in **how it decides the order to do the errands in** and **what it believes
while deciding**. Two axes, and each method is a point on them:

                             order chosen        belief the cost model is given
    ORACLE                   optimal, by truth   the true room of every object
    SAYPLAN                  none - the LLM's    (no cost model at all)
    GAVEL_MAP                once, offline       most likely room only
    GAVEL_STATIC             once, offline       the full distribution
    GAVEL                    online              the full distribution

Read down the last three and each rung changes exactly one thing: `GAVEL_MAP` to
`GAVEL_STATIC` is what the *distribution* is worth over its argmax, holding the ordering
policy fixed; `GAVEL_STATIC` to `GAVEL` is what *revising as perception arrives* is worth,
holding the belief fixed.

`SAYPLAN` isolates what ordering is worth at all: it runs the whole pipeline - decompose,
extract per errand, plan, verify, repair - and then simply does the errands in the order the
model wrote them. `GAVEL_STATIC` adds an ordering stage that runs once against the prior.
`GAVEL` adds revision at every errand boundary as perception comes in. `GAVEL_MAP` orders
once, like `GAVEL_STATIC`, but carries a single best guess per object - the maximum a posteriori
room - rather than a distribution, so it cannot tell a room it is sure about from one it merely
prefers, and a search it should have expected to pay for comes as a surprise. It isolates what
the *distribution* is worth, as against the argmax alone, with the ordering policy held fixed. `ORACLE` is the floor: it knows where everything is,
uses the benchmark's own reference subplans, and picks the ordering that is genuinely shortest
by driving all of them.

`EPOG` is the odd one out and the only baseline that does not ask the model for actions at
all: it collapses the belief to a single graph, subtracts it from the goal graph and reads the
plan off the difference (see `epog.py`). It shares GAVEL's inputs - the same grounded scene
graph, the same goal from the goal adapter - and the same nine primitives, so what separates it
from the others is symbolic planning against language-model planning.

`ORACLE` is not implementable - it enumerates every ordering and measures the truth - and it is
not meant to be. It says how much of the remaining distance is reachable at all, so the gap
between it and `GAVEL` is the part of the problem still open rather than a number to beat.
"""

import itertools
import time

import gavel
from build_tasks import seed_graph

ORACLE = "oracle"
SAYPLAN = "sayplan"
EPOG = "epog"
GAVEL_MAP = "gavel-map"
GAVEL_STATIC = "gavel-static"
GAVEL = "gavel"

#: In the order they should be reported: weakest first, the unreachable floor last.
ALL = (SAYPLAN, EPOG, GAVEL_MAP, GAVEL_STATIC, GAVEL, ORACLE)

LABELS = {
    SAYPLAN: "SayPlan",
    EPOG: "EPoG",
    GAVEL_MAP: "GAVEL-MAP",
    GAVEL_STATIC: "GAVEL Static",
    GAVEL: "GAVEL",
    ORACLE: "Oracle",
}


def uses_llm_plans(method):
    """Does this method plan with the model, or with the benchmark's reference subplans?

    `ORACLE` uses the benchmark's reference plans - it is the floor, and giving it a model's
    plan would measure the model rather than the floor. `EPOG` writes its own from the graph
    diff, which is the whole point of it. Everything else runs the pipeline it is a baseline
    for.
    """
    return method not in (ORACLE, EPOG)


def uses_repair(method):
    """May the world model edit this method's plan?

    `SAYPLAN` sees the machine's complaint and rewrites the plan itself - that is the baseline:
    validation and feedback, no repair. `EPOG` is not checked at all; it plans in one shot and
    the simulator is the only thing that judges it. Everything else gets the mender, at the
    single-errand level and again on the composed plan, which is what the single-task pipeline
    does and what these arms are baselines *for*.
    """
    return method not in (SAYPLAN, EPOG)


def run(method, task, subplans, scene_graph_dict, seed_graph_world, *, drive=None,
        goal=None, verbose=False):
    """Execute one method and return `gavel.solve`'s result dict for it.

    `drive` is a callable taking an ordered list of subplans and returning the simulator's
    driven distance, or None when it fails. `ORACLE` needs it, because its whole definition
    is "the ordering that is shortest when actually driven"; the others do not.

    The result carries `order_seconds`: the wall-clock this method spent **deciding the
    order**, with any simulator time subtracted out. That matters most for `ORACLE`, which
    drives every permutation to make its choice - charging it for that would report the
    simulator's speed rather than the method's, and it is not an implementable method anyway.
    Simulation time is charged separately, at `sim_eval.SIM_STEP_SECONDS` per control step.
    """
    n = len(subplans)
    started = time.perf_counter()
    spent_driving = [0.0]

    if drive is not None:
        inner = drive

        def drive(ordered):
            at = time.perf_counter()
            try:
                return inner(ordered)
            finally:
                spent_driving[0] += time.perf_counter() - at

    result = _dispatch(method, task, subplans, scene_graph_dict, seed_graph_world,
                       drive, goal, verbose)
    result["order_seconds"] = round(time.perf_counter() - started - spent_driving[0], 3)
    return result


def _dispatch(method, task, subplans, scene_graph_dict, seed_graph_world, drive, goal,
              verbose):
    n = len(subplans)

    if method == EPOG:
        # Not an ordering over somebody else's subplans - it writes its own from the graph
        # diff, so it hands back `steps` and the caller drives those directly.
        import epog

        # `goal`, not `goal or task["goal"]`. The fallback read the answer key whenever the
        # caller's goal adapter came back empty - which is every task when no goal model is
        # loaded - so EPoG was scored against the true goal on exactly the runs where its
        # own goal prediction had failed. An empty goal must produce an empty plan.
        steps, cost, scored = epog.plan(task, scene_graph_dict, goal or ())
        # No `orders_valid`: nothing filters these. Every sequence the enumeration produces is
        # executable by construction and the cheapest is returned unexamined - this baseline
        # plans in one shot, and the simulator is the only thing that judges it.
        return {"order": list(range(n)), "steps": steps, "walked": cost,
                "reorders": 0, "blocked": 0, "orders_tried": scored, "orders_valid": None}

    if method == SAYPLAN:
        # No ordering stage. Do them in the order they came out of the decomposition, which
        # for the pipeline is the order the model wrote and for a reference run is the order
        # the instruction names them.
        return gavel.solve(task, subplans, scene_graph_dict, seed_graph_world,
                           force=tuple(range(n)), verbose=verbose)

    if method == GAVEL_STATIC:
        return gavel.solve(task, subplans, scene_graph_dict, seed_graph_world,
                           reorder=False, verbose=verbose)

    if method == GAVEL:
        return gavel.solve(task, subplans, scene_graph_dict, seed_graph_world,
                           reorder=True, verbose=verbose)

    if method == GAVEL_MAP:
        # Ordered once, from the argmax, and then followed to the end. No revision: a planner
        # that believes it knows where everything is has no reason to reconsider, because
        # nothing it sees can contradict a belief it holds with certainty.
        return gavel.solve(task, subplans, scene_graph_dict, seed_graph_world,
                           reorder=False, collapse=True, verbose=verbose)

    if method == ORACLE:
        return oracle(task, subplans, seed_graph_world, drive, verbose=verbose)

    raise ValueError(f"unknown method {method!r}; expected one of {ALL}")


def oracle(task, subplans, seed_graph_world, drive, verbose=False):
    """The shortest ordering there is, measured rather than estimated.

    Knows where every object is, so nothing is searched for; uses the benchmark's own
    reference subplans, so nothing is mis-planned; and chooses the ordering by **driving all
    of them** and keeping the shortest. That last part is why it cannot be implemented: it
    reads the answer off the outcome. It is the floor the others are measured against.

    At two to five errands this is at most 120 drives per instruction, which is affordable.
    """
    n = len(subplans)
    truth = seed_graph(task)
    best, best_order, best_walked = None, None, None
    for order in itertools.permutations(range(n)):
        ordered = [subplans[i] for i in order]
        driven = drive(ordered) if drive else None
        if driven is None:
            continue
        if best is None or driven < best:
            # `walked` is reported for comparability with the other methods; the ordering
            # itself is chosen on `driven`, which is the definition.
            result = gavel.solve(task, subplans, truth, seed_graph_world, force=order)
            best, best_order, best_walked = driven, order, result["walked"]
    if best is None:
        result = gavel.solve(task, subplans, truth, seed_graph_world,
                             force=tuple(range(n)))
        result["driven"] = None
        return result
    result = gavel.solve(task, subplans, truth, seed_graph_world, force=best_order)
    result["driven"] = best
    result["walked"] = best_walked
    return result
