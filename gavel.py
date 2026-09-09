#!/usr/bin/env python3
"""Multi-task planning: decompose, plan each part, order them by expected cost, execute.

This is the single-task pipeline with three things added around it, and nothing inside it
changed. `replan.run` still writes and mends one plan for one instruction; `scene_graph`
still grounds; `GraphMachine` still validates; the 2-D simulator still judges. What is new:

    decompose    one instruction naming several errands becomes several instructions
    order        the errands are sequenced by what they are expected to *cost to walk*
    reorder      that sequence is recomputed at every subplan boundary, because executing
                 one errand reveals where other errands' objects are

The last is the only part that needs the robot to be embodied at all. A symbolic planner
has no reason to prefer one order over another - every ordering satisfies the same goal.
The order matters because the robot has to walk, and it has to *search*: it does not know
where the mug is, only a distribution over rooms, so reaching it costs an expected sweep
rather than a known distance. Sweeping the kitchen for the mug also reveals the bowl, and
that is what makes the remaining errands cheaper in one order than another.

**Why reordering online is not the same as ordering once.** The order chosen up front is
optimal against the *prior*. The moment the robot sweeps a room, the prior is wrong: every
object it saw is now localized and every object it did not see has lost a candidate. If the
plan is fixed at the start, that information is thrown away. Recomputing costs one
`N^2` matrix rebuild per boundary, which at 2-5 subgoals is milliseconds.
"""

import functools
import itertools
import json
import re
import time

import cost_matrix
import order as ordering
from build_tasks import seed_graph
from graph_machine import GraphMachine
from search_cost import Beliefs, rollout
from search_cost import ETA as ETA_SWEEP
from world_graph import WorldGraph

DECOMPOSE = """Split the instruction into the separate errands it asks for.

Write one errand per line, numbered. Copy the wording of the instruction - do not paraphrase
and do not add steps. Keep an errand whole even when it takes several clauses: a clause that
refers back to something already named continues the errand rather than starting a new one.

Instruction: put the book on the shelf, take the pie from the counter, heat it in the oven, and
put it on the table, and switch on the lamp

Errands:
1. put the book on the shelf
2. take the pie from the counter, heat it in the oven, and put it on the table
3. switch on the lamp

Instruction: {task}

Errands:"""


def decompose(text, generator, max_new_tokens=400):
    """The instruction, split into one sentence per errand.

    Falls back to splitting on the conjunctions the generator was given, so a model that
    refuses to number its output still produces something plannable rather than one
    enormous errand.

    **Each errand comes back lowercased**, to match the only convention every input the
    adapters were trained on obeys: all 8,000 training rows, all 100 single tasks and all 500
    instructions are lowercase, with no exceptions, while a model writing a numbered list
    capitalises the first word out of habit.

    This is normalisation, not a fix for anything measured. The suspicion was that the stray
    capital degraded the goal adapter - it is the only uppercase character that adapter is
    ever asked to read - but on 40 errands it answered identically either way, 40/40 correct
    both times. The lowercasing stays because feeding a fine-tuned model text outside its
    training distribution is not worth the coin flip, not because it bought anything here.
    """
    reply = generator(DECOMPOSE.format(task=text), max_new_tokens)
    parts = []
    for line in reply.splitlines():
        line = line.strip()
        match = re.match(r"^\(?(\d+)[.):]\s*(.+)$", line)
        if match and match.group(2).strip():
            parts.append(match.group(2).strip().rstrip(",;."))
    if len(parts) < 2:
        parts = [p.strip().rstrip(",;.") for p in re.split(r",\s+and\s+|,\s*(?=(?:put|take|get|bring|move|place|turn|switch|wash|cook|dry)\b)|\s+and\s+then\s+", text) if p.strip()]
    return [p.lower() for p in parts]


def observe(beliefs, truth, room, names, reachable=None):
    """What sweeping one room tells you about every object you were still looking for.

    A sweep is not a query about one object. The robot walks the room and sees what is in
    it, so *every* object of interest is resolved by that one visit: the ones that are
    there become localized, and the ones that are not lose that room from their candidate
    list. Dropping a candidate is worth as much as finding one - it is why the second
    errand in a room is nearly free, and why the estimate of a *third* errand elsewhere
    goes up once its likeliest room has been ruled out.

    When ruling out leaves an object with no candidate the robot can actually walk to, the
    belief has been refuted rather than sharpened: the object exists, so it is somewhere
    the robot has not looked. The distribution is reset to the rooms it can reach and has
    not yet ruled out, which is what an exhaustive searcher would do next. Without this the
    estimate goes infinite and every ordering ties - not because the errand is impossible,
    but because the guesser ran out of guesses.
    """
    from search_cost import _room_of

    seen = set()
    for name in names:
        if name in beliefs.located:
            continue
        where = _room_of(truth, name)
        if where is None:
            continue
        if where == room:
            beliefs.place(name, room)
            seen.add(name)
            continue
        if name not in beliefs.prior or room not in beliefs.prior[name]:
            beliefs.ruled.setdefault(name, set()).add(room)
            continue
        beliefs.ruled.setdefault(name, set()).add(room)
        del beliefs.prior[name][room]
        left = beliefs.prior[name]
        if reachable is not None and not (set(left) & reachable):
            # Refuted. Fall back to what the object's own RSN ranking said before the task
            # overrode it - a stated room being wrong is no reason to forget where this kind
            # of object usually lives - and failing that, sweep what is left.
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


def walk(plan, world, real, beliefs, distance, search, watching, start=None,
         reachable=None):
    """Execute one subplan against the *truth*, paying what the search actually costs.

    Two graphs move together here. `real` is the world as it is - execution advances it,
    and every room lookup asks it, because that is where objects genuinely are. `world` is
    the world as the robot believes it, advanced by the same steps so the estimator keeps
    scoring later orderings against a belief that has kept up with what the plan did.

    Reading rooms off the belief instead of the truth is the mistake this function is
    written to avoid: it makes the robot "find" a mug in whichever room the RSN guessed,
    so a wrong guess is never paid for and never corrected, and the whole point of
    reordering - that looking costs something and teaches you something - disappears.

    The estimator in `search_cost` charges an expectation over the belief. This charges
    what the robot really walks: it sweeps candidate rooms in ranked order until it reaches
    the room the object is genuinely in, observing each on the way. A lucky guess is cheap,
    an unlucky one is expensive, and both leave the belief better than they found it.

    Returns `(cost, end_room, world_after, real_after, seen)`.
    """
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
                # The same order the estimator charges for: likeliest first, nearest among
                # equals. Executing in a different order than was costed would make the
                # ordering stage's numbers describe a walk the robot never takes.
                route = sorted((r for r in ranked if distance(here, r) != float("inf")),
                               key=lambda r: (-dist.get(r, 0.0), distance(here, r), r))
            if target and target not in route:
                route.append(target)
            arrived = False
            for room in route:
                step = distance(here, room)
                if step == float("inf"):
                    # A room A* cannot reach from where the robot is standing. It is skipped
                    # rather than costed, because the robot cannot go there.
                    continue
                # What arriving costs, charged the way `search_cost` estimates it, so the two
                # meters describe the same walk. `expected_nav` charges a *known* object as a
                # plain drive with no sweep at all; a room entered while still searching costs
                # a full sweep; and the room the object is finally found in costs the partial
                # sweep `ETA` of one. Charging a sweep for a known object too - which this did -
                # added a term the estimator never charged, inflating `walked` by about a third
                # without being able to change any ordering, so the two meters could not be
                # compared even in principle.
                sweep = (0.0 if known is not None and room == target
                         else search(room) if room != target
                         else ETA_SWEEP * search(room))
                total += step + sweep
                here = room
                seen |= observe(beliefs, actual.graph, room, watching, reachable)
                if room == target:
                    arrived = True
                    break
            # **Only move the robot if it actually got there.** This used to set `here =
            # target` unconditionally, just past a loop that `continue`s over every leg A*
            # cannot walk - so an object behind a wall was reached for free, and, worse, every
            # later leg of the run was then measured from a room the robot had never entered.
            # It fired on 91 of 500 instructions - all fifty `Wainscott_0_int` and forty-one
            # `Pomaria_0_int` - across 145 navigation legs, and because the free teleport
            # skipped the sweep it *understated* distance for both arms unevenly: excluding the
            # affected tasks moves the measured reordering gain from +1.01% to +1.49%.
            #
            # When the robot cannot get there it stays where it is and learns nothing, which is
            # what really happens. The plan is then wrong about the world rather than the
            # measurement being wrong about the plan, and the simulator says so when it drives.
            if arrived:
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


def compose(graph, plans, goal):
    """Does running these subplans back to back, in this order, actually do the whole job?

    Each subplan was validated on its own against a graph in its starting state. That is not
    the same question as whether they compose: the second subplan runs in a world the first
    one changed. This is the only check that asks the machine about the *concatenation*, and
    it is asked against the union of the subgoals, so a subplan that quietly undoes another's
    goal is caught here and nowhere else.
    """
    steps = [s for plan in plans for s in plan]
    outcome = GraphMachine(graph.copy()).run(steps, goal)
    return steps, outcome


@functools.lru_cache(maxsize=None)
def _table(scene):
    """A* between every pair of room centroids. Seconds per scene, and it never changes -
    so it is built once and shared by every arm and every task in that house."""
    return cost_matrix.build(scene)


def _costs(scene):
    table = _table(scene)
    return (table,
            lambda a, b: table["distance"].get(f"{a}|{b}", float("inf")),
            lambda r: table["search"].get(r, 0.0))


def solve(task, subplans, scene_graph_dict, graph, *, reorder=True, verbose=False,
          force=None, collapse=False):
    """Stages 7 and 9-11: order the subplans, then commit-execute-observe-reorder.

    `reorder=False` is the ablation - the order is chosen once against the prior and then
    followed to the end, which is what a planner that sequences offline does. Everything
    else is identical, so the difference between the two arms is the online information and
    nothing else.

    **Note what that does and does not measure.** `reorder=False` still *optimises* on its
    first pass; it simply never revises. So the two arms differ by the online increment on
    top of an already-optimal order, not by the value of ordering at all. To measure that,
    pass `force` - an explicit permutation the executor must follow, with no optimisation -
    and compare against either arm. `force=tuple(range(n))` is "do them in the order the
    instruction happens to name them", which is the honest no-ordering baseline.
    """
    table, distance, search = _costs(task["scene"])
    truth = WorldGraph.from_scene_graph(seed_graph(task))
    watching = sorted({o for plan in subplans for a, o in plan if o})
    def mk():
        beliefs = Beliefs(scene_graph_dict, table["rooms"])
        if collapse:
            # Most likely room only. The tail is what tells the estimator an object is
            # *uncertain* - a sharp belief and a flat one over the same argmax cost the same
            # to reach under this, and differ by a whole room sweep under the full one.
            for name, dist in list(beliefs.prior.items()):
                if dist:
                    beliefs.prior[name] = {max(dist, key=dist.get): 1.0}
        return beliefs

    # The robot's own connected component. Half the scenes are in more than one piece, and
    # a room it cannot walk to is not somewhere it can be told to look.
    from search_cost import _room_of
    origin = _room_of(truth, "robot") or (table["rooms"][0] if table["rooms"] else None)
    reachable = {r for r in table["rooms"] if distance(origin, r) != float("inf")}

    beliefs = mk()
    remaining = list(range(len(subplans)))
    world, real, here = graph.copy(), truth.copy(), None
    sequence, walked, estimates, reorders, blocked = [], 0.0, [], 0, 0

    while remaining:
        plans = [subplans[i] for i in remaining]
        # Re-estimate from the belief as it stands *now*. On the first pass this is the
        # prior; on later passes it carries every room the robot has already looked in.
        snapshot = {"prior": {k: dict(v) for k, v in beliefs.prior.items()},
                    "located": dict(beliefs.located), "found": set(beliefs.found),
                    "ruled": {k: set(v) for k, v in beliefs.ruled.items()},
                    "fallback": {k: dict(v) for k, v in beliefs.fallback.items()}}

        def factory():
            fresh = mk()
            fresh.prior = {k: dict(v) for k, v in snapshot["prior"].items()}
            fresh.located = dict(snapshot["located"])
            fresh.found = set(snapshot["found"])
            fresh.ruled = {k: set(v) for k, v in snapshot["ruled"].items()}
            fresh.fallback = {k: dict(v) for k, v in snapshot["fallback"].items()}
            return fresh

        # The ablation follows the order its first optimisation chose, so after that first
        # pass there is nothing for it to decide - and building the matrix and testing every
        # ordering only to throw the answer away costs `n!` composition checks a boundary.
        settled = force is not None or (not reorder and sequence)
        if force is not None:
            # An order handed down from outside: follow it, optimise nothing. `remaining`
            # is rewritten in the caller's order on the first pass, so from then on the
            # head of `remaining` is always the next errand it asked for.
            if not sequence:
                remaining = [remaining[i] for i in force]
                plans = [subplans[i] for i in remaining]
            pick, estimate = tuple(range(len(plans))), 0.0
            A = head = None
        elif settled:
            pick, estimate = tuple(range(len(plans))), estimates[-1] if estimates else 0.0
            A = head = None
        else:
            A, head = ordering.pairwise(plans, world, factory, distance, search,
                                        start={here: 1.0} if here else None)
        # Stage 8, asked of every ordering the executor might adopt rather than once at the
        # start: take the cheapest order whose concatenation the machine says still runs.
        # Reordering is only sound because the errands are independent, and with subplans a
        # model wrote that independence is an expectation, not a guarantee - one that opens
        # a cabinet in one errand and reaches into it in another has made two errands
        # dependent whatever the instruction said. Checking here means a bad decomposition
        # costs distance instead of correctness.
        if not settled:
            pick, estimate = None, float("inf")
            for candidate in ordering.ranked(A, head, len(plans)):
                trial = [plans[i] for i in candidate]
                if compose(world, trial, ())[1].failed_at is None:
                    pick, estimate = candidate, ordering.score(candidate, A, head)
                    break
            if pick is None:
                pick, estimate = ordering.best_order(A, head, len(plans))
                blocked += 1
        estimates.append(estimate)

        if not settled and sequence and pick != tuple(range(len(plans))):
            # A reorder is any disagreement with the schedule already in hand, not just a
            # change of what comes next. Swapping the last two errands is a decision the
            # online information caused just as much as swapping the first two, and counting
            # only `pick[0] != 0` reports zero for runs that visibly resequenced.
            reorders += 1

        chosen = remaining[pick[0]]
        cost, here, world, real, seen = walk(subplans[chosen], world, real, beliefs,
                                             distance, search, watching, start=here,
                                             reachable=reachable)
        walked += cost
        sequence.append(chosen)
        remaining = [remaining[i] for i in pick[1:]]
        if verbose:
            print(f"    did subgoal {chosen} for {cost:.1f} m, now in {here}"
                  + (f", saw {sorted(seen)}" if seen else ""))

    return {"order": sequence, "walked": walked, "estimated": estimates[0] if estimates else 0.0,
            "reorders": reorders, "blocked": blocked, "end": here}
