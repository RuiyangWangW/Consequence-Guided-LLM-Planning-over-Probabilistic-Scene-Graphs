#!/usr/bin/env python3
"""An EPoG-style symbolic planner: the plan is the difference between two graphs.

GAVEL asks a language model for the actions and then checks them. This asks nothing, and it is
never checked: it takes the same two inputs GAVEL gets - the grounded scene graph from stage 3
and the goal state from the goal adapter - collapses the belief to a single deterministic
graph, subtracts it from the goal graph, and reads the actions off the difference.

    Ghat_b    every uncertain object at its maximum a posteriori room, so the world is a fact
              rather than a distribution. `scene_graph.populate` already records the argmax as
              each object's `room`, so collapsing the belief is reading that field.
    G_g       the goal state as a partial graph: the edges and flags that must hold at the end
    K         the manipulation actions that difference implies, as an unordered set:

                  remove a support relation      ->  GRASP
                  insert on_top(o, y)            ->  PLACE_ON_TOP(y)
                  insert object_inside(o, y)     ->  PLACE_INSIDE(y)
                  door    closed -> open         ->  OPEN
                  door    open -> closed         ->  CLOSE
                  switch  off -> on              ->  TOGGLE_ON
                  switch  on -> off              ->  TOGGLE_OFF
                  drop a held object, nowhere    ->  RELEASE

              and nothing else. There is no entry for a state an appliance confers.

Precedence comes from the action semantics - `GRASP(o)` before `PLACE(y)` for the same object,
`OPEN(c)` before reaching into a shut container and `CLOSE(c)` after. That yields a dependency
DAG; the planner takes the cheapest of its topological orders under `C_MAP` and only then
inserts navigation.

WHAT IT CANNOT SAY

The list above is the whole of it, and it is a list of *graph edits*. A goal condition that
names no edge to add or remove has no image in it and is dropped - `unsupported()` reports
which. `cooked`, `washed` and `dried` are exactly that case: nothing about the graph says that
"washed" comes about by putting a thing in a washing machine, shutting the door and running it.
That is a fact about appliances, and encoding it here would be handing this baseline a
nine-step template for the hardest instructions in the benchmark - knowledge the language model
it is compared against has to infer from the sentence. So it is not encoded, and the twenty-two
single-task instructions that ask for a conferred state are simply beyond this planner: it will
fetch the pie and put it on the table without ever heating it, and the simulator will mark the
goal unmet. That is the honest boundary of planning by graph difference.

**Navigation is not a graph edit.** It appears in the emitted plan as an atomic action, but
only to discharge the `near(x)` precondition the manipulation primitives carry: after the
ordering is fixed, a `NAVIGATE_TO` is inserted before every action whose stance differs from
where the robot already is. Consecutive actions on one receptacle therefore share a single
drive.

WHAT LIMITS THE ORDERING SEARCH, AND WHAT DOES NOT

The robot has one hand. That is a *resource* constraint rather than a precedence constraint, so
it does not live in the DAG - but it decides which topological orders are executable, and
enumerating the orders it forbids would be waste. Between a `GRASP` and the `PLACE` that
empties the hand, no second object's chain can begin. So the executable topological orders are
exactly:

    every ordering of the per-object chains, with the hand-free actions - `TOGGLE_ON`,
    `TOGGLE_OFF`, `OPEN`, `CLOSE` on a switch or door that is a goal in its own right -
    inserted at *any* point, including in the middle of a chain.

That last clause matters and an earlier version of this file did not have it. `TOGGLE_ON`'s
preconditions in `GraphMachine` are "beside it" and "it has a switch" - **the hand may be
full** - so switching a lamp on while carrying a newspaper past it is legal, and can be
strictly cheaper than any arrangement of whole chains. 107 of the 500 multi-task instructions
have a switch goal alongside placements, so refusing to interleave understated this baseline on
a fifth of them.

NO VALIDITY FILTER

Every sequence this enumerates is executable by construction, so it is scored by `C_MAP` alone
and the cheapest is returned unexamined. There is deliberately no `GraphMachine` pass to reject
candidates and no repair loop: this baseline plans in one shot and the simulator is the only
thing that ever judges it, which is the point of comparing it against a pipeline that validates
and mends. (The filter this replaced was measured to be inert in any case - over 50
instructions it accepted 2106 of 2116 orderings, and the split was all-or-nothing: on 47
instructions every ordering passed and on 3 none did, so it never once chose between a good
ordering and a bad one.)
"""

import itertools

from planner import OPENABLE
from world_graph import WorldGraph

#: What a placement goal turns into.
PLACE_FOR = {"on_top": "PLACE_ON_TOP", "object_inside": "PLACE_INSIDE"}

#: A ceiling on the sequences scored, so a task shape nobody has written yet cannot hang the
#: run. Five chains is 120 orderings, and one hand-free action inserted into a twenty-action
#: sequence multiplies that by twenty-one.
MAX_ORDERS = 20000


def _category(graph, name):
    return (graph.objects.get(name) or {}).get("category") or name


def _openable(graph, name):
    return _category(graph, name) in OPENABLE


def _support_of(graph, name):
    """What `name` is resting on or sitting in, as `(edge_type, support)`, or `None`."""
    for edge in ("object_inside", "on_top"):
        for _, support in graph.edges_of(edge, src=name):
            return edge, support
    return None


def _fetch(graph, obj):
    """Actions ending with `obj` in the hand, opening and re-shutting whatever holds it.

    Each is `(action, argument, stance)`, where the stance is what the robot must be standing
    at for the action to be legal - the container rather than the object when the object is
    shut inside one, because `NAVIGATE_TO` refuses an object it cannot see and arriving at the
    container brings its contents within reach.
    """
    held_by = _support_of(graph, obj)
    if held_by and held_by[0] == "object_inside" and _openable(graph, held_by[1]):
        box = held_by[1]
        return [("OPEN", box, box), ("GRASP", obj, box), ("CLOSE", box, box)]
    return [("GRASP", obj, obj)]


def _put(graph, obj, relation, destination):
    """Actions ending with `obj` in the goal relation to `destination`."""
    action = PLACE_FOR[relation]
    if relation == "object_inside" and _openable(graph, destination):
        return [("OPEN", destination, destination),
                (action, destination, destination),
                ("CLOSE", destination, destination)]
    return [(action, destination, destination)]


def _leaves_hand_full(chain):
    """Does this chain end holding something? `GRASP` fills the hand, a place or a release
    empties it, and the machine refuses a second `GRASP` while the first thing is still held -
    so a chain that ends full would break whichever chain runs next."""
    full = False
    for action, _, _ in chain:
        if action == "GRASP":
            full = True
        elif action in ("PLACE_ON_TOP", "PLACE_INSIDE", "RELEASE"):
            full = False
    return full


def edits(graph, goal):
    """The goal graph minus the current graph, as `(chains, free)`.

    A *chain* is the manipulation actions for one object, in the only order they can run.
    Conditions on the same object belong to one chain, not several: "heat the pie and put it on
    the table" is a single errand whose order is forced - the pie is cooked while it is in the
    oven and placed afterwards - and splitting it would let the enumeration try to place the
    pie before cooking it, which no ordering can repair.

    A *free* action is a switch or door that is a goal in its own right. It needs no hand and
    has no dependency, so it is not tied to any chain and may run at any point.
    """
    placements, free = {}, []
    for edge_type, src, dst in goal:
        if edge_type in PLACE_FOR:
            placements[src] = (edge_type, dst)
        elif edge_type == "toggled":
            free.append(("TOGGLE_ON" if dst else "TOGGLE_OFF", src, src))
        elif edge_type == "open":
            free.append(("OPEN" if dst else "CLOSE", src, src))
        # Anything else is outside what a graph edit can say. `cooked`, `washed` and `dried`
        # land here: they name no edge to add or remove, and the sequence that brings them
        # about is a fact about ovens and washing machines rather than about the graph. This
        # planner does not know that fact, so it plans nothing for them - see `unsupported`.

    chains = []
    for obj in placements:
        relation, destination = placements[obj]
        if graph.has_edge(relation, obj, destination):
            continue                 # already true; nothing to change
        chain = _fetch(graph, obj) + _put(graph, obj, relation, destination)
        if chain:
            # "Remove a held object without placement" is the ninth edit, and this is where it
            # arises: a goal that only asks for an object to be cooked leaves it in the hand,
            # and the machine then refuses the next chain's GRASP. The stance is None because
            # the hand can be emptied wherever the robot happens to be standing.
            if _leaves_hand_full(chain):
                chain.append(("RELEASE", None, None))
            chains.append(chain)
    return chains, free


def unsupported(goal):
    """The goal conditions this planner cannot express, as a list of tuples.

    Kept so a caller can tell "planned nothing because there was nothing to do" apart from
    "planned nothing because it did not understand the question". A non-empty answer here means
    the plan cannot satisfy the goal no matter how it is ordered or executed.
    """
    return [tuple(g) for g in goal
            if g[0] not in PLACE_FOR and g[0] not in ("toggled", "open")]


def _interleavings(base, free):
    """Every way of dropping the hand-free actions into `base`, at any point.

    This is what makes the search a topological-order enumeration rather than a permutation of
    chains: a switch that needs no hand has no place in the precedence order at all, so every
    position is legal and the cheapest is a question for the cost model, not the DAG.
    """
    if not free:
        yield base
        return
    head, rest = free[0], free[1:]
    for cut in range(len(base) + 1):
        yield from _interleavings(base[:cut] + [head] + base[cut:], rest)


def with_navigation(sequence):
    """Insert `NAVIGATE_TO` before every action whose stance differs from where we are.

    Navigation is not a graph edit and is not ordered by the search; it is inserted afterwards,
    purely to discharge the `near(x)` precondition. Actions that share a stance - opening a
    cabinet, taking something out of it and shutting it again - therefore share one drive.
    """
    plan, here = [], None
    for action, argument, stance in sequence:
        if stance is not None and stance != here:
            plan.append(("NAVIGATE_TO", stance))
            here = stance
        plan.append((action, argument))
    return plan


def nav_cost(plan, graph, distance):
    """`C_MAP`: what the drives cost with every object at its MAP room.

    Deterministic by construction - there is no belief left to be uncertain about - which is
    the whole point of the baseline, and the whole of its blindness: there is no search term
    here, so this planner cannot anticipate ever paying to look for something.

    Rooms that cannot be connected contribute nothing rather than infinity, so one unreachable
    leg cannot make every ordering tie.
    """
    from search_cost import _room_of

    total, here = 0.0, _room_of(graph, "robot")
    for action, arg in plan:
        if action != "NAVIGATE_TO" or not arg:
            continue
        room = _room_of(graph, arg)
        if here and room:
            leg = distance(here, room)
            if leg != float("inf"):
                total += leg
        if room:
            here = room
    return total


def plan(task, scene_graph_dict, goal, distance=None):
    """The EPoG plan for one instruction: `(steps, C_MAP, sequences_scored)`.

    `scene_graph_dict` is already the MAP graph - `scene_graph.populate` records each object's
    most likely room as its `room` - so collapsing the belief is reading that field rather than
    a step of its own.

    An empty `goal` gives an empty plan. That is not a degenerate case to guard against but the
    honest answer: this planner *is* the difference between the belief and the goal, so with
    nothing asked for there is nothing to do, and a run whose goal adapter returned nothing
    should be scored as the failure it is rather than quietly handed the answer key.
    """
    graph = WorldGraph.from_scene_graph(scene_graph_dict)
    goal = [tuple(g) for g in goal]
    chains, free = edits(graph, goal)
    if not chains and not free:
        return [], 0.0, 0
    if distance is None:
        import gavel
        table = gavel._table(task["scene"])
        distance = lambda a, b: table["distance"].get(f"{a}|{b}", float("inf"))

    best, best_cost, scored = None, None, 0
    for order in itertools.permutations(range(len(chains))):
        base = [step for i in order for step in chains[i]]
        for sequence in _interleavings(base, free):
            scored += 1
            if scored > MAX_ORDERS:
                return (best if best is not None else with_navigation(sequence),
                        best_cost or 0.0, scored - 1)
            steps = with_navigation(sequence)
            cost = nav_cost(steps, graph, distance)
            if best_cost is None or cost < best_cost:
                best, best_cost = steps, cost
    return best, best_cost, scored
