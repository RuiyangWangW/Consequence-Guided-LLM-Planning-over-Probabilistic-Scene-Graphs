#!/usr/bin/env python3
"""Expected navigation and search cost, rolled forward over the graph world model.

GAVEL orders subplans by what they are expected to cost to execute. That estimate has to
account for the fact that the robot does not know where things are: a `NAVIGATE_TO` on an
unseen object is not a drive, it is a search over the rooms the RSN ranked.

    C_search(r, o)   expected cost of finding `o` starting from room `r`
    C_nav(r, o)      that, or a plain drive if `o` has been localized
    rollout(...)     both, propagated through a subplan over a *copy* of the graph

The rollout is the part that matters. Object locations are not static: a
`PLACE_INSIDE(dishwasher)` moves the mug, and a later `NAVIGATE_TO(mug)` must be costed
against the dishwasher's room, not the mug's original prior. So the estimator replays the
plan through `GraphMachine` - the same effect model the checker uses - and reads locations
off the graph as it goes.
"""

import itertools

# The expected fraction of a room that must be swept before the object turns up, given it
# is in that room. An explicit modelling approximation, not a fitted constant: the
# simulator's frontier search starts from the doorway and is geometry-dependent, so the
# true fraction is neither uniform nor known in closed form. Half is the uninformative
# answer, and the formula is written so a measured eta(o, r) can replace it later without
# touching anything around it.
ETA = 0.5


def expected_search(start, candidates, belief, distance, search, eta=ETA):
    """C_search: expected cost of finding an object, searching `candidates` in order.

    Conditioned on the object being in the k-th candidate room, the robot drives to the
    first candidate, fully searches every room before the k-th, drives between them, and
    sweeps `eta` of the k-th. Weighted by the belief that it is there.
    """
    # A room A* cannot reach from here is not a candidate: the robot cannot search it, so
    # it carries no cost and contributes no chance of finding anything. Half the scenes are
    # in more than one piece, so without this every belief that touches the far piece makes
    # the whole estimate infinite and every ordering ties.
    candidates = [r for r in candidates if distance(start, r) != float("inf")]
    if not candidates:
        return float("inf")
    # Most probable first, and among rooms the model cannot tell apart - two bedrooms, which
    # the RSN scores identically because it scores types - the nearer one first. That tie is
    # common rather than exotic, and breaking it arbitrarily sends the robot across the house
    # to sweep a room its twin next door was equally likely to hold.
    candidates = sorted(candidates, key=lambda r: (-belief.get(r, 0.0), distance(start, r), r))
    total, mass = 0.0, 0.0
    for k, room in enumerate(candidates):
        p = belief.get(room, 0.0)
        if p <= 0:
            continue
        cost = distance(start, candidates[0])
        for m in range(k):
            cost += search(candidates[m]) + distance(candidates[m], candidates[m + 1])
        cost += eta * search(room)
        total += p * cost
        mass += p
    # The belief is a ranking over rooms the scene has, and a room ruled out during
    # execution is removed from it, so the mass need not sum to one. Normalising keeps the
    # estimate a cost rather than a cost-times-confidence, which would make an object the
    # robot is unsure about look cheap.
    return total / mass if mass > 0 else float("inf")


def expected_nav(start, target, world, distance, search, eta=ETA):
    """C_nav: a drive if the target's room is known, a search if it is not.

    `start` is a distribution over rooms, so this is an expectation over where the robot
    is as well as over where the object is.
    """
    belief, candidates, known = world.belief(target)
    out, mass = 0.0, 0.0
    for room, p in start.items():
        if p <= 0:
            continue
        cost = (distance(room, known) if known is not None
                else expected_search(room, candidates, belief, distance, search, eta))
        # A start room from which the target is unreachable is a room the robot is not
        # in - the plan was verified, so it can be carried out from wherever it really is.
        # Keeping the term would make one disconnected room poison the whole estimate.
        if cost == float("inf"):
            continue
        out += p * cost
        mass += p
    return out / mass if mass > 0 else float("inf")


class Beliefs:
    """Where each object is thought to be, and how that changes as a plan is rolled out.

    Reads the same structure `scene_graph.populate` produces - `{object: {room,
    probability, candidates}}` - so the estimator and the executor believe the same thing.
    An object the plan has moved is *localized*: its room is wherever the plan put it, and
    no search is paid for it again.
    """

    def __init__(self, graph_dict, rooms):
        self.rooms = list(rooms)
        self.prior = {}
        self.fallback = {}
        for name, record in (graph_dict.get("objects") or {}).items():
            self.fallback[name] = dict(record.get("fallback") or {})
            given = record.get("belief")
            if given:
                # `scene_graph` already normalised the RSN over every room this scene has,
                # so the searcher inherits a complete ranking that sums to one.
                self.prior[name] = dict(given)
                continue
            # No stored distribution - an older graph, or an object placed by hand. Spread
            # the top score's remainder over the rest of the ranking so the tail is worth
            # something, and normalise, so this is a probability like the branch above.
            candidates = list(record.get("candidates") or ([record["room"]] if record.get("room") else []))
            if not candidates:
                self.prior[name] = {}
                continue
            p = float(record.get("probability") or 0.0)
            spread = (1.0 - p) / max(len(candidates) - 1, 1)
            raw = {r: (p if i == 0 else spread) for i, r in enumerate(candidates)}
            bulk = sum(raw.values())
            self.prior[name] = ({r: v / bulk for r, v in raw.items()} if bulk > 0
                                else {r: 1.0 / len(raw) for r in raw})
        self.located = {}
        self.found = set()
        # Rooms an executor has looked in and not found this object in. Kept so a belief
        # that has been refuted can be rebuilt over what is left rather than over the house.
        self.ruled = {}

    def belief(self, name):
        """(distribution, ranked candidates, known room or None)."""
        if name in self.located:
            room = self.located[name]
            return {room: 1.0}, [room], room
        dist = self.prior.get(name) or {}
        if name in self.found and dist:
            # Searched once already: the robot knows where it is, but the *planner* still
            # only knows the distribution. So no sweep is paid again - `known` stays None
            # and the caller uses the distribution - and this is why a second visit to the
            # same object is not charged twice.
            return dist, list(dist), None
        return dist, list(dist), None

    def searched(self, name):
        self.found.add(name)

    def place(self, name, room):
        self.located[name] = room


def _room_of(graph, name):
    """The room a graph node is in, following what it rests on or sits inside."""
    seen = set()
    while name and name not in seen:
        seen.add(name)
        room = graph.room_of(name)
        if room:
            return room
        parent = None
        for edge in ("object_inside", "on_top"):
            for _, target in graph.edges_of(edge, src=name):
                parent = target
                break
            if parent:
                break
        name = parent
    return None


def rollout(plan, graph, beliefs, distance, search, start=None, eta=ETA):
    """Cost of one plan, and where the robot is expected to end up.

    Returns `(cost, end_distribution, graph_after)`. The graph is advanced by the same
    `GraphMachine` the checker uses, so a `PLACE_INSIDE(dishwasher)` really does move the
    mug and a later `NAVIGATE_TO(mug)` is costed against the dishwasher's room.

    The endpoint is whatever the rollout leaves the robot at - there is no special rule for
    grasping, placing or carrying. `PLACE_INSIDE` does not move the robot, so a plan ending
    on one ends wherever its last `NAVIGATE_TO` went.
    """
    from graph_machine import GraphMachine

    machine = GraphMachine(graph.copy(), copy=False)
    where = dict(start or {})
    if not where:
        room = _room_of(machine.graph, "robot")
        where = {room: 1.0} if room else {r: 1.0 / len(beliefs.rooms) for r in beliefs.rooms}

    total = 0.0
    for index, (action, arg) in enumerate(plan):
        if action == "NAVIGATE_TO" and arg:
            total += expected_nav(where, arg, beliefs, distance, search, eta)
            dist, _, known = beliefs.belief(arg)
            beliefs.searched(arg)
            # After arriving, the robot is where the object is - a point mass if the plan
            # has already put it somewhere, otherwise distributed as the belief.
            where = {known: 1.0} if known else dict(dist)
        result = machine.step(index, action, arg)
        if not result.ok:
            break
        if action in ("PLACE_ON_TOP", "PLACE_INSIDE") and arg:
            room = _room_of(machine.graph, arg)
            for held, _ in machine.graph.edges_of("on_top", dst=arg):
                if room:
                    beliefs.place(held, room)
            for held, _ in machine.graph.edges_of("object_inside", dst=arg):
                if room:
                    beliefs.place(held, room)
    return total, where, machine.graph
