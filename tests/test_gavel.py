#!/usr/bin/env python3
"""Checks for the multi-task stages: the cost decomposition, observation, and ordering.

The claims worth pinning here are the ones that were wrong at some point and would be
silently wrong again:

  * the pairwise matrix must reproduce what full rollouts cost, or the ordering is
    optimising a different objective than the one it reports;
  * `walk` must read rooms from the truth and not from the belief, or searching stops
    costing anything and stops teaching anything;
  * a sweep must rule rooms *out*, not only find things;
  * a belief the executor has refuted must not go infinite;
  * the RSN's distribution must be complete and sum to one, with same-type rooms tied.

Everything here is built by hand - no scene, no RSN, no GPU - so it runs in a second.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import itertools
import sys

import gavel
import order as ordering
from graph_machine import GraphMachine
from search_cost import Beliefs, expected_search, rollout
from world_graph import WorldGraph

passed = failed = 0


def check(label, got, want):
    global passed, failed
    if got == want:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL  {label}\n        got  {got}\n        want {want}")


# --- a four-room house in a line: a |2| b |3| c |4| d -------------------------------------
ROOMS = ["a", "b", "c", "d"]
STEP = {("a", "b"): 2.0, ("b", "c"): 3.0, ("c", "d"): 4.0}


def distance(x, y):
    if x == y:
        return 0.0
    order_ = ROOMS.index(x), ROOMS.index(y)
    lo, hi = min(order_), max(order_)
    return sum(STEP[(ROOMS[i], ROOMS[i + 1])] for i in range(lo, hi))


def search(room):
    return 10.0


def house():
    """Truth: the mug is in `a`, the book in `d`, and a table in every room."""
    g = WorldGraph()
    for room in ROOMS:
        g.rooms[room] = {"room_type": "kitchen" if room == "a" else "bedroom"}
        g.see_object(f"table_{room}", "breakfast_table", None, room)
    g.see_object("robot", "robot", None, "a")
    g.see_object("mug", "mug", None, "a")
    g.see_object("book", "book", None, "d")
    return g


def graph_dict(mug_belief, book_belief):
    return {"rooms": {r: {"room_type": "kitchen"} for r in ROOMS}, "edges": [],
            "objects": {"mug": {"room": max(mug_belief, key=mug_belief.get),
                                "belief": mug_belief, "candidates": list(mug_belief)},
                        "book": {"room": max(book_belief, key=book_belief.get),
                                 "belief": book_belief, "candidates": list(book_belief)}},
            "unplaced": {}, "relations": []}


# --- the decomposition is exact ----------------------------------------------------------
def test_pairwise_matches_rollouts():
    """`A[i][j] + head` must equal what rolling the plans forward really costs.

    This is the whole justification for building an N^2 matrix instead of doing N! rollouts.
    If it drifts, the ordering stage reports a number nobody pays.
    """
    truth = house()
    plans = [[("NAVIGATE_TO", "mug"), ("GRASP", "mug"),
              ("NAVIGATE_TO", "table_b"), ("PLACE_ON_TOP", "table_b")],
             [("NAVIGATE_TO", "book"), ("GRASP", "book"),
              ("NAVIGATE_TO", "table_c"), ("PLACE_ON_TOP", "table_c")],
             [("NAVIGATE_TO", "table_d")]]
    gd = graph_dict({"a": 0.6, "b": 0.4}, {"d": 0.7, "c": 0.3})
    mk = lambda: Beliefs(gd, ROOMS)

    A, head = ordering.pairwise(plans, truth, mk, distance, search)
    for perm in itertools.permutations(range(3)):
        total, where, world = 0.0, None, truth
        for i in perm:
            cost, where, world = rollout(plans[i], world, mk(), distance, search, start=where)
            total += cost
        check(f"pairwise == rollout for {perm}",
              round(ordering.score(perm, A, head), 6), round(total, 6))


def test_best_order_is_the_cheapest():
    A = [[0, 5, 9], [5, 0, 3], [9, 3, 0]]
    head = [1.0, 100.0, 100.0]
    best, cost = ordering.best_order(A, head, 3)
    want = min(itertools.permutations(range(3)), key=lambda o: ordering.score(o, A, head))
    check("best_order picks the minimum", best, want)
    check("best_order reports its cost", round(cost, 6), round(ordering.score(want, A, head), 6))
    check("ranked is sorted by cost",
          [round(ordering.score(o, A, head), 6) for o in ordering.ranked(A, head, 3)],
          sorted(round(ordering.score(o, A, head), 6)
                 for o in itertools.permutations(range(3))))


def test_best_order_never_returns_none():
    """Every ordering infinite - an object behind a wall - must still yield an answer."""
    inf = float("inf")
    best, cost = ordering.best_order([[0, inf], [inf, 0]], [inf, inf], 2)
    check("best_order falls back to the identity", best, (0, 1))
    check("and says the cost is infinite", cost, inf)


# --- observation --------------------------------------------------------------------------
def test_sweep_finds_and_rules_out():
    truth = house()
    gd = graph_dict({"a": 0.5, "b": 0.5}, {"c": 0.5, "d": 0.5})
    beliefs = Beliefs(gd, ROOMS)
    seen = gavel.observe(beliefs, truth, "a", ["mug", "book"], set(ROOMS))
    check("sweeping `a` finds the mug", sorted(seen), ["mug"])
    check("the mug is now located", beliefs.located.get("mug"), "a")
    check("`a` was not a candidate for the book, so nothing is dropped",
          sorted(beliefs.prior["book"]), ["c", "d"])

    seen = gavel.observe(beliefs, truth, "c", ["mug", "book"], set(ROOMS))
    check("sweeping `c` finds nothing", sorted(seen), [])
    check("but rules `c` out for the book", sorted(beliefs.prior["book"]), ["d"])
    check("and renormalises what is left", round(beliefs.prior["book"]["d"], 6), 1.0)
    check("a located object is not re-examined", beliefs.located.get("mug"), "a")


def test_refuted_belief_does_not_go_infinite():
    """Every reachable candidate ruled out is not proof of impossibility.

    The object exists, so it is somewhere nobody looked. Before this, the estimate went to
    infinity and every ordering tied - the ordering stage silently stopped working.
    """
    truth = house()
    gd = graph_dict({"a": 1.0}, {"b": 0.5, "c": 0.5})
    gd["objects"]["book"]["fallback"] = {}
    beliefs = Beliefs(gd, ROOMS)
    for room in ("b", "c"):
        gavel.observe(beliefs, truth, room, ["book"], set(ROOMS))
    dist, candidates, known = beliefs.belief("book")
    check("the refuted belief is not empty", bool(dist), True)
    check("it excludes the rooms already swept",
          sorted(set(dist) & {"b", "c"}), [])
    cost = expected_search("a", candidates, dist, distance, search)
    check("and the search cost is finite", cost != float("inf"), True)


def test_walk_reads_rooms_from_truth_not_belief():
    """The bug that made the whole stage meaningless.

    The belief says the mug is in `d`; it is really in `a`. An executor that trusts the
    belief "finds" it in `d` - paying nothing for being wrong and learning nothing. The real
    walk must end holding a mug it found in `a`.
    """
    truth = house()
    believed = house()
    gd = graph_dict({"d": 1.0}, {"d": 1.0})
    beliefs = Beliefs(gd, ROOMS)
    plan = [("NAVIGATE_TO", "mug"), ("GRASP", "mug")]
    cost, here, _, real, seen = gavel.walk(plan, believed, truth, beliefs, distance, search,
                                           ["mug", "book"], reachable=set(ROOMS))
    check("the robot ends where the mug really is", here, "a")
    check("the mug is located in its true room", beliefs.located.get("mug"), "a")
    check("and it paid for the wrong guess", cost > 0.0, True)


def test_unreachable_target_does_not_teleport_the_robot():
    """A room A* cannot reach must not be entered for free.

    The executor walks a route and skips any leg of infinite cost. It then used to set the
    robot's position to the target regardless, so an object behind a wall was reached at no
    charge - and every later leg was measured from a room the robot had never been in. It fired
    on 91 of 500 instructions, and because the free arrival also skipped the sweep it understated
    the distance both arms walked.
    """
    truth = house()
    gd = graph_dict({"a": 1.0}, {"d": 1.0})
    beliefs = Beliefs(gd, ROOMS)
    # `d` is walled off from everywhere: the book cannot be reached.
    walled = lambda x, y: float("inf") if (x == "d") != (y == "d") else distance(x, y)
    plan = [("NAVIGATE_TO", "book")]
    cost, here, _, _, seen = gavel.walk(plan, house(), truth, beliefs, walled, search,
                                        ["book"], reachable={"a", "b", "c"})
    check("the robot does not reach a walled-off room", here != "d", True)
    check("and is not charged for a walk it never took", cost, 0.0)
    check("and does not claim to have found what is there", "book" in beliefs.located, False)

    # The reachable case still works, so the guard has not broken ordinary navigation.
    fresh = Beliefs(graph_dict({"a": 1.0}, {"d": 1.0}), ROOMS)
    cost2, here2, _, _, _ = gavel.walk([("NAVIGATE_TO", "book")], house(), truth, fresh,
                                       distance, search, ["book"], reachable=set(ROOMS))
    check("a reachable target is still reached", here2, "d")
    check("and is charged for", cost2 > 0.0, True)
    check("and is located", fresh.located.get("book"), "d")


def test_compose_catches_a_broken_concatenation():
    """Two subplans, each fine alone, that break when run back to back."""
    truth = house()
    carry = [("NAVIGATE_TO", "mug"), ("GRASP", "mug")]
    _, alone = gavel.compose(truth, [carry], ())
    check("one grasp alone is applicable", alone.failed_at, None)
    steps, outcome = gavel.compose(truth, [carry, carry], ())
    check("grasping twice is not", outcome.failed_at is not None, True)


# --- the RSN's distribution ---------------------------------------------------------------
def test_beliefs_are_complete_and_normalised():
    gd = graph_dict({"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25}, {"d": 1.0})
    beliefs = Beliefs(gd, ROOMS)
    check("a stored distribution is used as given",
          round(sum(beliefs.prior["mug"].values()), 6), 1.0)
    # the legacy path: a ranking plus a top score, with no distribution stored
    legacy = {"rooms": {r: {} for r in ROOMS}, "edges": [], "unplaced": {}, "relations": [],
              "objects": {"pan": {"room": "a", "probability": 0.4,
                                  "candidates": ["a", "b", "c"]}}}
    old = Beliefs(legacy, ROOMS)
    check("a graph with no distribution is still normalised",
          round(sum(old.prior["pan"].values()), 6), 1.0)
    check("and keeps the ranking's head on top",
          max(old.prior["pan"], key=old.prior["pan"].get), "a")


def test_equal_rooms_are_broken_by_distance():
    """Two rooms the model cannot tell apart: sweep the near one first.

    From `a`, rooms `b` and `d` are equally likely, but `b` is 2 m away and `d` is 9 m. The
    estimator must sort them itself, so the answer cannot depend on the order the candidates
    arrive in - the RSN has no idea where the robot is standing, and hands them over ranked
    by probability alone.
    """
    belief = {"b": 0.5, "d": 0.5}
    near = expected_search("a", ["b", "d"], belief, distance, search)
    far = expected_search("a", ["d", "b"], belief, distance, search)
    check("the order candidates arrive in does not matter", round(near, 6), round(far, 6))
    # Sweeping `b` first: 0.5*(2 + 5) + 0.5*(2 + 10 + 7 + 5) = 15.5
    check("and it is the near-first cost", round(near, 6), 15.5)
    # Sweeping `d` first would be 0.5*(9 + 5) + 0.5*(9 + 10 + 7 + 5) = 22.5
    check("which beats sweeping the far room first", near < 22.5, True)
    # A room the robot cannot reach is not a candidate at all.
    unreachable = lambda x, y: float("inf") if "d" in (x, y) and x != y else distance(x, y)
    check("an unreachable room is dropped rather than costed",
          round(expected_search("a", ["b", "d"], belief, unreachable, search), 6),
          round(expected_search("a", ["b"], {"b": 0.5}, unreachable, search), 6))


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
