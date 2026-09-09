"""Offline tests for the world graph and the graph edit state machine.

No simulator. Runs in about a second, so the plan-checking half of the pipeline can be
developed without a fifteen-minute Isaac run in the loop.

The scene is the one the simulator test uses - `house_single_floor`, the potato/plate
plan - with the observations the robot would have made after searching both rooms filled
in by hand. That keeps the two halves comparable: the same plan, the same objects, the
same rooms, checked here symbolically and there for real.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import sys

from graph_machine import GraphMachine, check
from world_graph import ROBOT, WorldGraph

SCENE = "house_single_floor"

# What the robot ends up knowing after searching kitchen_0 and living_room_0. In the real
# run these come from `read_predicates` against the simulator; here they are written out
# so the machine can be tested on its own.
OBSERVED = [
    # name,                  category,        room
    ("potato",               "potato",        "kitchen_0"),
    ("plate",                "plate",         "kitchen_0"),
    ("countertop_kelker_0",  "countertop",    "kitchen_0"),
    ("oven_ffitak_0",        "oven",          "kitchen_0"),
    ("coffee_table_rlsebe_0", "coffee_table", "living_room_0"),
]

INITIAL_EDGES = [
    ("on_top", "potato", "countertop_kelker_0"),
    ("on_top", "plate", "countertop_kelker_0"),
    ("next_to", "potato", "plate"),
    ("under", "countertop_kelker_0", "potato"),
    ("under", "countertop_kelker_0", "plate"),
]

# The 16-step plan the simulator run executes, as (action, argument).
PLAN = [
    ("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
    ("NAVIGATE_TO", "plate"), ("PLACE_ON_TOP", "plate"),
    ("GRASP", "plate"),
    ("NAVIGATE_TO", "oven_ffitak_0"), ("OPEN", "oven_ffitak_0"),
    ("PLACE_INSIDE", "oven_ffitak_0"), ("CLOSE", "oven_ffitak_0"),
    ("TOGGLE_ON", "oven_ffitak_0"), ("TOGGLE_OFF", "oven_ffitak_0"),
    ("OPEN", "oven_ffitak_0"), ("GRASP", "plate"), ("CLOSE", "oven_ffitak_0"),
    ("NAVIGATE_TO", "coffee_table_rlsebe_0"), ("PLACE_ON_TOP", "coffee_table_rlsebe_0"),
]

# Cook the potato on the plate and set it on the table: the potato ends up on the plate,
# and the plate on the coffee table.
GOAL = [
    ("on_top", "potato", "plate"),
    ("on_top", "plate", "coffee_table_rlsebe_0"),
]


def observed_graph():
    """The graph as it stands after both rooms have been searched."""
    g = WorldGraph.from_scene_file(SCENE)
    for name, category, room in OBSERVED:
        g.see_object(name, category, [0.0, 0.0, 0.0], room=room)
    for edge_type, a, b in INITIAL_EDGES:
        g.add_edge(edge_type, a, b, note="observed")
    return g


PASS, FAIL = [], []


def case(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))


def main():
    print("=== the room graph seeds room_connect and nothing else ===")
    seed = WorldGraph.from_scene_file(SCENE)
    case("21 rooms, 49 room_connect edges",
         len(seed.rooms) == 21 and len(seed.edges_of("room_connect")) == 49,
         f"{len(seed.rooms)} rooms, {len(seed.edges_of('room_connect'))} edges")
    case("no object edges before anything is seen", not seed.objects)
    case("kitchen_0 connects to living_room_0",
         seed.has_edge("room_connect", "kitchen_0", "living_room_0"))
    case("next_to is symmetric",
         seed.add_edge("next_to", "a", "b") and not seed.add_edge("next_to", "b", "a"))

    print("\n=== the 16-step plan, against a fully observed graph ===")
    g = observed_graph()
    print(f"  starting graph: {g.summary()}")
    outcome = check(g, PLAN, GOAL)
    for line in outcome.report().splitlines():
        print("  " + line)
    case("plan is applicable end to end", outcome.failed_at is None)
    case("plan achieves the goal", outcome.goal_met)

    print("\n=== the potato rides along when the plate is grasped ===")
    m = GraphMachine(observed_graph())
    for i, (a, arg) in enumerate(PLAN[:5]):
        m.step(i, a, arg)
    case("on_top(potato, plate) survives GRASP(plate)",
         m.graph.has_edge("on_top", "potato", "plate"))
    case("plate is no longer on the counter",
         not m.graph.has_edge("on_top", "plate", "countertop_kelker_0"))
    case("plate is no longer next to the potato",
         not m.graph.has_edge("next_to", "plate", "potato"))

    print("\n=== carried objects move through the graph with the robot ===")
    m = GraphMachine(observed_graph())
    for i, (a, arg) in enumerate(PLAN[:15]):
        m.step(i, a, arg)
    case("the carried plate is in the living room",
         m.graph.room_of("plate") == "living_room_0", m.graph.room_of("plate"))
    case("the potato riding on it moved too",
         m.graph.room_of("potato") == "living_room_0", m.graph.room_of("potato"))
    case("the counter it came from did not move",
         m.graph.room_of("countertop_kelker_0") == "kitchen_0")
    # A rider two levels deep: something inside a box on a tray.
    g2 = observed_graph()
    g2.see_object("crumb", "crumb", [0, 0, 0], room="kitchen_0")
    g2.add_edge("on_top", "crumb", "potato")
    m2 = GraphMachine(g2)
    for i, (a, arg) in enumerate(PLAN[:15]):
        m2.step(i, a, arg)
    case("a rider two levels up moved as well",
         m2.graph.room_of("crumb") == "living_room_0", m2.graph.room_of("crumb"))

    print("\n=== the graph the LLM sees is the graph the machine checks ===")
    # `scene_graph.populate`'s output, as `pipeline.py` hands it to the planner: rooms and
    # their topology, the RSN's placements with the confidence that justified them, and
    # any relation the task stated outright.
    rsn = {
        "rooms": {"kitchen_0": {"room_type": "kitchen"},
                  "living_room_0": {"room_type": "living_room"}},
        "edges": [["kitchen_0", "living_room_0"]],
        "objects": {
            "potato": {"room": "kitchen_0", "probability": 1.0},
            "countertop": {"room": "kitchen_0", "probability": 0.99},
            "oven": {"room": "kitchen_0", "probability": 0.77},
            "coffee_table": {"room": "living_room_0", "probability": 0.81},
        },
        "relations": [{"from": "potato", "relation": "ON_TOP", "to": "countertop"}],
    }
    g = WorldGraph.from_scene_graph(rsn)
    case("the RSN's objects come across", set(g.object_names()) == set(rsn["objects"]),
         str(g.object_names()))
    case("so does where it thinks they are",
         g.room_of("potato") == "kitchen_0" and g.room_of("coffee_table") == "living_room_0")
    case("and the confidence that justified it",
         g.objects["oven"]["probability"] == 0.77)
    case("a relation the task stated is a kinematic edge",
         g.has_edge("on_top", "potato", "countertop")
         and g.has_edge("under", "countertop", "potato"))
    case("but nothing has a position - the RSN predicts rooms, not coordinates",
         all(g.position_of(n) is None for n in g.object_names()))
    case("the room topology comes across too",
         g.has_edge("room_connect", "kitchen_0", "living_room_0"))

    # This is the whole point: the plan the LLM was asked for is checkable against the
    # graph it was shown, with no observation and no simulator.
    outcome = check(WorldGraph.from_scene_graph(rsn),
                    [("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
                     ("NAVIGATE_TO", "coffee_table"),
                     ("PLACE_ON_TOP", "coffee_table")],
                    [("on_top", "potato", "coffee_table")])
    case("a plan can be checked against it before the robot has looked at anything",
         outcome.ok, outcome.steps[outcome.failed_at].reason
         if outcome.failed_at is not None else "goal not met")

    # A room named by a placement but missing from the topology is registered, not dropped.
    # Dropping it leaves the object in no room at all, `_require_here` finds nothing to
    # contradict, and every precondition about where the robot stands passes vacuously -
    # a broken plan comes back clean.
    partial = WorldGraph.from_scene_graph(
        {"objects": {"potato": {"room": "pantry_0"}}})
    case("a room the topology forgot is registered rather than dropped",
         partial.room_of("potato") == "pantry_0" and "pantry_0" in partial.rooms)
    case("so a plan that never navigates is still refused",
         check(partial, [("GRASP", "potato")], []).failed_at == 0)

    print("\n=== the specification, one case per precondition ===")

    def machine(at=None, near=(), holding=None, open_state=None):
        """A machine on the observed graph, put into a named starting state.

        `near` is what the robot is standing at - the `nearby` edges `NAVIGATE_TO` writes.
        Being in the right *room* is no longer enough to act on something, which is the
        whole point of the edge, so these cases have to say what the robot drove to.
        """
        m = GraphMachine(observed_graph())
        m.open = dict(open_state or {})
        if at is not None:
            m.location = at
        m.graph.set_nearby(near)
        if holding is not None:
            m.graph.set_held(holding)
        return m

    def refused(m, action, arg=None, because=""):
        result = m.step(0, action, arg)
        return (not result.ok) and because in (result.reason or "")

    def allowed(m, action, arg=None):
        return m.step(0, action, arg).ok

    OVEN, TABLE = "oven_ffitak_0", "coffee_table_rlsebe_0"
    COUNTER = "countertop_kelker_0"

    case("NAVIGATE_TO still has no preconditions for an object the graph holds",
         allowed(machine(at="kitchen_0"), "NAVIGATE_TO", TABLE))
    case("NAVIGATE_TO has no preconditions: carrying something is fine",
         allowed(machine(at="kitchen_0", holding="potato"), "NAVIGATE_TO", TABLE))
    case("RELEASE has no preconditions: an empty hand is a no-op, not a failure",
         allowed(machine(at="kitchen_0"), "RELEASE"))

    case("GRASP 1: refused while the hand is full",
         refused(machine(at="kitchen_0", near=["potato"], holding="plate"),
                 "GRASP", "potato", "already holding"))
    case("GRASP 2: refused when standing at something else in the same room",
         refused(machine(at="kitchen_0", near=[COUNTER]), "GRASP", "potato",
                 "NAVIGATE_TO it first"))
    m = machine(at="kitchen_0", near=["potato"], open_state={OVEN: False})
    m.graph.add_edge("object_inside", "potato", OVEN)
    case("GRASP 3: refused out of a closed container",
         refused(m, "GRASP", "potato", "which is closed"))
    m = machine(at="kitchen_0", near=["potato"], open_state={OVEN: True})
    m.graph.add_edge("object_inside", "potato", OVEN)
    case("GRASP 3: allowed once that container is open", allowed(m, "GRASP", "potato"))
    case("GRASP: refused on something that cannot be picked up",
         refused(machine(at="kitchen_0", near=[OVEN]), "GRASP", OVEN, "fixed furniture"))

    case("TOGGLE 1: refused when standing at something else",
         refused(machine(at="kitchen_0", near=[COUNTER]), "TOGGLE_ON", OVEN,
                 "NAVIGATE_TO it first"))
    case("TOGGLE 2: refused on something with no switch",
         refused(machine(at="kitchen_0", near=[COUNTER]), "TOGGLE_ON", COUNTER,
                 "has no switch"))
    case("TOGGLE: allowed on an oven",
         allowed(machine(at="kitchen_0", near=[OVEN]), "TOGGLE_ON", OVEN))

    case("OPEN 1: refused when standing at something else",
         refused(machine(at="kitchen_0", near=[COUNTER]), "OPEN", OVEN,
                 "NAVIGATE_TO it first"))
    case("OPEN 2: refused on something with no door",
         refused(machine(at="kitchen_0", near=[COUNTER]), "OPEN", COUNTER,
                 "does not open"))
    case("CLOSE: same two conditions",
         refused(machine(at="kitchen_0", near=[COUNTER]), "CLOSE", COUNTER,
                 "does not open"))

    case("PLACE_INSIDE 1: refused when standing at something else",
         refused(machine(at="kitchen_0", near=[COUNTER], holding="potato"),
                 "PLACE_INSIDE", OVEN, "NAVIGATE_TO it first"))
    case("PLACE_INSIDE 2: refused into a container that is not open",
         refused(machine(at="kitchen_0", near=[OVEN], holding="potato"),
                 "PLACE_INSIDE", OVEN, "OPEN it before placing inside"))
    case("PLACE_INSIDE 3: refused with an empty hand",
         refused(machine(at="kitchen_0", near=[OVEN], open_state={OVEN: True}),
                 "PLACE_INSIDE", OVEN, "nothing in the hand"))
    case("PLACE_INSIDE: allowed when all three hold",
         allowed(machine(at="kitchen_0", near=[OVEN], holding="potato",
                         open_state={OVEN: True}), "PLACE_INSIDE", OVEN))

    case("PLACE_ON_TOP 1: refused when standing at something else",
         refused(machine(at="kitchen_0", near=[COUNTER], holding="potato"),
                 "PLACE_ON_TOP", TABLE, "NAVIGATE_TO it first"))
    case("PLACE_ON_TOP 2: refused with an empty hand",
         refused(machine(at="kitchen_0", near=[COUNTER]), "PLACE_ON_TOP", COUNTER,
                 "nothing in the hand"))
    case("PLACE_ON_TOP: needs no door to be open",
         allowed(machine(at="kitchen_0", near=[COUNTER], holding="potato"),
                 "PLACE_ON_TOP", COUNTER))

    print("\n=== the specification, one case per effect ===")

    def carrying():
        """A machine holding the plate, with the potato riding on it, in the kitchen.

        The potato starts on the *counter*, not on the plate - putting it on the plate is
        what the first four steps of PLAN do - so the stack has to be built before it can
        be picked up. Grasping the plate straight away carries nothing, which is correct
        and not what these cases are about.
        """
        m = GraphMachine(observed_graph())
        for i, (a, arg) in enumerate(PLAN[:5]):
            m.step(i, a, arg)
        return m

    m = GraphMachine(observed_graph())
    m.step(0, "NAVIGATE_TO", TABLE)
    case("NAVIGATE_TO: the robot ends up where the object is",
         m.location == "living_room_0" == m.graph.room_of(TABLE), str(m.location))

    m = carrying()
    case("GRASP: the robot is holding it", m.graph.has_edge("holding", ROBOT, "plate"))
    case("GRASP: what it was resting on no longer supports it",
         not m.graph.has_edge("on_top", "plate", COUNTER))
    m.step(5, "NAVIGATE_TO", TABLE)
    case("GRASP: and it travels with the robot",
         m.graph.room_of("plate") == "living_room_0" == m.location)
    case("GRASP: so does whatever rides on it",
         m.graph.has_edge("on_top", "potato", "plate")
         and m.graph.room_of("potato") == "living_room_0")

    m = carrying()
    m.step(5, "NAVIGATE_TO", TABLE)
    m.step(6, "RELEASE")
    case("RELEASE: the hand is empty", m.held is None
         and not m.graph.edges_of("holding", src=ROBOT))
    case("RELEASE: the object stays in the room the robot released it in",
         m.graph.has_edge("room_inside", "plate", "living_room_0"),
         str(m.graph.edges_of("room_inside", src="plate")))
    case("RELEASE: and it rests on nothing - it is on the floor",
         not m.graph.edges_of("on_top", src="plate"))
    case("RELEASE: what was riding on it came down with it",
         m.graph.has_edge("room_inside", "potato", "living_room_0"))

    m = carrying()
    m.step(5, "NAVIGATE_TO", TABLE)
    m.step(6, "PLACE_ON_TOP", TABLE)
    case("PLACE_ON_TOP: the held object is now on top of the other",
         m.graph.has_edge("on_top", "plate", TABLE)
         and m.graph.has_edge("under", TABLE, "plate"))
    case("PLACE_ON_TOP: and the hand is empty", m.held is None)
    case("PLACE_ON_TOP: both take the room of what they were put on",
         m.graph.room_of("plate") == m.graph.room_of("potato") == "living_room_0")

    m = carrying()
    m.open[OVEN] = True
    m.step(5, "NAVIGATE_TO", OVEN)
    m.step(6, "PLACE_INSIDE", OVEN)
    case("PLACE_INSIDE: the held object is now inside the other",
         m.graph.has_edge("object_inside", "plate", OVEN))
    case("PLACE_INSIDE: and the hand is empty", m.held is None)
    case("PLACE_INSIDE: nothing claims the container is on top of it",
         not m.graph.edges_of("on_top", src="plate"))

    m = GraphMachine(observed_graph())
    m.step(0, "NAVIGATE_TO", OVEN)
    m.step(1, "TOGGLE_ON", OVEN)
    case("TOGGLE_ON: the object is on", m.toggled[OVEN] is True)
    m.step(2, "TOGGLE_OFF", OVEN)
    case("TOGGLE_OFF: and off again", m.toggled[OVEN] is False)
    m.step(3, "OPEN", OVEN)
    case("OPEN: the object is open", m.open[OVEN] is True)
    m.step(4, "CLOSE", OVEN)
    case("CLOSE: and shut again", m.open[OVEN] is False)
    case("neither touches the graph's edges",
         not any(e[0] in ("object_inside", "holding") for e in m.graph.edges))

    print("\n=== the robot is a node, and its state is edges ===")
    m = GraphMachine(observed_graph())
    case("a graph with no robot in it says nothing about where the robot is",
         m.location is None and m.held is None)
    m.step(0, "NAVIGATE_TO", "potato")
    case("NAVIGATE_TO writes room_inside(robot, ...)",
         m.graph.has_edge("room_inside", ROBOT, "kitchen_0"), str(m.location))
    m.step(1, "GRASP", "potato")
    case("GRASP writes holding(robot, potato)",
         m.graph.has_edge("holding", ROBOT, "potato"))
    case("and `held` reads that edge rather than a second copy of it",
         m.held == "potato" == m.graph.held_object())
    m.step(2, "NAVIGATE_TO", "coffee_table_rlsebe_0")
    case("driving somewhere rewrites the room rather than adding a second",
         m.graph.edges_of("room_inside", src=ROBOT) == [(ROBOT, "living_room_0")],
         str(m.graph.edges_of("room_inside", src=ROBOT)))
    case("and what is in the hand travels with it",
         m.graph.has_edge("holding", ROBOT, "potato")
         and m.graph.room_of("potato") == "living_room_0")
    case("a carried object has no room edge of its own",
         not m.graph.edges_of("room_inside", src="potato"),
         str(m.graph.edges_of("room_inside", src="potato")))
    case("but its room is still answerable - it is the robot's",
         m.graph.room_of("potato") == "living_room_0", str(m.graph.room_of("potato")))

    m.step(3, "PLACE_ON_TOP", "coffee_table_rlsebe_0")
    case("placing it empties the hand", not m.graph.edges_of("holding", src=ROBOT)
         and m.held is None)
    case("the object it was holding is now resting on the table",
         m.graph.has_edge("on_top", "potato", "coffee_table_rlsebe_0"))
    case("and putting it down writes the room edge back",
         m.graph.has_edge("room_inside", "potato", "living_room_0"),
         str(m.graph.edges_of("room_inside", src="potato")))

    # A rider goes with it: grasping the plate takes the potato's room edge too, and
    # putting the plate down gives both of them back.
    g = observed_graph()
    m2 = GraphMachine(g)
    for i, (a, arg) in enumerate(PLAN[:5]):
        m2.step(i, a, arg)
    case("a rider loses its room edge along with what it rides on",
         not m2.graph.edges_of("room_inside", src="potato")
         and not m2.graph.edges_of("room_inside", src="plate"))
    case("and both still answer with the robot's room",
         m2.graph.room_of("potato") == m2.graph.room_of("plate") == m2.location)
    m2.step(5, "NAVIGATE_TO", "coffee_table_rlsebe_0")
    m2.step(6, "PLACE_ON_TOP", "coffee_table_rlsebe_0")
    case("putting the stack down restores a room edge for every one of them",
         m2.graph.has_edge("room_inside", "plate", "living_room_0")
         and m2.graph.has_edge("room_inside", "potato", "living_room_0"),
         str(sorted(m2.graph.edges_of("room_inside", src="potato"))))

    # Re-observing an object must not put down what the robot is carrying: `holding` is
    # deliberately not one of the kinematic edges.
    g = observed_graph()
    g.place_robot("kitchen_0")
    g.set_held("potato")
    g.see_object("potato", "potato", [0.0, 0.0, 0.0], room="kitchen_0")
    g.clear_kinematic("potato")
    case("re-observing a carried object does not drop it",
         g.held_object() == "potato")

    # A machine handed a graph that already says where the robot is must not wipe it.
    g = observed_graph()
    g.place_robot("living_room_0")
    case("construction leaves a robot the caller already placed alone",
         GraphMachine(g).location == "living_room_0")

    print("\n=== the graph catches what the flat validator cannot ===")
    # Drop step 12, OPEN(oven), and the plate is being grasped out of a closed oven.
    no_reopen = [s for i, s in enumerate(PLAN) if i != 11]
    outcome = check(observed_graph(), no_reopen, GOAL)
    case("GRASP through a closed oven door is rejected",
         outcome.failed_at is not None and "closed" in (outcome.steps[-1].reason or ""),
         outcome.steps[-1].reason if outcome.failed_at is not None else "no failure")

    # Putting something into a shut oven is as impossible as taking something out of one,
    # and the machine refuses it on the same terms - but only when it knows the oven is
    # shut. Openability read off a category list is a guess, and rejecting a plan on a
    # guess throws out plans that are fine.
    no_open = [s for i, s in enumerate(PLAN) if i != 6]        # drop OPEN before filling
    machine = GraphMachine(observed_graph())
    machine.open = {"oven_ffitak_0": False}                    # the robot saw it shut
    outcome = machine.run(no_open, GOAL)
    case("PLACE_INSIDE into an oven known to be shut is rejected",
         outcome.failed_at is not None
         and "OPEN it before placing inside" in (outcome.steps[outcome.failed_at].reason or ""),
         outcome.steps[outcome.failed_at].reason if outcome.failed_at is not None
         else "no failure")

    # Not knowing the state is not the same as knowing it is open. A plan that never
    # opened the oven has not established what the placement needs, so it is rejected on
    # the same grounds as one that shut it.
    outcome = check(observed_graph(), no_open, GOAL)
    case("and rejected too when its state was never established",
         outcome.failed_at is not None
         and "has not been opened" in (outcome.steps[outcome.failed_at].reason or ""),
         outcome.steps[outcome.failed_at].reason if outcome.failed_at is not None
         else "no failure")

    # A container with no door has nothing to open, and the same fact cuts both ways: the
    # placement is excused the OPEN, and opening it is refused outright.
    g = observed_graph()
    g.see_object("bowl", "bowl", [0.0, 0.0, 0.0], room="kitchen_0")
    plan = [("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
            ("NAVIGATE_TO", "bowl"), ("PLACE_INSIDE", "bowl")]
    outcome = check(g, plan, [])
    case("PLACE_INSIDE something with no door needs no OPEN",
         outcome.failed_at is None,
         outcome.steps[outcome.failed_at].reason if outcome.failed_at is not None else "")
    outcome = check(g, plan[:3] + [("OPEN", "bowl")] + plan[3:], [])
    case("and OPEN on that same bowl is refused",
         outcome.failed_at == 3 and outcome.steps[3].fault == ("no_door", "bowl"),
         outcome.steps[3].reason if outcome.failed_at == 3 else "no failure")

    machine = GraphMachine(observed_graph())
    machine.open = {"oven_ffitak_0": False}
    outcome = machine.run(PLAN, GOAL)
    case("and the plan that does open it still runs", outcome.ok,
         outcome.steps[outcome.failed_at].reason if outcome.failed_at is not None
         else "goal not met")

    print("\n=== a plan that runs but does not do the task ===")
    # Everything up to taking the plate back out, then just put it down on the counter.
    # The NAVIGATE_TO is needed now: standing at the oven is not standing at the counter,
    # even though both are in the kitchen.
    wrong = PLAN[:14] + [("NAVIGATE_TO", "countertop_kelker_0"),
                         ("PLACE_ON_TOP", "countertop_kelker_0")]
    outcome = check(observed_graph(), wrong, GOAL)
    case("runs to completion", outcome.failed_at is None)
    case("goal reported as not met", not outcome.goal_met)
    case("names the missing edge",
         ("on_top", "plate", "coffee_table_rlsebe_0") in outcome.missing,
         str(outcome.missing))

    print("\n=== other rejections ===")
    outcome = check(observed_graph(), [("GRASP", "potato")], [])
    case("GRASP without NAVIGATE_TO is rejected", outcome.failed_at == 0,
         outcome.steps[0].reason)

    outcome = check(observed_graph(),
                    [("NAVIGATE_TO", "oven_ffitak_0"), ("GRASP", "oven_ffitak_0")], [])
    case("GRASP on a fixed appliance is rejected", outcome.failed_at == 1,
         outcome.steps[1].reason)

    outcome = check(observed_graph(),
                    [("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
                     ("GRASP", "plate")], [])
    case("a second GRASP while holding is rejected", outcome.failed_at == 2,
         outcome.steps[2].reason)

    outcome = check(observed_graph(),
                    [("NAVIGATE_TO", "potato"), ("PLACE_ON_TOP", "plate")], [])
    case("placing with an empty hand is rejected", outcome.failed_at == 1,
         outcome.steps[1].reason)

    outcome = check(observed_graph(), [("RELEASE", "potato")], [])
    case("RELEASE with an argument is rejected", outcome.failed_at == 0,
         outcome.steps[0].reason)

    print("\n=== names the pipeline never produced ===")
    # One vocabulary: the graph is built from the extractor's names, the goal is written in
    # them and the plan is asked for them. A name that is not a node is a name the model
    # invented, and it is refused rather than admitted - admitting it built a node for a
    # cupboard that does not exist and let the plan fill it.
    fresh = WorldGraph.from_scene_file(SCENE)
    outcome = check(fresh, [("NAVIGATE_TO", "potato")], [])
    case("NAVIGATE_TO an invented name is refused",
         outcome.failed_at == 0 and outcome.steps[0].fault == ("unknown", "potato"),
         outcome.steps[0].reason if outcome.failed_at == 0 else "no failure")
    outcome = check(fresh, [("GRASP", "potato")], [])
    case("GRASP on an invented name is refused",
         outcome.failed_at == 0 and outcome.steps[0].fault == ("unknown", "potato"),
         outcome.steps[0].reason if outcome.failed_at == 0 else "no failure")
    case("and nothing was added to the graph for it", "potato" not in fresh.objects)

    print("\n=== NAVIGATE_TO takes an object, not a room ===")
    import json as _json

    from build_tasks import seed_graph

    graph = WorldGraph.from_scene_graph(
        seed_graph(_json.load(open("data/tasks.json"))[0]))
    room = next(iter(graph.rooms))
    machine = GraphMachine(graph)
    refused = machine.step(0, "NAVIGATE_TO", room)
    case("NAVIGATE_TO a room is refused", not refused.ok, refused.reason or "")
    case("and the complaint names the room, so the loop can rewrite the step",
         room in (refused.reason or ""), refused.reason or "")
    case("no phantom object is invented for the room",
         room not in graph.objects, str(sorted(graph.objects)[:3]))

    print("\n=== a goal term resolves onto the graph when exactly one object could be it ===")
    # The goal adapter writes the sentence's words. The graph holds the dataset's. A term that
    # never resolves can never hold, so the loop refuses every plan for all five attempts and
    # reports whatever the last one wrote - which is how `Pomaria_0_int-04` threw away a correct
    # first plan over `toggled(tv)` against a graph holding `standing_tv`.
    g = WorldGraph()
    g.rooms["living_room_0"] = {"room_type": "living_room"}
    for node in ("standing_tv", "t_shirt", "bath_towel", "bottom_cabinet", "top_cabinet"):
        g.see_object(node, node, None, "living_room_0")
    m = GraphMachine(g)
    case("'tv' resolves to standing_tv", m._resolve_goal_name("tv") == "standing_tv",
         m._resolve_goal_name("tv"))
    case("'tshirt' resolves to t_shirt", m._resolve_goal_name("tshirt") == "t_shirt",
         m._resolve_goal_name("tshirt"))
    case("a plural resolves to its singular",
         m._resolve_goal_name("bath_towels") == "bath_towel", m._resolve_goal_name("bath_towels"))
    case("an exact name is untouched",
         m._resolve_goal_name("t_shirt") == "t_shirt", m._resolve_goal_name("t_shirt"))
    # The guard. This is what the strict version was written to protect, and it still holds:
    # `cabinet` could be either cabinet, so it resolves to neither and the condition stays unmet.
    case("an ambiguous term is NOT guessed at",
         m._resolve_goal_name("cabinet") == "cabinet", m._resolve_goal_name("cabinet"))
    case("a term nothing produced is left alone",
         m._resolve_goal_name("nonesuch") == "nonesuch", m._resolve_goal_name("nonesuch"))
    case("and the goal check follows the resolution",
         m.unmet([("toggled", "tv", False)]) == [], str(m.unmet([("toggled", "tv", False)])))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
