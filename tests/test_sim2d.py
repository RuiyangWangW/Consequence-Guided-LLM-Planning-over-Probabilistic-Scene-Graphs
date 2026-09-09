"""Offline tests for the 2-D symbolic simulator. No Isaac, no GPU, about ten seconds.

Three things are worth testing here, and they are the three the simulator adds over
`graph_machine.py`'s purely symbolic model:

    the loader     that a world position lands in the room the dataset says it is in.
                   Everything else rests on this: get the transform wrong and every room
                   label, every stance and every search is quietly about the wrong place.
    the geometry   that A* finds routes where routes exist and refuses where they do not,
                   and that an action is refused when the robot has not driven to it.
    the camera     that an object is revealed when the robot could have seen it and not
                   when it could not - in a closed fridge, behind a wall, out of range.

`Beechwood_0_int` is the scene throughout: its standable floor is one connected region, so
a navigation failure in these tests is a bug rather than a house.

    python test_sim2d.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import math
import sys

import numpy as np

from floor_world import DEFAULT_ROBOT_RADIUS, FloorWorld
from sim2d import (CAMERA_FOV, CAMERA_RANGE, SCAN_HEADINGS, Sim2D, astar,
                   cast_fov)
from world_graph import ROBOT

SCENE = "Beechwood_0_int"

# What these tests need out of the scene, loaded the way `execute_plan.py` restricts its
# load: the house starts empty and only named categories come in.
CATEGORIES = ["countertop", "fridge", "oven"]

PASS, FAIL = [], []


def case(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def sees(world, x, y, yaw, name):
    """Would one look from this pose, along this heading, reveal `name`?

    The wedge on its own, without the four-heading scan a stop does - otherwise every
    object near the robot is visible and the heading proves nothing.
    """
    view = np.zeros_like(world.free, dtype=np.uint8)
    cast_fov(world, x, y, yaw, CAMERA_FOV, CAMERA_RANGE, view)
    sim = Sim2D(world, start=(x, y, yaw), verbose=False)
    sim.graph.objects.pop(name, None)
    return sim.can_see(name, view)


def kitchen_world():
    """Beechwood with a potato and a plate on a kitchen counter, as the demo sets it up."""
    world = FloorWorld.load(SCENE, categories=CATEGORIES)
    support = world.truth.by_category("countertop")[0]
    world.add_object("potato", "potato", on_top=support)
    world.add_object("plate", "plate", on_top=support)
    return world, support


def main():
    print("=== the loader: floor plan, rooms, objects ===")
    world = FloorWorld.load(SCENE, categories=None)
    case("rooms come from the room graph the figures are drawn from",
         set(world.rooms) == set(world.room_graph["rooms"]), f"{len(world.rooms)} rooms")
    case("the floor is a grid at the requested resolution",
         world.free.shape == (world.n, world.n) and world.resolution == 0.1,
         f"{world.n}x{world.n} cells")
    case("the scene's furniture is loaded", len(world.truth.objects) > 100,
         f"{len(world.truth.objects)} objects")

    # The transform is the load-bearing part. Every object the scene JSON annotates with a
    # room should land in a room of that type when looked up through the grid. Doors and
    # windows are excluded: they sit in the wall between two rooms and belong to neither.
    agree = disagree = 0
    for name, record in world.truth.objects.items():
        position = record.get("position")
        if position is None or "door" in name or "window" in name:
            continue
        room = world.room_at(position[0], position[1])
        if room is None:
            continue
        annotated = world.truth.room_of(name)
        agree += room == annotated
        disagree += room != annotated
    case("world -> cell -> room agrees with where objects were placed",
         disagree == 0, f"{agree} agree, {disagree} disagree")

    round_trip = [world.to_cell(*world.to_world(r, c)) == (r, c)
                  for r, c in [(0, 0), (10, 40), (world.n - 1, world.n - 1)]]
    case("to_cell and to_world are inverses", all(round_trip))

    print("\n=== putting objects wherever we want ===")
    world = FloorWorld.load(SCENE, categories=CATEGORIES)
    rows, cols = np.nonzero(world.room_mask("kitchen_0"))
    cell = (int(rows[len(rows) // 2]), int(cols[len(cols) // 2]))
    world.add_object("widget", "widget", cell=cell)
    case("an object placed by grid cell is in that cell", world.cell_of("widget") == cell,
         f"{world.cell_of('widget')} vs {cell}")
    case("and the room label comes from the floor plan under it",
         world.room_of("widget") == "kitchen_0", world.room_of("widget"))

    support = world.truth.by_category("countertop")[0]
    world.add_object("mug", "mug", on_top=support)
    # Not the support's centre: the near edge of it, which is where an arm can put a mug
    # down and later pick it up again. Placing at the middle of a wide support put objects
    # further from any standable floor than the arm can reach.
    case("an object placed on a support lands on the support's footprint",
         world.to_cell(*world.position_of("mug")) in set(world.footprint(support)),
         f"mug at {world.cell_of('mug')}, support spans {len(world.footprint(support))} cells")
    case("and it lands where the robot could stand beside it",
         min(abs(r - world.cell_of("mug")[0]) + abs(c - world.cell_of("mug")[1])
             for r, c in np.argwhere(world.free)) * world.resolution < 0.6,
         f"{min(abs(r - world.cell_of('mug')[0]) + abs(c - world.cell_of('mug')[1]) for r, c in np.argwhere(world.free)) * world.resolution:.2f} m to free floor")
    case("and the relation is recorded", world.truth.has_edge("on_top", "mug", support))
    world.add_object("spoon", "spoon", room="bathroom_0")
    case("an object placed by room lands in that room",
         world.room_of("spoon") == "bathroom_0", world.room_of("spoon"))

    print("\n=== opening the doorways the raster leaves shut ===")
    shut = FloorWorld.load("Rs_int", categories=(), open_doorways=False)
    opened = FloorWorld.load("Rs_int", categories=())
    case("Rs_int ships with its rooms in several regions of floor",
         len(shut.regions()) > 1, f"{len(shut.regions())} regions")
    case("opening the thresholds joins them", len(opened.regions()) == 1,
         f"{len(opened.regions())} regions")
    case("and it says which ones it opened", len(opened.doorways) > 0,
         str([d["rooms"] for d in opened.doorways]))
    adjacent = {tuple(sorted(e)) for e in opened.room_graph["edges"]}
    case("only between rooms the room graph already calls adjacent",
         all(tuple(sorted(d["rooms"])) in adjacent for d in opened.doorways))
    case("every sill it carved was narrow enough to be a threshold",
         all(d["gap"] <= 0.5 for d in opened.doorways if d["kind"] == "sill"),
         str([d.get("gap") for d in opened.doorways]))
    case("opening doorways only ever adds floor",
         bool((opened.free | shut.free == opened.free).all()))

    print("\n=== A* on the grid ===")
    mask, labels = world.traversable(DEFAULT_ROBOT_RADIUS)
    rows, cols = np.nonzero(mask & world.room_mask("kitchen_0"))
    kitchen = (int(rows[0]), int(cols[0]))
    rows, cols = np.nonzero(mask & world.room_mask("living_room_0"))
    living = (int(rows[0]), int(cols[0]))
    route = astar(mask, kitchen, living)
    case("a route exists between two rooms of one region", route is not None,
         f"{len(route or [])} cells")
    case("every cell of the route is standable",
         route is not None and all(mask[cell] for cell in route))
    case("a route into a wall is refused",
         astar(mask, kitchen, (0, 0)) is None)
    case("a route to where you already are is the single cell",
         astar(mask, kitchen, kitchen) == [kitchen])

    print("\n=== the camera is a wedge, and it reveals what is in it ===")
    world, support = kitchen_world()
    here = world.position_of("potato")
    x, y = world.to_world(*world.nearest_free_cell(here[0], here[1],
                                                  DEFAULT_ROBOT_RADIUS))
    facing = math.atan2(here[1] - y, here[0] - x)
    case("an object in front of the camera is seen",
         sees(world, x, y, facing, "potato"))
    case("the same object behind the robot is not",
         not sees(world, x, y, facing + math.pi, "potato"))
    case("nor is it once it is out of range",
         not sees(world, x - (CAMERA_RANGE + 2), y, 0.0, "potato"))

    far = world.sample_free("living_room_0")
    case("an object in another room is not seen through the wall",
         not sees(world, far[0], far[1],
                  math.atan2(here[1] - far[1], here[0] - far[0]), "potato"))

    world, support = kitchen_world()
    fridge = world.truth.by_category("fridge")[0]
    at_fridge = world.position_of(fridge)
    world.move_object("potato", at_fridge)
    world.truth.clear_kinematic("potato")
    world.truth.add_edge("object_inside", "potato", fridge)
    x, y = world.to_world(*world.nearest_free_cell(at_fridge[0], at_fridge[1],
                                                   DEFAULT_ROBOT_RADIUS))
    facing = math.atan2(at_fridge[1] - y, at_fridge[0] - x)
    case("an object in a closed fridge is not seen",
         not sees(world, x, y, facing, "potato"))
    case("but the fridge itself is", sees(world, x, y, facing, fridge))
    world.open[fridge] = True
    case("open the fridge and it is", sees(world, x, y, facing, "potato"))

    print("\n=== the search, run the way nav_controller runs it ===")
    world, support = kitchen_world()
    start = world.sample_free("living_room_0")
    sim = Sim2D(world, start=(start[0], start[1], 0.0),
                room_hints={"potato": world.room_of(support)}, verbose=False)
    case("the potato is not known before the search", "potato" not in sim.graph.objects)
    result = sim.step("NAVIGATE_TO", "potato")
    case("searching its room finds it", result.ok, result.reason or "")
    case("and the robot ends up within reach",
         result.ok and result.at <= sim.reach, f"{result.at:.2f} m")
    case("the search cost real driving", sim.distance > 1.0, f"{sim.distance:.1f} m")
    case("and revealed the rest of the kitchen on the way",
         len(sim.graph.objects) > 3, f"{len(sim.graph.objects)} objects seen")

    hopeless = Sim2D(world, start=(start[0], start[1], 0.0),
                     room_hints={"potato": "bathroom_0"}, verbose=False)
    result = hopeless.step("NAVIGATE_TO", "potato")
    case("searching the wrong room rules it out rather than finding it",
         not result.ok and "bathroom_0" in result.reason, result.reason)
    case("and records which rooms it ruled out",
         hopeless.rooms_searched.get("potato") == ["bathroom_0"],
         str(hopeless.rooms_searched))

    # A wrong first guess is survivable: the RSN hands over a ranking, and the search works
    # down it without anyone re-planning - the plan only ever said NAVIGATE_TO(potato).
    world, support = kitchen_world()
    kitchen = world.room_of(support)
    recovering = Sim2D(world, start_room="living_room_0",
                       room_hints={"potato": ["bathroom_0", "corridor_0", kitchen]},
                       verbose=False)
    result = recovering.step("NAVIGATE_TO", "potato")
    case("a wrong first guess is recovered from by searching the next room",
         result.ok, result.reason or "")
    case("and the rooms it ruled out are the ones ahead of the right one",
         recovering.rooms_searched.get("potato") == ["bathroom_0", "corridor_0"],
         str(recovering.rooms_searched.get("potato")))
    case("no re-planning was needed - one NAVIGATE_TO did it all",
         result.ok and result.distance > 0)

    print("\n=== an action needs the robot to have driven to it ===")
    world, support = kitchen_world()
    kitchen = world.room_of(support)
    hints = {"potato": kitchen, "plate": kitchen,
             world.truth.by_category("fridge")[0]: kitchen}
    sim = Sim2D(world, start_room="living_room_0", room_hints=hints, verbose=False)
    result = sim.step("GRASP", "potato")
    case("GRASP from across the house is refused on distance",
         not result.ok and "beyond" in result.reason, result.reason)

    sim.step("NAVIGATE_TO", "potato")
    result = sim.step("GRASP", "potato")
    case("GRASP after navigating succeeds", result.ok, result.reason or "")
    case("the hand holds it", sim.held == "potato", str(sim.held))
    case("and it is no longer resting on the counter",
         not world.truth.has_edge("on_top", "potato", support))

    result = sim.step("GRASP", "plate")
    case("a second GRASP while holding is refused",
         not result.ok and "already holding" in result.reason, result.reason)

    fridge = world.truth.by_category("fridge")[0]
    result = sim.step("RELEASE")
    case("RELEASE takes no argument and empties the hand",
         result.ok and sim.held is None, result.reason or str(sim.held))
    case("RELEASE with an argument is refused",
         not sim.step("RELEASE", "potato").ok)
    sim.step("NAVIGATE_TO", fridge)
    result = sim.step("GRASP", fridge)
    case("GRASP on a fridge is refused as fixed furniture",
         not result.ok and "fixed furniture" in result.reason, result.reason)

    print("\n=== carrying, opening, and the effects that follow ===")
    world, support = kitchen_world()
    fridge = world.truth.by_category("fridge")[0]
    kitchen = world.room_of(support)
    hints = {"potato": kitchen, "plate": kitchen, fridge: kitchen}
    sim = Sim2D(world, start_room=kitchen, room_hints=hints, verbose=False)
    results = sim.run([("NAVIGATE_TO", "plate"), ("GRASP", "plate"),
                       ("NAVIGATE_TO", "potato"), ("PLACE_ON_TOP", "potato")])
    case("a four-step plan runs", all(r.ok for r in results),
         next((r.reason for r in results if not r.ok), ""))

    world, support = kitchen_world()
    sim = Sim2D(world, start_room=kitchen, room_hints=hints, verbose=False)
    results = sim.run([("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
                       ("NAVIGATE_TO", "plate"), ("PLACE_ON_TOP", "plate"),
                       ("GRASP", "plate"), ("NAVIGATE_TO", fridge),
                       ("PLACE_INSIDE", fridge)], stop_on_failure=False)
    case("PLACE_INSIDE a shut fridge is refused by the graph model",
         not results[-1].ok or any("closed" in w for w in results[-1].warnings),
         results[-1].reason or "; ".join(results[-1].warnings))

    world, support = kitchen_world()
    sim = Sim2D(world, start_room=kitchen, room_hints=hints, verbose=False)
    results = sim.run([("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
                       ("NAVIGATE_TO", "plate"), ("PLACE_ON_TOP", "plate"),
                       ("GRASP", "plate"), ("NAVIGATE_TO", fridge), ("OPEN", fridge),
                       ("PLACE_INSIDE", fridge), ("CLOSE", fridge)])
    case("the same plan with OPEN first runs to the end", all(r.ok for r in results),
         next((r.reason for r in results if not r.ok), ""))
    case("the fridge really is shut again", world.open[fridge] is False)
    case("the plate is inside it", world.truth.has_edge("object_inside", "plate", fridge))
    case("the potato rode along on the plate",
         world.truth.has_edge("on_top", "potato", "plate"))
    case("and both are at the fridge, not on the counter",
         world.to_cell(*world.position_of("potato")) in set(world.footprint(fridge)),
         f"potato at {world.cell_of('potato')}, fridge spans "
         f"{len(world.footprint(fridge))} cells")

    result = sim.step("TOGGLE_ON", fridge)
    case("TOGGLE_ON is refused on something with no switch",
         not result.ok and "no switch" in result.reason, result.reason or "")
    oven = (world.truth.by_category("oven") or [None])[0]
    if oven:
        sim.step("NAVIGATE_TO", oven)
        case("but an oven does have one", sim.step("TOGGLE_ON", oven).ok
             and world.toggled[oven] is True)
    sim.step("NAVIGATE_TO", support)
    result = sim.step("OPEN", support)
    case("OPEN is refused on a countertop",
         not result.ok and "does not open" in result.reason, result.reason or "")

    print("\n=== the robot is in the graph, in both of them ===")
    world, support = kitchen_world()
    kitchen = world.room_of(support)
    hints = {"potato": kitchen, "plate": kitchen}
    sim = Sim2D(world, start_room="living_room_0", room_hints=hints, verbose=False)
    case("the robot is a node from the start, in the belief and in the world",
         sim.graph.has_edge("room_inside", ROBOT, "living_room_0")
         and world.truth.has_edge("room_inside", ROBOT, "living_room_0"))
    case("and it is not counted as something the robot has seen",
         ROBOT not in sim.graph.object_names())
    case("its hand starts empty", sim.graph.held_object() is None)

    sim.step("NAVIGATE_TO", "potato")
    case("driving into another room moves the node with it",
         sim.graph.room_of(ROBOT) == kitchen, str(sim.graph.room_of(ROBOT)))
    case("the world agrees where it is",
         world.truth.room_of(ROBOT) == sim.graph.room_of(ROBOT))

    sim.step("GRASP", "potato")
    case("GRASP writes holding(robot, potato) in both graphs",
         sim.graph.has_edge("holding", ROBOT, "potato")
         and world.truth.has_edge("holding", ROBOT, "potato"))
    case("and `sim.held` is that edge", sim.held == "potato")
    case("the carried potato has no room edge of its own",
         not sim.graph.edges_of("room_inside", src="potato")
         and not world.truth.edges_of("room_inside", src="potato"))
    case("but its room still answers, as the robot's",
         sim.graph.room_of("potato") == sim.graph.room_of(ROBOT) == kitchen)
    before = sim.distance
    sim.step("NAVIGATE_TO", "plate")
    case("driving with it in hand writes no room edge back",
         not sim.graph.edges_of("room_inside", src="potato"),
         f"drove {sim.distance - before:.1f} m")

    sim.step("PLACE_ON_TOP", "plate")
    case("placing empties the hand in both graphs",
         not sim.graph.edges_of("holding", src=ROBOT)
         and not world.truth.edges_of("holding", src=ROBOT))
    case("and the room edge the grasp removed is written back, in both",
         sim.graph.has_edge("room_inside", "potato", kitchen)
         and world.truth.has_edge("room_inside", "potato", kitchen),
         str(sim.graph.edges_of("room_inside", src="potato")))
    case("one hand, so at most one holding edge ever",
         len(sim.graph.edges_of("holding", src=ROBOT)) <= 1)

    print("\n=== belief and world agree about what the robot did ===")
    report = sim.audit()
    written = [e for e in report["agree"] if e[0] in ("on_top", "object_inside", "under")]
    case("the edges the actions wrote are in both graphs", len(written) >= 3,
         str(written))
    case("the robot believes nothing the world denies",
         not report["believed_not_true"], str(report["believed_not_true"][:3]))
    case("the audit covers where the robot thinks it is",
         any(edge[0] == "room_inside" and edge[1] == ROBOT for edge in report["agree"]),
         str([e for e in report["agree"] if ROBOT in e]))
    fresh = Sim2D(FloorWorld.load(SCENE, categories=CATEGORIES),
                  start_room="living_room_0", verbose=False)
    case("a robot that has not moved knows only what it can see from there",
         0 < len(fresh.graph.object_names()) < len(fresh.world.truth.object_names()),
         f"{len(fresh.graph.object_names())}/{len(fresh.world.truth.object_names())}")

    print("\n=== the camera is a wedge, and a stop turns far enough to close the ring ===")
    empty = FloorWorld.load(SCENE, categories=[])
    probe = Sim2D(empty, start_room="living_room_0", verbose=False)
    ox, oy = probe.x, probe.y
    floor = np.argwhere(empty.free)
    for deg in range(0, 360, 15):
        a = math.radians(deg)
        tr, tc = empty.to_cell(ox + 2.0 * math.cos(a), oy + 2.0 * math.sin(a))
        i = np.hypot(floor[:, 0] - tr, floor[:, 1] - tc).argmin()
        empty.add_object(f"m{deg:03d}", "marker", cell=tuple(int(v) for v in floor[i]))
    ring = [f"m{d:03d}" for d in range(0, 360, 15) if f"m{d:03d}" in empty.truth.objects]

    facing = Sim2D(empty, start=(ox, oy, 0.0), verbose=False)
    facing.graph.objects.clear()
    ahead = {n for n in facing.observe() if n.startswith("m")}
    case("one look while driving sees only what is in front of it",
         0 < len(ahead) < len(ring), f"{len(ahead)} of {len(ring)}")

    facing.graph.objects.clear()
    around = {n for n in facing.look(scan=True) if n.startswith("m")}
    # Four headings of a 63.4 deg camera cover 254 of 360 and leave blind wedges on the
    # diagonals; the count is derived from the FOV so the ring closes.
    case("a stop sees the whole ring, with no blind wedge between headings",
         len(around) == len(ring), f"{len(around)} of {len(ring)}, "
         f"{SCAN_HEADINGS} headings x {math.degrees(facing.fov):.0f} deg")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
