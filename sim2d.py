"""A symbolic simulator: one robot, nine primitives, a 2-D floor plan and no physics.

The pipeline's expensive half is the simulator. `execute_plan.py` needs Isaac, a GPU and
about seventeen minutes to answer a question that is usually settled in the first thirty
seconds - did the robot get there, and did it find the thing. This runs the same nine
primitives against `floor_world.FloorWorld` in about a second, so the parts of the
pipeline worth iterating on can be iterated on.

What is kept, because dropping it would make the answers meaningless:

    navigation is real     A* over the floor plan eroded by the robot's footprint. A room
                           behind a blocked doorway is unreachable here too, and that is
                           the failure a symbolic model cannot produce.
    perception is real     a field-of-view cone, cast ray by ray and stopped at the first
                           wall. An object is seen when the robot could actually have seen
                           it, so "I searched the kitchen and it is not there" is earned.
    belief is separate     `world.truth` is what is true; `sim.graph` is what the robot
                           has found out. They are both `WorldGraph`s, so `audit()` is a
                           set difference rather than a translation.

What is dropped, because it is what costs the seventeen minutes:

    no arm, no contact, no dynamics. GRASP welds a name to the hand, PLACE teleports it
    onto a support, OPEN flips a flag. Every one of those is a state change with
    preconditions - the same ones `graph_machine.py` checks - and nothing more.

The effect model is not reimplemented here. `GraphMachine` already knows what each
primitive requires and what it writes, so this drives *two* of them: one over ground truth,
one over the robot's belief. The simulator's own job is the layer underneath - can the
robot get there, is it close enough to touch it - which is exactly the layer the graph
machine has to assume.

    world = FloorWorld.load("Rs_int", categories=["countertop", "fridge"])
    world.add_object("potato", "potato", on_top="countertop_tpuwys_0")
    sim = Sim2D(world, start_room="living_room_0")
    sim.run([("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
             ("NAVIGATE_TO", "fridge"), ("OPEN", "fridge"),
             ("PLACE_INSIDE", "fridge"), ("CLOSE", "fridge")])

    python sim2d.py --scene Rs_int --demo
    python sim2d.py --scene Beechwood_0_int --plan plan.json --gif figures/sim2d.gif
"""

import json
import math
import os

import numpy as np

from floor_world import DEFAULT_ROBOT_RADIUS, FloorWorld, astar
from graph_machine import GraphMachine
from world_graph import ROBOT, WorldGraph

# Coverage-grid values, numbered as `object_map.py` numbers them.
UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def _widen(mask, cells):
    """Grow a boolean mask by `cells` in each direction - a square dilation."""
    out = mask.copy()
    for _ in range(cells):
        out[1:, :] |= out[:-1, :]
        out[:-1, :] |= out[1:, :]
        out[:, 1:] |= out[:, :-1]
        out[:, :-1] |= out[:, 1:]
    return out

# Tiago's head camera, derived the way `object_map.camera_fov` and `exploration.camera_fov`
# derive it from a loaded sensor - `2*atan(aperture / 2*focal_length)` - using the values
# OmniGibson's VisionSensor ships (`sensors/vision_sensor.py:119,122`). Those two functions
# fall back to 1.2 rad when no camera object is available, and this file had hardcoded that
# fallback: 68.8 deg against the real 63.4, so the 2-D robot saw a wider wedge than the one
# in Isaac and the simulated search was easier than the real one by 5.4 deg of arc.
#
# The camera is the search's only source of evidence, so these numbers decide how many
# frontier moves a room costs - and whether a result here transfers to BEHAVIOR-1K at all.
CAMERA_FOCAL_LENGTH = 17.0        # mm, VisionSensor default
CAMERA_APERTURE = 20.995          # mm, VisionSensor default
CAMERA_FOV = 2 * math.atan(CAMERA_APERTURE / (2 * CAMERA_FOCAL_LENGTH))   # 1.106 rad

# Not a sensor limit: OmniGibson's default clipping range is effectively unbounded
# (0.001, 1e7), so the depth image runs to the far wall. 5 m is the range
# `object_map.observe` uses as the distance beyond which a detection is not trusted, and
# this matches it so the two searches explore at the same rate.
CAMERA_RANGE = 5.0
CAMERA_RAYS = 121                 # matches `object_map.observe_fov(n_rays=121)`

# How far from an object the camera has to have reached for the object to count as seen.
# An object is not on free floor - it is inside its own footprint, which is exactly what
# makes its cell untraversable - so no ray ever arrives at its centre and a rule that asked
# for one would make every object in the house invisible. What the camera really sees is
# the *floor beside it*: the near face of the counter the potato is on. 0.8 m is about the
# half-depth of a counter run, and the cells tested are the ones this look actually reached,
# so nothing on the far side of a wall can satisfy it.
SIGHT_MARGIN = 0.8

# How close the robot has to be to act on something. The real limit is set by the robot's
# body rather than the map - measured, Tiago works at 0.53 m to 1.02 m depending on the
# approach - and a stance is the nearest standable cell to the object, so it lands inside
# this for anything smaller than a dining table. Every action result reports the distance
# it acted at, so this can be tightened against a real run.
REACH = 1.5

# Stances are enumerated from the map and ranked by true distance to the object; only the
# closest few are ever routed to. Beyond that the route is long enough that a nearer cell
# would have worked, and each attempt costs an A* search.
STANCE_CANDIDATES = 8

# Headings observed at every stop. One look reports only what happened to be in front of
# the robot when it arrived - with a 63.4 deg camera that is a sixth of the room, and the
# search then shuffles between two cells half a metre apart, gaining nothing. Measured in
# `Rs_int`, one heading stalls at 21% coverage.
#
# The count is *derived* from the camera rather than chosen. It was fixed at four, which
# covers 4 x 63.4 = 254 of the 360 degrees and leaves four 26.6 deg blind wedges on the
# diagonals: a ring of markers every 15 deg around a stopped robot came back with exactly
# the four at 45, 135, 225 and 315 deg missing. An object standing in one of those wedges
# was invisible to a robot that had stopped and scanned specifically to find it.
SCAN_HEADINGS = math.ceil(2 * math.pi / CAMERA_FOV)

# Search parameters, matching `nav_controller.py` so that a search here and a search there
# give up at the same point.
COVERAGE_ENOUGH = 0.95
MIN_FRONTIER_DISTANCE = 0.6

# How far outside a room to look for a place to stand while searching it. An object at the
# edge of a room is seen from the floor on the other side of that edge.
ROOM_MARGIN = 1.5
MAX_FRONTIERS = 10

# How far the robot drives between observations. The camera runs while it moves, which is
# what lets one navigation reveal objects the plan never asked about.
OBSERVE_EVERY = 1.0


class ActionResult:
    """One primitive's outcome: whether it happened, and what it cost."""

    def __init__(self, action, arg, ok, reason=None, edits=(), warnings=(),
                 distance=0.0, at=None, seen=()):
        self.action = action
        self.arg = arg
        self.ok = ok
        self.reason = reason
        self.edits = list(edits)
        self.warnings = list(warnings)
        self.distance = distance      # metres driven by this action
        self.at = at                  # metres from the object it acted on
        self.seen = list(seen)        # objects observed for the first time

    def __repr__(self):
        head = f"{self.action}({self.arg or ''})"
        if not self.ok:
            return f"{head:32s} FAIL  {self.reason}"
        detail = []
        if self.distance:
            detail.append(f"drove {self.distance:.1f} m")
        if self.at is not None:
            detail.append(f"at {self.at:.2f} m")
        if self.seen:
            detail.append(f"saw {len(self.seen)} new")
        return f"{head:32s} ok    {', '.join(detail + self.edits) or 'state change'}"


class Sim2D:
    """The robot: a pose on the floor plan, a camera, a hand, and a belief."""

    def __init__(self, world, start=None, start_room=None, radius=DEFAULT_ROBOT_RADIUS,
                 fov=CAMERA_FOV, camera_range=CAMERA_RANGE, reach=REACH,
                 room_hints=None, focus=(), verbose=True):
        self.world = world
        # The objects the task is about. The robot still sees and records everything it
        # can; this is only what the pictures are drawn about, because a search that turns
        # up eight countertops on its way to the potato should not put eight countertops
        # in the figure.
        self.focus = set(focus)
        # Where the robot expects to find things it has never seen. A value is either one
        # room or the RSN's whole ranking, best first. A ranking is what makes a wrong
        # guess survivable: the robot searches the most likely room, does not find the
        # object, and the belief moves down the list - without asking the planner for
        # anything, because the plan only ever said `NAVIGATE_TO(potato)` and *finding* it
        # is this layer's job.
        self.room_hints = {name: [rooms] if isinstance(rooms, str) else list(rooms)
                           for name, rooms in (room_hints or {}).items()}
        self.rooms_searched = {}     # target -> rooms ruled out, in the order tried
        self.radius = radius
        self.fov = fov
        self.range = camera_range
        self.reach = reach
        self.verbose = verbose

        # What the robot knows: the rooms and their connectivity, and nothing else. Every
        # object in it got there by being seen.
        self.graph = WorldGraph.from_room_graph(world.room_graph)
        self.machine = GraphMachine(self.graph, allow_search=True, copy=False)

        # And the same effect model over what is actually true. Pointing the machine's
        # own open/toggled dicts at the world's makes `OPEN(fridge)` change the world
        # rather than a private copy of it.
        self.truth_machine = GraphMachine(world.truth, allow_search=False, copy=False)
        self.truth_machine.open = world.open
        self.truth_machine.toggled = world.toggled

        self.coverage = np.zeros_like(world.free, dtype=np.uint8)
        # Which headings the last observation covered. A stop scans four of them, and a
        # frame that records one yaw is a frame `sim2d_render` cannot replay: it
        # reconstructs coverage by re-casting from the recorded poses, and three quarters
        # of every scan went missing - objects appeared on the map with no mapped floor
        # anywhere near them.
        self._headings = []
        self.frames = []              # trace, for `sim2d_render.render`
        self.distance = 0.0           # metres driven, all told
        self.step_index = 0

        self.x, self.y, self.yaw = self._starting_pose(start, start_room)
        self._place_robot()
        # Look before the first snapshot, not after it: the frame has to carry the scan
        # that produced it, or the render replays a pose that observed nothing and the
        # objects found on arrival float on unmapped floor.
        self._snapshot("start", self.look())

    # ------------------------------------------------------------------ pose

    def _starting_pose(self, start, start_room):
        if start is not None:
            return (float(start[0]), float(start[1]),
                    float(start[2]) if len(start) > 2 else 0.0)
        mask, labels = self.world.traversable(self.radius)
        if start_room is not None:
            mask = mask & self.world.room_mask(start_room)
        else:
            # The largest connected region, not the largest pile of cells. A scene's
            # standable floor comes in pieces - `Beechwood_0_int` has five - and the
            # centroid of all of them together lands in whichever closet is nearest the
            # middle of the house, from which nothing else is reachable. Parking in the
            # biggest region is what makes the default start able to run a plan.
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            mask = mask & (labels == int(np.argmax(sizes)))
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            raise ValueError("no standable floor to start on"
                             + (f" in {start_room}" if start_room else ""))
        # The middle of the region, not a corner: a robot parked against a wall sees
        # nothing on its first look and the first search is spent walking out of it.
        i = int(np.argmin((rows - rows.mean()) ** 2 + (cols - cols.mean()) ** 2))
        x, y = self.world.to_world(int(rows[i]), int(cols[i]))
        return x, y, 0.0

    @property
    def xy(self):
        return (self.x, self.y)

    @property
    def room(self):
        return self.world.room_at(self.x, self.y)

    @property
    def held(self):
        return self.truth_machine.held

    def _place_robot(self):
        """Record where the robot is and what it is standing at, in both graphs.

        Its room is a `room_inside` edge like any object's, and what is within arm's reach
        is a set of `nearby` edges. Those are what `GraphMachine` reads for every
        manipulation's "the robot is beside it" precondition - so here they are *measured*,
        and the symbolic model gets the geometric answer without knowing any geometry.

        The belief only gets `nearby` for objects the robot has actually seen. Standing
        next to something it has never looked at is not knowledge it has.
        """
        room = self.room
        for graph in (self.graph, self.world.truth):
            graph.place_robot(room, self.xy)
        within = {name for name in self.world.truth.object_names()
                  if self.world.distance_to(name, self.x, self.y) <= self.reach}
        self.world.truth.set_nearby(within, note="measured")
        self.graph.set_nearby(within & set(self.graph.objects), note="measured")

    def _say(self, text):
        if self.verbose:
            print(f"  [sim2d] {text}")

    def _snapshot(self, event, seen=()):
        """One frame. `seen` is what *this* observation revealed, not the running total -
        a frame that lists everything found since the drive began cannot be used to say
        where a sighting came from, which is the one question the trace exists to answer.
        `ActionResult.seen` keeps the per-action total.
        """
        self.frames.append({
            "event": event,
            "x": self.x, "y": self.y, "yaw": self.yaw,
            "room": self.room,
            "held": self.held,
            "seen": list(seen),
            "headings": list(self._headings),
            "known": self.graph.object_names(),
            # Where each known object actually is, at this moment. Without it the render
            # draws every object at its final position in every frame, so the potato is
            # already on the table before the robot has picked it up.
            "positions": {name: self.world.truth.position_of(name)
                          for name in self.graph.object_names()},
            "edges": [list(e) for e in sorted(self.graph.edges) if e[0] != "room_connect"],
            "distance": self.distance,
        })

    # ------------------------------------------------------------------ perception

    def look(self, scan=True):
        """Turn on the spot and observe at each heading. What a stop is for.

        Without the turn a stop reports only what was in front of the robot when it
        arrived, which is a 1.2 rad slice of the room - not enough to move the frontier,
        so the next frontier is the one just vacated and the search walks in place.
        """
        if not scan:
            return self.observe()
        yaw0 = self.yaw
        seen, headings = [], []
        for k in range(SCAN_HEADINGS):
            self.yaw = yaw0 + k * 2.0 * math.pi / SCAN_HEADINGS
            seen += self.observe()
            headings += self._headings
        self.yaw = yaw0
        self._headings = headings
        return seen

    def observe(self):
        """Look once, along the current heading. Mark what the camera covered, and record
        what was in frame.

        Two separate results from one look, as in `object_map.py`: the coverage cone,
        which is what makes a search a search, and the objects, which is what makes the
        graph grow. An object is in frame when it is inside the cone, within range, and
        nothing structural stands between it and the camera.
        """
        # Cast into a scratch grid first: which cells *this* look reached is what decides
        # what was in frame, and the cumulative map cannot answer that.
        view = np.zeros_like(self.coverage)
        cast_fov(self.world, self.x, self.y, self.yaw, self.fov, self.range, view)
        np.maximum(self.coverage, view, out=self.coverage)
        self._headings = [self.yaw]

        new = []
        for name in self.world.truth.object_names():
            if name in self.graph.objects:
                continue
            if not self.can_see(name, view):
                continue
            record = self.world.truth.objects[name]
            self.graph.see_object(name, record.get("category"), record.get("position"),
                                  self.world.room_of(name))
            new.append(name)

        for name in new:
            self._copy_relations(name)
        if new:
            self._say(f"saw {', '.join(new[:6])}"
                      + (f" and {len(new) - 6} more" if len(new) > 6 else ""))
        return new

    def can_see(self, name, view=None):
        """Was this object in frame on the look that just happened?

        Three gates, and the third is the one that needs saying. Range and bearing put the
        object in the cone. The last asks whether the camera actually *reached* it, and it
        asks that of the floor within `SIGHT_MARGIN` of the object rather than of the
        object's own cell, because an object's own cell is never free floor.
        """
        if view is None:
            view = self.coverage
        position = self.world.truth.position_of(name)
        if position is None:
            return False
        # Inside a shut container is not visible, whatever the geometry says. The same
        # test `GraphMachine._blocked_by_container` makes for reaching.
        for _, container in self.world.truth.edges_of("object_inside", src=name):
            if container in self.world.open and not self.world.open[container]:
                return False
        dx, dy = position[0] - self.x, position[1] - self.y
        if self.world.distance_to(name, self.x, self.y) > self.range:
            return False
        bearing = math.atan2(dy, dx) - self.yaw
        if abs((bearing + math.pi) % (2 * math.pi) - math.pi) > self.fov / 2:
            return False
        # Anchor the margin at the part of the object nearest the robot, not its centre.
        # A bed seen from beside it is seen; a rule measured from the middle of the bed
        # says the floor 0.8 m from its centre was never observed, because that floor is
        # the bed.
        cells = self.world.footprint(name)
        row, col = self.world.to_cell(position[0], position[1])
        if len(cells) > 1:
            here = self.world.to_cell(self.x, self.y)
            row, col = min(cells, key=lambda rc: (rc[0] - here[0]) ** 2
                           + (rc[1] - here[1]) ** 2)
        margin = max(1, int(round(SIGHT_MARGIN / self.world.resolution)))
        patch = view[max(0, row - margin):row + margin + 1,
                     max(0, col - margin):col + margin + 1]
        return bool((patch == FREE).any())

    def _copy_relations(self, name):
        """Write the kinematic edges a first sighting reveals.

        Only pairs where *both* objects have been seen, exactly as `read_predicates` does:
        a relation between something seen and something not seen is not knowledge the
        robot has, and admitting it would mean that spotting a plate silently revealed the
        counter underneath it.
        """
        known = set(self.graph.objects)
        for edge_type, a, b in self.world.truth.edges:
            # `holding` is not an observation - the robot knows what is in its own hand
            # because it put it there, and its own actions write that edge.
            if edge_type in ("room_connect", "room_inside", "next_to", "holding"):
                continue
            if name in (a, b) and a in known and b in known:
                self.graph.add_edge(edge_type, a, b, note="observed")
        # `next_to` is derived from position rather than stored, so it is read here rather
        # than copied - which is also why it can never go stale.
        for other in self.world.neighbours(name):
            if other in known:
                self.graph.add_edge("next_to", name, other, note="observed")

    def coverage_of(self, room):
        """Fraction of a room's own cells the camera has covered."""
        mask = self.world.room_mask(room)
        total = int(mask.sum())
        if total == 0:
            return 1.0
        return float(((self.coverage > UNKNOWN) & mask).sum()) / total

    def frontiers(self, room):
        """Covered free cells inside the room that touch cells it has not covered yet."""
        mask = self.world.room_mask(room)
        free = (self.coverage == FREE) & mask
        unknown = (self.coverage == UNKNOWN) & mask
        touches = np.zeros_like(unknown)
        touches[:-1, :] |= unknown[1:, :]
        touches[1:, :] |= unknown[:-1, :]
        touches[:, :-1] |= unknown[:, 1:]
        touches[:, 1:] |= unknown[:, :-1]
        # Only somewhere the robot could stand is worth walking to.
        standable, _ = self.world.traversable(self.radius)
        rows, cols = np.nonzero(free & touches & standable)
        return list(zip(rows.tolist(), cols.tolist()))

    def nearest_frontier(self, room, exclude=()):
        best, best_d = None, float("inf")
        for cell in self.frontiers(room):
            if cell in exclude:
                continue
            fx, fy = self.world.to_world(*cell)
            d = math.hypot(fx - self.x, fy - self.y)
            if d < MIN_FRONTIER_DISTANCE or d >= best_d:
                continue
            best, best_d = cell, d
        return best

    def nearest_unvisited(self, room, exclude=()):
        """The closest standable cell in `room` the camera has not reached yet.

        A frontier is a *known* free cell beside an *unknown* one, so frontier search can
        only grow the observed region outwards from itself. A room split by a furniture run
        into pockets joined by a 0.10-0.20 m gap has pockets that are never adjacent to
        anything observed, so no frontier is ever generated for them and the sweep declares
        itself finished with a third of the room unseen - while A* can route into the pocket
        perfectly well. Measured on `Pomaria_0_int`, the robot gave up at 33% coverage with
        a routable stance 0.70 m from the television it was looking for.

        So when frontiers run out, fall back to the plainer question: where in this room
        have I not been that I can still get to? That uses the map and the coverage the
        robot has built, and nothing about where the object actually is.
        """
        # Only cells in the robot's own connected region - the filter `stances_for`
        # already applies, and the one that matters. Without it the nearest unvisited cell
        # is usually in the pocket the robot cannot enter, and the search spends its whole
        # budget re-picking single cells it can never drive to.
        mask, labels = self.world.traversable(self.radius)
        here = self.world.region_of(self.x, self.y, self.radius)
        # The room, widened. An object against a room's edge is looked at from the floor
        # just outside it - measured, all eight standable cells beside one television were
        # outside the mask of the room the television is in, so a search scoped strictly to
        # the room could never take the one viewpoint that sees it.
        room_mask = _widen(self.world.room_mask(room),
                           int(round(ROOM_MARGIN / self.world.resolution)))
        candidates = np.argwhere(room_mask & mask & (labels == here)
                                 & (self.coverage == UNKNOWN))
        best, best_d = None, float("inf")
        for row, col in candidates:
            cell = (int(row), int(col))
            if cell in exclude:
                continue
            fx, fy = self.world.to_world(*cell)
            d = math.hypot(fx - self.x, fy - self.y)
            if d < MIN_FRONTIER_DISTANCE or d >= best_d:
                continue
            best, best_d = cell, d
        return best

    # ------------------------------------------------------------------ navigation

    def route_to(self, cell):
        """A* from where the robot stands to a cell, on the eroded map. None if no route."""
        mask, _ = self.world.traversable(self.radius)
        start = self.world.nearest_free_cell(self.x, self.y, self.radius)
        if start is None:
            return None
        return astar(mask, start, tuple(cell))

    def stances_for(self, name):
        """Standable cells near an object, nearest first, in the robot's own region.

        Enumerated from the map rather than sampled around the object. Two filters, the
        two `nav_controller` kept: the cell has to be standable, and it has to be in the
        same connected region as the robot - which is the one thing sampling cannot check
        and the one that does most of the work.
        """
        position = self.graph.position_of(name) or self.world.truth.position_of(name)
        if position is None:
            return []
        mask, labels = self.world.traversable(self.radius)
        region = self.world.region_of(self.x, self.y, self.radius)
        rows, cols = np.nonzero(mask & (labels == region))
        if len(rows) == 0:
            return []
        row, col = self.world.to_cell(position[0], position[1])
        d2 = (rows - row) ** 2 + (cols - col) ** 2
        order = np.argsort(d2)[:STANCE_CANDIDATES]
        return [(int(rows[i]), int(cols[i])) for i in order]

    def drive(self, route, label=""):
        """Follow a route, looking as it goes. Returns the distance driven.

        There is no controller and no dynamics: the robot is placed at each waypoint in
        turn. What the drive is for is the *camera* - mapping while moving is why one
        navigation reveals objects the plan never asked about, and why the second
        navigation to a room is direct.
        """
        driven, since_look, seen = 0.0, 0.0, []
        for row, col in route[1:]:
            nx, ny = self.world.to_world(row, col)
            leg = math.hypot(nx - self.x, ny - self.y)
            self.yaw = math.atan2(ny - self.y, nx - self.x)
            self.x, self.y = nx, ny
            driven += leg
            since_look += leg
            if since_look >= OBSERVE_EVERY:
                since_look = 0.0
                fresh = self.observe()
                seen += fresh
                self._snapshot(f"moving{': ' + label if label else ''}", fresh)
        self.distance += driven
        self._carry()
        self._place_robot()
        return driven, seen

    def _carry(self):
        """Whatever is in the hand is where the robot is, and so is whatever rides on it."""
        if self.held is None:
            return
        for name in self.truth_machine._carried_with(self.held):
            self.world.move_object(name, (self.x, self.y))

    def go_to_room(self, room):
        """Drive to the nearest standable cell inside a room."""
        mask, labels = self.world.traversable(self.radius)
        region = self.world.region_of(self.x, self.y, self.radius)
        target = mask & self.world.room_mask(room) & (labels == region)
        rows, cols = np.nonzero(target)
        if len(rows) == 0:
            return None, f"no standable floor in {room} reachable from {self.room}"
        row, col = self.world.to_cell(self.x, self.y)
        order = np.argsort((rows - row) ** 2 + (cols - col) ** 2)
        for i in order[:STANCE_CANDIDATES]:
            route = self.route_to((int(rows[i]), int(cols[i])))
            if route is not None:
                return route, None
        return None, f"no route from {self.room} to {room}"

    def navigate_to(self, target, room=None):
        """Get the robot to an object, searching a room for it if it has never been seen.

        The two paths `nav_controller.navigate_to` takes, on a grid instead of in Isaac.
        An object already in the graph gets one approach; an unknown one gets driven to
        its room and frontier-searched until it appears or the room is ruled out.
        """
        # Refused for the same reason `GraphMachine` refuses it, and with the same words:
        # the two have to agree about what a plan may say, or a plan passes validation and
        # dies here.
        if target in self.world.rooms:
            return (None, 0.0, [],
                    f"'{target}' is a room, not an object; NAVIGATE_TO takes the object "
                    f"you are about to act on - name the thing in {target}, not the room")

        name = self.graph.resolve(target)
        if name is not None and self.graph.position_of(name) is not None:
            return self._approach(name)

        rooms = [room] if room else self._candidate_rooms(target)
        if not rooms:
            return None, 0.0, [], f"'{target}' has never been seen and no room was given"

        # Work down the ranking. A room searched and ruled out is evidence, and the next
        # room is the RSN's next best answer - no replanning, because the plan has not
        # changed and does not need to.
        driven, seen, ruled_out = 0.0, [], []
        for candidate in rooms:
            found, leg, more, problem = self._search_room(target, candidate)
            driven += leg
            seen += more
            if found is not None:
                self.rooms_searched[target] = ruled_out
                return found, driven, seen, None
            # The belief said the object was here and it is not. Retract it, so nothing
            # downstream keeps asserting a room the robot has just swept. This is what
            # makes a wrong *stated* location survivable: the task said "the office
            # cabinet", the office had no cabinet, and the search moves to the RSN's next
            # room instead of the plan dying. Only the belief is touched - the ground-truth
            # graph is not a belief and has nothing to retract.
            self.graph.rule_out_room(target, candidate)
            ruled_out.append(candidate)
            if len(ruled_out) < len(rooms):
                self._say(f"{target} is not in {candidate}; trying "
                          f"{rooms[len(ruled_out)]} next")
        self.rooms_searched[target] = ruled_out
        return (None, driven, seen,
                f"'{target}' is in none of {', '.join(ruled_out)} "
                f"({len(self.graph.object_names())} objects seen)")

    def _search_room(self, target, room):
        """Drive into one room and frontier-search it. Returns (found, m, seen, problem)."""
        self._say(f"searching {room} for {target}")
        route, problem = self.go_to_room(room)
        if route is None:
            return None, 0.0, [], problem
        driven, seen = self.drive(route, f"enter {room}")
        fresh = self.look()
        seen += fresh
        self._snapshot(f"looking in {room}", fresh)

        blocked = set()
        for move in range(MAX_FRONTIERS):
            # Only the *belief* is consulted. Asking ground truth whether the object
            # exists is the cheat this layer exists to remove.
            found = self.graph.resolve(target)
            if found is not None and self.graph.position_of(found) is not None:
                self._say(f"found {found} after {move} frontier moves, "
                          f"{self.coverage_of(room):.0%} of {room} covered")
                route_result = self._approach(found)
                return (route_result[0], driven + route_result[1],
                        seen + route_result[2], route_result[3])
            if self.coverage_of(room) >= COVERAGE_ENOUGH:
                break
            cell = self.nearest_frontier(room, exclude=blocked)
            if cell is None:
                # Frontiers exhausted does not mean the room is searched - only that the
                # observed region cannot grow outwards. Ask where else in this room the
                # robot can still drive to that it has not seen.
                cell = self.nearest_unvisited(room, exclude=blocked)
            if cell is None:
                break
            blocked.add(cell)
            route = self.route_to(cell)
            if route is None:
                continue
            leg, more = self.drive(route, f"frontier {move + 1}")
            driven += leg
            fresh = self.look()
            seen += more + fresh
            self._snapshot(f"frontier {move + 1} in {room}", fresh)

        return (None, driven, seen,
                f"'{target}' is not in {room} ({self.coverage_of(room):.0%} covered, "
                f"{len(self.graph.object_names())} objects seen)")

    def _candidate_rooms(self, target):
        """Where to look for something never seen, best guess first.

        Nothing here is allowed to know where the object actually is - that is the cheat
        this layer exists to remove. Two sources are legitimate: the ranking the caller
        supplied, and the room another instance of the same category was seen in.
        """
        if target in self.room_hints:
            # A room already searched for this object is not a candidate again. Two
            # NAVIGATE_TO calls for the same thing would otherwise re-sweep the same wrong
            # room, and a repair loop that retries the plan would do it five times over.
            return [r for r in self.room_hints[target]
                    if not self.graph.is_ruled_out(target, r)]
        for name in self.graph.object_names():
            record = self.graph.objects[name]
            if record.get("category") == target:
                room = self.graph.room_of(name)
                return [room] if room else []
        return []

    def _approach(self, name):
        """Drive to the nearest stance the robot can route to. Returns (name, m, seen, err)."""
        driven, seen = 0.0, []
        for cell in self.stances_for(name):
            route = self.route_to(cell)
            if route is None:
                continue
            leg, more = self.drive(route, f"approach {name}")
            driven += leg
            seen += more
            # Face what you came for, then look at it.
            position = self.world.truth.position_of(name)
            if position is not None:
                self.yaw = math.atan2(position[1] - self.y, position[0] - self.x)
            fresh = self.look()
            seen += fresh
            self._snapshot(f"looking at {name}", fresh)
            return name, driven, seen, None
        return None, driven, seen, f"no stance near '{name}' that the robot can route to"

    # ------------------------------------------------------------------ the actions

    def step(self, action, arg=None):
        """Run one primitive: the physical gate first, then the two effect models."""
        index = self.step_index
        self.step_index += 1
        warnings = []
        held_before = self.held

        if action == "NAVIGATE_TO":
            name, driven, seen, problem = self.navigate_to(arg)
            if problem is not None:
                self._snapshot(f"FAILED {action}({arg})")
                return ActionResult(action, arg, False, reason=problem,
                                    distance=driven, seen=seen)
            # The room the robot ends up in is the room it is in - not the object's room.
            # A stance beside a fridge set into a wall can legitimately be next door.
            self._sync_location(self.room)
            edits = [f"robot -> {self.room}"]
            if self.held is not None:
                edits.append(f"carried {self.held}")
            result = ActionResult(action, arg, True, edits=edits, distance=driven,
                                  at=self.world.distance_to(name, self.x, self.y),
                                  seen=seen)
            # No observation happens at this instant - the sightings belong to the drive
            # and scan frames that came before it, and are recorded there.
            self._snapshot(f"{action}({arg})")
            return result

        # --- everything else acts on something within arm's reach ---------------------
        name = self.world.truth.resolve(arg) if arg else None
        if action != "RELEASE":
            if name is None:
                return self._fail(action, arg, f"'{arg}' is not in the world")
            distance = self.world.distance_to(name, self.x, self.y)
            if distance > self.reach:
                return self._fail(action, arg,
                                  f"'{name}' is {distance:.2f} m away, beyond the "
                                  f"{self.reach:.2f} m the robot can reach; NAVIGATE_TO it first")
            # Affordances are not checked here. They are preconditions, and the
            # preconditions live in `GraphMachine` - one specification, applied by both
            # callers. The truth machine below reads its `open` and `toggled` straight
            # out of this world, so it decides them on this scene's own state rather than
            # on a category guess.
        else:
            distance = None

        # The room the robot is in is the room it is in. The machines' "beside it" check
        # reads `nearby` edges, which `_place_robot` measures, so there is no longer any
        # need to tell them the robot is in the object's room when it is not.
        self._sync_location(self.room)

        truth_step = self.truth_machine.step(index, action, name)
        if not truth_step.ok:
            return self._fail(action, arg, truth_step.reason, distance)
        self._apply_geometry(action, name, held_before)
        self._place_robot()

        belief_step = self.machine.step(index, action,
                                        self.graph.resolve(arg) if arg else None)
        if not belief_step.ok:
            # The action happened; the robot's model of it did not keep up. Worth saying
            # out loud rather than silently patching, because it is a real divergence.
            warnings.append(f"belief refused this step: {belief_step.reason}")
        # Both machines run the same effect model, so they raise the same warnings.
        for warning in truth_step.warnings + belief_step.warnings:
            if warning not in warnings:
                warnings.append(warning)

        seen = self.observe()      # one look: the robot is facing what it just acted on
        result = ActionResult(action, arg, True, edits=truth_step.edits,
                              warnings=warnings, at=distance, seen=seen)
        self._snapshot(f"{action}({arg or ''})", seen)
        return result

    def _fail(self, action, arg, reason, distance=None):
        self._snapshot(f"FAILED {action}({arg or ''})")
        return ActionResult(action, arg, False, reason=reason, at=distance)

    def _sync_location(self, room):
        self.machine.location = room
        self.truth_machine.location = room

    def _apply_geometry(self, action, name, held_before):
        """Move the object the action moved. The only geometry any manipulation has.

        `GraphMachine` rewrites the relations; positions are this simulator's business,
        and without them the robot could put the plate in the fridge and still see it on
        the counter.
        """
        if action == "GRASP":
            self._carry()
        elif action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
            # Where the robot can reach on that support, not the support's centre. A robot
            # places at arm's length; putting the object at the middle of a sofa let it set
            # something down and then be unable to pick it up again, which is not a thing
            # that happens to a robot with an arm.
            target = (self.world.reachable_point_on(name, near=(self.x, self.y))
                      or self.world.truth.position_of(name))
            if target is not None and held_before is not None:
                # Everything riding on what was placed lands with it: putting down the
                # plate puts down the potato that was on it.
                for moved in self.truth_machine._carried_with(held_before):
                    self.world.move_object(moved, target)
        elif action == "RELEASE" and held_before is not None:
            for moved in self.truth_machine._carried_with(held_before):
                self.world.move_object(moved, (self.x, self.y))

    # ------------------------------------------------------------------ whole plans

    def run(self, plan, stop_on_failure=True):
        """Execute a plan. `plan` is (action, arg) pairs, or the pipeline's step dicts."""
        results = []
        for entry in plan:
            if isinstance(entry, dict):
                action, arg = entry["action"], entry.get("object")
            else:
                action, arg = entry[0], entry[1] if len(entry) > 1 else None
            result = self.step(action, arg)
            results.append(result)
            if self.verbose:
                print(f"{len(results):3d}. {result!r}")
                for warning in result.warnings:
                    print(f"       warning: {warning}")
            if not result.ok and stop_on_failure:
                break
        return results

    # ------------------------------------------------------------------ afterwards

    def left_unsafe(self):
        """What this run opened and never shut, and switched on and never switched off.

        Read from the *world*, not from the plan: the truth machine tracks what its own
        actions disturbed, and this is that set filtered by the state the world is
        actually in. A run can succeed at every step, reach its goal, and still leave an
        open fridge and a lit hob behind - which is not a run anyone should be pleased
        with, and is invisible to a check that only asks whether the goal edges hold.
        """
        machine = self.truth_machine
        return {
            "open": sorted(n for n in machine.opened if self.world.open.get(n)),
            "on": sorted(n for n in machine.switched_on if self.world.toggled.get(n)),
        }

    def audit(self):
        """Where the robot's belief and the world disagree, over what it has seen.

        Both sides are `WorldGraph`s, so this is a set difference. Edges about objects the
        robot never saw are not disagreements - they are things it does not claim.
        """
        # The robot is audited too: `room_inside(robot, ...)` and `holding(robot, ...)`
        # are edges like any other, and a belief about where the robot is that the world
        # disagrees with is exactly the kind of divergence worth catching. Rooms are in the
        # node set for the same reason - without them every `room_inside` edge has one
        # endpoint outside the audit and is silently skipped, the robot's included.
        known = set(self.graph.objects) | {ROBOT} | set(self.graph.rooms)
        believed = {(t, a, b) for t, a, b in self.graph.edges
                    if t != "room_connect" and a in known and b in known}
        actual = self.world.true_edges(known)
        return {
            "seen": len(self.graph.object_names()),
            "of": len(self.world.truth.object_names()),
            "agree": sorted(believed & actual),
            "believed_not_true": sorted(believed - actual),
            "true_not_believed": sorted(actual - believed),
        }

    def summary(self):
        return (f"{self.world.scene}: robot in {self.room}, holding {self.held}, "
                f"{self.distance:.1f} m driven, {self.graph.summary()}")

    def save_trace(self, path, categories=None):
        """Write the run's frames, plus what it takes to rebuild the world they are in."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"scene": self.world.scene,
                       "resolution": self.world.resolution,
                       "trav_map": self.world.trav_map,
                       "categories": categories,
                       "focus": sorted(self.focus),
                       "objects": {n: self.world.truth.objects[n]
                                   for n in self.world.truth.object_names()},
                       "frames": self.frames}, f)
        return path


# ---------------------------------------------------------------------- primitives

def cast_fov(world, x, y, yaw, fov, max_range, grid, n_rays=CAMERA_RAYS):
    """Mark a camera cone on a coverage grid, stopping each ray at the first wall.

    Occlusion is the whole point. A disc of "seen" around the robot claims the far side of
    a counter without looking at it, and the room is then declared searched on evidence
    nobody gathered - the false negative that sends replanning down a branch the robot
    never ruled out.
    """
    step = world.resolution
    for i in range(n_rays):
        frac = (i / (n_rays - 1)) - 0.5 if n_rays > 1 else 0.0
        angle = yaw + frac * fov
        dx, dy = math.cos(angle), math.sin(angle)
        for s in range(int(max_range / step) + 1):
            d = s * step
            row, col = world.to_cell(x + d * dx, y + d * dy)
            if not world.in_bounds(row, col):
                break
            if not world.free[row, col]:
                grid[row, col] = OCCUPIED
                break
            grid[row, col] = FREE
    return grid


# ---------------------------------------------------------------------- command line

def categories_for(plan_steps):
    """Which of the scene's own categories a plan needs loaded.

    The same rule `execute_plan.py` applies before starting Isaac: load the structure plus
    the categories the plan names, and nothing else. Here it costs nothing to load more,
    but a world holding only what the task is about is a world whose graph can be read.
    """
    return sorted({step["object"] if isinstance(step, dict) else step[1]
                   for step in plan_steps
                   if (step.get("object") if isinstance(step, dict) else
                       (step[1] if len(step) > 1 else None))})


def stage_plan(world, saved, rng=None, verbose=True):
    """Ground a pipeline plan onto this world, and put into it what the plan believes.

    The two setup steps `execute_plan.py` does before a run, and for the same reasons.

    **Grounding.** The planner names *categories* - `countertop` - because that is what the
    RSN predicts, and the world holds *instances*: `Beechwood_0_int` has nine countertops.
    `WorldGraph.resolve` refuses an ambiguous category on purpose, so something has to
    choose, and choosing badly is not a small error - it sends the robot to a counter in
    another room. The instance in the room the graph predicts wins; failing that, the first.

    **Injecting.** BEHAVIOR scenes are furniture-only, so the potato has to be put there.
    Where it goes is the graph's own claim made concrete: an object the graph relates to
    another - `potato ON_TOP countertop`, which is what `scene_graph.populate` writes for a
    task that says where something is - goes onto the grounded instance of that support.
    The rest land on free floor in the room the RSN chose. Ignoring the relation is not a
    detail: dropped on the floor instead, the potato was 2.56 m from the counter the plan
    drives to, and `GRASP` failed on reach for a plan whose real fault was three steps later.

    Returns `(grounded steps, room hints, what was injected, the task's objects)`.
    """
    graph = saved.get("graph", {})
    steps = saved["plan"]["steps"]
    wanted = {step["object"] for step in steps if step.get("object")}
    binding, injected = {}, []

    for name in sorted(wanted):
        if name in world.truth.objects:
            binding[name] = name
            continue
        instances = world.truth.by_category(name)
        if not instances:
            continue
        room = (graph.get("objects", {}).get(name) or {}).get("room")
        binding[name] = next((i for i in instances if world.room_of(i) == room),
                             instances[0])
        if verbose and len(instances) > 1:
            print(f"  grounded {name:18s} -> {binding[name]} "
                  f"(of {len(instances)} in the scene)")

    relations = {r["from"]: r for r in graph.get("relations", [])}
    for name, info in sorted((graph.get("objects") or {}).items()):
        if name in binding:
            continue
        relation = relations.get(name)
        support = binding.get(relation["to"]) if relation else None
        if support is not None:
            key = "inside" if relation["relation"].upper() == "INSIDE" else "on_top"
            world.add_object(name, name, **{key: support})
            injected.append((name, f"{relation['relation'].lower()} {support}"))
        elif info.get("room") in world.rooms:
            world.add_object(name, name, room=info["room"], rng=rng)
            injected.append((name, f"on the floor of {info['room']}"))
        else:
            continue
        binding[name] = name

    grounded = [{**step, "object": binding.get(step.get("object"), step.get("object"))}
                for step in steps]
    # The whole RSN ranking where there is one, so a wrong first guess is a room ruled out
    # rather than a failed navigation.
    hints = {binding[name]: [binding.get(r, r) for r in
                             (info.get("candidates") or [info["room"]])]
             for name, info in (graph.get("objects") or {}).items()
             if info.get("room") and name in binding}
    return grounded, hints, injected, set(binding.values())


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", default="Beechwood_0_int")
    parser.add_argument("--plan", help="JSON from pipeline.py --json")
    parser.add_argument("--demo", action="store_true",
                        help="spawn a potato and a plate and run a fetch-and-heat plan")
    parser.add_argument("--start-room")
    parser.add_argument("--categories", nargs="*",
                        help="scene categories to load; by default only the ones the "
                             "plan or the demo names")
    parser.add_argument("--full-scene", action="store_true",
                        help="load every category the scene declares")
    parser.add_argument("--resolution", type=float, default=0.1)
    parser.add_argument("--radius", type=float, default=DEFAULT_ROBOT_RADIUS)
    parser.add_argument("--trace", help="write the run's frames here")
    parser.add_argument("--gif", help="render the run to this path (implies --trace)")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    # An empty house, plus exactly the categories this run needs - the load policy
    # `execute_plan.py` and `test_primitives.py` use before starting Isaac.
    categories = (None if args.full_scene else
                  args.categories if args.categories is not None else
                  _categories_wanted(args))
    world = FloorWorld.load(args.scene, resolution=args.resolution,
                            categories=categories, radius=args.radius)
    print(world.summary())
    if world.doorways:
        print(f"opened {len(world.doorways)} doorways the raster had shut")


    plan, hints, focus = [], {}, set()
    if args.plan:
        with open(args.plan) as f:
            saved = json.load(f)
        if saved.get("scene") and saved["scene"] != args.scene:
            print(f"note: plan is for {saved['scene']}, running it in {args.scene}")
        # Ground the plan's category names onto instances, and inject what the scene
        # does not have. The hints that come back are the RSN's guess, which is what the
        # robot goes on when it has never seen the object.
        plan, hints, injected, focus = stage_plan(world, saved)
        for name, where in injected:
            print(f"  put {name} {where}")
    elif args.demo:
        hints, plan = _demo_plan(world)
        focus = {arg for _, arg in plan if arg}

    sim = Sim2D(world, start_room=args.start_room, radius=args.radius,
                room_hints=hints, focus=focus, verbose=not args.quiet)

    # Which rooms the robot can actually drive to from where it is parked. The room graph
    # says which rooms adjoin; A* says which it can reach, and in half the scenes those
    # are not the same set. Worth knowing before a plan fails on step 6 rather than after.
    groups = world.regions(args.radius)
    here = groups.get(world.region_of(sim.x, sim.y, args.radius), [])
    stranded = sorted(r for rooms in groups.values() for r in rooms if rooms is not here)
    if stranded:
        print(f"warning: from {sim.room} the robot can reach {', '.join(here)} "
              f"but not {', '.join(stranded)}")
    print(f"start: {sim.summary()}\n")

    results = sim.run(plan)
    ok = sum(1 for r in results if r.ok)
    print(f"\n{ok}/{len(plan)} actions succeeded, {sim.distance:.1f} m driven")
    print(sim.summary())

    unsafe = sim.left_unsafe()
    if unsafe["open"] or unsafe["on"]:
        print("\nWARNING: the run left things as it found them only in part -")
        if unsafe["open"]:
            print(f"  still open:         {', '.join(unsafe['open'])}")
        if unsafe["on"]:
            print(f"  still switched on:  {', '.join(unsafe['on'])}")

    report = sim.audit()
    print(f"\naudit: saw {report['seen']} of {report['of']} objects; "
          f"{len(report['agree'])} edges agree, "
          f"{len(report['believed_not_true'])} believed but not true, "
          f"{len(report['true_not_believed'])} true but not believed")
    for edge in report["believed_not_true"][:5]:
        print(f"  believed, not true: {edge[0]}({edge[1]}, {edge[2]})")

    if args.trace or args.gif:
        path = sim.save_trace(args.trace or "figures/sim2d_trace.json",
                              categories=categories)
        print(f"wrote {path} ({len(sim.frames)} frames)")
    if args.gif:
        from sim2d_render import render

        out, n = render(world, sim.frames, args.gif, focus=sim.focus)
        print(f"wrote {out} ({n} frames)")


# What the demo needs out of the scene: something to stand things on, and something with a
# door to put them in.
DEMO_CATEGORIES = ["countertop", "breakfast_table", "coffee_table", "table", "desk",
                   "oven", "microwave", "dishwasher", "fridge"]


def _categories_wanted(args):
    """The categories a run needs, read off the plan or the demo before the world loads."""
    if args.demo:
        return DEMO_CATEGORIES
    if args.plan:
        with open(args.plan) as f:
            return categories_for(json.load(f)["plan"]["steps"])
    return []


def _demo_plan(world):
    """A fetch-and-heat plan against whatever this scene happens to have.

    Picks a counter to start from and an openable appliance to use, so the demo runs in
    any scene rather than only in the two the simulator tests were written for.
    """
    def first(*categories):
        for category in categories:
            hits = world.truth.by_category(category)
            if hits:
                return hits[0]
        return None

    support = first("countertop", "breakfast_table", "coffee_table", "table", "desk")
    # Something to heat in, by preference: an oven has both a door and a switch, so the
    # demo exercises OPEN/CLOSE and TOGGLE_ON/OFF. A fridge only has the door, and asking
    # it to switch on is refused - correctly, which is not what a demo is for.
    appliance = first("oven", "microwave", "dishwasher", "fridge")
    if support is None or appliance is None:
        raise SystemExit(f"{world.scene} has no counter or no appliance to demo with")

    world.add_object("potato", "potato", on_top=support)
    world.add_object("plate", "plate", on_top=support)
    room = world.room_of(support)
    print(f"  put a potato and a plate on {support} in {room}; "
          f"the appliance is {appliance}")
    hints = {"potato": room, "plate": room, appliance: world.room_of(appliance),
             support: room}
    return hints, [
        ("NAVIGATE_TO", "potato"), ("GRASP", "potato"),
        ("NAVIGATE_TO", "plate"), ("PLACE_ON_TOP", "plate"),
        ("GRASP", "plate"),
        ("NAVIGATE_TO", appliance), ("OPEN", appliance),
        ("PLACE_INSIDE", appliance), ("CLOSE", appliance),
    ] + ([("TOGGLE_ON", appliance), ("TOGGLE_OFF", appliance)]
         if appliance in world.toggled else []) + [
        ("OPEN", appliance), ("GRASP", "plate"), ("CLOSE", appliance),
        ("NAVIGATE_TO", support), ("PLACE_ON_TOP", support),
    ]


if __name__ == "__main__":
    main()
