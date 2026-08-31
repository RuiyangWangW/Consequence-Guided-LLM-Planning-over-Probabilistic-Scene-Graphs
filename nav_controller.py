"""The low-level navigation controller: one NAVIGATE_TO becomes a series of drives.

The nine primitives are high-level actions. `NAVIGATE_TO(potato)` says where the robot
should end up, not how to get there, and it says nothing at all about the case the plan
cannot see: the robot does not know where the potato *is*. Something has to turn that one
high-level action into the sequence of drives that finds out.

That is this module, and it sits strictly between the primitive and the simulator:

    NAVIGATE_TO(potato)                      high-level, from the plan
        |
        v
    NavigationController.navigate_to         <- here: decides *where to drive next*
        room sub-goal -> frontier -> frontier -> approach
        |
        v
    controller._drive_to / _navigate_near    the simulator's navigation, unchanged

Nothing here drives the robot itself. Each sub-goal is handed to the existing drive layer,
which still does A* over the eroded map and the rotate-drive-rotate following. What this
adds is the *decision* of which pose to ask for, which is the part that needs to know
whether the object has been seen yet.

Two paths through `navigate_to`:

    the object is in the world graph      one approach sub-goal, straight to it. This is
                                          what every navigation does once the object has
                                          been found, and it is the old behaviour exactly.

    the object has never been seen        drive into its room, look, and keep driving to
                                          frontiers until it appears or the room runs out
                                          of unexplored floor. Finding it writes it into
                                          the graph, so the *next* NAVIGATE_TO to the same
                                          object takes the first path.

Which room to search is ground truth here, by design: the point of this controller is the
in-room search, not the room-level belief, and the RSN ranking that would supply the room
is a separate question the pipeline already answers.
"""

import math

from object_map import ObjectSemanticMap, observe
from world_graph import read_predicates, room_of_object

# How many frontiers to drive to before giving up on a room. Each one is a full drive plus
# an observation, so this bounds the search rather than the room's size doing it.
MAX_FRONTIERS = 10

# How far a frontier must be to be worth driving to. Below this the robot shuffles on the
# spot and the camera sees the same thing again.
MIN_FRONTIER_DISTANCE = 0.6

# How far the camera is trusted to have seen, in metres. Beyond this a reading is too
# oblique to call the floor searched.
SIGHT_RANGE = 5.0

# Coverage above which a room counts as searched even with frontiers left over - a safety
# valve, not the usual exit. Running out of frontiers already ends a sweep, and in every
# run so far that or the move budget is what fired: the highest coverage observed was 56%,
# because cells under and behind furniture are never seen and stay UNKNOWN forever. So this
# threshold has never actually triggered, and its exact value has not yet mattered.
COVERAGE_ENOUGH = 0.95

# How far to pitch the head down while searching, radians. The head camera sits about
# 1.2 m up and looks level by default, so from a normal standing distance its vertical
# field of view spans roughly 0.85-1.7 m - which is *above* most of what a plan refers to.
# Measured: a plate on a coffee table at 0.28 m was never once in frame, and a potato on a
# counter at 0.92 m sat on the very bottom edge and took nine frontier moves to catch.
#
# `head_2_joint` rotates about the parent's -y (axis (0,0,1) under an rpy of (90,0,0)) and
# its limits run -1.047 to +0.785, the extra travel being downward, so negative looks down.
# The resulting camera pitch is printed on the first aim so the log confirms the sign
# rather than this comment being the only evidence for it.
# Two looks per stop, level and down, rather than one compromise angle. Measured: level
# only, the plate on a 0.28 m coffee table was never in frame and the final graph still
# said it was on the counter; tilted to -0.6 rad instead, the counter at 0.92 m went out
# of frame and the potato was not found at all - 34 objects seen against 55, and the
# search gave up. Neither angle sees both. Looking twice does, and costs a render rather
# than a drive, which is the expensive part of a search.
# Headings to observe from at each stop, spread over a full turn. One look per stop sees
# only the camera's ~69 deg of horizontal field of view, so the robot drives past things
# without ever facing them: measured, a potato 0.6 m from where it entered the kitchen took
# nine frontier moves to catch in one run and was never caught in another, while coverage
# and object counts climbed the whole time. Turning on the spot multiplies what a stop is
# worth, and an in-place turn is far cheaper than the drive that reached the stop.
SCAN_HEADINGS = 4

# The building itself, and the robot. These fill most of every camera frame and are not
# things a plan ever refers to, so recording them would add hundreds of nodes and make the
# predicate pass - which is quadratic in what has been seen - quadratic in the walls.
IGNORED_CATEGORIES = frozenset({
    "floors", "walls", "ceilings", "roof", "driveway", "lawn", "agent", "robot",
})


class NavResult:
    """What one high-level NAVIGATE_TO turned into, and how it ended.

    `status` is the distinction that matters to the caller:

        direct       the object was already in the graph; drove straight to it
        found        searched the room and found it
        not_in_room  searched the room out and it is not there. The room is ruled out -
                     this is evidence, not a failure to try
        gave_up      ran out of frontier budget or could not reach what was left. The
                     object may still be here, so the room is NOT ruled out
    """

    def __init__(self, status, target, obj=None, subgoals=None, omap=None, seen=0):
        self.status = status
        self.target = target
        self.obj = obj
        self.subgoals = subgoals or []
        self.omap = omap
        self.seen = seen

    @property
    def ok(self):
        return self.status in ("direct", "found")

    def __repr__(self):
        return (f"<NavResult {self.status} {self.target} "
                f"via {len(self.subgoals)} sub-goals>")


class NavigationController:
    """Expands high-level navigation into drives, and keeps the world graph fed."""

    def __init__(self, env, robot, controller, graph, step_cb=None, verbose=True):
        self.env = env
        self.robot = robot
        self.controller = controller
        self.graph = graph
        self.step_cb = step_cb
        self.verbose = verbose
        self.maps = {}          # room -> ObjectSemanticMap, kept across searches
        # One unmasked map over the whole floor plan, written by every observation
        # alongside the room's own. The room maps decide where to search; this is what the
        # robot knows about the world, including objects seen through a doorway into a
        # room it has never searched.
        self.scene_map = None
        self.on_observe = None  # optional callback(event) after each look, for tracing

    # ------------------------------------------------------------------ plumbing

    def _say(self, text):
        if self.verbose:
            print(f"    [search] {text}")

    def _robot_xy(self):
        return self.robot.get_position_orientation()[0][:2].tolist()

    def _run(self, generator):
        """Hand a drive to the simulator, one action at a time."""
        for action in generator:
            self.env.step(action)
            if self.step_cb is not None:
                self.step_cb()

    def _look(self, omap, scan=True, head_tilt=None, notify=True):
        """Look around from where the robot stands, and write what it saw to the graph.

        `scan` turns the base through `SCAN_HEADINGS` headings, observing at each. Without
        it a stop reports only what happened to be in front of the robot when it arrived.

        `head_tilt` pitches the head to an absolute joint angle for this look only. It is
        for one narrow case: inspecting something low from close up. The head's default of
        -0.45 rad is right for searching a room - four runs spent overriding it all made
        the search worse - but a plate at 0.28 m seen from 0.89 m away sits below the
        bottom of the frame at that pitch, and no amount of turning brings it in. Leave it
        None for searching.

        The camera has its own position `JointController`, so the head holds a persistent
        drive target: setting the joint state alone would be undone on the next physics
        step. Both are set, then the render products are regenerated.
        """
        import math as _math

        import omnigibson as og
        import omnigibson.utils.transform_utils as T

        if head_tilt is not None:
            try:
                names = list(self.robot.camera_joint_names)
                pitch = [n for n in names if n.endswith("2_joint")]
                joint = self.robot.joints.get(pitch[0]) if pitch else None
            except Exception:
                joint = None
            if joint is not None:
                joint.set_pos(head_tilt, drive=False)
                if joint.driven:
                    joint.set_pos(head_tilt, drive=True)
                og.sim.render()
                og.sim.render()
                self._say(f"head pitched to {head_tilt:+.2f} rad for a close low look")

        if self.scene_map is None:
            self.scene_map = ObjectSemanticMap.for_scene(self.env.scene)

        raw = set()
        pos, orn = self.robot.get_position_orientation()
        x, y = float(pos[0]), float(pos[1])
        yaw0 = float(T.quat2euler(orn)[2])
        headings = SCAN_HEADINGS if scan else 1
        for k in range(headings):
            if k:
                try:
                    self._run(self.controller._drive_to(
                        (x, y, yaw0 + k * 2.0 * _math.pi / headings)))
                except Exception:
                    # A heading the base cannot reach is one fewer view, not a failure -
                    # the rest of the scan still counts.
                    continue
            raw |= observe(self.env, self.robot, omap, max_range=SIGHT_RANGE,
                           also=(self.scene_map,), ignore=IGNORED_CATEGORIES)
        scene = self.env.scene
        seen, fresh = set(), []
        for name in sorted(raw):
            obj = scene.object_registry("name", name)
            if obj is None or getattr(obj, "category", None) in IGNORED_CATEGORIES:
                continue
            if obj is self.robot:
                continue
            seen.add(name)
            # First sighting only. Seeing something again tells the robot nothing new: the
            # world is static unless the robot moves something, and when it does, that
            # action's own graph edits record it. Re-recording here would overwrite what
            # the action established with whatever the predicates happen to say now.
            if self.graph.knows(name):
                continue
            pos = obj.get_position_orientation()[0].tolist()
            self.graph.see_object(name, obj.category, pos,
                                  room=room_of_object(obj, scene))
            fresh.append(name)
        if fresh:
            read_predicates(self.graph, fresh, scene, verbose=False)
            self._say(f"first sight of {', '.join(fresh)}")
        if notify and self.on_observe is not None:
            self.on_observe("look" + (f": saw {', '.join(fresh)}" if fresh else ""))
        return seen

    def room_here(self):
        """The room the robot is standing in, from the segmentation."""
        import torch as th

        seg = getattr(self.env.scene, "_seg_map", None)
        if seg is None:
            return None
        rx, ry = self._robot_xy()
        try:
            return seg.get_room_instance_by_point(th.tensor([rx, ry], dtype=th.float32))
        except Exception:
            return None

    def observe_here(self):
        """One cheap look into the map for whatever room the robot is in.

        Called while the robot is moving, not only when it stops. Mapping only at
        designated stops leaves the map frozen for the whole of the rest of a plan - the
        entry scan discovers a room, and then nothing changes again however far the robot
        drives. Building the map continuously is also what a real system does.

        Maps are created per room on demand, so driving into the living room starts
        mapping the living room rather than writing into the kitchen's grid.
        """
        room = self.room_here()
        if room is None:
            return None, set()
        omap = self.maps.get(room)
        if omap is None:
            omap = ObjectSemanticMap.for_room(self.env.scene, room)
            if omap is None:
                return None, set()
            self.maps[room] = omap
        return room, self._look(omap, scan=False, notify=False)

    # ------------------------------------------------------------------ sub-goals

    def _room_pose(self, room, omap):
        """A pose inside `room` the robot can actually stand in, nearest to it.

        Uses the same eroded navigation map the drive layer plans on, so a pose accepted
        here is one A* can route to. Candidates come from the room's own cells, which is
        what keeps the robot from "entering the kitchen" by parking in the corridor
        outside its door.
        """
        import torch as th

        free, labels = self.controller._nav_map_now()
        tmap = self.env.scene.trav_map
        rx, ry = self._robot_xy()

        # The robot's own component, read from the nearest free cell rather than the one
        # it stands in - at this erosion radius its own position is often eroded away.
        rows, cols = (free > 0).nonzero()
        if len(rows) == 0:
            return None
        m = tmap.world_to_map(th.tensor([rx, ry], dtype=th.float32))
        d2 = (rows - int(m[0])) ** 2 + (cols - int(m[1])) ** 2
        near = d2.argmin()
        robot_label = labels[rows[near], cols[near]]

        # The room's cells and where they land on the navigation map. Neither depends on
        # where the robot is, so this is computed once per room and kept - without the
        # cache it is one tensor conversion per cell on every call, thousands of them.
        cells = getattr(omap, "_nav_cells", None)
        if cells is None:
            cells = []
            for row in range(omap.h):
                for col in range(omap.w):
                    # `in_room`, not a grid state. The room mask moved out of the grid
                    # when walls needed to be markable as OCCUPIED, and this check was
                    # left testing a state the grid no longer holds - so it never fired,
                    # and cells outside the room were candidates for standing in it.
                    if not bool(omap.in_room[row, col]):
                        continue
                    wx, wy = omap.to_world(row, col)
                    mm = tmap.world_to_map(th.tensor([wx, wy], dtype=th.float32))
                    r, c = int(mm[0]), int(mm[1])
                    if 0 <= r < free.shape[0] and 0 <= c < free.shape[1]:
                        cells.append((wx, wy, r, c))
            omap._nav_cells = cells

        best, best_d = None, float("inf")
        for wx, wy, r, c in cells:
            if free[r, c] <= 0 or labels[r, c] != robot_label:
                continue
            d = math.hypot(wx - rx, wy - ry)
            if d < best_d:
                best, best_d = (wx, wy), d
        if best is None:
            return None
        # Face the middle of the room on arrival, so the first look is into it rather
        # than back out through the door the robot came in by.
        cx = (omap.x0 + omap.x1) / 2.0
        cy = (omap.y0 + omap.y1) / 2.0
        yaw = math.atan2(cy - best[1], cx - best[0])
        return (best[0], best[1], yaw)

    def _drive(self, pose, label):
        """One navigation sub-goal, forwarded to the simulator's navigation."""
        self.controller._nav_target = None      # no door swing to avoid on a bare drive
        self._say(f"sub-goal {label}: ({pose[0]:.2f}, {pose[1]:.2f})")
        self._run(self.controller._tuck_arm())
        self._run(self.controller._drive_to(pose))

    # ------------------------------------------------------------------ the entry point

    def navigate_to(self, target, room=None, max_frontiers=MAX_FRONTIERS):
        """Get the robot to `target`, searching `room` for it if it has never been seen."""
        scene = self.env.scene
        subgoals = []

        # --- already known: one approach sub-goal, the old behaviour ----------------
        name = self.graph.resolve(target)
        if name is not None and self.graph.position_of(name) is not None:
            obj = scene.object_registry("name", name)
            if obj is not None:
                self._say(f"{target} is already in the graph; driving straight to it")
                subgoals.append(("approach", name))
                self._run(self.controller._navigate_near(obj))
                return NavResult("direct", target, obj, subgoals)

        if room is None:
            self._say(f"{target} has never been seen and no room was given")
            return NavResult("gave_up", target, None, subgoals)

        # --- unknown: drive into the room and search it -----------------------------
        omap = self.maps.get(room)
        if omap is None:
            omap = ObjectSemanticMap.for_room(scene, room)
            if omap is None:
                self._say(f"no segmentation for {room}; cannot search it")
                return NavResult("gave_up", target, None, subgoals)
            self.maps[room] = omap
        self._say(f"searching {room} for {target}; {omap.summary()}")

        # Tuck before anything reads the navigation map. The map is eroded by the robot's
        # *current* bounding box and the footprint is measured once and cached for the
        # whole run, so whoever touches it first decides the radius for good. Measured,
        # skipping this pinned it at the untucked 0.99 m instead of 0.77 m, which eroded
        # every cell in the kitchen away and reported "no reachable floor" for a room the
        # robot drives through routinely. `_navigate_near` tucks first for this reason;
        # so must this.
        self._run(self.controller._tuck_arm())

        pose = self._room_pose(room, omap)
        if pose is None:
            self._say(f"no reachable floor inside {room}")
            return NavResult("gave_up", target, None, subgoals, omap)
        subgoals.append(("room", room))
        self._drive(pose, f"enter {room}")

        seen = self._look(omap)
        found = self._match(target, seen, scene)
        if found is not None:
            return self._approach(target, found, subgoals, omap, len(seen))

        blocked = set()
        for step in range(max_frontiers):
            if omap.coverage() >= COVERAGE_ENOUGH:
                self._say(f"{room} is {omap.coverage():.0%} covered; that is a search")
                break
            rx, ry = self._robot_xy()
            frontier = omap.nearest_frontier(rx, ry, MIN_FRONTIER_DISTANCE, exclude=blocked)
            if frontier is None:
                self._say(f"{room} has no frontier left at {omap.coverage():.0%} coverage")
                break
            yaw = math.atan2(frontier[1] - ry, frontier[0] - rx)
            subgoals.append(("frontier", frontier))
            # Never pick the same cell twice. The drive layer arrives within its own
            # tolerance rather than exactly on the cell, so a frontier can survive the
            # observation that was supposed to consume it - measured, one cell was chosen
            # three times out of ten moves, each costing a full drive.
            blocked.add(omap.to_cell(*frontier))
            try:
                self._drive((frontier[0], frontier[1], yaw), f"frontier {step + 1}")
            except Exception as exc:
                # Do NOT mark this area seen. Claiming coverage the robot never earned is
                # what turns an unreachable corner into a false "the object is not here".
                blocked.add(omap.to_cell(*frontier))
                self._say(f"frontier unreachable ({type(exc).__name__}); skipping it")
                continue
            seen = self._look(omap)
            found = self._match(target, seen, scene)
            if found is not None:
                self._say(f"found {found.name} after {step + 1} frontier moves, "
                          f"{omap.coverage():.0%} covered")
                return self._approach(target, found, subgoals, omap, len(seen))

        # --- searched and not found -------------------------------------------------
        exhausted = not blocked
        status = "not_in_room" if exhausted else "gave_up"
        self._say(f"{target} is not in {room} ({omap.coverage():.0%} covered, "
                  f"{len(omap.objects)} objects seen, status {status})")
        return NavResult(status, target, None, subgoals, omap)

    def _match(self, target, seen, scene):
        """Has the target been found - in this frame, or at any point already?

        Checking only the current frame is not enough now that the robot maps while it
        drives. A navigation tick that catches the potato in passing writes it into the
        graph, and a search that ignores the graph would carry on hunting for something it
        already knows the location of, stopping only if the object happened to be in view
        again at the next stop. Seeing it once is finding it, whenever that happened.
        """
        for name in seen:
            if name == target:
                return scene.object_registry("name", name)
        for name in seen:
            obj = scene.object_registry("name", name)
            if obj is not None and getattr(obj, "category", None) == target:
                return obj
        known = self.graph.resolve(target)
        if known is not None and self.graph.position_of(known) is not None:
            return scene.object_registry("name", known)
        return None

    def _approach(self, target, obj, subgoals, omap, seen):
        """Final sub-goal: now that the object is known, drive to it as usual."""
        subgoals.append(("approach", obj.name))
        self._say(f"approaching {obj.name}")
        self._run(self.controller._navigate_near(obj))
        return NavResult("found", target, obj, subgoals, omap, seen)
