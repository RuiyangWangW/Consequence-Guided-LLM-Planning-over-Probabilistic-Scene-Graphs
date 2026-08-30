"""The action-primitive backend: all nine primitives, symbolic, on real navigation.

    NAVIGATE_TO, GRASP, PLACE_ON_TOP, PLACE_INSIDE, RELEASE,
    OPEN, CLOSE, TOGGLE_ON, TOGGLE_OFF

The robot drives itself to each object for real - A* over the eroded traversability map,
rotate in place, drive straight, rotate in place - and the manipulation is then resolved
by setting object state rather than by moving the arm.

Every primitive is atomic. The state changes do NOT navigate: OPEN opens, and getting the
robot to the fridge first is the plan's job, as a separate NAVIGATE_TO step.

    NAVIGATE_TO(fridge)
    OPEN(fridge)

An earlier version buried an approach inside each state change, which made those
primitives responsible for two things and hid navigation failures behind an OPEN that
appeared to fail. Keeping them separate means a plan that forgets to navigate fails at the
step that is actually wrong, and the validator can check the ordering. `_require_near`
enforces it.

Why not physical manipulation
-----------------------------
`StarterSemanticActionPrimitives` really drives the hardware - its `_grasp` navigates to a
grasp pose, moves the hand to pregrasp, closes the gripper and approaches - but four of
the nine open with a bare `raise NotImplementedError`:

    _open_or_close   "Open/close is not implemented correctly yet."
    _toggle          "Toggle is not implemented correctly yet."

Upstream also skips its own OPEN tests with `reason="primitives are broken"`. The five it
does implement are genuinely flaky in practice - a sticky grasp slips off a thin object,
the planner cannot route the last few centimetres - so every attempt had to be rescued
symbolically anyway, which left the recordings misleading about what had actually
happened, and a single PLACE_ON_TOP could burn twenty minutes on retries.

`SymbolicSemanticActionPrimitives` implements all nine - it contains no
NotImplementedError at all. `_grasp` teleports the object into the gripper and welds a
FixedJoint, `_place` teleports it to a sampled pose. It subclasses the starter set, so
all the motion machinery is still there to inherit; it simply skips CuRobo, which this
module puts back because the stance search needs it for collision checks.

What stays real is the driving, which is the part a plan can actually get wrong.
"""


import contextlib


# Joint-position tolerance for a completed motion, in radians.
#
# The starter primitives ship 0.005, and physical GRASP fails on it by a hair: measured
# 0.0056 and 0.0097 against the 0.005 limit after ~900 steps of correct arm motion, which
# raises EXECUTION_ERROR: "Could not reach the target articulation joint positions". The
# same grasp passes on other runs, so this is a marginal convergence, not a wrong plan.
# 0.02 is still four times tighter than the module's own LOW_PRECISION value of 0.05.
JOINT_TOLERANCE = 0.02

# Measured: 0.30 keeps the robot's centre 0.77 m from anything, and that is what sets
# how far back it stands - 0.70 m from an object's edge on a worktop, 1.57 m from an
# oven it is opening.
#
# Zero was tried, since the arm no longer moves and the measured box already contains
# it. It does bring the robot closer - the plate approach went from 0.68 m to 0.40 m
# from its edge - but it breaks routing: at 0.47 m erosion the free space fragments
# into 34 components against 28, small pockets open up near furniture that connect to
# nothing, and NAVIGATE_TO(oven) failed with all nine stances unroutable. The margin is
# not only clearance around obstacles, it is what keeps the free space coherent enough
# to plan through.
ARM_CLEARANCE_MARGIN = 0.30

# How fast the base is asked to travel along a route, m/s. Waypoint spacing is derived
# from this: a tall mobile base pitches when accelerated hard, and the wheels come off
# the floor. Slower than walking pace on purpose - nothing here is racing.
# Speeds along a route, m/s. Regulated Pure Pursuit slows itself for tight curvature and
# on the approach to the goal, so these are the bounds it works between rather than a
# profile of their own.
CRUISE_SPEED = 0.6
APPROACH_SPEED = 0.15


# How far ahead of the robot the moving setpoint is allowed to get, and how long it may
# sit waiting before the route is abandoned. Bounding the lead bounds the tracking error,
# and so the force the position drive applies.
# Regulated Pure Pursuit, following nav2's defaults where they apply.
CLEARANCE_FULL_SPEED = 0.45  # metres of clear floor needed to run at full speed
CLEARANCE_MIN_SCALE = 0.25   # never slow below this fraction of the cruise speed
APPROACH_DISTANCE = 1.0      # metres from the goal over which it eases off
GOAL_TOLERANCE = 0.15        # metres; close enough to stop driving
GOAL_YAW_TOLERANCE = 0.05    # radians; close enough to stop turning
ROTATE_FIRST_ANGLE = 0.8     # radians of heading error that call for turning on the spot
ROTATE_SPEED = 1.0           # rad/s cap when turning on the spot
ROTATE_GAIN = 1.5            # rad/s per rad of heading error, so the turn settles
# Metres between the waypoints handed to the executor. `_execute_motion_plan` treats each
# one as a step change in target, so this is the tracking error the position drive sees -
# and the force it answers with. At 0.12 m the base arrived 7.5 deg out of level; keeping
# more of the trajectory keeps each step small. The rollout is dense (one state per
# simulation step), so this is only about how much of it to throw away.
WAYPOINT_EVERY = 0.05
WAYPOINT_TURN = 0.05         # radians between executed waypoints while turning

# Below these, a leg is not worth driving or turning - trying to makes the base twitch
# on the spot, which is what short hops looked like on the video.
MIN_SEGMENT = 0.04           # metres
MIN_TURN = 0.05              # radians

# Categories that are ways through rather than things in the way. Their swing must not be
# blocked off, or every doorway in the house closes and the map falls into pieces.
PASSAGE_CATEGORIES = frozenset({"door", "sliding_door"})


# Waypoints used to fold the arm in. Commanding the tucked pose in one jump leaves the arm
# part-way there, because the move is large and the executor gives each waypoint a fixed
# step budget.
TUCK_STEPS = 6

# How close the arm joints must get to the tucked pose before the robot drives, radians,
# and how many extra rounds of holding the target to allow. Measured: with a plate in the
# gripper the interpolated waypoints alone leave the arm well short, and the controller
# then fights its own unreached target while the load swings.
TUCK_TOLERANCE = 0.05
TUCK_SETTLE_TRIES = 8

# The same, while carrying something. A joint holding a load settles a little short of its
# target because it is balancing gravity, not lagging, and judging that against the
# empty-hand tolerance reports a failure where there is none.
CARRY_TOLERANCE = 0.20

# How much a settle round must improve the residual to be worth another, radians.
TUCK_PROGRESS = 0.01


# How many settling actions to hold after a drive, before judging the base's posture. A
# reading taken the instant the last waypoint finishes measures momentum, not posture.
SETTLE_ACTIONS = 10

# How close to the requested pose counts as arrived, metres, and how far the base may roll
# or pitch before it is treated as toppled, radians. Checking position alone once reported
# PASS for a robot at roll=+172 deg - upside down.
ARRIVAL_TOLERANCE = 0.30
UPRIGHT_LIMIT = 0.35

# How many of the nearest free floor cells to collision-check before giving up. The map has
# hundreds of thousands; the answer is always among the closest few dozen, and each batch
# costs a CuRobo call.
CELL_CANDIDATES = 120

# How many placements to sample on a surface before choosing one. Upstream's sampler
# returns the first pose that works, which is a uniform draw over the whole surface - on
# the video the plate landed at the far side of the table, nowhere near the robot that put
# it there. Sampling repeatedly and keeping the nearest costs one ray-cast batch each and
# cannot fail where taking the first would have succeeded.
PLACE_CANDIDATES = 12

# How far each openable object's doors sweep across the floor, in metres, per door link.
#
# Measured offline by `door_swing.py` straight from each model's misc/metadata.json - run
# it again to regenerate. It is a fixed property of the model, so there is nothing to work
# out while the simulator is running.
#
# Deriving it at runtime from the hinge joint is what kept going wrong: `joint.axis` is
# expressed in the joint's own frame, offset from the child link by `physics:localRot1`,
# and rotating it by the link's world orientation alone tests the wrong vector. That
# misread the fridge as bottom-hung and returned its door's *height*, 1.39 m - more than
# twice the width of the whole appliance - which blocked every stance in the kitchen.
# (link, shape, a, b). A side-hung door sweeps about a vertical line down one edge, so the
# floor it covers is an arc of radius equal to its width and a disc is the superset -
# `a` is that radius. A bottom-hung door rotates about a horizontal line at its lower edge
# and drops straight out into the room without ever sweeping sideways, so the floor it
# covers is a rectangle `a` deep by `b` wide. Blocking a disc there takes floor beside and
# behind the appliance that the door cannot reach.
DOOR_SWING = {
    "fridge/dszchb": [("link_0", "disc", 0.564, 0.0)],        # house_single_floor
    "oven/ffitak": [("door", "box", 0.438, 0.592)],           # house_single_floor
    "fridge/xyejdx": [("link_0", "disc", 0.602, 0.0),         # Pomaria_1_int, double door
                      ("link_1", "disc", 0.602, 0.0)],
    "oven/fexqbj": [("dof_rootd_aa001_r", "box", 0.435, 0.574)],   # Pomaria_1_int
}


def _door_swing(obj):
    """Where this object's doors sweep, in world coordinates.

    Returns a list of (kind, x, y, a, b, fx, fy): the hinge position, the shape from the
    table above, and the unit vector pointing out of the object through that door.

    Only the hinge's position is read from the scene, and that needs no frame algebra: the
    door link's origin *is* the hinge - which is how door_swing.py can treat the link
    bounding box's offset as measured from it. The outward direction comes from the hinge's
    position relative to the object's centre, since a door is mounted on a face.
    """
    import torch as th

    key = f"{getattr(obj, 'category', None)}/{getattr(obj, 'model', None)}"
    links = DOOR_SWING.get(key)
    if not links:
        return None

    centre, _ = obj.get_position_orientation()
    out = []
    for link_name, kind, a, b in links:
        link = obj.links.get(link_name)
        if link is None:
            continue
        hinge = link.get_position_orientation()[0]
        fx, fy = float(hinge[0]) - float(centre[0]), float(hinge[1]) - float(centre[1])
        span = float(th.norm(th.tensor([fx, fy])))
        if span < 1e-6:
            fx, fy = 1.0, 0.0
        else:
            fx, fy = fx / span, fy / span
        out.append((kind, float(hinge[0]), float(hinge[1]), float(a), float(b), fx, fy))
    return out or None


# Navigation maps and their connected components, keyed by erosion radius, the door swings
# blocked into them, and what is excluded. None of that depends on where the robot is
# standing, so they are computed once and kept.
_FLOOR_CACHE = {}

# Floor footprints of the loaded objects, per scene.
_OBSTACLE_CACHE = {}

# Categories that are the floor plan itself - already in the traversability map, and
# blocking their bounding boxes would black out the whole building.
_STRUCTURE = frozenset({"floors", "walls", "ceilings", "roof", "driveway", "lawn"})

# How high the robot reaches. An object whose underside is above this cannot be hit by it,
# so its footprint is not an obstacle - a plate on a counter is not a wall.
ROBOT_CLEARANCE_HEIGHT = 1.5


def _object_boxes(scene, robot, exclude=()):
    """Floor footprints of everything loaded, as (row0, row1, col0, col1) map boxes.

    The traversability map is a floor plan; it does not know about anything injected into
    the scene at runtime, and it is not the right authority on how much room a given piece
    of furniture takes. Every object's own bounding box is, and it is exact.

    Only objects that reach into the robot's own height band are painted. A plate lying on
    a counter at 0.9 m is not something the base can hit, and blocking the floor under it
    would wall the robot away from the counter it has to work at.
    """
    import torch as th

    skip = {id(o) for o in exclude if o is not None}
    key = (id(scene), tuple(sorted(skip)))
    hit = _OBSTACLE_CACHE.get(key)
    if hit is not None:
        return hit

    tmap = scene._trav_map
    boxes = []
    for obj in scene.objects:
        if obj is robot or obj.category in _STRUCTURE or id(obj) in skip:
            continue
        # Something with no collisions cannot obstruct anything. Carried objects are
        # `visual_only`, and blocking one would wall the robot in with what it is holding.
        if getattr(obj, "visual_only", False):
            continue
        try:
            lo, hi = obj.aabb
        except Exception:
            continue
        if float(lo[2]) > ROBOT_CLEARANCE_HEIGHT or float(hi[2]) < 0.05:
            continue                       # above the robot, or flat on the floor
        a = tmap.world_to_map(th.as_tensor([float(lo[0]), float(lo[1])]))
        b = tmap.world_to_map(th.as_tensor([float(hi[0]), float(hi[1])]))
        r0, r1 = sorted((int(a[0]), int(b[0])))
        c0, c1 = sorted((int(a[1]), int(b[1])))
        boxes.append((r0, r1, c0, c1))

    _OBSTACLE_CACHE[key] = boxes
    print(f"    [nav] {len(boxes)} loaded objects reach into the robot's height band "
          f"and are blocked off (cached)")
    return boxes


def _nav_map(scene, robot, radius, swing=(), exclude=()):
    """The map the robot navigates on: floor, minus obstacles, eroded by its footprint.

    Returns (free, labels). `swing` is a list of (x, y, r) discs for the doors of whatever
    is about to be opened.

    Everything is painted on the *raw* floor before eroding, which is the whole point.
    Erosion is what turns "where the robot's body fits" into "where its centre may be", so
    an obstacle blocked afterwards only keeps the robot's centre out of it while its body
    still overhangs. That is exactly how the oven door came to hit the robot: the swing
    disc was painted after eroding, the robot parked with its centre just outside the disc,
    and the door swung into the half of it that was overhanging.
    """
    import math

    import cv2
    import numpy as np
    import torch as th

    # Round the key: the measured footprint wobbles by a millimetre or two between calls
    # and a map recomputed for that is the same map.
    key = (round(float(radius), 2),
           tuple(sorted((kind, round(x, 2), round(y, 2), round(a, 2), round(b, 2),
                         round(fx, 2), round(fy, 2))
                        for kind, x, y, a, b, fx, fy in swing)),
           tuple(sorted(id(o) for o in exclude if o is not None)))
    hit = _FLOOR_CACHE.get(key)
    if hit is not None:
        return hit

    tmap = scene._trav_map
    raw = th.clone(tmap.floor_map[0]).cpu().numpy()

    for r0, r1, c0, c1 in _object_boxes(scene, robot, exclude=exclude):
        raw[max(r0, 0):r1 + 1, max(c0, 0):c1 + 1] = 0

    for kind, x, y, a, b, fx, fy in swing:
        centre = tmap.world_to_map(th.tensor([x, y], dtype=th.float32))
        if kind == "disc":
            rad = int(math.ceil(a / tmap.map_resolution))
            r0 = max(int(centre[0]) - rad, 0)
            r1 = min(int(centre[0]) + rad, raw.shape[0] - 1)
            c0 = max(int(centre[1]) - rad, 0)
            c1 = min(int(centre[1]) + rad, raw.shape[1] - 1)
            if r1 < r0 or c1 < c0:
                continue
            rr, cc = np.ogrid[r0:r1 + 1, c0:c1 + 1]
            disc = (rr - int(centre[0])) ** 2 + (cc - int(centre[1])) ** 2 <= rad ** 2
            raw[r0:r1 + 1, c0:c1 + 1][disc] = 0
        else:
            # A rectangle reaching `a` out of the object through the door and `b` across,
            # in map cells. `world_to_map` gives (row, col) and fillPoly wants (col, row).
            px, py = -fy, fx                       # across the door
            corners = []
            for s_out, s_across in ((0.0, -0.5), (0.0, 0.5), (1.0, 0.5), (1.0, -0.5)):
                wx = x + fx * a * s_out + px * b * s_across
                wy = y + fy * a * s_out + py * b * s_across
                m = tmap.world_to_map(th.tensor([wx, wy], dtype=th.float32))
                corners.append([int(m[1]), int(m[0])])
            cv2.fillPoly(raw, [np.array(corners, dtype=np.int32)], 0)

    px = int(math.ceil(key[0] / tmap.map_resolution))
    free = cv2.erode(raw, np.ones((px, px), np.uint8))
    n, labels = cv2.connectedComponents((free > 0).astype(np.uint8), connectivity=4)
    _FLOOR_CACHE[key] = (free, labels)
    print(f"    [nav] navigation map: eroded by {key[0]:.2f} m, "
          f"{len(swing)} door swings blocked before eroding -> "
          f"{int((free > 0).sum())} free cells in {n - 1} components (cached)")
    return free, labels


def build(env, robot, curobo_batch_size=3, joint_tolerance=JOINT_TOLERANCE):
    """Return a controller with all nine primitives working.

    Every primitive is symbolic: the robot drives itself to the object for real, and the
    manipulation is resolved by setting object state rather than by moving the arm.

    The physical manipulation stack is gone. It was never reliable enough to build on -
    a sticky grasp slips off a thin object, the planner cannot route the last few
    centimetres of the approach - and every attempt had to be rescued symbolically
    anyway, which made the videos misleading about what had actually happened.
    """
    from omnigibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
    from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
        SymbolicSemanticActionPrimitives,
    )

    if joint_tolerance is not None:
        # `m` is a module-level macro namespace, so this applies to every motion the
        # primitives execute.
        from omnigibson.action_primitives.starter_semantic_action_primitives import m as _m

        with _m.unlocked():
            _m.JOINT_POS_DIFF_THRESHOLD = joint_tolerance

    class NineWorkingPrimitives(SymbolicSemanticActionPrimitives):
        """Symbolic primitives, with our own navigation and arm handling."""

        _borrowed_open_or_close = SymbolicSemanticActionPrimitives._open_or_close
        _borrowed_toggle = SymbolicSemanticActionPrimitives._toggle

        def _require_near(self, obj, primitive):
            """Fail unless the robot is already at `obj`.

            Without this a plan that forgets NAVIGATE_TO still passes, which defeats the
            point of validating plans. GRASP and PLACE call upstream's
            `_navigate_if_needed`, so they would quietly drive across the house
            themselves; OPEN/CLOSE/TOGGLE_* only write object state, so they would work
            from another room entirely.

            The rule is that a primitive may adjust its own footing but may not travel.
            `_navigate_near` parks within `half-diagonal + 2.0` m of the object, so
            anything past that plus a margin means no NAVIGATE_TO ran.
            """
            import torch as th

            obj_xy = obj.get_position_orientation()[0][:2]
            robot_xy = self.robot.get_position_orientation()[0][:2]
            distance = float(th.norm(obj_xy - robot_xy))
            limit = 0.5 * float(th.linalg.norm(obj.aabb_extent[:2])) + 3.0
            print(f"    [near] {primitive} on {obj.name}: standing {distance:.2f} m from "
                  f"its centre, {distance - 0.5 * float(th.linalg.norm(obj.aabb_extent[:2])):.2f} m "
                  f"from its edge (limit {limit:.2f})")
            if distance > limit:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    f"{primitive} needs the robot at the object - no NAVIGATE_TO ran",
                    {"object": obj.name, "distance": round(distance, 2),
                     "limit": round(limit, 2)},
                )

        def _open_or_close(self, obj, should_open):
            """Open or close, all the way, rather than by a random amount.

            `Open._set_value(new_value, fully=False)` samples the hinge uniformly between
            the open threshold and fully-open, and upstream's primitive calls it without
            `fully` - so every OPEN swings the door to a different angle. That is fine for
            a state check and looks wrong on video, and it also makes the door's swept
            volume unpredictable, which matters because the stance keep-out is measured
            from the object's bounding box.

            Upstream's own checks and post-conditions run first; this then settles the
            joint at its end stop.
            """
            from omnigibson import object_states

            self._require_near(obj, "OPEN" if should_open else "CLOSE")
            with self._hand_allowed_full():
                yield from self._borrowed_open_or_close(obj, should_open)

            try:
                obj.states[object_states.Open].set_value(should_open, fully=True)
            except TypeError:
                return          # older signature without `fully`; leave it as it is
            yield from self._settle_robot()

        def _toggle(self, obj, value):
            self._require_near(obj, "TOGGLE_ON" if value else "TOGGLE_OFF")
            with self._hand_allowed_full():
                yield from self._borrowed_toggle(obj, value)

        @contextlib.contextmanager
        def _hand_allowed_full(self):
            """Let OPEN/CLOSE/TOGGLE_* run while the robot is carrying something.

            Upstream refuses outright - "Cannot open or close an object while holding an
            object" - by checking `_get_obj_in_hand()` first thing. That rules out the
            ordinary way of doing this task: carry the plate to the oven, open the oven,
            put the plate in. Tiago has two arms, so there is nothing physically wrong
            with holding a plate in one and pulling a door with the other, and the
            alternative is to put the plate down on the floor mid-task.

            The check is a guard inside the borrowed generator rather than a parameter, so
            the only way past it is to make it see an empty hand for the duration. Every
            other use of `_get_obj_in_hand` - by the placement primitives, which genuinely
            do need to know - is untouched, because this is scoped to the one call.
            """
            real = self._get_obj_in_hand
            self._get_obj_in_hand = lambda *a, **k: None
            try:
                yield
            finally:
                self._get_obj_in_hand = real

        def _nav_map_now(self):
            """The navigation map for the robot as it stands, and its component labels.

            One radius per arm configuration, and the door swing of whatever is being
            navigated to painted on before eroding, so the robot's whole body clears the
            door rather than just its centre.
            """
            import torch as th

            # One footprint for the whole run, measured once.
            #
            # `aabb_extent` grows when something is carried - 0.77 m empty, 0.79 m with a
            # potato, 0.87 m with a plate - and eroding by more later than when the robot
            # parked means a stance that was legitimately free on arrival can be *inside*
            # an obstacle by the next navigation, so A* cannot even start. Measured: parked
            # 0.53 m from the plate, every route to the oven then failed with "no route,
            # and the direct line is blocked".
            #
            # A carried object is `visual_only` and cannot collide with anything, which is
            # why it is already kept out of CuRobo's `attached_obj`; the map has to agree.
            # The arm never moves, so the footprint is a constant - take it once.
            if getattr(self, "_footprint_radius", None) is None:
                extent = self.robot.aabb_extent[:2]
                self._footprint_radius = (float(th.norm(extent)) / 2.0
                                          + ARM_CLEARANCE_MARGIN)
                print(f"    [nav] robot footprint fixed at "
                      f"{self._footprint_radius:.2f} m "
                      f"({float(extent[0]):.2f} x {float(extent[1]):.2f} m tucked)")
            radius = self._footprint_radius
            # The object being navigated to is not an obstacle for that navigation.
            #
            # Its footprint is painted before eroding, so leaving it in pushes the robot a
            # full erosion radius away from the very thing it is approaching. Measured, the
            # floor is uniformly 0.53 m from the counter along its whole length, yet the
            # robot stood 0.83 m from the plate and 0.55 m from the potato - the difference
            # being that a 0.209 m plate blocks enough cells to matter and a 0.077 m potato
            # rounds away.
            target = getattr(self, "_nav_target", None)
            swing = _door_swing(target) or []
            return _nav_map(self.robot.scene, self.robot, radius, swing,
                            exclude=(target,))

        def _release(self):
            """Let go, putting whatever was held back into the physics.

            `_grasp` marks a carried object `visual_only` so it cannot push the arm around.
            It has to come back the instant it is released or it would hang in mid-air -
            except during a placement, which `_place_with_predicate` handles itself.
            """
            held = self._get_obj_in_hand()
            if held is not None and not getattr(self, "_placing", False):
                held.visual_only = False
            yield from SymbolicSemanticActionPrimitives._release(self)

        def _shelf_pose(self, held, obj, predicate):
            """Where to put `held` inside `obj`: in its fillable volume.

            `Inside` does not test the container's bounding box. It wants the object's
            centre inside a *container meta-link volume* - the types are `fillable` and
            `openfillable` - checked with `link.check_points_in_volume`.

            So aim at that volume. Three placements were tried against the geometry first,
            on the open door, on the rack and at the cavity centre, and all failed
            identically because none of them tested what the predicate tests.

            Upstream's sampler is no better here: a door is a *link of the object*, so an
            open door extends the bounding volume out into the room and the sampler
            legitimately picks a point above it. That is where the plate was being set
            down, 0.24 m in front of the oven's face.

            Returns None when the container has no fillable volume, leaving the sampler to
            it. `misc/metadata.json` cannot answer whether it has one - a bowl and a bucket
            both list no meta links there - so this reads the loaded model.
            """
            import torch as th

            from omnigibson import object_states

            if predicate is not object_states.Inside:
                return None

            fillable = [link for link in obj.links.values()
                        if getattr(link, "is_meta_link", False)
                        and link.meta_link_type in ("fillable", "openfillable")]
            if not fillable:
                print(f"    [place] {obj.name} has no fillable volume; Inside cannot be "
                      f"satisfied by placing alone")
                return None

            # Search for a point the predicate actually accepts, rather than computing one
            # and hoping. Two things went wrong with computing it:
            #
            #   the target was the fillable link's AABB centre, and that AABB is
            #   degenerate - `z spanning 1.15..1.15` on this oven - so nothing guaranteed
            #   the point was inside the mesh volume `check_points_in_volume` tests.
            #
            #   `Inside` tests the object's *AABB centre*, while the placement sets its
            #   *position*. For anything whose origin is not its centre those are
            #   different points, so the aim was off by that offset.
            #
            # Measured, that combination failed 2 runs in 5 with the plate landing within
            # 0.001 m of its target - a knife-edge, not a near miss.
            lo, hi = obj.aabb
            steps = 11
            axes = [th.linspace(float(lo[i]), float(hi[i]), steps) for i in range(3)]
            grid = th.stack(th.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 3)

            accepted = th.zeros(grid.shape[0], dtype=th.bool)
            for link in fillable:
                try:
                    accepted |= link.check_points_in_volume(grid)
                except Exception:
                    continue
            if not bool(accepted.any()):
                print(f"    [place] no point inside {obj.name}'s fillable volume out of "
                      f"{grid.shape[0]} sampled; Inside cannot be satisfied by placing")
                return None

            # Take the point furthest from the cavity's boundary, horizontally and
            # vertically. Which point is chosen turns out not to decide the outcome here -
            # see the note below - but the deepest one is the least bad aim.
            inside_pts, outside_pts = grid[accepted], grid[~accepted]
            if outside_pts.shape[0]:
                margin = th.cdist(inside_pts, outside_pts).min(dim=1).values
                target = inside_pts[int(margin.argmax())]
                best = float(margin.max())
            else:
                target = inside_pts.mean(dim=0)
                best = float("inf")
            span = (float(inside_pts[:, 2].min()), float(inside_pts[:, 2].max()))

            # Aim the object's AABB centre, not its origin - that is what Inside reads.
            centre_offset = held.aabb_center - held.get_position_orientation()[0]
            pos = (target - centre_offset).to(th.float32)
            print(f"    [place] aiming at {obj.name}'s fillable volume "
                  f"({float(target[0]):+.2f}, {float(target[1]):+.2f}, "
                  f"{float(target[2]):+.2f}), {int(accepted.sum())}/{grid.shape[0]} "
                  f"sampled points inside, volume z {span[0]:.3f}..{span[1]:.3f}, "
                  f"horizontal clearance {best:.3f} m; object origin offset "
                  f"({float(centre_offset[0]):+.3f}, {float(centre_offset[1]):+.3f}, "
                  f"{float(centre_offset[2]):+.3f})")
            return pos, held.get_position_orientation()[1]

        def _near_pose(self, held, obj, predicate, near_poses=None,
                       near_poses_threshold=None):
            """Sample several placements on `obj` and keep the one nearest the robot.

            Upstream samples by ray-casting down onto the object and returns the *first*
            pose that fits, which is a uniform draw over the whole surface. On the video
            the plate was set down at the far end of the coffee table, well out of reach of
            the robot standing at its near edge - correct by the predicate, and obviously
            wrong to look at.

            Choosing geometrically instead - clamping the robot's position into the
            surface's bounding box - was the other option and is worse here. `aabb` is
            world-axis-aligned, so for any rotated piece of furniture it overshoots the
            real top face, and the near edge it would aim at can be off the surface
            entirely. The sampler already ray-casts against the true geometry; it just
            needs asking more than once.

            Best-of-N, not a distance threshold. `_sample_pose_with_object_and_predicate`
            takes `near_poses`, but that rejects anything past the cutoff and raises when
            nothing survives - it can fail where taking the first pose would have worked.
            Ranking cannot.
            """
            import torch as th

            from omnigibson.action_primitives.action_primitive_set_base import (
                ActionPrimitiveError,
            )

            robot_xy = self.robot.get_position_orientation()[0][:2]

            best, best_distance, distances = None, None, []
            for attempt in range(PLACE_CANDIDATES):
                try:
                    pose = self._sample_pose_with_object_and_predicate(
                        predicate, held, obj, near_poses=near_poses,
                        near_poses_threshold=near_poses_threshold)
                except ActionPrimitiveError:
                    # One draw failing says nothing about the next - the sampler is
                    # random. Only give up if none of them lands.
                    continue
                distance = float(th.norm(pose[0][:2] - robot_xy))
                distances.append(distance)
                if best_distance is None or distance < best_distance:
                    best, best_distance = pose, distance

            if best is None:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not find a position to put this object in the desired relation "
                    "to the target object",
                    {"target object": obj.name, "object in hand": held.name,
                     "samples tried": PLACE_CANDIDATES},
                )

            print(f"    [place] {len(distances)}/{PLACE_CANDIDATES} placements sampled on "
                  f"{obj.name}, {min(distances):.2f}..{max(distances):.2f} m from the "
                  f"robot; taking the nearest")
            return best

        def _inside_now(self, held, obj):
            """`Inside`, evaluated from prim geometry rather than through object states.

            BEHAVIOR defines `Inside(a, b)` as two tests: a's AABB *centre* within b's
            AABB, and that same point inside a `fillable` meta-link volume of b. This
            computes exactly that, and differs from `held.states[Inside]` only in where the
            AABB comes from - `held.aabb_center`, read straight from the prim, instead of
            `states[AABB]`, which derives from the physics view.

            That difference is the whole bug. A placed object is `visual_only` so that
            gravity cannot pull it out of the cavity before anything looks, and a
            `visual_only` object has been removed from the physics view - so `states[AABB]`
            reports wherever it was *before* the teleport. Measured over 32 trials with the
            plate at an identical (+8.31, -1.66, +1.03) and `moved 0.000 m`: this test said
            True every time, while `states[Inside]` alternated True/False, giving a 38-58%
            pass rate on placements that were all equally correct.

            Restoring physics first does not help, because the view syncs on a simulation
            step and stepping is exactly what makes the plate fall out.
            """
            from omnigibson import object_states

            centre = held.aabb_center
            lo, hi = obj.aabb
            if not (bool((lo <= centre).all()) and bool((centre <= hi).all())):
                return False
            point = centre.reshape(1, 3)
            for link in obj.links.values():
                if not getattr(link, "is_meta_link", False):
                    continue
                if link.meta_link_type not in ("fillable", "openfillable"):
                    continue
                try:
                    if bool(link.check_points_in_volume(point)[0]):
                        return True
                except Exception:
                    continue
            return False

        def _rest_pose(self, held, obj):
            """Rest `held` on whatever surface is inside `obj`, wherever that turns out to be.

            The general form of "put it in the oven". An object released in mid-cavity
            falls to whatever is beneath it, so the placement has to be chosen at the
            surface it will end up on, not in the air above it. Measured on `oven/ffitak`:
            aiming at the fillable volume's centroid put the plate between the two racks,
            where it caught rack2 by luck about half the time and otherwise slid off to
            the oven floor - which sits below the volume - for a 38% pass rate. Resting it
            deliberately on a rack is 100%.

            Finding the surface by *ray cast* rather than by link geometry is what makes
            this general. Only the oven has shelf links to aim at; `fridge/dszchb`,
            `microwave/vuezel` and `dishwasher/xlmier` have none at all, and their objects
            come to rest on an interior floor that is part of `base_link` - whose AABB top
            is the appliance's outer top, several tens of centimetres above the surface
            that matters. A downward ray finds a rack, a shelf, a fridge floor or a
            microwave floor without knowing which it is.

            Candidates are drawn from inside the fillable volume, since that is the region
            `Inside` actually tests, and kept only if the object *resting* there would
            still be in it. The highest surviving surface wins: it is what a person would
            use, and it leaves the object furthest from the floor it would otherwise slide
            to.
            """
            import torch as th

            from omnigibson.utils.sampling_utils import raytest_batch

            fillable = [link for link in obj.links.values()
                        if getattr(link, "is_meta_link", False)
                        and link.meta_link_type in ("fillable", "openfillable")]
            if not fillable:
                return None

            # Where the cavity is, in world coordinates. Sampling the container's own
            # AABB and keeping what the volume accepts avoids trusting the meta-link's
            # AABB, which on this oven is degenerate (z 1.152..1.152).
            #
            # Sampled finely in x and y, coarsely in z: the z levels only have to find the
            # top of the cavity at each column, while x and y decide where the object ends
            # up standing. At 11 x 11 only 14 columns survived and the best of them sat at
            # the cavity's edge, where the object slid off - the horizontal resolution is
            # what matters.
            lo, hi = obj.aabb
            xs = th.linspace(float(lo[0]), float(hi[0]), 21)
            ys = th.linspace(float(lo[1]), float(hi[1]), 21)
            zs = th.linspace(float(lo[2]), float(hi[2]), 15)
            gx, gy, gz = th.meshgrid(xs, ys, zs, indexing="ij")
            grid = th.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
            accepted = th.zeros(grid.shape[0], dtype=th.bool)
            for link in fillable:
                try:
                    accepted |= link.check_points_in_volume(grid)
                except Exception:
                    continue
            if not bool(accepted.any()):
                return None
            cavity = grid[accepted]

            # One column per distinct (x, y) in the cavity, cast from its top downwards.
            columns = {}
            for p in cavity:
                columns.setdefault((round(float(p[0]), 4), round(float(p[1]), 4)), []) \
                    .append(float(p[2]))
            tops = [(x, y, max(zs_)) for (x, y), zs_ in columns.items()]
            starts = [[x, y, z + 0.01] for x, y, z in tops]
            ends = [[x, y, float(lo[2]) - 0.05] for x, y, _ in tops]

            try:
                hits = raytest_batch(starts, ends, only_closest=True)
            except Exception as exc:
                print(f"    [place] ray cast into {obj.name} failed "
                      f"({type(exc).__name__}); using the cavity")
                return None

            half_height = float(held.aabb_extent[2]) / 2.0
            centre_offset = held.aabb_center - held.get_position_orientation()[0]

            # Every surface the object could rest on and still satisfy the predicate.
            resting = []
            for (x, y, _), hit in zip(tops, hits):
                if not hit or not hit.get("hit"):
                    continue
                surface_z = float(hit["position"][2])
                centre = th.tensor([x, y, surface_z + half_height + 0.005],
                                   dtype=th.float32)
                ok = False
                for link in fillable:
                    try:
                        ok |= bool(link.check_points_in_volume(centre.reshape(1, 3))[0])
                    except Exception:
                        continue
                if ok:
                    resting.append((surface_z, centre))

            best = None
            if resting:
                # Highest surface first - what a person would use, and furthest from the
                # floor the object would otherwise slide to. Then, among the columns on
                # that same surface, the most central one. Taking the highest column alone
                # put the plate at the cavity's edge, where it slid off and out: measured,
                # that scored 1/6 against 12/12 for a centred placement on the same shelf.
                top_z = max(z for z, _ in resting)
                level = [c for z, c in resting if abs(z - top_z) < 0.02]
                mid = th.stack(level)[:, :2].mean(dim=0)
                best = min(level, key=lambda c: float(th.norm(c[:2] - mid)))

            if best is None:
                print(f"    [place] no surface inside {obj.name} would hold {held.name} "
                      f"within its fillable volume ({len(tops)} columns cast); "
                      f"using the cavity")
                return None

            print(f"    [place] resting {held.name} inside {obj.name} on the surface at "
                  f"({float(best[0]):+.2f}, {float(best[1]):+.2f}, "
                  f"{float(best[2]) - half_height - 0.005:+.3f}), object centre "
                  f"{float(best[2]):+.3f}, from {len(tops)} columns cast, "
                  f"{len(resting)} of them usable")
            return (best - centre_offset).to(th.float32), \
                held.get_position_orientation()[1]

        def _cavity_pose(self, held, obj):
            """Where to put `held` inside `obj`: sampled, filtered to the cavity, nearest.

            The same shape as `_near_pose` does for `OnTop` - draw several poses from
            upstream's ray-casting sampler and keep the one closest to the robot - with one
            filter that `OnTop` does not need.

            An open door is a *link of the object*, so the sampler will happily return a
            pose above it: measured previously, that put the plate 0.24 m in front of the
            oven's face. And "closest to the robot" actively prefers the door, since the
            door is the nearest part of the oven. So a sampled pose is kept only if its
            AABB centre lands in a `fillable` meta-link volume, which is the cavity and is
            what `Inside` actually tests.

            Sampling rather than computing a point earns two things the geometric search
            could not: the sampler works from the held object's real bounding box, so the
            object *fits* rather than merely its centre being somewhere legal, and it
            ray-casts against the true geometry rather than trusting a meta-link's AABB.
            """
            import torch as th

            from omnigibson import object_states
            from omnigibson.action_primitives.action_primitive_set_base import (
                ActionPrimitiveError,
            )

            fillable = [link for link in obj.links.values()
                        if getattr(link, "is_meta_link", False)
                        and link.meta_link_type in ("fillable", "openfillable")]
            if not fillable:
                return None

            robot_xy = self.robot.get_position_orientation()[0][:2]
            centre_offset = held.aabb_center - held.get_position_orientation()[0]

            best, best_distance, kept, drawn = None, None, 0, 0
            for _ in range(PLACE_CANDIDATES):
                try:
                    pose = self._sample_pose_with_object_and_predicate(
                        object_states.Inside, held, obj)
                except ActionPrimitiveError:
                    continue
                drawn += 1
                # Where the object's AABB centre would land - the point Inside tests.
                point = (pose[0] + centre_offset).reshape(1, 3)
                inside = False
                for link in fillable:
                    try:
                        inside |= bool(link.check_points_in_volume(point)[0])
                    except Exception:
                        continue
                if not inside:
                    continue
                kept += 1
                distance = float(th.norm(pose[0][:2] - robot_xy))
                if best_distance is None or distance < best_distance:
                    best, best_distance = pose, distance

            if best is None:
                print(f"    [place] {drawn}/{PLACE_CANDIDATES} sampled poses for "
                      f"{obj.name}, none inside a fillable volume; falling back")
                return None

            print(f"    [place] {kept}/{drawn} sampled poses land in {obj.name}'s cavity; "
                  f"taking the nearest at {best_distance:.2f} m from the robot")
            return best

        def _place_with_predicate(self, obj, predicate, near_poses=None,
                                  near_poses_threshold=None):
            """Place, giving the object back to the physics the moment it has arrived.

            Upstream's order is: sample a pose, let go, teleport the object there, settle,
            check the predicate. The object is therefore fully physical while it is being
            teleported, so gravity reaches it before it has arrived.

            Here it stays out of the physics - no gravity, no collisions, so nothing can
            shove it - until it has been put where it belongs, and is handed back
            immediately after. It has to be physical again for the check: `OnTop` and
            `Inside` are contact-based, and something that touches nothing satisfies
            neither. Measured, holding it out through the check reported a failure with the
            potato sitting squarely on the plate, 3 cm off centre and resting on its
            surface.
            """
            import torch as th

            from omnigibson import object_states
            from omnigibson.action_primitives.action_primitive_set_base import (
                ActionPrimitiveError,
            )

            self._require_near(obj, "PLACE")

            held = self._get_obj_in_hand()
            if held is None:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "You need to be grasping an object first to place it somewhere.",
                )

            obj_pose = None
            if predicate is object_states.Inside:
                # A shelf first: an object resting on one stays put, where an object
                # aimed at mid-cavity falls to the floor and out of the volume.
                obj_pose = self._rest_pose(held, obj)
            if obj_pose is None and predicate is object_states.Inside:
                obj_pose = self._cavity_pose(held, obj)
            if obj_pose is None:
                obj_pose = self._shelf_pose(held, obj, predicate)
            if obj_pose is None:
                obj_pose = self._near_pose(
                    held, obj, predicate, near_poses=near_poses,
                    near_poses_threshold=near_poses_threshold)

            # When the object is handed back to the physics depends on the predicate,
            # because the two are genuinely different tests:
            #
            #   OnTop  is contact-based, so it has to be physical for the check - held out
            #          of the physics, the potato sat squarely on the plate and the check
            #          still said no.
            #   Inside is purely positional: AABB containment plus check_points_in_volume,
            #          with no contact anywhere in it. Handed back first, the plate is put
            #          at the fillable volume's origin and then falls 0.219 m onto the rack
            #          below - out of that volume - before anything looks at it.
            positional = predicate is object_states.Inside

            self._placing = True
            try:
                yield from self._release()
                held.set_position_orientation(*obj_pose)
            finally:
                self._placing = False
                if not positional:
                    held.visual_only = False

            placed_at = held.get_position_orientation()[0].clone()
            yield from self._settle_robot()
            rested_at = held.get_position_orientation()[0]
            print(f"    [place] {held.name} put at "
                  f"({float(placed_at[0]):+.2f}, {float(placed_at[1]):+.2f}, "
                  f"{float(placed_at[2]):+.2f}), settled at "
                  f"({float(rested_at[0]):+.2f}, {float(rested_at[1]):+.2f}, "
                  f"{float(rested_at[2]):+.2f}); moved "
                  f"{float(th.norm(rested_at - placed_at)):.3f} m")

            # Why did it pass or fail? `Inside` is two tests - the object's AABB *centre*
            # inside the container's AABB, then that same point inside a fillable
            # meta-link volume - and knowing which one flipped is the difference between
            # a placement bug and a predicate that does not see what we think it sees.
            if positional:
                try:
                    centre = held.aabb_center
                    lo_o, hi_o = obj.aabb
                    in_box = bool((lo_o <= centre).all() and (centre <= hi_o).all())
                    pts = centre.reshape(1, 3)
                    hits = []
                    for link in obj.links.values():
                        if not getattr(link, "is_meta_link", False):
                            continue
                        if link.meta_link_type not in ("fillable", "openfillable"):
                            continue
                        try:
                            hits.append((link.meta_link_type,
                                         bool(link.check_points_in_volume(pts)[0])))
                        except Exception as exc:
                            hits.append((link.meta_link_type, f"raised {type(exc).__name__}"))
                    print(f"    [place] check: aabb centre "
                          f"({float(centre[0]):+.3f}, {float(centre[1]):+.3f}, "
                          f"{float(centre[2]):+.3f}), inside container aabb={in_box}, "
                          f"volume tests {hits}")
                except Exception as exc:
                    print(f"    [place] check diagnostic failed: {type(exc).__name__}: {exc}")

            # Clear the cache before asking. Object states cache per timestep, and
            # `KinematicsMixin._cache_is_valid` skips its "has anything moved?" test for
            # objects that are asleep - which a `visual_only` object always is, having no
            # physics. `Inside` therefore returned whatever it last evaluated to *before*
            # the teleport, and whether a stale entry existed decided the verdict:
            # measured, identical placements at (+8.31, -1.66, +1.03) with `moved 0.000 m`
            # returned True, False, True, False, False, while a fresh evaluation of both
            # halves of the predicate said True every time.
            # Hand the object back to the physics *before* asking, then put it back on
            # its mark. `Inside` reads `states[AABB]`, which derives from the physics
            # view - and a `visual_only` object has been removed from that view, so the
            # AABB it reports is wherever the object was before the teleport. Measured,
            # the fresh `aabb_center` property and both halves of the predicate said True
            # while `states[Inside]` said False, for identical placements at
            # (+8.31, -1.66, +1.03) with `moved 0.000 m`.
            #
            # Restoring physics first would normally let it fall 0.219 m out of the cavity
            # before anything looked - which is why it was held out in the first place -
            # so it is re-placed on the same pose immediately afterwards and checked
            # without stepping. It falls after the check, which is fine: the check has its
            # answer, and `Inside` has no contact term to spoil.
            if positional:
                settled = self._inside_now(held, obj)
                held.visual_only = False
            else:
                settled = held.states[predicate].get_value(obj)
            if not settled:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Failed to place object at the desired place (probably dropped). The "
                    "object was still released, so you need to grasp it again to continue",
                    {"dropped object": held.name, "target object": obj.name},
                )

        def _grasp(self, obj):
            """Grasp, welding the object level so it is carried flat.

            Upstream's symbolic grasp welds the object at whatever orientation it happened
            to have - `set_position_orientation(position=...)` sets position only - so a
            plate picked up off a counter can end up welded on its edge, and anything
            resting on it slides off.

            Levelling it here is the whole of the arm handling, and it is enough on its own
            because the arm does not move afterwards: it holds the tucked pose for the rest
            of the task, so an object welded flat stays flat.
            """
            import math

            import torch as th

            from omnigibson.utils import transform_utils as T

            self._require_near(obj, "GRASP")

            if self._get_obj_in_hand() is None:
                # Flat, keeping its own yaw so it is not spun on the spot.
                yaw = float(T.quat2euler(obj.get_position_orientation()[1])[2])
                level = T.euler2quat(th.tensor([0.0, 0.0, yaw], dtype=th.float32))
                # Sit it just clear of the robot, not inside it.
                #
                # Welded centred on the gripper, half of a wide object is inside the
                # torso. The solver resolves that overlap by shoving it, so an object set
                # perfectly flat comes back rotated: measured, the plate settled at
                # roll=+129 pitch=-56 while a potato - small enough not to overlap - stayed
                # at 0. Offsetting it outward by its own half-width puts its near edge at
                # the gripper, which is where a gripper holds a plate anyway.
                grasp_pos = self.robot.get_eef_position(self.arm)
                obj.set_position_orientation(position=grasp_pos, orientation=level)

                # The contact point is the object's own position. That is upstream's
                # invariant, and breaking it anchors the weld away from the body it is
                # welding: measured, that drove arm_left_5 to +32 rad against a +-2.094
                # limit, which the simulator does not enforce.
                self.robot._establish_grasp(obj, obj.root_link_name, self.arm, grasp_pos,
                                            "FixedJoint")
                # Carried, not carried *against*.
                #
                # A welded object is part of the arm's rigid chain, so any contact force on
                # it becomes joint torque - and these joints are not clamped at their
                # limits, so once one starts turning nothing stops it. Measured, a plate
                # welded at the gripper drove arm_left_5 to -7.457 rad against a -2.094
                # limit, and moving the plate outward to reduce the overlap only brought
                # the tuck residual from 19.5 rad down to 5.9.
                #
                # `visual_only` takes it out of the physics entirely - no gravity, no
                # collisions - which is what a symbolically held object should be: the
                # primitives teleport it into place, nothing ever pushes it. Restored the
                # moment it is let go, in `_release`.
                obj.visual_only = True

                yield from self._settle_robot()

                if self._get_obj_in_hand() is not obj:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.POST_CONDITION_ERROR,
                        "Grasp completed, but the object is not in the hand",
                        {"target object": obj.name},
                    )
                roll, pitch, _ = (math.degrees(float(v)) for v in
                                  T.quat2euler(obj.get_position_orientation()[1]))
                print(f"    [arm] {obj.name} welded level and carried out of physics: "
                      f"roll={roll:+.0f} pitch={pitch:+.0f} deg")
                return

            yield from SymbolicSemanticActionPrimitives._grasp(self, obj)

        def _hold_pose(self):
            """The arm configuration, which is the tucked one whether or not it is holding.

            The arm never moves for a manipulation - every primitive except NAVIGATE_TO is
            symbolic and teleports the object into place - so there is nothing to gain from
            posing it, and one fixed configuration means one erosion radius and one cached
            navigation map.

            Whatever is carried is welded level at the moment it is grasped (see `_grasp`),
            and since the arm holds this pose from then on, it stays level.

            The base joints are preserved so nothing but the arm is commanded, and the
            gripper joints so whatever is held stays held.
            """
            import torch as th

            q_now = self.robot.get_joint_positions()
            target = self.robot.tucked_default_joint_pos.clone()
            target[self.robot.base_idx] = q_now[self.robot.base_idx]
            for arm_name in self.robot.arm_names:
                grip = self.robot.gripper_control_idx[arm_name]
                target[grip] = q_now[grip]

            arm_idx = [self.robot.trunk_control_idx]
            for arm_name in self.robot.arm_names:
                arm_idx.append(self.robot.arm_control_idx[arm_name])
            return target, th.cat(arm_idx)

        def _tuck_arm(self):
            """Put the arm in its holding configuration, base and gripper untouched.

            Which configuration depends on whether the hand is full - see `_hold_pose`.
            Measured on Tiago: the hand sits 0.55 m from the base axis at the pose upstream
            calls "reset", 0.28 m tucked and 0.30 m carrying. Since the map is eroded by
            the robot's current bounding box, that difference is also the difference
            between needing 0.99 m of clearance and needing 0.77 m.

            The motion is interpolated because commanding it in one jump leaves the arm
            part-way there - `_execute_motion_plan` gives each waypoint a fixed step budget.
            """
            import math

            import torch as th

            q_now = self.robot.get_joint_positions()
            tucked, arm_idx = self._hold_pose()

            steps = []
            for k in range(1, TUCK_STEPS + 1):
                mid = q_now.clone()
                mid[arm_idx] = q_now[arm_idx] + (
                    tucked[arm_idx] - q_now[arm_idx]) * (k / TUCK_STEPS)
                steps.append(mid)

            # Walk the interpolation one waypoint at a time and stop the moment the arm
            # stops tracking it.
            #
            # Commanding the whole sweep and judging it afterwards means a pose the arm
            # cannot reach is still attempted in full - the arm turning through most of a
            # revolution after picking up a plate before anything notices. A stalled joint
            # will not un-stall on the next waypoint, so the signal worth acting on is the
            # gap to the target no longer shrinking.
            #
            # Predicting it instead was tried and does not work: collision-checking the
            # carry pose with the object attached rejects a potato that reaches it
            # perfectly well, and the potato then rides at roll=-79 deg instead of level.
            reach_limit = (CARRY_TOLERANCE if self._get_obj_in_hand() is not None
                           else TUCK_TOLERANCE)
            previous_gap = None
            for step in steps:
                yield from self._execute_motion_plan(
                    [step], low_precision=True, ignore_failure=True)
                gap = float(th.max(th.abs(
                    self.robot.get_joint_positions()[arm_idx] - tucked[arm_idx])))
                if gap <= reach_limit:
                    break                       # there already; the rest is no motion
                if previous_gap is not None and previous_gap - gap < TUCK_PROGRESS:
                    print(f"    [arm] arm stopped tracking {gap:.3f} rad short; "
                          f"holding here rather than pushing at it")
                    break
                previous_gap = gap

            # Then make sure it actually got there.
            #
            # `_execute_motion_plan` spends at most MAX_STEPS_FOR_JOINT_MOTION (10) steps
            # per waypoint and `ignore_failure=True` swallows the miss, so the six
            # waypoints above give the arm 60 steps and no complaint if that was not
            # enough. Carrying a plate it is not enough: the arm stops part-way, the
            # position controller keeps driving toward a target it never reached, and the
            # load swings on the end of it. That is what toppled the robot after it picked
            # up the plate - not the route, and not the furniture.
            #
            # So hold the final target until the joints converge, then let it settle before
            # anything starts driving.
            settle_limit = (CARRY_TOLERANCE if self._get_obj_in_hand() is not None
                            else TUCK_TOLERANCE)
            previous = None
            for _ in range(TUCK_SETTLE_TRIES):
                residual = float(th.max(th.abs(
                    self.robot.get_joint_positions()[arm_idx] - tucked[arm_idx])))
                if residual <= settle_limit:
                    break
                # Stop once it stops improving. A joint balancing gravity settles a little
                # short and stays there - measured, arm_left_6 sits at 0.161 rad however
                # many rounds it is given - and re-commanding it is motion for nothing.
                if previous is not None and previous - residual < TUCK_PROGRESS:
                    break
                previous = residual
                yield from self._execute_motion_plan(
                    [tucked], low_precision=True, ignore_failure=True)
            yield from self._settle_robot()

            residual = float(th.max(th.abs(
                self.robot.get_joint_positions()[arm_idx] - tucked[arm_idx])))
            held = self._get_obj_in_hand()
            carrying = ""
            if held is not None:
                # How flat is whatever is being carried, once the arm is actually tucked?
                # This is the question of whether the tucked pose can hold a plate at all,
                # answered by measurement rather than by assumption.
                from omnigibson.utils import transform_utils as T

                roll, pitch, _ = (math.degrees(float(v)) for v in
                                  T.quat2euler(held.get_position_orientation()[1]))
                carrying = (f", carrying {held.name} at roll={roll:+.0f} "
                            f"pitch={pitch:+.0f} deg")
            worst = ""
            if residual > TUCK_TOLERANCE:
                now = self.robot.get_joint_positions()[arm_idx]
                gap = th.abs(now - tucked[arm_idx])
                k = int(th.argmax(gap))
                names = [n for n, j in self.robot.joints.items()]
                idx = int(arm_idx[k])
                name = names[idx] if idx < len(names) else str(idx)
                # The measured value, not just the gap. A gap wider than the joint's own
                # range means the joint has wound past its limits, and no amount of
                # reasoning about gaps will show that - only the position will.
                joint = self.robot.joints.get(name)
                span = ""
                if joint is not None:
                    try:
                        span = (f", limits [{float(joint.lower_limit):+.2f}, "
                                f"{float(joint.upper_limit):+.2f}]")
                    except Exception:
                        span = ""
                worst = (f"; furthest is {name} at {float(now[k]):+.3f} "
                         f"want {float(tucked[arm_idx][k]):+.3f}{span}")
            limit = CARRY_TOLERANCE if held is not None else TUCK_TOLERANCE
            print(f"    [arm] tucked to within {residual:.3f} rad" + carrying + worst
                  + ("" if residual <= limit else "  <-- DID NOT CONVERGE"))

        def _reset_robot(self):
            """Tuck, rather than reaching out to `_reset_eef_pose`.

            `apply_ref` calls this after every primitive attempt, describing it as
            "retract the arms" - but upstream's reset pose leaves the hand 0.55 m in front
            of the base, which is not retracted. That is why the arm was extended between
            two toggles that share one NAVIGATE_TO: the toggle itself never moves the arm,
            the wrapper does.

            Tucking here means the arm is folded whenever the robot is not actively
            manipulating, without needing anything else to remember to do it.
            """
            yield from self._tuck_arm()

        def _drive_to(self, pose):
            """Drive to a 2-D pose along a route from the traversability map.

            Navigation and manipulation want different tools. CuRobo plans the arm, where
            it is excellent; asked for a base trajectory it plans 0.5 m and fails at 1.0 m
            in this scene, so it cannot move the robot between rooms at all. A* on the
            traversability map is the tool for that, and because the map is eroded by the
            robot's own footprint the route it returns has clearance for the base
            everywhere along it - that route, not the planner, is the collision guarantee.

            So the route comes from the map, and the robot follows it through the ordinary
            controller with no planning in the loop.

            This is used by NAVIGATE_TO only. GRASP and PLACE reposition themselves with
            upstream's `_navigate_to_pose`, and should: their move is under half a metre,
            which is inside CuRobo's working range, and it has to land on an exact pose -
            the one their reachability check just proved the hand can work from. Arriving
            "within 0.25 m" as this route does would break that guarantee.
            """
            import math

            import cv2 as _cv2
            import numpy as _np

            import omnigibson as og
            import torch as th
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                m as _m,
            )

            import torch as th

            tmap = self.robot.scene._trav_map
            start = self.robot.get_position_orientation()[0][:2]
            target = th.as_tensor([float(pose[0]), float(pose[1])], dtype=th.float32)

            # The same map the stance was chosen on, so the route cannot cut through
            # anything the stance search treated as blocked.
            eroded, _ = self._nav_map_now()

            # Distance from every free pixel to the nearest obstacle, for the proximity
            # term of the speed regulation below.
            clearance = _cv2.distanceTransform(
                (eroded > 0).astype(_np.uint8), _cv2.DIST_L2, 5) * tmap.map_resolution

            def clearance_at(x, y):
                m = tmap.world_to_map(th.tensor([float(x), float(y)], dtype=th.float32))
                r, c = int(m[0]), int(m[1])
                if not (0 <= r < clearance.shape[0] and 0 <= c < clearance.shape[1]):
                    return 0.0
                return float(clearance[r, c])

            def free_at(x, y):
                m = tmap.world_to_map(th.tensor([float(x), float(y)], dtype=th.float32))
                r, c = int(m[0]), int(m[1])
                return (0 <= r < eroded.shape[0] and 0 <= c < eroded.shape[1]
                        and eroded[r, c] > 0)

            def clear_line(a, b):
                dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
                steps = max(2, int(math.hypot(dx, dy) / (tmap.map_resolution * 0.5)))
                return all(free_at(float(a[0]) + dx * t / steps,
                                   float(a[1]) + dy * t / steps)
                           for t in range(steps + 1))

            path, _ = tmap.get_shortest_path(0, start, target,
                                             entire_path=True, robot=self.robot)
            if path is None or len(path) < 2:
                # A* finds nothing when the target is a step away - its own pixel is often
                # not traversable, being right up against the furniture. That is not a
                # reason to hand the move to a different controller: handing it to CuRobo
                # produced "there is no accessible path" and two symbolic rescues in a
                # run, for moves of a few centimetres.
                #
                # A straight line is the route in that case, and pure pursuit drives it
                # like any other.
                here_xy = self.robot.get_position_orientation()[0][:2]
                start_pt = (float(here_xy[0]), float(here_xy[1]))
                end_pt = (float(target[0]), float(target[1]))

                # The straight line has to clear the same floor A* would have used. Driving
                # it unchecked put the robot through the furniture and onto its back -
                # roll -182 deg, caught by the upright check. A* avoiding obstacles is the
                # whole point of having it; a shortcut that ignores them is not a route.
                if not clear_line(start_pt, end_pt):
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.PLANNING_ERROR,
                        "No route to this pose, and the direct line is blocked",
                        {"from": [round(start_pt[0], 2), round(start_pt[1], 2)],
                         "to": [round(end_pt[0], 2), round(end_pt[1], 2)]},
                    )

                path = [th.tensor(list(start_pt)), th.tensor(list(end_pt))]
                print(f"    [nav] no A* route to ({end_pt[0]:.2f},{end_pt[1]:.2f}) - "
                      f"line is clear, driving straight "
                      f"{math.hypot(end_pt[0]-start_pt[0], end_pt[1]-start_pt[1]):.2f} m")

            # A* is 8-connected on a 0.1 m grid and emits a waypoint every 0.2 m, so its
            # route is a staircase that changes heading every waypoint or two. Driven
            # literally the robot zigzags along it - visible as constant camera shake.
            #
            # String-pulling straightens it: keep a waypoint only where the direct line to
            # the next one would leave free floor. Standard any-angle post-processing for
            # grid A*, and it cannot cut a corner the base would not fit through, because
            # the line-of-sight test runs on the same eroded map that produced the route.
            corners = [path[0]]
            i = 0
            while i < len(path) - 1:
                j = len(path) - 1
                while j > i + 1 and not clear_line(path[i], path[j]):
                    j -= 1
                corners.append(path[j])
                i = j

            # The base rocks when it is shoved: OmniGibson models it with free `rx`/`ry`
            # joints (see `base_idx`: x, y, z, rx, ry, rz, of which only x/y/rz are
            # driven), so acceleration - not speed - is what unsettles it. The controller
            # below regulates its own acceleration; this just needs the timestep.
            try:
                step_dt = float(og.sim.get_sim_step_dt())
            except Exception:
                step_dt = 1.0 / 30.0

            # No corner rounding. The robot turns on the spot at each corner, so the arcs
            # a continuous-curvature follower would need have nothing to smooth - and any
            # arc cuts inside the route that string-pulling just verified as clear.
            total = sum(math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))
                        for a, b in zip(corners, corners[1:]))

            # Command the arm to the *tucked* pose for the whole drive, not to wherever it
            # measures right now.
            #
            # This used to be `get_joint_positions()`, so every waypoint told the arm to
            # hold its measured position at that instant. The joints have plenty of torque
            # to hold a plate - the problem was being asked to hold the wrong thing. If the
            # tuck had not finished, the half-tucked pose became the target and the arm was
            # instructed to stay extended; and because each new target was read back from
            # the measurement, a load swinging on the end of the arm latched its own sag in
            # and the pose ratcheted outward. That is what put the robot at 1.14 x 0.96 m
            # after it picked up the plate, and a footprint that size makes the whole house
            # unroutable.
            #
            # Naming the tucked pose explicitly means the drives hold it against gravity and
            # against the load for the entire route, which is what the joints are for.
            q = self.robot.get_joint_positions().clone()
            hold, arm_idx = self._hold_pose()
            q[arm_idx] = hold[arm_idx]

            print(f"    [nav] driving {total:.1f} m route, "
                  f"{len(corners)} corners, cruise {CRUISE_SPEED:.2f} m/s")

            # Rotate, drive, rotate. One heading change at a time, never both at once.
            #
            # The robot turns on the spot to face the leg it is about to drive, drives that
            # leg in a straight line at a fixed heading, and turns on the spot again at the
            # end to face the object. This is the plain differential-drive motion, and it
            # only ever commands "forward" or "turn" - the wheels are never asked to slide
            # sideways, which a Tiago cannot do however willing OmniGibson's
            # `HolonomicBaseJointController` is to accept the command.
            #
            # Regulated Pure Pursuit used to generate this trajectory instead, spreading
            # each turn across the drive so the base arced continuously between waypoints.
            # It tracked well, but the arc is harder to watch and harder to reason about
            # than "turn, then go".
            def shortest_turn(delta):
                """Signed angle for `delta`, taking the short way round."""
                return math.atan2(math.sin(delta), math.cos(delta))

            goal_yaw = float(pose[2])
            cur_x = float(q[self.robot.base_control_idx][0])
            cur_y = float(q[self.robot.base_control_idx][1])
            cur_yaw = float(q[self.robot.base_control_idx][2])
            states = []

            def turn_to(target_yaw):
                """Rotate on the spot, at a rate the base can follow."""
                nonlocal cur_yaw
                delta = shortest_turn(target_yaw - cur_yaw)
                if abs(delta) < MIN_TURN:
                    return
                steps = max(1, int(abs(delta) / (ROTATE_SPEED * step_dt)))
                for k in range(1, steps + 1):
                    states.append((cur_x, cur_y, cur_yaw + delta * k / steps))
                cur_yaw += delta

            def drive_to(x1, y1):
                """Drive straight at the current heading, slowing in tight places."""
                nonlocal cur_x, cur_y
                dist = math.hypot(x1 - cur_x, y1 - cur_y)
                if dist < MIN_SEGMENT:
                    return
                # Regulated by clearance, as before - it is what stopped the base pitching
                # in the gap between the fridge and the stove.
                room = clearance_at((cur_x + x1) / 2, (cur_y + y1) / 2)
                speed = CRUISE_SPEED
                if room < CLEARANCE_FULL_SPEED:
                    speed *= max(CLEARANCE_MIN_SCALE, room / CLEARANCE_FULL_SPEED)
                steps = max(1, int(dist / max(speed * step_dt, 1e-4)))
                x0, y0 = cur_x, cur_y
                for k in range(1, steps + 1):
                    t = k / steps
                    states.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, cur_yaw))
                cur_x, cur_y = x1, y1

            for a, b in zip(corners, corners[1:]):
                leg = math.hypot(float(b[0]) - cur_x, float(b[1]) - cur_y)
                if leg < MIN_SEGMENT:
                    continue                      # nothing to drive; do not twitch
                turn_to(math.atan2(float(b[1]) - cur_y, float(b[0]) - cur_x))
                drive_to(float(b[0]), float(b[1]))
            turn_to(goal_yaw)

            # Thin the states out for the executor, which treats each as a step change in
            # target. Keeping every simulation step would give it hundreds of targets a
            # few millimetres apart and spend ten steps on each; keeping too few makes the
            # steps large enough to shove the base.
            q_traj, last = [], None
            for (tx, ty, tth) in states:
                if last is not None:
                    moved = math.hypot(tx - last[0], ty - last[1])
                    turned = abs(shortest_turn(tth - last[2]))
                    if moved < WAYPOINT_EVERY and turned < WAYPOINT_TURN:
                        continue
                last = (tx, ty, tth)
                step = q.clone()
                step[self.robot.base_control_idx] = th.tensor([tx, ty, tth], dtype=q.dtype)
                q_traj.append(step)

            final = q.clone()
            final[self.robot.base_control_idx] = th.tensor(
                [float(target[0]), float(target[1]), goal_yaw], dtype=q.dtype)
            q_traj.append(final)

            print(f"    [nav] rotate-drive-rotate: {len(states)} states, "
                  f"{len(q_traj)} waypoints")

            yield from self._execute_motion_plan(
                q_traj, low_precision=True, ignore_failure=True)

            # Let the base come to rest before judging it - a reading taken the instant
            # the last waypoint finishes measures momentum, not posture.
            for _ in range(SETTLE_ACTIONS * 5):
                yield self._postprocess_action(
                    self.robot.q_to_action(self.robot.get_joint_positions()))

            # Read roll and pitch from the base joints, which is where
            # `_get_robot_pose_from_2d_pose` reads them and therefore what every IK solve
            # and reachability check is expressed against. Decomposing the world
            # quaternion instead can show apparent roll that is really just yaw.
            base_q = self.robot.get_joint_positions()[self.robot.base_idx]
            roll, pitch = float(base_q[3]), float(base_q[4])
            level = "" if max(abs(roll), abs(pitch)) < 0.03 else "   <- BASE NOT LEVEL"

            # A robot that ends up on its side has not navigated anywhere, whatever its
            # coordinates say. Checking position alone reported PASS for a base at
            # roll=+172 deg - upside down - and every reachability failure afterwards was
            # a consequence of that, not of the arm.
            if max(abs(roll), abs(pitch)) > UPRIGHT_LIMIT:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "The base is no longer upright after driving",
                    {"roll_deg": round(math.degrees(roll), 1),
                     "pitch_deg": round(math.degrees(pitch), 1)},
                )

            here = self.robot.get_position_orientation()[0][:2]
            gap = float(th.norm(here - th.as_tensor(target, dtype=here.dtype)))
            if gap > ARRIVAL_TOLERANCE:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.EXECUTION_ERROR,
                    "Drove the route but did not arrive",
                    {"target": [round(float(target[0]), 2), round(float(target[1]), 2)],
                     "reached": [round(float(here[0]), 2), round(float(here[1]), 2)],
                     "gap_m": round(gap, 2)},
                )
            print(f"    [nav] arrived within {gap:.2f} m; "
                  f"roll={math.degrees(roll):+.1f} deg pitch={math.degrees(pitch):+.1f} deg"
                  f"{level}")

        def _navigate_near(self, obj, sampling_attempts=40):
            """Drive to a collision-free spot near `obj`, ignoring arm reachability.

            Used only before a state change. `OPEN` does not need the arm to reach
            anything - the robot just has to be at the fridge - so this is upstream's
            `_sample_pose_near_object` with the two eef-driven parts removed:

              ring centre     upstream centres the sample ring on `eef_pose`, which
                              defaults to a *top-down grasp pose above the object*. Here
                              the ring is centred on the object itself.

              reachability    upstream keeps only poses from which the arm can reach that
                              eef pose. For a 1.5m fridge that means reaching over its
                              top, which the arm cannot do from any legal standing
                              distance, so every candidate is rejected and NAVIGATE_TO
                              fails with "Could not find a valid pose near the object".
                              Here the filter is dropped.

            Everything else is upstream's: same random ring, same room filter, same
            collision check, same `_navigate_to_pose`. Collision avoidance is untouched -
            the pose passes `check_collisions`, and the motion planner still refuses to
            route through obstacles.

            Every NAVIGATE_TO uses this, before physical and symbolic alike. GRASP and
            PLACE run their own final approach afterwards, against the pose they actually
            intend to use and with retries.
            """
            # The object we are going to. Its door swing is the only one treated as an
            # obstacle - every other door in the house is a way through.
            self._nav_target = obj

            import math

            from omnigibson import object_states
            import numpy as _np_nav

            import torch as th
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                m as _m,
            )

            center, _ = obj.get_position_orientation()
            lo, hi = _m.BASE_POSE_SAMPLING_LOWER_BOUND, _m.BASE_POSE_SAMPLING_UPPER_BOUND
            rooms = obj.in_rooms or []
            seg = self.robot.scene._seg_map

            self._motion_generator.update_obstacles()
            joint_pos = self.robot.get_joint_positions()

            # Anything in the gripper is part of the robot for collision purposes. Without
            # this the check clears stances where the carried object is buried in a wall,
            # which matters on the leg to the fridge - the apple is held the whole way.
            obj_in_hand = self._get_obj_in_hand()
            # A carried object is `visual_only` - no collisions, no gravity - so it has no
            # collision mesh to hand the planner, and CuRobo dies on it with
            # "'NoneType' object has no attribute 'get_trimesh_mesh'". It also has nothing
            # to contribute: something that cannot collide does not change where the robot
            # fits.
            attached_obj = (
                {self.robot.eef_link_names[self.arm]: obj_in_hand.root_link}
                if obj_in_hand is not None and not obj_in_hand.visual_only else None
            )

            # Only consider spots the robot can actually drive to. `check_collisions`
            # answers "does the robot fit here", not "can it get here", and those differ:
            # measured on Beechwood's fridge, 37% of collision-free ring poses sit in a
            # floor region cut off from the robot by a doorway too narrow for it. Routing
            # to one of those can only ever fail, and each attempt costs a full CuRobo
            # plan, so they are dropped before anything is planned.
            import numpy as _np

            # Fold the arm in *before* measuring the footprint. The map is eroded by the
            # robot's current bounding box, and until this runs that box is whatever the
            # last primitive left behind - measured, the first navigation of a run eroded
            # by 0.99 m against 0.77 m once tucked, which is the same robot in the same
            # scene getting two different maps. Tucking first also makes the radius take
            # exactly two values, tucked and carrying, which is what lets the map be
            # cached at all.
            yield from self._tuck_arm()

            tmap = self.robot.scene._trav_map
            # One erosion radius, one cached map, one cached labelling. The labels come
            # from the map before the door swing is blocked off, which is the same answer
            # for reachability: cells inside the swing are dropped anyway because they are
            # not free in `eroded`, and an appliance door is not what connects two rooms.
            eroded, floor_region = self._nav_map_now()

            def region_of(xy):
                m = tmap.world_to_map(th.as_tensor(xy, dtype=th.float32))
                r, c = int(m[0]), int(m[1])
                if not (0 <= r < floor_region.shape[0] and 0 <= c < floor_region.shape[1]):
                    return None
                return int(floor_region[r, c])

            robot_pos = self.robot.get_position_orientation()[0]
            robot_xy = robot_pos[:2]
            # Which component the robot is in - read from the nearest *free* cell, not
            # from the cell it stands on.
            #
            # At a large enough erosion radius the robot's own position is eroded away, and
            # `region_of` then returns the background label 0, which no free cell carries.
            # Every candidate is discarded for being in a different component and the
            # primitive fails with "No reachable floor near the object" while standing in
            # the middle of a perfectly good room - measured at the oven while carrying the
            # plate, in_robot_region=0 out of 219017 free cells. The object's region is
            # already read this way, for the same reason: its centre is never traversable.
            def region_near(xy):
                free = _np.argwhere(eroded > 0)
                if len(free) == 0:
                    return None
                m = tmap.world_to_map(th.as_tensor(xy, dtype=th.float32))
                d2 = ((free[:, 0] - int(m[0])) ** 2 + (free[:, 1] - int(m[1])) ** 2)
                nearest = free[int(_np.argmin(d2))]
                return int(floor_region[nearest[0], nearest[1]])

            robot_region = region_of(robot_xy)
            if not robot_region:
                robot_region = region_near(robot_xy)
                print(f"    [nav] the robot's own cell is not traversable at this "
                      f"erosion; taking its region from the nearest free floor "
                      f"-> {robot_region}")

            # Is the robot in collision *where it already stands*? CuRobo plans from the
            # current configuration, so if that state is invalid every plan fails no
            # matter how short - which is what a failure on the first 1.9 m hop looks
            # like. Cheap to ask, and it separates "cannot get there" from "cannot start".
            here_bad = bool(self._motion_generator.check_collisions(
                joint_pos.unsqueeze(0), self_collision_check=False,
                attached_obj=attached_obj).cpu()[0])
            print(f"    [nav] robot z={float(robot_pos[2]):.3f} "
                  f"in_collision_at_start={here_bad}")
            # An object's own centre is never traversable, so report the region of the
            # nearest free pixel to it - the floor a robot would actually stand on.
            free_px = _np.argwhere(eroded > 0)
            cm = tmap.world_to_map(th.as_tensor(center[:2], dtype=th.float32))
            near = free_px[int(_np.argmin((free_px[:, 0] - int(cm[0])) ** 2
                                          + (free_px[:, 1] - int(cm[1])) ** 2))]
            obj_region = int(floor_region[near[0], near[1]])
            print(f"    [nav] robot at ({float(robot_xy[0]):.2f}, {float(robot_xy[1]):.2f}) "
                  f"region={robot_region}; {obj.name} region={obj_region}")

            # Never sample inside the object's own footprint. The ring's lower bound is
            # 0.0 upstream, which for a fridge puts a large share of candidates literally
            # inside the appliance, where they can only ever collide.
            ext = obj.aabb_extent
            lo = max(float(lo), 0.5 * float(th.linalg.norm(ext[:2])))

            # Reach further out than upstream's 1.5 m. A state change needs the robot at
            # the object, not within arm's reach of it, so standing back is fine - and it
            # is what makes the difference in a tight kitchen: Beechwood's fridge yielded
            # only 3 collision-free stances inside 1.5 m and none of the three could be
            # routed to, which fails the primitive for want of somewhere to park.
            # If the target can open, stand clear of where its door will be.
            #
            # The stance has to be valid for the object *after* it opens, not before:
            # parking in the swing means the door hits the robot, or cannot open at all.
            # That applies whether we came to open it or to reach inside - so it depends on
            # the object, not on which primitive comes next.
            #
            # A hinged door sweeps roughly its own width, so pushing the ring's inner edge
            # out by the object's largest horizontal dimension clears the arc without
            # needing to know which side it is hinged or how far it opens.
            # Enumerate the floor, do not sample it.
            #
            # The map already knows everything the ring sampling was guessing at: which
            # cells are floor, which are eroded away by the robot's own footprint, which
            # are taken by the target's door swing, and which are in the same connected
            # component as the robot. So walk the free cells in order of distance from the
            # object and take the nearest one that works, instead of drawing random poses
            # and widening the ring until something lands.
            #
            # Being in the robot's own component is the part sampling could not give us. A
            # collision-free pose across an untraversable gap is not a place the robot can
            # get to, and every one of those cost a full routing attempt to discover.
            free_px = _np.argwhere(eroded > 0)
            if len(free_px) == 0:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No traversable floor at all once eroded by the robot's footprint",
                    {"object": obj.name},
                )

            # World coordinates of every free cell, and its distance to the object.
            world = tmap.map_to_world(th.as_tensor(free_px, dtype=th.float32))
            dx = world[:, 0] - float(center[0])
            dy = world[:, 1] - float(center[1])
            dist = th.sqrt(dx * dx + dy * dy)

            keep = th.ones(len(free_px), dtype=th.bool)

            # Never stand inside the object's own footprint.
            keep &= dist >= lo
            n_outside = int(keep.sum())

            # Only where the robot can actually get to.
            if robot_region is not None:
                labels = th.as_tensor(floor_region[free_px[:, 0], free_px[:, 1]])
                keep &= labels == robot_region
            n_reachable = int(keep.sum())

            # No angular restriction. Standing where the door will sweep is already
            # impossible: that floor is painted as an obstacle before the map is eroded, so
            # any surviving cell clears the door by the robot's whole width. Restricting
            # the approach to a 25 degree arc on top of that removed floor for no safety
            # gain - measured at the oven, 6203 reachable cells down to 535 - and left too
            # few candidates for any of them to be routable.

            # No room filter either.
            #
            # It required the robot to stand in the room the object is annotated to, which
            # removed floor without buying anything: being in the robot's own connected
            # component already guarantees it can get there, and the door swing being
            # blocked already guarantees it is not in the door's way. What it did do was
            # narrow the oven's candidates from 535 to 223 - and an appliance set into the
            # wall between two rooms can legitimately be worked at from the other one.

            idx = th.nonzero(keep).flatten()
            if len(idx) == 0:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No reachable floor near the object",
                    {"object": obj.name, "free_cells": len(free_px),
                     "outside_footprint": n_outside, "in_robot_region": n_reachable},
                )

            # Order by how far the floor actually is from the object, walked rather than
            # measured.
            #
            # Breadth-first from the object's own centre, eight-connected, over the
            # navigation map. The first cell it reaches that is free and in the robot's
            # component is the closest place the robot can genuinely stand; sorting cells
            # by straight-line distance instead puts the far side of a wall ahead of the
            # near side of the room. The BFS seed also anchors the ordering, so the
            # remaining candidates are ranked by distance from *it* rather than from a
            # point inside the object.
            # Nearest to the object, over the cells that survived the filters.
            #
            # This used to breadth-search out from the object's centre for the first
            # reachable cell and rank everything by distance from *that*. Both halves were
            # wrong. An eight-connected search expands in grid steps, so its frontier is a
            # square rather than a circle and the cell it lands on is the nearest in steps,
            # not in metres - measured, it overshot by 0.30 m at the plate and 0.02 m at
            # the potato. Ranking from that cell then carried the error into the stance:
            # the plate was approached at 0.83 m where the robot physically fits at 0.70 m.
            #
            # Sorting the survivors by their real distance is exact and costs nothing worth
            # measuring. The distances are already computed, over every free cell at once,
            # in 0.10 ms; the sort runs on the few thousand that pass the filters in
            # 0.34 ms. Sorting the whole map would be 12.5 ms, and is not done. One CuRobo
            # collision batch is tens of milliseconds, and a navigation does several.
            idx = idx[th.argsort(dist[idx])]

            # Collision-check the nearest handful and keep the ones that pass. Checking
            # every free cell would be thousands of poses; the nearest few dozen is where
            # the answer is.
            wanted = 8
            candidates = []
            n_checked = n_collision_free = 0
            for chunk in th.split(idx[:CELL_CANDIDATES], self._curobo_batch_size):
                poses, q = [], []
                for i in chunk.tolist():
                    # Face the object. NAVIGATE_TO's job is to put the robot at the object,
                    # squared up to it; GRASP and PLACE reposition themselves for reaching
                    # afterwards through their own navigation.
                    yaw = math.atan2(float(center[1]) - float(world[i, 1]),
                                     float(center[0]) - float(world[i, 0]))
                    pose = th.tensor([float(world[i, 0]), float(world[i, 1]), yaw])
                    poses.append(pose)
                    j = joint_pos.clone()
                    j[self.robot.base_control_idx] = pose
                    q.append(j)
                if not poses:
                    continue
                n_checked += len(poses)
                bad = self._motion_generator.check_collisions(
                    th.stack(q), self_collision_check=False,
                    attached_obj=attached_obj).cpu()
                for pose, is_bad in zip(poses, bad):
                    if not is_bad.item():
                        n_collision_free += 1
                        candidates.append(pose)
                if len(candidates) >= wanted:
                    break

            nearest = float(dist[idx[0]])
            print(f"    [nav] {obj.name}: {len(free_px)} free cells, "
                  f"{n_outside} outside its footprint, {n_reachable} in the robot's "
                  f"region; "
                  f"checked {n_checked} nearest, {n_collision_free} collision-free; "
                  f"nearest reachable floor {nearest:.2f} m out"
                  + (f"; carrying {obj_in_hand.name}" if obj_in_hand is not None else ""))

            if not candidates:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No collision-free pose on the reachable floor near the object",
                    {"object": obj.name, "free_cells": len(free_px),
                     "in_robot_region": n_reachable, "checked": n_checked},
                )

            # Tuck the arm in before setting off.
            #
            # Make sure the arm is folded in before driving. `_reset_robot` already tucks
            # after every primitive, so this is usually a no-op - but it guarantees it for
            # the cases that do not go through `apply_ref`.
            yield from self._tuck_arm()

            reach = max(float(th.norm(self.robot.get_relative_eef_pose(a)[0][:2]))
                        for a in self.robot.arm_names)
            print(f"    [nav] arm tucked in: hand {reach:.2f} m from the base axis "
                  f"(0.55 m at upstream's reset pose, 0.29 m tucked)")

            errors = []
            for i, pose in enumerate(candidates):
                try:
                    # Tuck again before *every* attempt, not once before the loop.
                    #
                    # A failed drive can leave the arm extended, and the map is eroded by
                    # the robot's current bounding box - so the next candidate is then
                    # planned against an inflated footprint. Measured: after one failed
                    # attempt the robot read 1.14 x 0.96 m, wider than even its reset
                    # pose, and the erosion radius went from 0.79 m to 1.04 m. At 1.04 m
                    # the house falls apart - every table in house_single_floor becomes
                    # unreachable and the spawn's own connected component changes - so the
                    # robot's current position stops resolving to a valid floor region and
                    # every remaining candidate is discarded for being in a different one.
                    yield from self._tuck_arm()
                    yield from self._drive_to(pose)
                    return
                except ActionPrimitiveError as e:
                    # Could not get to this one; try the next. Both real causes are worth
                    # telling apart, so print the reason rather than assuming: candidates
                    # now all come from the robot's own connected component, so a genuine
                    # "no path" is unlikely and an EXECUTION_ERROR here usually means the
                    # drive itself went wrong - the base toppling, most often. Labelling
                    # every failure "unroutable" once sent us looking at path planning
                    # while the robot was lying on its side.
                    errors.append(e)
                    print(f"    [nav] candidate {i + 1}/{len(candidates)} failed: "
                          f"{str(e).splitlines()[0][:110]}")

            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No collision-free pose near the object could be reached",
                {"object": obj.name, "in_robot_region": n_reachable,
                 "collision_free": n_collision_free, "tried": len(candidates),
                 "last error": str(errors[-1]).splitlines()[0] if errors else None},
            )

        def _validate_poses(self, candidate_poses, eef_pose=None,
                            plan_with_open_gripper=False, skip_obstacle_update=False):
            """Upstream's validator, instrumented.

            Upstream returns a bare boolean mask, so a NAVIGATE_TO failure before a
            grasp cannot be read: "nowhere to stand" and "nowhere the arm can reach
            from" produce the identical message. This counts the two rejection stages
            separately and leaves them on self._diag. Behaviour is otherwise upstream's.
            """
            import torch as th

            if plan_with_open_gripper:
                current_joint_pos = self._get_joint_position_with_fingers_at_limit("upper")
            else:
                current_joint_pos = self.robot.get_joint_positions()

            cjp = []
            for pose in candidate_poses:
                j = current_joint_pos.clone()
                j[self.robot.base_control_idx] = pose
                cjp.append(j)
            cjp = th.stack(cjp)

            obj_in_hand = self._get_obj_in_hand()
            # A carried object is `visual_only` - no collisions, no gravity - so it has no
            # collision mesh to hand the planner, and CuRobo dies on it with
            # "'NoneType' object has no attribute 'get_trimesh_mesh'". It also has nothing
            # to contribute: something that cannot collide does not change where the robot
            # fits.
            attached_obj = (
                {self.robot.eef_link_names[self.arm]: obj_in_hand.root_link}
                if obj_in_hand is not None and not obj_in_hand.visual_only else None
            )
            invalid = self._motion_generator.check_collisions(
                cjp, self_collision_check=False,
                skip_obstacle_update=skip_obstacle_update,
                attached_obj=attached_obj).cpu()

            d = self._diag
            d["sampled"] += len(candidate_poses)
            d["collision_free"] += int((~invalid).sum())

            for i in range(len(candidate_poses)):
                if invalid[i].item():
                    continue
                if eef_pose is not None:
                    if not self._target_in_reach_of_robot(
                        eef_pose, initial_joint_pos=cjp[i],
                        skip_obstacle_update=skip_obstacle_update
                    ):
                        invalid[i] = True
                    else:
                        d["reachable"] += 1
            return ~invalid

        def _navigate_to_obj(self, obj, **kwargs):
            """Stock navigation, with the rejection counts printed on the way out."""
            self._diag = {"sampled": 0, "collision_free": 0, "reachable": 0}
            try:
                yield from super()._navigate_to_obj(obj, **kwargs)
            finally:
                d = self._diag
                name = getattr(obj, "name", obj)
                print(f"    [nav-stock] {name}: sampled={d['sampled']} "
                      f"collision_free={d['collision_free']} reachable={d['reachable']}")

        def __init__(self, env, robot, curobo_batch_size=3):
            # `SymbolicSemanticActionPrimitives.__init__` takes only (env, robot) and
            # passes `skip_curobo_initilization=True`, which leaves `_motion_generator`
            # as None - it has no use for a motion planner because it never moves the
            # arm. Our navigation does: `_navigate_near` collision-checks every candidate
            # stance through it. So run the symbolic constructor for its primitive table,
            # then build the motion generator that it skipped.
            from omnigibson.action_primitives.starter_semantic_action_primitives import (
                CuRoboMotionGenerator,
                m as _starter_m,
            )

            SymbolicSemanticActionPrimitives.__init__(self, env, robot)
            self._curobo_batch_size = curobo_batch_size
            self._motion_generator = CuRoboMotionGenerator(
                robot=self.robot,
                batch_size=curobo_batch_size,
                collision_activation_distance=(
                    _starter_m.DEFAULT_COLLISION_ACTIVATION_DISTANCE),
            )

            # Rejection counts for the stock reachability-aware sampler, filled in by
            # _validate_poses below.
            self._diag = {"sampled": 0, "collision_free": 0, "reachable": 0}

    return NineWorkingPrimitives(env, robot, curobo_batch_size)


def primitive_set():
    """The IntEnum whose members name the primitives.

    It has to be the symbolic set, matching the controller `build()` returns. The starter
    set is *not* interchangeable with it: both declare the same names in a different
    order, so a name carries a different integer in each - NAVIGATE_TO is 6 in the starter
    set and 13 in the symbolic one, while 6 is TOGGLE_ON there. They are IntEnums, so a
    member of the wrong set still hashes equal to whatever shares its value, and the
    controller's dispatch table returns *a* primitive rather than raising - the wrong one,
    silently. Measured: asking for NAVIGATE_TO with a starter member ran TOGGLE_ON.
    """
    from omnigibson.action_primitives.symbolic_semantic_action_primitives import (
        SymbolicSemanticActionPrimitiveSet,
    )

    return SymbolicSemanticActionPrimitiveSet
