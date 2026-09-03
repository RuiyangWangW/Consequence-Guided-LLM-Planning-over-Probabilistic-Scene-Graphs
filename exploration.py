"""Active search: find an object the RSN only guessed the location of.

The plan says NAVIGATE_TO(stove), but nothing in the pipeline knows where the stove is -
the RSN said "probably the kitchen, 0.66", which is a belief, not a position. Executing
that plan directly is only possible because the executor cheats: it looks the object up
in the scene registry, which is information the robot could not have.

This module removes the cheat. The robot goes to the *room* the RSN considers most
likely, looks around, and only navigates to the object once it has actually seen it:

    1. navigate to the most probable room for the object
    2. observe - read seg_instance from the robot camera, record every object seen and
       mark the observed area on an occupancy map
    3. if the target was seen, navigate to it and we are done
    4. otherwise pick the nearest frontier - the boundary between explored and unexplored
       free space inside this room - move there, and observe again
    5. when the room is fully explored and the target has not appeared, report failure
       with the room ruled out

Step 5 is what feeds replanning: the caller drops that room type from the object's
distribution, takes the RSN's next most likely room, rebuilds the graph and asks the LLM
for a new plan. That loop - belief, search, disconfirmation, revised belief - is the part
this project exists to study, so it lives in the caller rather than being hidden here.
"""

import math

# Occupancy grid values.
UNKNOWN, FREE, OCCUPIED = 0, 1, 2


class LocalMap:
    """What the robot has seen: an occupancy grid plus the objects observed in it.

    Deliberately built only from the robot's own observations, never from the scene
    registry. It starts empty and fills in as the robot looks around, which is what makes
    "explored the room and did not find it" a meaningful statement.
    """

    def __init__(self, bounds, resolution=0.1):
        import torch as th

        (self.x0, self.y0), (self.x1, self.y1) = bounds
        self.resolution = resolution
        self.w = max(1, int((self.x1 - self.x0) / resolution))
        self.h = max(1, int((self.y1 - self.y0) / resolution))
        self.grid = th.zeros((self.h, self.w), dtype=th.uint8)
        self.objects = {}  # name -> {"position": [x, y, z], "category": str}

    def to_cell(self, x, y):
        return (int((y - self.y0) / self.resolution),
                int((x - self.x0) / self.resolution))

    def to_world(self, row, col):
        return (self.x0 + (col + 0.5) * self.resolution,
                self.y0 + (row + 0.5) * self.resolution)

    def in_bounds(self, row, col):
        return 0 <= row < self.h and 0 <= col < self.w

    def mark_observed_fov(self, x, y, yaw, h_fov, max_range, depth_fn=None,
                          traversable_fn=None, n_rays=61):
        """Mark the camera's actual field of view as observed, stopping at obstacles.

        Casts `n_rays` across the horizontal FoV and walks each one outward, marking cells
        FREE until it reaches either `max_range` or the depth reading for that bearing -
        the first surface along the ray. The cell at the depth hit is marked OCCUPIED and
        the ray stops, so space *behind* an obstacle stays UNKNOWN.

        That occlusion is the point. A disc around the robot would claim the far side of
        a counter as explored without ever seeing it, and the room would be declared
        "fully explored, object not here" on evidence the robot never gathered - exactly
        the false negative that would send replanning down the wrong branch.
        """
        import math

        for i in range(n_rays):
            # Bearing of this ray, spread evenly across the horizontal FoV.
            frac = (i / (n_rays - 1)) - 0.5 if n_rays > 1 else 0.0
            angle = yaw + frac * h_fov
            reach = max_range if depth_fn is None else min(max_range, depth_fn(frac))

            steps = max(1, int(reach / self.resolution))
            hit = False
            for s in range(steps + 1):
                d = s * self.resolution
                row, col = self.to_cell(x + d * math.cos(angle), y + d * math.sin(angle))
                if not self.in_bounds(row, col):
                    break
                # An obstacle ends the ray: it is the surface we can see, and everything
                # past it is hidden.
                if traversable_fn is not None and d > 0 and not traversable_fn(
                        x + d * math.cos(angle), y + d * math.sin(angle)):
                    if self.grid[row, col] == UNKNOWN:
                        self.grid[row, col] = OCCUPIED
                    hit = True
                    break
                if self.grid[row, col] == UNKNOWN:
                    self.grid[row, col] = FREE
            if not hit and depth_fn is not None and reach < max_range:
                # The depth reading itself is a surface, even if the traversability map
                # does not know about it (an object sitting on the floor, say).
                row, col = self.to_cell(x + reach * math.cos(angle),
                                        y + reach * math.sin(angle))
                if self.in_bounds(row, col) and self.grid[row, col] == UNKNOWN:
                    self.grid[row, col] = OCCUPIED

    def record_object(self, name, category, position):
        self.objects[name] = {"category": category, "position": list(position)}

    def coverage(self):
        """Fraction of cells that are no longer unknown."""
        import torch as th

        return float((self.grid != UNKNOWN).sum()) / float(self.grid.numel())

    def frontiers(self):
        """Free cells adjacent to unknown space - the boundary worth walking to.

        Standard frontier-based exploration: anywhere the known-free region touches the
        unknown is somewhere new information can be gained by standing there.
        """
        out = []
        for row in range(self.h):
            for col in range(self.w):
                if self.grid[row, col] != FREE:
                    continue
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r2, c2 = row + dr, col + dc
                    if self.in_bounds(r2, c2) and self.grid[r2, c2] == UNKNOWN:
                        out.append(self.to_world(row, col))
                        break
        return out

    def nearest_frontier(self, x, y, min_distance=0.5, exclude=()):
        """Closest frontier at least `min_distance` away, so the robot actually moves.

        `exclude` holds grid cells already found unreachable, so a frontier the motion
        planner cannot get to is not selected again on the next iteration.
        """
        best, best_d = None, float("inf")
        for fx, fy in self.frontiers():
            if self.to_cell(fx, fy) in exclude:
                continue
            d = math.hypot(fx - x, fy - y)
            if d < min_distance or d >= best_d:
                continue
            best, best_d = (fx, fy), d
        return best


def _robot_camera(robot):
    """The robot's head/main vision sensor, or None."""
    from omnigibson.sensors.vision_sensor import VisionSensor

    cams = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    if not cams:
        return None
    # Prefer a head-mounted camera when there is one; otherwise take the first.
    for cam in cams:
        if "head" in cam.name.lower() or "eyes" in cam.name.lower():
            return cam
    return cams[0]


def camera_fov(cam, default=1.1064):
    """Horizontal field of view in radians, from the camera's own intrinsics.

        h_fov = 2 * atan(horizontal_aperture / (2 * focal_length))

    The default is what OmniGibson's own VisionSensor defaults produce - 1.1064 rad, 63.4
    deg - rather than a round number. It used to be 1.2, which is 5.4 deg wider than any
    camera the robot actually has, so a run that fell back to it explored faster than the
    real robot could.
    """
    import math

    try:
        aperture = float(cam.horizontal_aperture)
        focal = float(cam.focal_length)
        if focal > 0:
            return 2.0 * math.atan(aperture / (2.0 * focal))
    except Exception:
        pass
    return default


def observe(env, robot, local_map, max_range=5.0):
    """Read the robot camera, record what is visible, and mark the seen area.

    Two things come out of one observation:

      objects   from `seg_instance`, which names the object each pixel belongs to, so
                only what is genuinely in frame is recorded
      coverage  from `depth_linear` cast across the camera's real field of view, so the
                explored region is the shape the camera actually sees and stops at the
                first surface along each bearing

    Positions still come from the scene registry once an object has been seen, standing
    in for the depth-based localization a real system would do. The visibility gate is
    what matters for exploration; the position is a convenience.
    """
    import torch as th

    obs, info = robot.get_obs()

    # --- what is in frame ---
    seen = set()
    for key, value in obs.items():
        if not key.endswith("seg_instance"):
            continue
        mapping = {}
        for info_key, info_val in (info or {}).items():
            if info_key.endswith("seg_instance") and isinstance(info_val, dict):
                mapping = info_val
                break
        for ident in th.unique(value).tolist():
            name = mapping.get(ident, mapping.get(str(ident)))
            if isinstance(name, str):
                seen.add(name)

    scene = env.scene
    for name in seen:
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        pos, _ = obj.get_position_orientation()
        local_map.record_object(name, obj.category, pos.tolist())

    # --- how much of the room that observation covered ---
    pos, orn = robot.get_position_orientation()
    x, y = float(pos[0]), float(pos[1])

    import omnigibson.utils.transform_utils as T

    yaw = float(T.quat2euler(orn)[2])

    cam = _robot_camera(robot)
    h_fov = camera_fov(cam) if cam is not None else 1.2

    # Depth along each bearing, so rays stop at the first surface instead of sweeping
    # through walls. `frac` runs -0.5..0.5 across the FoV; map it to an image column and
    # take the minimum over that column, which is the nearest surface in that direction.
    depth_fn = None
    depth = next((v for k, v in obs.items() if k.endswith("depth_linear")), None)
    if depth is not None and getattr(depth, "ndim", 0) == 2:
        width = depth.shape[1]

        def depth_fn(frac, _d=depth, _w=width):
            col = int(round((frac + 0.5) * (_w - 1)))
            col = max(0, min(_w - 1, col))
            column = _d[:, col]
            column = column[th.isfinite(column) & (column > 0.05)]
            return float(column.min()) if column.numel() else max_range

    trav = getattr(scene, "trav_map", None)

    def traversable(wx, wy):
        if trav is None:
            return True
        try:
            return bool(trav.has_node(0, th.tensor([wx, wy])))
        except Exception:
            return True

    local_map.mark_observed_fov(x, y, yaw, h_fov, max_range,
                                depth_fn=depth_fn, traversable_fn=traversable)
    return seen


def room_bounds(scene, room_instance, pad=0.5):
    """World-frame (min, max) xy box for a room, from the segmentation map."""
    import torch as th

    seg = scene._seg_map
    ins_id = seg.room_ins_name_to_ins_id.get(room_instance)
    if ins_id is None:
        return None
    rows, cols = th.nonzero(th.tensor(seg.room_ins_map == ins_id), as_tuple=True)
    if rows.numel() == 0:
        return None
    corners = [seg.map_to_world(th.tensor([float(r), float(c)]))
               for r, c in ((rows.min(), cols.min()), (rows.max(), cols.max()))]
    xs = sorted(float(c[0]) for c in corners)
    ys = sorted(float(c[1]) for c in corners)
    return ((xs[0] - pad, ys[0] - pad), (xs[1] + pad, ys[1] + pad))


def search_room(env, robot, controller, room_instance, target_category,
                max_steps=8, max_range=5.0, step_cb=None, verbose=True):
    """Look for `target_category` inside one room.

    Returns (object or None, result) where result is:

        {"map": LocalMap, "coverage": float, "status": str, "unreachable": int}

    `status` distinguishes the three ways this ends, and the difference matters to the
    caller:

        "found"      the object was seen
        "exhausted"  no frontier left - the room really was searched, so it can be ruled
                     out and the RSN's belief revised against it
        "gave_up"    hit max_steps with frontiers remaining, or could not reach the ones
                     left. NOT the same as exhausted: the object may still be here, and
                     eliminating the room on this basis would be a false negative

    Turning "gave_up" into "ruled out" is the failure mode worth guarding, because it
    sends replanning down a branch the evidence does not support.
    """
    bounds = room_bounds(env.scene, room_instance)
    if bounds is None:
        if verbose:
            print(f"    no bounds for {room_instance}; cannot search it")
        return None, {"map": None, "coverage": 0.0, "status": "no_bounds",
                      "unreachable": 0}

    local_map = LocalMap(bounds)
    unreachable = 0

    def look():
        observe(env, robot, local_map, max_range)
        for name, rec in local_map.objects.items():
            if rec["category"] == target_category:
                return env.scene.object_registry("name", name)
        return None

    def done(found, status):
        return found, {"map": local_map, "coverage": local_map.coverage(),
                       "status": status, "unreachable": unreachable}

    found = look()
    if found is not None:
        if verbose:
            print(f"    saw {found.name} on arrival")
        return done(found, "found")

    # Frontiers already tried and found unreachable, so the loop does not pick the same
    # one forever. Kept as world coordinates rounded to the grid.
    blocked = set()

    for step in range(max_steps):
        x, y = robot.get_position_orientation()[0][:2].tolist()
        frontier = local_map.nearest_frontier(x, y, exclude=blocked)
        if frontier is None:
            if verbose:
                print(f"    {room_instance} exhausted at {local_map.coverage():.0%} "
                      f"coverage; {target_category} is not here")
            # Frontiers can run out because everything left was unreachable rather than
            # because the room was covered. That is a give-up, not a clean sweep.
            return done(None, "exhausted" if unreachable == 0 else "gave_up")

        if verbose:
            print(f"    frontier {step + 1}: ({frontier[0]:.1f}, {frontier[1]:.1f}), "
                  f"coverage {local_map.coverage():.0%}")
        try:
            # `_navigate_to_pose` takes a 2d pose (x, y, yaw); face the frontier so the
            # camera looks at the unexplored side when the robot arrives.
            import math

            yaw = math.atan2(frontier[1] - y, frontier[0] - x)
            for action in controller._navigate_to_pose((frontier[0], frontier[1], yaw)):
                env.step(action)
                if step_cb is not None:
                    step_cb()
        except Exception as e:
            # Do NOT mark this area observed. Claiming coverage the robot never earned
            # is what turns an unreachable corner into a false "object is not here".
            unreachable += 1
            blocked.add(local_map.to_cell(*frontier))
            if verbose:
                print(f"    frontier unreachable ({type(e).__name__}); skipping it")
            continue

        found = look()
        if found is not None:
            if verbose:
                print(f"    found {found.name} after {step + 1} moves")
            return done(found, "found")

    if verbose:
        print(f"    gave up after {max_steps} moves at {local_map.coverage():.0%} "
              f"coverage - {target_category} may still be in {room_instance}")
    return done(None, "gave_up")


def rsn_room_ranking(object_name, graph, model_path="models/rsn_cal.pt", device=None):
    """Rooms in this scene ranked by P(object | room type), most likely first.

    Returns [(room_instance, room_type, probability), ...]. The caller walks this list:
    search the top room, and if the object is not there, drop to the next. That descent
    through the ranking is the belief revision - each failed search is evidence against
    one room type, and the prior supplies the next hypothesis.
    """
    import torch

    from query_rsn import load, predict_rooms

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load(model_path, device)
    room_types = ckpt["room_types"]
    probs = predict_rooms(model, ckpt, object_name, device)

    # Largest instance of each type, matching how scene_graph.populate picks one.
    best = {}
    for rid, info in graph["rooms"].items():
        t = info["room_type"]
        if t not in best or info["pixels"] > graph["rooms"][best[t]]["pixels"]:
            best[t] = rid

    ranked = [(best[t], t, float(probs[room_types.index(t)]))
              for t in best if t in room_types]
    return sorted(ranked, key=lambda r: -r[2])


def find_object(env, robot, controller, object_name, graph, max_rooms=3,
                model_path="models/rsn_cal.pt", step_cb=None, verbose=True):
    """Search for an object the RSN only has a belief about.

    Walks the RSN's room ranking: go to the most likely room, explore it, and if the
    object is not there, drop to the next. Returns

        (object or None, {
            "searched":    [room_instance, ...],   every room entered
            "ruled_out":   [room_instance, ...],   searched to exhaustion, object absent
            "inconclusive":[room_instance, ...],   gave up early - NOT evidence of absence
            "ranking":     [(room, type, p), ...], the RSN prior that drove the order
            "rooms":       {room: result},
        })

    Only `ruled_out` is evidence. A caller revising the scene graph should remove those
    rooms from the object's distribution and leave `inconclusive` ones alone - treating a
    partial search as a negative result would eliminate the right answer and send the
    replan down a branch nothing supports.

    The replanning itself is left to the caller: rebuild the graph without the ruled-out
    rooms and ask the LLM again. That loop is the research question, not plumbing to be
    buried here.
    """
    ranking = rsn_room_ranking(object_name, graph, model_path)
    searched, ruled_out, inconclusive, rooms = [], [], [], {}

    for room_instance, room_type, p in ranking[:max_rooms]:
        if verbose:
            print(f"  searching {room_instance} for {object_name} "
                  f"(RSN: {p:.2f} for {room_type})")
        searched.append(room_instance)

        found, result = search_room(env, robot, controller, room_instance,
                                    object_name, step_cb=step_cb, verbose=verbose)
        rooms[room_instance] = result

        if found is not None:
            return found, {"searched": searched, "ruled_out": ruled_out,
                           "inconclusive": inconclusive, "ranking": ranking,
                           "rooms": rooms}

        if result["status"] == "exhausted":
            ruled_out.append(room_instance)
            if verbose:
                print(f"  ruled out {room_instance}")
        else:
            inconclusive.append(room_instance)
            if verbose:
                print(f"  {room_instance} inconclusive ({result['status']}, "
                      f"{result['coverage']:.0%} covered) - not counted as evidence")

    return None, {"searched": searched, "ruled_out": ruled_out,
                  "inconclusive": inconclusive, "ranking": ranking, "rooms": rooms}
