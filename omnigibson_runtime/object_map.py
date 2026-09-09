"""The object semantic map: what the robot has actually looked at, and what it saw there.

Starts entirely UNKNOWN and fills in from the robot's own camera. Two things come out of
one observation, and they answer different questions:

    coverage   which cells the camera has actually seen, cast across its real field of
               view and stopped at the first surface along each bearing. This is what
               makes "I searched the kitchen" a claim with evidence behind it.
    objects    which objects were in frame, from `seg_instance`. Positions are then read
               from the scene registry, standing in for the depth-based localisation a
               real system would do - the *visibility gate* is what matters here, and the
               position is a convenience once the gate has been passed.

The map is masked to one room. Without that, the bounding box of `kitchen_0` includes a
slice of the corridor and a chunk of the living room, so "fully explored" would be a claim
about floor the robot was never asked to search, and frontier selection would happily walk
the robot out of the room it was told to look in.

Occlusion is the other thing worth keeping. A disc of "seen" around the robot would claim
the far side of a counter without ever looking at it, and the room would be declared
searched on evidence nobody gathered - the false negative that sends replanning down a
branch the robot never ruled out.
"""

import os as _os, sys as _sys
# The repo root, found by marker rather than by counting parents, so these run from
# wherever they are filed. They import `world_graph` and `graph_machine` from there.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, 'src')):
    _d = _os.path.dirname(_d)
_roots = [_d, _os.path.join(_d, 'omnigibson_runtime')]
_roots += [_f.path for _r in ('src', 'benchmark')
           for _f in _os.scandir(_os.path.join(_d, _r))
           if _f.is_dir() and not _f.name.startswith(('.', '_'))]
for _p in _roots:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import math

# How many pixels an object must occupy to count as seen. One pixel is a sliver through a
# door crack or a corner of something behind a chair leg, and calling that an observation
# lets the graph claim knowledge the robot has not really got. Small, because the objects
# that matter are often small and far: a 0.077 m potato at 5 m still covers about 60 px at
# 640x360, so this filters noise without costing real sightings.
MIN_PIXELS = 4

# Cell states. Room membership is *not* one of them - it lives in `in_room`, so that a
# wall the robot genuinely saw can be marked OCCUPIED even though the segmentation puts it
# on a room's boundary rather than inside it.
UNKNOWN, FREE, OCCUPIED = 0, 1, 2

# Refuse to build a map larger than this many cells. `for_room` does one segmentation
# lookup per cell, which is nothing for a kitchen (a few thousand) and ruinous for the
# outdoors: `garden_0` in house_single_floor covers 1.4 million floor-plan pixels, and
# mapping it on demand because the robot clipped one garden cell would stall a run.
MAX_CELLS = 250_000

# Room types that are outdoors. They are rooms in the segmentation like any other, and
# `garden_0` in house_single_floor covers 1.4 million floor-plan pixels - twenty times the
# house. Including them in the scene map's extent leaves the building a smudge in the
# middle of a field, so the bounds are taken from the indoor rooms and the outdoors is
# simply off the edge of the map.
OUTDOOR_ROOMS = frozenset({"garden", "lawn", "driveway", "porch", "patio", "yard"})


class ObjectSemanticMap:
    """An occupancy grid over one room, plus the objects seen in it."""

    def __init__(self, bounds, resolution=0.1, room=None):
        import torch as th

        (self.x0, self.y0), (self.x1, self.y1) = bounds
        self.resolution = resolution
        self.room = room
        self.w = max(1, int((self.x1 - self.x0) / resolution))
        self.h = max(1, int((self.y1 - self.y0) / resolution))
        self.grid = th.zeros((self.h, self.w), dtype=th.uint8)
        # Which cells belong to this room. Kept apart from the grid so that a wall the
        # robot genuinely sees can be marked OCCUPIED even though the segmentation puts it
        # on the room's boundary rather than inside it. Coverage and frontier selection
        # still count only the room's own cells.
        self.in_room = th.ones((self.h, self.w), dtype=th.bool)
        # Which room instance each cell belongs to, 0 for none. Static, and only used for
        # drawing - it is what lets a scene-wide map still show where one room ends and
        # the next begins.
        self.room_id = th.zeros((self.h, self.w), dtype=th.int32)
        self.room_names = {}       # room instance id -> name
        self.objects = {}          # name -> {"category": str, "position": [x, y, z]}

    # ------------------------------------------------------------------ geometry

    def to_cell(self, x, y):
        return (int((y - self.y0) / self.resolution),
                int((x - self.x0) / self.resolution))

    def to_world(self, row, col):
        return (self.x0 + (col + 0.5) * self.resolution,
                self.y0 + (row + 0.5) * self.resolution)

    def in_bounds(self, row, col):
        return 0 <= row < self.h and 0 <= col < self.w

    def _fill_rooms(self, seg):
        """Label every cell with the room instance it falls in, in one pass.

        `world_to_map` is `flip(xy / resolution + size / 2)`, so the whole grid can be
        mapped with arithmetic instead of one call per cell. The per-cell loop this
        replaces was the slowest thing in map construction and would have been ruinous on
        a scene-wide grid of a quarter of a million cells.
        """
        import torch as th

        rooms = th.as_tensor(seg.room_ins_map)
        xs = self.x0 + (th.arange(self.w, dtype=th.float32) + 0.5) * self.resolution
        ys = self.y0 + (th.arange(self.h, dtype=th.float32) + 0.5) * self.resolution
        gy, gx = th.meshgrid(ys, xs, indexing="ij")
        rows = (gy / seg.map_resolution + seg.map_size / 2.0).long()
        cols = (gx / seg.map_resolution + seg.map_size / 2.0).long()
        good = ((rows >= 0) & (rows < rooms.shape[0])
                & (cols >= 0) & (cols < rooms.shape[1]))
        self.room_id[good] = rooms[rows[good], cols[good]].to(th.int32)
        self.room_names = dict(getattr(seg, "room_ins_id_to_ins_name", {}) or {})

    @classmethod
    def for_room(cls, scene, room_instance, resolution=0.1, pad=0.5):
        """A map covering one room, with `in_room` marking which cells are its own.

        The mask is read from the scene's room segmentation, one lookup per cell. A room
        at 0.1 m is a few thousand cells, so this costs nothing and is done once.
        """
        import torch as th

        seg = scene._seg_map
        ins_id = seg.room_ins_name_to_ins_id.get(room_instance)
        if ins_id is None:
            return None
        rows, cols = th.nonzero(th.as_tensor(seg.room_ins_map == ins_id), as_tuple=True)
        if rows.numel() == 0:
            return None
        corners = [seg.map_to_world(th.tensor([float(r), float(c)]))
                   for r, c in ((rows.min(), cols.min()), (rows.max(), cols.max()))]
        xs = sorted(float(c[0]) for c in corners)
        ys = sorted(float(c[1]) for c in corners)
        bounds = ((xs[0] - pad, ys[0] - pad), (xs[1] + pad, ys[1] + pad))

        omap = cls(bounds, resolution=resolution, room=room_instance)
        if omap.h * omap.w > MAX_CELLS:
            return None
        omap._fill_rooms(seg)
        omap.in_room = omap.room_id == ins_id
        return omap

    @classmethod
    def for_scene(cls, scene, resolution=0.1, pad=1.0, max_cells=MAX_CELLS):
        """One map over the whole floor plan, unmasked.

        The per-room maps answer "have I searched this room", and their mask is what makes
        that claim mean something. This answers a different question - "what do I know
        about the world" - and so has no mask at all: an object seen through a doorway
        into the next room is real knowledge, and belongs here even though it must not
        count towards having searched the room the robot is standing in.

        The resolution is coarsened until the grid fits `max_cells`. A whole house at
        0.1 m can run to millions of cells, and this map is for accumulating and drawing,
        not for frontier selection, so a coarser grid costs nothing that matters.
        """
        import torch as th

        seg = getattr(scene, "_seg_map", None)
        if seg is None or seg.room_ins_map is None:
            return None
        rooms = th.as_tensor(seg.room_ins_map)
        indoors = rooms > 0
        for ident, name in (getattr(seg, "room_ins_id_to_ins_name", {}) or {}).items():
            if name.rsplit("_", 1)[0] in OUTDOOR_ROOMS:
                indoors &= rooms != ident
        rows, cols = th.nonzero(indoors, as_tuple=True)
        if rows.numel() == 0:
            rows, cols = th.nonzero(rooms > 0, as_tuple=True)
        if rows.numel() == 0:
            return None
        corners = [seg.map_to_world(th.tensor([float(r), float(c)]))
                   for r, c in ((rows.min(), cols.min()), (rows.max(), cols.max()))]
        xs = sorted(float(c[0]) for c in corners)
        ys = sorted(float(c[1]) for c in corners)
        bounds = ((xs[0] - pad, ys[0] - pad), (xs[1] + pad, ys[1] + pad))

        while True:
            omap = cls(bounds, resolution=resolution, room="scene")
            if omap.h * omap.w <= max_cells or resolution > 1.0:
                omap._fill_rooms(seg)
                return omap
            resolution *= 2

    # ------------------------------------------------------------------ observation

    def observe_fov(self, x, y, yaw, h_fov, max_range, traversable_fn=None, n_rays=121):
        """Mark the camera's field of view as seen, stopping at structure.

        Casts `n_rays` across the horizontal FoV and walks each outward, marking cells
        FREE until it reaches a cell the traversability map calls blocked - a wall or a
        piece of furniture - which is marked OCCUPIED and ends the ray. Space behind it
        stays UNKNOWN, which is the occlusion that makes "I searched this room" mean
        something.

        Depth used to decide where a ray stopped, and that was wrong in a way that capped
        the whole map: the head sits pitched 0.45 rad down by default, so every ray's first
        depth return is the *floor*, about 2.5 m out. Rays therefore never reached the
        walls at all - which is why walls never appeared, obstacles were a sparse scatter,
        and coverage plateaued. Structure is what occludes; the floor is not.
        """
        for i in range(n_rays):
            frac = (i / (n_rays - 1)) - 0.5 if n_rays > 1 else 0.0
            angle = yaw + frac * h_fov
            steps = max(1, int(max_range / self.resolution))
            for s in range(steps + 1):
                d = s * self.resolution
                wx, wy = x + d * math.cos(angle), y + d * math.sin(angle)
                row, col = self.to_cell(wx, wy)
                if not self.in_bounds(row, col):
                    break
                if traversable_fn is not None and d > 0 and not traversable_fn(wx, wy):
                    if self.grid[row, col] == UNKNOWN:
                        self.grid[row, col] = OCCUPIED
                    break
                if self.grid[row, col] == UNKNOWN:
                    self.grid[row, col] = FREE

    def record_object(self, name, category, position):
        self.objects[name] = {"category": category, "position": list(position)}

    # ------------------------------------------------------------------ progress

    def coverage(self):
        """Fraction of the room's own cells that are no longer unknown."""
        import torch as th

        total = int(self.in_room.sum())
        if total == 0:
            return 1.0
        seen = ((self.grid == FREE) | (self.grid == OCCUPIED)) & self.in_room
        return float(seen.sum()) / total

    def frontiers(self):
        """Free cells inside the room that touch unknown space inside the room.

        The boundary between what has been seen and what has not is where standing gains
        new information. Vectorised, because this runs after every observation.
        """
        import torch as th

        free = (self.grid == FREE) & self.in_room
        unknown = (self.grid == UNKNOWN) & self.in_room
        touches = th.zeros_like(unknown)
        touches[:-1, :] |= unknown[1:, :]
        touches[1:, :] |= unknown[:-1, :]
        touches[:, :-1] |= unknown[:, 1:]
        touches[:, 1:] |= unknown[:, :-1]
        rows, cols = th.nonzero(free & touches, as_tuple=True)
        return [self.to_world(int(r), int(c)) for r, c in zip(rows, cols)]

    def nearest_frontier(self, x, y, min_distance=0.5, exclude=()):
        """Closest frontier at least `min_distance` away, so the robot actually moves.

        `exclude` holds cells already found unreachable, so a frontier the planner cannot
        get to is not chosen again on the next pass.
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

    def summary(self):
        import torch as th

        inside = int(self.in_room.sum())
        return (f"{self.room}: {self.coverage():.0%} of {inside} cells seen, "
                f"{len(self.objects)} objects, {len(self.frontiers())} frontiers")


# -------------------------------------------------------------------- camera plumbing

def robot_camera(robot):
    """The robot's head vision sensor, or None."""
    from omnigibson.sensors.vision_sensor import VisionSensor

    cams = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
    if not cams:
        return None
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
    try:
        aperture = float(cam.horizontal_aperture)
        focal = float(cam.focal_length)
        if focal > 0:
            return 2.0 * math.atan(aperture / (2.0 * focal))
    except Exception:
        pass
    return default


def observe(env, robot, omap, max_range=5.0, also=(), ignore=()):
    """One look: record what is in frame, and mark what the camera covered.

    Returns the set of object names seen. Nothing here reads the scene registry to decide
    *what exists* - only `seg_instance` does that, so an object the camera never saw is
    never recorded.
    """
    import torch as th
    import omnigibson.utils.transform_utils as T

    obs, info = robot.get_obs()

    # `robot.get_obs()` is keyed by *sensor*, and each sensor's entry is itself a dict of
    # modalities: obs["robot0:eyes:Camera:0"]["seg_instance"]. Matching modality names
    # against the top-level keys therefore finds nothing at all - which is exactly what
    # happened: coverage climbed while the robot drove a kitchen full of furniture and
    # reported zero objects seen. Walk one level down, and tolerate a flat dict too.
    def modality(name):
        """Every (data, info-mapping) pair for one modality, over all sensors."""
        out = []
        for key, value in (obs or {}).items():
            if isinstance(value, dict):
                if name in value:
                    out.append((value[name], (info or {}).get(key, {}).get(name, {})))
            elif key.endswith(name):
                flat = {}
                for info_key, info_val in (info or {}).items():
                    if info_key.endswith(name) and isinstance(info_val, dict):
                        flat = info_val
                        break
                out.append((value, flat))
        return out

    # --- what is in frame ---
    seen = set()
    for seg, mapping in modality("seg_instance"):
        idents, counts = th.unique(seg, return_counts=True)
        for ident, count in zip(idents.tolist(), counts.tolist()):
            if count < MIN_PIXELS:
                continue
            name = mapping.get(ident, mapping.get(str(ident)))
            if isinstance(name, str):
                seen.add(name)

    scene = env.scene
    maps = [m for m in (omap,) + tuple(also) if m is not None]

    # The building is not an object. Walls, floors and ceilings fill most of every frame,
    # and recording them would add hundreds of nodes for things no plan ever refers to.
    ignore = set(ignore)
    seen = {n for n in seen
            if getattr(scene.object_registry("name", n), "category", None) not in ignore}

    for name in seen:
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        pos, _ = obj.get_position_orientation()
        for m in maps:
            m.record_object(name, obj.category, pos.tolist())

    # --- how much of the room that look covered ---
    pos, orn = robot.get_position_orientation()
    x, y = float(pos[0]), float(pos[1])
    yaw = float(T.quat2euler(orn)[2])

    cam = robot_camera(robot)
    h_fov = camera_fov(cam) if cam is not None else 1.2

    trav = getattr(scene, "trav_map", None)

    def traversable(wx, wy):
        if trav is None:
            return True
        try:
            return bool(trav.has_node(0, th.tensor([wx, wy])))
        except Exception:
            return True

    for m in maps:
        m.observe_fov(x, y, yaw, h_fov, max_range, traversable_fn=traversable)

    # Object footprints are deliberately *not* painted in.
    #
    # An earlier version stamped each seen object's bounding box as OCCUPIED, to get solid
    # obstacle shapes instead of the dotted arcs the rays were leaving. Those arcs were a
    # symptom of a different bug - the rays were stopping on the floor at 2.5 m - and with
    # that fixed the traversability map already marks furniture, so 121 rays hitting real
    # geometry draw the surfaces on their own.
    #
    # Painting did two things wrong that no amount of tuning fixes. A *carried* object was
    # stamped at every observation along the route, smearing a black trail across the map
    # behind the robot; and an object seen but not drawn - the `relevant` filter keeps only
    # the task's - left a solid obstacle the picture never explained. `sim2d` has no
    # equivalent, which is the other reason to drop it: the two maps are meant to be the
    # same picture.
    return seen
