"""The 2-D world: a scene's floor plan as a grid, its rooms as labels, its contents as state.

This is the environment half of the symbolic simulator - everything that is true about the
house, as opposed to what the robot believes about it. It is built from the same three
files the room graph is built from, so the world the robot drives around in is the world
the figures in `figures/` are pictures of:

    layout/floor_trav_no_door_0.png   free floor, furniture carved out   -> the grid
    layout/floor_insseg_0.png         which room each pixel belongs to   -> room labels
    room_graph.py                     which rooms are adjacent           -> room_connect
    json/<scene>_best.json            the furniture, by category         -> objects

Nothing here loads Isaac, and nothing here is physics. An object is a name, a category, a
position and a set of relations; a fridge is open because a flag says so. What the 2-D
grid buys is the part a symbolic model cannot fake: **whether the robot can actually get
there**, and **whether it could actually have seen it**. Those two are where plans really
fail, and they are the two things `graph_machine.py` has to assume.

Ground truth is stored as a `WorldGraph`, the same class the robot's belief uses. That is
not thrift - it is what makes the audit at the end of a run a set difference between two
objects of the same type, rather than a translation between two representations.

**The scene starts empty and you add what the task needs**, which is what
`test_primitives.py` does in Isaac: `load_object_categories` is restricted to the
structure plus the categories the plan touches, and the small objects are injected. Here
the default goes one step further and loads no furniture at all, because in two dimensions
an object has no extent and therefore nothing to stand in the way.

    world = FloorWorld.load("Rs_int", categories=["countertop", "oven"])
    world.add_object("potato", "potato", on_top="countertop_tpuwys_0")
    world.add_object("plate", "plate", room="kitchen_0")

Everything is 2-D. A position is `[x, y]` in metres, and that is the whole of an object's
geometry - there is no height, no extent and no support surface, because none of the nine
primitives is decided by one. An object on a counter is *at* the counter, and `on_top` is
what says it is on it. Coordinates are OmniGibson's: origin at the centre of the floor
plan, x to the right and y up. `to_cell`/`to_world` convert, and the mapping is the one `scene_setup.py`
uses (row = y/res + n/2, col = x/res + n/2), checked against every object's `in_rooms`
annotation in `test_sim2d.py`.
"""

import os as _os, sys as _sys
# Runnable as a script from anywhere. The other stages are sibling folders under src/, which
# are not on the path when this file is the one being executed, so find the repo root by
# marker and add every stage. A no-op when an entry point has already done it.
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


import glob
import heapq
import json
import math
import os

import numpy as np

from planner import NOT_GRASPABLE, OPENABLE, TOGGLEABLE
from world_graph import ROBOT, WorldGraph

DEFAULT_DATASET = os.environ.get(
    "BEHAVIOR_ASSETS",
    "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/datasets/behavior-1k-assets",
)

# The layout PNGs ship at 1 cm per pixel. The simulator works at 10 cm, which is
# OmniGibson's own `map_resolution` and the resolution `scene_setup.py` erodes at, so a
# stance this world calls free is one the real navigation map also calls free.
NATIVE_RESOLUTION = 0.01
DEFAULT_RESOLUTION = 0.1

# The traversability maps a scene ships. The default keeps the furniture and drops the
# doors.
#
# The doors go because there is no door-opening primitive: a shut interior door is not an
# obstacle the robot could do anything about, it is one that silently partitions the house.
# With the doors in, `house_single_floor` breaks into 10 pieces and `Beechwood_0_int` into
# 4; without them, both are one.
#
# The furniture stays even though the scene loads empty, which is what OmniGibson does:
# `load_object_categories` restricts which objects exist, and the traversability raster is
# read from the file regardless. Keeping it is also what makes the rest of the simulator
# worth anything - on the bare floor (`no_obj`) nothing occludes, so a room is "searched"
# from the doorway, and nothing blocks, so the robot parks in the middle of the counter it
# is reaching for. Measured: on `no_obj` the demo's stance lands 0.00 m from the potato and
# the kitchen reads 98% covered without the potato ever being seen.
TRAV_MAPS = {
    "no_door": "floor_trav_no_door_0.png",     # furniture, but no doors (default)
    "closed": "floor_trav_0.png",              # the raster as the scene ships it
    "open_door": "floor_trav_open_door_0.png",
    "no_obj": "floor_trav_no_obj_0.png",       # the bare floor: walls only
}

# Categories that are the building, not things in it. They are never loaded as objects:
# a plan cannot act on a wall, and 60 wall panels and 40 floor tiles in the world graph
# bury the handful of objects it is about. `execute_plan.py` refuses to navigate to the
# same set. Doors go with them - see TRAV_MAPS.
STRUCTURAL = {
    "walls", "wall", "floors", "floor", "ceilings", "ceiling", "roofs", "roof",
    "door", "doors", "window", "windows", "openable_window", "lawn", "driveway",
}

# `_erode_trav_map` uses radius = |chassis_extent_xy| / 2 + 0.2 with a square kernel;
# measured on a loaded Tiago that is 0.892 m, and that is the number to use when the
# question is whether the *real* robot would have fitted.
TIAGO_ERODE_RADIUS = 0.892

# What this simulator erodes by instead. The real map is eroded harder because CuRobo
# checks the whole 3-D body against the scene and the base has to clear a counter lip it
# would otherwise drive under; there is no 3-D body here, so eroding by Tiago's number
# takes floor away for a collision that cannot happen. At 0.892 m in `Rs_int` the kitchen
# keeps 4 standable cells and the bathroom none - a house no plan can run in. At 0.35 m,
# roughly Tiago's base radius, every room in every scene tried keeps floor to stand on.
DEFAULT_ROBOT_RADIUS = 0.35

# Doorways the robot cannot drive through because the raster says so rather than because
# the house does. Measured across all 51 scenes: 24 come out with their rooms in several
# regions of standable floor, and at the broken doorways the gap in the *un-eroded* floor
# map is 0.2 m to 0.3 m - a door threshold, a strip of non-floor pixels where the frame
# sits, not a wall. The room graph, built from dilated floor-plan adjacency, calls those
# rooms adjacent and is right to. `open_doorways` reconciles the two.
#
# A sill wider than this is not a threshold, and is left shut.
MAX_SILL = 0.5

# How wide a doorway is opened. It has to survive the footprint erosion that comes after,
# so it is set from the robot rather than from architecture: erosion takes `radius` off
# each side, and the margin is what is left to drive through.
DOOR_MARGIN = 0.2

# How many times one pair of rooms may have a doorway carved between them. More than one
# is worth having - a room with two doorways can have the first opened at the wrong end -
# but it has to be bounded: unbounded, `Wainscott_0_int` cuts 34 openings over ten passes
# and ends in the two regions it started in. Measured over all 51 scenes, a budget of 5
# reaches every scene an unbounded search reaches.
MAX_ATTEMPTS = 5

# An opening more than this many cells long is not a doorway. It is a detour through a
# room, and carving it would be inventing a corridor rather than clearing a threshold.
MAX_OPENING_CELLS = 40

# Two objects count as `next_to` when their centres are this close. The real predicate is
# an extent-aware ray cast; this is the symbolic stand-in for it, and it is derived rather
# than stored so that moving an object cannot leave a stale adjacency behind.
NEXT_TO_DISTANCE = 1.0


_NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
               (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def astar(mask, start, goal, corners=False):
    """Shortest path between two cells of a boolean map, 8-connected. None if none exists.

    `corners` allows the path to slip diagonally between two blocked cells. It is off for
    driving - the base would clip both - and on for finding where a doorway is, where two
    floors meeting at a single corner is exactly the case that has to be recognised.

    A* is the collision guarantee, not a planner in the loop: the map it runs on is already
    eroded by the robot's footprint, so any route it returns has clearance for the base
    everywhere along it.
    """
    if not (mask[start] and mask[goal]):
        return None
    if start == goal:
        return [start]
    height, width = mask.shape

    def heuristic(cell):
        dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
        return (math.sqrt(2) - 1) * min(dr, dc) + max(dr, dc)

    open_set = [(heuristic(start), 0.0, start)]
    came_from = {start: None}
    best = {start: 0.0}
    while open_set:
        _, cost, cell = heapq.heappop(open_set)
        if cell == goal:
            path = []
            while cell is not None:
                path.append(cell)
                cell = came_from[cell]
            return path[::-1]
        if cost > best.get(cell, float("inf")):
            continue
        row, col = cell
        for dr, dc, weight in _NEIGHBOURS:
            nxt = (row + dr, col + dc)
            if not (0 <= nxt[0] < height and 0 <= nxt[1] < width) or not mask[nxt]:
                continue
            # No cutting a corner the base would clip: a diagonal needs both of the
            # orthogonal cells it passes between.
            if dr and dc and not corners and not (mask[row + dr, col] and mask[row, col + dc]):
                continue
            new_cost = cost + weight
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                came_from[nxt] = cell
                heapq.heappush(open_set, (new_cost + heuristic(nxt), new_cost, nxt))
    return None


def _erode(mask, cells):
    """Shrink a boolean mask by a square kernel `cells` wide, as `_erode_trav_map` does."""
    if cells <= 1:
        return mask.copy()
    from scipy import ndimage

    return ndimage.binary_erosion(mask, np.ones((cells, cells), bool), border_value=0)


def reachable_rooms(scene, resolution=DEFAULT_RESOLUTION, radius=DEFAULT_ROBOT_RADIUS,
                    graphs_path=None):
    """The rooms the robot can actually get into, in this scene, fully furnished.

    A room the robot cannot enter is not part of its world. Objects in it can never be
    picked up, searched for or placed on; a plan that names one cannot succeed however good
    it is; and a probability the RSN spends on it is probability taken away from the rooms
    that can hold something. Half of `Wainscott_0_int` is like this - six of its twelve
    rooms, holding nine of its thirty-five pieces of furniture.

    **This is the one definition, and it is the simulator's own.** `Sim2D._search_room`
    refuses a room with "no standable floor in X reachable from Y", and that is exactly the
    test here: erode the map by the robot's radius, take the connected region the robot
    starts in, and keep the rooms with at least one cell in it. Two other notions were in
    use and both were wrong in a way that cost real tasks. Routing to a *stance beside an
    object* answers a different question - the stance can sit just outside the room, on
    reachable floor, while the room itself has none, which is how a console table in an
    unreachable bedroom looked reachable and broke nineteen tasks. And A* between *room
    centroids*, which `cost_matrix` used, is stricter than either: a centroid can be buried
    under furniture in a room the robot enters perfectly well.

    Measured on the fully furnished scene on purpose. The simulator loads only the
    categories a task mentions, so its floor is more open than the real house and a room
    can look reachable in one task and not the next. Reachability has to be a property of
    the building, or the benchmark means something different for every task in it.
    """
    key = (scene, round(resolution, 6), round(radius, 6), graphs_path)
    if key in _REACHABLE:
        return _REACHABLE[key]
    kwargs = {"graphs_path": graphs_path} if graphs_path else {}
    # `prune_unreachable=False` or this recurses: pruning asks this function what to keep.
    world = FloorWorld.load(scene, resolution=resolution, categories=None,
                            prune_unreachable=False, **kwargs)
    mask, labels = world.traversable(radius)
    cells = np.argwhere(mask)
    if not len(cells):
        _REACHABLE[key] = set()
        return _REACHABLE[key]
    # Where the robot starts: the middle of the largest standable region, which is what
    # `Sim2D._starting_pose` picks when no start room is given.
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    home = int(np.argmax(sizes))
    keep = {room for room in world.rooms
            if int((world.room_mask(room) & mask & (labels == home)).sum()) > 0}
    _REACHABLE[key] = keep
    return keep


_REACHABLE = {}


class FloorWorld:
    """One scene: the floor as a grid, the rooms as labels, the objects as ground truth."""

    def __init__(self, scene, resolution, free, room_ids, id_to_room, room_graph,
                 trav_map="no_door"):
        self.scene = scene
        self.trav_map = trav_map
        self.resolution = resolution
        self.free = free                # (n, n) bool: floor the robot could stand on
        self.room_ids = room_ids        # (n, n) int: floor-plan segment id, 0 = none
        self.id_to_room = id_to_room    # segment id -> room name, as room_graph names them
        self.n = free.shape[0]
        self.room_graph = room_graph
        self._footprints = {}
        self.rooms = room_graph["rooms"]

        # Ground truth, in the same class the robot's belief uses. Seeded with the rooms
        # and their adjacency; objects are added on top of that.
        self.truth = WorldGraph.from_room_graph(room_graph)
        self.open = {}      # object -> bool, for openable objects only
        self.toggled = {}   # object -> bool, for switchable objects only

        self._eroded = {}   # radius -> (mask, component labels)
        self.doorways = []  # thresholds `open_doorways` had to clear, for the record

    # ------------------------------------------------------------------ construction

    @classmethod
    def load(cls, scene, resolution=DEFAULT_RESOLUTION, dataset_root=DEFAULT_DATASET,
             graphs_path="data/room_graphs.json", categories=(), trav_map="no_door",
             open_doorways=True, radius=None, prune_unreachable=True):
        """Build the world for one scene from its shipped floor plans.

        `categories` says which of the scene's own furniture to load, and mirrors
        OmniGibson's `load_object_categories`:

            ()           an empty house - rooms, walls and floor, nothing else. The
                         default, and what most runs want: add what the task needs.
            ["oven", …]  those categories only, as `test_primitives.py` restricts its load
            None         everything the scene JSON declares

        Structural categories are never loaded whatever is asked for; see `STRUCTURAL`.
        `trav_map` picks which raster is the floor (`TRAV_MAPS`), and `open_doorways`
        clears the thresholds that leave adjacent rooms unreachable from one another.

        `prune_unreachable` drops the rooms the robot cannot get into, and everything in
        them, before anything downstream ever sees them - see `reachable_rooms`. It is on
        by default because a room behind a wall is not part of the robot's world in any
        sense that matters: it cannot be searched, nothing in it can be picked up, and a
        plan naming it cannot succeed. Keeping it only gave every stage its own chance to
        trip over it. `reachable_rooms` itself loads with this off, since that is the
        measurement it makes.
        """
        from PIL import Image

        # Our own dataset's floor plans, not untrusted input; the larger scenes trip PIL's
        # decompression-bomb guard.
        Image.MAX_IMAGE_PIXELS = None

        layout = os.path.join(dataset_root, "scenes", scene, "layout")
        trav_img = Image.open(os.path.join(layout, TRAV_MAPS[trav_map]))
        ins_img = Image.open(os.path.join(layout, "floor_insseg_0.png"))

        n = max(1, int(round(trav_img.size[0] * NATIVE_RESOLUTION / resolution)))
        free = np.array(trav_img.resize((n, n), Image.NEAREST)) > 0
        room_ids = np.array(ins_img.resize((n, n), Image.NEAREST)).astype(np.int32)

        room_graph = _load_room_graph(scene, graphs_path, dataset_root)
        id_to_room = {info["segment_id"]: name
                      for name, info in room_graph["rooms"].items()
                      if "segment_id" in info}

        world = cls(scene, resolution, free, room_ids, id_to_room, room_graph, trav_map)
        if open_doorways:
            world.open_doorways(radius or DEFAULT_ROBOT_RADIUS)
        if prune_unreachable:
            # The canonical set, measured on the fully furnished house, so that a room is
            # reachable or not as a fact about the building rather than about which
            # categories this particular load happens to want.
            world.keep_rooms(reachable_rooms(scene, resolution=resolution,
                                             radius=radius or DEFAULT_ROBOT_RADIUS,
                                             graphs_path=graphs_path))
        if categories is None or categories:
            world.load_scene_objects(dataset_root, categories)
        return world

    def keep_rooms(self, keep):
        """Drop every room not in `keep`, and every object standing in one.

        The floor raster is left alone - the robot may still walk over cells belonging to a
        dropped room, and pretending otherwise would carve holes in the map it navigates.
        What goes is the room's *name*: it stops being somewhere the RSN can rank, somewhere
        the searcher can be sent, a node in the graph the planner reasons over, and a room
        an instruction can name. That is the whole point - one rule, applied where the world
        is built, instead of every stage downstream having to remember.
        """
        keep = set(keep)
        dropped = [r for r in self.rooms if r not in keep]
        if not dropped:
            return self
        self.dropped_rooms = dropped
        # `room_at` answers from `id_to_room`, which is about to lose the dropped rooms - so
        # a point inside one would come back `None` and read as "roomless" rather than
        # "somewhere the robot cannot go". Keep the full map to tell those two apart.
        self._id_to_room_all = dict(self.id_to_room)
        self.rooms = {r: v for r, v in self.rooms.items() if r in keep}
        self.room_graph = dict(self.room_graph)
        self.room_graph["rooms"] = self.rooms
        self.room_graph["edges"] = [e for e in self.room_graph.get("edges", [])
                                    if e[0] in keep and e[1] in keep]
        self.id_to_room = {i: r for i, r in self.id_to_room.items() if r in keep}
        self.truth = WorldGraph.from_room_graph(self.room_graph)
        return self

    def load_scene_objects(self, dataset_root=DEFAULT_DATASET, categories=None):
        """Add the furniture the scene JSON declares, at its recorded position.

        `categories` restricts the load to those categories, the way
        `config["scene"]["load_object_categories"]` does in Isaac; None loads everything.
        Structure is skipped either way.

        Room membership comes from the *segmentation under the object's own position*, not
        from the JSON's `in_rooms` annotation: room ids here are the floor-plan-derived
        ones the rest of the pipeline uses, and the two namings need not coincide. Where
        both exist they agree on the room type for all but doors and windows, which sit in
        the wall between two rooms and belong to neither.
        """
        paths = sorted(glob.glob(os.path.join(dataset_root, "scenes", self.scene,
                                              "json", "*.json")))
        best = [p for p in paths if "best" in os.path.basename(p)]
        if not (best or paths):
            return 0
        with open((best or paths)[0]) as f:
            data = json.load(f)

        wanted = None if categories is None else set(categories)
        init = data["objects_info"]["init_info"]
        registry = data["state"]["registry"]["object_registry"]
        added = 0
        for name, info in init.items():
            category = info.get("args", {}).get("category")
            state = registry.get(name)
            if not category or not isinstance(state, dict) or "root_link" not in state:
                continue
            if category in STRUCTURAL or (wanted is not None and category not in wanted):
                continue
            position = [float(v) for v in state["root_link"]["pos"][:2]]
            # The dataset records switch state per object; use it where it exists rather
            # than assuming every oven starts off.
            non_kin = state.get("non_kin") or {}
            toggled = (non_kin.get("ToggledOn") or {}).get("value")
            opened = (non_kin.get("Open") or {}).get("value")
            # An object in a room the robot cannot enter is not part of its world either.
            # Loading it would let `ground` bind a plan's word to it and let the searcher
            # be sent after it, which is exactly the failure this pruning exists to remove.
            if getattr(self, "dropped_rooms", None):
                if self._true_room_at(*position) in set(self.dropped_rooms):
                    continue
            self.add_object(name, category, position=position,
                            toggled=toggled, opened=opened)
            added += 1
        return added

    # ------------------------------------------------------------------ geometry

    def to_cell(self, x, y):
        return (int(y / self.resolution + self.n / 2.0),
                int(x / self.resolution + self.n / 2.0))

    def to_world(self, row, col):
        return ((col + 0.5 - self.n / 2.0) * self.resolution,
                (row + 0.5 - self.n / 2.0) * self.resolution)

    def in_bounds(self, row, col):
        return 0 <= row < self.n and 0 <= col < self.n

    def is_free(self, x, y):
        row, col = self.to_cell(x, y)
        return bool(self.in_bounds(row, col) and self.free[row, col])

    def room_at(self, x, y, search=0.6):
        """Which room a point is in, or None.

        Points inside walls and furniture carry no segment id, and that is where objects
        that matter often sit - a door is in the wall between two rooms, a fridge is set
        into a counter run. Rather than call those roomless, take the majority room within
        `search` metres, which is the room a person would name.
        """
        row, col = self.to_cell(x, y)
        if not self.in_bounds(row, col):
            return None
        here = int(self.room_ids[row, col])
        if here in self.id_to_room:
            return self.id_to_room[here]
        r = int(round(search / self.resolution))
        lo_r, hi_r = max(0, row - r), min(self.n, row + r + 1)
        lo_c, hi_c = max(0, col - r), min(self.n, col + r + 1)
        patch = self.room_ids[lo_r:hi_r, lo_c:hi_c]
        ids, counts = np.unique(patch[patch > 0], return_counts=True)
        for seg in ids[np.argsort(-counts)]:
            if int(seg) in self.id_to_room:
                return self.id_to_room[int(seg)]
        return None

    def _true_room_at(self, x, y, search=0.6):
        """`room_at`, but answering from the map as it was before any room was pruned.

        Needed for exactly one question: is this object standing in a room the robot cannot
        reach? After pruning, `room_at` returns None for such a point, which is the same
        answer it gives for a point in a wall - and the two have to be told apart, or every
        object in a dropped room is kept as "roomless".
        """
        full = getattr(self, "_id_to_room_all", None)
        if full is None:
            return self.room_at(x, y, search)
        row, col = self.to_cell(x, y)
        if not self.in_bounds(row, col):
            return None
        here = int(self.room_ids[row, col])
        if here in full:
            return full[here]
        r = int(round(search / self.resolution))
        lo_r, hi_r = max(0, row - r), min(self.n, row + r + 1)
        lo_c, hi_c = max(0, col - r), min(self.n, col + r + 1)
        patch = self.room_ids[lo_r:hi_r, lo_c:hi_c]
        ids, counts = np.unique(patch[patch > 0], return_counts=True)
        for seg in ids[np.argsort(-counts)]:
            if int(seg) in full:
                return full[int(seg)]
        return None

    def room_mask(self, room):
        """Boolean mask of the cells belonging to one room."""
        segments = [seg for seg, name in self.id_to_room.items() if name == room]
        mask = np.zeros_like(self.free)
        for seg in segments:
            mask |= self.room_ids == seg
        return mask

    def traversable(self, radius=DEFAULT_ROBOT_RADIUS):
        """Free floor eroded by the robot's footprint, plus its connected components.

        Returns `(mask, labels)`. Both are cached per radius: neither depends on where the
        robot stands or what it is going to, and re-eroding a 750 000-cell map for every
        navigation is the cost that made this worth caching in the real system too.
        """
        key = round(float(radius), 3)
        if key not in self._eroded:
            from scipy import ndimage

            cells = int(math.ceil(radius / self.resolution))
            cells += 1 - cells % 2                     # odd, so erosion stays centred
            mask = _erode(self.free, cells)
            labels, _ = ndimage.label(mask, structure=np.ones((3, 3), int))
            self._eroded[key] = (mask, labels)
        return self._eroded[key]

    def region_of(self, x, y, radius=DEFAULT_ROBOT_RADIUS):
        """Which connected component of standable floor a point belongs to.

        Read from the *nearest* standable cell, not the cell itself: an object's centre is
        never traversable, and at a large erosion radius neither is the spot the robot is
        standing on. Reading the label directly returns the background, which no free cell
        carries, and every candidate stance is then discarded as unreachable.
        """
        mask, labels = self.traversable(radius)
        cell = self.nearest_free_cell(x, y, radius)
        return 0 if cell is None else int(labels[cell])

    def nearest_free_cell(self, x, y, radius=DEFAULT_ROBOT_RADIUS, limit=None):
        """The standable cell closest to a point, or None if nothing is near enough."""
        mask, _ = self.traversable(radius)
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return None
        row, col = self.to_cell(x, y)
        d2 = (rows - row) ** 2 + (cols - col) ** 2
        i = int(np.argmin(d2))
        if limit is not None and math.sqrt(d2[i]) * self.resolution > limit:
            return None
        return int(rows[i]), int(cols[i])

    def footprint(self, name, span=1.2):
        """The cells `name` occupies: the untraversable blob containing its centre.

        An object here is stored as a point, which is the middle of the thing. That is fine
        for a mug and misleading for a bed: the camera sees the near edge of a bed from a
        metre away, and the arm reaches the near edge, but a check measured from the centre
        of a two-metre bed says both are out of range. Flooding the blob gives the extent
        back, bounded by `span` so a sofa against a wall does not annex the wall.

        Returns a list of `(row, col)`, or the single centre cell for something that sits
        on free floor and therefore has no footprint of its own.
        """
        centre = self.cell_of(name)
        if centre is None:
            return []
        cached = self._footprints.get((name, span))
        if cached is not None:
            return cached
        row0, col0 = centre
        if self.free[row0, col0]:
            cells = [centre]
        else:
            limit = int(round(span / self.resolution))
            seen, stack = {centre}, [centre]
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < self.free.shape[0] and 0 <= nc < self.free.shape[1]):
                        continue
                    if abs(nr - row0) > limit or abs(nc - col0) > limit:
                        continue
                    if not self.free[nr, nc] and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            cells = sorted(seen)
        self._footprints[(name, span)] = cells
        return cells

    def reachable_point_on(self, support, near=None, span=1.2):
        """A point on `support` that a robot can actually stand beside.

        Objects here are points, and a support's point is its centre. That is fine for a
        mug and wrong for a sofa: placing a clock "on the sofa" put it at the sofa's middle,
        2.6 m from any floor the robot can stand on, so the robot could put the clock down
        and then never pick it up again. A real placement happens at arm's length, on the
        near edge of the furniture, and this returns that edge.

        The footprint is the blob of untraversable cells containing the support's centre,
        flooded no further than `span` metres so a sofa against a wall does not become the
        wall. Among those cells, the one closest to free floor wins - or closest to `near`
        (the robot) when it is given, which is the difference between "somewhere you could
        reach" and "where you are standing now".
        """
        centre = self.cell_of(support)
        if centre is None:
            return None
        row0, col0 = centre
        cells = self.footprint(support, span)
        if not cells or self.free[row0, col0]:
            return self.to_world(row0, col0)        # already standable; nothing to do

        # The footprint cells that touch free floor: where a robot could reach onto it.
        edge = [(r, c) for r, c in cells
                if any(0 <= r + dr < self.free.shape[0] and 0 <= c + dc < self.free.shape[1]
                       and self.free[r + dr, c + dc]
                       for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)))]
        if not edge:
            return self.to_world(row0, col0)
        if near is not None:
            target = self.to_cell(near[0], near[1])
        else:
            target = None
        if target is None:
            # Nearest to the centre keeps the object on the support rather than flung to
            # the far end of a long counter.
            best = min(edge, key=lambda rc: (rc[0] - row0) ** 2 + (rc[1] - col0) ** 2)
        else:
            best = min(edge, key=lambda rc: (rc[0] - target[0]) ** 2 + (rc[1] - target[1]) ** 2)
        return self.to_world(best[0], best[1])

    def sample_free(self, room=None, radius=DEFAULT_ROBOT_RADIUS, rng=None):
        """A standable point, optionally inside one room. Where to drop a new object."""
        mask, _ = self.traversable(radius)
        if room is not None:
            mask = mask & self.room_mask(room)
        rows, cols = np.nonzero(mask)
        if len(rows) == 0:
            return None
        rng = rng or np.random.default_rng(0)
        i = int(rng.integers(len(rows)))
        return self.to_world(int(rows[i]), int(cols[i]))

    # ------------------------------------------------------------------ contents

    def add_object(self, name, category=None, position=None, cell=None, room=None,
                   on_top=None, inside=None, toggled=None, opened=None, rng=None):
        """Put an object in the world, and say where.

        Position comes from whichever of these is given, in order: an explicit `position`
        in metres or `cell` as a `(row, col)` grid cell; a support or container, in which
        case the object takes its place and the relation is recorded; a room, in which case
        it lands on free floor there. An object with none of them exists but has no
        position, which is a legitimate state - it is what an object a plan mentions and
        nothing has placed yet looks like.
        """
        category = category or name.rsplit("_", 2)[0]
        if position is None and cell is not None:
            position = self.to_world(int(cell[0]), int(cell[1]))
        support = on_top or inside
        if position is None and support is not None:
            base = self.truth.position_of(support)
            if base is None:
                raise ValueError(f"cannot place '{name}' on '{support}': it has no position")
            # In two dimensions, on a counter and in a drawer are the same place. Which of
            # the two it is, is what the `on_top` / `object_inside` edge says.
            #
            # Not the support's *centre*, though: the near edge a robot could actually
            # stand beside. Spawning at the centre of a wide bed or a wedged-in coffee
            # table put 12 of the benchmark's objects further from any reachable floor than
            # the camera's 0.8 m sight margin or the arm's 1.5 m reach, so a known-good
            # plan could neither see nor grasp them.
            position = self.reachable_point_on(support) or list(base)
        if position is None and room is not None:
            position = self.sample_free(room, rng=rng)
            if position is None:
                raise ValueError(f"no standable floor in '{room}' to place '{name}'")
        position = list(position[:2]) if position is not None else None

        where = room or (self.room_at(position[0], position[1]) if position else None)
        self.truth.see_object(name, category, position, where)
        if on_top:
            self.truth.add_edge("on_top", name, on_top, note="placed at setup")
            self.truth.add_edge("under", on_top, name, note="placed at setup")
        if inside:
            self.truth.add_edge("object_inside", name, inside, note="placed at setup")

        if category in OPENABLE:
            self.open[name] = bool(opened) if opened is not None else False
        if category in TOGGLEABLE:
            self.toggled[name] = bool(toggled) if toggled is not None else False
        return self.truth.objects[name]

    def move_object(self, name, position, room=None):
        """Teleport an object, and re-derive which room it is in."""
        record = self.truth.objects.get(name)
        if record is None:
            raise KeyError(name)
        position = list(position[:2])
        record["position"] = position
        # An object in the robot's hand has no room of its own - `room_of` derives it from
        # the robot. Writing one here would put back the duplicate GRASP just removed, on
        # every metre the robot drives.
        if self.truth.carried(name):
            return record
        where = room or self.room_at(position[0], position[1])
        if where is not None:
            for _, old in self.truth.edges_of("room_inside", src=name):
                if old != where:
                    self.truth.remove_edge("room_inside", name, old, note="moved")
            self.truth.add_edge("room_inside", name, where, note="moved")
        return record

    def position_of(self, name):
        return self.truth.position_of(name)

    def cell_of(self, name):
        """Which grid cell an object occupies, or None if it has no position yet."""
        position = self.truth.position_of(name)
        return None if position is None else self.to_cell(position[0], position[1])

    def room_of(self, name):
        return self.truth.room_of(name)

    def category_of(self, name):
        return (self.truth.objects.get(name) or {}).get("category", name)

    def is_openable(self, name):
        return name in self.open

    def is_graspable(self, name):
        return self.category_of(name) not in NOT_GRASPABLE

    def distance_to(self, name, x, y):
        """How far the robot at (x, y) is from `name` - measured to its NEAR EDGE.

        An object is stored as a point, and for a bed or a sofa that point is a metre from
        anything the robot can touch. An arm reaches the edge of a bed, not its middle, so
        measuring to the centre refused grasps that a real robot makes easily: measured on
        the benchmark, twelve known-good plans failed this check on furniture the robot was
        standing right beside.
        """
        position = self.truth.position_of(name)
        if position is None:
            return float("inf")
        straight = math.hypot(position[0] - x, position[1] - y)
        cells = self.footprint(name)
        if len(cells) <= 1:
            return straight
        row, col = self.to_cell(x, y)
        nearest = min((r - row) ** 2 + (c - col) ** 2 for r, c in cells) ** 0.5
        return min(straight, nearest * self.resolution)

    def neighbours(self, name, distance=NEXT_TO_DISTANCE):
        """Objects close enough to `name` to count as beside it.

        Derived on demand rather than stored, so that moving an object can never leave a
        stale `next_to` behind - the one kinematic relation nothing in the action model
        explicitly maintains.
        """
        here = self.truth.position_of(name)
        if here is None or name == ROBOT:
            return []
        out = []
        for other, record in self.truth.objects.items():
            there = record.get("position")
            if other == name or other == ROBOT or there is None:
                continue
            if math.hypot(here[0] - there[0], here[1] - there[1]) <= distance:
                out.append(other)
        return sorted(out)

    def true_edges(self, names):
        """Ground-truth kinematic edges among `names`, with `next_to` derived.

        What the robot's belief should look like if it had seen everything in `names` and
        nothing had moved since. `audit` compares against exactly this, which is why
        `next_to` has to be derived here as well - stored on one side and derived on the
        other, every adjacency reads as a disagreement.
        """
        names = set(names)
        edges = {(t, a, b) for t, a, b in self.truth.edges
                 if t not in ("room_connect", "next_to") and a in names and b in names}
        # `next_to` is about two things standing beside each other on the floor. The robot
        # is beside almost everything it acts on and that is not a fact about the world.
        for name in names:
            for other in self.neighbours(name):
                if other in names:
                    edges.add(("next_to", *sorted((name, other))))
        return edges

    def contents_of(self, name):
        """Everything inside or on top of `name`, one level deep."""
        return sorted({a for _, a in [(t, a) for t, a, b in self.truth.edges
                                      if b == name and t in ("on_top", "object_inside")]})

    def open_doorways(self, radius=DEFAULT_ROBOT_RADIUS, max_sill=MAX_SILL, passes=8):
        """Clear the thresholds that leave adjacent rooms in different regions of floor.

        The room graph and the traversability raster disagree about this house, and the
        disagreement is one-sided. `room_graph.py` derives adjacency from dilated
        floor-plan pixels, which is what recovers archways and open-plan boundaries that
        carry no door object; the raster is a floor mesh, and at a doorway it has a 0.2 m
        to 0.3 m strip of non-floor where the threshold sits. Left alone, that strip cuts
        24 of the 51 scenes into pieces the robot cannot drive between - `Rs_int` into
        four, `Wainscott_1_int` into thirteen - and no plan that crosses a house can run.

        Two things are opened, and only where the room graph already says there is a way
        through:

            a sill    the two rooms are in different components of the *raw* floor, with a
                      gap no wider than `max_sill`. That is a threshold; it is carved.
            a pinch   the raw floor connects them but the footprint erosion does not, so
                      the doorway is narrower than the robot. The cells of the raw route
                      that erosion took are widened to a real door.

        Both are opened to `2 * radius + DOOR_MARGIN`, which is the width that survives the
        erosion that follows. Nothing is opened where the room graph has no edge, where the
        sill is wider than `max_sill`, or where the route needs more than
        `MAX_OPENING_CELLS` - those are walls, and they stay up.

        Records what it did in `self.doorways` and returns it.
        """
        from scipy import ndimage

        width = 2 * radius + DOOR_MARGIN
        half = max(1, int(round(width / 2 / self.resolution)))
        self.doorways = []

        # A pair gets a bounded number of attempts. More than one is worth having - a room
        # with two doorways can have the first one opened at the wrong end, and the second
        # pass finds the right one - but carving is unbounded otherwise: `Wainscott_0_int`
        # cut 34 openings over ten passes and ended in the two regions it started in.
        attempts = {}
        for _ in range(passes):
            opened = 0
            for room_a, room_b in self.room_graph["edges"]:
                if attempts.get((room_a, room_b), 0) >= MAX_ATTEMPTS:
                    continue
                # Re-read the labelling for every pair. Carving renumbers the components,
                # so a label captured before an earlier carve names a different region
                # afterwards - and opening one doorway routinely settles several pairs.
                mask, labels = self.traversable(radius)
                label_a = self._region_label(room_a, mask, labels)
                label_b = self._region_label(room_b, mask, labels)
                if label_a is None or label_b is None or label_a == label_b:
                    continue
                attempts[(room_a, room_b)] = attempts.get((room_a, room_b), 0) + 1
                if self._open_between(room_a, room_b, mask & (labels == label_a),
                                      mask & (labels == label_b), mask, half, max_sill,
                                      ndimage):
                    opened += 1
            if not opened:
                break
        return self.doorways

    def _region_label(self, room, mask, labels):
        """Which region of standable floor a room mostly sits in, or None if it has none."""
        cells = self.room_mask(room) & mask
        return int(np.bincount(labels[cells]).argmax()) if cells.any() else None

    def _open_between(self, room_a, room_b, side_a, side_b, mask, half, max_sill, ndimage):
        """Open one doorway between two regions of standable floor. True if it opened.

        Everything here is restricted to the two rooms' own floors. Without that, the
        narrowest gap between two *regions* is wherever in the house those regions happen
        to come closest, which is generally not this doorway and is sometimes a wall
        between two entirely different rooms.

        Where the two rooms' floors come closest is the doorway, and it is opened there
        whether the gap is a threshold to carve or floor already continuous but too narrow
        to drive. Taking the route first instead was wrong in a way worth recording: in
        `restaurant_brunch` the bathroom and the dining room touch at 0.10 m, and A* on the
        raw floor happily returned a 135-cell detour round the back of the building, 99
        cells of it too narrow - over the cap, so nothing opened and the bathroom stayed
        sealed off behind a gap two cells wide.
        """
        floor_a = self.free & self.room_mask(room_a)
        floor_b = self.free & self.room_mask(room_b)
        near_a = side_a & floor_a
        near_b = side_b & floor_b
        if not (near_a.any() and near_b.any()):
            return False

        start = self._closest_cell(near_a, near_b, ndimage)
        goal = self._closest_cell(near_b, near_a, ndimage)

        # Where the passage is, found across the two rooms' own floors and the unlabelled
        # cells of the boundary between them. Corner-cutting is allowed *here only*: two
        # floors that meet at a single diagonal cell are a doorway the raster has all but
        # closed, and refusing to see it is what left `restaurant_brunch`'s bathroom
        # sealed behind a two-cell gap while A* returned a 135-cell detour round the back
        # of the building instead.
        local = self.free & (self.room_mask(room_a) | self.room_mask(room_b)
                             | (self.room_ids == 0))
        route = astar(local, start, goal, corners=True)
        if route is not None:
            pinched = [cell for cell in route if not mask[cell]]
            if pinched and len(pinched) <= MAX_OPENING_CELLS:
                self._carve(pinched, half)
                self.doorways.append({"rooms": (room_a, room_b), "kind": "pinch",
                                      "cells": len(pinched)})
                return True

        # No floor joins them, or joining it would mean carving a corridor rather than
        # clearing a threshold. Carve where their own floors come closest, if that is
        # narrow enough to be a threshold at all.
        gap, p, q = self._narrowest_gap(floor_a, floor_b, ndimage)
        if gap is None or gap > max_sill:
            return False
        self._carve(_line(p, q), half)
        self.doorways.append({"rooms": (room_a, room_b), "kind": "sill",
                              "gap": round(gap, 2)})
        return True

    def _closest_cell(self, mask_from, mask_to, ndimage):
        """The cell of `mask_from` nearest to `mask_to`."""
        if not (mask_from.any() and mask_to.any()):
            return None
        distance = ndimage.distance_transform_edt(~mask_to)
        distance = np.where(mask_from, distance, np.inf)
        flat = int(np.argmin(distance))
        return (flat // self.n, flat % self.n)

    def _narrowest_gap(self, mask_a, mask_b, ndimage):
        """How far apart two masks are at their closest, and the pair of cells there."""
        p = self._closest_cell(mask_a, mask_b, ndimage)
        q = self._closest_cell(mask_b, mask_a, ndimage)
        if p is None or q is None:
            return None, None, None
        return math.dist(p, q) * self.resolution, p, q

    def _carve(self, cells, half):
        """Open a corridor `2 * half + 1` cells wide through the given cells."""
        for row, col in cells:
            lo_r, hi_r = max(0, row - half), min(self.n, row + half + 1)
            lo_c, hi_c = max(0, col - half), min(self.n, col + half + 1)
            self.free[lo_r:hi_r, lo_c:hi_c] = True
        self._eroded.clear()      # the map changed; every cached erosion is stale

    def regions(self, radius=DEFAULT_ROBOT_RADIUS):
        """The rooms grouped by which region of standable floor they sit in.

        Worth asking before a run rather than after. The room graph says which rooms are
        adjacent on the floor plan, which is not the same as the robot being able to drive
        between them: a doorway a chair stands in is an edge in the graph and a wall to
        A*. Measured across all 51 scenes, 27 come out as one region and 24 do not, and in
        one of those a plan that crosses the house cannot run however good it is.

        Returns `{region_label: [room, ...]}`, with rooms that have no standable floor at
        all under the key `None`.
        """
        mask, labels = self.traversable(radius)
        groups = {}
        for room in sorted(self.rooms):
            cells = self.room_mask(room) & mask
            key = int(np.bincount(labels[cells]).argmax()) if cells.any() else None
            groups.setdefault(key, []).append(room)
        return groups

    def summary(self):
        mask, labels = self.traversable()
        return (f"{self.scene}: {len(self.rooms)} rooms, {len(self.room_graph['edges'])} "
                f"connections, {len(self.truth.objects)} objects, "
                f"{int(self.free.sum())} free cells "
                f"({int(mask.sum())} standable in {labels.max()} components)")

    def __repr__(self):
        return f"<FloorWorld {self.summary()}>"


def _line(p, q):
    """The cells on the straight segment between two cells, endpoints included."""
    steps = max(abs(q[0] - p[0]), abs(q[1] - p[1]), 1)
    return [(round(p[0] + (q[0] - p[0]) * i / steps),
             round(p[1] + (q[1] - p[1]) * i / steps)) for i in range(steps + 1)]


def _load_room_graph(scene, graphs_path, dataset_root):
    """The room graph for a scene, from the cache when it is current, else rebuilt.

    The cached graphs predate `segment_id`, which is what ties a room name to the pixels
    it owns, so a cache without it is stale for this purpose and gets rebuilt.
    """
    if graphs_path and os.path.exists(graphs_path):
        with open(graphs_path) as f:
            graphs = json.load(f)
        graph = graphs.get(scene)
        if graph and all("segment_id" in r for r in graph["rooms"].values()):
            return graph

    from room_graph import build_room_graph

    return build_room_graph(scene, dataset_root)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="Rs_int")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--categories", nargs="*", default=[],
                        help="scene categories to load; none by default, "
                             "--categories with no names loads nothing, "
                             "--full-scene loads everything")
    parser.add_argument("--full-scene", action="store_true")
    parser.add_argument("--trav-map", default="no_door", choices=sorted(TRAV_MAPS))
    parser.add_argument("--no-doorways", action="store_true",
                        help="leave the raster's door thresholds shut")
    args = parser.parse_args()

    world = FloorWorld.load(args.scene, args.resolution, args.dataset_root,
                            categories=None if args.full_scene else args.categories,
                            trav_map=args.trav_map,
                            open_doorways=not args.no_doorways)
    print(world.summary())
    if world.doorways:
        print(f"opened {len(world.doorways)} doorways the raster had shut:")
        for door in world.doorways:
            detail = (f"{door['gap']:.2f} m sill" if door["kind"] == "sill"
                      else f"{door['cells']} cells too narrow")
            print(f"  {door['rooms'][0]:20s} <-> {door['rooms'][1]:20s} {detail}")
    groups = world.regions()
    tiny = groups.pop(None, [])
    if tiny:
        print(f"\n{len(tiny)} rooms are too small for the robot to stand in: "
              f"{', '.join(tiny)}")
    if len(groups) > 1:
        print(f"\nwarning: the rooms fall into {len(groups)} regions the robot cannot "
              f"drive between:")
        for key, rooms in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            print(f"  region {key:<15d} {', '.join(rooms)}")
    print()
    mask, labels = world.traversable()
    for name in sorted(world.rooms):
        room = world.room_mask(name)
        standable = int((room & mask).sum())
        contents = [n for n in world.truth.objects if world.room_of(n) == name]
        print(f"  {name:22s} {int(room.sum()):6d} cells, {standable:5d} standable, "
              f"{len(contents):3d} objects")


if __name__ == "__main__":
    main()
