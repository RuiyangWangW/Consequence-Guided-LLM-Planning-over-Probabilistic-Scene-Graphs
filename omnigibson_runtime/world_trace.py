"""Record the world model as it is built, and render it as an animation.

Two halves, deliberately separated. `Trace` runs inside the simulator and does nothing but
append cheap snapshots - a copy of the room's occupancy grid, the objects known so far, and
the graph's edges. `render` runs afterwards, offline, and turns those into frames. Keeping
matplotlib out of the simulation loop matters: a run is already seventeen minutes, and a
drawing bug should not be able to kill it.

The animation answers a question the logs cannot. A graph printed at the end says what the
robot ended up believing; it says nothing about *when* each edge appeared, which is the
part worth checking. An edge that shows up at the right moment - `on_top(potato, plate)` the
instant the placement succeeds, not before - is evidence the model is tracking the world
rather than coincidentally agreeing with it at the end.
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


import json
import os


class Trace:
    """Snapshots of the map and the graph, appended as the run proceeds."""

    def __init__(self, path, focus=()):
        self.path = path
        self.focus = list(focus)
        self.frames = []

    def add(self, event, graph, omap=None, robot_pose=None, held=None,
            distance=0.0):
        """One snapshot, in the schema `sim2d_render` draws.

        The two animations are meant to be read side by side, so this records exactly what
        that renderer wants rather than a shape of its own - plus the coverage grid, which
        the 2D simulator reconstructs from poses and this side already has for real.
        """
        coverage = None
        bounds = None
        rooms = None
        if omap is not None:
            coverage = [[int(v) for v in row] for row in omap.grid.tolist()]
            bounds = [omap.x0, omap.y0, omap.x1, omap.y1, omap.resolution]
            rooms = {"id": [[int(v) for v in row] for row in omap.room_id.tolist()],
                     "names": {str(k): v for k, v in (omap.room_names or {}).items()}}
        x, y, yaw = robot_pose or (0.0, 0.0, 0.0)
        self.frames.append({
            "event": event,
            "x": float(x), "y": float(y), "yaw": float(yaw),
            "distance": float(distance),
            "held": held,
            "known": sorted(graph.objects),
            "positions": {n: (r.get("position") or [0.0, 0.0, 0.0])
                          for n, r in graph.objects.items()},
            "edges": [list(e) for e in sorted(graph.edges) if e[0] != "room_connect"],
            "coverage": coverage,
            "bounds": bounds,
            # Not "rooms" - `sim2d_render` has no such field, and an earlier version of
            # this used that name twice in one dict literal, so the layer was silently
            # dropped on every frame.
            "room_map": rooms,
        })

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"focus": self.focus, "frames": self.frames}, f)
        return self.path


# ------------------------------------------------------------------ rendering
#
# Delegated to `sim2d_render`, deliberately. Both animations answer the same question and
# are meant to be read side by side, and the only way to keep them identical is to draw
# them with the same code rather than with two implementations of the same conventions.
# An earlier version of this file reimplemented them and drifted: uniform green floor
# instead of room tints, outlines instead of colour, no trail, no field-of-view wedge, and
# every object drawn instead of the task's.


class _TraceWorld:
    """The little of `floor_world.World` that `sim2d_render` actually reads.

    The 2D simulator owns a world and derives its map from it. Here the map came from a
    real camera in Isaac and was recorded frame by frame, so the world has to be
    reconstructed from the recording - the floor, the room labels, and the cell/world
    conversion - which is all the renderer asks for.
    """

    class _Truth:
        def position_of(self, name):
            return None

    def __init__(self, frame):
        import numpy as np

        info = frame.get("room_map") or {}
        self.room_id = np.array(info.get("id") or [[0]])
        names = info.get("names") or {}
        self.x0, self.y0, self.x1, self.y1, self.resolution = (
            frame.get("bounds") or (0.0, 0.0, 1.0, 1.0, 0.1))
        self.free = self.room_id > 0
        self.n = max(self.room_id.shape)
        self.truth = self._Truth()
        self._ids = {}
        self.rooms = {}
        for key, name in names.items():
            try:
                ident = int(key)
            except (TypeError, ValueError):
                continue
            if not (self.room_id == ident).any():
                continue
            self._ids[name] = ident
            self.rooms[name] = {"room_type": name.rsplit("_", 1)[0]}

    def to_cell(self, x, y):
        return (int((y - self.y0) / self.resolution),
                int((x - self.x0) / self.resolution))

    def room_mask(self, name):
        return self.room_id == self._ids.get(name, -1)


def render(trace_path, out_path, fps=2, dpi=90, stride=1, focus=()):
    """Read a trace and write the side-by-side animation, via `sim2d_render`."""
    import sim2d_render

    with open(trace_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        frames, recorded = payload.get("frames", []), payload.get("focus", [])
    else:
        frames, recorded = payload, []
    focus = list(focus) or list(recorded)
    if not frames:
        raise SystemExit(f"{trace_path} has no frames")

    world = _TraceWorld(frames[-1])
    return sim2d_render.render(world, frames, out_path, fps=fps, stride=stride,
                               focus=focus)


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="figures/world_trace.json")
    parser.add_argument("--out", default="figures/world_trace.gif")
    parser.add_argument("--fps", type=float, default=2)
    parser.add_argument("--focus", nargs="*", default=None,
                        help="object names the pictures are about; anything more than "
                             "one relation away is kept in the graph but not drawn. "
                             "Defaults to what the run recorded.")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every nth 'moving' frame; action frames are always kept")
    args = parser.parse_args()

    path, n = render(args.trace, args.out, fps=args.fps, stride=args.stride,
                     focus=args.focus or ())
    print(f"wrote {path} ({n} frames)")


if __name__ == "__main__":
    main()
