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

import json
import os


class Trace:
    """Snapshots of the map and the graph, appended as the run proceeds."""

    def __init__(self, path, focus=()):
        self.path = path
        self.focus = list(focus)
        self.frames = []

    def add(self, event, graph, omap=None, robot_xy=None, held=None):
        """One snapshot. Cheap enough to call after every look and every action."""
        grid = None
        bounds = None
        in_room = None
        rooms = None
        if omap is not None:
            grid = [[int(v) for v in row] for row in omap.grid.tolist()]
            in_room = [[bool(v) for v in row] for row in omap.in_room.tolist()]
            bounds = [omap.x0, omap.y0, omap.x1, omap.y1]
            # Room labels are static; recorded per frame for simplicity, and they compress
            # to almost nothing next to the grid itself.
            rooms = {"id": [[int(v) for v in row] for row in omap.room_id.tolist()],
                     "names": {str(k): v for k, v in (omap.room_names or {}).items()}}
        self.frames.append({
            "event": event,
            "grid": grid,
            "in_room": in_room,
            "rooms": rooms,
            "bounds": bounds,
            "robot": list(robot_xy) if robot_xy is not None else None,
            "held": held,
            "objects": {n: (r.get("position") or [0, 0, 0])
                        for n, r in graph.objects.items()},
            "categories": {n: r.get("category") for n, r in graph.objects.items()},
            "edges": [list(e) for e in sorted(graph.edges) if e[0] != "room_connect"],
            "rooms": {n: graph.room_of(n) for n in graph.objects},
        })

    def save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"focus": self.focus, "frames": self.frames}, f)
        return self.path


# ------------------------------------------------------------------ rendering

# One colour per edge type, so a glance at the graph panel says which relation appeared.
EDGE_COLORS = {
    "room_inside": "#9aa0a6",
    "object_inside": "#d17b0f",
    "on_top": "#1a73e8",
    "under": "#8430ce",
    "next_to": "#188038",
}

# The same colours and the same conventions `sim2d_render.py` uses, so the two animations
# read the same way: unknown is a neutral grey, obstacles near-black, and *seen floor is
# tinted by its room's type* with the palette from `visualize_graph.py` - so a kitchen is
# the same colour here as in `figures/<scene>_graph.png`. Rooms are shown by colour rather
# than by outline, which is what makes a room the robot has not entered read as a grey
# hole in the middle of the picture.
UNKNOWN_COLOR = (0.373, 0.388, 0.408)      # #5f6368
OBSTACLE_COLOR = (0.122, 0.122, 0.122)     # #1f1f1f
UNROOMED_COLOR = (0.784, 0.902, 0.788)     # seen floor belonging to no room

CELL_COLORS = ["#5f6368", "#c8e6c9", "#1f1f1f", "#000000"]

# Grid state values, mirroring object_map's UNKNOWN / FREE / OCCUPIED.
UNKNOWN_V, FREE_V, OCCUPIED_V = 0, 1, 2


def relevant(frame, focus):
    """Which objects the pictures are drawn about, at this point in the run.

    The task's own objects, plus whatever the graph relates *directly* to one of them -
    the plate the potato ended up on, the oven it went into. Everything else the robot
    found on the way is knowledge it genuinely has and clutter in a figure: searching a
    kitchen for a potato turns up eight countertops, and drawing all eight buries the four
    the task is about. The set grows as relations appear, so an object joins the picture
    at the moment the robot relates it to the task and not before.

    One hop, deliberately. Following the relations transitively walks the whole counter
    run back in - each countertop is `next_to` the next one - and puts six of the eight
    back on the map by way of a chain that has nothing to do with the task.

    Nothing is discarded from the graph: this decides what is *drawn*, not what is known.
    With no `focus` there is nothing to be relevant to, and everything known is drawn.

    The same rule, and the same reasoning, as `sim2d_render.relevant` - the two animations
    are meant to be read side by side.
    """
    known = set(frame["objects"])
    task = {name for name in focus if name in known}
    if not task:
        return known
    keep = set(task)
    for edge_type, a, b in frame["edges"]:
        if edge_type == "room_inside":
            continue
        for near, far in ((a, b), (b, a)):
            if near in task and far in known:
                keep.add(far)
    return keep


def _room_tints(rid, names, np, plt):
    """A light colour per cell, keyed to the room *type*, as the figures key it."""
    from visualize_graph import ROOM_TYPE_ORDER

    palette = [plt.get_cmap("tab20")(i / 20.0) for i in range(20)]
    palette += [plt.get_cmap("tab20b")(i / 20.0) for i in range(20)]

    tint = np.tile(np.array(UNROOMED_COLOR), rid.shape + (1,))
    for key, name in (names or {}).items():
        try:
            ident = int(key)
        except (TypeError, ValueError):
            continue
        room_type = name.rsplit("_", 1)[0]
        index = (ROOM_TYPE_ORDER.index(room_type) if room_type in ROOM_TYPE_ORDER
                 else hash(room_type) % len(palette))
        colour = np.array(palette[index % len(palette)][:3])
        tint[rid == ident] = colour * 0.45 + 0.55
    return tint


def _short(name):
    """`countertop_kelker_0` -> `countertop`, so labels fit."""
    parts = name.split("_")
    if len(parts) >= 3 and len(parts[-2]) == 6:
        return "_".join(parts[:-2])
    return name


def render(trace_path, out_path, fps=2, dpi=90, stride=1, focus=()):
    """Read a trace and write a side-by-side animation of map and graph.

    `stride` keeps every nth frame. Mapping while the robot drives produces a snapshot
    every 30 simulation steps, which is the right resolution for watching the map fill in
    and far too many frames for a legible animation. Frames naming an action are always
    kept - those are the ones where an edge appears, and dropping one would hide the very
    moment worth seeing.
    """
    import matplotlib
    matplotlib.use("Agg")

    import imageio.v2 as imageio
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    with open(trace_path) as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        frames, recorded_focus = payload.get("frames", []), payload.get("focus", [])
    else:
        frames, recorded_focus = payload, []
    focus = list(focus) or list(recorded_focus)
    if not frames:
        raise SystemExit(f"{trace_path} has no frames")

    if stride > 1:
        frames = [f for i, f in enumerate(frames)
                  if i % stride == 0 or not f["event"].startswith("moving")
                  or i == len(frames) - 1]

    # A fixed layout across every frame, computed once from the final graph. Re-running a
    # spring layout per frame makes nodes jump around and the animation unreadable - the
    # point is to watch edges appear, not to watch the layout churn.
    final = frames[-1]
    nodes = list(final["objects"]) + sorted({r for r in final["rooms"].values() if r})
    scaffold = nx.Graph()
    scaffold.add_nodes_from(nodes)
    for etype, a, b in final["edges"]:
        if a in nodes and b in nodes:
            scaffold.add_edge(a, b)
    for name, room in final["rooms"].items():
        if room:
            scaffold.add_edge(name, room)
    pos = nx.spring_layout(scaffold, seed=7, k=0.9)

    images, trail = [], []
    for i, frame in enumerate(frames):
        fig, (ax_map, ax_graph) = plt.subplots(1, 2, figsize=(13.6, 6.2))

        # --- left: the object semantic map ---
        #
        # The robot's map, not the floor plan. Cells the camera has never reached stay
        # grey; what it has seen is drawn in its room's colour and what it stopped against
        # is drawn as obstacle, so a room it has not entered is a grey hole in the middle
        # of the picture. Same conventions as `sim2d_render.py`.
        x0, y0, x1, y1 = frame["bounds"] or (0.0, 0.0, 1.0, 1.0)
        if frame["grid"] is not None:
            grid = np.array(frame["grid"])
            info = frame.get("rooms") or {}
            rid = np.array(info.get("id")) if info.get("id") else np.zeros_like(grid)
            tint = _room_tints(rid, info.get("names"), np, plt)

            image = np.tile(np.array(UNKNOWN_COLOR), grid.shape + (1,))
            seen_free = grid == FREE_V
            image[seen_free] = tint[seen_free]
            image[grid == OCCUPIED_V] = np.array(OBSTACLE_COLOR)

            # Crop to the floor with a margin - most of a scene's raster is empty.
            rows, cols = np.nonzero(rid > 0)
            if len(rows):
                pad = max(2, int(0.1 * max(grid.shape)))
                r0, r1 = max(0, rows.min() - pad), min(grid.shape[0], rows.max() + pad + 1)
                c0, c1 = max(0, cols.min() - pad), min(grid.shape[1], cols.max() + pad + 1)
            else:
                r0, r1, c0, c1 = 0, grid.shape[0], 0, grid.shape[1]
            res_x = (x1 - x0) / grid.shape[1]
            res_y = (y1 - y0) / grid.shape[0]
            ext = [x0 + c0 * res_x, x0 + c1 * res_x, y0 + r0 * res_y, y0 + r1 * res_y]
            ax_map.imshow(image[r0:r1, c0:c1], origin="lower", extent=ext,
                          interpolation="nearest")
            ax_map.set_xlim(ext[0], ext[1])
            ax_map.set_ylim(ext[2], ext[3])

            # Name only the rooms the robot has actually been shown some of.
            for key, name in (info.get("names") or {}).items():
                try:
                    ident = int(key)
                except (TypeError, ValueError):
                    continue
                mask = (rid == ident) & (grid != UNKNOWN_V)
                if mask.sum() < 20:
                    continue
                rr, cc = np.nonzero(mask)
                ax_map.annotate(name, (x0 + (cc.mean() + 0.5) * res_x,
                                       y0 + (rr.mean() + 0.5) * res_y),
                                fontsize=6.5, color="#202124", ha="center", zorder=2,
                                path_effects=[pe.withStroke(linewidth=2.0,
                                                            foreground="white",
                                                            alpha=0.85)])

        # Fan the labels out vertically when objects sit close together, which they do -
        # a potato and a plate 0.9 m apart on the same counter overprint each other into
        # an unreadable smear at this scale.
        shown = relevant(frame, focus)
        placed = []
        for name, p in sorted(((n, p) for n, p in frame["objects"].items()
                               if n in shown), key=lambda kv: kv[1][0]):
            ax_map.plot(p[0], p[1], "o", ms=6, color="#d93025", zorder=4)
            offset = 5
            for prev_x, prev_off in placed:
                if abs(p[0] - prev_x) < (x1 - x0) * 0.06:
                    offset = prev_off + 9
            placed.append((p[0], offset))
            ax_map.annotate(_short(name), (p[0], p[1]), fontsize=6, color="#202124",
                            xytext=(4, offset), textcoords="offset points", zorder=5,
                            path_effects=[pe.withStroke(linewidth=1.8,
                                                        foreground="white", alpha=0.85)])

        # The robot, and where it has been.
        trail.append(tuple(frame["robot"]) if frame["robot"] else None)
        path = [p for p in trail if p]
        if len(path) > 1:
            ax_map.plot([p[0] for p in path], [p[1] for p in path], "-",
                        color="#d93025", linewidth=1.2, alpha=0.8, zorder=5)
        if frame["robot"]:
            ax_map.plot(frame["robot"][0], frame["robot"][1], marker="o", ms=9,
                        color="#202124", markeredgecolor="white", markeredgewidth=1.2,
                        zorder=6, linestyle="none")

        mapped = int((np.array(frame["grid"]) != UNKNOWN_V).sum()) if frame["grid"] else 0
        found = (f"{len(shown)} of {len(frame['objects'])} objects shown" if focus
                 else f"{len(frame['objects'])} objects found")
        ax_map.set_title(f"object semantic map - {mapped} cells seen, {found}",
                         fontsize=10)
        ax_map.set_aspect("equal")
        ax_map.axis("off")

        # --- right: the world graph ---
        g = nx.MultiDiGraph()
        g.add_nodes_from(n for n in frame["objects"] if n in shown)
        rooms_here = {r for n, r in frame["rooms"].items() if r and n in shown}
        g.add_nodes_from(rooms_here)
        here = {n: pos[n] for n in g.nodes if n in pos}
        room_nodes = [n for n in g.nodes if n in rooms_here]
        obj_nodes = [n for n in g.nodes if n not in rooms_here]
        nx.draw_networkx_nodes(g, here, nodelist=room_nodes, node_color="#e8eaed",
                               node_shape="s", node_size=900, ax=ax_graph)
        nx.draw_networkx_nodes(g, here, nodelist=obj_nodes, node_color="#fde293",
                               node_size=560, ax=ax_graph)
        nx.draw_networkx_labels(g, here, labels={n: _short(n) for n in g.nodes},
                                font_size=6, ax=ax_graph)
        for etype, a, b in frame["edges"]:
            if a not in here or b not in here:
                continue
            if a not in shown and a not in rooms_here:
                continue
            if b not in shown and b not in rooms_here:
                continue
            nx.draw_networkx_edges(g, here, edgelist=[(a, b)], ax=ax_graph,
                                   edge_color=EDGE_COLORS.get(etype, "#000000"),
                                   width=1.8, arrows=True, arrowsize=11,
                                   connectionstyle="arc3,rad=0.08")
        counts = {}
        for etype, _, _ in frame["edges"]:
            counts[etype] = counts.get(etype, 0) + 1
        legend = [matplotlib.lines.Line2D([], [], color=c, lw=2,
                                          label=f"{t} ({counts.get(t, 0)})")
                  for t, c in EDGE_COLORS.items()]
        ax_graph.legend(handles=legend, fontsize=6.5, loc="upper left", framealpha=0.9)
        ax_graph.set_title("world graph", fontsize=10)
        ax_graph.axis("off")

        held = f"    holding {_short(frame['held'])}" if frame.get("held") else ""
        fig.suptitle(f"{i + 1}/{len(frames)}   {frame['event']}{held}", fontsize=12)
        fig.tight_layout()

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        images.append(image)
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".gif"):
        imageio.mimsave(out_path, images, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(out_path, images, fps=fps)
    return out_path, len(images)


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
