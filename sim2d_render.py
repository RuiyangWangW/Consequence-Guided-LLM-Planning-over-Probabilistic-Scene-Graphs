"""Draw a `sim2d` run: the object semantic map on the left, the world graph on the right.

The same two panels `world_trace.py` renders for a simulator run, and the same question:
not what the robot ended up believing, but *when* each belief appeared. An edge that shows
up at the right moment - `on_top(potato, plate)` the instant the placement succeeds, not
before - is evidence the model is tracking the world rather than coincidentally agreeing
with it at the end.

**The left panel is the robot's map, not the floor plan.** Cells the camera has never
reached are grey and stay grey; what it has seen is drawn in its room's colour, and what
it stopped against is drawn as obstacle. A room the robot has not entered is a grey hole
in the middle of the picture, which is the point - the picture is what the robot knows.

Kept out of the simulator for the reason `world_trace.py` gives - a drawing bug should not
be able to kill a run - and because coverage is a pure function of where the robot stood,
so replaying `cast_fov` from the recorded poses reconstructs it exactly and a frame in the
trace stays a dozen numbers rather than a grid.

    python sim2d.py --scene Beechwood_0_int --demo --gif figures/sim2d.gif
    python sim2d_render.py --trace figures/sim2d_trace.json --out figures/sim2d.gif
"""

import json
import math
import os

import numpy as np

from sim2d import CAMERA_FOV, CAMERA_RANGE, FREE, OCCUPIED, UNKNOWN, cast_fov
from world_graph import ROBOT

# One colour per edge type, the assignment `world_trace.py` uses, so the two animations
# read the same way.
EDGE_COLORS = {
    "room_inside": "#9aa0a6", "object_inside": "#d17b0f", "on_top": "#1a73e8",
    "under": "#8430ce", "next_to": "#188038", "holding": "#d93025",
}

# Unknown is the state the map spends most of its time in, so it reads better as a neutral
# grey than as something darker than the obstacles.
UNKNOWN_COLOR = np.array([0.373, 0.388, 0.408])     # #5f6368
OBSTACLE_COLOR = np.array([0.122, 0.122, 0.122])    # #1f1f1f
UNROOMED_COLOR = np.array([0.784, 0.902, 0.788])    # free floor outside any room


def _room_tints(world):
    """A light colour per cell, keyed to the room *type* as the figures key it.

    Same palette and same order as `visualize_graph.py`, so a kitchen is the same colour
    here as in `figures/<scene>_graph.png`.
    """
    import matplotlib.pyplot as plt

    from visualize_graph import ROOM_TYPE_ORDER

    palette = [plt.get_cmap("tab20")(i / 20.0) for i in range(20)]
    palette += [plt.get_cmap("tab20b")(i / 20.0) for i in range(20)]

    tint = np.tile(UNROOMED_COLOR, world.free.shape + (1,))
    for name, info in world.rooms.items():
        room_type = info["room_type"]
        index = (ROOM_TYPE_ORDER.index(room_type) if room_type in ROOM_TYPE_ORDER
                 else hash(room_type) % len(palette))
        colour = np.array(palette[index % len(palette)][:3])
        tint[world.room_mask(name)] = colour * 0.45 + 0.55
    return tint


def _extent(world):
    """Crop to the floor, with a margin. Most of a scene's raster is empty."""
    rows, cols = np.nonzero(world.free)
    pad = int(2.0 / world.resolution)
    return (max(0, int(rows.min()) - pad), min(world.n, int(rows.max()) + pad + 1),
            max(0, int(cols.min()) - pad), min(world.n, int(cols.max()) + pad + 1))


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

    With no `focus` there is nothing to be relevant *to*, and everything known is drawn.
    With one, the answer can legitimately be nothing: early in a run the robot has found
    only furniture the task never mentions, and showing all of it because none of the task
    is known yet would fill the first frames with exactly what the filter exists to leave
    out.
    """
    known = set(frame["known"])
    # The robot is always relevant - it is the thing doing the task - and it is not an
    # observation, so it is never in `known`.
    if any(name == ROBOT for edge in frame["edges"] for name in edge[1:]):
        known.add(ROBOT)
    if not focus:
        return known
    task = {name for name in focus if name in known} | (known & {ROBOT})
    keep = set(task)
    for edge_type, a, b in frame["edges"]:
        if edge_type == "room_inside":
            continue
        for near, far in ((a, b), (b, a)):
            if near in task and far in known:
                keep.add(far)
    return keep


def render(world, frames, out_path, fps=3, stride=1, graph_panel=True, focus=()):
    """Write an animation of a run. Returns `(path, frames written)`."""
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    if not frames:
        raise ValueError("nothing to render: the run recorded no frames")
    if stride > 1:
        frames = [f for i, f in enumerate(frames)
                  if i % stride == 0 or not f["event"].startswith("moving")
                  or i == len(frames) - 1]

    tint = _room_tints(world)
    r0, r1, c0, c1 = _extent(world)
    coverage = np.zeros_like(world.free, dtype=np.uint8)
    positions = _graph_layout(frames, focus) if graph_panel else None

    images, trail = [], []
    for i, frame in enumerate(frames):
        # Every heading the robot observed at, not just the one it ended up facing: a
        # stop is a four-heading scan, and replaying one of them leaves three quarters of
        # what the robot saw off the map.
        # A run that measured its own coverage hands it over directly. The 2D simulator
        # replays `cast_fov` because a pose is a dozen numbers where a grid is thousands;
        # the BEHAVIOR-1K side has a real camera and a real occupancy grid already, and
        # re-deriving it here from poses would draw something subtly different from what
        # the robot actually knew.
        if frame.get("coverage") is not None:
            coverage = np.asarray(frame["coverage"], dtype=np.uint8)
        else:
            for heading in frame.get("headings") or [frame["yaw"]]:
                cast_fov(world, frame["x"], frame["y"], heading, CAMERA_FOV,
                         CAMERA_RANGE, coverage)
        trail.append((frame["x"], frame["y"]))

        panels = 2 if graph_panel else 1
        fig, axes = plt.subplots(1, panels, figsize=(6.8 * panels, 6.2))
        ax_map = axes[0] if graph_panel else axes

        image = np.tile(UNKNOWN_COLOR, world.free.shape + (1,))
        image[coverage == FREE] = tint[coverage == FREE]
        image[coverage == OCCUPIED] = OBSTACLE_COLOR
        ax_map.imshow(image[r0:r1, c0:c1], origin="lower", interpolation="nearest")
        # Pin the axes to the floor plan. The camera wedge is 5 m of patch and the robot
        # spends much of a run near a wall, so autoscaling to fit it zooms the map out to
        # accommodate empty space outside the building - the map shrinks and the wedge
        # gains nothing, since it is only meaningful where there is floor. Patches clip to
        # the axes, so freezing the limits crops the wedge at the edge of the plan.
        ax_map.autoscale(False)

        shown = relevant(frame, focus)
        _label_rooms(ax_map, world, coverage, r0, c0, pe)
        _draw_objects(ax_map, world, frame, r0, c0, pe, shown)
        _draw_robot(ax_map, world, frame, trail, r0, c0, Wedge)

        mapped = int((coverage > UNKNOWN).sum())
        found = (f"{len(shown)} of {len(frame['known'])} objects" if focus
                 else f"{len(frame['known'])} objects found")
        ax_map.set_title(f"object semantic map — {mapped} cells seen, {found}",
                         fontsize=10)
        ax_map.axis("off")

        if graph_panel:
            _draw_graph(axes[1], frame, positions, matplotlib, shown)

        held = f"    holding {_short(frame['held'])}" if frame.get("held") else ""
        fig.suptitle(f"{i + 1}/{len(frames)}   {frame['event']}   "
                     f"{frame['distance']:.1f} m driven{held}", fontsize=12)
        fig.tight_layout()
        fig.canvas.draw()
        images.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])
        plt.close(fig)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if out_path.endswith(".gif"):
        imageio.mimsave(out_path, images, duration=1.0 / fps, loop=0)
    else:
        imageio.mimsave(out_path, images, fps=fps)
    return out_path, len(images)


def _short(name):
    """`countertop_kelker_0` -> `countertop`, so labels fit. As `world_trace._short`."""
    parts = name.split("_")
    if len(parts) >= 3 and len(parts[-2]) == 6:
        return "_".join(parts[:-2])
    return name


def _label_rooms(ax, world, coverage, r0, c0, pe):
    """Name each room the robot has actually been shown some of."""
    for name in world.rooms:
        mask = world.room_mask(name) & (coverage > UNKNOWN)
        if mask.sum() < 20:
            continue
        rows, cols = np.nonzero(mask)
        ax.annotate(name, (cols.mean() - c0, rows.mean() - r0), fontsize=6.5,
                    color="#202124", ha="center", zorder=2,
                    path_effects=[pe.withStroke(linewidth=2.0, foreground="white",
                                                alpha=0.85)])


# Above this many objects on the map, only the ones an action has touched are named. The
# house loads empty so a run normally has a dozen, which all fit; `--full-scene` has 190,
# and naming those turns the kitchen into a block of overlapping text.
MAX_MAP_LABELS = 40

# How far apart two labels have to be, in metres, for both to be drawn. A counter run is
# eight countertop instances within a couple of metres, and eight names on top of each
# other read as one piece of mangled text - "untertop" - which is worse than no name.
LABEL_SPACING = 1.6


def _draw_objects(ax, world, frame, r0, c0, pe, shown):
    """Only what the robot has found, where it was at this moment in the run.

    These are the *object* half of an object semantic map - the coverage grid is what the
    camera swept, and these are what it found in it. Objects an action has written a
    relation about are picked out in red and named in bold; the rest are the furniture the
    search turned up on the way, and they are named too, in grey, so that a dot on the map
    is never something the reader has to guess at.
    """
    acting = {n for t, a, b in frame["edges"] if t not in ("room_inside", "next_to")
              for n in (a, b)}
    if frame.get("held"):
        acting.add(frame["held"])
    positions = frame.get("positions") or {}
    label_all = len(shown) <= MAX_MAP_LABELS
    spacing = LABEL_SPACING / world.resolution
    placed = []

    # Objects an action touched are drawn last, so they sit on top, but they claim their
    # label first, so a counter the robot happens to be beside can never crowd out the
    # potato's name.
    for name in sorted(shown, key=lambda n: n not in acting):
        # The robot is on this panel already, as the triangle that shows where it faces.
        if name == ROBOT:
            continue
        position = positions.get(name) or world.truth.position_of(name)
        if position is None:
            continue
        row, col = world.to_cell(position[0], position[1])
        touched = name in acting
        ax.plot(col - c0, row - r0, "o", markersize=8.0 if touched else 4.5,
                color="#ea4335" if touched else "#1a73e8",
                markeredgecolor="white", markeredgewidth=0.8,
                zorder=4 if touched else 3)
        if not (touched or label_all):
            continue
        crowded = any(math.dist((row, col), other) < spacing for other in placed)
        if crowded and not touched:
            continue
        placed.append((row, col))
        # An object an action touched always keeps its name; if a name is already written
        # there, this one goes underneath instead. The potato is *inside* the oven, so the
        # two share a cell and one of the two labels has to move or they overprint - which
        # is how "oven" and "potato" came out as `oveno`.
        offset = (6, -11) if crowded else (6, 4)
        # A white stroke behind every label. Half the map is the dark grey of unexplored
        # space and the other half is pale room colour, so no single text colour is
        # legible on both - unstroked, the found furniture's names vanished wherever they
        # happened to fall outside what the robot had mapped.
        ax.annotate(_short(name), (col - c0, row - r0),
                    fontsize=7.5 if touched else 6.0, xytext=offset,
                    textcoords="offset points", zorder=5,
                    color="#a50e0e" if touched else "#263238",
                    fontweight="bold" if touched else "normal",
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white",
                                                alpha=0.9)])


def _draw_robot(ax, world, frame, trail, r0, c0, Wedge):
    cells = [world.to_cell(x, y) for x, y in trail]
    ax.plot([c - c0 for _, c in cells], [r - r0 for r, _ in cells],
            "-", color="#d93025", linewidth=1.2, alpha=0.8, zorder=5)
    # The wedge is drawn as an outline, not a fill. Filled, it claims everything in front
    # of the robot out to 5 m, including the far side of walls it cannot see through -
    # what the camera actually reached is the mapped floor underneath it.
    row, col = world.to_cell(frame["x"], frame["y"])
    heading = math.degrees(frame["yaw"])
    half = math.degrees(CAMERA_FOV) / 2
    ax.add_patch(Wedge((col - c0, row - r0), CAMERA_RANGE / world.resolution,
                       heading - half, heading + half, facecolor="none",
                       edgecolor="#f9ab00", linewidth=1.4, linestyle="--", alpha=0.85,
                       zorder=3))
    # A triangle pointing where the robot faces, not a disc. The objects an action has
    # touched are red discs, and a red disc for the robot on top of them is one dot too
    # many to tell apart - the shape is what separates the robot from what it is holding.
    ax.plot(col - c0, row - r0, marker=(3, 0, heading - 90), markersize=13,
            color="#202124", markeredgecolor="white", markeredgewidth=1.2, zorder=6,
            linestyle="none")


def _graph_layout(frames, focus=()):
    """One layout for every frame, from the final graph.

    Re-running a spring layout per frame makes the nodes jump and the animation
    unreadable. The point is to watch edges appear, not to watch the layout churn.
    """
    import networkx as nx

    final = frames[-1]
    shown = relevant(final, focus)
    graph = nx.Graph()
    graph.add_nodes_from(shown)
    graph.add_node(ROBOT)
    for edge_type, a, b in final["edges"]:
        if a in shown and (b in shown or edge_type == "room_inside"):
            graph.add_edge(a, b)
    return nx.spring_layout(graph, seed=7, k=1.1) if graph.number_of_nodes() else {}


def _draw_graph(ax, frame, positions, matplotlib, shown):
    """The world graph as it stands: objects, the rooms they are in, and the relations."""
    rooms = {b for t, a, b in frame["edges"] if t == "room_inside" and a in shown}
    drawn = {n for n in list(shown) + list(rooms) if n in positions}
    edges = [(t, a, b) for t, a, b in frame["edges"] if a in drawn and b in drawn]

    # `under(b, a)` is the converse of `on_top(a, b)` and lands on the same two nodes, so
    # one would hide the other. Drawing the wider one first leaves a blue core inside a
    # purple sheath, which reads as the pair it is.
    order = {"room_inside": 0, "next_to": 1, "under": 2, "object_inside": 3, "on_top": 4,
             "holding": 5}
    widths = {"room_inside": 0.9, "next_to": 0.9, "under": 4.0, "object_inside": 2.6,
              "on_top": 1.8, "holding": 2.6}
    for edge_type, a, b in sorted(edges, key=lambda e: order.get(e[0], 5)):
        ax.plot([positions[a][0], positions[b][0]], [positions[a][1], positions[b][1]],
                color=EDGE_COLORS.get(edge_type, "#888"), zorder=1,
                linewidth=widths.get(edge_type, 2.0),
                alpha=0.4 if edge_type in ("room_inside", "next_to") else 1.0)

    acting = {n for t, a, b in edges if t not in ("room_inside", "next_to")
              for n in (a, b)} | ({frame["held"]} if frame.get("held") else set())
    for name in sorted(drawn):
        x, y = positions[name]
        is_room, is_robot = name in rooms, name == ROBOT
        # The robot is a triangle here as it is on the map, so the node carrying the
        # `holding` edge is recognisable as the same thing driving around.
        marker = "^" if is_robot else ("s" if is_room else "o")
        ax.plot(x, y, marker,
                markersize=11 if is_robot else (9 if name in acting else (7 if is_room else 5)),
                color="#202124" if is_robot else
                      ("#ea4335" if name in acting else
                       ("#5f6368" if is_room else "#1a73e8")),
                markeredgecolor="white", markeredgewidth=0.6, zorder=2)
        ax.annotate(_short(name), (x, y),
                    fontsize=7.5 if (is_room or is_robot or name in acting) else 6.5,
                    xytext=(5, 5), textcoords="offset points", zorder=3,
                    fontweight="bold" if (name in acting or is_robot) else "normal",
                    color="#a50e0e" if name in acting else
                          ("#202124" if (is_room or is_robot) else "#5f6368"))

    counts = {}
    for edge_type, _, _ in frame["edges"]:
        counts[edge_type] = counts.get(edge_type, 0) + 1
    handles = [matplotlib.lines.Line2D([], [], color=colour, lw=2,
                                       label=f"{t} ({counts.get(t, 0)})")
               for t, colour in EDGE_COLORS.items()]
    # Below the axes, not inside them: the spring layout puts nodes wherever it likes and
    # a legend in a corner lands on top of one about half the time.
    ax.legend(handles=handles, fontsize=7, loc="upper center", ncol=5, frameon=False,
              bbox_to_anchor=(0.5, -0.01))
    ax.set_title(f"world graph — {len(drawn - rooms - {ROBOT})} objects, "
                 f"{len(edges)} relations", fontsize=10)
    ax.axis("off")
    if drawn:
        ax.margins(0.18)


def main():
    import argparse

    from floor_world import FloorWorld

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", default="figures/sim2d_trace.json")
    parser.add_argument("--out", default="figures/sim2d.gif")
    parser.add_argument("--fps", type=float, default=3)
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every nth 'moving' frame; action frames are always kept")
    parser.add_argument("--no-graph", action="store_true", help="map panel only")
    args = parser.parse_args()

    with open(args.trace) as f:
        saved = json.load(f)
    # Rebuild the world the run happened in, not the default one: a trace records which
    # categories were loaded because the picture is wrong without them.
    world = FloorWorld.load(saved["scene"], resolution=saved.get("resolution", 0.1),
                            categories=saved.get("categories", []),
                            trav_map=saved.get("trav_map", "no_door"))
    for name, record in (saved.get("objects") or {}).items():
        if name not in world.truth.objects and record.get("position"):
            world.add_object(name, record.get("category"), position=record["position"])

    path, n = render(world, saved["frames"], args.out, fps=args.fps, stride=args.stride,
                     graph_panel=not args.no_graph, focus=saved.get("focus") or ())
    print(f"wrote {path} ({n} frames)")


if __name__ == "__main__":
    main()
