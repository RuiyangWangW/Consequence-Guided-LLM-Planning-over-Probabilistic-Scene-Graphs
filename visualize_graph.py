"""Render a scene's floor-plan segmentation next to the room graph derived from it.

Side-by-side visual check that the graph matches the ground-truth layout:
  left   the room-instance segmentation, each region tinted by room type and labeled
  right  the derived graph, nodes drawn at each room's true centroid so the topology
         can be compared against the plan directly

Room colors are keyed to the room *type* from a fixed global palette, so the same
category reads the same across every scene and figures can be compared side by side.

    python visualize_graph.py --scene Rs_int
    python visualize_graph.py --scene Rs_int --overlay      # graph on top of the plan
    python visualize_graph.py --all                         # every scene in the dataset
"""

import argparse
import os

import numpy as np

from room_graph import DEFAULT_DATASET, build_room_graph, connected_components

# Room types present in BEHAVIOR-1K, ordered by how often they occur across the 51
# scenes. Fixing the order pins each type to a stable color: any type at the same
# index gets the same color in every figure, which is what makes scenes comparable.
ROOM_TYPE_ORDER = [
    "bathroom", "corridor", "storage_room", "bedroom", "living_room", "closet",
    "private_office", "kitchen", "dining_room", "childs_room", "meeting_room",
    "garden", "utility_room", "empty_room", "shared_office", "playroom",
    "grocery_store", "bar", "conference_hall", "copy_room", "lobby", "garage",
    "sauna", "entryway", "locker_room", "break_room", "phone_room", "pantry_room",
    "staircase", "gym", "spa", "television_room", "exercise_room", "hammam",
    "biology_lab", "chemistry_lab", "computer_lab", "infirmary", "classroom",
]


def _type_colors():
    """Stable color per room *type*, consistent across all scenes.

    tab20 and tab20b together give 40 distinguishable hues, one more than the 39
    room types in the dataset. Any type not in ROOM_TYPE_ORDER (a dataset update)
    falls back to a hash of its name so it still gets a deterministic color.
    """
    import matplotlib.pyplot as plt

    palette = [plt.get_cmap("tab20")(i / 20.0) for i in range(20)]
    palette += [plt.get_cmap("tab20b")(i / 20.0) for i in range(20)]
    colors = {t: palette[i] for i, t in enumerate(ROOM_TYPE_ORDER)}
    return colors, palette


def _color_for(room_type, colors, palette):
    if room_type not in colors:
        colors[room_type] = palette[hash(room_type) % len(palette)]
    return colors[room_type]


def _short(name):
    """Compact node label: 'living_room_2' -> 'living\nroom 2'."""
    base, _, idx = name.rpartition("_")
    return f"{base.replace('_', ' ')}\n{idx}" if base else name


def render(scene, dataset_root=DEFAULT_DATASET, out_path=None, dilate_radius=4, overlay=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None

    graph = build_room_graph(scene, dataset_root, dilate_radius)
    rooms = graph["rooms"]
    if not rooms:
        raise SystemExit(f"no rooms found for scene {scene}")

    layout = os.path.join(dataset_root, "scenes", scene, "layout")
    ins = np.array(Image.open(os.path.join(layout, "floor_insseg_0.png")))

    base_colors, palette = _type_colors()
    colors = {n: _color_for(i["room_type"], base_colors, palette) for n, i in rooms.items()}

    # Paint each room region with its type color; everything else stays white.
    rgb = np.ones(ins.shape + (3,), dtype=float)
    for name, info in rooms.items():
        rgb[ins == info["segment_id"]] = colors[name][:3]

    # Crop to the occupied area so small plans are not lost in whitespace. Only the
    # regions that became rooms count: discarded speckle can sit far from the plan
    # and would otherwise stretch the crop across mostly empty pixels.
    kept = np.isin(ins, [i["segment_id"] for i in rooms.values()])
    ys, xs = np.nonzero(kept)
    pad = max(10, int(0.02 * max(ins.shape)))
    y0, y1 = max(0, ys.min() - pad), min(ins.shape[0], ys.max() + pad)
    x0, x1 = max(0, xs.min() - pad), min(ins.shape[1], xs.max() + pad)

    # Match figure proportions to the plan's aspect ratio so wide or tall layouts
    # are not squeezed into a square panel.
    h, w = y1 - y0, x1 - x0
    panel_w = 9.0
    panel_h = float(np.clip(panel_w * h / max(w, 1), 4.5, 14.0))

    n_panels = 1 if overlay else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, panel_h))
    axes = [axes] if n_panels == 1 else list(axes)

    n_comp = len(connected_components(graph))
    status = "connected" if n_comp == 1 else f"{n_comp} components"

    # Label size shrinks as rooms get denser, so 25-room scenes stay readable.
    lab = float(np.clip(90.0 / max(len(rooms), 1), 4.5, 8.5))

    # --- left panel: segmentation ---
    ax = axes[0]
    ax.imshow(rgb[y0:y1, x0:x1], interpolation="nearest")
    for name, info in rooms.items():
        cx, cy = info["centroid"]
        ax.text(
            cx - x0, cy - y0, _short(name),
            ha="center", va="center", fontsize=lab, weight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
        )
    ax.set_title(f"{scene}\nroom segmentation (ground truth)", fontsize=11)
    ax.axis("off")

    # --- right panel (or overlay): derived graph ---
    ax = axes[0] if overlay else axes[1]
    if not overlay:
        # Faint plan behind the graph for spatial reference.
        faded = 1.0 - (1.0 - rgb[y0:y1, x0:x1]) * 0.18
        ax.imshow(faded, interpolation="nearest")

    for a, b in graph["edges"]:
        ax_, ay_ = rooms[a]["centroid"]
        bx_, by_ = rooms[b]["centroid"]
        ax.plot([ax_ - x0, bx_ - x0], [ay_ - y0, by_ - y0],
                color="0.25", lw=1.6, zorder=2, alpha=0.85)

    # Past ~10 rooms a full name inside a node marker is unreadable, so nodes get a
    # number and the legend carries the key. Sparse scenes keep the inline names,
    # which are easier to read when they fit.
    dense = len(rooms) > 10
    order = sorted(rooms)
    node_s = float(np.clip(2600.0 / max(len(rooms), 1), 150.0, 460.0))
    for i, name in enumerate(order, 1):
        cx, cy = rooms[name]["centroid"]
        ax.scatter(cx - x0, cy - y0, s=node_s, color=colors[name],
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.text(cx - x0, cy - y0, str(i) if dense else _short(name),
                ha="center", va="center", fontsize=(7.5 if dense else lab * 0.8),
                weight="bold", zorder=4,
                color="white" if dense and sum(colors[name][:3]) < 1.5 else "black")

    title = f"derived room graph\n{len(rooms)} rooms, {len(graph['edges'])} edges ({status})"
    ax.set_title(f"{scene}\n{title.split(chr(10))[1]}" if overlay else title, fontsize=11)
    ax.axis("off")

    # Legend: on dense scenes it keys the numbered nodes, otherwise it names the
    # room types present in this scene.
    if dense:
        handles = [
            Line2D([], [], marker="o", ls="", markersize=7, markeredgecolor="black",
                   markeredgewidth=0.6, color=colors[name],
                   label=f"{i}  {name.replace('_', ' ')}")
            for i, name in enumerate(order, 1)
        ]
        ncol = min(6, max(3, (len(handles) + 3) // 4))
    else:
        present = sorted({i["room_type"] for i in rooms.values()})
        handles = [
            Line2D([], [], marker="o", ls="", markersize=7, markeredgecolor="black",
                   markeredgewidth=0.6, color=_color_for(t, base_colors, palette),
                   label=t.replace("_", " "))
            for t in present
        ]
        ncol = min(len(present), 8)
    # Lay out the panels first, then reserve a strip under them for the legend, so a
    # tall multi-row key never overlaps the floor plan.
    fig.tight_layout()
    n_rows = -(-len(handles) // ncol)
    strip = min(0.30, 0.035 + 0.028 * n_rows)
    # `top` leaves room for the two-line panel titles, which the reserved-strip
    # layout below would otherwise push off the canvas.
    fig.subplots_adjust(bottom=strip, top=0.90)
    fig.legend(
        handles=handles, loc="lower center", ncol=ncol,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    out_path = out_path or f"figures/{scene}_graph.png"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=110, facecolor="white")
    plt.close(fig)
    return out_path, graph, n_comp


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", help="single scene to render")
    parser.add_argument("--all", action="store_true", help="render every scene in the dataset")
    parser.add_argument("--dataset-root", default=os.environ.get("BEHAVIOR_ASSETS", DEFAULT_DATASET))
    parser.add_argument("--dilate-radius", type=int, default=4)
    parser.add_argument("--overlay", action="store_true", help="draw the graph on the plan itself")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--out", help="output path (single scene only)")
    args = parser.parse_args()

    if not args.scene and not args.all:
        parser.error("pass --scene SCENE or --all")

    if args.scene:
        out = args.out or os.path.join(args.out_dir, f"{args.scene}_graph.png")
        path, graph, n_comp = render(
            args.scene, args.dataset_root, out, args.dilate_radius, args.overlay
        )
        status = "connected" if n_comp == 1 else f"{n_comp} components"
        print(f"scene: {args.scene}")
        print(f"  rooms: {len(graph['rooms'])}  edges: {len(graph['edges'])}  ({status})")
        for a, b in graph["edges"]:
            print(f"    {a} <-> {b}")
        print(f"wrote {path}")
        return

    scenes_dir = os.path.join(args.dataset_root, "scenes")
    scenes = sorted(
        s for s in os.listdir(scenes_dir)
        if os.path.exists(os.path.join(scenes_dir, s, "layout", "floor_insseg_0.png"))
    )
    disconnected = []
    for i, scene in enumerate(scenes, 1):
        out = os.path.join(args.out_dir, f"{scene}_graph.png")
        _, graph, n_comp = render(scene, args.dataset_root, out, args.dilate_radius, args.overlay)
        flag = "" if n_comp == 1 else f"  <- {n_comp} components"
        print(f"[{i:2d}/{len(scenes)}] {scene:28s} "
              f"{len(graph['rooms']):2d} rooms {len(graph['edges']):3d} edges{flag}")
        if n_comp > 1:
            disconnected.append((scene, len(graph["rooms"]), n_comp))
    print(f"\nwrote {len(scenes)} figures to {args.out_dir}/")
    print(f"fully connected: {len(scenes) - len(disconnected)}/{len(scenes)}")
    for scene, n, c in disconnected:
        print(f"  {scene:28s} {n:2d} rooms in {c} components")


if __name__ == "__main__":
    main()
