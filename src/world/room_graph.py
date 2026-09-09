"""Build a room-adjacency graph for a BEHAVIOR-1K scene from ground-truth layout data.

Each scene ships a floor-plan segmentation pair under `layout/`:
  floor_insseg_0.png  room *instance* regions (one id per distinct room)
  floor_semseg_0.png  room *category* per pixel, indexing metadata/room_categories.txt

Two rooms are adjacent when their pixel regions touch or come within a few pixels of
each other on the floor plan. Dilating each region by a small radius before testing
overlap bridges the wall thickness that separates them in the raster.

Why not derive adjacency from door objects: only 3 of 51 scenes come out fully
connected that way, because archways and open-plan boundaries carry no door object.
Pixel adjacency recovers the real floor-plan topology.

The semantic id -> room category mapping is `room_categories.txt[id - 1]`, verified
against the `in_rooms` annotations of several scenes.
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


import json
import os
from collections import defaultdict

import numpy as np

DEFAULT_DATASET = os.environ.get(
    "BEHAVIOR_ASSETS",
    "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/datasets/behavior-1k-assets",
)


def _load_room_categories(dataset_root):
    path = os.path.join(dataset_root, "metadata", "room_categories.txt")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _dilate(mask, radius):
    """Binary dilation by a square structuring element, via cumulative shifts."""
    out = mask.copy()
    for _ in range(radius):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        out = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
        )
    return out


def build_room_graph(
    scene,
    dataset_root=DEFAULT_DATASET,
    dilate_radius=4,
    min_pixels=50,
    max_dimension=2048,
    min_boundary_pixels=40,
):
    """Return the room graph for one scene.

    {
      "scene": str,
      "rooms": {room_id: {"room_type": str, "pixels": int}},
      "edges": [[room_id_a, room_id_b], ...],
    }

    Room ids are `<room_type>_<n>`, numbered per type in descending area order so the
    naming is deterministic. They are layout-derived and need not coincide with the
    `in_rooms` instance names in the scene JSON.
    """
    from PIL import Image

    # These are our own dataset's floor plans, not untrusted input; a few scenes have
    # very large rasters that trip PIL's decompression-bomb guard.
    Image.MAX_IMAGE_PIXELS = None

    layout = os.path.join(dataset_root, "scenes", scene, "layout")
    ins_img = Image.open(os.path.join(layout, "floor_insseg_0.png"))
    sem_img = Image.open(os.path.join(layout, "floor_semseg_0.png"))

    # Dilation cost grows with area, so downsample very large plans. NEAREST keeps
    # region ids intact; the scale factor is applied to the dilation radius so the
    # adjacency threshold stays constant in real-world terms.
    scale = 1
    if max(ins_img.size) > max_dimension:
        scale = max(ins_img.size) / max_dimension
        new_size = (max(1, int(ins_img.width / scale)), max(1, int(ins_img.height / scale)))
        ins_img = ins_img.resize(new_size, Image.NEAREST)
        sem_img = sem_img.resize(new_size, Image.NEAREST)
        dilate_radius = max(1, int(round(dilate_radius / scale)))
        # min_pixels is an area threshold, so it scales quadratically; the boundary
        # contact threshold is a length, so it scales linearly.
        min_pixels = max(4, int(min_pixels / (scale**2)))
        min_boundary_pixels = max(4, int(min_boundary_pixels / scale))

    ins = np.array(ins_img)
    sem = np.array(sem_img)
    categories = _load_room_categories(dataset_root)

    region_ids = [int(v) for v in np.unique(ins) if v != 0]

    regions = {}
    for rid in region_ids:
        mask = ins == rid
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue  # speckle, not a room
        # The region's room category is the majority semantic label inside it.
        sem_vals = sem[mask]
        sem_vals = sem_vals[sem_vals != 0]
        if len(sem_vals) == 0:
            continue
        sem_id = int(np.bincount(sem_vals).argmax())
        if not 0 < sem_id <= len(categories):
            continue
        ys, xs = np.nonzero(mask)
        regions[rid] = {
            "room_type": categories[sem_id - 1],
            "pixels": n_pixels,
            "mask": mask,
            # Centroid in original image coordinates, so graph nodes can be drawn at
            # their true positions on the floor plan.
            "centroid": [float(xs.mean() * scale), float(ys.mean() * scale)],
            "segment_id": rid,
        }

    # Deterministic names: largest room of each type gets index 0.
    by_type = defaultdict(list)
    for rid, info in regions.items():
        by_type[info["room_type"]].append(rid)
    names = {}
    for room_type, rids in by_type.items():
        for i, rid in enumerate(sorted(rids, key=lambda r: -regions[r]["pixels"])):
            names[rid] = f"{room_type}_{i}"

    # Adjacency: dilate each region, then measure how much the dilations overlap.
    #
    # Requiring a substantial overlap - not merely any contact - is what separates a
    # real shared wall or doorway from two rooms that meet at a single corner. With a
    # bare `.any()` test, rooms touching diagonally at one pixel register as connected,
    # which produces edges a robot could not actually traverse.
    dilated = {rid: _dilate(info["mask"], dilate_radius) for rid, info in regions.items()}
    min_contact = max(min_boundary_pixels, dilate_radius * 2)
    edges = set()
    rids = sorted(regions)
    for i, a in enumerate(rids):
        for b in rids[i + 1 :]:
            contact = int((dilated[a] & dilated[b]).sum())
            if contact >= min_contact:
                edges.add(tuple(sorted((names[a], names[b]))))

    return {
        "scene": scene,
        "rooms": {
            names[rid]: {
                "room_type": info["room_type"],
                "pixels": info["pixels"],
                "centroid": info["centroid"],
                "segment_id": info["segment_id"],
            }
            for rid, info in regions.items()
        },
        "edges": [list(e) for e in sorted(edges)],
    }


def connected_components(graph):
    """List of room-id sets, one per connected component."""
    adj = defaultdict(set)
    for a, b in graph["edges"]:
        adj[a].add(b)
        adj[b].add(a)
    seen, components = set(), []
    for room in sorted(graph["rooms"]):
        if room in seen:
            continue
        stack, comp = [room], set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(adj[u] - seen)
        components.append(comp)
    return components


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", help="single scene; omit to build all scenes")
    parser.add_argument("--dataset-root", default=os.environ.get("BEHAVIOR_ASSETS", DEFAULT_DATASET))
    parser.add_argument("--dilate-radius", type=int, default=4)
    parser.add_argument("--out", default="data/room_graphs.json")
    args = parser.parse_args()

    scenes_dir = os.path.join(args.dataset_root, "scenes")
    scenes = [args.scene] if args.scene else sorted(os.listdir(scenes_dir))

    graphs, disconnected = {}, []
    for scene in scenes:
        layout = os.path.join(scenes_dir, scene, "layout", "floor_insseg_0.png")
        if not os.path.exists(layout):
            continue
        g = build_room_graph(scene, args.dataset_root, args.dilate_radius)
        graphs[scene] = g
        comps = connected_components(g)
        if len(comps) > 1:
            disconnected.append((scene, len(g["rooms"]), len(comps)))
        if args.scene:
            print(f"scene: {scene}")
            print(f"rooms ({len(g['rooms'])}):")
            for name, info in sorted(g["rooms"].items()):
                print(f"  {name:24s} {info['pixels']:6d} px")
            print(f"edges ({len(g['edges'])}):")
            for a, b in g["edges"]:
                print(f"  {a} <-> {b}")

    if not args.scene:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(graphs, f, indent=1)
        n_rooms = sum(len(g["rooms"]) for g in graphs.values())
        n_edges = sum(len(g["edges"]) for g in graphs.values())
        print(f"built {len(graphs)} scene graphs: {n_rooms} rooms, {n_edges} edges")
        print(f"fully connected: {len(graphs) - len(disconnected)}/{len(graphs)}")
        for scene, n, c in disconnected[:10]:
            print(f"  {scene:28s} {n:2d} rooms in {c} components")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
