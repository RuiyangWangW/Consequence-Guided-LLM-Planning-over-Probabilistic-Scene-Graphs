#!/usr/bin/env python
"""Measure how far each openable object's door sweeps, offline from the asset metadata.

Run this once, paste the table it prints into DOOR_SWING in primitive_patches.py. It
never launches the simulator - everything here comes out of each model's
`misc/metadata.json`, which is ground truth shipped with the dataset.

    python door_swing.py                 # the models the test scenes use
    python door_swing.py fridge/dszchb   # a specific one

How the radius is derived
-------------------------
`link_bounding_boxes[<link>]["visual"]["axis_aligned"]["transform"]` places the door
panel's bounding box in the link's own frame, and the link's origin sits on the hinge.
So the translation of that transform is the vector from the hinge to the centre of the
panel, and which way it points says how the door is hung:

    offset mostly horizontal -> the hinge is a vertical line down one side of the panel.
                                A side-hung door, like a fridge. Opening sweeps the
                                panel's *width* across the floor; its height stays put.

    offset mostly vertical   -> the hinge is the panel's bottom edge, lying flat.
                                A bottom-hung door, like an oven. Opening drops the panel
                                outward, projecting its *height* into the room.

Either way the panel centre is half a panel from the hinge, so twice the dominant offset
is the radius swept. Checked against the panel extents below, the two agree to within the
hinge's inset from the edge, and every radius comes out smaller than the object's own
footprint - which is the bound a door swing can never exceed.
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
import math
import os
import sys

DATASET = os.environ.get(
    "BEHAVIOR_ASSETS",
    "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/datasets/behavior-1k-assets/objects",
)

# The openable objects the two viable scenes use, from SCENE_SETUP in test_primitives.py.
DEFAULT_MODELS = [
    ("fridge", "dszchb"), ("oven", "ffitak"),     # house_single_floor
    ("fridge", "xyejdx"), ("oven", "fexqbj"),     # Pomaria_1_int
]

# A panel whose centre sits on its own hinge is not a door - it is a shelf, a rack or a
# pane of glass that happens to hang off a joint. Below this the offset is measurement
# noise and doubling it says nothing.
MIN_OFFSET = 0.05


def door_links(meta):
    """Which links are doors, preferring the dataset's own 'openable' tag."""
    tagged = [name for name, tags in (meta.get("link_tags") or {}).items()
              if "openable" in (tags or [])]
    if tagged:
        return tagged
    # oven/ffitak carries no tags. Fall back to every non-base link and let MIN_OFFSET
    # throw out the racks and the glass, whose centres sit on their origins.
    return [n for n in meta["link_bounding_boxes"] if n != "base_link"]


def measure(category, model):
    path = os.path.join(DATASET, category, model, "misc", "metadata.json")
    if not os.path.exists(path):
        return None, f"no metadata at {path}"
    meta = json.load(open(path))
    footprint = math.hypot(*meta["bbox_size"][:2])

    out = []
    for link in door_links(meta):
        box = meta["link_bounding_boxes"][link]["visual"]["axis_aligned"]
        tf, extent = box["transform"], box["extent"]
        ox, oy, oz = tf[0][3], tf[1][3], tf[2][3]
        horizontal, vertical = math.hypot(ox, oy), abs(oz)

        if max(horizontal, vertical) < MIN_OFFSET:
            continue
        if horizontal > vertical:
            # Side-hung: the panel sweeps about a vertical line down one edge, so the floor
            # it covers is an arc of radius equal to its width. A disc is the superset.
            hung, shape = "side", ("disc", 2 * horizontal, 0.0)
            width = max(extent[0], extent[1])
        else:
            # Bottom-hung: the panel rotates about a horizontal line at its lower edge and
            # drops straight out into the room. It never sweeps sideways, so the floor it
            # covers is a rectangle - as deep as the panel is tall, as wide as the panel.
            depth = 2 * vertical
            span = max(extent[0], extent[1])
            hung, shape = "bottom", ("box", depth, span)
            width = extent[2]

        out.append({"link": link, "hung": hung, "shape": shape, "radius": shape[1],
                    "panel": width, "footprint": footprint,
                    "offset": (ox, oy, oz)})
    return out, None


def main(argv):
    if argv:
        models = [tuple(a.split("/", 1)) for a in argv]
    else:
        models = DEFAULT_MODELS

    table = {}
    for category, model in models:
        rows, err = measure(category, model)
        if err:
            print(f"{category}/{model}: {err}")
            continue
        print(f"{category}/{model}")
        for r in rows:
            ox, oy, oz = r["offset"]
            kind, a, b = r["shape"]
            flag = "" if a <= r["footprint"] else "   <-- EXCEEDS FOOTPRINT"
            print(f"    [{r['link']}] hinge from offset ({ox:+.3f},{oy:+.3f},{oz:+.3f}) "
                  f"-> {r['hung']}-hung")
            if kind == "disc":
                print(f"    {'':{len(r['link']) + 2}} disc, radius {a:.3f} m "
                      f"(panel {r['panel']:.3f} m, footprint {r['footprint']:.3f} m)"
                      f"{flag}")
            else:
                print(f"    {'':{len(r['link']) + 2}} box, {a:.3f} m out by {b:.3f} m "
                      f"wide (panel {r['panel']:.3f} m, footprint "
                      f"{r['footprint']:.3f} m){flag}")
            table.setdefault(f"{category}/{model}", []).append(
                (r["link"], kind, round(a, 3), round(b, 3)))
        print()

    print("# Paste into primitive_patches.py:")
    print("DOOR_SWING = {")
    for key, links in table.items():
        entries = ", ".join(f'("{n}", "{k}", {a}, {b})' for n, k, a, b in links)
        print(f'    "{key}": [{entries}],')
    print("}")


if __name__ == "__main__":
    main(sys.argv[1:])
