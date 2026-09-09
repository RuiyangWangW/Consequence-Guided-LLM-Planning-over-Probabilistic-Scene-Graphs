#!/usr/bin/env python3
"""Choose the furniture each suite uses, offline, before the simulator starts.

Ranking furniture at runtime meant every scene change cost a full Isaac startup just to
find out the chosen surface was unusable. Everything that decides usability is in the
shipped floor plan and scene JSON, so it can be settled in seconds instead:

  standing room   free floor in the annulus the robot stands in to reach the object,
                  on the traversability map eroded by the robot's footprint
  component       which connected region of floor the object sits beside; a support in a
                  different component from the fridge can never have its contents carried
                  there, however well the primitives work

`_erode_trav_map` uses radius = |chassis_extent_xy| / 2 + 0.2 with a square kernel, and
the exact value needs a loaded robot, so the sweep below reports several radii. A support
that wins at every radius is a safe pick; one that wins only at the smallest is not.

    python scene_setup.py Beechwood_0_int
    python scene_setup.py --all
"""

import os as _os, sys as _sys
# The repo root, found by marker rather than by counting parents, so these run from
# wherever they are filed. They import `world_graph` and `graph_machine` from there.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.exists(_os.path.join(_d, 'graph_machine.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)

import argparse
import glob
import json
import math
import os

import cv2
import numpy as np

SCENES = "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/datasets/behavior-1k-assets/scenes"
RES, DEFAULT_RES = 0.1, 0.01
# `_erode_trav_map` uses radius = |chassis_extent_xy| / 2 + 0.2. Measured on a loaded
# Tiago: extent (0.91, 1.043) m -> 0.892 m, a 9-pixel kernel at this map resolution. That
# is the number that decides which scenes are usable, and guessing it low (0.30 m) made
# every scene look fine when almost none are. The neighbours guard against a scene that
# only works at exactly one width.
TIAGO_EROSION_RADIUS = 0.892
RADII = (0.85, TIAGO_EROSION_RADIUS, 0.95)

# Only categories test_primitives.py actually loads.
SUPPORTS = ("countertop", "breakfast_table", "coffee_table")
TARGETS = ("fridge", "oven", "stove", "microwave")


def trav_map(scene, radius):
    img = cv2.imread(f"{SCENES}/{scene}/layout/floor_trav_0.png", cv2.IMREAD_GRAYSCALE)
    size = int(img.shape[0] * DEFAULT_RES / RES)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_NEAREST)
    px = int(math.ceil(radius / RES))          # same square kernel as _erode_trav_map
    return cv2.erode(img, np.ones((px, px), np.uint8)), size


def objects(scene):
    js = sorted(glob.glob(f"{SCENES}/{scene}/json/*.json"))
    js = [j for j in js if "best" in j] or js
    if not js:
        return {}
    data = json.load(open(js[0]))
    init = data["objects_info"]["init_info"]
    reg = data["state"]["registry"]["object_registry"]
    out = {}
    for name, info in init.items():
        cat = info.get("args", {}).get("category")
        st = reg.get(name)
        if cat and st and "root_link" in st:
            out[name] = (cat, st["root_link"]["pos"][:2])
    return out


def to_px(xy, size):
    return xy[1] / RES + size / 2.0, xy[0] / RES + size / 2.0      # row, col


def analyse(scene, radius):
    eroded, size = trav_map(scene, radius)
    _, labels = cv2.connectedComponents(eroded, connectivity=4)
    free = np.argwhere(eroded > 0)
    if len(free) == 0:
        return None

    def near(xy):
        row, col = to_px(xy, size)
        d = np.hypot(free[:, 0] - row, free[:, 1] - col)
        n = free[int(np.argmin(d))]
        return int(labels[n[0], n[1]]), (row, col)

    def room(xy, lo, hi):
        row, col = to_px(xy, size)
        d = np.hypot(free[:, 0] - row, free[:, 1] - col) * RES
        return int(((d >= lo) & (d <= hi)).sum())

    origin_comp = int(labels[int(size / 2), int(size / 2)])
    info = {"origin_comp": origin_comp, "n_comp": int(labels.max()),
            "supports": [], "targets": {}}
    for name, (cat, xy) in objects(scene).items():
        if cat in SUPPORTS:
            comp, _ = near(xy)
            info["supports"].append((room(xy, 0.5, 1.3), comp, name, cat))
        elif cat in TARGETS:
            comp, _ = near(xy)
            info["targets"].setdefault(cat, []).append((room(xy, 0.8, 1.6), comp, name))
    info["supports"].sort(reverse=True)
    for cat in info["targets"]:
        info["targets"][cat].sort(reverse=True)
    return info


def report(scene):
    print(f"\n=== {scene} " + "=" * (58 - len(scene)))
    per_radius = {}
    for r in RADII:
        a = analyse(scene, r)
        if a is None:
            print(f"  r={r}: no traversable floor at all")
            return None
        per_radius[r] = a

    base = per_radius[TIAGO_EROSION_RADIUS]   # decide on the measured footprint

    # The robot spawns at the world origin (nothing in the config or DummyTask moves it),
    # so anything it must touch has to be in the origin's own region of floor. This is the
    # criterion that actually rules scenes out: at Tiago's real width a single narrow
    # doorway seals the kitchen off, and no amount of pose sampling recovers from it.
    origin_region = base["origin_comp"]
    print(f"  components={base['n_comp']}  robot spawns at origin in component "
          f"{base['origin_comp']}")

    fridge = base["targets"].get("fridge", [])
    if not fridge:
        print("  no fridge -> scene cannot run this task")
        return None
    f_room, f_comp, f_name = fridge[0]
    if f_comp != origin_region:
        print(f"  fridge  {f_name:32s} c{f_comp} but robot spawns in c{origin_region}"
              f" -> unreachable, scene rejected")
        return None
    print(f"  fridge  {f_name:32s} room={f_room:4d} c{f_comp}")

    for cat in ("oven", "stove", "microwave"):
        for room_, comp, name in base["targets"].get(cat, [])[:1]:
            flag = "" if comp == f_comp else "   <- different component from fridge"
            print(f"  {cat:7s} {name:32s} room={room_:4d} c{comp}{flag}")

    print("  supports (must share the fridge's component):")
    chosen = None
    for room_, comp, name, cat in base["supports"][:8]:
        ok = comp == f_comp == origin_region and room_ > 0
        # require it to hold up at every radius, not just the forgiving one
        # Component labels are renumbered for every erosion radius, so a label from one
        # radius means nothing at another. Re-derive the fridge's component at each
        # radius and compare within that radius only.
        def shares_fridge_component(r):
            a = per_radius[r]
            fr = a["targets"].get("fridge")
            if not fr:
                return False
            fc = fr[0][1]
            return any(n == name and c == fc and rm > 0
                       for rm, c, n, _ in a["supports"])

        stable = all(shares_fridge_component(r) for r in RADII)
        mark = "  OK" if ok and stable else ("  unstable" if ok else "  skip")
        print(f"    {name:32s} room={room_:4d} c{comp}{mark}")
        if ok and stable and chosen is None:
            chosen = name
    if chosen is None:
        print("  -> no usable support; do not run this scene")
        return None
    print(f"  -> support = {chosen}")
    chosen_cat = next(c for _, _, n, c in base["supports"] if n == chosen)
    return {"support": chosen, "support_category": chosen_cat, "fridge": f_name,
            "oven": (base["targets"].get("oven") or base["targets"].get("stove")
                     or [(0, 0, None)])[0][2]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default="Beechwood_0_int")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    names = sorted(os.listdir(SCENES)) if args.all else [args.scene]
    picked = {}
    for s in names:
        if not os.path.isdir(f"{SCENES}/{s}"):
            continue
        try:
            r = report(s)
        except Exception as e:
            print(f"\n=== {s}: FAILED {type(e).__name__}: {e}")
            continue
        if r:
            picked[s] = r
    print("\n\nSCENE_SETUP = " + json.dumps(picked, indent=4).replace('"', '"'))
