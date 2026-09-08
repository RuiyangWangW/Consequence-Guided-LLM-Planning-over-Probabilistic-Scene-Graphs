#!/usr/bin/env python
"""Which stance does the navigation pick? Answered from the map, without the simulator.

`_navigate_near` gathers the reachable floor cells near an object, orders them, and drives
to the first that survives a collision check. This reproduces that selection offline and
compares the two orderings:

    by seed    - distance from the cell the BFS first reached, which is what the code did
    by object  - distance from the object itself

The collision check needs CuRobo, so it is stood in for by a threshold: `--fits`, the
distance at which the robot actually fits, measured with `test_primitives.py --probe`. For
house_single_floor's countertop that is 0.70 m for both the potato and the plate, while the
map opens up at 0.55 m - the robot's body against the worktop is stricter than its
footprint against the floor plan.

    python test_stance_order.py
"""

import argparse
import collections
import math

import numpy as np

import scene_setup as S


def eroded_map(scene, radius):
    eroded, size = S.trav_map(scene, radius)
    return eroded, size, np.argwhere(eroded > 0)


def bfs_seed(eroded, labels, region, start_rc):
    """First free cell reached from `start_rc`, eight-connected - what the code does."""
    rows, cols = eroded.shape
    seen = np.zeros(eroded.shape, dtype=bool)
    r0 = int(min(max(start_rc[0], 0), rows - 1))
    c0 = int(min(max(start_rc[1], 0), cols - 1))
    queue = collections.deque([(r0, c0)])
    seen[r0, c0] = True
    steps = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while queue:
        r, c = queue.popleft()
        if eroded[r, c] > 0 and int(labels[r, c]) == region:
            return r, c
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc]:
                seen[nr, nc] = True
                queue.append((nr, nc))
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default="house_single_floor")
    ap.add_argument("--radius", type=float, default=0.77,
                    help="erosion radius the robot navigates on (default 0.77)")
    ap.add_argument("--fits", type=float, default=0.70,
                    help="distance at which the robot actually fits, from --probe")
    ap.add_argument("--keep", type=int, default=120,
                    help="how many nearest cells are collision-checked (CELL_CANDIDATES)")
    args = ap.parse_args()

    import cv2

    eroded, size, free = eroded_map(args.scene, args.radius)
    _, labels = cv2.connectedComponents((eroded > 0).astype(np.uint8), connectivity=4)
    # The robot's region: where it stands when it starts, near the middle of the map.
    origin = int(labels[int(size / 2), int(size / 2)])
    xs = (free[:, 1] - size / 2.0) * S.RES
    ys = (free[:, 0] - size / 2.0) * S.RES
    in_region = labels[free[:, 0], free[:, 1]] == origin
    print(f"robot region {origin}: {int(in_region.sum())} of {len(free)} free cells "
          f"are actually reachable")
    print(f"{args.scene}: eroded by {args.radius} m, {len(free)} free cells; "
          f"the robot fits from {args.fits} m out\n")

    # Where the objects end up, from the runs.
    for name, (ox, oy) in (("potato", (4.37, -0.95)), ("plate", (4.35, -0.05))):
        seed = bfs_seed(eroded, labels, origin, S.to_px((ox, oy), size))
        sx = (seed[1] - size / 2.0) * S.RES
        sy = (seed[0] - size / 2.0) * S.RES

        to_obj = np.hypot(xs - ox, ys - oy)
        to_seed = np.hypot(xs - sx, ys - sy)

        rows = []
        for label, order in (("by seed", np.argsort(to_seed)),
                             ("by object", np.argsort(to_obj))):
            considered = order[:args.keep]
            # Stand in for the collision check: anything closer than `fits` is rejected.
            # Only cells the robot can actually reach, and only those it fits in.
            usable = [i for i in considered
                      if in_region[i] and to_obj[i] >= args.fits]
            chosen = to_obj[usable[0]] if usable else float("nan")
            rows.append((label, chosen, len(considered), len(usable)))

        print(f"{name}: BFS seed {np.hypot(sx - ox, sy - oy):.2f} m from it")
        for label, chosen, n, u in rows:
            print(f"   ordered {label:9s} -> stands {chosen:.2f} m away "
                  f"({u} of the {n} nearest are usable)")
        print(f"   best possible {args.fits:.2f} m\n")


if __name__ == "__main__":
    main()
