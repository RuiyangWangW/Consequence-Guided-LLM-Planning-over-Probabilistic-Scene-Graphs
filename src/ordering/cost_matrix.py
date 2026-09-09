#!/usr/bin/env python3
"""Room-to-room navigation cost, and the cost of searching a room.

GAVEL's ordering stage needs two numbers the rest of the pipeline never had to compute:

    D(r_i, r_j)   what it costs to drive from one room to another
    s(r)          what it costs to search one room completely

Both are measured on the same eroded traversability map the simulator drives on, not on
the room graph. The room graph says which rooms adjoin; it does not say how far apart they
are, and in half the scenes it disagrees with A* about whether they are connected at all.
A cost model built on edge counts would rank an ordering by a distance the robot never
travels.

    python cost_matrix.py --scene Beechwood_0_int
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


import argparse
import json

import numpy as np

from floor_world import DEFAULT_ROBOT_RADIUS, FloorWorld, astar

# A room is searched by driving to frontiers until nothing unseen remains, so its cost
# scales with area rather than with its diameter. This converts traversable cells into a
# comparable distance: the robot must cover the room with a sensor footprint, so the
# distance travelled is roughly the area divided by the width of the swathe it clears.
SWATHE_M = 1.2


def _centroid_cell(world, room):
    """A traversable cell near the middle of a room - where 'being in the room' means."""
    mask = world.room_mask(room) & world.traversable(DEFAULT_ROBOT_RADIUS)[0]
    cells = np.argwhere(mask)
    if not len(cells):
        return None
    middle = cells.mean(axis=0)
    # The centroid of an L-shaped room can fall outside it, so take the traversable cell
    # closest to it rather than the centroid itself.
    return tuple(cells[np.argmin(((cells - middle) ** 2).sum(axis=1))])


def search_costs(world, rooms=None):
    """s(r): metres the robot must drive to sweep each room, from its traversable area."""
    mask, _ = world.traversable(DEFAULT_ROBOT_RADIUS)
    out = {}
    for room in (rooms or world.rooms):
        cells = int((world.room_mask(room) & mask).sum())
        area = cells * (world.resolution ** 2)
        out[room] = area / SWATHE_M
    return out


def distance_matrix(world, rooms=None):
    """D(r_i, r_j) in metres, by A* on the eroded map between room centroids.

    Unreachable pairs are `inf` rather than a large number: a plan that needs one is not
    expensive, it is impossible, and the ordering stage must not be able to buy its way
    through a wall by paying enough.
    """
    rooms = sorted(rooms or world.rooms)
    mask, _ = world.traversable(DEFAULT_ROBOT_RADIUS)
    cells = {r: _centroid_cell(world, r) for r in rooms}
    out = {}
    for a in rooms:
        for b in rooms:
            if a == b:
                out[(a, b)] = 0.0
                continue
            if cells[a] is None or cells[b] is None:
                out[(a, b)] = float("inf")
                continue
            route = astar(mask, cells[a], cells[b])
            out[(a, b)] = (len(route) - 1) * world.resolution if route else float("inf")
    return out


def build(scene, graphs_path="data/room_graphs.json"):
    """Both tables for one scene, keyed by room id."""
    world = FloorWorld.load(scene, graphs_path=graphs_path, categories=None)
    rooms = sorted(world.rooms)
    return {"scene": scene, "rooms": rooms,
            "distance": {f"{a}|{b}": v for (a, b), v in distance_matrix(world, rooms).items()},
            "search": search_costs(world, rooms)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--json")
    args = parser.parse_args()
    table = build(args.scene)
    d = table["distance"]
    finite = [v for v in d.values() if v != float("inf")]
    print(f"{args.scene}: {len(table['rooms'])} rooms")
    print(f"  distances: {len(finite)}/{len(d)} reachable, "
          f"mean {sum(finite)/max(len(finite),1):.1f} m, max {max(finite):.1f} m")
    for r, s in sorted(table["search"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"  search {r:22s} {s:6.1f} m")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(table, handle, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    raise SystemExit(main())
