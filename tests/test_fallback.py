#!/usr/bin/env python3
"""The belief was wrong: does the robot recover, and does the graph stop lying?

Extraction gets locations wrong in two ways, and the pipeline has to survive both:

    the stated ROOM is wrong      "the office cabinet", but it is in the kitchen
    the stated RELATION is wrong  "the potato on the counter", but it is in the fridge

Neither is exotic - the extractor is a 1.7B model reading free text - so a pipeline that
only works when extraction is right is a pipeline that mostly does not work. What makes
them survivable is that a stated location is the *first* candidate, never the only one:
`scene_graph.populate` keeps the RSN's whole ranking behind it, the searcher walks down
that ranking, and each miss retracts the belief instead of leaving the graph asserting a
room the robot has already swept.

    python test_fallback.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import sys

from floor_world import FloorWorld
from sim2d import Sim2D

SCENE = "Beechwood_0_int"
CATEGORIES = ["countertop", "fridge", "oven"]
PASS, FAIL = [], []


def case(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def main():
    # A potato really on a kitchen countertop.
    world = FloorWorld.load(SCENE, categories=CATEGORIES)
    counter = world.truth.by_category("countertop")[0]
    world.add_object("potato", "potato", on_top=counter)
    truth_room = world.truth.room_of("potato")

    wrong = next(r for r in world.truth.rooms if r != truth_room and "kitchen" not in r)
    print(f"potato is really in {truth_room}; the belief will say {wrong}\n")

    # --- a wrong stated room, with the ranking behind it -----------------------------
    sim = Sim2D(world, start_room="living_room_0", verbose=False,
                room_hints={"potato": [wrong, truth_room]})
    found, driven, _, problem = sim.navigate_to("potato")
    case("a wrong stated room is recovered from",
         found is not None and problem is None, problem or f"found after {driven:.1f} m")
    case("the ruled-out room is retracted from the belief graph",
         sim.graph.is_ruled_out("potato", wrong),
         f"ruled_out={sim.graph.ruled_out.get('potato')}")
    case("the graph no longer asserts the object is in the wrong room",
         ("room_inside", "potato", wrong) not in sim.graph.edges)

    # --- a wrong stated room with NO fallback: this is what used to happen ------------
    world2 = FloorWorld.load(SCENE, categories=CATEGORIES)
    world2.add_object("potato", "potato", on_top=world2.truth.by_category("countertop")[0])
    only = Sim2D(world2, start_room="living_room_0", verbose=False,
                 room_hints={"potato": [wrong]})
    found2, _, _, problem2 = only.navigate_to("potato")
    case("with a single candidate the same search fails (the bug this fixes)",
         found2 is None and problem2 is not None, str(problem2)[:70])

    # --- the room is searched once, not once per NAVIGATE_TO -------------------------
    world3 = FloorWorld.load(SCENE, categories=CATEGORIES)
    world3.add_object("potato", "potato", on_top=world3.truth.by_category("countertop")[0])
    twice = Sim2D(world3, start_room="living_room_0", verbose=False,
                  room_hints={"potato": [wrong, truth_room]})
    twice.navigate_to("potato")
    twice.graph.objects.pop("potato", None)          # forget we found it, keep the evidence
    case("a ruled-out room is not searched again",
         wrong not in twice._candidate_rooms("potato"),
         str(twice._candidate_rooms("potato")))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
