"""Map BEHAVIOR's fine-grained object categories onto everyday nouns.

BEHAVIOR splits common nouns into material/mounting/context variants: nine kinds of
sink, four kinds of cabinet, ten kinds of light. This module collapses variants that
share an everyday name and a functional role.

Merging is a *training-label* decision, not a query-vocabulary one - the RSN's text
encoder already accepts arbitrary object names at query time. It matters because
splitting one everyday noun across many rare variants fragments the supervision: the
model sees `pedestal_sink` in 16 room instances instead of `sink` in 84. Training the
RSN without merging measurably hurts (AUC 0.879 vs 0.904, AP 0.535 vs 0.602).

Merging is by explicit table, not by substring matching on names. Names are misleading
here in both directions:
  - `periodic_table` and `pool_table` contain "table" but are not furniture tables.
  - `hall_tree` contains "tree" but is a coatrack, an indoor object.
  - `bench_press_machine` contains "bench" but is gym equipment.
Substring rules would silently merge these and corrupt the prior, so anything not
listed below passes through unchanged.

Merges only collapse variants that a filter should treat interchangeably. Distinctions
that carry real placement information are preserved:
  - `bottom_cabinet` vs `top_cabinet` stay apart (floor vs wall mounted).
  - `freezer`, `wine_fridge`, `display_fridge` stay apart from `fridge`; a display
    fridge implies a shop, not a kitchen.
  - `urinal` is never merged into `toilet` (it implies a public restroom).
  - `bathtub`, `shower`, and `shower_stall` stay apart.

Run `python category_merge.py` to print every group that actually merges, with counts.
"""

import os as _os, sys as _sys
# Run from anywhere. Find the repo root by marker, put it on the import path, and make
# it the working directory - every path in this file is written 'data/...', so without
# the chdir a build invoked from inside its own folder would quietly write benchmark/data/.
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
_os.chdir(_d)


# fine-grained category -> everyday noun.
MERGE_MAP = {
    # --- sinks: nine variants differing only in mounting and setting ---
    "commercial_kitchen_sink": "sink",
    "drop_in_sink": "sink",
    "furniture_sink": "sink",
    "multi_station_furniture_sink": "sink",
    "multi_station_tabletop_sink": "sink",
    "multi_station_wall_mounted_sink": "sink",
    "pedestal_sink": "sink",
    "tabletop_sink": "sink",
    "wall_mounted_sink": "sink",

    # --- chairs: material/style variants of a seat for one person ---
    "armchair": "chair",
    "eames_chair": "chair",
    "garden_chair": "chair",
    "rocking_chair": "chair",
    "straight_chair": "chair",
    "swivel_chair": "chair",
    "music_stool": "chair",
    "taboret": "chair",
    # NOTE: `bench`, `booth`, `ottoman`, `sofa` stay distinct - multi-person or
    # backless seating carries different room information.

    # --- tables: free-standing surfaces on legs ---
    "breakfast_table": "table",
    "coffee_table": "table",
    "commercial_kitchen_table": "table",
    "conference_table": "table",
    "console_table": "table",
    "garden_coffee_table": "table",
    "lab_table": "table",
    "pedestal_table": "table",
    # NOTE: `periodic_table` (a wall chart) and `pool_table` (game furniture) are NOT
    # tables in this sense and are deliberately absent.
    # `desk` and `reception_desk` stay distinct - they imply workspaces.

    # --- lights: ceiling/wall fixtures that light a room ---
    "downlight": "light",
    "rectangular_light": "light",
    "room_light": "light",
    "spotlight": "light",
    "square_light": "light",
    "track_light": "light",
    "wall_mounted_light": "light",
    "light_bulb": "light",
    # NOTE: `floor_lamp`, `table_lamp`, `chandelier`, `garden_light`, `fairy_light`,
    # `paper_lantern` stay distinct - they are furniture or decor, not fixtures.

    # --- televisions ---
    "standing_tv": "tv",
    "wall_mounted_tv": "tv",

    # --- mirrors ---
    "makeup_mirror": "mirror",
    "standing_mirror": "mirror",

    # --- loudspeakers ---
    "standing_loudspeaker": "loudspeaker",
    "wall_mounted_loudspeaker": "loudspeaker",

    # --- soap dispensers ---
    "wall_mounted_soap_dispenser": "soap_dispenser",

    # --- toilet paper fixtures ---
    "toilet_paper_holder": "toilet_paper_dispenser",

    # --- trash cans ---
    "public_trash_can": "trash_can",

    # --- fireplaces: fuel type does not change the room ---
    "gas_fireplace": "fireplace",
    "wood_fireplace": "fireplace",

    # --- windows: fixed vs openable is a mechanism detail ---
    "fixed_window": "window",
    "openable_window": "window",
    # NOTE: `window_blind` is a separate object that hangs on a window.

    # --- shelving units ---
    "commercial_kitchen_shelf": "shelf",
    "grocery_shelf": "shelf",
    "candy_dispenser_shelf": "shelf",
    "dry_food_dispenser_shelf": "shelf",
    "ceiling_rack": "shelf",

    # --- plants and trees ---
    "garden_plant": "plant",
    "hanging_plant": "plant",
    "pot_plant": "plant",
    "greenery": "plant",
    "flower": "plant",
    "bush": "plant",
    "low_resolution_tree": "tree",
    # NOTE: `hall_tree` is a coatrack, not a tree, and is deliberately absent.
    # `vine` stays distinct (climbing, usually structural).

    # --- artwork hung on walls ---
    "painting": "picture",
    "portrait": "picture",

    # --- doors: `door` absorbs only plain variants ---
    "sliding_door": "door",
    # NOTE: `garage_door`, `elevator_door`, `gate` stay distinct - each strongly
    # implies a specific room type.
}


def merge_category(category):
    """Map one fine-grained category to its everyday noun (identity if unmapped)."""
    return MERGE_MAP.get(category, category)


def merged_vocabulary(categories):
    """Apply the mapping to a category list, returning the sorted merged vocabulary."""
    return sorted({merge_category(c) for c in categories})


def main():
    import collections
    import csv
    import json
    import os

    groups = collections.defaultdict(list)
    for fine, coarse in MERGE_MAP.items():
        groups[coarse].append(fine)

    counts = {}
    placements = "data/placements.csv"
    if os.path.exists(placements):
        rooms = collections.defaultdict(set)
        with open(placements) as f:
            for r in csv.DictReader(f):
                rooms[(r["scene"], r["room_instance"])].add(r["object_category"])
        c = collections.Counter()
        for cats in rooms.values():
            for cat in cats:
                c[cat] += 1
        counts = c

    print(f"{len(MERGE_MAP)} fine categories -> {len(groups)} everyday nouns\n")
    for coarse in sorted(groups):
        members = sorted(groups[coarse])
        total = sum(counts.get(m, 0) for m in members)
        detail = ", ".join(f"{m}({counts.get(m, 0)})" for m in members) if counts else ", ".join(members)
        print(f"{coarse:24s} <- [{detail}]" + (f"  => {total} room instances" if counts else ""))

    vocab_path = "data/vocab.json"
    if os.path.exists(vocab_path):
        with open(vocab_path) as f:
            cats = json.load(f)["object_categories"]
        merged = merged_vocabulary(cats)
        print(f"\nvocabulary: {len(cats)} -> {len(merged)} categories")
        unused = sorted(set(MERGE_MAP) - set(cats))
        if unused:
            print(f"mapped but absent from scene vocabulary: {', '.join(unused)}")


if __name__ == "__main__":
    main()
