"""Extract (scene, room_type, object_category) placements from the BEHAVIOR-1K dataset.

Each scene ships a single `*_best.json` describing its instantiated objects. Every
object carries an `in_rooms` list of room *instances* (e.g. "bedroom_1"); the room
*type* is that name with its trailing instance index stripped.

Writes two files to --out-dir:
  placements.csv  one row per (scene, room_instance, object) pair
  vocab.json      the room-type / object-category vocabularies and scene list
"""

import os as _os, sys as _sys
# Run from anywhere. Find the repo root by marker, put it on the import path, and make
# it the working directory - every path in this file is written 'data/...', so without
# the chdir a build invoked from inside its own folder would quietly write benchmark/data/.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.exists(_os.path.join(_d, 'graph_machine.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)
_os.chdir(_d)


import argparse
import csv
import json
import os
import re
from collections import Counter

from category_merge import merge_category

# Structural geometry that is part of the building rather than its contents.
# These carry no `in_rooms` annotation and are not useful as MLP targets.
STRUCTURAL_CATEGORIES = {
    "walls",
    "ceilings",
    "floors",
    "roof",
    "background",
    "driveway",
    "lawn",
    "pillar",
}

ROOM_INSTANCE_RE = re.compile(r"_\d+$")

DEFAULT_DATASET = "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/datasets/behavior-1k-assets"


def room_type_of(room_instance):
    """"bedroom_1" -> "bedroom"."""
    return ROOM_INSTANCE_RE.sub("", room_instance)


def iter_placements(scenes_dir):
    """Yield one record per (object, room) pair across every scene."""
    for scene in sorted(os.listdir(scenes_dir)):
        json_dir = os.path.join(scenes_dir, scene, "json")
        if not os.path.isdir(json_dir):
            continue
        for fname in sorted(os.listdir(json_dir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(json_dir, fname)) as f:
                data = json.load(f)
            init_info = data.get("objects_info", {}).get("init_info", {})
            for obj_name, info in init_info.items():
                args = info.get("args", {})
                category = args.get("category")
                if category is None or category in STRUCTURAL_CATEGORIES:
                    continue
                for room_instance in args.get("in_rooms") or []:
                    yield {
                        "scene": scene,
                        "scene_file": fname,
                        "object_name": obj_name,
                        "object_category": category,
                        # Everyday-noun form; equals object_category when unmapped.
                        "merged_category": merge_category(category),
                        "model": args.get("model", ""),
                        "room_instance": room_instance,
                        "room_type": room_type_of(room_instance),
                    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=os.environ.get("BEHAVIOR_ASSETS", DEFAULT_DATASET))
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()

    scenes_dir = os.path.join(args.dataset_root, "scenes")
    if not os.path.isdir(scenes_dir):
        raise SystemExit(f"scenes directory not found: {scenes_dir}")

    rows = list(iter_placements(scenes_dir))
    if not rows:
        raise SystemExit(f"no placements extracted from {scenes_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    fields = [
        "scene",
        "scene_file",
        "object_name",
        "object_category",
        "merged_category",
        "model",
        "room_instance",
        "room_type",
    ]
    csv_path = os.path.join(args.out_dir, "placements.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    scenes = sorted({r["scene"] for r in rows})
    room_types = sorted({r["room_type"] for r in rows})
    categories = sorted({r["object_category"] for r in rows})
    merged_categories = sorted({r["merged_category"] for r in rows})

    # The canonical room vocabulary shipped with the dataset, so the model can
    # score room types that happen to have no instances in any scene.
    meta_rooms_path = os.path.join(args.dataset_root, "metadata", "room_categories.txt")
    if os.path.exists(meta_rooms_path):
        with open(meta_rooms_path) as f:
            meta_rooms = sorted({line.strip() for line in f if line.strip()})
        room_types = sorted(set(room_types) | set(meta_rooms))

    vocab = {
        "scenes": scenes,
        "room_types": room_types,
        "object_categories": categories,
        "merged_categories": merged_categories,
    }
    with open(os.path.join(args.out_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    room_counts = Counter(r["room_type"] for r in rows)
    print(f"placements:       {len(rows)}")
    print(f"scenes:           {len(scenes)}")
    print(f"room types:       {len(room_types)}")
    print(f"object categories:{len(categories)} (merged: {len(merged_categories)})")
    print(f"room instances:   {len({(r['scene'], r['room_instance']) for r in rows})}")
    print(f"wrote {csv_path} and {os.path.join(args.out_dir, 'vocab.json')}")
    print("\ntop room types by placement count:")
    for room, n in room_counts.most_common(10):
        print(f"  {room:20s} {n}")


if __name__ == "__main__":
    main()
