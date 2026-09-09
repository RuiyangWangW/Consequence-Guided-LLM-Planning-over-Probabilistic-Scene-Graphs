"""Turn extracted placements into a labeled (room_type, object_category) dataset.

The raw placements are positive-only, so negatives are constructed by treating each
concrete *room instance* as a closed world: within one room we know every object that
is present, so every category absent from it is a true negative.

Label semantics: y = 1 iff object category c appears at least once in room instance r.
Training on these pairs makes the model estimate

    P(category c is present in a room of type T)

which is the quantity a safety filter wants when asking "is a toilet plausibly in a
kitchen?".

Emits data/pairs.csv with one row per (room instance, category) pair.
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


import argparse
import csv
import json
import os
from collections import defaultdict

DEFAULT_MIN_ROOM_OBJECTS = 3


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placements", default="data/placements.csv")
    parser.add_argument("--vocab", default="data/vocab.json")
    parser.add_argument("--out", default="data/pairs_merged.csv")
    parser.add_argument(
        "--fine-grained",
        dest="fine_grained",
        action="store_true",
        help="build over raw BEHAVIOR categories instead of merged everyday nouns. "
        "Merged is the default because splitting a noun across rare variants fragments "
        "supervision and measurably hurts the RSN; see category_merge.py",
    )
    parser.add_argument(
        "--min-room-objects",
        type=int,
        default=DEFAULT_MIN_ROOM_OBJECTS,
        help="drop room instances with fewer objects than this; a nearly empty room is "
        "not evidence of absence and would inject false negatives",
    )
    args = parser.parse_args()

    with open(args.placements) as f:
        rows = list(csv.DictReader(f))
    with open(args.vocab) as f:
        vocab = json.load(f)

    cat_field = "object_category" if args.fine_grained else "merged_category"
    vocab_key = "object_categories" if args.fine_grained else "merged_categories"
    if cat_field not in rows[0]:
        raise SystemExit(f"{args.placements} has no '{cat_field}' column; re-run extract_scene_data.py")
    categories = vocab[vocab_key]

    # room instance -> set of categories present
    present = defaultdict(set)
    room_meta = {}
    for r in rows:
        key = (r["scene"], r["room_instance"])
        present[key].add(r[cat_field])
        room_meta[key] = r["room_type"]

    kept = {k: v for k, v in present.items() if len(v) >= args.min_room_objects}
    dropped = len(present) - len(kept)

    out_rows = []
    for (scene, room_instance), cats in sorted(kept.items()):
        room_type = room_meta[(scene, room_instance)]
        for category in categories:
            out_rows.append(
                {
                    "scene": scene,
                    "room_instance": room_instance,
                    "room_type": room_type,
                    "object_category": category,
                    "label": int(category in cats),
                }
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scene", "room_instance", "room_type", "object_category", "label"]
        )
        writer.writeheader()
        writer.writerows(out_rows)

    n_pos = sum(r["label"] for r in out_rows)
    print(f"categories:          {len(categories)} ({'fine-grained' if args.fine_grained else 'merged'})")
    print(f"room instances kept: {len(kept)} (dropped {dropped} with <{args.min_room_objects} objects)")
    print(f"pairs:               {len(out_rows)}")
    print(f"positives:           {n_pos} ({n_pos / len(out_rows):.2%})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
