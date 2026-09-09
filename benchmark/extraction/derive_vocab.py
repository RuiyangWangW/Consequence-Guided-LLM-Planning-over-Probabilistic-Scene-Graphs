#!/usr/bin/env python3
"""Derive the household vocabulary from BEHAVIOR's own data, not from a hand-written list.

The training generator draws objects, supports and room qualifiers. Left to the raw dataset
it drew a `periodic_table` as furniture, a `graduated_cylinder` off a shelf, and put a
dishwasher in a bedroom - because the pools were regexes over all 51 scenes, which include
chemistry labs and restaurants, and the room was a uniform choice. Instructions like that
are not household tasks, and a model trained on them learns that the words carry less than
they do.

Everything needed to fix that is already in BEHAVIOR, in two files nobody was reading:

  activity_definitions/*/problem0.bddl   1018 real household activities. Each declares the
                                         synsets it uses and, through `inroom`, the rooms it
                                         happens in.
  generated_data/combined_room_object_list.json
                                         what each of the 51 scenes is actually furnished
                                         with, room by room.

So "is this a household object" becomes "does a household activity use it", and "where does
this belong" becomes "where do the activities that use it happen, and where is it installed
in the houses". Both are answers BEHAVIOR already gives; neither is a judgement of mine.

    python derive_vocab.py            # writes data/household_vocab.json
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
import collections
import csv
import json
import os
import re

BDDL = "/mnt/check/ruiyangw/omnigibson/BEHAVIOR-1K/bddl3/bddl"
OUT = "data/household_vocab.json"

def house_rooms(bddl=BDDL):
    """The room types the residential scenes actually have.

    Read off the scenes rather than written down. BEHAVIOR's room vocabulary spans 39 types
    including `chemistry_lab`, `grocery_store` and `sauna`; the fifteen `_int` houses use 19
    of them, and those are the rooms this pipeline plans in. Listing them by hand was the
    first version and it let a `sauna_bench` through, because I had guessed that a sauna was
    a room a house has.
    """
    path = os.path.join(bddl, "generated_data", "combined_room_object_list.json")
    with open(path) as handle:
        scenes = json.load(handle).get("scenes") or {}
    return {re.sub(r"_\d+$", "", room)
            for scene, rooms in scenes.items() if scene.endswith("_int")
            for room in rooms}


def synset_to_categories(bddl=BDDL):
    """synset -> the object categories that realise it, from `category_mapping.csv`."""
    out = collections.defaultdict(set)
    path = os.path.join(bddl, "generated_data", "category_mapping.csv")
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            synset, category = (row.get("synset") or "").strip(), (row.get("category") or "").strip()
            if synset and category:
                out[synset].add(category)
    return out


def activities(bddl=BDDL):
    """Every activity's (synsets, rooms), read off its BDDL problem definition."""
    root = os.path.join(bddl, "activity_definitions")
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "problem0.bddl")
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            text = handle.read()
        block = re.search(r"\(:objects(.*?)\n\s*\)", text, re.S)
        synsets = set()
        if block:
            # "water.n.06_1 - water.n.06" - the type after the dash is the synset.
            for line in block.group(1).splitlines():
                if "-" in line:
                    synsets.add(line.rsplit("-", 1)[1].strip())
        rooms = set(re.findall(r"\(inroom\s+\S+\s+(\w+)\s*\)", text))
        yield name, synsets, rooms


def scene_fixtures(bddl=BDDL, indoor=(), houses_only=True):
    """category -> {room_type: count}, from what the scenes are actually furnished with.

    With `houses_only` off this is every scene, which is how a category is recognised as
    *furniture* at all: a `massage_bed` and a `sauna_bench` are installed somewhere, just
    never in a house, and that is the difference between them and a `mug`, which no house
    installs because tasks bring it.
    """
    path = os.path.join(bddl, "generated_data", "combined_room_object_list.json")
    with open(path) as handle:
        scenes = json.load(handle).get("scenes") or {}
    out = collections.defaultdict(collections.Counter)
    for scene, rooms in scenes.items():
        if houses_only and not scene.endswith("_int"):
            continue
        for room, contents in rooms.items():
            room_type = re.sub(r"_\d+$", "", room)
            if room_type not in indoor:
                continue
            for entry in contents:
                # entries look like `bottom_cabinet-immwzb`; the model id is after the dash
                out[entry.rsplit("-", 1)[0]][room_type] += 1
    return out


def derive(bddl=BDDL):
    """The household vocabulary, and where each of its categories belongs."""
    indoor = house_rooms(bddl)
    mapping = synset_to_categories(bddl)
    used = collections.Counter()
    where = collections.defaultdict(collections.Counter)
    kept = dropped = 0
    for name, synsets, rooms in activities(bddl):
        here = {r for r in rooms if r in indoor}
        if rooms and not here:
            dropped += 1          # happens only outdoors - not a task this pipeline plans
            continue
        kept += 1
        for synset in synsets:
            realisations = mapping.get(synset, ())
            if not realisations:
                continue
            # One synset is realised by several categories - `bed.n.01` is a bed and also a
            # massage bed - and crediting each of them the whole activity made the odd ones
            # look as common as the ordinary one. The activity says a bed was used, not
            # which; split the credit rather than multiply it.
            share = 1.0 / len(realisations)
            for category in realisations:
                used[category] += share
                for room in here:
                    where[category][room] += share

    # Two different questions, and the answer differs by what kind of thing it is.
    #
    # A *fixture* is part of the building, so a household fixture is one a house is actually
    # built with: a `massage_bed` and a `sauna_bench` are installed somewhere, never in a
    # house, and no household instruction names them. That is a crisp exclusion.
    #
    # A *movable* is brought to the task, so no house installs it - `mug` appears in cafe
    # scenes and in no house, and it is obviously a kitchen object. There is no crisp line
    # to draw, so none is drawn: an indoor activity using it is enough, and how often the
    # activities use it is what the weight carries.
    from extraction_data import movable_pool

    movable = set(movable_pool())
    fixtures = scene_fixtures(bddl, indoor)
    for category in list(used):
        if category not in movable and category not in fixtures:
            del used[category]
            where.pop(category, None)
    # A category the houses are furnished with belongs in a household vocabulary whether or
    # not an activity happens to name it - the robot has to be able to talk about the sofa
    # it walks past. Its rooms come from where it is actually installed, which is a stronger
    # answer than where an activity mentioning it happened to be set.
    for category, rooms in fixtures.items():
        used[category] += 1
        for room, n in rooms.items():
            where[category][room] += n

    return {
        "categories": sorted(used),
        # How much household use BEHAVIOR actually records for each - the generator draws
        # against this, so an object one activity mentions stays as rare as it really is
        # instead of being as likely as a mug.
        "weight": {c: round(w, 4) for c, w in used.items()},
        "rooms": {c: dict(r) for c, r in where.items() if r},
        "kept_activities": kept,
        "dropped_outdoor_activities": dropped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bddl", default=BDDL)
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    vocab = derive(args.bddl)
    with open(args.out, "w") as handle:
        json.dump(vocab, handle, indent=1, sort_keys=True)
    print(f"{len(vocab['categories'])} household categories from "
          f"{vocab['kept_activities']} indoor activities "
          f"({vocab['dropped_outdoor_activities']} outdoor-only dropped), "
          f"{len(vocab['rooms'])} with a room -> {args.out}")


if __name__ == "__main__":
    raise SystemExit(main())
