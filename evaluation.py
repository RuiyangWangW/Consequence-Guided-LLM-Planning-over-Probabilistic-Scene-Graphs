"""Shared evaluation utilities: scene-grouped splitting and ranking metrics.

sklearn is not installed in the `behavior` conda env, so ROC-AUC and average precision
are implemented here. Both were verified against brute-force computation, including
tied-score cases.
"""

import csv
import random
from collections import defaultdict

import numpy as np


def load_pairs(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def scene_split(rows, val_frac, seed):
    """Split scenes so every room type keeps at least one instance in train.

    Grouping by scene (rather than splitting individual pairs) means validation
    measures generalization to an unseen building, which is what the filter faces. A
    pair-level split would leak, since the same room instance would appear on both
    sides.

    Scenes are shuffled and assigned to validation greedily, skipping any scene that
    would strip the training set of the last instance of some room type - several room
    types have only one instance in the whole dataset.
    """
    scenes = sorted({r["scene"] for r in rows})
    scene_rooms = defaultdict(set)
    for r in rows:
        scene_rooms[r["scene"]].add(r["room_type"])

    room_scene_count = defaultdict(int)
    for scene, types in scene_rooms.items():
        for t in types:
            room_scene_count[t] += 1

    rng = random.Random(seed)
    order = scenes[:]
    rng.shuffle(order)

    target = max(1, int(round(len(scenes) * val_frac)))
    val, remaining = set(), dict(room_scene_count)
    for scene in order:
        if len(val) >= target:
            break
        # Only move this scene to val if each of its room types survives in train.
        if all(remaining[t] > 1 for t in scene_rooms[scene]):
            val.add(scene)
            for t in scene_rooms[scene]:
                remaining[t] -= 1

    train = [r for r in rows if r["scene"] not in val]
    val_rows = [r for r in rows if r["scene"] in val]
    return train, val_rows, sorted(val)


def roc_auc(y_true, y_score):
    """ROC-AUC via the rank (Mann-Whitney U) identity, with ties averaged."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def average_precision(y_true, y_score):
    """Area under the precision-recall curve, computed as the step-wise sum."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-y_score, kind="mergesort")
    hits = y_true[order]
    cum_tp = np.cumsum(hits)
    precision = cum_tp / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)
