"""Populate a room graph with the objects a task needs, using the RSN as the prior.

The room graph from `room_graph.py` is pure topology: which rooms exist and which
connect. It says nothing about contents. This module adds the missing half - for each
object the task requires, the RSN's P(object | room type) decides which room instance
it is placed in, producing a *scene graph* the planner can reason over:

    rooms      -> nodes, with edges from the floor-plan adjacency
    objects    -> attached to a room node, each with the probability that justified it

Placement uses the RSN over room *types*, but the graph has room *instances*
(`bathroom_0`, `bathroom_1`). When a type has several instances the object goes in the
largest, which is the one most likely to be the "main" room of that type.

Objects below the confidence floor are recorded as `unplaced` rather than dropped: the
planner needs to know an object is probably absent, which is exactly the signal the
safety filter exists to provide.
"""

import json
import os

DEFAULT_GRAPHS = "data/room_graphs.json"
DEFAULT_MODEL = "models/rsn_cal.pt"

# No threshold by default: every requested object is placed in its most probable room
# and the probability travels with it.
#
# Thresholding discards real objects. `Rs_int` genuinely contains a laptop, but the RSN
# scores it 0.13 for living_room - laptops appear in few scenes, so the prior is weak -
# and a 0.30 cut marked it absent, blocking a task the scene could actually support.
# Carrying the number instead lets the planner weigh a doubtful object against a
# confident one, and leaves the accept/reject decision to whatever consumes the graph
# rather than hiding it here.
DEFAULT_THRESHOLD = None


def load_room_graph(scene, graphs_path=DEFAULT_GRAPHS, dataset_root=None):
    """Room graph for one scene, from the cache or rebuilt from the floor plans.

    The cache is used when it has the fields we need. Older caches predate `centroid`
    and were written before the edge filter changed, so we rebuild rather than trust
    a stale schema.
    """
    if os.path.exists(graphs_path):
        with open(graphs_path) as f:
            graphs = json.load(f)
        if scene in graphs:
            g = graphs[scene]
            if all("centroid" in r for r in g["rooms"].values()):
                return g

    from room_graph import DEFAULT_DATASET, build_room_graph

    return build_room_graph(scene, dataset_root or DEFAULT_DATASET)


def populate(
    scene_or_graph,
    objects,
    dependent=None,
    model_path=DEFAULT_MODEL,
    threshold=DEFAULT_THRESHOLD,
    graphs_path=DEFAULT_GRAPHS,
    device=None,
):
    """Attach `objects` to the rooms of a scene graph using RSN probabilities.

    Each object goes in its single most probable room among the types this scene has,
    and the probability is kept alongside it:

        {object_name: {"room": room_id, "probability": float, "room_type": str}}

    With `threshold` set, objects whose best room falls below it are diverted to an
    `unplaced` map instead. Left as None (the default) nothing is discarded, and the
    probability is reported so the consumer can judge for itself.
    """
    import torch

    from query_rsn import load, predict_rooms

    graph = (
        load_room_graph(scene_or_graph, graphs_path)
        if isinstance(scene_or_graph, str)
        else scene_or_graph
    )
    rooms = graph["rooms"]

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load(model_path, device)
    room_types = ckpt["room_types"]

    # Largest instance per type: with several bathrooms, the biggest is the main one.
    best_instance = {}
    for rid, info in rooms.items():
        t = info["room_type"]
        if t not in best_instance or info["pixels"] > rooms[best_instance[t]]["pixels"]:
            best_instance[t] = rid

    placed, unplaced = {}, {}
    for name in objects:
        probs = predict_rooms(model, ckpt, name, device)

        # Only room types this scene actually has are candidates: a high score for
        # `garage` is irrelevant in a scene with no garage.
        candidates = [
            (probs[room_types.index(t)], t)
            for t in best_instance
            if t in room_types
        ]
        if not candidates:
            unplaced[name] = {"reason": "no known room type in scene", "probability": 0.0}
            continue

        p, room_type = max(candidates)
        if threshold is not None and p < threshold:
            unplaced[name] = {
                "reason": f"below threshold ({p:.3f} < {threshold:.2f})",
                "probability": float(p),
                "best_room_type": room_type,
            }
            continue

        placed[name] = {
            "room": best_instance[room_type],
            "room_type": room_type,
            "probability": float(p),
        }

    # Dependent objects are not guessed at: the task stated where they are, so they get
    # a deterministic edge to their container or support. A potato "from the fridge" is
    # inside the fridge with probability 1, and its room follows from the fridge's.
    relations = []
    for dep in dependent or []:
        target = dep["target"]
        room = placed.get(target, {}).get("room")
        placed[dep["object"]] = {
            "room": room,
            "room_type": rooms[room]["room_type"] if room in rooms else None,
            "probability": 1.0,
            "via": {"relation": dep["relation"], "target": target},
        }
        relations.append({
            "from": dep["object"], "relation": dep["relation"], "to": target,
            "probability": 1.0,
        })

    out = dict(graph)
    out["objects"] = placed
    out["unplaced"] = unplaced
    out["relations"] = relations
    out["threshold"] = threshold
    return out


def format_for_llm(graph):
    """Render the scene graph as compact text for an LLM prompt.

    Deliberately plain: room list, adjacency list, and contents per room. An LLM plans
    better over an explicit adjacency list than over coordinates, and the probabilities
    are included so it can prefer certain objects over doubtful ones.
    """
    from collections import defaultdict

    lines = [f"Scene: {graph['scene']}", "", "Rooms:"]
    for rid in sorted(graph["rooms"]):
        lines.append(f"  {rid}")

    lines += ["", "Connections (robot can move directly between these):"]
    if graph["edges"]:
        for a, b in graph["edges"]:
            lines.append(f"  {a} <-> {b}")
    else:
        lines.append("  (none)")

    by_room = defaultdict(list)
    for name, info in graph.get("objects", {}).items():
        by_room[info["room"]].append((name, info["probability"]))

    relations = graph.get("relations", [])
    dependent_names = {r["from"] for r in relations}

    lines += ["", "Objects and where they are most likely to be (0-1 confidence):"]
    guessed = {rid: [(n, p) for n, p in items if n not in dependent_names]
               for rid, items in by_room.items()}
    if any(guessed.values()):
        for rid in sorted(guessed):
            if not guessed[rid]:
                continue
            items = ", ".join(f"{n} ({p:.2f})" for n, p in sorted(guessed[rid]))
            lines.append(f"  {rid}: {items}")
    else:
        lines.append("  (none)")

    # Stated by the task, so certain - kept separate from the probabilistic placements so
    # the planner can tell a fact from a guess.
    if relations:
        lines += ["", "Known object positions (certain):"]
        for r in sorted(relations, key=lambda r: r["from"]):
            word = "inside" if r["relation"] == "INSIDE" else "on top of"
            room = graph.get("objects", {}).get(r["from"], {}).get("room")
            where = f" (in {room})" if room else ""
            lines.append(f"  {r['from']} is {word} the {r['to']}{where}")

    if graph.get("unplaced"):
        lines += ["", "Objects likely NOT present in this scene:"]
        for name, info in sorted(graph["unplaced"].items()):
            lines.append(f"  {name} ({info['reason']})")

    return "\n".join(lines)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="optional: drop objects whose best room scores below this. "
                             "Off by default - every object is placed in its most "
                             "likely room and the probability is reported instead")
    parser.add_argument("--json", action="store_true", help="emit the graph as JSON")
    args = parser.parse_args()

    graph = populate(args.scene, args.objects, args.model, args.threshold)
    print(json.dumps(graph, indent=1) if args.json else format_for_llm(graph))


if __name__ == "__main__":
    main()
