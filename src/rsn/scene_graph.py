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


import collections
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


# What a person calls a room, mapped to what BEHAVIOR calls it. An instruction says "the
# office cabinet" and the scene graph says `private_office`; without this the stated room
# matched nothing and was silently discarded, which is worse than not extracting it - the
# work was done and then thrown away. Eight of the eighteen room words the extractor can
# produce needed one of these.
ROOM_SYNONYMS = {
    "office": "private_office", "study": "private_office", "den": "living_room",
    "pantry": "pantry_room", "hallway": "corridor", "hall": "corridor",
    "playroom": "childs_room", "nursery": "childs_room", "lounge": "living_room",
    "washroom": "bathroom", "toilet": "bathroom", "laundry": "utility_room",
    "laundry_room": "utility_room", "storage": "storage_room", "foyer": "entryway",
    "hallway_corridor": "corridor", "basement": "storage_room", "cellar": "storage_room",
    "entrance": "entryway", "front_hall": "entryway", "vestibule": "entryway",
}


def resolve_room_type(word, available):
    """The scene room type a stated room word means, or None if the scene has no such room."""
    if word in available:
        return word
    alias = ROOM_SYNONYMS.get(word)
    return alias if alias in available else None


def populate(
    scene_or_graph,
    objects,
    dependent=None,
    stated=None,
    model_path=DEFAULT_MODEL,
    threshold=DEFAULT_THRESHOLD,
    graphs_path=DEFAULT_GRAPHS,
    device=None,
):
    """Place the task's objects in the scene's rooms, using what the task said first.

    Takes the three disjoint classes `task_objects.extract` produces:

        objects     the task said nothing about where these are - the RSN guesses
        stated      {object: room} the task named the room of
        dependent   [{object, relation, target}] the task named the support of

    and resolves them into one map:

        {object: {"room": room_id, "room_type": str, "probability": float,
                  "candidates": [room_id, ...]}}

    **A location is resolved, not guessed, wherever the task allows.** A dependent object's
    room is its *root's* room - follow the chain of relations to the object nothing else
    hangs off, and take that one's room. "the mug in the office cabinet" puts the mug in
    the office without the RSN ever being asked about mugs, and a chain of any depth works
    the same way.

    **Everything keeps a ranked fallback, including stated rooms.** `candidates` is the
    order the robot should search, and it always ends with the RSN's full ranking. A stated
    room goes first because the task said so, but if the robot searches it and the object
    is not there, the belief was wrong and there is somewhere else to look. Before this,
    a stated room produced a one-element list and a wrong statement was unrecoverable -
    the object was simply unfindable and the plan died.
    """
    import torch

    from query_rsn import load, predict_rooms

    graph = (
        load_room_graph(scene_or_graph, graphs_path)
        if isinstance(scene_or_graph, str)
        else scene_or_graph
    )
    rooms = graph["rooms"]

    # **Rooms the robot cannot get into are not part of its world, so they are not part of
    # its belief either.** Leaving them in gave the RSN somewhere to put probability that
    # can never hold anything: the searcher was sent to `bedroom_0` in `Wainscott_0_int`,
    # got "no standable floor" back, ruled it out and moved on - a wasted sweep every time -
    # and the mass spent on six dead rooms was mass taken from the six that can actually
    # hold the object. Worse, `sim_eval.ground` could bind a plan's word to a piece of
    # furniture standing in one, which is what made nineteen tasks unsolvable before any
    # planning happened.
    #
    # `floor_world.reachable_rooms` is the single definition, and it is the simulator's own:
    # a room is reachable if it has standable floor in the region the robot starts in. The
    # remaining probabilities are renormalised over what is left, so the ranking still sums
    # to one and still covers every room the robot can reach.
    if isinstance(scene_or_graph, str):
        from floor_world import reachable_rooms

        keep = reachable_rooms(scene_or_graph, graphs_path=graphs_path)
        if keep and any(r not in keep for r in rooms):
            rooms = {r: info for r, info in rooms.items() if r in keep}
            graph = dict(graph)
            graph["rooms"] = rooms
            graph["edges"] = [e for e in graph.get("edges", [])
                              if e[0] in keep and e[1] in keep]

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, ckpt = load(model_path, device)
    room_types = ckpt["room_types"]

    # Largest instance per type: with several bathrooms, the biggest is the main one.
    best_instance = {}
    for rid, info in rooms.items():
        t = info["room_type"]
        if t not in best_instance or info["pixels"] > rooms[best_instance[t]]["pixels"]:
            best_instance[t] = rid

    # Every room of every type, so a type with two instances can hold the object in either.
    # Keeping only the largest instance - which this did - means an object that is really in
    # the smaller living room is not merely mis-ranked, it is unreachable by search: the room
    # never enters the candidate list, so no amount of looking can ever turn it up.
    instances = collections.defaultdict(list)
    for rid, info in rooms.items():
        instances[info["room_type"]].append(rid)

    def rsn_ranking(name):
        """A complete distribution over this scene's rooms, most probable first.

        The RSN scores room *types*; a scene has room *instances*. The type's mass is split
        equally across its instances, because the model has said nothing to tell two bedrooms
        apart - they are a genuine tie. The tie is broken where the information to break it
        exists: `search_cost` orders equal-probability rooms by how far they are from
        wherever the robot is standing, so it sweeps the near bedroom first. That is a fact
        about the robot's position, which changes as it moves, so it cannot be baked in here.

        The result is normalised over the rooms this scene actually has, so it sums to one
        and ranks every room. That matters downstream: the search estimator weighs a
        candidate by its probability, and a distribution that sums to the top score alone -
        which is what the raw RSN output gives - makes an object the model is unsure about
        look cheap to find rather than expensive.

        Rooms whose type the RSN has no label for keep a floor of the smallest scored mass,
        rather than zero: the model having no opinion about a room is not evidence that the
        object is not in it.
        """
        probs = predict_rooms(model, ckpt, name, device)
        # Only room types this scene actually has are candidates: a high score for
        # `garage` is irrelevant in a scene with no garage.
        scored = {t: float(probs[room_types.index(t)]) for t in instances if t in room_types}
        if not scored:
            return [], 0.0, None, {}
        floor = min(scored.values()) if scored else 0.0
        weights = {}
        for room_type, rids in instances.items():
            mass = scored.get(room_type, floor)
            for rid in rids:
                weights[rid] = mass / len(rids)
        bulk = sum(weights.values())
        belief = ({r: w / bulk for r, w in weights.items()} if bulk > 0
                  else {r: 1.0 / len(weights) for r in weights})
        ranked = sorted(belief, key=lambda r: (-belief[r], r))
        best_type = rooms[ranked[0]]["room_type"]
        return ranked, float(scored.get(best_type, belief[ranked[0]])), best_type, belief

    stated = stated or {}
    dependent = dependent or []
    # Follow each dependent object to the root of its chain - the thing nothing else it
    # rests on hangs off. That root is the only object whose room has to be established;
    # everything above it inherits.
    support = {d["object"]: d["target"] for d in dependent}

    def root_of(name):
        seen = set()
        while name in support and name not in seen:
            seen.add(name)
            name = support[name]
        return name

    # The roots, plus every free-standing object, are what need a room of their own.
    roots = {root_of(d["object"]) for d in dependent}
    to_place = list(dict.fromkeys(list(objects) + list(stated) + list(roots)))
    to_place = [n for n in to_place if n not in support]

    placed, unplaced = {}, {}
    for name in to_place:
        ranked, p, best_type, belief = rsn_ranking(name)
        said = resolve_room_type(stated.get(name), best_instance) if name in stated else None
        if said:
            # The task said so, so search there first - but keep the RSN's ranking behind
            # it, because a stated room can still be wrong and the robot needs somewhere
            # else to look when it is.
            first = best_instance[said]
            placed[name] = {
                "room": first,
                "room_type": said,
                "probability": 1.0,
                "candidates": [first] + [r for r in ranked if r != first],
                # The task named the room, so that is where the robot looks; the RSN's
                # distribution stays behind it as the belief to fall back on if it is wrong.
                "belief": {first: 1.0},
                "fallback": belief,
                "stated": True,
            }
            continue
        if not ranked:
            unplaced[name] = {"reason": "no known room type in scene", "probability": 0.0}
            continue
        if threshold is not None and p < threshold:
            unplaced[name] = {
                "reason": f"below threshold ({p:.3f} < {threshold:.2f})",
                "probability": p,
                "best_room_type": best_type,
            }
            continue
        placed[name] = {
            "room": best_instance[best_type],
            "room_type": best_type,
            "probability": p,
            "candidates": ranked,
            "belief": belief,
        }

    # Now the dependent objects, outward from the roots. A relation is a fact the task
    # stated, so the object is where its support is - and it inherits the support's search
    # order too, so that if the support turns out to be somewhere else, the object is
    # looked for there as well.
    relations = []
    remaining = list(dependent)
    for _ in range(len(remaining) + 1):
        progressed = False
        for dep in list(remaining):
            target = dep["target"]
            if target not in placed and target in support:
                continue                       # its own support is not resolved yet
            anchor = placed.get(target, {})
            room = anchor.get("room")
            # Three tiers of fallback, weakest assumption last:
            #   1. where the task says the support is
            #   2. where the support might be instead, if that room is ruled out
            #   3. where this object itself tends to live - the tier that saves the task
            #      when the *relation* was wrong, not just the room. A potato reported on
            #      the countertop but actually in the fridge is only findable if the
            #      potato's own ranking is in the list.
            own, _, _, own_belief = rsn_ranking(dep["object"])
            order = list(anchor.get("candidates") or ([room] if room else [])) + own
            placed[dep["object"]] = {
                "room": room,
                "room_type": rooms[room]["room_type"] if room in rooms else None,
                "probability": 1.0,
                "candidates": list(dict.fromkeys(order)),
                # Where its support is, if that is known; otherwise the support's own
                # belief, and failing that this object's - the same three tiers as the
                # candidate order above, carrying probability rather than just rank.
                "belief": ({room: 1.0} if room else
                           (anchor.get("belief") or own_belief)),
                "fallback": anchor.get("belief") or own_belief,
                "via": {"relation": dep["relation"], "target": target},
            }
            relations.append({
                "from": dep["object"], "relation": dep["relation"], "to": target,
                "probability": 1.0,
            })
            remaining.remove(dep)
            progressed = True
        if not progressed:
            break
    for dep in remaining:                      # a cycle in the stated relations
        unplaced[dep["object"]] = {"reason": "relation chain has no root",
                                   "probability": 0.0}

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

    graph = populate(args.scene, args.objects, model_path=args.model,
                     threshold=args.threshold)
    print(json.dumps(graph, indent=1) if args.json else format_for_llm(graph))


if __name__ == "__main__":
    main()
