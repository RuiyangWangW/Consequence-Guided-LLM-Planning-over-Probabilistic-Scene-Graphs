"""The world graph: what the robot has actually seen, as typed edges.

The room graph from `room_graph.py` is the starting point and the only thing known before
the robot moves - which rooms exist and which are connected. Everything else is empty. As
the robot searches a room it sees objects, and each observation writes edges here: which
room the object is in, and how it sits relative to the other objects already seen.

Eight edge types, and where each comes from:

    room_connect    two rooms are traversably adjacent   from the room graph, known up front
    room_inside     an object is in a room               ground truth, when the object is seen
    object_inside   an object is inside another          simulator predicate, when both are seen
    on_top          an object rests on another           simulator predicate, when both are seen
    under           an object is beneath another         simulator predicate, when both are seen
    next_to         two objects are beside each other    simulator predicate, when both are seen
    holding         the robot has this in its hand       from the robot's own actions
    nearby          the robot is beside this              from where it navigated to

The robot is a node too, `ROBOT`, and its facts are edges like any other:
`room_inside(robot, kitchen_0)` says which room it is in, `holding(robot, potato)` says
what it has, and `nearby(robot, oven)` says what it is standing at. That last one is what
every manipulation's "the robot is beside it" precondition reads, and it is why a room is
no longer the finest thing the model knows about where the robot is. A carried object has **no `room_inside` edge of its own**: it is wherever the
robot is, and `room_of` derives that by following `holding` back to the robot. Storing it
as well would be a second record of the same fact, kept in step by rewriting it on every
navigation - which is one more thing that can fall out of step. That is what makes "where is the robot and what is it carrying" a question
answered by reading the graph rather than by asking whoever happens to be executing the
plan. `open` and `toggled` stay off the graph, because they are properties of one node
rather than relations between two.

The gate is the same for all five observed types: **an edge is only written once the robot
has seen the objects it connects**. That is what makes the graph a belief rather than a
copy of the scene registry, and what makes "I searched the kitchen and the potato is not
there" a statement the robot has earned.

Edges are never silently dropped once written. An object that goes out of frame does not
stop existing, and the robot should not forget where it left the plate. Re-observing an
object *replaces* its kinematic edges (`object_inside`, `on_top`, `under`, `next_to`),
because those are the ones an action can change; `room_inside` and `room_connect` persist.
"""

import json

# The six edge types, in the order they are defined above.
EDGE_TYPES = ("room_connect", "room_inside", "object_inside", "on_top", "under",
              "next_to", "holding", "nearby")

# The robot's own node. Reserved: nothing observed is ever called this.
ROBOT = "robot"

# Edges whose two endpoints are interchangeable. Stored with the endpoints sorted, so
# next_to(a, b) and next_to(b, a) are one edge and not two.
SYMMETRIC = frozenset({"room_connect", "next_to"})

# Edges that describe how an object is resting, as opposed to where it lives. A primitive
# that moves an object invalidates exactly these, which is also why re-observing an object
# replaces them rather than adding to them. `holding` is deliberately not among them: an
# object in the hand is not resting on anything, and re-observing it must not put it down.
KINEMATIC = ("object_inside", "on_top", "under", "next_to")

# Predicates read from the simulator, and the edge each one writes. `Under` is the
# converse of `OnTop` for most furniture, but not always - a rug is under a table without
# the table being on top of it in any useful sense - so both are read rather than derived.
PREDICATE_EDGES = (("object_inside", "Inside"), ("on_top", "OnTop"),
                   ("under", "Under"), ("next_to", "NextTo"))


class WorldGraph:
    """Rooms, objects, and typed edges between them.

    Nodes are plain strings: room instance ids (`kitchen_0`) and object instance names
    (`potato`, `oven_ffitak_0`). The two namespaces are kept apart by `self.rooms` rather
    than by decorating the ids, so an id that appears in a plan can be looked up directly.
    """

    def __init__(self):
        self.rooms = {}      # room id -> {"room_type": str}
        self.objects = {}    # object name -> {"category": str, "position": [x, y, z]}
        self.edges = set()   # (edge_type, src, dst)
        self.log = []        # ordered record of every edit, for inspection and rollback

    # ---------------------------------------------------------------- construction

    @classmethod
    def from_room_graph(cls, graph):
        """Seed with the room graph: rooms and their connectivity, nothing else.

        This is everything the robot knows before it has looked at anything.
        """
        g = cls()
        for room_id, info in graph.get("rooms", {}).items():
            g.rooms[room_id] = {"room_type": info.get("room_type")}
        for a, b in graph.get("edges", []):
            g.add_edge("room_connect", a, b, note="room graph")
        return g

    @classmethod
    def from_scene_graph(cls, graph):
        """Seed from the RSN's scene graph: the rooms, and where it thinks things are.

        `scene_graph.populate` produces a plain dict - `{potato: {room: kitchen_0,
        probability: 0.66}}` plus the relations a task stated outright - and that dict is
        what the LLM is shown and what `planner.validate` checks. Without this it is not
        what `GraphMachine` checks: `from_room_graph` reads `rooms` and `edges` and drops
        `objects`, so the machine was validating against a graph with nothing in it and
        every `GRASP` failed for want of an object the planner had been told about.

        What comes across is a *belief*, and it is written as one. The room is the RSN's
        guess and the probability travels with it, so a consumer can tell a 0.99 from a
        0.13. There are no positions: the RSN predicts rooms, not coordinates, and an
        object here has a room and no place in it until the robot looks.
        """
        g = cls.from_room_graph(graph)
        for name, info in (graph.get("objects") or {}).items():
            room = info.get("room")
            # A room the placement names but the topology does not declare is registered
            # rather than dropped. Dropping it is quiet and total: an object with no room
            # has no room to be *in*, `_require_here` finds nothing to contradict, and
            # every precondition that turns on where the robot is standing passes
            # vacuously - a broken plan comes back clean.
            if room is not None and room not in g.rooms:
                g.rooms[room] = {"room_type": None}
            # The RSN keys its objects by category, so the name is the category - until
            # the plan is grounded onto instances, when `category` carries it instead.
            # Getting this wrong is quiet and total too: the category decides every
            # affordance, so an oven whose category reads `oven_wuinhm_0` is an oven that
            # does not open.
            g.see_object(name, info.get("category") or name, None, room=room)
            if info.get("probability") is not None:
                g.objects[name]["probability"] = float(info["probability"])
        # A task that says where something is - "the potato on the counter" - is a fact,
        # not a guess, and `populate` records it as a relation. It is a kinematic edge
        # here, which is what makes GRASP(potato) resolvable before the robot has looked.
        relations = {"ON_TOP": "on_top", "INSIDE": "object_inside"}
        for relation in graph.get("relations") or []:
            edge = relations.get(str(relation.get("relation", "")).upper())
            src, dst = relation.get("from"), relation.get("to")
            if edge is None or src not in g.objects or dst not in g.objects:
                continue
            g.add_edge(edge, src, dst, note="scene graph")
            if edge == "on_top":
                g.add_edge("under", dst, src, note="scene graph")
        return g

    @classmethod
    def from_scene_file(cls, scene, path="data/room_graphs.json"):
        with open(path) as f:
            return cls.from_room_graph(json.load(f)[scene])

    # ---------------------------------------------------------------- edge algebra

    @staticmethod
    def _key(edge_type, src, dst):
        if edge_type in SYMMETRIC:
            src, dst = sorted((src, dst))
        return (edge_type, src, dst)

    def add_edge(self, edge_type, src, dst, note=None):
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"unknown edge type {edge_type!r}; expected one of {EDGE_TYPES}")
        if src == dst:
            return False
        key = self._key(edge_type, src, dst)
        if key in self.edges:
            return False
        self.edges.add(key)
        self.log.append(("+", key, note))
        return True

    def remove_edge(self, edge_type, src, dst, note=None):
        key = self._key(edge_type, src, dst)
        if key not in self.edges:
            return False
        self.edges.discard(key)
        self.log.append(("-", key, note))
        return True

    def has_edge(self, edge_type, src, dst):
        return self._key(edge_type, src, dst) in self.edges

    def edges_of(self, edge_type, src=None, dst=None):
        """Every edge of a type, optionally pinned at one end.

        For symmetric types either endpoint matches, since the stored order is only a
        canonical form and carries no meaning.
        """
        out = []
        for t, a, b in self.edges:
            if t != edge_type:
                continue
            if edge_type in SYMMETRIC:
                if src is not None and src not in (a, b):
                    continue
                if dst is not None and dst not in (a, b):
                    continue
            else:
                if src is not None and a != src:
                    continue
                if dst is not None and b != dst:
                    continue
            out.append((a, b))
        return sorted(out)

    def clear_kinematic(self, name, note=None, keep_riders=False):
        """Drop the resting relations involving `name`.

        `keep_riders` is the difference between picking something up and re-observing it.
        Lifting a plate ends every relation in which the plate was the *supported* thing -
        it is no longer on the counter, no longer next to the potato - but anything
        resting *on* the plate travels with it, so `on_top(potato, plate)` survives. That
        edge is the whole reason the graph is worth keeping: it is what makes the later
        `GRASP(plate)` known to carry the potato to the oven.

        Re-observing clears both directions instead, because the predicates are about to
        be read again and whatever still holds will be written back.
        """
        # Whatever is resting on `name` travels with it, and so does the `under` edge
        # that says the same thing the other way round. Keeping one without the other
        # would leave the graph asserting that the potato is on the plate while the
        # plate is under nothing.
        riders = {a for t, a, b in self.edges if t == "on_top" and b == name}
        for t, a, b in [e for e in self.edges if e[0] in KINEMATIC]:
            if name not in (a, b):
                continue
            if keep_riders and t == "on_top" and b == name:
                continue
            if keep_riders and t == "under" and a == name and b in riders:
                continue
            self.remove_edge(t, a, b, note=note)

    # ---------------------------------------------------------------- observation

    def see_object(self, name, category, position, room=None):
        """Record that the robot has seen this object. Returns True the first time."""
        first = name not in self.objects
        self.objects[name] = {"category": category,
                              "position": None if position is None else list(position)}
        if room is not None:
            # room_inside is single-valued: an object is in one room.
            for _, old_room in self.edges_of("room_inside", src=name):
                if old_room != room:
                    self.remove_edge("room_inside", name, old_room, note="moved rooms")
            self.add_edge("room_inside", name, room, note="seen")
        return first

    # ---------------------------------------------------------------- the robot

    def place_robot(self, room=None, position=None):
        """Put the robot in the graph, and record which room it is in.

        `room_inside` is single-valued for the robot exactly as it is for an object: it is
        in one room, so moving rewrites the edge rather than adding a second.
        """
        record = self.objects.setdefault(ROBOT, {"category": "robot", "position": None})
        if position is not None:
            record["position"] = list(position)
        if room is not None:
            for _, old in self.edges_of("room_inside", src=ROBOT):
                if old != room:
                    self.remove_edge("room_inside", ROBOT, old, note="robot moved")
            self.add_edge("room_inside", ROBOT, room, note="robot moved")
        return record

    def carried_with(self, name):
        """`name` plus everything that travels with it, transitively.

        Grasping a plate lifts the potato resting on it; grasping a box lifts what is
        inside the box, and whatever is resting on *that*. The graph already records those
        relations, so carrying is a reachability question over the `on_top` and
        `object_inside` edges that point at the thing in the hand.
        """
        moving, stack = {name}, [name]
        while stack:
            here = stack.pop()
            for edge_type in ("on_top", "object_inside"):
                for rider, _ in self.edges_of(edge_type, dst=here):
                    if rider not in moving:
                        moving.add(rider)
                        stack.append(rider)
        return moving

    def carried(self, name):
        """Is this travelling with the robot - in the hand, or riding on what is?"""
        held = self.held_object()
        return held is not None and name in self.carried_with(held)

    def held_object(self):
        """What the robot has in its hand, from the graph. One hand, so at most one."""
        found = self.edges_of("holding", src=ROBOT)
        return found[0][1] if found else None

    def set_held(self, name, note=None):
        """Grasp or let go. Passing None empties the hand."""
        for _, old in self.edges_of("holding", src=ROBOT):
            self.remove_edge("holding", ROBOT, old, note=note)
        if name is not None:
            self.objects.setdefault(ROBOT, {"category": "robot", "position": None})
            self.add_edge("holding", ROBOT, name, note=note)

    def set_nearby(self, names, note=None):
        """Replace what the robot is standing at.

        Driving somewhere is the only thing that changes which objects are within arm's
        reach, so the set is replaced rather than added to - otherwise the robot
        accumulates a memory of everywhere it has ever been and every precondition about
        being beside something passes forever after.
        """
        wanted = set(names)
        for _, old in self.edges_of("nearby", src=ROBOT):
            if old not in wanted:
                self.remove_edge("nearby", ROBOT, old, note=note)
        if wanted:
            self.objects.setdefault(ROBOT, {"category": "robot", "position": None})
        for name in sorted(wanted):
            self.add_edge("nearby", ROBOT, name, note=note)

    def is_near(self, name):
        """Is the robot standing at this? The precondition every manipulation reads."""
        return self.has_edge("nearby", ROBOT, name)

    def near_objects(self):
        return sorted(b for _, b in self.edges_of("nearby", src=ROBOT))

    def object_names(self):
        """The objects the robot has *seen*. The robot is a node, not an observation."""
        return sorted(name for name in self.objects if name != ROBOT)

    def room_of(self, name):
        """Which room something is in, or None if the graph does not say.

        A carried object has no room edge - it has no room of its own while it is in the
        hand - so the answer is derived: whatever room the robot is in. That keeps one
        record of the fact instead of two.
        """
        found = self.edges_of("room_inside", src=name)
        if found:
            return found[0][1]
        if name != ROBOT and self.carried(name):
            return self.room_of(ROBOT)
        return None

    def position_of(self, name):
        rec = self.objects.get(name)
        return None if rec is None else rec.get("position")

    def knows(self, name):
        """Has the robot seen this object? The question NAVIGATE_TO has to ask."""
        return name in self.objects

    def by_category(self, category):
        return sorted(n for n, r in self.objects.items() if r.get("category") == category)

    def resolve(self, name):
        """A plan names `potato`; the scene may hold `potato_lqjear_0`. Accept either.

        Returns the instance name if it is unambiguous, else None. Categories with several
        instances seen are ambiguous on purpose - picking one silently is how a plan ends
        up acting on the wrong object.
        """
        if name in self.objects or name in self.rooms:
            return name
        matches = self.by_category(name)
        return matches[0] if len(matches) == 1 else None

    # ---------------------------------------------------------------- serialisation

    def to_dict(self):
        return {"rooms": self.rooms, "objects": self.objects,
                "edges": sorted(list(e) for e in self.edges)}

    @classmethod
    def from_dict(cls, d):
        g = cls()
        g.rooms = dict(d.get("rooms", {}))
        g.objects = dict(d.get("objects", {}))
        g.edges = {tuple(e) for e in d.get("edges", [])}
        return g

    def copy(self):
        g = WorldGraph()
        g.rooms = {k: dict(v) for k, v in self.rooms.items()}
        g.objects = {k: dict(v) for k, v in self.objects.items()}
        g.edges = set(self.edges)
        return g

    def summary(self):
        counts = {t: 0 for t in EDGE_TYPES}
        for t, _, _ in self.edges:
            counts[t] = counts.get(t, 0) + 1
        parts = [f"{len(self.rooms)} rooms", f"{len(self.object_names())} objects seen"]
        parts += [f"{t}={counts[t]}" for t in EDGE_TYPES if counts[t]]
        return ", ".join(parts)

    def __repr__(self):
        return f"<WorldGraph {self.summary()}>"


# -------------------------------------------------------------------- simulator side

def room_of_object(obj, scene=None):
    """Ground-truth room for a loaded object.

    Read only once the robot has seen the object - this stands in for recognising which
    room you are in when you spot something, which a real system gets from its own pose.

    Two sources, in order. `in_rooms` is the scene's own annotation and is authoritative
    where it exists, but it only exists for objects the scene file declares: anything
    spawned at runtime - the potato and the plate in the test plan - has none, and the
    first version of this returned None for exactly the two objects the search is meant
    to find. So fall back to the room segmentation under the object's own position, which
    is the more general answer anyway: an object is in the room it is standing in.
    """
    rooms = getattr(obj, "in_rooms", None)
    if rooms:
        return rooms[0]

    scene = scene if scene is not None else getattr(obj, "scene", None)
    seg = getattr(scene, "_seg_map", None)
    if seg is None:
        return None

    # Sample a small ring as well as the point itself. `get_room_instance_by_point`
    # returns None on a room-*boundary* pixel, and an object standing against a wall or on
    # furniture at the edge of a room lands on one often enough to matter: measured, the
    # plate ended on a coffee table in the living room and its lookup came back None, so
    # the graph silently kept the kitchen it had been seen in and reported the plate in the
    # wrong room. A miss should widen the search, not leave a stale answer standing.
    import math

    import torch as th

    try:
        xy = obj.get_position_orientation()[0][:2]
        x, y = float(xy[0]), float(xy[1])
    except Exception:
        return None

    offsets = [(0.0, 0.0)]
    for radius in (0.15, 0.30, 0.45):
        for k in range(8):
            angle = k * math.pi / 4.0
            offsets.append((radius * math.cos(angle), radius * math.sin(angle)))

    for dx, dy in offsets:
        try:
            room = seg.get_room_instance_by_point(
                th.tensor([x + dx, y + dy], dtype=th.float32))
        except Exception:
            continue
        if room:
            return room
    return None


def read_predicates(graph, names, scene, verbose=False):
    """Write the kinematic edges for objects seen for the **first** time.

    `names` is first sightings only, not everything currently in frame. The world is
    static unless the robot moves something, and when it does, the action's own graph
    edits say so - `GRASP` clears what the object was resting on, `PLACE_ON_TOP` writes
    what it now rests on. So an object already in the graph needs no re-derivation, and
    re-deriving it does active harm: reading the predicates again would first have to
    clear the existing edges, and anything the simulator declines to confirm is then lost.
    That is how `on_top(potato, plate)` disappeared - established by a placement, erased by
    the next look, because the potato is `visual_only` and touches nothing.

    Only pairs where *both* objects have been seen are considered. A predicate between
    something seen and something not seen is not knowledge the robot has, and admitting it
    would mean spotting a plate silently revealed the counter under it.
    """
    from omnigibson import object_states

    known = list(graph.object_names())
    written = 0

    for name in names:
        if name not in graph.objects:
            continue
        obj = scene.object_registry("name", name)
        if obj is None:
            continue
        # No collisions means no meaningful kinematic predicates: OnTop needs contact,
        # Inside and Under need adjacency ray casts against collision geometry. They would
        # all read False wherever the object actually is.
        if getattr(obj, "visual_only", False):
            continue
        for other in known:
            if other == name:
                continue
            other_obj = scene.object_registry("name", other)
            if other_obj is None or getattr(other_obj, "visual_only", False):
                continue
            # Both directions: the pair is only visited once, from the new object's side.
            for subject, target, s_obj, t_obj in ((name, other, obj, other_obj),
                                                  (other, name, other_obj, obj)):
                for edge_type, state_name in PREDICATE_EDGES:
                    state = getattr(object_states, state_name, None)
                    if state is None or state not in s_obj.states:
                        continue
                    try:
                        value = s_obj.states[state].get_value(t_obj)
                    except Exception:
                        # A predicate can legitimately refuse a pair - Inside needs the
                        # other object to have a volume, NextTo needs both to have
                        # extents. That is a "no", not an error worth stopping for.
                        continue
                    if value and graph.add_edge(edge_type, subject, target,
                                                note="observed"):
                        written += 1
                        if verbose:
                            print(f"      [graph] {edge_type}({subject}, {target})")
    return written
