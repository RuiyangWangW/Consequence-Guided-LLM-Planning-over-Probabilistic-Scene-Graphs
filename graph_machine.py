"""Execute a plan against the world graph, as graph edits.

Two callers, one effect model:

    offline   `check(graph, plan, goal)` runs the whole plan against a *copy* and reports
              whether it is applicable and whether it reaches the goal. No simulator, about
              a millisecond, so a bad plan is rejected before a long run proves it wrong.
    online    a machine built with `copy=False` is stepped as each primitive *succeeds* in
              the simulator, editing the live world graph. An action the robot performed is
              evidence as good as an observation - better, in fact, since the robot need
              not still be looking at the plate to know it put it on the table.

The two together are what make the graph complete: observation supplies what the robot
found, action effects supply what the robot did.


`planner.py:validate` already replays a plan against a flat world model - what is held,
which room the robot is in, what is open. This does the same job against the *graph*, and
the difference is what the graph can answer that the flat model cannot:

    PLACE_ON_TOP(plate) after GRASP(potato)   flat: legal, something is held
                                              graph: writes on_top(potato, plate), so the
                                                     later GRASP(plate) is known to carry
                                                     the potato with it

    GRASP(plate) while the plate is in a
    closed oven                               flat: legal, the robot is at the oven
                                              graph: object_inside(plate, oven) and the
                                                     oven is closed, so the hand cannot
                                                     reach it

Each action is a **temporal graph edit**: preconditions read the graph as it stands after
every earlier action, and effects rewrite it. Running the plan therefore produces a
sequence of graphs, and the last one is what the goal is checked against. Nothing here
touches Isaac - it runs in about a millisecond, so a plan can be rejected before a
fifteen-minute simulator run is spent proving it wrong.

Two ways a plan fails, reported separately because they mean different things:

    inapplicable   some action's preconditions do not hold. The plan is wrong.
    goal not met   every action ran, but the required edges are absent at the end. The
                   plan is executable and does not do the task.
"""

from planner import NOT_GRASPABLE, OPENABLE
from world_graph import WorldGraph

# The robot's hand is not a graph node - an object being carried has no resting relation
# to anything, which is exactly what `clear_kinematic` expresses. Held-ness is machine
# state instead, as are open and toggled, because they are properties of a node rather
# than relations between two.


class StepResult:
    """One action's outcome: whether it applied, and what it did to the graph."""

    def __init__(self, index, action, arg, ok, reason=None, edits=(), warnings=()):
        self.index = index
        self.action = action
        self.arg = arg
        self.ok = ok
        self.reason = reason
        self.edits = list(edits)
        self.warnings = list(warnings)

    def __repr__(self):
        head = f"{self.index + 1:2d}. {self.action}" + (f"({self.arg})" if self.arg else "()")
        if not self.ok:
            return f"{head:34s} FAIL  {self.reason}"
        return f"{head:34s} ok    {', '.join(self.edits) or 'no graph change'}"


class Outcome:
    """What running a whole plan produced."""

    def __init__(self, steps, graph, goal, goal_met, missing, failed_at):
        self.steps = steps
        self.graph = graph
        self.goal = goal
        self.goal_met = goal_met
        self.missing = missing
        self.failed_at = failed_at

    @property
    def ok(self):
        return self.failed_at is None and self.goal_met

    def report(self):
        lines = [repr(s) for s in self.steps]
        for s in self.steps:
            for w in s.warnings:
                lines.append(f"    warning (step {s.index + 1}): {w}")
        if self.failed_at is not None:
            lines.append(f"\nplan is inapplicable at step {self.failed_at + 1}: "
                         f"{self.steps[self.failed_at].reason}")
        elif self.goal_met:
            lines.append(f"\nplan completes the task: all {len(self.goal)} goal edges hold")
        else:
            lines.append("\nplan runs to completion but does NOT do the task; missing:")
            lines += [f"    {t}({a}, {b})" for t, a, b in self.missing]
        lines.append(f"final graph: {self.graph.summary()}")
        return "\n".join(lines)


class GraphMachine:
    """Applies atomic actions to a `WorldGraph` and checks each one's preconditions.

    `allow_search` says whether NAVIGATE_TO may target an object the graph has never
    seen. It defaults to True because that is what the low-level navigation controller
    does: it drives to the object's room and searches for it. With it False the machine
    demands a fully observed graph, which is the right setting for checking a plan
    *after* exploration rather than before.
    """

    def __init__(self, graph, allow_search=True, verbose=False, copy=True):
        # `copy=False` runs the machine *on* a live graph rather than a snapshot of it,
        # which is how the same effect model serves two jobs. Offline it checks a plan
        # against a copy and leaves the original alone. Online it edits the world graph as
        # each primitive succeeds, so the graph knows the plate is on the table because
        # the robot put it there - not only if the robot later happens to look at it.
        self.graph = graph.copy() if copy else graph
        self.allow_search = allow_search
        self.verbose = verbose
        self.location = None     # room the robot is in
        self.held = None         # object in the hand
        self.open = {}           # object -> bool
        self.toggled = {}        # object -> bool

    # ------------------------------------------------------------------ helpers

    def _category(self, name):
        rec = self.graph.objects.get(name)
        return (rec or {}).get("category") or name

    def _resolve(self, name):
        """Map a plan's name onto a graph node, or None if it is not there."""
        return self.graph.resolve(name)

    def _room_of(self, name):
        if name in self.graph.rooms:
            return name
        return self.graph.room_of(name)

    def _reachable(self, room):
        """Can the robot get from where it is to `room`?

        Anywhere is reachable before the first navigation - the robot has not committed
        to a position yet. After that the room graph has to connect them, though not
        necessarily directly: a route through other rooms is fine and is what the
        low-level controller would drive.
        """
        if self.location is None or room is None or room == self.location:
            return True
        seen, stack = {self.location}, [self.location]
        while stack:
            here = stack.pop()
            for a, b in self.graph.edges_of("room_connect", src=here):
                nxt = b if a == here else a
                if nxt == room:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def _blocked_by_container(self, name):
        """Is `name` inside something that is currently closed?"""
        for _, container in self.graph.edges_of("object_inside", src=name):
            if self._category(container) in OPENABLE and not self.open.get(container, False):
                return container
        return None

    def _carried_with(self, name):
        """`name` plus everything that travels with it, transitively.

        Grasping a plate lifts the potato resting on it; grasping a box lifts what is
        inside the box, and whatever is resting on *that*. The graph already records those
        relations, so carrying is a reachability question over `on_top` and
        `object_inside` edges pointing at the thing in the hand.
        """
        moving, stack = {name}, [name]
        while stack:
            here = stack.pop()
            for edge_type in ("on_top", "object_inside"):
                for rider, _ in self.graph.edges_of(edge_type, dst=here):
                    if rider not in moving:
                        moving.add(rider)
                        stack.append(rider)
        return moving

    def _move_to_room(self, names, room):
        """Rewrite room_inside for everything in `names`. Returns what changed."""
        edits = []
        for name in sorted(names):
            current = self.graph.room_of(name)
            if current == room:
                continue
            if current is not None:
                self.graph.remove_edge("room_inside", name, current, note="carried")
            self.graph.add_edge("room_inside", name, room, note="carried")
            edits.append(f"room_inside({name}, {room})")
        return edits

    def _require_here(self, name):
        """The robot has to be in the object's room to act on it."""
        room = self._room_of(name)
        if room is None:
            return None            # room unknown; nothing to contradict
        if self.location != room:
            return (f"robot is in {self.location or 'no room yet'} but '{name}' is in "
                    f"{room}; NAVIGATE_TO it first")
        return None

    # ------------------------------------------------------------------ the actions

    def step(self, index, action, arg):
        edits, warnings = [], []

        def fail(reason):
            return StepResult(index, action, arg, False, reason=reason, warnings=warnings)

        def ok():
            return StepResult(index, action, arg, True, edits=edits, warnings=warnings)

        # --- arity, the same rule the validator applies -----------------------------
        if action == "RELEASE":
            if arg:
                return fail("RELEASE takes no argument")
        elif not arg:
            return fail(f"{action} needs an object")

        name = self._resolve(arg) if arg else None

        # --- NAVIGATE_TO -----------------------------------------------------------
        if action == "NAVIGATE_TO":
            if name is None:
                if not self.allow_search:
                    return fail(f"'{arg}' has not been seen and search is disabled")
                # The low-level controller handles this: drive to the object's room and
                # search it. The machine cannot know which room, so it admits the object
                # without one and leaves `location` where it is.
                self.graph.see_object(arg, arg, None)
                warnings.append(f"'{arg}' unseen; the navigation controller must search "
                                f"for it before this step can run")
                edits.append(f"+node {arg} (unseen)")
                return ok()
            room = self._room_of(name)
            if room is not None and not self._reachable(room):
                return fail(f"{room} is not reachable from {self.location}")
            if room is not None:
                self.location = room
                edits.append(f"robot -> {room}")
                # Whatever is in the hand travels with the robot, and so does anything
                # riding on it - carrying the plate carries the potato on the plate.
                if self.held is not None:
                    edits += self._move_to_room(self._carried_with(self.held), room)
            return ok()

        # RELEASE is exempt: it takes no argument, so `name` is None by construction and
        # there is no object for the graph to have seen. Every other primitive names one.
        if name is None and action != "RELEASE":
            return fail(f"'{arg}' is not in the graph; the robot has never seen it")

        # --- GRASP -----------------------------------------------------------------
        if action == "GRASP":
            if self.held is not None:
                return fail(f"already holding '{self.held}'; place or release it first")
            if self._category(name) in NOT_GRASPABLE:
                return fail(f"'{name}' is fixed furniture and cannot be picked up")
            problem = self._require_here(name)
            if problem:
                return fail(problem)
            shut = self._blocked_by_container(name)
            if shut:
                return fail(f"'{name}' is inside '{shut}', which is closed; OPEN it first")
            # Everything resting on the grasped object comes with it, and everything it
            # was resting on is no longer supporting it.
            carried = [a for t, a, b in sorted(self.graph.edges)
                       if t == "on_top" and b == name]
            self.graph.clear_kinematic(name, note="grasped", keep_riders=True)
            edits.append(f"-kinematic({name})")
            if carried:
                warnings.append(f"'{', '.join(carried)}' rode along on '{name}'")
            self.held = name
            edits.append(f"held = {name}")
            return ok()

        # --- PLACE_ON_TOP / PLACE_INSIDE -------------------------------------------
        if action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
            if self.held is None:
                return fail("nothing in the hand to place")
            if self.held == name:
                return fail(f"cannot place '{name}' on itself")
            problem = self._require_here(name)
            if problem:
                return fail(problem)
            if action == "PLACE_INSIDE":
                if self._category(name) in OPENABLE and not self.open.get(name, False):
                    warnings.append(f"'{name}' is closed; PLACE_INSIDE will fail unless "
                                    f"it is opened first")
                self.graph.add_edge("object_inside", self.held, name, note="placed")
                edits.append(f"object_inside({self.held}, {name})")
            else:
                self.graph.add_edge("on_top", self.held, name, note="placed")
                self.graph.add_edge("under", name, self.held, note="placed")
                edits.append(f"on_top({self.held}, {name})")
            room = self._room_of(name)
            if room is not None:
                edits += self._move_to_room(self._carried_with(self.held), room)
            self.held = None
            edits.append("held = none")
            return ok()

        # --- RELEASE ---------------------------------------------------------------
        if action == "RELEASE":
            if self.held is None:
                return fail("nothing in the hand to release")
            if self.location is not None:
                edits += self._move_to_room(self._carried_with(self.held), self.location)
            edits.append(f"released {self.held} (now on the floor)")
            self.held = None
            return ok()

        # --- OPEN / CLOSE ----------------------------------------------------------
        if action in ("OPEN", "CLOSE"):
            problem = self._require_here(name)
            if problem:
                return fail(problem)
            want = action == "OPEN"
            if self.open.get(name) == want:
                warnings.append(f"'{name}' is already {'open' if want else 'closed'}")
            self.open[name] = want
            edits.append(f"{name}.open = {want}")
            return ok()

        # --- TOGGLE_ON / TOGGLE_OFF ------------------------------------------------
        if action in ("TOGGLE_ON", "TOGGLE_OFF"):
            problem = self._require_here(name)
            if problem:
                return fail(problem)
            want = action == "TOGGLE_ON"
            if self.toggled.get(name) == want:
                warnings.append(f"'{name}' is already toggled {'on' if want else 'off'}")
            self.toggled[name] = want
            edits.append(f"{name}.toggled = {want}")
            return ok()

        return fail(f"unknown action {action!r}")

    # ------------------------------------------------------------------ whole plans

    def run(self, plan, goal=()):
        """Apply every action in order, then check the goal edges.

        `plan` is a list of (action, argument-or-None). `goal` is a list of
        (edge_type, src, dst) that must be present in the final graph.
        """
        steps, failed_at = [], None
        for i, (action, arg) in enumerate(plan):
            result = self.step(i, action, arg)
            steps.append(result)
            if self.verbose:
                print(repr(result))
            if not result.ok:
                failed_at = i
                break

        missing = []
        if failed_at is None:
            for edge_type, src, dst in goal:
                a = self._resolve(src) or src
                b = self._resolve(dst) or dst
                if not self.graph.has_edge(edge_type, a, b):
                    missing.append((edge_type, src, dst))
        else:
            missing = list(goal)

        return Outcome(steps, self.graph, list(goal), failed_at is None and not missing,
                       missing, failed_at)


def check(graph, plan, goal=(), allow_search=True, verbose=False):
    """Convenience: build a machine, run the plan, hand back the outcome."""
    return GraphMachine(graph, allow_search=allow_search, verbose=verbose).run(plan, goal)
