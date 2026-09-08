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

import json
import os
import re

from object_names import match
from planner import CONFERS, MUST_SWITCH_OFF, NOT_GRASPABLE, OPENABLE, TOGGLEABLE
from world_graph import ROBOT, WorldGraph

_CATALOGUE = None


def _object_categories():
    """Every object category BEHAVIOR ships, cached; empty if the catalogue is missing.

    Category-level and scene-free - it says `bar` and `toilet` are things that exist in the
    world, never that this house has one. It is here to settle the handful of words that
    name both a room and an object.
    """
    global _CATALOGUE
    if _CATALOGUE is None:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "vocab.json")) as handle:
                vocab = json.load(handle)
            _CATALOGUE = frozenset(vocab.get("object_categories", ())) | frozenset(
                vocab.get("merged_categories", ()))
        except (OSError, ValueError):
            _CATALOGUE = frozenset()
    return _CATALOGUE

# The robot is a node, and the two facts about it that relate it to something else are
# edges: `room_inside(robot, kitchen_0)` and `holding(robot, potato)`. `location` and
# `held` below are views onto those edges rather than a second copy of them - two records
# of where the robot is, is two records that can disagree. What an object in the hand is
# *not* is resting on anything, which is what `clear_kinematic` expresses and why
# `holding` is not one of the kinematic edges.
#
# `open` and `toggled` stay off the graph: they are properties of a single node rather
# than relations between two, and an edge is the wrong shape for them.


class StepResult:
    """One action's outcome: whether it applied, and what it did to the graph."""

    def __init__(self, index, action, arg, ok, reason=None, edits=(), warnings=(),
                 fault=None):
        self.index = index
        self.action = action
        self.arg = arg
        self.ok = ok
        self.reason = reason
        # `reason` is the sentence a human (or an LLM) reads; `fault` is the same refusal
        # as a pair, `(kind, object)`, for code to branch on. They are set together at
        # every refusal so the two cannot drift, and the pair is what lets the repair loop
        # and the complaint writer key off the precondition that actually failed rather
        # than grepping the English.
        self.fault = fault
        self.edits = list(edits)
        self.warnings = list(warnings)

    def __repr__(self):
        head = f"{self.index + 1:2d}. {self.action}" + (f"({self.arg})" if self.arg else "()")
        if not self.ok:
            return f"{head:34s} FAIL  {self.reason}"
        return f"{head:34s} ok    {', '.join(self.edits) or 'no graph change'}"


class Outcome:
    """What running a whole plan produced."""

    def __init__(self, steps, graph, goal, goal_met, missing, failed_at,
                 left_open=(), left_on=()):
        self.steps = steps
        self.graph = graph
        self.goal = goal
        self.goal_met = goal_met
        self.missing = missing
        self.failed_at = failed_at
        # Doors the plan opened and never shut, and hazardous appliances it turned on and
        # never turned off. A plan can be applicable and reach its goal and still walk away
        # from an open fridge and a lit hob, which is not a plan anyone should run. A lamp
        # left burning is not in that class - see `planner.MUST_SWITCH_OFF`.
        self.left_open = list(left_open)
        self.left_on = list(left_on)

    @property
    def safe(self):
        """Did the plan put back what it disturbed?"""
        return not (self.left_open or self.left_on)

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
        if self.left_open:
            lines.append("left open: " + ", ".join(self.left_open))
        if self.left_on:
            lines.append("left switched on: " + ", ".join(self.left_on))
        lines.append(f"final graph: {self.graph.summary()}")
        return "\n".join(lines)


class GraphMachine:
    """Applies atomic actions to a `WorldGraph` and checks each one's preconditions.

    The specification, in full. `near(x)` is "the robot is standing at x", and it is a
    graph fact: the `nearby(robot, x)` edge `NAVIGATE_TO` writes. It used to be
    `room_inside(robot) == room_inside(x)`, and a room turned out to be far too coarse -
    measured against the 2-D simulator, a fifth of the plans this model accepted were
    undrivable and every one of them was a plan that drove to one thing in the kitchen and
    then acted on another thing three metres away in the same kitchen. With the edge, that
    whole class is gone. `open(x)` and `toggled(x)` are node properties the machine tracks;
    `openable(x)` and `switchable(x)` are affordances, read from the tracked state where it
    exists and from the category lists in `planner.py` otherwise.

    | action              | preconditions                                    |
    | ------------------- | ------------------------------------------------ |
    | `NAVIGATE_TO(x)`    | if `inside(x, c)` and `c` has a door, then `open(c)` |
    | `RELEASE()`         | none                                             |
    | `GRASP(x)`          | not holding anything; `near(x)`; if `inside(x, c)` then `open(c)`; `graspable(x)` |
    | `TOGGLE_ON/OFF(x)`  | `near(x)`; `switchable(x)`                       |
    | `OPEN/CLOSE(x)`     | `near(x)`; `openable(x)`                         |
    | `PLACE_INSIDE(x)`   | `near(x)`; `open(x)` if `openable(x)`; holding something |
    | `PLACE_ON_TOP(x)`   | `near(x)`; holding something                     |

And what each one does. "The stack" is the held object plus everything riding on it,
    which `carried_with` walks over `on_top` and `object_inside`.

    | action | in words | in edges |
    | --- | --- | --- |
    | `NAVIGATE_TO(x)` | the robot is now standing at x, and at whatever is on or inside it | `room_inside(robot)` := room of x; `nearby(robot)` := x and its contents, plus whatever is in the hand |
    | `GRASP(x)` | the robot is holding x, and x travels with it | `+holding(robot, x)`; drop x's kinematic edges but keep its riders; drop `room_inside` for the whole stack, so its room derives from the robot's |
    | `RELEASE()` | the hand is empty; what it held stays in the room the robot released it in, resting on nothing | `-holding`; `room_inside(stack)` := the robot's room |
    | `PLACE_ON_TOP(x)` | what was held is now on top of x, and is no longer held | `+on_top(held, x)`, `+under(x, held)`; `-holding`; `room_inside(stack)` := room of x |
    | `PLACE_INSIDE(x)` | what was held is now inside x, and is no longer held | `+object_inside(held, x)`; `-holding`; `room_inside(stack)` := room of x |
    | `OPEN/CLOSE(x)` | x is now open / shut | `open(x)` := True / False |
    | `TOGGLE_ON/OFF(x)` | x is now on / off | `toggled(x)` := True / False |

    Three of those are worth a sentence. `GRASP` keeps the riders - lifting a plate does
    not put down the potato on it - and that surviving `on_top(potato, plate)` is what
    makes a later `GRASP(plate)` known to carry the potato with it. It also *drops*
    `room_inside` rather than rewriting it, because a carried object has no room of its
    own; the placements write it back. And `PLACE_INSIDE` writes only `object_inside`,
    where `PLACE_ON_TOP` writes both directions: there is no "contains" edge, so one
    direction is the whole record.

    One thing is checked that is not a precondition and is not in the table, because it is
    about the plan being well formed rather than about the world: an action's arity -
    `RELEASE` takes no argument, the rest take one.

    The line the machine draws is between knowing what a *kind* of thing is and knowing
    what is true of *this house*. Whether a fridge has a door is a fact about fridges, and
    a robot that recognises one knows it; where the fridge is, and whether it is shut right
    now, are things the belief graph has to guess. So `planner.OPENABLE` is consulted and
    the guess is not second-guessed.

    That one fact has to cut both ways or it is not knowledge. A container with a door must
    be opened before anything is taken out of it or put into it; a bowl, a sink, an
    open-topped bin has nothing to open, so `PLACE_INSIDE` needs no `OPEN` first *and*
    `OPEN` on it is refused outright. Excusing the one while permitting the other was the
    incoherent middle: it claimed the machine could not know a bin has no lid at the moment
    it refused, and did know at the moment it excused.

    The same holds for the other two affordances, and for the same reason: a fridge cannot
    be lifted and a countertop has no switch, so `GRASP` and `TOGGLE` refuse them. What is
    refused is always what the *kind* of thing cannot do, never what this particular one
    happens not to be doing.

    `planner.CONFERS` is the same kind of category fact, read the same way: a dishwasher
    washes what is inside it, so a goal can ask for a clean plate.

    """

    def __init__(self, graph, verbose=False, copy=True):
        # `copy=False` runs the machine *on* a live graph rather than a snapshot of it,
        # which is how the same effect model serves two jobs. Offline it checks a plan
        # against a copy and leaves the original alone. Online it edits the world graph as
        # each primitive succeeds, so the graph knows the plate is on the table because
        # the robot put it there - not only if the robot later happens to look at it.
        self.graph = graph.copy() if copy else graph
        self.verbose = verbose
        self.open = {}           # object -> bool
        self.toggled = {}        # object -> bool
        # What running an appliance has done to things. A task that says "heat the pie" is
        # not finished by the pie arriving on the table, and "load the dishwasher and run
        # it" is not finished by loading it - both need the machine to have run, and
        # neither `on_top` nor `toggled` can say so. Append-only, like `opened`: a plan that
        # cooked something and then moved it still cooked it.
        self.states = {name: set() for name in CONFERS}
        # Every object an executed action named. A goal written in the instruction's words
        # is read against these first - the plan is what says which cabinet was meant.
        self.touched = set()
        # The room ids the pipeline holds, plus the bare word each one is built from -
        # `kitchen_0` gives `kitchen`. Both spellings are the planner's own vocabulary:
        # `scene_graph.format_for_llm` prints the ids into its prompt under "Rooms:", and
        # the word is what is left when the instance number is taken off. Nothing here
        # comes from the scene's true contents.
        #
        # Exact equality on both, and no fuzzy matching in either direction. Fuzzy is the
        # obvious way to do this and it is wrong: `object_names.same` calls `kitchen_table`
        # a `kitchen`, `bathroom_sink` a `bathroom`, and `bar_soap` a `bar` - and `bar_soap`
        # is a real object in this benchmark while `bar` is a real BEHAVIOR-1K room type.
        from scene_graph import ROOM_SYNONYMS

        types = {re.sub(r"_\d+$", "", room) for room in self.graph.rooms}
        self.room_words = set(self.graph.rooms) | types | {
            word for word, room_type in ROOM_SYNONYMS.items() if room_type in types}
        # What *this plan* has ever opened or switched on. Append-only: a plan that opens
        # a door and shuts it still opened it, which is what lets a caller ask whether the
        # goal requires putting it back. `left_open` / `left_on` filter these by the state
        # at the end, so they are the ones the plan actually walked away from. Discarding
        # on CLOSE was tried and made that question unanswerable - a tidy plan looked like
        # a plan that had touched nothing.
        self.opened = set()
        self.switched_on = set()

    # ------------------------------------------------------------------ robot state

    # Both of these live in the graph. Nothing is initialised here: a graph handed to the
    # machine may already say where the robot is and what it is carrying, and overwriting
    # that with None would throw away the state the caller just set up.

    @property
    def location(self):
        """The room the robot is in, or None if the graph does not say."""
        return self.graph.room_of(ROBOT)

    @location.setter
    def location(self, room):
        if room is None:
            for _, old in self.graph.edges_of("room_inside", src=ROBOT):
                self.graph.remove_edge("room_inside", ROBOT, old, note="robot location lost")
        else:
            self.graph.place_robot(room)

    @property
    def held(self):
        """The object in the hand, or None."""
        return self.graph.held_object()

    @held.setter
    def held(self, name):
        self.graph.set_held(name, note="grasped" if name else "let go")

    # ------------------------------------------------------------------ helpers

    def _is_room(self, name):
        """Does this name a room rather than a thing to act on?

        A handful of words name both - `bar`, `toilet` and `locker` are BEHAVIOR object
        categories *and* room types - and for those the object wins, because a task that
        scrubs a toilet must be able to say `toilet`.

        The tie-break asks the object catalogue, not this graph's nodes. Asking the graph
        is the obvious thing and it is exactly wrong: `graph.objects` is filled from the
        extractor's own words with nothing filtering room names out, so an extractor that
        answers `kitchen` creates a node called `kitchen` - and the check that exists to
        catch `kitchen` would then switch itself off, precisely when it was needed. The
        catalogue is fixed, scene-free, and nothing upstream can write to it.
        """
        if name not in self.room_words:
            return False
        catalogue = _object_categories()
        if catalogue:
            return name not in catalogue
        # No catalogue to consult: fall back to the graph, which is poisonable but better
        # than refusing every `toilet` in the benchmark.
        return name not in self.graph.objects

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

    def _openable(self, name):
        """Does this have a door - per BDDL's category annotations, or per the plan?

        Whether a fridge has a door is a fact about fridges, not about this house, and a
        robot that recognises one knows it. That makes it different in kind from where the
        fridge is or whether it is currently shut, which are the things the belief graph
        has to guess at. `name in self.open` comes first so a plan that has already opened
        something is believed over the table.
        """
        return name in self.open or self._category(name) in OPENABLE

    def _switchable(self, name):
        """Has a switch to flip, per BDDL - or the plan has already flipped it."""
        return name in self.toggled or self._category(name) in TOGGLEABLE

    def _contents(self, name):
        """Everything in or on this object, and everything riding on those."""
        found, frontier = set(), [name]
        while frontier:
            here = frontier.pop()
            for edge in ("object_inside", "on_top"):
                for item, _ in self.graph.edges_of(edge, dst=here):
                    if item not in found:
                        found.add(item)
                        frontier.append(item)
        return found


    def _inside_only(self, name):
        """What is *inside* this object, and inside those - not what is resting on it.

        `_contents` walks `on_top` as well, which is right for reach: standing at a table
        puts the robot within reach of what is on it. It is wrong for an appliance. An oven
        cooks what is inside it and not the pie someone left on the lid, and using
        `_contents` here credited exactly that - a plan that wrote PLACE_ON_TOP(washer)
        instead of PLACE_INSIDE(washer) was scored as having washed the towels.
        """
        found, frontier = set(), [name]
        while frontier:
            here = frontier.pop()
            for item, _ in self.graph.edges_of("object_inside", dst=here):
                if item not in found:
                    found.add(item)
                    frontier.append(item)
        return found

    def _blocked_by_container(self, name):
        """Is `name` inside something that is currently closed?"""
        for _, container in self.graph.edges_of("object_inside", src=name):
            if self._category(container) in OPENABLE and not self.open.get(container, False):
                return container
        return None

    def _carried_with(self, name):
        """`name` plus everything that travels with it - the graph answers this."""
        return self.graph.carried_with(name)

    def _move_to_room(self, names, room):
        """Rewrite room_inside for everything in `names`. Returns what changed.

        Reads the *stored* edge, not `room_of`. `room_of` derives a carried object's room
        from the robot's, so asking it here says "already in kitchen_0" about an object
        that has no room edge at all - and the restore on putting it down silently does
        nothing. Measured: the potato came out of a completed plan with no room.
        """
        edits = []
        for name in sorted(names):
            stored = self.graph.edges_of("room_inside", src=name)
            current = stored[0][1] if stored else None
            if current == room:
                continue
            if current is not None:
                self.graph.remove_edge("room_inside", name, current, note="carried")
            self.graph.add_edge("room_inside", name, room, note="carried")
            edits.append(f"room_inside({name}, {room})")
        return edits

    def _require_here(self, name):
        """The robot has to be standing at the object to act on it.

        Read off the `nearby` edges `NAVIGATE_TO` writes, not off room membership. The room
        was the finest thing this model used to know about where the robot was, and a room
        is not fine enough: measured against the 2-D simulator over 200 plans, a fifth of
        the plans it accepted were undrivable, and **every one** was this - the plan drove
        to one thing in the kitchen and then acted on another thing in the same kitchen,
        three metres away, without driving to it. Same room, so the check passed.
        """
        if self.graph.is_near(name):
            return None
        standing_at = self.graph.near_objects()
        where = f"standing at {', '.join(standing_at)}" if standing_at else "not at anything"
        return f"robot is {where}, not at '{name}'; NAVIGATE_TO it first"

    def _within_reach_of(self, name):
        """`name` and whatever is on or inside it - all of it is within arm's reach.

        Driving to a counter puts what is on the counter in reach; driving to an open oven
        puts what is in the oven in reach. Without this the robot would have to navigate to
        the potato *and* to the counter it is on, which is not how reaching works and would
        reject plans that are fine.
        """
        return self.graph.carried_with(name)

    # ------------------------------------------------------------------ the actions

    def step(self, index, action, arg=None):
        # `arg` defaults because RELEASE genuinely has none, and making every caller pass
        # an explicit None for the one action that takes no object is a wart.
        edits, warnings = [], []

        def fail(reason, fault=None):
            return StepResult(index, action, arg, False, reason=reason,
                              warnings=warnings, fault=fault)

        def ok():
            return StepResult(index, action, arg, True, edits=edits, warnings=warnings)

        # --- arity, the same rule the validator applies -----------------------------
        if action == "RELEASE":
            if arg:
                return fail("RELEASE takes no argument", ("arity", arg))
        elif not arg:
            return fail(f"{action} needs an object", ("arity", None))

        # A room is not something any primitive acts on. This sits before resolution and
        # before every branch, because the alternative is worse in two different ways:
        # `NAVIGATE_TO(kitchen)` used to be admitted as an object node called `kitchen`
        # that corresponds to nothing, and `PLACE_ON_TOP(kitchen_0)` used to be refused for
        # standing in the wrong place - a true sentence about a step whose real problem is
        # that it names a room.
        if arg and self._is_room(arg):
            return fail(f"'{arg}' is a room, not an object; {action} takes the object you "
                        f"are acting on - name the thing in {arg}, not the room",
                        ("room", arg))

        name = self._resolve(arg) if arg else None
        if name:
            self.touched.add(name)

        # --- NAVIGATE_TO -----------------------------------------------------------
        if action == "NAVIGATE_TO":
            # A room is not a destination for this primitive. NAVIGATE_TO takes the
            # object the robot is about to act on, and a plan that drives to `kitchen_0`
            # has not said which thing it means to reach - the next GRASP then has no
            # `nearby` edge and fails several steps later, where the cause is hard to see.
            #
            # Refusing here is what makes it repairable: the complaint names the room and
            # the loop rewrites the step to the object. Admitting it instead invented a
            # node called `kitchen_0` and validated a plan the simulator then refused on
            # its very first step - 23 of the 27 plans that passed validation and died when
            # driven.
            if name is None:
                return fail(f"'{arg}' is not one of the objects this task is about; use "
                            f"the names listed above", ("unknown", arg))
            # Driving to something sealed inside a shut container is a step the simulator
            # cannot execute. It has to *see* the object to go to it, and a bottle behind a
            # closed fridge door is invisible: `sim2d` searches every believed room, finds
            # nothing, and the run dies on the step. The machine used to allow it, because
            # NAVIGATE_TO had no preconditions at all - and that gap was 9 of the 4B's 11
            # plans that passed validation and then failed when driven, every one of them
            # reaching for something behind a door it had not opened yet.
            #
            # Refusing here makes it repairable instead of fatal, and needs no new rule:
            # `repair.py` answers `closed` by inserting OPEN(container), the `not_near`
            # rule then supplies the drive to the container, and the two compose into
            # "go to the container, open it, then come to the object".
            shut = self._blocked_by_container(name)
            if shut is not None:
                return fail(f"'{name}' is inside '{shut}', which is closed - open "
                            f"'{shut}' before driving to '{name}'", ("closed", shut))
            # No other preconditions. Whether the robot can actually get there is a question
            # about floor, and the room graph's answer to it is too coarse to be worth
            # refusing a plan over - `sim2d` runs A* over the eroded map and answers it
            # properly. An unreachable room is reported there, where it is known.
            room = self._room_of(name)
            if room is not None and not self._reachable(room):
                warnings.append(f"{room} may not be reachable from {self.location}")
            if room is not None:
                self.location = room
                edits.append(f"robot -> {room}")
            # Standing at the object, and at whatever is on or inside it. Whatever is in
            # the hand stays in reach too - it is in the hand. The set is *replaced*: the
            # robot is no longer beside what it drove away from.
            reach = set(self._within_reach_of(name))
            if self.held is not None:
                reach |= self._carried_with(self.held)
                edits.append(f"carrying {', '.join(sorted(self._carried_with(self.held)))}")
            self.graph.set_nearby(reach, note="navigated")
            edits.append(f"nearby({', '.join(sorted(reach))})")
            return ok()

        # A name that does not resolve is admitted rather than refused. Every object a
        # task needs is placed by the RSN before planning starts, so "the robot has never
        # seen it" was never true - what it really meant was that the plan spelled the
        # object differently from the graph. Refusing that made the machine complain about
        # its own vocabulary, which is not something the plan can fix.

        if name is None and action != "RELEASE":
            return fail(f"'{arg}' is not one of the objects this task is about; use the "
                        f"names listed above", ("unknown", arg))

        # --- GRASP -----------------------------------------------------------------
        if action == "GRASP":
            # 1. the hand is empty
            if self.held is not None:
                return fail(f"already holding '{self.held}'; place or release it first",
                            ("holding", self.held))
            # 2. the robot is beside it
            problem = self._require_here(name)
            if problem:
                return fail(problem, ("not_near", name))
            # 3. if it is inside something, that something is open
            # A robot can no more lift a fridge than switch on a countertop. Same fact as
            # the door, in a different suit: what a *kind* of thing affords is known, where
            # this one is and what state it is in is believed.
            if self._category(name) in NOT_GRASPABLE:
                return fail(f"'{name}' is fixed furniture and cannot be picked up",
                            ("not_graspable", name))
            shut = self._blocked_by_container(name)
            if shut:
                return fail(f"'{name}' is inside '{shut}', which is closed; OPEN it "
                            f"first", ("closed", shut))
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
            # It is in the hand, so it stays in reach however the graph is rearranged.
            self.graph.set_nearby(set(self.graph.near_objects())
                                  | self._carried_with(name), note="grasped")
            # A thing in the hand is not in a room of its own. Its room is the robot's,
            # and `room_of` derives it, so the stored edge would only be a duplicate to
            # keep in step on every drive. Placing it down writes the edge back.
            for moving in sorted(self._carried_with(name)):
                for _, room in self.graph.edges_of("room_inside", src=moving):
                    self.graph.remove_edge("room_inside", moving, room, note="picked up")
                    edits.append(f"-room_inside({moving})")
            return ok()

        # --- PLACE_ON_TOP / PLACE_INSIDE -------------------------------------------
        if action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
            # 1. the robot is beside the target
            problem = self._require_here(name)
            if problem:
                return fail(problem, ("not_near", name))
            # 2. something is in the hand
            if self.held is None:
                return fail("nothing in the hand to place", ("empty_hand", None))
            if action == "PLACE_INSIDE":
                # 3. if it has a door, that door is open. Putting something into a shut
                # oven is exactly as impossible as taking something out of one, which
                # GRASP already refuses via `_blocked_by_container`. Not knowing the state
                # is not the same as knowing it is open, so it fails too - a plan that
                # never opened the container has not established what this step needs.
                # Something with no door at all - a bowl, a sink - has nothing to open.
                # Only a container that has a door has to be opened. A bowl or a sink has
                # nothing to open, and demanding it would ask for a step the robot cannot
                # perform - the same knowledge that refuses `OPEN(bowl)` below excuses the
                # plan from needing it here, which is what makes the pair coherent.
                if self._openable(name) and not self.open.get(name, False):
                    known = "is closed" if name in self.open else "has not been opened"
                    return fail(f"'{name}' {known}; OPEN it before placing inside",
                                ("not_open", name))
                self.graph.add_edge("object_inside", self.held, name, note="placed")
                edits.append(f"object_inside({self.held}, {name})")
            else:
                self.graph.add_edge("on_top", self.held, name, note="placed")
                self.graph.add_edge("under", name, self.held, note="placed")
                edits.append(f"on_top({self.held}, {name})")
            room = self._room_of(name)
            if room is not None:
                edits += self._move_to_room(self._carried_with(self.held), room)
            # Just put down at arm's length, so still in reach.
            self.graph.set_nearby(set(self.graph.near_objects())
                                  | self._carried_with(self.held), note="placed")
            self.held = None
            edits.append("held = none")
            return ok()

        # --- RELEASE ---------------------------------------------------------------
        if action == "RELEASE":
            # No preconditions. Opening an empty hand is not an error, it is a step that
            # does nothing, and a plan is not wrong for containing one.
            if self.held is None:
                warnings.append("nothing in the hand; RELEASE does nothing here")
                return ok()
            if self.location is not None:
                edits += self._move_to_room(self._carried_with(self.held), self.location)
            # Dropped at the robot's feet, so still in reach.
            self.graph.set_nearby(set(self.graph.near_objects())
                                  | self._carried_with(self.held), note="released")
            edits.append(f"released {self.held} (now on the floor)")
            self.held = None
            return ok()

        # --- OPEN / CLOSE ----------------------------------------------------------
        if action in ("OPEN", "CLOSE"):
            # 1. beside it, 2. it has a door
            problem = self._require_here(name)
            if problem:
                return fail(problem, ("not_near", name))
            # The other half of the same fact. If the machine is entitled to excuse a
            # `PLACE_INSIDE(bowl)` from needing a door opened, it is entitled to say that
            # opening the bowl is not a thing that can happen - and a model that knows one
            # and not the other is not a model, it is a preference.
            if not self._openable(name):
                return fail(f"'{name}' is a {self._category(name)} and does not open",
                            ("no_door", name))
            want = action == "OPEN"
            if self.open.get(name) == want:
                warnings.append(f"'{name}' is already {'open' if want else 'closed'}")
            self.open[name] = want
            if want:
                self.opened.add(name)
            edits.append(f"{name}.open = {want}")
            return ok()

        # --- TOGGLE_ON / TOGGLE_OFF ------------------------------------------------
        if action in ("TOGGLE_ON", "TOGGLE_OFF"):
            # 1. beside it, 2. it has a switch
            problem = self._require_here(name)
            if problem:
                return fail(problem, ("not_near", name))
            if not self._switchable(name):
                return fail(f"'{name}' is a {self._category(name)} and has no switch",
                            ("no_switch", name))
            want = action == "TOGGLE_ON"
            if self.toggled.get(name) == want:
                warnings.append(f"'{name}' is already toggled {'on' if want else 'off'}")
            self.toggled[name] = want
            if want:
                self.switched_on.add(name)
                category = self._category(name)
                for state, appliances in CONFERS.items():
                    if category in appliances:
                        for item in self._inside_only(name):
                            self.states[state].add(item)
                            edits.append(f"{item}.{state} = True")
            edits.append(f"{name}.toggled = {want}")
            return ok()

        return fail(f"unknown action {action!r}", ("unknown_action", action))

    # ------------------------------------------------------------------ whole plans

    def _resolve_goal_name(self, name):
        """The graph node a goal term names, or the term itself.

        One vocabulary: the goal is written in the same names the belief graph is built
        from, so this is equality. It used to match loosely, with the plan breaking ties,
        because the goal model wrote the sentence's words - `cabinet` for `bottom_cabinet`
        - on 16 of 100 tasks. That gap is closed at the source now: the instruction names
        the dataset's category and the goal is canonicalised on the way in. A term that
        still does not resolve is a term nothing produced, and leaving it unresolved makes
        the condition unmet, which is the honest answer.
        """
        return name if name in self.graph.objects else name

    def unmet(self, goal):
        """Which of these goal conditions do not hold in this machine's world?

        Separate from `run` so the *simulator* can ask it of the true world after driving a
        plan. Replaying symbolically and executing with a camera have to be judged by one
        definition of done, or the difference between them measures the definition rather
        than the perception.
        """
        missing = []
        for edge_type, src, dst in goal:
            a = self._resolve_goal_name(src)
            # A door nobody touched is shut, and a switch nobody touched is off. Reading an
            # untracked object as `None` made "leave the oven shut" *unmet* for a plan that
            # never opened the oven - so a goal asserting the safe state of something the
            # plan does not disturb could never be satisfied, and five attempts would be
            # spent failing to satisfy it.
            if edge_type == "open":
                held = self.open.get(a, False) == dst
            elif edge_type in self.states:
                held = (a in self.states[edge_type]) == bool(dst)
            elif edge_type == "toggled":
                held = self.toggled.get(a, False) == dst
            else:
                held = self.graph.has_edge(edge_type, a, self._resolve_goal_name(dst))
            if not held:
                missing.append((edge_type, src, dst))
        return missing

    def run(self, plan, goal=()):
        """Apply every action in order, then check the goal.

        `plan` is a list of (action, argument-or-None). A goal entry is a triple. Usually
        it is an edge - `("on_top", "potato", "table")` - that must be present at the end.

        The two node properties can be asked for as well: `("open", "oven", False)` and
        `("toggled", "oven", True)`. Without them half the tasks a kitchen suggests are
        inexpressible - "turn the oven on", "shut the fridge" - because what they change is
        a property of one node rather than a relation between two, and a goal language of
        edges alone cannot say it.
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

        missing = self.unmet(goal) if failed_at is None else list(goal)

        # Everything the plan opened has to be shut - an open door is an open door.
        # Switches are not symmetric: only an appliance that heats or runs a cycle is worth
        # walking back for, so `MUST_SWITCH_OFF` filters these. That is what makes "turn on
        # the lamp" a task the machine can accept instead of one it fails for succeeding.
        left_open = sorted(n for n in self.opened if self.open.get(n))
        left_on = sorted(n for n in self.switched_on
                         if self.toggled.get(n) and self._category(n) in MUST_SWITCH_OFF)
        return Outcome(steps, self.graph, list(goal), failed_at is None and not missing,
                       missing, failed_at, left_open, left_on)


def check(graph, plan, goal=(), verbose=False):
    """Convenience: build a machine, run the plan, hand back the outcome."""
    return GraphMachine(graph, verbose=verbose).run(plan, goal)
