"""Task shapes: the sequences a household task actually decomposes into.

Every shape returns the `spawn`, `goal` and `plan` of a task, and every one is at least
ten actions long. Writing them as shapes rather than as a hundred bespoke plans is what
keeps the length honest - the action count is a property of the shape, so it cannot drift
while someone edits a plan by hand - and it makes the dataset's difficulty describable:
`heat_and_serve` is thirteen actions in every scene it appears in.

The shapes are named for what a person would call the job, because that is what the task
text has to say. `fetch_heat_serve` is "take the soup out of the fridge, heat it in the
microwave and put it on the table", and the fifteen actions are what that costs.

**Every shape ends safe, and says so in its goal.** Anything a plan opens it shuts again,
anything it switches on it switches off, and the goal asserts both - so a plan that walks
away from a lit oven fails the goal rather than passing it. `build_tasks.py` checks the
invariant against `Outcome.left_open` / `left_on` rather than trusting the shapes to have
got it right.
"""

from planner import FILLABLE, OPENABLE

ON_TOP, INSIDE = "ON_TOP", "INSIDE"


def _nav_grasp(item):
    return [["NAVIGATE_TO", item], ["GRASP", item]]


def _open_grasp_close(container, item):
    """Take `item` out of a container that has a door, and shut it again."""
    return [["NAVIGATE_TO", container], ["OPEN", container],
            ["GRASP", item], ["CLOSE", container]]


def _open_put_close(container, inside=True):
    return [["NAVIGATE_TO", container], ["OPEN", container],
            ["PLACE_INSIDE" if inside else "PLACE_ON_TOP", container],
            ["CLOSE", container]]


# --- short shapes, for the multi-task set ------------------------------------------
#
# The same construction as every shape above - a spawn, a goal and a reference plan the
# builder replays before the task counts - only shorter. A long instruction is several of
# these, and the point of the multi-task set is the *order* they are done in, so each one
# has to be small enough that three or four fit in an instruction a person would say.
#
# Four primitives is the shortest errand with a goal worth verifying: fetch it, carry it,
# put it down. Two (drive somewhere and flip a switch) asserts nothing about placement.
#
# There is no shape here that must follow another. If one errand has to happen before a
# second - wash, then dry - they are one subtask, not two, so a multi-task instruction
# never carries precedence and every ordering of its subgoals is legal.


def carry_one(item, source, destination):
    """4 - fetch one thing from a surface and put it down on another."""
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}],
        "goal": [["on_top", item, destination]],
        "plan": _nav_grasp(item) + [["NAVIGATE_TO", destination],
                                    ["PLACE_ON_TOP", destination]],
    }


def stow_one(item, source, container):
    """4 - fetch one thing and put it inside something doorless: a bookcase, a bin, a sink.

    Doorless on purpose. A cabinet would need an OPEN and a CLOSE and the errand would be
    six primitives, which is long enough to crowd out a second subgoal.
    """
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}],
        "goal": [["object_inside", item, container]],
        "plan": _nav_grasp(item) + [["NAVIGATE_TO", container],
                                    ["PLACE_INSIDE", container]],
    }


def retrieve_one(item, container, destination):
    """4 - take one thing out of a doorless container and set it down."""
    return {
        "spawn": [{"name": item, "relation": INSIDE, "target": container}],
        "goal": [["on_top", item, destination]],
        "plan": _nav_grasp(item) + [["NAVIGATE_TO", destination],
                                    ["PLACE_ON_TOP", destination]],
    }


def switch_one(device):
    """2 - leave one switch on. The shortest errand with a goal worth checking.

    `toggled(device, true)` is an end state, and the safety rule leaves a lamp alone
    because `MUST_SWITCH_OFF` covers only heat sources, water sources and appliances that
    run a cycle - so "leave the lamp on" is expressible without contradicting it.
    """
    return {
        "spawn": [],
        "goal": [["toggled", device, True]],
        "plan": [["NAVIGATE_TO", device], ["TOGGLE_ON", device]],
    }


def stow_closed(item, source, cabinet):
    """6 - put one thing away behind a door, and shut it again.

    The doorful counterpart of `stow_one`. The CLOSE is not in the goal: `GraphMachine`
    derives "put back what you disturbed" from what the plan opened, which is strictly
    stronger than asserting it here.
    """
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}],
        "goal": [["object_inside", item, cabinet]],
        "plan": _nav_grasp(item) + _open_put_close(cabinet),
    }


def _conferred(appliance, *items):
    """What running `appliance` does to the things inside it, as goal conditions.

    Without these a plan that never switches the machine on satisfies "heat the pie and put
    it on the table" - the placement holds and the appliance, never touched, reads as off
    and shut. The safety conditions that used to stand here (`open(x, false)`,
    `toggled(x, false)`) said nothing: `GraphMachine` derives them from what the plan
    actually disturbed, which is strictly stronger, and asserting them in the goal only
    made them look enforced.
    """
    from planner import CONFERS

    state = next((k for k, v in CONFERS.items() if appliance in v), None)
    return [[state, item, True] for item in items] if state else []


def confer_in_place(item, source, appliance):
    """Fetch one thing, run it through an appliance, and leave it there. Eight actions.

    The short form of a conferred-state errand. `heat_and_serve` and `laundry_cycle` carry a
    relocation after the cycle - which is the interesting part of them, because it forces
    "cook, then place" - but that costs five to nine extra actions, and an instruction built
    from four of those is longer than anything a person would say in one breath. This asks for
    the state change alone: "put the plate in the dishwasher and run it", finishing with the
    thing still inside, the door shut and the switch off.

    The door is shut before the cycle and stays shut. `GraphMachine` does not require that -
    `TOGGLE_ON` confers on whatever is inside whether the door is open or not - but running an
    oven with its door open is not a plan anybody should be scored for writing, and shortening
    these by exploiting a gap in the world model would make the benchmark less faithful rather
    than more compact.
    """
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}],
        "goal": [["object_inside", item, appliance]] + _conferred(appliance, item),
        "plan": [["NAVIGATE_TO", item], ["GRASP", item],
                 ["NAVIGATE_TO", appliance], ["OPEN", appliance],
                 ["PLACE_INSIDE", appliance], ["CLOSE", appliance],
                 ["TOGGLE_ON", appliance], ["TOGGLE_OFF", appliance]],
    }


def heat_and_serve(item, source, appliance, destination):
    """13 - put it in, run it, take it out again, and set it down somewhere else."""
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}],
        "goal": [["on_top", item, destination]] + _conferred(appliance, item),
        "plan": _nav_grasp(item) + _open_put_close(appliance) + [
            ["TOGGLE_ON", appliance], ["TOGGLE_OFF", appliance],
            ["OPEN", appliance], ["GRASP", item], ["CLOSE", appliance],
            ["NAVIGATE_TO", destination], ["PLACE_ON_TOP", destination]],
    }


def fetch_heat_serve(item, container, appliance, destination):
    """15 - the same, but it starts inside something that has to be opened first."""
    return {
        "spawn": [{"name": item, "relation": INSIDE, "target": container}],
        "goal": [["on_top", item, destination]] + _conferred(appliance, item),
        "plan": [["NAVIGATE_TO", container], ["OPEN", container], ["GRASP", item],
                 ["CLOSE", container]] + _open_put_close(appliance) + [
            ["TOGGLE_ON", appliance], ["TOGGLE_OFF", appliance],
            ["OPEN", appliance], ["GRASP", item], ["CLOSE", appliance],
            ["NAVIGATE_TO", destination], ["PLACE_ON_TOP", destination]],
    }


def laundry_cycle(item, source, washer, dryer):
    """17 - wash it, take it out, dry it, and leave both machines off and shut."""
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}],
        "goal": ([["object_inside", item, dryer]]
                 + _conferred(washer, item) + _conferred(dryer, item)),
        "plan": _nav_grasp(item) + _open_put_close(washer) + [
            ["TOGGLE_ON", washer], ["TOGGLE_OFF", washer],
            ["OPEN", washer], ["GRASP", item], ["CLOSE", washer]]
            + _open_put_close(dryer)
            + [["TOGGLE_ON", dryer], ["TOGGLE_OFF", dryer]],
    }


def two_into_container(first, second, source, container):
    """12 - two things away into the same cupboard, shut behind each."""
    return {
        "spawn": [{"name": first, "relation": ON_TOP, "target": source},
                  {"name": second, "relation": ON_TOP, "target": source}],
        "goal": [["object_inside", first, container],
                 ["object_inside", second, container]],
        "plan": (_nav_grasp(first) + _open_put_close(container)
                 + _nav_grasp(second) + _open_put_close(container)),
    }


def load_and_run(first, second, source, appliance):
    """14 - load two things into an appliance, run it, and switch it off after."""
    return {
        "spawn": [{"name": first, "relation": ON_TOP, "target": source},
                  {"name": second, "relation": ON_TOP, "target": source}],
        "goal": ([["object_inside", first, appliance],
                  ["object_inside", second, appliance]]
                 + _conferred(appliance, first, second)),
        "plan": (_nav_grasp(first) + _open_put_close(appliance)
                 + _nav_grasp(second) + _open_put_close(appliance)
                 + [["TOGGLE_ON", appliance], ["TOGGLE_OFF", appliance]]),
    }


def move_three(items, sources, destination):
    """12 - three things carried to one place."""
    plan = []
    for item in items:
        plan += _nav_grasp(item) + [["NAVIGATE_TO", destination],
                                    ["PLACE_ON_TOP", destination]]
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}
                  for item, source in zip(items, sources)],
        "goal": [["on_top", item, destination] for item in items],
        "plan": plan,
    }


def unload_two(first, second, container, destination):
    """12 - take two things out of a cupboard, shutting it each time."""
    plan = []
    for item in (first, second):
        plan += [["NAVIGATE_TO", container], ["OPEN", container], ["GRASP", item],
                 ["CLOSE", container], ["NAVIGATE_TO", destination],
                 ["PLACE_ON_TOP", destination]]
    return {
        "spawn": [{"name": first, "relation": INSIDE, "target": container},
                  {"name": second, "relation": INSIDE, "target": container}],
        "goal": [["on_top", first, destination], ["on_top", second, destination]],
        "plan": plan,
    }


def carry_two(first, second, sources, destination):
    """8 - two things carried to one place, and nothing else.

    Split out from `carry_two_and_switch` for the two tasks whose trailing clause ran a
    coffee maker or a tap. Neither can be asked to end switched *on* - both are in
    `planner.MUST_SWITCH_OFF`, so the safety rule requires them off, and a goal wanting
    them on would contradict it. Running them leaves no other trace the goal can name, so
    the clause was dropped rather than made unscoreable.
    """
    plan = []
    for item in (first, second):
        plan += _nav_grasp(item) + [["NAVIGATE_TO", destination],
                                    ["PLACE_ON_TOP", destination]]
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}
                  for item, source in zip((first, second), sources)],
        "goal": [["on_top", first, destination], ["on_top", second, destination]],
        "plan": plan,
    }


def carry_two_and_switch(first, second, sources, destination, switch):
    """11 - two things moved, then something left switched on.

    It used to be "switched on and off again", whose goal was `toggled(switch, false)` - a
    condition a plan satisfies by never going near the switch, since an untouched switch
    reads as off. The clause was unscoreable, so the task was really a two-object carry
    with decoration. Asking for the switch to be left ON is an end state, and a lamp is not
    hazardous, so the safety rule leaves it alone.
    """
    plan = []
    for item in (first, second):
        plan += _nav_grasp(item) + [["NAVIGATE_TO", destination],
                                    ["PLACE_ON_TOP", destination]]
    plan += [["NAVIGATE_TO", switch], ["TOGGLE_ON", switch]]
    return {
        "spawn": [{"name": item, "relation": ON_TOP, "target": source}
                  for item, source in zip((first, second), sources)],
        "goal": [["on_top", first, destination], ["on_top", second, destination],
                 ["toggled", switch, True]],
        "plan": plan,
    }


def stack_then_store(rider, tray, source, tray_source, container):
    """11 - take one thing out of a cupboard, set it on another, and put the pair away.

    The only shape that leans on carrying: grasping the tray takes the rider with it, so
    the last `PLACE_INSIDE` puts both away and the goal asks for both. A plan that puts the
    rider away separately does not satisfy it.

    The rider starts *inside* `source` rather than on a worktop, which is what makes this
    eleven actions without padding. An earlier version ended on a `NAVIGATE_TO` back to
    where the rider came from, and that was a step the task text had no reason to mention -
    so the extraction ground truth named an object the instruction never did.
    """
    return {
        "spawn": [{"name": rider, "relation": INSIDE, "target": source},
                  {"name": tray, "relation": ON_TOP, "target": tray_source}],
        "goal": [["on_top", rider, tray], ["object_inside", tray, container]],
        "plan": ([["NAVIGATE_TO", source], ["OPEN", source], ["GRASP", rider],
                  ["CLOSE", source]]
                 + [["NAVIGATE_TO", tray], ["PLACE_ON_TOP", tray], ["GRASP", tray]]
                 + _open_put_close(container)),
    }


def swap_places(first, second, source_a, source_b, spare):
    """12 - each of two things ends up where the other started.

    The robot has one hand, so the swap needs somewhere to put the first thing down: the
    `spare` surface is not decoration, it is what makes the task twelve actions instead of
    eight and a contradiction.

    A source that has a door holds its object *inside* rather than on top, and the swap
    follows: the thing is taken out of it and the other thing goes in. Assuming ON_TOP
    everywhere made one task contradict its own sentence - "swap the detergent bottle **in**
    the utility room bottom cabinet" spawned the bottle on the cabinet's roof and demanded
    the soap end up there too. The plan the model wrote put the soap inside, which is what
    the words say and what a person would do, and it was marked wrong for it.
    """
    def relation(target):
        # Fillable, not openable: what decides "in" versus "on" is whether things go inside
        # it, not whether it has a door.
        return INSIDE if target in FILLABLE else ON_TOP

    # Two different questions. `fillable` says whether a thing goes *in* it - a bookcase
    # holds books inside it - and `openable` says whether a door has to be dealt with first.
    # A bookcase and a bin are the first without the second, so deciding both from one
    # annotation had the plan open a bookcase.
    def take(item, source):
        return (_open_grasp_close(source, item) if source in OPENABLE
                else _nav_grasp(item))

    def put(target):
        inside = target in FILLABLE
        if target in OPENABLE:
            return _open_put_close(target, inside=inside)
        return [["NAVIGATE_TO", target],
                ["PLACE_INSIDE" if inside else "PLACE_ON_TOP", target]]

    def predicate(item, target):
        return ["object_inside" if relation(target) == INSIDE else "on_top", item, target]

    return {
        "spawn": [{"name": first, "relation": relation(source_a), "target": source_a},
                  {"name": second, "relation": relation(source_b), "target": source_b}],
        "goal": [predicate(first, source_b), predicate(second, source_a)],
        "plan": (take(first, source_a) + put(spare)
                 + take(second, source_b) + put(source_a)
                 + take(first, spare) + put(source_b)),
    }
