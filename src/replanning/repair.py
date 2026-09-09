#!/usr/bin/env python3
"""Mechanically repair a plan the graph machine refused, where the edit is derivable.

The complaint loop asks the LLM to rewrite the whole plan. That works when the model can
see what it got wrong, and on the 4B it mostly cannot: 39 of its 47 failures are plans the
checker refused, 36 of those burn all five attempts, and the refused step sits on average
nine actions into a fourteen-action plan. Half of them are one omission - acting on
something the robot never drove to - and the correction is a single line inserted before
the failing step.

The machine already knows that line. `StepResult.fault` names the precondition that failed
and the object it failed on, so the edit follows from the refusal rather than from reading
the English. This module turns the ones that follow into plan edits and leaves the rest
alone.

    repaired, notes = repair(graph, plan, goal)

Two of the refusals are not derivable from the graph at all. `GRASP` while already holding
and `PLACE` with an empty hand are the same mistake seen from two sides - the model plans as
though the robot had two hands, so it grasps twice before placing, or places twice having
grasped once - and in both the machine knows the hand is wrong but not what the plan meant.
The *goal* knows: it is the one thing in the pipeline that states where each object is
going. So the missing grasp is whatever the goal wants at that destination, and the answer
to a full hand is to finish the errand the goal gives the held object rather than to drop
it. Measured over the surviving failures of both models, the goal named the missing object
in 28 of 32.

What stays with the LLM is what neither the graph nor the goal can supply: a hand-state
fault on an object the goal never mentions, and a `not_near` whose subject is a room rather
than a thing.

Every edit is re-validated by the same machine, and an edit that does not strictly improve
the plan is rolled back - so a repair can be speculative, but it cannot make things worse.
"""

REPAIRABLE = ("not_near", "closed", "not_open", "room", "no_door", "no_switch",
              "not_graspable", "empty_hand", "holding")

# Which goal predicate each placement writes, so a placement can be read backwards: a
# `PLACE_INSIDE(cabinet)` with an empty hand is asking for whatever the goal wants inside
# that cabinet.
PLACES = {"PLACE_ON_TOP": "on_top", "PLACE_INSIDE": "object_inside"}
PLACE_FOR = {v: k for k, v in PLACES.items()}

# Headroom for a plan that needs several inserts, and far below anything that could run
# away. A round that changes nothing ends the loop regardless, so this is a backstop rather
# than a budget.
MAX_ROUNDS = 12

PARTNER = {"OPEN": "CLOSE", "CLOSE": "OPEN",
           "TOGGLE_ON": "TOGGLE_OFF", "TOGGLE_OFF": "TOGGLE_ON"}


def _same(a, b):
    """One vocabulary: the goal and the graph use the same names, so this is equality."""
    return a is not None and b is not None and str(a) == str(b)


def _edit(graph, plan, index, kind, subject, goal=()):
    """The plan edit this refusal implies, or None if it implies none.

    `graph` is the world as it stands *at the refusal*, not the world the plan started in.
    The difference decides the hand-state rules: asked what is missing from a table, the
    seed graph says "everything", because nothing has been placed yet in it. Reading the
    seed had the repair re-grasp the object the plan had just put down.

    Returns a whole replacement plan. Guards live here rather than at the call site because
    each is specific to one rule, and a guard that fails means "no edit", not "edit anyway".
    """
    action, arg = plan[index]

    if kind == "not_near":
        # `WorldGraph.resolve` hands back room ids unchanged, so `subject` can be a room -
        # and `NAVIGATE_TO` refuses rooms. Synthesising one would produce a step the very
        # next round refuses, which the round after repairs by inserting it again.
        if subject in graph.rooms or subject not in graph.objects:
            return None
        return plan[:index] + [("NAVIGATE_TO", subject)] + plan[index:]

    if kind in ("closed", "not_open"):
        # `subject` is the container standing in the way, and the whole edit is the OPEN.
        # It does not need the drive to the container spliced in with it, nor the drive
        # back: inserting the bare OPEN makes the next round raise `not_near` on the
        # container, which rule 1 answers, and the round after that raises `not_near` on
        # whatever the failing step was reaching for. Two rules compose into what one
        # three-action splice used to do, and the splice had to guess where the robot had
        # been standing to write the drive back.
        if subject in graph.rooms or subject not in graph.objects:
            return None
        return plan[:index] + [("OPEN", subject)] + plan[index:]

    if kind == "not_graspable":
        # A fixture is not cargo, whatever precedes the step. Dropping it is the only edit
        # that does not invent intent - and it is nearly always the model running the
        # go-grasp-place template over the *destination*, which it had already driven to.
        return plan[:index] + plan[index + 1:]

    if kind == "room":
        # The step can never apply, whatever precedes it. Dropping it is the only edit that
        # does not invent intent: a room is not a destination for NAVIGATE_TO, and the step
        # that follows names the thing the plan actually meant to reach.
        return plan[:index] + plan[index + 1:]

    if kind in ("no_door", "no_switch"):
        # The object has no door, or no switch, so every step that assumes one is equally
        # doomed - and they come in pairs. Removing the OPEN alone leaves its CLOSE to be refused next
        # round, and a plan that opens and closes a lidless bin three times would need six
        # rounds; removing the pair at once needs one.
        partner = PARTNER.get(action)
        return [(a, o) for a, o in plan
                if not (o == arg and (a == action or a == partner))]

    # The two hand-state faults. Neither is derivable from the graph alone - the machine
    # knows the hand is wrong but not what the plan meant - and both are derivable from the
    # goal, which is the only thing in the pipeline that states intent. They are one mistake
    # wearing two faces: the model plans as though the robot had two hands, so it grasps
    # twice before placing, or places twice having grasped once. Together they are 22 of the
    # 8B's 23 surviving planning failures.
    if kind == "empty_hand" and action in PLACES:
        # "Place into the cabinet" with nothing held: the goal says what belongs in that
        # cabinet, so the missing steps are the drive and the grasp for whichever of those
        # things is not there yet.
        want = PLACES[action]
        for relation, moved, destination in goal:
            if relation != want or not _same(destination, arg):
                continue
            if moved in graph.objects and not graph.has_edge(want, moved, arg):
                return (plan[:index] + [("NAVIGATE_TO", moved), ("GRASP", moved)]
                        + plan[index:])
        return None

    if kind == "holding":
        # Grasping with a full hand. `subject` is what is already held, and the goal says
        # where it was going - so finish that errand first rather than dropping it, which
        # is what a bare RELEASE would do and what could undo a goal already met.
        for relation, moved, destination in goal:
            if relation in PLACE_FOR and _same(moved, subject) \
                    and destination in graph.objects:
                return (plan[:index] + [("NAVIGATE_TO", destination),
                                        (PLACE_FOR[relation], destination)]
                        + plan[index:])
        return None

    return None


def _discharge(plan, outcome):
    """Close what the plan left open and switch off what it left on, at the end.

    Only reached once every action applies. An inserted OPEN has to be undone or the plan
    fails the safety check, but *where* it is undone cannot be decided while the plan is
    still being made applicable - a door has to stay open while things go in and out, so
    the only point known to be after the last use is the end.

    Each undo is driven to. CLOSE and TOGGLE_OFF both require the robot to be standing at
    the object, and by the end of a plan it is standing wherever it put the last thing
    down, which is not where the door it left open is.
    """
    tail = []
    for name in outcome.left_open:
        tail += [("NAVIGATE_TO", name), ("CLOSE", name)]
    for name in outcome.left_on:
        tail += [("NAVIGATE_TO", name), ("TOGGLE_OFF", name)]
    return plan + tail


def _rank(outcome, length):
    """How good an outcome is, for deciding whether an edit was an improvement.

    Applicability first, and safety only after it. The other order looks reasonable and is
    wrong: a plan refused at step 2 never opened anything, so it is vacuously safe, and
    ranking safety above applicability makes that vacuum beat a repaired plan that really
    runs and leaves one door open - the repair is rejected and the door is never even
    reached. Goal, then how far it got, then length, which breaks ties towards the shorter
    plan so a repair cannot win by padding.
    """
    return (outcome.failed_at is None, outcome.goal_met, outcome.safe,
            outcome.failed_at if outcome.failed_at is not None else len(outcome.steps),
            -length)


def repair(graph, plan, goal=(), rounds=MAX_ROUNDS, machine=None):
    """Fix what the machine's refusals imply, and hand back the best plan found.

    `graph` is the seed WorldGraph; it is copied for every trial, never mutated. Returns
    `(plan, notes)` - the plan is the original if nothing could be improved, and `notes` is
    one line per accepted edit, for the record and for the tests.
    """
    from graph_machine import GraphMachine

    def run(candidate):
        return GraphMachine(graph.copy()).run(candidate, goal)

    start_out = run(plan)
    best, best_out = list(plan), start_out
    current, notes, seen = list(plan), [], {tuple(plan)}

    # Edits are applied and the plan re-run, but an edit is NOT required to improve the
    # outcome on its own. It cannot be: the whole point of the small rules is that they
    # compose, and inserting a bare OPEN moves the plan sideways - same failing index, one
    # action longer - until the next round supplies the drive that makes it count. Judging
    # each edit alone stalled exactly there and repaired nothing.
    #
    # So the loop runs freely and only the *best plan seen* is kept, compared at the end
    # against the plan it started from. The guarantee is unchanged - a repair can never
    # hand back something worse than it was given - but it is now a guarantee about the
    # result rather than about every step towards it.
    for _ in range(rounds):
        out = run(current)
        if _rank(out, len(current)) > _rank(best_out, len(best)):
            best, best_out = list(current), out
        if out.failed_at is None:
            break
        kind, subject = out.steps[out.failed_at].fault or (None, None)
        if kind not in REPAIRABLE:
            break
        candidate = _edit(out.graph, current, out.failed_at, kind, subject, goal)
        # A cycle is the one thing the loop cannot reason its way out of, so a plan it has
        # already tried ends it.
        if candidate is None or tuple(candidate) in seen:
            break
        seen.add(tuple(candidate))
        notes.append(f"{kind}({subject}) at step {out.failed_at + 1}: "
                     f"{len(current)} -> {len(candidate)} actions")
        current = candidate

    # Safety is checked only at the end of a plan that ran to the end, so this comes after
    # the applicability loop rather than inside it.
    if best_out.failed_at is None and not best_out.safe:
        candidate = _discharge(best, best_out)
        out = run(candidate)
        if _rank(out, len(candidate)) > _rank(best_out, len(best)):
            notes.append(f"discharged {len(candidate) - len(best)} left-open/left-on")
            best, best_out = candidate, out

    # Nothing found that beats what we were handed: give it back untouched, and say so by
    # returning no notes.
    if _rank(best_out, len(best)) <= _rank(start_out, len(plan)):
        return list(plan), []

    return best, notes
