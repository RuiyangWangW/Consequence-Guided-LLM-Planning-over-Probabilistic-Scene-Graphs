#!/usr/bin/env python3
"""Plan, check, and hand the checker's complaint back to the LLM until the plan holds.

`planner.generate` is deliberately single-shot: it asks once and surfaces the raw failure.
This closes the loop around it. One scene graph serves both halves - the RSN's guesses are
what the LLM is shown *and* what the plan is checked against - so the checker can only
complain about things the planner was told, which is what makes its complaints repairable.

    task + scene
      -> objects the task needs                      task_objects.extract
      -> the RSN's guess at where they are           scene_graph.populate
      -> one WorldGraph, for the prompt and check    WorldGraph.from_scene_graph
      loop, up to `--attempts` times:
          LLM writes a plan                          planner.build_prompt / parse_plan
          the graph machine runs it                  GraphMachine.run
          if it holds, stop; else quote the step it refused and why, and ask again

The feedback is the machine's own reason string, against the numbered plan, with the
offending step marked. That is the whole point of the preconditions being specified: a
refusal is a sentence the model can act on - "robot is standing at countertop, not at
'oven'; NAVIGATE_TO it first" - and not merely a rejection.

    python replan.py --scene Beechwood_0_int --task "heat the potato in the oven"
    python replan.py --scene Rs_int --task "..." --objects potato oven --attempts 3
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


import argparse
import json

from graph_machine import GraphMachine
from planner import (PRIMITIVES, build_prompt, get_generator, parse_goal, parse_plan)
from world_graph import WorldGraph

DEFAULT_ATTEMPTS = 5


# One refusal is not about the plan's *order* but about the argument: a closet is a room
# however carefully you drive to it. For it the primitive's own contract is no help and is
# worse than none - NAVIGATE_TO's reads "requires: nothing", directly beneath a rejected
# NAVIGATE_TO. So it gets its own note and the generic contract is suppressed.
FAULT_NOTES = {
    "no_door": ("{name} has no door or lid, so it can never be opened or closed. Not "
                "every container opens: an open-topped bin, a bowl, a sink, a basket all "
                "take PLACE_INSIDE directly, with no OPEN before and no CLOSE after. "
                "Remove this step and the one that matches it."),
    "no_switch": ("{name} has no switch and can never be toggled on or off. Remove this "
                  "step and the one that matches it."),
    "not_graspable": ("{name} is fixed to the building and can never be picked up, no "
                      "matter what the plan does first. Only movable objects are grasped. "
                      "If {name} is where something is meant to go, drive to it and use "
                      "PLACE_ON_TOP({name}) or PLACE_INSIDE({name}) - the destination is "
                      "never grasped, and what gets put down is whatever the robot is "
                      "already holding."),
    "room": ("{name} is a room, not an object. NAVIGATE_TO drives to the thing the robot "
             "is about to act on - name that thing instead. If the step after this one "
             "already drives to it, this step is simply unnecessary: drop it."),
}



def compose_complaint(errand_goal, foreign, missing):
    """What to tell the model about an errand whose plan undid another errand.

    Each errand is planned and checked on its own, against the world as it stands when that
    errand begins. Nothing in that check can see the errand *after* it, so a plan that picks
    up somebody else's object and puts it somewhere of its own passes every test and still
    ruins the instruction. `gavel.compose` is the only thing that looks at the concatenation,
    and until now its verdict was recorded and thrown away.

    The complaint names the objects this errand touched that were none of its business, and
    the conditions that stopped holding as a result, and says what the errand is actually
    for. It deliberately does not quote the *other* errand's plan: the model is being asked
    to write this errand narrowly, not to reason about an ordering it was never shown.
    """
    wanted = ", ".join(f"{k}({s}, {d})" for k, s, d in errand_goal) or "the errand above"
    undone = ", ".join(f"{k}({s}, {d})" for k, s, d in missing)
    touched = ", ".join(sorted(foreign))
    return (f"""
Your plan for this errand moved objects it was not asked about: {touched}.

Another part of the instruction depends on those staying where they were, and running your
plan leaves these conditions no longer true: {undone}.

This errand is only responsible for: {wanted}. Write it again, touching nothing else - do not
pick up, open, or switch anything that is not needed for this errand alone.""")


def syntax_prompt(task, graph, reply, goal=(), declare_goal=False):
    """Ask again after a reply that parsed to no actions at all, saying so.

    The old behaviour here was to rebuild the *same* prompt and ask again. Under greedy
    decoding that is not a retry - the model is handed identical input and returns an
    identical reply, so five attempts are one attempt counted five times, and an errand
    whose only fault was its punctuation was recorded as five planning failures.

    So the complaint has to say what went wrong, and it is a syntax complaint rather than a
    planning one: the model is not being told its plan was wrong, because nothing here knows
    that - only that nothing in the reply could be read as an action. It gets the required
    form, the primitives it may use, and its own reply back to compare against.
    """
    from planner import PRIMITIVES, build_prompt

    forms = "\n".join(f"  {name}({'object' if s['takes_object'] else ''})"
                       for name, s in PRIMITIVES.items())
    quoted = "\n".join(f"  | {line}" for line in (reply or "").strip().splitlines()[:12])
    return (build_prompt(task, graph, with_goal=declare_goal, goal=goal)
            + f"""

Your previous reply could not be read. No line in it was a recognisable action:

{quoted or "  | (empty reply)"}

Write one action per line and nothing else, using exactly this form - the primitive's name,
then the object in round brackets:

{forms}

For example: NAVIGATE_TO(breakfast_table) then GRASP(mug) then PLACE_ON_TOP(countertop).
Do not use any other punctuation between the name and the object, and do not add prose.
""")


def repair_prompt(task, graph, steps, outcome, mended=()):
    """The planning prompt again, plus the plan that failed and the step that failed it.

    The whole plan is quoted back with the refused step marked, rather than only the
    error. A model handed "step 3 was wrong" has to reconstruct what step 3 was; a model
    handed its own plan with an arrow on the line does not, and the corrected plan comes
    back whole instead of as a fragment to splice in.
    """
    lines = []
    # Only when the plan ran to the end. `left_open` is read off the execution, and an
    # execution that stopped at step 10 never reached the CLOSE at step 12 - marking that
    # OPEN "never undone" reports a fault the plan does not have. Eleven of the 4B's 38
    # rejected plans carried at least one such false mark, on top of the real refusal.
    abandoned = (set(outcome.left_open) | set(outcome.left_on)
                 if outcome.failed_at is None else set())
    # A plan whose every action applied and whose goal is still unmet is nearly always the
    # wrong placement verb: PLACE_ON_TOP where the goal asks for `object_inside`, or the
    # mirror. Marking those steps "ok" is true - they were applicable - and useless: the
    # model is then shown twelve correct-looking steps and told a relation is missing, with
    # nothing connecting the two. Measured over both models, nine plans failed this way and
    # seven of them burned all five attempts re-sending the same placement. So find them.
    PLACES = {"on_top": "PLACE_ON_TOP", "object_inside": "PLACE_INSIDE"}
    misplaced = {}
    if outcome.failed_at is None:
        for relation, _moved, destination in outcome.missing:
            want = PLACES.get(relation)
            if want is None:
                continue
            for index, step in enumerate(steps):
                if (step["action"] in PLACES.values() and step["action"] != want
                        and step.get("object") == destination):
                    misplaced[index] = (step["action"], want)
    for index, step in enumerate(steps):
        arg = step.get("object") or ""
        mark = ""
        if index == outcome.failed_at:
            mark = f"   <-- REJECTED: {outcome.steps[index].reason}"
        elif arg in abandoned and step["action"] in ("OPEN", "TOGGLE_ON"):
            mark = "   <-- never undone"
        elif index in misplaced:
            wrong, want = misplaced[index]
            mark = f"   <-- applied, but {wrong} is not {want} - see below"
        elif index < len(outcome.steps) and outcome.steps[index].ok:
            mark = "   ok"
        lines.append(f"  {index + 1:2d}. {step['action']}({arg}){mark}")

    if outcome.failed_at is None:
        # A runnable plan can be wrong in two ways at once, and they used to be an if/elif:
        # a plan that left the oven on *and* missed the goal was told only about the oven,
        # fixed that, and learned about the goal on the next attempt. Two of five attempts
        # spent on what is one message. Both faults are known at the same moment, so both
        # are said at the same moment.
        faults = []
        if not outcome.safe:
            left = ([f"    still open:        {n}" for n in outcome.left_open]
                    + [f"    still switched on: {n}" for n in outcome.left_on])
            faults.append("it leaves the house in a state it should not:\n"
                          + "\n".join(left)
                          + "\n  Anything the robot opens it must close again, and "
                            "anything it switches on it must switch off - at the right "
                            "points, since a door has to stay open while something is "
                            "being put in or taken out.")
        if outcome.missing:
            detail = ("it does not achieve the task. These are still missing at the "
                      "end:\n"
                      + "\n".join(f"    {t}({a}, {b})" for t, a, b in outcome.missing))
            if misplaced:
                # Quote both contracts, for the same reason the rejected-step branch does:
                # knowing *that* the goal is unmet is not knowing what to write instead.
                verbs = sorted({v for pair in misplaced.values() for v in pair})
                detail += ("\n  The marked steps are why. They applied, but they produce "
                           "the wrong relation:\n"
                           + "\n".join(f"    {v}: {PRIMITIVES.get(v, {}).get('effect', '')}"
                                       for v in verbs)
                           + "\n  Change the marked steps to the placement whose effect "
                             "matches what is missing. Leave the rest of the plan as it is.")
            faults.append(detail)
        complaint = ("Every action was applicable, but the plan has "
                     + ("two problems" if len(faults) > 1 else "a problem") + ".\n\n"
                     + "\n\n".join(f"  {i}. {f}" for i, f in enumerate(faults, 1))
                     + "\n\nFix "
                     + ("both" if len(faults) > 1 else "it") + " in one plan.")
    else:
        # Quote the failed action's own contract back. Knowing *that* a step is wrong is
        # not the same as knowing what to write instead: told "cannot place 'potato' on
        # itself", a 7B model reproduced the identical plan five times, because the
        # complaint never says that PLACE_INSIDE takes the container and the held object
        # is implicit.
        action = steps[outcome.failed_at]["action"]
        spec = PRIMITIVES.get(action, {})
        kind, subject = (outcome.steps[outcome.failed_at].fault or (None, None))
        if kind in FAULT_NOTES:
            contract = "\n\n" + FAULT_NOTES[kind].format(name=f"'{subject}'")
        else:
            signature = f"{action}({'object' if spec.get('takes_object') else ''})"
            contract = (f"\n\nRemember what {signature} needs:\n"
                        f"  requires: {spec.get('requires', '')}\n"
                        f"  then:     {spec.get('effect', '')}")
            if action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
                contract += ("\nThe argument is the *destination* - the surface or "
                             "container being placed onto or into. What is being put down "
                             "is whatever the robot is already holding, and is never "
                             "named.")
        # Not "everything before it is fine". The machine checked applicability, and a
        # step can be applicable and still be the wrong thing to do - one rejected plan
        # was told its first eleven steps were fine when two of them put the milk back in
        # the fridge the task had asked to take it out of.
        complaint = ("The plan was rejected at the marked step. Every step before it was "
                     "applicable, which means only that the robot could carry it out - if "
                     "the plan up to there does not actually do what the task asked, fix "
                     "that too. Then fix the marked step and anything after it that "
                     "depended on it." + contract)

    # The plan quoted back is the mended one, so say so. A model shown steps it did not
    # write, with no explanation, has to work out whether it misremembers its own answer -
    # and the inserted drives are exactly the steps it keeps forgetting, so it is worth its
    # seeing that they were needed.
    preface = "Your previous attempt:"
    if mended:
        preface = ("Your previous attempt, with the missing steps already filled in for "
                   "you:\n" + "\n".join(f"  - {note}" for note in mended)
                   + "\n\nDo not undo those. What is left is the part they cannot fix:")

    return (f"{build_prompt(task, graph, goal=outcome.goal)}\n\n"
            f"---\n\n"
            f"{preface}\n\n" + "\n".join(lines) + "\n\n"
            f"{complaint}\n\n"
            f"Reply with ONLY the corrected action sequence, one action per line.")


def run(task, graph, goal=(), attempts=DEFAULT_ATTEMPTS, model_name=None,
        max_new_tokens=512, verbose=True, declare_goal=False, mend="loop",
        complaint=None):
    """Ask, mend, complain, ask again. The whole transcript.

    One attempt is: the model writes a plan, and the machine then validates and mends it
    over and over until it has nothing left to do - either the plan holds, or what is left
    is a fault the machine cannot derive an edit for. Only then is the model asked again,
    and it is asked about what survived the mending rather than about what it wrote.

    That ordering is the point. Mending only after the last attempt - which is what this
    did first - spends every retry on faults the machine could have removed itself, so the
    model is asked five times to insert a NAVIGATE_TO and never once about the thing that
    actually defeats it. Mending inside the loop means each retry is spent on a question
    only the model can answer.

    `goal` is optional. Without it the machine only asks whether every action was
    applicable, which is the question the pipeline can pose on its own - nothing upstream
    produces goal edges from a task description, and inventing them here would be checking
    the plan against a target nobody stated.

    `mend` says where the machine may edit: "loop" (the default, above), "end" to mend
    only the plan the attempts settled on, or False not at all. "end" is what this used to
    do, kept so the two placements can be measured against each other.

    Mending at all is on by default because measuring said so. Over the hundred-task benchmark the
    machine's own repairs are worth +22 tasks to the 4B and +17 to the 8B on top of five
    attempts of complaining, and break none; and a single plan mended once beats five
    attempts unmended for both models - 67 against 56, and 74 against 71 - at a third of
    the wall clock. Asking the model again is the expensive way to fix a missing
    NAVIGATE_TO.
    """
    from repair import repair

    generator = get_generator(model_name) if model_name else get_generator()
    seed = WorldGraph.from_scene_graph(graph)
    history = []
    prompt = build_prompt(task, graph, with_goal=declare_goal, goal=goal)
    if complaint:
        # An objection raised before the first attempt rather than after it. The composition
        # check is the only caller: it has already seen this errand's plan break another
        # errand's work, which is not a fault any single-errand check can see, so the model is
        # told about it up front instead of being asked the same question that produced it.
        prompt += "\n" + complaint

    for attempt in range(1, attempts + 1):
        reply = generator(prompt, max_new_tokens)
        steps = parse_plan(reply)
        if declare_goal:
            # The model's own reading of what "done" means. It is checked against the
            # plan, never against ground truth, so a wrong goal is the model's mistake to
            # make - and a plan that satisfies its own stated goal is at least internally
            # consistent, which is more than the loop could ask before.
            stated = parse_goal(reply)
            if stated:
                goal = stated

        if not steps:
            # Say what was wrong instead of asking the identical question again. See
            # `syntax_prompt`: re-issuing the same prompt to a greedy decoder is not a retry.
            history.append({"attempt": attempt, "steps": [], "outcome": None,
                            "unparsed": (reply or "")[:400]})
            prompt = syntax_prompt(task, graph, reply, goal=goal, declare_goal=declare_goal)
            continue

        plan = [(s["action"], s.get("object")) for s in steps]
        written = list(steps)
        # `repair` is itself the iteration: it validates, edits, re-validates, and stops
        # when the plan holds or the fault that remains is one it has no edit for. It
        # hands back the best plan it saw, which is the original if nothing helped.
        notes = []
        if mend == "loop" or mend is True:
            plan, notes = repair(seed, plan, goal)
            if notes:
                steps = [{"action": a, "object": o} for a, o in plan]
        outcome = GraphMachine(seed.copy()).run(plan, goal)
        # `written` is what the model actually returned, before any mending. The unchecked
        # arm is read off attempt 1, and mending overwrote `steps` in place - so the
        # "model's first answer, kept whatever it says" was in fact a repaired plan on 60 of
        # the 4B's 75 successes, and the checker's measured contribution was that much too
        # small.
        record = {"attempt": attempt, "steps": steps, "written": written,
                  "outcome": outcome}
        if notes:
            record["mended"] = notes
        history.append(record)

        if verbose:
            print(f"\nattempt {attempt}: {len(steps)} actions"
                  + (f", mended: {'; '.join(notes)}" if notes else ""))
            for index, step in enumerate(steps):
                arg = step.get("object") or ""
                flag = ""
                if index == outcome.failed_at:
                    flag = f"   REJECTED: {outcome.steps[index].reason}"
                print(f"  {index + 1:2d}. {step['action']}({arg}){flag}")

        # A plan is only accepted if it applies, tidies up after itself, and - where a
        # goal was given - reaches it. Leaving the oven on used to count as success,
        # because the loop had nothing to check but preconditions.
        if outcome.failed_at is None and outcome.safe and (not goal or outcome.goal_met):
            record["accepted"] = True
            return history
        # Complain about the mended plan, not the written one. The steps the machine
        # inserted are part of what the model is being asked to fix now, and quoting the
        # plan without them would mark a step number that is no longer there.
        prompt = repair_prompt(task, graph, steps, outcome, mended=notes)

    # The old placement, kept so the two can be measured against each other: the attempts
    # are spent, and only now does the machine get to touch the plan they settled on. Every
    # retry above was therefore spent on faults it could have removed itself.
    if mend == "end" and history and history[-1]["steps"]:
        steps = history[-1]["steps"]
        fixed, notes = repair(seed, [(s["action"], s.get("object")) for s in steps], goal)
        if notes:
            outcome = GraphMachine(seed.copy()).run(fixed, goal)
            record = {"attempt": len(history) + 1, "mended": notes, "outcome": outcome,
                      "steps": [{"action": a, "object": o} for a, o in fixed]}
            if outcome.failed_at is None and outcome.safe and (not goal or outcome.goal_met):
                record["accepted"] = True
            history.append(record)
            if verbose:
                print(f"\nmended by the graph machine: {'; '.join(notes)}")

    return history


def main():
    from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, format_for_llm, populate
    from task_objects import extract

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--objects", nargs="+", help="skip extraction, use these")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rsn-model", default=DEFAULT_MODEL)
    parser.add_argument("--json", help="write the whole transcript here")
    args = parser.parse_args()

    stated = {}
    if args.objects:
        objects, dependent = list(args.objects), []
    else:
        # Reuse the planner's generator so extraction costs no second model load, as
        # `pipeline.extract_objects` does. `uncertain` is what the RSN has to place;
        # `dependent` is what the task already said the location of.
        found = extract(args.task, args.model, generator=get_generator(args.model))
        objects, dependent = found["uncertain"], found["dependent"]
        stated = found["stated"]
    if dependent:
        print("dependent: " + ", ".join(
            f"{d['object']} {d['relation']} {d['target']}" for d in dependent))

    graph = populate(args.scene, objects, dependent, stated=stated,
                     model_path=args.rsn_model, threshold=DEFAULT_THRESHOLD)
    print(f"task:  {args.task}")
    print(f"scene: {args.scene}\n")
    print(format_for_llm(graph))

    history = run(args.task, graph, attempts=args.attempts, model_name=args.model)
    winner = next((h for h in history if h.get("accepted")), None)

    print()
    if winner:
        print(f"ACCEPTED on attempt {winner['attempt']} of {args.attempts}: "
              f"{len(winner['steps'])} actions, every precondition holds")
    else:
        last = history[-1]
        print(f"FAILED after {len(history)} attempts; the last refusal was "
              f"{last['outcome'].steps[last['outcome'].failed_at].reason}"
              if last["outcome"].failed_at is not None
              else f"FAILED after {len(history)} attempts")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({"task": args.task, "scene": args.scene, "graph": graph,
                       "attempts": [{"attempt": h["attempt"], "steps": h["steps"],
                                     "accepted": h.get("accepted", False),
                                     "failed_at": h["outcome"].failed_at,
                                     "reason": (h["outcome"].steps[h["outcome"].failed_at].reason
                                                if h["outcome"].failed_at is not None else None)}
                                    for h in history],
                       "plan": {"steps": (winner or history[-1])["steps"],
                                "errors": [] if winner else ["no valid plan"],
                                "warnings": []}}, f, indent=1)
        print(f"wrote {args.json}")
    return 0 if winner else 1


if __name__ == "__main__":
    raise SystemExit(main())
