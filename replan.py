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

import argparse
import json

from graph_machine import GraphMachine
from planner import PRIMITIVES, build_prompt, get_generator, parse_plan
from world_graph import WorldGraph

DEFAULT_ATTEMPTS = 5

# The first ask is greedy - one question, one answer. Retries sample, because greedy
# decoding makes a repair loop pointless: measured, a rejected plan came back
# byte-identical on all five attempts, the model having already given its best answer to a
# prompt it was not persuaded by. Rising temperature widens the search as the obvious
# repairs run out.
TEMPERATURES = (0.0, 0.5, 0.7, 0.9, 1.0)


def repair_prompt(task, graph, steps, outcome):
    """The planning prompt again, plus the plan that failed and the step that failed it.

    The whole plan is quoted back with the refused step marked, rather than only the
    error. A model handed "step 3 was wrong" has to reconstruct what step 3 was; a model
    handed its own plan with an arrow on the line does not, and the corrected plan comes
    back whole instead of as a fragment to splice in.
    """
    lines = []
    abandoned = set(outcome.left_open) | set(outcome.left_on)
    for index, step in enumerate(steps):
        arg = step.get("object") or ""
        mark = ""
        if index == outcome.failed_at:
            mark = f"   <-- REJECTED: {outcome.steps[index].reason}"
        elif arg in abandoned and step["action"] in ("OPEN", "TOGGLE_ON"):
            mark = "   <-- never undone"
        elif index < len(outcome.steps) and outcome.steps[index].ok:
            mark = "   ok"
        lines.append(f"  {index + 1:2d}. {step['action']}({arg}){mark}")

    if outcome.failed_at is None and not outcome.safe:
        # The one thing the loop can fault a *runnable* plan for without being told what
        # the task wants. The machine already knows what this plan opened and did not
        # shut, and what it switched on and did not switch off - no goal required, because
        # "put back what you disturbed" is not a property of the task, it is a property of
        # every task.
        left = ([f"    still open:        {n}" for n in outcome.left_open]
                + [f"    still switched on: {n}" for n in outcome.left_on])
        complaint = ("Every action was applicable, but the plan leaves the house in a "
                     "state it should not:\n" + "\n".join(left)
                     + "\n\nAnything the robot opens it must close again, and anything "
                       "it switches on it must switch off. Add the missing CLOSE and "
                       "TOGGLE_OFF actions, at the right points - a door has to stay open "
                       "while something is being put in or taken out.")
    elif outcome.failed_at is None:
        complaint = ("Every action was applicable, but the plan does not achieve the "
                     "task. These are still missing at the end:\n"
                     + "\n".join(f"    {t}({a}, {b})" for t, a, b in outcome.missing))
    else:
        # Quote the failed action's own contract back. Knowing *that* a step is wrong is
        # not the same as knowing what to write instead: told "cannot place 'potato' on
        # itself", a 7B model reproduced the identical plan five times, because the
        # complaint never says that PLACE_INSIDE takes the container and the held object
        # is implicit.
        action = steps[outcome.failed_at]["action"]
        spec = PRIMITIVES.get(action, {})
        signature = f"{action}({'object' if spec.get('takes_object') else ''})"
        contract = (f"\n\nRemember what {signature} needs:\n"
                    f"  requires: {spec.get('requires', '')}\n"
                    f"  then:     {spec.get('effect', '')}")
        if action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
            contract += ("\nThe argument is the *destination* - the surface or container "
                         "being placed onto or into. What is being put down is whatever "
                         "the robot is already holding, and is never named.")
        complaint = ("The plan was rejected at the marked step. Everything before it is "
                     "fine; fix that step and anything after it that depended on it."
                     + contract)

    return (f"{build_prompt(task, graph)}\n\n"
            f"---\n\n"
            f"Your previous attempt:\n\n" + "\n".join(lines) + "\n\n"
            f"{complaint}\n\n"
            f"Reply with ONLY the corrected action sequence, one action per line.")


def run(task, graph, goal=(), attempts=DEFAULT_ATTEMPTS, model_name=None,
        max_new_tokens=512, verbose=True):
    """Ask, check, complain, ask again. Returns the transcript of every attempt.

    `goal` is optional. Without it the machine only asks whether every action was
    applicable, which is the question the pipeline can pose on its own - nothing upstream
    produces goal edges from a task description, and inventing them here would be checking
    the plan against a target nobody stated.
    """
    generator = get_generator(model_name) if model_name else get_generator()
    seed = WorldGraph.from_scene_graph(graph)
    history = []
    prompt = build_prompt(task, graph)

    for attempt in range(1, attempts + 1):
        temperature = TEMPERATURES[min(attempt - 1, len(TEMPERATURES) - 1)]
        reply = generator(prompt, max_new_tokens, temperature)
        steps = parse_plan(reply)
        plan = [(s["action"], s.get("object")) for s in steps]
        outcome = GraphMachine(seed.copy(), allow_search=True).run(plan, goal)
        record = {"attempt": attempt, "steps": steps, "outcome": outcome}
        history.append(record)

        if verbose:
            print(f"\nattempt {attempt} (temperature {temperature}): "
                  f"{len(steps)} actions")
            for index, step in enumerate(steps):
                arg = step.get("object") or ""
                flag = ""
                if index == outcome.failed_at:
                    flag = f"   REJECTED: {outcome.steps[index].reason}"
                print(f"  {index + 1:2d}. {step['action']}({arg}){flag}")

        if not steps:
            prompt = build_prompt(task, graph)      # nothing parsed; ask again cleanly
            continue
        # A plan is only accepted if it applies, tidies up after itself, and - where a
        # goal was given - reaches it. Leaving the oven on used to count as success,
        # because the loop had nothing to check but preconditions.
        if outcome.failed_at is None and outcome.safe and (not goal or outcome.goal_met):
            record["accepted"] = True
            return history
        prompt = repair_prompt(task, graph, steps, outcome)

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
