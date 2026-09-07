#!/usr/bin/env python3
"""Does checking the plan help? Run the dataset with and without the graph machine.

Both arms come from **one** run per task, which is what makes the comparison exact rather
than approximate:

    without validation   the model's first answer, kept whatever it says. Attempt 1 is
                         decoding is deterministic, so it is the same plan in both arms -
                         the arms differ only in whether anything is done about a bad one.
    with validation      the repair loop of `replan.py`: check with `GraphMachine`, hand
                         the refusal back, ask again, up to `--attempts` times.

Scoring is against **ground truth**, not against the graph the planner was given. A plan
that is valid in the RSN's imagined house and impossible in the real one is a failure, and
scoring it against the RSN's guess would hide exactly the error this pipeline exists to
catch. `data/tasks.json` carries the true object placements and the goal.

**The ground truth is used to score, never to decide.** Whether a run succeeded is settled
entirely by replaying its plan against the true world: every action applies, the goal
holds, nothing was left open or switched on. The dataset's own reference plan and its
extraction answer take no part in that - they exist to prove the task is solvable and to
*explain* a failure after the fact. A run whose extraction differs from the reference and
whose plan works anyway is a success, because it is one.

Failures are then attributed to the first stage that went wrong, because they cascade - an
object extraction missed is an object the RSN cannot place, so the planner cannot use it
and the plan cannot possibly work:

    extraction      the objects pulled out of the task text do not cover what it needs,
                    or a stated location was read as the wrong relation
    grounding       extraction was right, but the RSN put an object in a room that has no
                    such thing, so the plan acts on something that is not there
    planning        the graph was right and the plan is still inapplicable
    goal            every action applies and the task is not done
    safety          the task is done and something was left open or switched on
    unparsed        nothing that looked like an action came back

    python evaluate.py                       # all 100
    python evaluate.py --limit 10 --json out.json
"""

import argparse
import json
import time

from build_tasks import furniture_rooms, seed_graph, world_for
from graph_machine import GraphMachine
from object_names import match, same
from planner import CONFERS, get_generator, release_generator
from replan import run as replan_run
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from task_objects import extract
from world_graph import WorldGraph

TASKS = "data/tasks.json"


def same_object(a, b):
    """Do these two names refer to the same thing?

    One implementation, in `object_names`, shared with the simulator. They used to be two:
    the scorer matched loosely and the simulator matched exactly, so a plan the scorer
    counted correct was one the robot could not locate - 17 search failures in a single run
    were that disagreement rather than the robot.
    """
    return same(a, b)


def resolve(name, candidates):
    """The name in `candidates` this one refers to, or None."""
    return match(name, candidates)


def extraction_error(task, found):
    """Did stage 1 recover what the task names, and where the task says it is?"""
    truth = task["extraction"]
    wanted = set(truth["uncertain"]) | {d["object"] for d in truth["dependent"]}
    got = set(found["uncertain"]) | {d["object"] for d in found["dependent"]}
    missed = sorted(w for w in wanted if not any(same_object(w, g) for g in got))
    if missed:
        return f"did not extract {', '.join(missed)}"

    stated = {d["object"]: (d["relation"], d["target"]) for d in truth["dependent"]}
    heard = {d["object"]: (d["relation"], d["target"]) for d in found["dependent"]}
    for name, (relation, target) in stated.items():
        match = next((h for h in heard if same_object(name, h)), None)
        if match is None:
            return f"did not read a stated location for '{name}'"
        got_relation, got_target = heard[match]
        if got_relation != relation or not same_object(got_target, target):
            return (f"read '{name}' as {got_relation} {got_target} "
                    f"rather than {relation} {target}")

    # And the other direction: a relation stage 1 *invented*. The loop above only walks the
    # relations the task really states, so an extra one was invisible - and the extra one is
    # the error that matters most, because it is what reading a destination as a location
    # looks like. "carry the mug from the kitchen to the coffee table" came back as
    # `mug ON_TOP coffee_table`, the belief was built with the mug already at its
    # destination, and the robot believed the task was done before it moved. That was
    # recorded as a planning failure.
    for name, (relation, target) in heard.items():
        if any(same_object(name, w) for w in stated):
            continue
        return (f"invented a location for '{name}': {relation} {target}, which the task "
                f"does not state")
    return None


def grounding_error(task, graph):
    """Did the RSN leave the planner unable to name something the task needs?

    Not "a different room from the reference": a scene with breakfast tables in the kitchen
    and the living room has two right answers, and marking one of them wrong invents a
    failure that never happened.

    Returns `(message, names)` - the names matter because a failure is only *caused* by
    grounding when the step that failed acted on an object the RSN could not place. Blaming
    every failure in a task where anything at all was unplaceable is the correlational
    mistake this file already corrected once for extraction: a run was recorded as a
    grounding failure over a `table_lamp` the plan never reached, when what it actually
    missed was `on_top(gaming_controller, bed)`.

    Nor is a wrong first guess one. The RSN returns a *ranking*, and the search layer works
    down it - a room searched and ruled out costs metres, not the task, and the plan never
    changes because it only ever said `NAVIGATE_TO(potato)`. So the only grounding failure
    that survives is the one nothing downstream can recover from: an object the RSN could
    not place at all, which is therefore absent from the graph the planner was shown and
    cannot be named in a plan.
    """
    world = world_for(task["scene"])
    unplaced = sorted(graph.get("unplaced") or {})
    if unplaced:
        return f"the RSN could not place {', '.join(unplaced)}", set(unplaced)

    rooms = furniture_rooms(task["scene"])
    hopeless = []
    for name, info in (graph.get("objects") or {}).items():
        if name not in rooms:
            continue
        candidates = info.get("candidates") or [info.get("room")]
        if not any(any(world.category_of(n) == name and world.room_of(n) == room
                       for n in world.truth.object_names())
                   for room in candidates):
            hopeless.append(name)
    if not hopeless:
        return None, set()
    return ("; ".join(f"{n}: no room the RSN ranked has one" for n in hopeless),
            set(hopeless))


def score(task, steps):
    """Replay a plan against ground truth. Returns (verdict, detail, failing object).

    The plan's object names are grounded onto the true world's first, the way stage 5 does
    before running anything in OmniGibson. Without that, a plan that says `GRASP(soup)`
    fails against a world holding `bottle_of_soup` - and the pipeline never saw the longer
    name, because its own extraction produced the shorter one from the instruction. That is
    a naming mismatch, not a planning error, and scoring it as one buries the result.
    """
    if not steps:
        return "unparsed", "no actions parsed from the reply", None
    graph = seed_graph(task)
    known = set(graph["objects"])
    # The task's own ground-truth vocabulary, resolved against first. The world holds every
    # category in the house on purpose - a plan that reaches the goal by way of a different
    # real cupboard should be marked right - but that also means an under-specified name
    # matches several of them equally. `cabinet` is as good a name for `top_cabinet` as for
    # `bottom_cabinet`, and matching against the whole house picked the top one: a cabinet
    # in the playroom, for a task about the bathroom, marking five correct plans wrong.
    #
    # The task says which it means. Its spawn, plan and goal name the categories it is
    # about, and they are ground truth rather than a guess - so an ambiguous name resolves
    # among those first, and only falls back to the whole house when the task's own words
    # do not settle it.
    # Only what the *setup* names - the objects the task injects and the rooms it pins.
    # The reference plan and the goal were in here too, and they are the answer: a plan
    # saying "towel" was resolved against the reference's `hand_towel` before the house was
    # consulted, which is help no unseen task can give. Removing them moves 3-10 tasks per
    # stored run, every one of them ok -> worse.
    named = ({s["target"] for s in task.get("spawn", ())}
             | {s["name"] for s in task.get("spawn", ())}
             | set(task.get("rooms") or {})) & known

    def ground(name):
        return resolve(name, named) or resolve(name, known) or name

    plan = [(s["action"], ground(s["object"]) if s.get("object") else None)
            for s in steps]
    machine = GraphMachine(WorldGraph.from_scene_graph(graph))
    outcome = machine.run(plan, [tuple(g) for g in task["goal"]])
    if outcome.failed_at is not None:
        step = outcome.steps[outcome.failed_at]
        return ("planning", f"step {outcome.failed_at + 1} {step.action}: {step.reason}",
                step.arg)
    if not outcome.goal_met:
        return ("goal",
                "missing " + ", ".join(f"{t}({a}, {b})" for t, a, b in outcome.missing),
                None)
    if not outcome.safe:
        return ("safety", "left " + ", ".join([f"{n} open" for n in outcome.left_open]
                                              + [f"{n} on" for n in outcome.left_on]), None)
    return "ok", f"{len(plan)} actions", None


def evaluate(tasks, attempts=5, model=None, verbose=True, out=None,
             extractor=None, simulate=False, declare_goal=False, goal_model=None,
             mend="loop"):
    """Run every task. Rows are written after each one, not at the end.

    A hundred tasks is an hour of GPU time, and a run that dies at task 90 with nothing on
    disk has cost an hour and produced nothing. Writing as it goes means a crash, an OOM or
    a stop leaves an analysable partial result.
    """
    generator = get_generator(model) if model else get_generator()
    # A fine-tuned extractor is a separate, much smaller model that does stage 1 only; the
    # planner stays whatever `model` names. It gets the short instruction it was trained
    # on rather than the twelve worked examples, which is the point of having tuned it.
    extract_prompt = None
    if extractor:
        from finetune_extraction import INSTRUCTION

        extractor_gen = get_generator(adapter=extractor)
        extract_prompt = INSTRUCTION + "\n"

    # The third small model. It reads the finished state out of the instruction so the
    # checker can ask whether a plan *does the task*, which it could not before: the
    # benchmark's goal is ground truth kept for scoring, and handing it to the planner
    # would be telling it the answer. Asking the planner for its own goal was tried and
    # made things worse - its goal was right half the time, and validating against a wrong
    # goal accepts plans that then fail the real one.
    if goal_model:
        from finetune_extraction import GOAL_INSTRUCTION
        from planner import parse_goal

        goal_gen = get_generator(adapter=goal_model)
    rows = []
    for index, task in enumerate(tasks, 1):
        started = time.time()

        # --- stage 1, shared by both arms so they differ only in the validation loop ---
        found = extract(task["task"],
                        generator=extractor_gen if extractor else generator,
                        prompt=extract_prompt)
        stage1 = extraction_error(task, found)

        graph = populate(task["scene"], found["uncertain"], found["dependent"],
                         stated=found["stated"], model_path=DEFAULT_MODEL,
                         threshold=DEFAULT_THRESHOLD)
        stage3, unnameable = grounding_error(task, graph)
        # Is the task already finished in the world the robot believes in? If so no plan can
        # be judged - the checker will accept one that does nothing, and the failure surfaces
        # only when driven. It always means an earlier stage put an object where the task
        # wanted it to end up, which is a grounding fault however the plan then behaves.
        # `build_tasks` refuses any task whose goal holds before the robot moves, but it
        # tests the *true* seed graph; nothing tested the belief.
        believed_done = not GraphMachine(
            WorldGraph.from_scene_graph(graph)).run([], [tuple(g) for g in task["goal"]]).missing

        # Pass the model through. Without it `replan.run` falls back to its own default,
        # so extraction runs on the model under test and *planning* runs on whatever that
        # default is - two models resident on one card, and a comparison that silently
        # measures the wrong one.
        # What the pipeline believes "done" means. Never `task["goal"]` - that is the
        # answer key, and the whole point is that a wrong prediction shows up as a failure
        # rather than being hidden by it.
        predicted = ()
        if goal_model:
            predicted = parse_goal("GOAL:\n" + goal_gen(
                GOAL_INSTRUCTION.format(task=task["task"]), 200))

        history = replan_run(task["task"], graph, goal=predicted, attempts=attempts,
                             model_name=model, verbose=False,
                             declare_goal=declare_goal, mend=mend)
        first = (history[0].get("written") or history[0]["steps"]) if history else []
        winner = next((h for h in history if h.get("accepted")), None)
        final = (winner or history[-1])["steps"] if history else []

        # `replan.run` mends the plan itself when its attempts run out, so the edits are
        # already in `final`; this only reads what it did, for the record.
        mended = (winner or history[-1]).get("mended") if history else None

        # The verdict is the plan's, and only the plan's.
        plain, plain_why, plain_at = score(task, first)
        checked, checked_why, checked_at = score(task, final)

        # And the same plan, *driven*. `score` replays symbolically: a NAVIGATE_TO always
        # succeeds, so a plan that depends on finding a mug in the wrong room scores the
        # same as one that does not. Running it in `sim2d` makes the robot search for the
        # object with a camera, and the belief it searches on is the one this pipeline
        # produced - so a wrong extraction costs metres here in a way it cannot symbolically.
        simulated = None
        if simulate and final:
            from sim_eval import run_plan

            try:
                simulated = run_plan(task, graph, final, verbose=False)
            except Exception as exc:
                simulated = {"ok": False, "why": f"{type(exc).__name__}: {exc}",
                             "error": True}

        # Attribution runs only over failures, and only to explain them. An earlier stage
        # going wrong is the honest cause - the planner cannot use an object nobody
        # extracted - but a run whose extraction differs from the reference and whose plan
        # works anyway is a success, not an extraction failure.
        def blame(verdict, steps, failed_on):
            """Which stage *caused* this failure - not merely which stage differed.

            The old rule said "extraction" whenever stage 1 disagreed with the reference at
            all. Measured, that was wrong nearly every time: of the 10 failures it blamed on
            extraction for the 8B, every single one actually died on a planner error the
            plan's own words show - navigating to a room, grasping while already holding,
            acting on something it had not driven to. Extraction had differed, so extraction
            got the name, and the biggest failure class was undercounted by a fifth.

            Extraction can only *cause* a failure by leaving the planner unable to name
            something. If the plan names an object anyway, extraction missing it changed
            nothing about what happened. So the question is not "did stage 1 differ" but
            "did the plan want something stage 1 never surfaced".
            """
            if verdict == "ok":
                return None
            # Only when the step that failed is the object nothing could name. Absent that,
            # an unplaceable object elsewhere in the task did not cause this failure.
            if stage3 and failed_on and any(same_object(failed_on, n) for n in unnameable):
                return "grounding"
            # A belief that already satisfies the goal is a stage-1/3 fault, and naming the
            # planner for it hides the cause entirely.
            if believed_done:
                return "extraction" if stage1 else "grounding"
            # A predicted goal is a new way to fail, and it has to be named as its own
            # stage. If the loop accepted a plan because it satisfied the goal stage 2b
            # predicted, and the plan then misses the real one, the planner did what it was
            # asked - the target was wrong. Blaming the planner for that hides the cause.
            if verdict == "goal" and predicted:
                # Compare only the predicates stage 2b is asked to produce. It omits
                # `open`/`toggled` on purpose - the machine derives those from what the plan
                # disturbed - so measuring it against the benchmark's full goal blamed it
                # for every goal failure, including seven where its prediction was exactly
                # right.
                spoken = {g[0] for g in predicted} | {"on_top", "object_inside"} | set(CONFERS)
                truth_goal = {(g[0], str(g[1]), str(g[2]).lower()) for g in task["goal"]
                              if g[0] in spoken}
                said = {(g[0], str(g[1]), str(g[2]).lower()) for g in predicted}
                if not all(any(t[0] == s_[0] and same_object(t[1], s_[1])
                               and t[2] == s_[2] for s_ in said) for t in truth_goal):
                    return "goal_model"
            named = {s.get("object") for s in steps if s.get("object")}
            truth = task["extraction"]
            wanted = (set(truth["uncertain"]) | set(truth["stated"])
                      | {d["object"] for d in truth["dependent"]})
            got = (set(found["uncertain"]) | set(found.get("stated", {}))
                   | {d["object"] for d in found["dependent"]})
            missed = [w for w in wanted if not any(same_object(w, g) for g in got)]
            # Only the ones the plan never managed to name are the ones extraction cost it.
            unusable = [w for w in missed if not any(same_object(w, n) for n in named)]
            if unusable:
                return "extraction"

            # The other way stage 1 can cause a failure: it read a location the task did
            # not state, the graph got a wrong edge from it, and the plan acted on that
            # edge - opening a cabinet for a mug that was on the counter all along. Only
            # counts when the step that failed is the object whose location was misread.
            stated = {d["object"]: (d["relation"], d["target"]) for d in truth["dependent"]}
            heard = {d["object"]: (d["relation"], d["target"]) for d in found["dependent"]}
            for name, want_rel in stated.items():
                match = next((h for h in heard if same_object(name, h)), None)
                misread = match is None or heard[match][0] != want_rel[0] \
                    or not same_object(heard[match][1], want_rel[1])
                if misread and failed_on and same_object(name, failed_on):
                    return "extraction"
            return verdict

        row = {
            "id": task["id"], "scene": task["scene"], "task": task["task"],
            "attempts": len(history), "accepted_at": winner["attempt"] if winner else None,
            "extraction": stage1, "grounding": stage3,
            "plain": plain, "plain_why": plain_why, "plain_cause": blame(plain, first, plain_at),
            "checked": checked, "checked_why": checked_why,
            "checked_cause": blame(checked, final, checked_at),
            # The object the failing step acted on, kept so a run can be re-attributed
            # later without replaying every plan.
            "checked_at": checked_at, "plain_at": plain_at,
            "mended": mended,
            "simulated": simulated,
            "seconds": round(time.time() - started, 1),
            # Kept so the whole thing can be re-scored later without re-running the LLM.
            "extracted": found, "predicted_goal": [list(g) for g in predicted],
            "first_plan": [[s["action"], s.get("object")] for s in first],
            "final_plan": [[s["action"], s.get("object")] for s in final],
        }
        rows.append(row)
        if out:
            with open(out, "w") as f:
                json.dump({"model": model, "complete": len(rows) == len(tasks),
                           "rows": rows}, f, indent=1)
        if verbose:
            print(f"{index:3d}/{len(tasks)} {row['id']:22s} "
                  f"first={row['plain']:9s} checked={row['checked']:9s} "
                  f"({row['attempts']} tries, {row['seconds']:.0f}s)  "
                  f"{(row['checked_cause'] or '') and row['checked_cause'] + ': '}"
                  f"{row['checked_why'][:46]}")
    return rows


def summarise(rows):
    from collections import Counter
    total = len(rows)
    lines = [f"\n{total} tasks\n"]
    for arm, key in (("without validation (first answer)", "plain"),
                     ("with validation (repair loop)", "checked")):
        ok = sum(1 for r in rows if r[key] == "ok")
        lines.append(f"  {arm:34s} {ok:3d}/{total} succeeded ({ok / total:.0%})")
        causes = Counter(r[f"{key}_cause"] for r in rows if r[key] != "ok")
        for cause, n in causes.most_common():
            lines.append(f"      {n:3d}  {cause}")
    # The third arm, when it was run: the same accepted plan, actually driven.
    driven = [r for r in rows if r.get("simulated")]
    if driven:
        ok = sum(1 for r in driven if r["simulated"]["ok"])
        lines.append(f"  {'executed in the 2-D simulator':34s} {ok:3d}/{len(driven)} "
                     f"succeeded ({ok / len(driven):.0%})")
        why = Counter()
        for r in driven:
            s_ = r["simulated"]
            if s_["ok"]:
                continue
            w = s_.get("why", "")
            why["the plan was already wrong" if r["checked"] != "ok"
                else "object not found by search" if "is in none of" in w
                else "could not get within reach" if "beyond the" in w
                else "goal not met once driven" if "goal not met" in w
                else "left unsafe" if "unsafe" in w
                else "other"] += 1
        for cause, n in why.most_common():
            lines.append(f"      {n:3d}  {cause}")
        metres = [r["simulated"].get("driven", 0.0) for r in driven if r["simulated"]["ok"]]
        if metres:
            lines.append(f"      {sum(metres) / len(metres):.1f} m driven on average "
                         f"by a run that finished")

    fixed = sum(1 for r in rows if r["plain"] != "ok" and r["checked"] == "ok")
    broke = sum(1 for r in rows if r["plain"] == "ok" and r["checked"] != "ok")
    lines.append(f"\n  repaired by the loop: {fixed}    made worse: {broke}")
    tries = Counter(r["accepted_at"] for r in rows if r["accepted_at"])
    lines.append("  accepted on attempt: "
                 + ", ".join(f"{k}: {v}" for k, v in sorted(tries.items())))

    # Where the errors come from, which is the question worth more than the headline rate.
    lines.append("\n  where the failures come from (with validation):")
    stage = Counter()
    for r in rows:
        if r["checked"] == "ok":
            continue
        stage["task_objects extraction" if r["checked_cause"] == "extraction"
              else "goal model read the task wrong" if r["checked_cause"] == "goal_model"
              else "RSN could not place it" if r["checked_cause"] == "grounding"
              else "LLM plan invalid" if r["checked_cause"] in ("planning", "unparsed")
              else "LLM plan valid but does not do the task" if r["checked_cause"] == "goal"
              else "left unsafe" if r["checked_cause"] == "safety"
              else str(r["checked_cause"])] += 1
    failures = sum(stage.values()) or 1
    for source, n in stage.most_common():
        lines.append(f"      {n:3d}  ({n / failures:4.0%} of failures)  {source}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default=TASKS)
    parser.add_argument("--limit", type=int, help="only the first N tasks")
    parser.add_argument("--scene", help="only this scene")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--repair-at", choices=("loop", "end", "off"), default="loop",
                        help="where the graph machine gets to mend: inside every attempt "
                             "(the default), only once after the attempts are spent, or "
                             "not at all. The last two are for ablations")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--models", nargs="+",
                        help="run the whole dataset once per model, freeing the GPU "
                             "between them, and write one file per model")
    parser.add_argument("--extractor", help="LoRA directory to do stage 1 with, instead "
                        "of prompting the planner's model for it")
    parser.add_argument("--goal-model", help="LoRA directory that reads the goal state out "
                        "of the task, so the checker can test whether the plan does it")
    parser.add_argument("--declare-goal", action="store_true",
                        help="have the planner state the finished world before it plans, "
                             "and check the plan against that instead of nothing")
    parser.add_argument("--simulate", action="store_true",
                        help="also drive each accepted plan in the 2-D simulator, so the "
                             "robot has to find its objects with a camera")
    parser.add_argument("--json", default="data/evaluation.json")
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)
    if args.scene:
        tasks = [t for t in tasks if t["scene"] == args.scene]
    if args.limit:
        tasks = tasks[:args.limit]

    for model in (args.models or [args.model]):
        print(f"\n{'=' * 72}\n{model}\n{'=' * 72}")
        out = (args.json if not args.models
               else args.json.replace(".json", f"-{model.split('/')[-1]}.json"))
        rows = evaluate(tasks, attempts=args.attempts, model=model, out=out,
                        extractor=args.extractor, simulate=args.simulate,
                        declare_goal=args.declare_goal, goal_model=args.goal_model,
                        mend=False if args.repair_at == "off" else args.repair_at)
        print(summarise(rows))
        with open(out, "w") as f:
            json.dump({"model": model, "complete": True, "rows": rows}, f, indent=1)
        print(f"\nwrote {out}")
        release_generator(model)


if __name__ == "__main__":
    raise SystemExit(main())
