#!/usr/bin/env python3
"""The multi-task benchmark: five ordering methods over one instruction naming several errands.

    SayPlan       the pipeline with no ordering stage - the errands in the order the model
                  decomposed them into
    GAVEL-MAP     reorders online, but carries one best guess per object - the maximum a
                  posteriori room - rather than a distribution, so it cannot tell a room it
                  is sure of from one it merely prefers
    GAVEL Static  orders once against the prior and never revises
    GAVEL         reorders online, carrying the full distribution
    Oracle        knows every object's room, uses the benchmark's reference subplans, and
                  takes the ordering that is genuinely shortest by driving all of them

`baselines.py` defines them; this drives them. **The decomposition and the subplans are
produced once and shared by all five**, so what separates the arms is the ordering and nothing
else - not a different sample of the model. Stages 1 to 6 are the single-task pipeline
unchanged, run once per errand.

Every arm is judged by the 2-D simulator against the true world. Two distances are reported:
`walked`, what the cost model charged, and `driven`, what the simulator really drove.

    python evaluate_multi.py --tasks data/multitask.json --limit 50 --out data/multi-4b.json
"""

import os as _os, sys as _sys
# Walk up to the repo root - the directory holding the library modules - so this file
# runs from wherever it is filed. Anchored on a marker rather than a fixed number of
# parents, so moving it a level deeper does not silently break the import.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.exists(_os.path.join(_d, 'graph_machine.py')):
    _d = _os.path.dirname(_d)
_sys.path.insert(0, _d)


import argparse
import json
import time

from evaluate import extraction_error, grounding_error
from graph_machine import GraphMachine
from object_names import same
from planner import get_generator, parse_goal
from build_tasks import seed_graph
from repair import repair
from replan import compose_complaint, run as replan_run
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, populate
from sim_eval import route_cost, route_matrix, run_plan
from task_objects import extract
from world_graph import WorldGraph

import baselines
import gavel
from build_multitask import merge_extraction


def ground(task, found):
    """Stage 3, shared by every arm so they differ only in what happens after it."""
    return populate(task["scene"], found["uncertain"], found["dependent"],
                    stated=found.get("stated") or {},
                    model_path=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD)


def canonical_goal(goal, graph):
    """Rewrite a predicted goal's object names into the names the belief graph uses.

    The goal adapter answers in the sentence's words. Asked about "switch on the playroom table
    lamp" it writes `toggled(playroom_table_lamp, True)`, and the graph holds `table_lamp`.
    `GraphMachine.unmet` matches goal names by equality, so the condition is unmeetable: every
    plan is refused for all five attempts and the run reports whatever the last attempt wrote.
    Both errands the loop never accepted in the first multi-task run were exactly this, and the
    same fault costs the single-task benchmark three tasks in each arm.

    `object_names.same` already knows these are the same thing - `tv`/`standing_tv`,
    `tshirt`/`t_shirt`, `bowls`/`bowl`. Nothing was asking it. This asks it, here in the
    harness, on the way in: it is the "canonicalised on the way in" that
    `GraphMachine._resolve_goal_name`'s docstring says already happens and that `parse_goal`,
    which normalises only the predicate vocabulary, does not do.

    Deliberately local to this harness. The single-task pipeline has the same gap and is the
    reference this work is measured against, so it is left exactly as it is.
    """
    names = list((graph.get("objects") or {}))
    def resolve(term):
        if not isinstance(term, str) or term in names:
            return term
        hits = [n for n in names if same(term, n)]
        return hits[0] if len(hits) == 1 else term
    return [(kind, resolve(src), resolve(dst) if isinstance(dst, str) else dst)
            for kind, src, dst in goal]


def plan_one(text, graph, model, attempts, goal_gen, mend, predicted=None, complaint=None):
    """One errand: read its goal, write a plan, let the machine mend it. Stages 5 and 6.

    `graph` is the scene-graph dict `populate` returns, not a `WorldGraph` - `replan.run`
    builds its own seed from it and needs the object records to write the prompt.
    """
    from finetune_extraction import GOAL_INSTRUCTION

    at = time.perf_counter()
    if predicted is not None:
        goal_gen = None                  # already read for this errand; do not ask twice
    else:
        predicted = ()
    if goal_gen:
        predicted = canonical_goal(
            parse_goal("GOAL:\n" + goal_gen(GOAL_INSTRUCTION.format(task=text), 200)), graph)
    goal_seconds = time.perf_counter() - at
    at = time.perf_counter()
    history = replan_run(text, graph, goal=predicted, attempts=attempts, model_name=model,
                         verbose=False, mend=mend, complaint=complaint)
    plan_seconds = time.perf_counter() - at
    if not history:
        return [], predicted, False, {"errand": text, "goal": [list(g) for g in predicted],
                                      "accepted": False, "why": "no reply parsed",
                                      "goal_seconds": round(goal_seconds, 2),
                                      "plan_seconds": round(plan_seconds, 2)}
    winner = next((h for h in history if h.get("accepted")), None)
    record = winner or history[-1]
    steps = [(s["action"], s.get("object")) for s in record["steps"]]
    outcome = record.get("outcome")
    # Enough to explain a failure without re-running it: what goal the errand was planned
    # against, what the mender did, and what the checker still objected to. Without the goal
    # in particular a failure is unreadable - the mender infers what to pick up from it, so
    # an errand planned against an empty goal has no repairs available and fails with no
    # grasp in it at all, which looks like the model forgetting rather than the goal missing.
    note = {"errand": text, "goal": [list(g) for g in predicted], "accepted": bool(winner),
            "attempts": len(history), "mended": record.get("mended") or [],
            "goal_seconds": round(goal_seconds, 2), "plan_seconds": round(plan_seconds, 2),
            "plan": [[a, o] for a, o in steps],
            "why": ("" if winner else
                    f"failed_at={getattr(outcome, 'failed_at', None)} "
                    f"missing={[list(m) for m in (getattr(outcome, 'missing', None) or [])]} "
                    f"safe={getattr(outcome, 'safe', None)}")}
    return steps, predicted, bool(winner), note


#: How many times a broken composition may send an errand back to the model.
COMPOSE_ROUNDS = 2


def blame(subplans, notes, outcome):
    """Which errand's plan reached outside its own errand, and what it broke.

    Composition can only fail two ways that a per-errand check cannot see: a step no longer
    applies because a predecessor changed the world, or a condition an errand established
    stops holding because a *later* errand undid it. The second is the one worth blaming,
    and the culprit is not the errand whose goal is now unmet - that errand did its job.

    So: take the objects named in the unmet conditions, and find an errand whose plan acts on
    one of them while its own goal never mentions it. On `Rs_int-m41` that is the tablespoon
    errand, whose plan opens with `GRASP(sliced_roast_beef)` and locks the beef in a cabinet
    another errand had just put on the table.

    Returns `(index, foreign_objects, missing)` or `(None, set(), missing)` when nothing in
    the plans explains it - a step that simply stopped applying, which the mender handles.
    """
    missing = [tuple(m) for m in (getattr(outcome, "missing", None) or [])]
    hurt = {m[1] for m in missing}
    hurt |= {m[2] for m in missing if isinstance(m[2], str)}
    for index, (plan, note) in enumerate(zip(subplans, notes)):
        own = set()
        for condition in (note.get("goal") or []):
            own.add(condition[1])
            if isinstance(condition[2], str):
                own.add(condition[2])
        # Only what the errand PICKS UP. An errand's own goal names its destination and
        # never its source - "take the sponge out of the pedestal sink and put it on the
        # coffee table" is `on_top(sponge, coffee_table)` - so counting every object a plan
        # navigates to or opens blames an errand for touching the very container it was sent
        # to empty. Undoing another errand's placement requires lifting the object out of it,
        # and `GRASP` is the only action that does.
        touched = {o for action, o in plan if o and action == "GRASP"}
        foreign = (touched & hurt) - own
        if foreign:
            return index, foreign, missing
    return None, set(), missing


def run_task(task, *, model, attempts, generator, extractor_gen, extract_prompt,
             goal_gen, mend, arms=baselines.ALL):
    goal = [tuple(g) for g in task["goal"]]
    row = {"id": task["id"], "scene": task["scene"], "subgoals": len(task["subgoals"]),
           "arms": {}}

    # Decompose and plan ONCE. Every method is handed the identical subplans, so what
    # separates them is the ordering and nothing else - not a different sample of the model.
    at = time.perf_counter()
    parts = gavel.decompose(task["task"], generator)
    t_decompose = time.perf_counter() - at

    at = time.perf_counter()
    pieces = []
    for part in parts:
        got = extract(part, generator=extractor_gen, prompt=extract_prompt)
        pieces.append({"extraction": {"uncertain": got["uncertain"],
                                      "stated": got.get("stated") or {},
                                      "dependent": got["dependent"]}})
    merged = merge_extraction(pieces) if pieces else None
    t_extract = time.perf_counter() - at

    at = time.perf_counter()
    graph = ground(task, merged)
    seed = WorldGraph.from_scene_graph(graph)
    t_ground = time.perf_counter() - at
    subplans, accepted, notes, predicted = [], [], [], []
    for part in parts:
        steps, goals, ok, note = plan_one(part, graph, model, attempts, goal_gen, mend)
        subplans.append(steps)
        accepted.append(ok)
        notes.append(note)
        predicted += [tuple(g) for g in goals]
    # Stage 8b: does the concatenation still do the job? Every errand was checked against the
    # world as it stood when that errand began, and no such check can see the errand that runs
    # after it. `gavel.compose` is the only thing that looks at the whole sequence, and its
    # verdict used to be recorded and thrown away - on `Rs_int-m41` it correctly reported that
    # the tablespoon errand had picked the roast beef off the table another errand had just
    # put it on and shut it in a cabinet, and nothing acted on that.
    #
    # Checked against `predicted`, never `task["goal"]`. This feeds back into the plan, so
    # steering it on the answer key would be measuring a pipeline that cannot exist.
    at = time.perf_counter()
    composed_fixes = []
    for _ in range(COMPOSE_ROUNDS):
        live = [i for i, p in enumerate(subplans) if p]
        if not live:
            break
        _, outcome = gavel.compose(seed, [subplans[i] for i in live], predicted)
        if outcome.failed_at is None and not outcome.missing and outcome.safe:
            break
        where, foreign, missing = blame([subplans[i] for i in live],
                                        [notes[i] for i in live], outcome)
        if where is None or not foreign:
            break                    # a step that stopped applying; the mender handles those
        index = live[where]
        complaint = compose_complaint(
            [tuple(g) for g in (notes[index].get("goal") or [])], foreign, missing)
        steps, _goals, ok, note = plan_one(
            parts[index], graph, model, attempts, goal_gen, mend,
            predicted=[tuple(g) for g in (notes[index].get("goal") or [])],
            complaint=complaint)
        composed_fixes.append({"errand": parts[index], "foreign": sorted(foreign),
                               "undone": [list(m) for m in missing],
                               "accepted": bool(ok)})
        subplans[index], notes[index], accepted[index] = steps, note, ok
    t_compose = time.perf_counter() - at
    row["composed_fixes"] = composed_fixes

    plans = [p for p in subplans if p]

    # SayPlan plans for itself. It gets the same decomposition, the same beliefs, the same goal
    # and the same five attempts against the same `GraphMachine` - it sees the machine's
    # complaint and rewrites - but `mend=False`, so nothing edits the plan on its behalf. That
    # is the baseline: validation and feedback without graph repair, and without a cost model.
    # It therefore cannot share the other arms' subplans, which were mended; sharing them was
    # crediting SayPlan with a repair stage it is defined not to have.
    def replan_pass(tries, mender):
        """Plan every errand again under a different validation regime, reusing the goal the
        adapter already read so the pass differs only in what it was allowed to do about a
        refusal - not in what it was aiming at."""
        at = time.perf_counter()
        got, got_notes = [], []
        for part, note in zip(parts, notes):
            steps, _g, _ok, made = plan_one(part, graph, model, tries, goal_gen, mender,
                                            predicted=[tuple(g) for g in note["goal"]])
            got.append(steps)
            got_notes.append(made)
        return [p for p in got if p], got_notes, time.perf_counter() - at

    # Only pay for a pass whose arm was asked for. SayPlan sees the machine's complaint and
    # rewrites, five times, but nothing edits its plan; LLM-only gets one attempt and is taken
    # as written. Neither can share the mended subplans the GAVEL arms use, which was crediting
    # them with a repair stage they are defined not to have.
    say_plans, say_notes, t_plan_sayplan = ([], [], 0.0)
    if baselines.SAYPLAN in arms:
        say_plans, say_notes, t_plan_sayplan = replan_pass(attempts, False)
        row["sayplan_errands"] = say_notes
    only_plans, only_notes, t_plan_only = ([], [], 0.0)
    if baselines.LLM_ONLY in arms:
        only_plans, only_notes, t_plan_only = replan_pass(1, False)
        row["llm_only_errands"] = only_notes
    row["extraction"] = extraction_error(task, merged) if merged else None
    row["extracted"] = {"uncertain": sorted((merged or {}).get("uncertain") or []),
                        "dependent": (merged or {}).get("dependent") or [],
                        "stated": (merged or {}).get("stated") or {}}
    row["grounding"] = grounding_error(task, graph)[0]
    row["parts"] = len(parts)
    row["errands"] = notes
    row["accepted"] = all(accepted) if accepted else False

    # What each module of the pipeline cost. The methods do not all run the same modules, so
    # a single "pipeline time" would overcharge the ones that skip a stage: EPoG needs the
    # instruction split, the objects extracted and the goal read, but writes its own actions
    # from the graph diff and never calls the planner or the repair loop, and Oracle is handed
    # the reference plans and calls no model at all.
    at = time.perf_counter()
    gavel._table(task["scene"])          # the A* room-distance table, built once per scene
    t_setup = time.perf_counter() - at
    row["module_seconds"] = {
        "decompose": round(t_decompose, 2),
        "extract": round(t_extract, 2),
        "ground": round(t_ground, 2),
        "goal": round(sum(n.get("goal_seconds") or 0.0 for n in notes), 2),
        "plan": round(sum(n.get("plan_seconds") or 0.0 for n in notes), 2),
        "plan_sayplan": round(t_plan_sayplan, 2),
        "plan_llm_only": round(t_plan_only, 2),
        "compose": round(t_compose, 2),
        "cost_table": round(t_setup, 2),
    }

    def drive(ordered):
        steps, _ = gavel.compose(seed, ordered, goal)
        try:
            return run_plan(task, graph, steps, verbose=False)["driven"]
        except Exception:
            return None

    # ORACLE is given ground truth in every sense, not only when it chooses. The other arms
    # are driven against `graph`, the belief, so the simulator makes them sweep for anything
    # the RSN put in the wrong room; driving Oracle that way made it pay a search cost its
    # ordering could not possibly have anticipated, and its own route computation then
    # disagreed with what it drove on 2 of 9 tasks. Handed the truth graph it grounds to the
    # true instance and walks straight to it, so the computed route and the driven one are the
    # same number and one simulation is exactly optimal.
    truth_graph = seed_graph(task)
    oracle_legs = oracle_start = None

    def oracle_drive(ordered):
        steps, _ = gavel.compose(seed, ordered, goal)
        try:
            return run_plan(task, truth_graph, steps, verbose=False)["driven"]
        except Exception:
            return None

    def oracle_measure(order):
        nonlocal oracle_legs, oracle_start
        steps, _ = gavel.compose(seed, [reference[i] for i in order], goal)
        if oracle_legs is None:
            oracle_legs, oracle_start = route_matrix(task, truth_graph, steps)
        return route_cost(steps, oracle_legs, oracle_start)

    # The benchmark's own reference subplans, for the methods that do not plan with the model.
    # `baselines.uses_llm_plans` says which those are, and ORACLE is the reason it exists: it
    # is the floor, so ordering the *model's* subplans makes it inherit the model's redundant
    # drives. On `Merom_1_int-m01` that cost it six extra actions and 21 m, and EPoG - which
    # writes its own plan from the graph diff and so never emits them - came in under the
    # "floor". A floor another method gets under is not a floor.
    reference = [[tuple(s) for s in sub["plan"]] for sub in task["subgoals"]]

    for arm in arms:
        started = time.time()
        row["arms"][arm] = {}
        uses_llm = baselines.uses_llm_plans(arm)
        theirs = (say_plans if arm == baselines.SAYPLAN else
                  only_plans if arm == baselines.LLM_ONLY else
                  plans if uses_llm else reference)
        if not theirs or not any(theirs):
            row["arms"][arm].update({"ok": False, "why": "no plan", "walked": None,
                                     "seconds": 0.0})
            continue
        # EPoG is given the goal the *pipeline* predicted, not the answer key - the same
        # information GAVEL plans against, or the comparison would be measuring the goal
        # adapter rather than the planner.
        modules = row["module_seconds"]
        if arm == baselines.ORACLE:
            uses = ()                     # reference plans, true positions, no model at all
        elif arm == baselines.EPOG:
            uses = ("decompose", "extract", "ground", "goal", "cost_table")
        elif arm == baselines.SAYPLAN:
            # its own planning pass, not the mended one every other arm shares
            uses = ("decompose", "extract", "ground", "goal", "plan_sayplan")
        elif arm == baselines.LLM_ONLY:
            uses = ("decompose", "extract", "ground", "goal", "plan_llm_only")
        else:
            uses = ("decompose", "extract", "ground", "goal", "plan", "compose",
                    "cost_table")
        row["arms"][arm]["module_seconds"] = round(sum(modules[k] for k in uses), 2)
        # ORACLE alone is allowed the answer key - knowing where everything is and what is
        # being asked for is its definition. Everybody else plans against `predicted`, the
        # goal the adapter read off the instruction, **including when that is empty**: an
        # `or goal` fallback there would hand EPoG the true goal on exactly the runs where
        # its goal adapter failed, which is the one case it should be marked down for.
        is_oracle = arm == baselines.ORACLE
        result = baselines.run(
            arm, task, theirs, truth_graph if is_oracle else graph, seed,
            drive=oracle_drive if is_oracle else drive,
            measure=oracle_measure if is_oracle else None,
            goal=goal if is_oracle else predicted)
        entry = row["arms"][arm]
        if result.get("steps") is not None:
            steps = result["steps"]
            outcome = GraphMachine(seed.copy()).run(steps, goal)
        else:
            ordered = [theirs[i] for i in result["order"]]
            steps = [s for plan in ordered for s in plan]
            if baselines.uses_repair(arm):
                # The mender, on the whole plan, exactly as the single-task pipeline runs it
                # on a single plan - and against `predicted`, so it never sees the answer key.
                steps, fixes = repair(seed.copy(), steps, predicted)
                if fixes:
                    entry["compose_mended"] = fixes
            outcome = GraphMachine(seed.copy()).run(steps, goal)
        # Stage 8's answer, against the graph the robot believes: do these subplans still
        # do the job when run back to back? It is the only check that sees the composition.
        composed_ok = outcome.failed_at is None and outcome.safe and not outcome.missing
        # Why it disagreed, when it does. Stage 8 judges on the graph the robot *believes*
        # and the simulator judges the world as it is, so the two can part company - and
        # without the reason recorded there is no way to tell a real composition fault from
        # a belief that was simply wrong about where something was.
        composed_why = ("" if composed_ok else
                        f"failed_at={outcome.failed_at} "
                        f"missing={[list(m) for m in (outcome.missing or [])]} "
                        f"safe={outcome.safe}")

        # And the verdict that counts: driven in the simulator, against the true world.
        # `run_plan` grounds the plan *and* the goal from categories to instances, which a
        # bare `GraphMachine` on the truth graph does not - the goal says `breakfast_table`
        # and the house holds `breakfast_table_skczfi_1`, so checking it unbound marks a run
        # that did exactly what was asked as having missed.
        try:
            # Every arm, ORACLE included, is driven through the same simulator here. ORACLE
            # already drove each candidate ordering to choose between them, so re-driving the
            # winner is one extra run and it is what makes its step count, its control time
            # and its success measured the same way as everyone else's rather than asserted.
            sim = run_plan(task, truth_graph if is_oracle else graph, steps, verbose=False)
        except Exception as exc:                       # a simulator fault is not a plan fault
            sim = {"ok": False, "why": f"{type(exc).__name__}: {exc}", "error": True,
                   "failed_at": None, "driven": None, "missing": [], "unsafe": []}
        entry.update({
            "ok": bool(sim.get("ok")),
            "why": sim.get("why", ""),
            "composed_ok": composed_ok,
            "composed_why": composed_why,
            "plan": [[a, o] for a, o in steps],
            "failed_at": sim.get("failed_at"),
            "goal_met": sim.get("goal_met"),
            "missing": sim.get("missing") or [],
            "unsafe": sim.get("unsafe") or [],
            "steps": len(steps),
            "order": result["order"],
            "driven": sim.get("driven"),
            "sim_steps": sim.get("sim_steps"),
            "sim_seconds": sim.get("sim_seconds"),
            "order_seconds": result.get("order_seconds"),
            # The method's own cost estimate, kept for diagnosis and deliberately not
            # reported: the methods do not all estimate the same quantity, so the column is
            # not comparable across rows. EPoG's `C_MAP` charges navigation only, because a
            # collapsed belief leaves nothing to search for; GAVEL's also charges the search.
            "walked": round(result["walked"], 1) if result.get("walked") is not None else None,
            "estimated": (round(result["estimated"], 1)
                          if result.get("estimated") is not None else None),
            "reorders": result.get("reorders", 0),
            "orders_tried": result.get("orders_tried"),
            "orders_valid": result.get("orders_valid"),
            "seconds": round(time.time() - started, 1),
        })
    return row


def summarise(rows):
    """Success rate, travel distance and running time, per method.

    Running time is everything the method spends: every module of the algorithm it actually
    runs - the model queries included - plus the robot's own time in the simulator at
    `sim_eval.SIM_STEP_SECONDS` per control step. The methods do not run the same modules, so
    the compute column is not a constant offset between rows: EPoG skips the planner and the
    repair loop, SayPlan skips the cost model, Oracle calls no model at all.

    **Distance and time are reported over the tasks every method solved.** A failed run stops
    where it failed, so its distance and its simulator time are truncated: averaging those
    over each method's own success set pays a method for failing early, and it did - an arm
    that solved 24 of 50 came out 6.6% "shorter" than one that solved the same 24 and drove
    them all to the end. The per-method means over each arm's own successes are printed
    underneath, where they can be read for what they are.
    """
    arms = [a for a in baselines.ALL if any(a in r["arms"] for r in rows)]
    mean = lambda xs: sum(xs) / len(xs) if xs else 0.0
    got = lambda r, a: r["arms"].get(a) or {}
    solved = lambda r, a: got(r, a).get("ok") and got(r, a).get("driven") is not None
    common = [r for r in rows if all(solved(r, a) for a in arms)]

    print(f"\n{len(rows)} instructions, {len(common)} solved by every method\n")
    print(f"  {'method':14s} {'success':>9s} {'driven':>10s} "
          f"{'compute':>9s} {'sim time':>10s} {'total':>9s}")
    for arm in arms:
        every = [got(r, arm) for r in rows if arm in r["arms"]]
        ok = sum(1 for g in every if g.get("ok"))
        here = [got(r, arm) for r in common]
        driven = mean([g["driven"] for g in here])
        compute = mean([(g.get("module_seconds") or 0.0) + (g.get("order_seconds") or 0.0)
                        for g in here])
        simt = mean([g.get("sim_seconds") or 0.0 for g in here])
        print(f"  {baselines.LABELS[arm]:14s} {ok:4d}/{len(every):<4d} "
              f"{driven:9.1f}m {compute:8.1f}s {simt:9.1f}s {compute + simt:8.1f}s")
    print(f"\n  success is over all {len(rows)} instructions; distance and time over the "
          f"{len(common)} every method solved.")
    print("  compute = every module the method runs, model queries included."
          "  sim = control steps x 0.1 s.")
    print("  distance is what the robot DROVE; each method's internal cost estimate is stored"
          " but not reported.")

    print("\n  for reference, each method over its own successes (not comparable across rows"
          " - different task sets):")
    for arm in arms:
        mine = [got(r, arm) for r in rows
                if arm in r["arms"] and solved(r, arm)]
        print(f"    {baselines.LABELS[arm]:14s} {len(mine):3d} tasks  "
              f"{mean([g['driven'] for g in mine]):7.1f}m")

    # Where each method's compute goes, so the totals above are auditable.
    print("\n  compute breakdown (mean seconds per instruction):")
    keys = ("decompose", "extract", "ground", "goal", "plan", "plan_sayplan",
            "plan_llm_only", "compose", "cost_table")
    print("    " + "".join(f"{k:>11s}" for k in keys))
    shared = {k: mean([(r.get("module_seconds") or {}).get(k) or 0.0 for r in rows])
              for k in keys}
    print("    " + "".join(f"{shared[k]:>10.2f}s" for k in keys))
    for arm in arms:
        every = [got(r, arm) for r in rows if arm in r["arms"]]
        print(f"    {baselines.LABELS[arm]:14s} runs "
              f"{mean([g.get('module_seconds') or 0.0 for g in every]):6.1f}s of those, "
              f"plus {mean([g.get('order_seconds') or 0.0 for g in every]):.2f}s ordering")

    base = baselines.GAVEL
    print("\n  paired against GAVEL, on driven distance:")
    for arm in arms:
        if arm == base:
            continue
        # Both arms must have SOLVED the task, not merely have a distance recorded: a run
        # that failed halfway drove half as far, and pairing on that reads as a saving.
        pairs = [(r["arms"][arm], r["arms"][base]) for r in rows
                 if arm in r["arms"] and base in r["arms"]
                 and r["arms"][arm].get("ok") and r["arms"][base].get("ok")
                 and r["arms"][arm].get("driven") is not None
                 and r["arms"][base].get("driven") is not None]
        if not pairs:
            continue
        a = sum(x["driven"] for x, _ in pairs); b = sum(y["driven"] for _, y in pairs)
        print(f"    {baselines.LABELS[arm]:14s} {a:8.0f} m  ->  GAVEL {b:8.0f} m   "
              f"{100*(a-b)/max(a,1e-9):+6.1f}%   over {len(pairs)} tasks")

    bad1 = sum(1 for r in rows if r.get("extraction"))
    print(f"\n  stage-1 extraction errors {bad1}/{len(rows)}, "
          f"stage-3 grounding errors {sum(1 for r in rows if r.get('grounding'))}/{len(rows)}")
    stuck = [(r["id"], e) for r in rows for e in (r.get("errands") or [])
             if isinstance(e, dict) and not e.get("accepted")]
    print(f"  errands the validation loop never accepted: {len(stuck)}")
    for tid, e in stuck[:10]:
        print(f"    {tid:22s} {e['errand'][:52]:52s} {e.get('why','')[:56]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="data/multitask.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=1,
                        help="take every Nth task, to sample the scenes evenly")
    parser.add_argument("--shard", type=int, default=0,
                        help="which slice of the selected tasks this process takes")
    parser.add_argument("--shards", type=int, default=1,
                        help="how many processes are splitting the work; the selection is "
                             "made first and then dealt round-robin, so every shard gets "
                             "the same mix of scenes and errand counts")
    parser.add_argument("--model", default=None)
    parser.add_argument("--extractor", default=None)
    parser.add_argument("--goal-model", dest="goal_model", default=None)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--mend", default="loop")
    parser.add_argument("--arms", default=",".join(baselines.ALL))
    parser.add_argument("--out")
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))[::args.stride]
    if args.limit:
        tasks = tasks[:args.limit]
    if args.shards > 1:
        tasks = tasks[args.shard::args.shards]

    generator = get_generator(args.model) if args.model else get_generator()
    extractor_gen, extract_prompt = generator, None
    if args.extractor:
        from finetune_extraction import INSTRUCTION
        extractor_gen = get_generator(adapter=args.extractor)
        extract_prompt = INSTRUCTION + "\n"
    goal_gen = get_generator(adapter=args.goal_model) if args.goal_model else None

    from build_multitask import stamp_of
    stamp = stamp_of(json.load(open(args.tasks)), None, None)

    rows = []
    for index, task in enumerate(tasks, 1):
        row = run_task(task, model=args.model, attempts=args.attempts,
                       generator=generator,
                       extractor_gen=extractor_gen, extract_prompt=extract_prompt,
                       goal_gen=goal_gen, mend=args.mend,
                       arms=tuple(args.arms.split(",")))
        rows.append(row)
        marks = " ".join(f"{a}={'ok' if row['arms'][a].get('ok') else 'X'}"
                         f"/{row['arms'][a].get('walked')}" for a in row["arms"])
        print(f"[{index}/{len(tasks)}] {row['id']:24s} {marks}", flush=True)
        if args.out:
            with open(args.out, "w") as handle:
                json.dump(rows, handle, indent=1)
            with open(args.out.replace(".json", "-stamp.json"), "w") as handle:
                # What the run cost, when a hosted model served it. Written beside the
                # results so a column can be priced later without re-running it, and so a
                # rate-limited or truncated run is visible as a call count that does not
                # match the instruction count.
                import api_models
                json.dump({**stamp, "model": args.model,
                           "api_usage": api_models.spent() or None}, handle, indent=1)
    summarise(rows)


if __name__ == "__main__":
    raise SystemExit(main())
