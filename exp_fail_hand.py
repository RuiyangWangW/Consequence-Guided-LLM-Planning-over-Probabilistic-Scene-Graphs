#!/usr/bin/env python3
"""Why the mender never fixed the 4B's four "nothing in the hand to place" refusals.

Four 4B rows die on `PLACE_ON_TOP: nothing in the hand to place` after five attempts, and
the 8B has none.  `repair._edit` has a rule for exactly that fault - read the placement
backwards through the goal and splice in the drive-and-grasp for whatever the goal wants at
that destination - so the question is why the rule declined.

This script does not guess.  It

  1. recovers the model's RAW final attempt by inverting the recorded mend notes, and
     proves the recovery by re-running `repair` on it and demanding the recorded notes and
     the recorded `final_plan` come back byte for byte;
  2. runs the machine on the final plan and prints the failing step and its `fault`;
  3. re-runs `repair` with the PREDICTED goal and with the TRUE goal, on the raw attempt,
     on the recorded final plan, and on attempt 1, and reports what each does;
  4. checks the known `unmet` name-comparison defect on these four goals;
  5. tests one counterfactual mender rule - a stranded placement whose goal is already
     satisfied is DELETED rather than left alone - and drives whatever it accepts through
     the 2D simulator against the TRUE goal.

Read-only with respect to every other module.
"""

import json
import sys

from build_tasks import seed_graph
from world_graph import WorldGraph
from graph_machine import GraphMachine
from object_names import same, candidates
import repair as R
from sim_eval import run_plan

IDS = ["Benevolence_1_int-07", "Benevolence_1_int-09",
       "Ihlen_1_int-07", "Pomaria_1_int-08"]

TASKS = {t["id"]: t for t in json.load(open("data/tasks.json"))}


def rows(path):
    return {r["id"]: r for r in json.load(open(path))["rows"]}


FOUR = rows("data/v8-4b.json")
EIGHT = rows("data/v8-8b.json")


def show(plan, mark=None, indent="    "):
    for i, (a, o) in enumerate(plan, 1):
        flag = "   <-- REFUSED" if mark is not None and i == mark + 1 else ""
        print(f"{indent}{i:2d}. {a}({o or ''}){flag}")


def accepted(out):
    """The loop's own acceptance test, verbatim from replan.run."""
    return out.failed_at is None and out.safe and out.goal_met


def verdict(out):
    if out.failed_at is None:
        return (f"ran to the end; goal_met={out.goal_met} safe={out.safe} "
                f"missing={[tuple(m) for m in out.missing]}")
    s = out.steps[out.failed_at]
    return f"step {out.failed_at + 1} {s.action}: {s.reason}   fault={s.fault}"


# ---------------------------------------------------------------- invert the mend notes

def parse_note(note):
    """'holding(mug) at step 4: 18 -> 20 actions' -> ('holding','mug',3,18,20)."""
    if note.startswith("discharged"):
        return ("discharge", None, None, None, int(note.split()[1]))
    head, tail = note.split(" at step ", 1)
    kind, subject = head.split("(", 1)
    subject = subject.rstrip(")")
    step, lengths = tail.split(": ", 1)
    before, after = lengths.replace(" actions", "").split(" -> ")
    return (kind, subject, int(step) - 1, int(before), int(after))


def unapply(plan, note):
    """Undo one recorded edit, exactly inverting `repair._edit`."""
    kind, subject, at, before, after = parse_note(note)
    if kind == "discharge":
        return plan[:len(plan) - after]
    if kind in ("not_near", "closed", "not_open"):        # inserted 1 at `at`
        return plan[:at] + plan[at + 1:]
    if kind in ("holding", "empty_hand"):                  # inserted 2 at `at`
        return plan[:at] + plan[at + 2:]
    if kind == "not_graspable":                            # deleted the GRASP at `at`
        return plan[:at] + [("GRASP", subject)] + plan[at:]
    if kind == "room":                                     # deleted the NAVIGATE at `at`
        return plan[:at] + [("NAVIGATE_TO", subject)] + plan[at:]
    raise SystemExit(f"no inverse for {note!r}")


def recover_written(row):
    """The model's raw last attempt, and whether the recovery is proven."""
    plan = [(a, o) for a, o in row["final_plan"]]
    notes = row["mended"] or []
    for note in reversed(notes):
        plan = unapply(plan, note)
    return plan


# ---------------------------------------- counterfactual: delete a satisfied placement

def _edit2(graph, plan, index, kind, subject, goal=()):
    """`repair._edit` plus one rule.

    A `PLACE_ON_TOP` with an empty hand whose destination already holds everything the
    goal wants there is not a missing grasp - there is nothing left to fetch.  It is a
    redundant step, and the edit that does not invent intent is to drop it, exactly as
    `not_graspable` drops a grasp on furniture.
    """
    action, arg = plan[index]
    edit = R._edit(graph, plan, index, kind, subject, goal)
    if edit is not None or kind != "empty_hand" or action not in R.PLACES:
        return edit
    want = R.PLACES[action]
    outstanding = [m for rel, m, d in goal
                   if rel == want and R._same(d, arg)
                   and m in graph.objects and not graph.has_edge(want, m, arg)]
    if outstanding:
        return None
    return plan[:index] + plan[index + 1:]


def repair2(graph, plan, goal=(), rounds=R.MAX_ROUNDS):
    """`repair.repair` with `_edit2`.  Structure copied so the comparison is like-for-like."""
    def run(candidate):
        return GraphMachine(graph.copy()).run(candidate, goal)

    start_out = run(plan)
    best, best_out = list(plan), start_out
    current, notes, seen = list(plan), [], {tuple(plan)}
    for _ in range(rounds):
        out = run(current)
        if R._rank(out, len(current)) > R._rank(best_out, len(best)):
            best, best_out = list(current), out
        if out.failed_at is None:
            break
        kind, subject = out.steps[out.failed_at].fault or (None, None)
        if kind not in R.REPAIRABLE:
            break
        candidate = _edit2(out.graph, current, out.failed_at, kind, subject, goal)
        if candidate is None or tuple(candidate) in seen:
            break
        seen.add(tuple(candidate))
        notes.append(f"{kind}({subject}) at step {out.failed_at + 1}: "
                     f"{len(current)} -> {len(candidate)} actions")
        current = candidate
    if best_out.failed_at is None and not best_out.safe:
        candidate = R._discharge(best, best_out)
        out = run(candidate)
        if R._rank(out, len(candidate)) > R._rank(best_out, len(best)):
            notes.append(f"discharged {len(candidate) - len(best)} left-open/left-on")
            best, best_out = candidate, out
    if R._rank(best_out, len(best)) <= R._rank(start_out, len(plan)):
        return list(plan), []
    return best, notes



# ---------------------------------------------------- counterfactual 2: self-placement

def unself(plan, goal, graph):
    """Rewrite `PLACE_ON_TOP(x)` issued while holding `x` to the goal's destination for x.

    The machine does not refuse this today - it writes `on_top(x, x)` and carries on - so
    this stands in for the pair of changes it would take: a new refusal in
    `GraphMachine.step` and the matching rule in `repair._edit`.  Applied as a prescan so
    the rest of the pipeline is untouched.
    """
    out, held, notes = [], None, []
    for action, arg in plan:
        if action == "GRASP":
            held = arg
        elif action == "RELEASE":
            held = None
        elif action in R.PLACES and held is not None:
            if R._same(arg, held):
                want = R.PLACES[action]
                dest = next((d for rel, m, d in goal
                             if rel == want and R._same(m, held)), None)
                if dest is not None:
                    notes.append(f"self-placement {action}({arg}) -> {action}({dest})")
                    arg = dest
            held = None
        out.append((action, arg))
    return out, notes


# ------------------------------------------------- counterfactual 3: stop when done

def truncate(graph, plan, goal):
    """The shortest prefix that applies, meets the goal and is safe - if there is one."""
    for n in range(1, len(plan) + 1):
        out = GraphMachine(graph.copy()).run(plan[:n], goal)
        if accepted(out):
            return plan[:n], n
    return None, None


def counts(plan):
    from collections import Counter
    c = Counter(a for a, _ in plan)
    return f"GRASP={c['GRASP']} PLACE={c['PLACE_ON_TOP'] + c['PLACE_INSIDE']} RELEASE={c['RELEASE']}"


# ------------------------------------------------------------------------------- report

def name_defect(goal, graph):
    """Does `unmet`'s `==` on goal object names bite on this goal?"""
    bad = []
    for rel, src, dst in goal:
        for term in (src, dst):
            if not isinstance(term, str) or term in graph.objects:
                continue
            hit = [o for o in graph.objects if same(term, o)]
            bad.append((term, hit))
    return bad


def main():
    for tid in IDS:
        task, row = TASKS[tid], FOUR[tid]
        seed = WorldGraph.from_scene_graph(seed_graph(task))
        predicted = [tuple(g) for g in row["predicted_goal"]]
        truth = [tuple(g) for g in task["goal"]]
        first = [(a, o) for a, o in row["first_plan"]]
        final = [(a, o) for a, o in row["final_plan"]]

        print("=" * 88)
        print(f"{tid}   4B accepted_at={row['accepted_at']}  attempts={row['attempts']}")
        print(f"  task: {task['task']}")
        print(f"  predicted goal: {predicted}")
        print(f"  true goal     : {truth}")
        print(f"  goal identical to truth: {set(map(tuple, predicted)) == set(map(tuple, truth))}")
        print(f"  unmet()-name defect on predicted goal: {name_defect(predicted, seed) or 'none'}")
        print(f"  recorded mend notes: {row['mended']}")

        out_final = GraphMachine(seed.copy()).run(final, predicted)
        print("\n  -- FINAL PLAN as scored by the loop (predicted goal)")
        show(final, out_final.failed_at)
        print(f"     {verdict(out_final)}")
        print(f"     recorded checked_why: {row['checked_why']}")
        if out_final.failed_at is not None:
            f = out_final.steps[out_final.failed_at]
            g = out_final.graph
            act, arg = final[out_final.failed_at]
            print(f"     at the refusal the hand holds: {GraphMachine(seed.copy()).held!r} "
                  f"(seed) / {g.__class__.__name__} state below")
            want = R.PLACES.get(act)
            if want:
                for rel, m, d in predicted:
                    if rel == want and R._same(d, arg):
                        print(f"     goal wants {rel}({m},{d}); already true in the world "
                              f"at the refusal? {g.has_edge(want, m, d)}")

        # --- recover the raw written attempt and prove it
        written = recover_written(row)
        redone, notes2 = R.repair(seed, written, predicted)
        proof = (notes2 == (row["mended"] or []) and
                 [list(x) for x in redone] == row["final_plan"])
        print(f"\n  -- RAW final attempt recovered by inverting the notes "
              f"({len(written)} actions); re-running repair reproduces the "
              f"recorded notes AND final_plan: {proof}")
        show(written)
        print("  -- RAW attempt 1 (row['first_plan'], never mended)")
        show(first)

        # --- what repair does with each goal, on each plan
        print("\n  -- repair() outcomes")
        for label, plan in (("attempt 1 raw", first),
                            ("attempt 5 raw", written),
                            ("attempt 5 mended (final_plan)", final)):
            for gname, goal in (("PREDICTED", predicted), ("TRUE", truth)):
                fixed, notes = R.repair(seed, plan, goal)
                out = GraphMachine(seed.copy()).run(fixed, goal)
                print(f"     {label:30s} {gname:9s} notes={notes or '[]'}")
                print(f"       -> {verdict(out)}   ACCEPTED={accepted(out)}")

        # --- the counterfactual mender
        print("\n  -- counterfactual mender (delete a placement the goal already satisfies)")
        for label, plan in (("attempt 5 raw", written),
                            ("attempt 5 mended (final_plan)", final)):
            for gname, goal in (("PREDICTED", predicted), ("TRUE", truth)):
                fixed, notes = repair2(seed, plan, goal)
                out = GraphMachine(seed.copy()).run(fixed, goal)
                ok = accepted(out)
                print(f"     {label:30s} {gname:9s} notes={notes or '[]'}")
                print(f"       -> {verdict(out)}   ACCEPTED={ok}")
                if ok and gname == "PREDICTED":
                    show(fixed, indent="         ")
                    drive = run_plan(task, seed_graph(task),
                                     [{"action": a, "object": o} for a, o in fixed])
                    print(f"       DRIVEN in sim2d: ok={drive['ok']} why={drive['why']!r} "
                          f"goal_met={drive.get('goal_met')} "
                          f"missing={drive.get('missing')} unsafe={drive.get('unsafe')}")

        # --- where the empty hand comes from
        print("\n  -- provenance of the empty hand")
        for label, plan in (("attempt 1 raw", first), ("attempt 5 raw", written),
                            ("attempt 5 mended", final)):
            out = GraphMachine(seed.copy()).run(plan, predicted)
            fault = (out.steps[out.failed_at].fault if out.failed_at is not None else None)
            print(f"     {label:18s} {counts(plan):34s} first fault = {fault}")

        # --- counterfactual 2 + 3
        print("\n  -- counterfactual 2: refuse+rewrite a self-placement, then mend")
        for label, plan in (("attempt 1 raw", first), ("attempt 5 raw", written)):
            for gname, goal in (("PREDICTED", predicted), ("TRUE", truth)):
                pre, pnotes = unself(plan, goal, seed)
                fixed, notes = repair2(seed, pre, goal)
                out = GraphMachine(seed.copy()).run(fixed, goal)
                ok = accepted(out)
                print(f"     {label:14s} {gname:9s} prescan={pnotes or '[]'} notes={notes or '[]'}")
                print(f"       -> {verdict(out)}   ACCEPTED={ok}")
                if ok and gname == "PREDICTED":
                    drive = run_plan(task, seed_graph(task),
                                     [{"action": a, "object": o} for a, o in fixed])
                    print(f"       DRIVEN in sim2d: ok={drive['ok']} why={drive['why']!r} "
                          f"goal_met={drive.get('goal_met')} missing={drive.get('missing')}")

        print("\n  -- counterfactual 3: stop the plan as soon as the goal holds")
        for label, plan in (("attempt 1 raw", first), ("attempt 5 raw", written),
                            ("attempt 5 mended", final)):
            for gname, goal in (("PREDICTED", predicted), ("TRUE", truth)):
                cut, n = truncate(seed, plan, goal)
                print(f"     {label:18s} {gname:9s} shortest accepted prefix: "
                      + (f"{n} of {len(plan)} actions" if cut else "none"))
                if cut and gname == "PREDICTED":
                    drive = run_plan(task, seed_graph(task),
                                     [{"action": a, "object": o} for a, o in cut])
                    print(f"       DRIVEN in sim2d: ok={drive['ok']} why={drive['why']!r} "
                          f"goal_met={drive.get('goal_met')} missing={drive.get('missing')}")

        # --- the 8B on the same task
        e = EIGHT[tid]
        print(f"\n  -- 8B on the same task: accepted_at={e['accepted_at']} "
              f"attempts={e['attempts']} checked={e['checked']} ({e['checked_why']}) "
              f"mended={e['mended']}")
        print("     8B first_plan:")
        show([(a, o) for a, o in e["first_plan"]], indent="       ")
        print()


def sweep():
    """Does the extra rule cost anything anywhere else?  All 200 rows, both models.

    For every row, mend `first_plan` with the recorded predicted goal under the shipped
    mender and under the counterfactual one, and compare the loop's own verdict.  A rule
    that is worth having must gain tasks and lose none.
    """
    changed, gained, lost = 0, [], []
    for path, table in (("4b", FOUR), ("8b", EIGHT)):
        for tid, row in table.items():
            task = TASKS[tid]
            plan = [(a, o) for a, o in row["first_plan"]]
            if not plan:
                continue
            goal = [tuple(g) for g in row["predicted_goal"]]
            seed = WorldGraph.from_scene_graph(seed_graph(task))
            a1, _ = R.repair(seed, plan, goal)
            a2, _ = repair2(seed, plan, goal)
            if a1 == a2:
                continue
            changed += 1
            o1 = GraphMachine(seed.copy()).run(a1, goal)
            o2 = GraphMachine(seed.copy()).run(a2, goal)
            if accepted(o2) and not accepted(o1):
                drive = run_plan(task, seed_graph(task),
                                 [{"action": a, "object": o} for a, o in a2])
                gained.append((path, tid, row["checked"], drive["ok"], drive["why"]))
            elif accepted(o1) and not accepted(o2):
                lost.append((path, tid))
    print(f"\nSWEEP over all 200 rows, mending attempt 1 with the predicted goal")
    print(f"  plans the extra rule changed at all: {changed}")
    print(f"  attempt-1 plans it turns from refused into ACCEPTED: {len(gained)}")
    for row in gained:
        print(f"     {row[0]} {row[1]:24s} recorded checked={row[2]:9s} "
              f"driven ok={row[3]} why={row[4]!r}")
    print(f"  plans it turns from accepted into refused: {len(lost)}  {lost}")


def census():
    """How often each model writes the mistakes that end in an empty hand."""
    from collections import Counter
    print("\nCENSUS over attempt-1 plans, both models")
    for name, table in (("4B", FOUR), ("8B", EIGHT)):
        tally, surplus, selfplace = Counter(), 0, 0
        for tid, row in table.items():
            plan = [(a, o) for a, o in row["first_plan"]]
            if not plan:
                continue
            goal = [tuple(g) for g in row["predicted_goal"]]
            seed = WorldGraph.from_scene_graph(seed_graph(task_of(tid)))
            out = GraphMachine(seed.copy()).run(plan, goal)
            fault = out.steps[out.failed_at].fault[0] if out.failed_at is not None else "-"
            tally[fault] += 1
            g = sum(1 for a, _ in plan if a == "GRASP")
            pl = sum(1 for a, _ in plan if a in R.PLACES)
            surplus += pl > g
            held = None
            for a, o in plan:
                if a == "GRASP":
                    held = o
                elif a == "RELEASE":
                    held = None
                elif a in R.PLACES:
                    selfplace += held is not None and R._same(o, held)
                    held = None
        print(f"  {name}: first fault on the model's own attempt-1 plan: "
              f"{dict(tally.most_common())}")
        print(f"      plans with more placements than grasps: {surplus}"
              f"   placements onto the very object being carried: {selfplace}")


def task_of(tid):
    return TASKS[tid]


if __name__ == "__main__":
    main()
    sweep()
    census()
