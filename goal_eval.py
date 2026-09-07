#!/usr/bin/env python3
"""How well can a model read the finished state out of a task description?

The checker can test whether a plan reaches a goal, but nothing upstream produced one: the
benchmark's goal conditions are ground truth kept for scoring, and handing them to the
planner would be telling it the answer. So the loop only ever asked whether every action
applied - and a plan that ran flawlessly and did the wrong thing was accepted without
complaint, which was 16 to 21 of the 100 tasks.

Asking the *planner* to state its own goal was tried and made things worse: measured, its
goal was exactly right only half the time, and validating against a wrong goal accepted
plans that then failed the real one. This measures the alternative - a small model trained
on the job, the way the object extractor was.

    python goal_eval.py --adapters models/goal-1.7b-2000
"""

import argparse
import json

from object_names import same


def as_set(goal):
    return {(k, str(a), str(b).lower()) for k, a, b in (tuple(g) for g in goal)}


def matches(want, got):
    """Is this wanted condition present, allowing the usual naming slack?

    The slack has to apply to BOTH names, not just the first. Comparing the target as a
    literal string marked `on_top(paper, trash_can)` wrong against
    `on_top(paper, public_trash_can)` - the same condition, spelled the way the sentence
    spells it - and did that on 11 of 100 tasks, which is most of what looked like a
    reasoning failure. `true` and `false` still compare exactly; they are values, not names.
    """
    k, a, b = want
    return any(k == k2 and same(a, a2)
               and (b == b2 if b in ("true", "false") else same(b, b2))
               for k2, a2, b2 in got)


def measure(tasks, answers):
    exact = hit = want_n = got_n = 0
    unsat = 0
    for task, goal in zip(tasks, answers):
        want, got = as_set(task["goal"]), as_set(goal)
        found = [w for w in want if matches(w, got)]
        hit += len(found)
        want_n += len(want)
        got_n += len(got)
        # A condition the machine cannot satisfy is worse than none: it rejects every plan.
        unsat += sum(1 for k, a, b in got
                     if k in ("open", "toggled") and not any(same(a, a2) for _, a2, _ in want)
                     and b == "false")
        exact += len(found) == len(want) == len(got)
    n = max(len(tasks), 1)
    return {"exact": exact / n, "recall": hit / max(want_n, 1),
            "precision": hit / max(got_n, 1), "unsatisfiable": unsat}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="data/tasks.json")
    parser.add_argument("--adapters", nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--json", default="data/goal-extraction.json")
    args = parser.parse_args()

    from finetune_extraction import GOAL_INSTRUCTION
    from planner import get_generator, parse_goal

    tasks = json.load(open(args.tasks))[:args.limit]
    print(f"{len(tasks)} tasks\n")
    print(f"  {'model':22s} {'exact':>6s} {'recall':>7s} {'precis':>7s} {'unsat':>6s}")
    out = {}
    for path in args.adapters:
        generate = get_generator(adapter=path)
        answers = [parse_goal("GOAL:\n" + generate(GOAL_INSTRUCTION.format(task=t["task"]),
                                                   200))
                   for t in tasks]
        got = measure(tasks, answers)
        out[path] = {"metrics": got, "answers": answers}
        print(f"  {path.split('/')[-1]:22s} {got['exact']:>5.0%} {got['recall']:>7.0%} "
              f"{got['precision']:>7.0%} {got['unsatisfiable']:>6d}")
    with open(args.json, "w") as f:
        json.dump({k: v for k, v in out.items()}, f, indent=1)
    print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
