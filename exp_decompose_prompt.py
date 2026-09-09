#!/usr/bin/env python3
"""Which decomposition prompt recovers the benchmark's own errand boundaries?

`gavel.decompose` is one model call, and on the multi-task benchmark it over-splits: a three
clause errand - "take the casserole out of the fridge, warm it in the microwave, and leave it on
the breakfast table" - comes back as three numbered lines. The clauses that begin with a pronoun
lose their referent, the goal adapter then answers about an object called `it` (or invents one
from the verb, `cooked(warm_it, True)`), and the errand cannot be planned at all. Measured over
the 50-instruction sample: decomposition matched the benchmark on 34 and over-split 16, and
GAVEL scored 82% on the first group against 31% on the second.

The prompt is why. Its constraints are asymmetric - it forbids merging two errands and says
nothing about splitting one - and its last line ("every object named must appear in exactly one
errand") actively rewards splitting, because cutting at every clause is the tidiest way to give
each object a line of its own.

This scores candidate prompts against `task["subgoals"]`, which is where the instruction was
assembled from and therefore the ground truth for what its errands are.

    python exp_decompose_prompt.py --variant v1 --out logs/decomp-v1.json

Scored two ways. `exact` is the strict one: the same errands, in the same order, word for word
after lowercasing and stripping punctuation - the decomposition a perfect run produces, since
the prompt tells the model to copy the instruction's wording. `boundaries` is the one that
matters for the pipeline: the same NUMBER of errands with the same movable objects in each, so
a decomposition that says "put the mail on the armchair" where the benchmark says "put the mail
from the desk on the armchair" still counts - the planner can do that errand.
"""

import argparse
import json
import re
import sys
import time

import gavel
from planner import get_generator

#: The prompt in `gavel.py` today. Kept here so a run can measure the baseline unchanged.
V0 = gavel.DECOMPOSE

#: Make the constraint symmetric, and say what a pronoun clause is.
V1 = """Split the instruction into the separate errands it asks for.

Write one errand per line, numbered. Copy the wording of the instruction - do not paraphrase
and do not add steps.

An errand is one complete job and may take several clauses. Do not merge two errands into one
line, and do not split one errand across several lines. A clause beginning with a pronoun -
"heat it in the oven", "put it on the table", "and run it" - is the same errand continuing: it
belongs on the same line as the clause that says what "it" is.

Instruction: {task}

Errands:"""

#: The same rule, shown rather than stated.
V2 = """Split the instruction into the separate errands it asks for.

Write one errand per line, numbered. Copy the wording of the instruction - do not paraphrase
and do not add steps. Keep an errand whole even when it takes several clauses: a clause that
refers back to something already named continues the errand rather than starting a new one.

Instruction: put the book on the shelf, take the pie from the counter, heat it in the oven, and
put it on the table, and switch on the lamp

Errands:
1. put the book on the shelf
2. take the pie from the counter, heat it in the oven, and put it on the table
3. switch on the lamp

Instruction: {task}

Errands:"""

#: Define the unit by what it acts on, which is what makes errands independent.
V3 = """Split the instruction into the separate errands it asks for.

Write one errand per line, numbered. Copy the wording of the instruction - do not paraphrase
and do not add steps.

Each errand deals with ONE movable thing from start to finish. Everything the instruction says
about that thing - where to take it from, what to do to it on the way, where to leave it -
belongs on one line, however many clauses that takes. Start a new line only when the
instruction turns to a different thing.

Instruction: {task}

Errands:"""

VARIANTS = {"v0": V0, "v1": V1, "v2": V2, "v3": V3}

MOVABLE = re.compile(r"[a-z_]+")


def normalise(text):
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def movables(task, errand):
    """The movable objects this errand sentence names, by the task's own spawn list."""
    names = {s["target"] for s in task.get("spawn", ())} | {s["name"] for s in task.get("spawn", ())}
    words = normalise(errand).replace(" ", "_")
    return {n for n in names if n.replace("_", "") in words.replace("_", "")}


def split_with(prompt, text, generator, max_new_tokens=400):
    """`gavel.decompose`, with the prompt swapped. Same parsing, same fallback, same casing."""
    reply = generator(prompt.format(task=text), max_new_tokens)
    parts = []
    for line in reply.splitlines():
        line = line.strip()
        match = re.match(r"^\(?(\d+)[.):]\s*(.+)$", line)
        if match and match.group(2).strip():
            parts.append(match.group(2).strip().rstrip(",;."))
    if len(parts) < 2:
        parts = [p.strip().rstrip(",;.") for p in re.split(
            r",\s+and\s+|,\s*(?=(?:put|take|get|bring|move|place|turn|switch|wash|cook|dry)\b)"
            r"|\s+and\s+then\s+", text) if p.strip()]
    return [p.lower() for p in parts]


def score(task, got):
    want = [s["task"] for s in task["subgoals"]]
    exact = [normalise(g) for g in got] == [normalise(w) for w in want]
    same_count = len(got) == len(want)
    # The pipeline-relevant test: same number of errands, each naming the same movables.
    boundaries = same_count and all(
        movables(task, g) == movables(task, w) for g, w in zip(got, want))
    return {"exact": exact, "boundaries": boundaries, "count": len(got),
            "want_count": len(want),
            "over": len(got) > len(want), "under": len(got) < len(want),
            "got": got, "want": want}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--tasks", default="data/multitask.json")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tasks = json.load(open(args.tasks))[::args.stride]
    if args.limit:
        tasks = tasks[:args.limit]
    generator = get_generator()
    prompt = VARIANTS[args.variant]

    rows, at = [], time.perf_counter()
    for i, task in enumerate(tasks, 1):
        got = split_with(prompt, task["task"], generator)
        row = score(task, got)
        row["id"] = task["id"]
        rows.append(row)
        if i % 10 == 0 or i == len(tasks):
            e = sum(1 for r in rows if r["exact"])
            b = sum(1 for r in rows if r["boundaries"])
            o = sum(1 for r in rows if r["over"])
            print(f"[{args.variant}] {i}/{len(tasks)}  exact {e}  boundaries {b}  over-split {o}",
                  flush=True)
    json.dump({"variant": args.variant, "seconds": round(time.perf_counter() - at, 1),
               "rows": rows}, open(args.out, "w"), indent=1)
    e = sum(1 for r in rows if r["exact"])
    b = sum(1 for r in rows if r["boundaries"])
    print(f"\n{args.variant}: exact {e}/{len(rows)}   boundaries {b}/{len(rows)}   "
          f"over-split {sum(1 for r in rows if r['over'])}   "
          f"under-split {sum(1 for r in rows if r['under'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
