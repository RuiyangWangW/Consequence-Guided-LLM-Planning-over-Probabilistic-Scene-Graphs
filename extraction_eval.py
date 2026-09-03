#!/usr/bin/env python3
"""Measure object-extraction accuracy on its own, and compare ways of improving it.

Extraction is one LLM call, so it can be measured in seconds where the whole pipeline
takes an hour - and it is worth measuring on its own, because after the safety feedback
landed it is **the largest remaining source of failure**: 43% of what the 4B still gets
wrong and 61% of the 8B's.

The measured failure is specific. Asked to list what a task needs, the model returns the
places and forgets the things: of 70 dropped items across the 8B's failures, 55 were the
objects being moved and 15 were furniture, and it volunteered a *room* as an object 16
times despite being told not to.

Variants are compared on a generated dev set rather than on the benchmark, so that tuning
a prompt is not tuning on the test set.

    python extraction_eval.py --variants baseline normalized restructured
    python extraction_eval.py --on data/tasks.json --variants baseline best
"""

import argparse
import json
import os
import re

from evaluate import same_object

# Names that are rooms rather than objects. The prompt says to skip them; measured, it
# returns one anyway about once every six tasks, and a room in the object list becomes a
# room the RSN is asked to place inside another room.
ROOM = re.compile(r"^(.*_)?(room|kitchen|bathroom|bedroom|corridor|closet|pantry|playroom|"
                  r"entryway|garage|office|hallway|hall|staircase|lobby|attic|basement|"
                  r"utility|garden|porch|balcony)(_\d+)?$")


def is_room(name):
    return bool(ROOM.match(name.lower()))


def objects_of(e):
    """Every object named, across all three classes."""
    return (set(e.get("uncertain", ())) | set(e.get("stated", {}))
            | {d["object"] for d in e.get("dependent", ())})


def measure(rows, extracted):
    """Compare answers against ground truth, per class.

    Four things are scored separately, because they fail separately and the fix differs:

        objects    did it name the right things at all
        class      did it put each in the right class - a support beats a room beats nothing
        relation   are the object->object triples right
        stated     are the object->room facts right

    Recall and precision are both reported for the relations and rooms because the two
    failure directions have different costs. Omitting a stated location costs a search;
    inventing one sends the robot to the wrong place and the plan dies there.
    """
    totals = dict(exact=0, obj_hit=0, obj_want=0, obj_got=0, leaks=0,
                  cls_hit=0, cls_want=0,
                  rel_hit=0, rel_want=0, rel_got=0,
                  room_hit=0, room_want=0, room_got=0)
    for row, got in zip(rows, extracted):
        want = row["extraction"]
        w_obj, g_obj = objects_of(want), objects_of(got)
        totals["obj_want"] += len(w_obj)
        totals["obj_got"] += len(g_obj)
        miss = [w for w in w_obj if not any(same_object(w, g) for g in g_obj)]
        totals["obj_hit"] += len(w_obj) - len(miss)
        totals["leaks"] += sum(1 for g in g_obj
                               if is_room(g) and not any(same_object(g, w) for w in w_obj))

        # Which class did each wanted object land in?
        def class_of(e, name):
            for d in e.get("dependent", ()):
                if same_object(d["object"], name):
                    return "dependent"
            for o in e.get("stated", {}):
                if same_object(o, name):
                    return "stated"
            for o in e.get("uncertain", ()):
                if same_object(o, name):
                    return "uncertain"
            return None

        for name in w_obj:
            totals["cls_want"] += 1
            if class_of(want, name) == class_of(got, name):
                totals["cls_hit"] += 1

        heard = {d["object"]: (d["relation"], d["target"]) for d in got.get("dependent", ())}
        totals["rel_want"] += len(want["dependent"])
        totals["rel_got"] += len(got.get("dependent", ()))
        rel_ok = True
        for d in want["dependent"]:
            m = next((h for h in heard if same_object(d["object"], h)), None)
            if m and heard[m][0] == d["relation"] and same_object(heard[m][1], d["target"]):
                totals["rel_hit"] += 1
            else:
                rel_ok = False

        w_rooms, g_rooms = want.get("stated", {}), got.get("stated", {})
        totals["room_want"] += len(w_rooms)
        totals["room_got"] += len(g_rooms)
        for obj, room in w_rooms.items():
            m = next((g for g in g_rooms if same_object(obj, g)), None)
            if m and g_rooms[m] == room:
                totals["room_hit"] += 1

        totals["exact"] += 1 if not miss and rel_ok else 0

    def over(hit, of):
        return totals[hit] / max(totals[of], 1)

    return {"exact": totals["exact"] / max(len(rows), 1),
            "recall": over("obj_hit", "obj_want"),
            "precision": over("obj_hit", "obj_got"),
            "class_acc": over("cls_hit", "cls_want"),
            "rel_recall": over("rel_hit", "rel_want"),
            "rel_precision": over("rel_hit", "rel_got"),
            "room_recall": over("room_hit", "room_want"),
            "room_precision": over("room_hit", "room_got"),
            "room_leaks": totals["leaks"]}


# ---------------------------------------------------------------------- the variants

# One LLM call per (prompt, task, temperature), shared across variants. Two variants that
# differ only in a post-filter would otherwise pay twice for the same greedy decode.
_CACHE = {}


def _ask(text, generator, prompt, temperature, tokens, normalize):
    from task_objects import extract

    key = (id(prompt), id(generator), text, temperature, normalize)
    if key not in _CACHE:
        _CACHE[key] = extract(text, generator=generator, prompt=prompt,
                              temperature=temperature, max_new_tokens=tokens,
                              normalize=normalize)
    return _CACHE[key]


def variant(prompt=None, normalize=True, samples=(0.0,), tokens=160, generator=None):
    """Build one extraction strategy out of the three things worth varying.

    `prompt`     which wording to use (None = the one in task_objects)
    `normalize`  split room qualifiers off the names afterwards, per `strip_room`.
                 Deterministic, and it cannot cost recall: it only ever shortens a name
                 towards a real category.
    `samples`    temperatures to ask at, unioned. Recall failures that are stochastic get
                 fixed by asking again; ones that are systematic do not, which is what
                 makes this a useful control on the prompt changes.
    `generator`  a model of this variant's own, for comparing a fine-tuned extractor
                 against the prompted one in a single run. Defaults to the shared model.
    """
    own = generator

    def run(text, generator):
        uncertain, dependent, stated = [], [], {}
        for temperature in samples:
            got = _ask(text, own or generator, prompt, temperature, tokens, normalize)
            stated.update(got.get("stated", {}))
            for name in got["uncertain"]:
                if not any(same_object(name, s) for s in uncertain):
                    uncertain.append(name)
            for d in got["dependent"]:
                if not any(same_object(d["object"], e["object"]) for e in dependent):
                    dependent.append(d)
        # The three classes stay disjoint after a union across samples, or the scorer sees
        # an object in two of them and the downstream resolver has to guess which.
        placed = {d["object"] for d in dependent}
        stated = {o: r for o, r in stated.items() if o not in placed}
        uncertain = [n for n in uncertain if n not in placed and n not in stated]
        return {"uncertain": uncertain, "stated": stated, "dependent": dependent}

    return run


def _prompts():
    from extraction_prompts import BASELINE, LONG_EXAMPLES, RESTRUCTURED

    return BASELINE, LONG_EXAMPLES, RESTRUCTURED


def build_variants(adapters=()):
    """Named strategies. Each prompt change is measured alone as well as combined, so a
    gain can be attributed to one of them rather than to the pair."""
    base, long_examples, restructured = _prompts()
    # `variant()` with no prompt uses whatever `task_objects` ships, which is now the
    # winner - so "current" tracks the shipped pipeline and needs no copy kept in step.
    built = {
        "baseline": variant(base, normalize=False),
        "normalized": variant(base),
        "long_examples": variant(long_examples),
        "restructured": variant(restructured, tokens=280),
        "current": variant(tokens=200),
        "union": variant(base, samples=(0.0, 0.7, 1.0)),
        "union_current": variant(samples=(0.0, 0.7, 1.0), tokens=200),
        "union_restructured": variant(restructured, samples=(0.0, 0.7, 1.0), tokens=280),
    }
    # A fine-tuned extractor gets the short instruction it was trained on, not the twelve
    # worked examples - those exist to demonstrate a format it has already learned.
    from finetune_extraction import INSTRUCTION
    from planner import get_generator

    for path in adapters:
        # `strip_room` stays on as a *backstop*, not as the mechanism. A trained model
        # states the room itself; measured, leaving the recovery on changes nothing except
        # room recall, 95% -> 100%, by catching the few names it still leaves fused.
        built[os.path.basename(path)] = variant(
            INSTRUCTION + "\n", tokens=200,
            generator=get_generator(adapter=path))
    return built


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--on", default="data/extraction-dev.json")
    parser.add_argument("--variants", nargs="+", default=["baseline"])
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--adapters", nargs="*", default=[],
                        help="LoRA directories to measure alongside the prompted model")
    parser.add_argument("--json", default="data/extraction-results.json")
    args = parser.parse_args()

    from planner import get_generator

    rows = json.load(open(args.on))
    if args.limit:
        rows = rows[:args.limit]
    variants = build_variants(args.adapters)
    # Every adapter carries its own model, so the shared one is loaded only if a variant
    # that needs it was actually asked for - two models on one card is how a run OOMs.
    prompted = [n for n in args.variants if n not in {os.path.basename(a)
                                                      for a in args.adapters}]
    generator = get_generator(args.model) if prompted else None

    which = f"model {args.model}" if prompted else "adapters only"
    print(f"{len(rows)} instructions from {args.on}, {which}\n")
    print(f"  {'variant':18s} {'exact':>6s} | {'obj rec':>7s} {'obj pre':>7s} {'class':>6s} | "
          f"{'rel rec':>7s} {'rel pre':>7s} | {'rm rec':>6s} {'rm pre':>6s}")
    results = {}
    for name in args.variants:
        answers = [variants[name](r["task"], generator) for r in rows]
        got = measure(rows, answers)
        results[name] = {"metrics": got, "answers": answers}
        print(f"  {name:18s} {got['exact']:>5.0%} | {got['recall']:>7.0%} "
              f"{got['precision']:>7.0%} {got['class_acc']:>6.0%} | "
              f"{got['rel_recall']:>7.0%} {got['rel_precision']:>7.0%} | "
              f"{got['room_recall']:>6.0%} {got['room_precision']:>6.0%}")

    with open(args.json, "w") as f:
        json.dump({"model": args.model, "on": args.on,
                   "results": {k: v["metrics"] for k, v in results.items()},
                   "answers": {k: v["answers"] for k, v in results.items()}}, f, indent=1)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
    raise SystemExit(main())
