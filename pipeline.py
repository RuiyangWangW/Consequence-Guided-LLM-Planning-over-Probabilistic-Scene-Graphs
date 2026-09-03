"""End-to-end: task description + scene -> validated BEHAVIOR-1K action sequence.

Four stages:

  1. task_objects  ask the LLM which objects the task needs
  2. room_graph    load the scene's room topology from the floor plans
  3. scene_graph   place uncertain objects with the RSN prior (P(obj | room type)), and
                   attach dependent objects to their container with a certain edge
  4. planner       LLM proposes primitives; a symbolic validator rejects infeasible ones

Stage 3 is what the RSN is for: the room graph knows the layout but not the contents,
and the robot cannot see the whole house at once. The RSN supplies a plausibility prior
over where things are, so the planner can reason about a furnished house instead of an
empty floor plan.

    python pipeline.py --scene Rs_int --task "put the mug in the fridge"
"""

import argparse
import json

from planner import format_plan, generate
from scene_graph import DEFAULT_MODEL, DEFAULT_THRESHOLD, format_for_llm, populate


def extract_objects(task, model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Objects the task needs, split into uncertain and dependent.

    Reuses the planner's generator, so extraction costs no extra model load.
    """
    from planner import get_generator
    from task_objects import extract

    return extract(task, model_name, generator=get_generator(model_name))


def run(task, scene, model_name="Qwen/Qwen2.5-7B-Instruct",
        rsn_model=DEFAULT_MODEL, threshold=DEFAULT_THRESHOLD, objects=None,
        strict=True, verbose=True):
    """Run the full pipeline and return every intermediate stage."""
    # --- stage 1: what does the task need? ---
    dependent, stated = [], {}
    if objects is None:
        found = extract_objects(task, model_name)
        objects, dependent = found["uncertain"], found["dependent"]
        # The task named a room for some of these ("the office bottom cabinet"). That is
        # stated, so it leads the search order - with the RSN's ranking behind it, in case
        # the statement is wrong.
        stated = found["stated"]
    if verbose:
        print(f"task:    {task}")
        print(f"scene:   {scene}")
        print(f"uncertain: {', '.join(objects) if objects else '(none)'}")
        if dependent:
            print("dependent: " + ", ".join(
                f"{d['object']} {d['relation']} {d['target']}" for d in dependent))
        print()

    # --- stages 2+3: topology from the floor plan, contents from the RSN ---
    graph = populate(scene, objects, dependent, stated=stated, model_path=rsn_model,
                     threshold=threshold)
    if verbose:
        print(format_for_llm(graph))
        print()

    # --- stage 4: plan, then validate against the primitives' real semantics ---
    if verbose:
        print("planning:")
    result = generate(task, graph, model_name, strict=strict, verbose=verbose)

    return {"task": task, "scene": scene, "objects": objects,
            "dependent": dependent, "graph": graph, "plan": result}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", required=True, help="natural-language task description")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--rsn-model", default=DEFAULT_MODEL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="optional: drop objects whose best room scores below this. "
                             "Off by default - every object is placed in its most "
                             "likely room and the probability is reported instead")
    parser.add_argument("--objects", nargs="+", help="skip extraction, use these objects")
    parser.add_argument("--lenient", action="store_true",
                        help="treat missing navigation as a warning, not an error")
    parser.add_argument("--json", help="write the full result to this path")
    args = parser.parse_args()

    out = run(args.task, args.scene, args.model, args.rsn_model,
              args.threshold, args.objects, strict=not args.lenient)

    print("\nplan:")
    print(format_plan(out["plan"]))

    status = "EXECUTABLE" if not out["plan"]["errors"] else "REJECTED"
    print(f"\n{status}: {len(out['plan']['steps'])} actions, "
          f"{len(out['plan']['errors'])} errors, {len(out['plan']['warnings'])} warnings")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
