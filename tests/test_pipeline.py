"""Tests for plan parsing and validation - no simulator required.

The validator is what makes the pipeline's output trustworthy, so its rejections are
tested directly: each case asserts that a specific infeasible plan is caught, and that a
correct plan passes clean.

This is the fast half of the test suite. It runs in under a second against a hand-built
scene graph, so it can be run on every edit. `test_primitives.py` is the slow half - it
launches Isaac Sim and checks that each primitive actually moves the world.

    python test_pipeline.py
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


from planner import parse_plan, validate

# Minimal two-room scene: a fridge and a mug in the kitchen, a towel one room away.
GRAPH = {
    "scene": "test",
    "rooms": {
        "kitchen_0": {"room_type": "kitchen", "pixels": 100, "centroid": [0, 0]},
        "bathroom_0": {"room_type": "bathroom", "pixels": 50, "centroid": [1, 1]},
        "bedroom_0": {"room_type": "bedroom", "pixels": 50, "centroid": [9, 9]},
    },
    "edges": [["bathroom_0", "kitchen_0"]],
    "objects": {
        "fridge": {"room": "kitchen_0", "room_type": "kitchen", "probability": 0.83},
        "mug": {"room": "kitchen_0", "room_type": "kitchen", "probability": 0.58},
        "towel": {"room": "bathroom_0", "room_type": "bathroom", "probability": 0.71},
    },
    "unplaced": {"chainsaw": {"reason": "below threshold", "probability": 0.01}},
}

CASES = []


def case(name):
    def wrap(fn):
        CASES.append((name, fn))
        return fn
    return wrap


def check(plan, expect_errors, strict=True):
    errors, warnings = validate(parse_plan(plan), GRAPH, strict)
    return errors, warnings


@case("a correct plan validates clean")
def _():
    errors, _ = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        NAVIGATE_TO(fridge)
        OPEN(fridge)
        PLACE_INSIDE(fridge)
        CLOSE(fridge)
    """, False)
    assert not errors, errors


@case("placing with an empty hand is rejected")
def _():
    errors, _ = check("NAVIGATE_TO(fridge)\nPLACE_INSIDE(fridge)", True)
    assert any("nothing in hand" in e for e in errors), errors


@case("grasping twice without releasing is rejected")
def _():
    errors, _ = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        GRASP(fridge)
    """, True)
    assert any("already holding" in e for e in errors), errors


@case("acting on an object in another room is rejected")
def _():
    errors, _ = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        PLACE_ON_TOP(towel)
    """, True)
    assert any("navigate there first" in e for e in errors), errors


@case("objects the RSN says are absent are rejected")
def _():
    errors, _ = check("NAVIGATE_TO(chainsaw)\nGRASP(chainsaw)", True)
    assert any("NOT present" in e for e in errors), errors


@case("objects outside the scene graph are rejected")
def _():
    errors, _ = check("NAVIGATE_TO(unicorn)", True)
    assert any("not in the scene graph" in e for e in errors), errors


@case("RELEASE takes no argument")
def _():
    errors, warnings = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        RELEASE(mug)
    """, False)
    assert not errors, errors
    assert any("takes no argument" in w for w in warnings), warnings


@case("a primitive missing its object argument is rejected")
def _():
    errors, _ = check("GRASP()", True)
    assert any("requires an object" in e for e in errors), errors


@case("placing an object onto itself is rejected")
def _():
    errors, _ = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        PLACE_ON_TOP(mug)
    """, True)
    assert any("onto itself" in e for e in errors), errors


@case("unreachable rooms warn rather than fail")
def _():
    errors, warnings = check("NAVIGATE_TO(mug)\nNAVIGATE_TO(bedroom_0)", False)
    assert not errors, errors
    assert any("not directly connected" in w for w in warnings), warnings


@case("toggle validates cleanly now that it is implemented")
def _():
    # primitive_patches.apply() installs a working _toggle, so TOGGLE_ON is a normal
    # executable step rather than something the validator has to warn about.
    errors, warnings = check("NAVIGATE_TO(fridge)\nTOGGLE_ON(fridge)", False)
    assert not errors, errors
    assert not any("cannot execute" in w for w in warnings), warnings


@case("placing inside a closed container warns")
def _():
    _, warnings = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        NAVIGATE_TO(fridge)
        PLACE_INSIDE(fridge)
    """, False)
    assert any("usually opened" in w for w in warnings), warnings


@case("an unfinished plan is rejected, not merely flagged")
def _():
    # A plan that ends mid-carry has not done the task; reporting it EXECUTABLE would
    # be misleading, so this is an error the retry loop can act on.
    errors, _ = check("NAVIGATE_TO(mug)\nGRASP(mug)", True)
    assert any("still holding" in e for e in errors), errors


@case("lenient mode downgrades navigation errors to warnings")
def _():
    errors, warnings = check("""
        NAVIGATE_TO(mug)
        GRASP(mug)
        PLACE_ON_TOP(towel)
    """, False, strict=False)
    assert not errors, errors
    assert any("implicit navigation" in w for w in warnings), warnings


@case("a state change reached without navigating is flagged")
def _():
    # OPEN and the toggles no longer navigate - the plan is expected to include its own
    # NAVIGATE_TO first (see primitive_patches). In the simulator the fridge would open
    # from across the room, so nothing there catches a plan that skips the approach; the
    # validator is the only thing that can.
    errors, warnings = check("OPEN(fridge)", True)
    assert errors or warnings, "expected the missing approach to be reported"


@case("parser survives numbering, bullets and prose")
def _():
    steps = parse_plan("""
        Here is the plan:
        1. NAVIGATE_TO(mug)
        - GRASP(mug)
        ```
        2) RELEASE()
        ```
        That completes the task.
    """)
    assert [s["action"] for s in steps] == ["NAVIGATE_TO", "GRASP", "RELEASE"], steps
    assert steps[2]["object"] is None, steps


@case("grasping a fixed appliance is rejected")
def _():
    # The LLM reliably writes GRASP(fridge) before OPEN(fridge). The robot cannot pick
    # up a fridge, so this is caught here rather than failing in the simulator.
    errors, _ = check("NAVIGATE_TO(fridge)\nGRASP(fridge)\nOPEN(fridge)", True)
    assert any("cannot be picked up" in e for e in errors), errors


@case("opening an appliance without grasping it validates")
def _():
    errors, _ = check("NAVIGATE_TO(fridge)\nOPEN(fridge)\nCLOSE(fridge)", False)
    assert not errors, errors


@case("parser repairs common LLM malformations")
def _():
    # Real Qwen output: it fixed the plan logic on retry but wrote the call wrong.
    # Discarding these steps makes a sound plan look incomplete.
    steps = parse_plan("""
        NAVIGATE_TO(bathroom_0)
        GRASP(towel)
        PLACE_ON_TOP.bed()
        TOGGLE_ON: fridge
        NAVIGATE_TO mug
    """)
    assert [(s["action"], s["object"]) for s in steps] == [
        ("NAVIGATE_TO", "bathroom_0"),
        ("GRASP", "towel"),
        ("PLACE_ON_TOP", "bed"),
        ("TOGGLE_ON", "fridge"),
        ("NAVIGATE_TO", "mug"),
    ], steps


@case("parser ignores invented primitives")
def _():
    steps = parse_plan("PICK_UP(mug)\nGRASP(mug)\nTELEPORT(kitchen_0)")
    assert [s["action"] for s in steps] == ["GRASP"], steps


def main():
    failed = 0
    for name, fn in CASES:
        try:
            fn()
            print(f"  pass  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}\n          {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
