#!/usr/bin/env python3
"""The one rule for deciding whether two object names mean the same thing.

Worth its own file because the rule used to exist twice - loosely in the scorer, exactly in
the simulator - and the disagreement cost 17 tasks in a single run: plans the scorer counted
correct named objects the robot then searched the whole house for and never recognised.

    python test_object_names.py
"""

import sys

from object_names import candidates, match, same

PASS, FAIL = [], []
CATS = {"t_shirt", "bath_towel", "hand_towel", "bottom_cabinet", "top_cabinet", "bed",
        "public_trash_can", "can", "mug", "soup", "bottle_of_soup", "countertop",
        "bathroom_countertop", "coffee_table", "breakfast_table"}


def case(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


def main():
    print("=== the tiers, in order ===")
    for name, want, why in [
        ("mug", "mug", "exact"),
        ("tshirt", "t_shirt", "same letters, different punctuation"),
        ("bath_towels", "bath_towel", "plural"),
        ("childs_bed", "bed", "less specific, last resort"),
        ("teapot", None, "nothing matches, and it says so"),
    ]:
        got = match(name, CATS)
        case(f"{name!r} -> {want!r} ({why})", got == want, f"got {got!r}")

    print("\n=== specificity beats brevity ===")
    # The task that forced this: "bring the wrapping paper, the carton and the *can* to the
    # *trash can*". Shortest-match-wins sends the destination to the thing being carried.
    case("'trash_can' prefers 'public_trash_can' over 'can'",
         match("trash_can", CATS) == "public_trash_can", str(match("trash_can", CATS)))
    # Stronger than ordering: tier 3 succeeds, so the less-specific tier is never reached
    # and `can` is not a candidate at all.
    case("'can' is not even offered as a candidate",
         "can" not in candidates("trash_can", CATS), str(candidates("trash_can", CATS)))
    # But it would be, if nothing more specific existed - that is the last-resort tier.
    case("with no finer option, 'can' is the last resort",
         candidates("trash_can", {"can", "mug"}) == ["can"],
         str(candidates("trash_can", {"can", "mug"})))

    print("\n=== a tie is a question about the world, not the words ===")
    tied = candidates("cabinet", CATS)
    case("'cabinet' fits both cabinets", set(tied) == {"top_cabinet", "bottom_cabinet"},
         str(tied))
    case("and a caller that knows the room decides it",
         match("cabinet", CATS, prefer=lambda c: c == "bottom_cabinet") == "bottom_cabinet")

    print("\n=== things that must NOT match ===")
    for a, b in [("bottom_cabinet", "top_cabinet"), ("mug", "bed"),
                 ("coffee_table", "breakfast_table")]:
        case(f"{a!r} is not {b!r}", not same(a, b))

    print("\n=== and things that must ===")
    for a, b in [("soup", "bottle_of_soup"), ("t_shirt", "tshirt"),
                 ("countertop", "bathroom_countertop")]:
        case(f"{a!r} is {b!r}", same(a, b))

    case("`same` is symmetric", all(same(a, b) == same(b, a)
                                    for a in CATS for b in list(CATS)[:6]))

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
