#!/usr/bin/env python3
"""One rule for deciding whether two object names mean the same thing.

The pipeline names objects out of free text and the world names them by BEHAVIOR category,
and the two agree on the thing while disagreeing on the string: `tshirt` and `t_shirt`,
`bath_towels` and `bath_towel`, `trash_can` and `public_trash_can`. Every consumer needs
that judgement, and for a while each had its own version of it - the scorer matched loosely,
the simulator matched exactly, and a plan the scorer counted correct was one the robot could
not locate. Seventeen search failures in one run were that disagreement, not the robot.

So it lives here once. `match(name, candidates)` picks which of several names is meant;
`same(a, b)` is the yes/no question defined in terms of it, so the two cannot drift apart.

**Specificity decides.** A plan names a thing loosely and the house names it precisely, so a
candidate whose words are a *superset* wins before one whose words are a subset. That is
what stops `trash_can` resolving to `can` in a task that names both - "bring the wrapping
paper, the carton and the can to the trash can".
"""


def tokens(name):
    return set(name.replace("-", "_").lower().split("_"))


def _squash(name):
    return name.replace("-", "").replace("_", "").lower()


def candidates(name, pool):
    """Every candidate that could be `name`, best first. Empty if none could.

    The tiers, in order:

        1. exact
        2. the same letters, ignoring punctuation      tshirt   -> t_shirt
        3. more specific than what was asked for       trash_can -> public_trash_can
        4. the same, after a plural is singularised    bath_towels -> bath_towel
        5. less specific, only when nothing finer fits childs_bed -> bed

    Tier 3 before tier 5 is the load-bearing order. Ties within a tier are returned
    together rather than resolved here - `cabinet` matches both `top_cabinet` and
    `bottom_cabinet`, and which is meant is a question about the *world*, not the words.
    The caller knows which room the robot believes it is in; this does not.
    """
    pool = list(pool)
    if name in pool:
        return [name]
    flat = _squash(name)
    same_letters = [c for c in pool if _squash(c) == flat]
    if same_letters:
        return sorted(same_letters, key=lambda c: (len(c), c))

    want = tokens(name)
    singular = {t[:-1] if t.endswith("s") and len(t) > 3 else t for t in want}
    for probe in (want, singular):
        finer = [c for c in pool if probe <= tokens(c)]
        if finer:
            return sorted(finer, key=lambda c: (len(c), c))
    coarser = [c for c in pool if tokens(c) <= want]
    return sorted(coarser, key=lambda c: (-len(c), c))


def match(name, pool, prefer=None):
    """The one candidate `name` means, or None.

    `prefer` picks between equally good candidates - pass a predicate that knows something
    this module cannot, such as which of three cabinets is in the room the robot believes
    it is standing in. Without it the shortest name wins, which is a coin flip dressed up
    as a rule.
    """
    ranked = candidates(name, pool)
    if not ranked:
        return None
    if prefer is not None:
        for option in ranked:
            if prefer(option):
                return option
    return ranked[0]


def same(a, b):
    """Could these two names refer to the same object?

    Symmetric, and defined in terms of `candidates` so that it cannot disagree with
    `match` about a pair it is shown.
    """
    return bool(candidates(a, [b]) or candidates(b, [a]))


# --------------------------------------------------------------- the canonical vocabulary

# Two BEHAVIOR categories nobody says out loud. A person says "the dryer" and "the trash
# can"; the dataset says `clothes_dryer` and `public_trash_can`. Writing the dataset's words
# into the instruction would be stilted, and letting the difference travel down the pipeline
# is what had five separate components each inventing their own way to paper over it.
#
# So it is resolved once, on the way in, and everything after extraction - the belief graph,
# the goal, the plan, the checker, the scorer - compares names with `==`.
#
# This is a fixed table on purpose, not a lookup against the object catalogue. The catalogue
# in `data/vocab.json` lists what the *scenes* contain, and tasks inject objects the scenes
# do not have - `t_shirt` and `bath_towel` are absent from it - so canonicalising against it
# would silently rewrite half the vocabulary and leave the other half alone.
ALIASES = {
    "trash_can": "public_trash_can",
    "dryer": "clothes_dryer",
}


def canonical(name):
    """The dataset's own word for `name`, unchanged when there is nothing to say.

    Nothing ambiguous is decided here. `cabinet` is a `top_cabinet` exactly as much as a
    `bottom_cabinet`, and choosing needs a world this function does not have - so it is
    left alone, the machine refuses a plan that says it, and the loop asks the model which
    it meant. A coin flip would be worse, because a coin flip is silent.
    """
    return ALIASES.get(name, name)
