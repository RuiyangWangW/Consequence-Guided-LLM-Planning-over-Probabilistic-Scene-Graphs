#!/usr/bin/env python3
"""Choose the order to do independent subplans in, by expected embodied cost.

The subtasks in one instruction share no object, so no ordering is forbidden and none changes
what another costs *through the world*. That makes the objective decompose, approximately:

    J(sigma) = sum_k internal(sigma_k)        constant, the same for every ordering
             + sum_k A[sigma_(k-1), sigma_k]  an asymmetric path through a pairwise matrix

so the matrix is built once with N^2 rollouts and every ordering is scored by adding N
numbers, rather than rolling the graph forward N! times. That is what makes re-optimising
at each subplan boundary cheap enough to do online.

**"Exactly" is too strong, and the reason is worth knowing.** Sharing no object rules out one
errand *changing the world* another acts on; it does not rule out one errand *telling the robot
something* about another's objects. Sweeping the kitchen for the mug localises the bowl, and
`pairwise` hands every rollout a fresh `Beliefs`, so the matrix cannot see that coupling. Fitting
the executed cost back onto the `head`/`A` form gives R^2 = 1.000 at three errands - where the
form is exactly determined and so proves nothing - but 0.93 at four and 0.89 at five. The
functional form is a good approximation and the ordering it selects is sound; it is not an
identity, and the earlier claim that it was is the kind of thing that stops people checking.

The matrix is **asymmetric**: doing the kitchen errand then the bedroom one leaves the
robot in the bedroom, and the reverse leaves it in the kitchen, so cost(i->j) != cost(j->i).
This is an asymmetric Hamiltonian *path*, not a symmetric tour, which is why Christofides
does not apply - it needs a symmetric metric obeying the triangle inequality, and builds a
minimum spanning tree and a perfect matching, neither defined on a directed graph.

At the sizes here - 2 to 5 subgoals - exhaustive enumeration is exact and instant, so no
approximation is used at all. Past about eight, Held-Karp would be the exact choice; it
handles asymmetry natively.
"""

import itertools


def pairwise(subplans, graph, beliefs_factory, distance, search, start=None):
    """`(A, head)` - the pairwise transition matrix and the cost of going first.

    `head[j]` is what subplan `j` costs run first, from the robot's actual room.
    `A[i][j]` is what `j` costs run from wherever `i` left the robot.

    Every rollout gets a fresh `Beliefs` from the factory, so one candidate ordering cannot
    leak a localized object into another. That is the assumption the decomposition rests
    on, and it holds only because the subtasks share no object.
    """
    from search_cost import rollout

    n = len(subplans)
    head, ends = [0.0] * n, [None] * n
    for j, plan in enumerate(subplans):
        head[j], ends[j], _ = rollout(plan, graph, beliefs_factory(), distance, search,
                                      start=start)
    A = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A[i][j], _, _ = rollout(subplans[j], graph, beliefs_factory(), distance,
                                    search, start=ends[i])
    return A, head


def enumerate_orders(n):
    """Every ordering. There is no precedence to respect - subtasks share no object."""
    return itertools.permutations(range(n))


def best_order(A, head, n):
    """The cheapest ordering, exactly, by trying all of them."""
    # The identity, so an instance every ordering of which is infinite still gets an answer.
    # That happens when a subgoal's object sits behind a wall A* cannot cross: the ordering
    # is then genuinely arbitrary, and returning nothing would strand the executor.
    best, best_cost = tuple(range(n)), float("inf")
    for order in enumerate_orders(n):
        total = head[order[0]]
        for a, b in zip(order, order[1:]):
            total += A[a][b]
        if total < best_cost:
            best, best_cost = order, total
    return best, best_cost


def ranked(A, head, n):
    """Every ordering, cheapest first. Used when the cheapest one has to pass a test.

    Cost is not the only thing that decides an order. The subtasks are meant to share no
    object, so any permutation should be applicable - but the subplans are written by a
    model, and a model that opens a cabinet in one errand and reaches into it in another has
    made them dependent whatever the task said. The executor walks this list and takes the
    first ordering that the machine says still runs, so an unlucky decomposition costs a
    little distance rather than a broken plan.
    """
    return sorted(enumerate_orders(n), key=lambda o: score(o, A, head))


def score(order, A, head):
    total = head[order[0]]
    for a, b in zip(order, order[1:]):
        total += A[a][b]
    return total
