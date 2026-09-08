# LLM safety filter for BEHAVIOR-1K

Turns a natural-language task and a scene into a **validated sequence of BEHAVIOR-1K action
primitives**, then drives it.

An LLM asked to plan for a household robot will confidently name objects that are not in the
house, grasp things that cannot be picked up, and put down objects it never picked up. A robot
cannot see a whole house at once, so it cannot check those claims itself. This pipeline
supplies what is missing — a learned prior over where objects are, and a symbolic model of what
the primitives permit — and refuses plans that violate either.

```
task text ──> objects ────┐                       ┌─> graph machine ──> repair ──┐
          └─> goal state ──┐ ├──> scene graph ──┐    │                              │
floor plan ──> rooms ──────┼─┘       (RSN)      └─> LLM plan                        │
                           └────────────────────────────> 2-D simulator <───────────┘
                                                          or OmniGibson (video)
```

## The pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| 1 | `task_objects.py` | task text -> the objects the task needs |
| 2 | `finetune_extraction.py` | a second small model reads the **goal state** out of the same sentence |
| 3 | `room_graph.py` | floor plans -> room adjacency |
| 4 | `scene_graph.py` | the RSN places the objects in rooms, with confidences |
| 5 | `planner.py` + `graph_machine.py` | the LLM proposes primitives; the machine checks every precondition |
| 6 | `repair.py` + `replan.py` | the machine mends what its refusals imply; what it cannot mend goes back to the LLM, up to 5 times |
| 7 | `sim2d.py` / `execute_plan.py` | the plan is **driven** — 2-D in a second, or OmniGibson with video |

Supporting: `derive_vocab.py` derives the household vocabulary from BEHAVIOR's own activity
definitions; `tasks.py` + `task_shapes.py` + `build_tasks.py` define and **verify** the 100-task
benchmark, replaying every reference plan before it counts. `build_tasks.py` refuses to write
`data/tasks.json`: that file carries corrections applied by hand, and regenerating discards
them silently — which has happened. A rebuild goes elsewhere and is diffed; `evaluate.py` runs the experiment; `ablate_plan.py` measures where the
two checkers disagree.

```bash
source ~/safety_filter/setup_behavior_env.sh

python pipeline.py --scene Beechwood_0_int --task "open the fridge in the kitchen" --json plan.json
python sim2d.py --plan plan.json --scene Beechwood_0_int --gif figures/run.gif   # ~1 s
python execute_plan.py --plan plan.json --video figures/run.mp4                  # ~17 min
```

## The action space

The nine primitives of `StarterSemanticActionPrimitiveSet` — which are nine of the fourteen
`SymbolicSemanticActionPrimitiveSet` declares; `SOAK_UNDER`, `SOAK_INSIDE`, `WIPE`, `CUT` and
`PLACE_NEAR_HEATING_ELEMENT` are excluded:

    GRASP(obj)         PLACE_ON_TOP(obj)   PLACE_INSIDE(obj)
    OPEN(obj)          CLOSE(obj)          NAVIGATE_TO(obj)
    TOGGLE_ON(obj)     TOGGLE_OFF(obj)     RELEASE()

Arity matches the controller signatures — `RELEASE` takes none (`_release(self)`), the other
eight take one object — transcribed into `planner.PRIMITIVES` rather than introspected at
runtime. The robot is Tiago.

## Stage 1 — what the task needs

The task sentence names things; the planner needs a list. A fine-tuned 1.7B LoRA on
Qwen3-1.7B does it, on a ~1.3k-character instruction against the prompted model's 3.3k.

Three findings shaped it:

- **Most apparent misreading was misnaming.** The model returned the right thing under the
  wrong word — `soup` for `bottle_of_soup`, `t_shirt` for `tshirt`. `object_names.py` holds one
  canonicalisation used everywhere, so a naming difference never scores as an extraction error.
- **The relation must not be the template's signature.** Trained on templated sentences, the
  model learned that a phrasing implies a relation rather than reading the preposition. The
  generator now mixes relations across templates.
- **The vocabulary is derived, not curated.** `derive_vocab.py` reads BEHAVIOR's 1018 activity
  definitions and the scenes' contents to decide what counts as a household object and where it
  belongs — 731 categories. Written by hand, the pools drew a `periodic_table` as furniture.

Output has three classes, not two: objects whose location the task **states**, objects it makes
**dependent** on another ("the mug on the tray"), and objects it merely names, whose location is
**uncertain** and left to the RSN.

## Stage 2 — the goal state

A second adapter reads, from the same sentence, the conditions that must hold when the task is
done. This is what makes "the plan ran flawlessly and did the wrong thing" detectable.

The adapter is asked for five forms, and told explicitly not to mention doors or switches —
those are checked separately, from what the plan disturbs:

    on_top(x, y)         x ends resting on y
    inside(x, y)         x ends inside y
    cooked(x, true)      x was heated in an oven, microwave or hob
    washed(x, true)      x went through a washer or dishwasher
    dried(x, true)       x went through a dryer

The last three matter: without them, "heat the pie and put it on the table" is satisfied by a
plan that never opens the oven. `planner.CONFERS` maps each state to the set of
appliances that confer it, and `TOGGLE_ON` writes that state onto the appliance's **contents**.

`GraphMachine.unmet` can additionally test `open(x, bool)` and `toggled(x, bool)`. The
benchmark uses only `toggled`, on 9 tasks, and all 9 want the appliance left **on** — six lamps
and three televisions. No benchmark goal uses `open` at all.

Nothing asks for a device to be left **off**. It used to: six tasks said "switch the lamp on and
off again", whose goal is `toggled(lamp, False)` — a state indistinguishable from never having
touched the switch. A goal describes how the world ends, so a task whose only requirement is
satisfied by doing nothing cannot be scored, and those six were removed rather than kept as
noise. Leaving an appliance off is still enforced, but by the safety check, which is a
different question from the goal: `MUST_SWITCH_OFF` is derived from BDDL's `heatSource` and
`waterSource` properties, so a plan that leaves the oven running fails whether or not the task
mentioned the oven.

**Known inconsistency:** `finetune_extraction.goal_target` emits whatever the training row's
goal holds, so 1,007 of the 8,000 training targets contain `open(...)` conditions the prompt
forbids. The benchmark no longer contains any, so training data and benchmark disagree on this
predicate. Harmless in practice — a stray `open` is redundant with the safety check — but it
means the adapter's behaviour on that one predicate is arbitrary.

**This predicted goal is shown to the planner in every attempt**, initial and retry. It is the
pipeline's own reading of the instruction, never the benchmark's answer key — so it leaks
nothing, and it closes a real gap: plans used to be validated against a goal the planner had
never been shown.

## Stage 3 — room graphs

Room adjacency is read off the floor-plan rasters by dilating each room's pixels and asking
which others it touches. 44 of 51 scenes come out connected; the other 7 genuinely are not one
region of floor.

## Stage 4 — placing objects with the RSN

The RSN gives `P(object | room type)` for open vocabulary. A stated location is the first guess
but never the only one: the graph carries a **ranking**, and the search layer works down it, so
a wrong first guess costs metres rather than the task. Only an object the RSN cannot place at
all is unrecoverable.

## Stage 5 — planning and validation

The LLM is given the action space with each primitive's `requires` and `then`, the believed
scene with per-object confidences, the predicted goal, nine rules, and worked examples.

`graph_machine.py` then replays the plan as typed graph edits in about a millisecond:

| action | preconditions |
| --- | --- |
| `NAVIGATE_TO(x)` | if `x` is inside a container with a door, that container is open |
| `GRASP(x)` | hand empty; `near(x)`; if inside a container, it is open; `x` is graspable |
| `PLACE_ON_TOP(x)` | `near(x)`; holding something |
| `PLACE_INSIDE(x)` | `near(x)`; holding something; `x` open if it has a door |
| `OPEN`/`CLOSE(x)` | `near(x)`; `x` has a door |
| `TOGGLE_ON`/`OFF(x)` | `near(x)`; `x` has a switch |
| `RELEASE()` | none |

Two of these were learned the hard way.

**`near` is an edge, not a room.** It used to mean "same room", and a fifth of the plans the
machine accepted were undrivable — every one drove to something in the kitchen and then acted
on something else three metres away in the same kitchen. `NAVIGATE_TO` now writes a `nearby`
edge to the object and its contents, and that class is gone.

**`NAVIGATE_TO` is not free.** It had no preconditions at all, but the simulator has to *see* an
object to drive to it, and a bottle behind a shut fridge door is invisible: the robot searches
every believed room and the run dies on the step. It now refuses, and the repair rules compose
into go-to-container → open → go-to-object.

### Safety

Anything the plan opens it must shut. Switches are not symmetric: only appliances in
`planner.MUST_SWITCH_OFF` must be switched off — derived as BDDL `heatSource | waterSource` plus
the `CONFERS` appliances, 39 categories. An oven, a dishwasher, a washer, a dryer, a coffee
maker and all nine sink types qualify; a lamp or a television does not, so "turn on the lamp"
is a task the machine can accept instead of failing for succeeding.

This is checked automatically from what the plan disturbed. Goals do **not** restate it. They
used to be required to, and that requirement was deleted: it duplicated a check the machine
already makes, in a form that enforced nothing — an untouched switch reads as off, so the
condition was satisfied by never approaching the object — and it made "leave the lamp on"
inexpressible.

## Stage 6 — repair, then replanning

The machine's refusals name a fault and an object, so many corrections follow from the refusal
rather than from reading English. `repair.py` applies them inside every attempt:

| fault | edit |
| --- | --- |
| `not_near` | insert `NAVIGATE_TO(subject)` |
| `closed` / `not_open` | insert `OPEN(container)` |
| `room` | drop the step — a room is not a `NAVIGATE_TO` destination |
| `no_door` / `no_switch` | delete the `OPEN`/`CLOSE` or `TOGGLE` pair in one edit |
| `not_graspable` | drop the `GRASP` on a fixture |
| `empty_hand` | insert `NAVIGATE_TO(x) + GRASP(x)` for what the goal wants there |
| `holding` | finish the held object's errand rather than dropping it |
| end of plan | append `CLOSE` / `TOGGLE_OFF` for anything left disturbed |

The rules compose: a bare `OPEN` moves the plan sideways until the next round supplies the
drive that makes it count. So the loop runs freely and keeps only the **best plan seen**,
compared against the plan it started from — a repair can be speculative but can never hand back
something worse.

Two faults are not derivable from the graph and are derived from the **goal** instead:
`GRASP` with a full hand and `PLACE` with an empty one are one mistake seen from two sides, and
the goal is the only thing that states where each object is going.

What it cannot mend goes back to the LLM as the plan quoted with the offending step marked,
plus the failing action's own contract — because knowing *that* a step is wrong is not knowing
what to write instead. When a plan runs to the end and still misses the goal, placement steps
whose destination the goal wants a different relation at are marked too, with both verbs'
effects quoted; that confusion between "on top of" and "inside" is the single largest failure
class in both models.

## Stage 7 — execution

`sim2d.py` drives the plan over the real floor plan: A* on an eroded traversability map, a
wedge camera, frontier search for objects the robot has not seen. It runs a `GraphMachine` over
ground truth and supplies its `nearby` edges from measured distance, so both checkers run the
same preconditions in the same code and differ only in where the edges came from.
`execute_plan.py` does the same in OmniGibson and records video.

### Where the two checkers disagree

`ablate_plan.py --fuzz` builds plans the graph machine accepts, then drives them. With `near`
as a room, about a fifth were undrivable; with the `nearby` edge, 480 of 480 across two scenes
were drivable, and the remaining disagreements were the graph model being *conservative* —
demanding a `NAVIGATE_TO` geometry says is unnecessary, which costs a step rather than the run.

That fuzzer cannot generate the case that matters most, though: it never spawns an unseen
object inside a shut container, which is exactly the gap `NAVIGATE_TO`'s new precondition
closes. Treat "480 of 480" as a statement about the cases the fuzzer reaches.

## The benchmark

100 tasks over 10 scenes, in `tasks.py`, expanded by `task_shapes.py` and verified by
`build_tasks.py`. A task is only written out if its reference plan runs and reaches its goal.

A task has to be **sayable** (the instruction names everything needed), **reachable** (the
objects are in one region of floor), and **mean one thing**. The extraction ground truth is
derived from the task's own structure, not written by hand. Where a sentence gives a
preposition it is decisive; otherwise BEHAVIOR's `fillable` annotation decides in-versus-on.

Predicate counts: `object_inside` 102, `on_top` 97, `washed` 17, `cooked` 12, `toggled` 9,
`dried` 3.

BEHAVIOR scenes are furniture-only — across all ten benchmark scenes there are **three**
objects a robot could pick up — so anything to manipulate is injected, from categories the
object dataset ships.

## The experiment

`evaluate.py` runs the whole pipeline over the benchmark. Both arms come from **one run per
task**, so the comparison is exact: decoding is greedy, so the first plan is the same plan in
both arms, and they differ only in whether anything is done about a bad one.

| arm | what it is |
| --- | --- |
| without validation | the model's first answer, kept whatever it says |
| with validation | the repair loop — mend what the machine can, hand back what it cannot, up to 5 attempts |

**Both arms are driven.** There is no symbolic scoring: a plan's verdict is what happened when
the robot ran it. An earlier version replayed plans symbolically against the benchmark's goal,
which handed out credit for satisfying an answer key the pipeline never sees — two tasks scored
`ok` while the validation loop had rejected them for five straight attempts. That scorer is
gone.

The summary reports three numbers per model: the two driven arms, and **how often the loop
believed it was finished against how often it actually was**. The gap between those is the cost
of a wrong goal prediction, which nothing measured before.

```bash
HF_HOME=/mnt/check/ruiyangw/hf_cache python evaluate.py \
    --model Qwen/Qwen3-8B --extractor models/h1-1.7b-v2 --goal-model models/state-1.7b-v2 \
    --json data/run.json
```

**The pipeline never sees ground truth.** It gets the instruction, its own extraction, and the
RSN's guesses. The simulator scores against truth — it is the examiner, and it earns the verdict
by executing.

### Where the errors come from

Failures cascade, so each is attributed to the stage that **caused** it, not the stage that
differed. Extraction can only cause a failure two ways: the plan wanted an object it never
surfaced, or it read a location the task did not state and the failing step is that object.
Anything else is the planner's. Measured on an earlier run, the old "blame extraction if
extraction differed" rule was wrong on every one of 11 tasks.

A wrong RSN room is not a failure — the search layer works down the ranking.

### Results

100 tasks, both arms driven in the 2-D simulator.

| | Qwen3-4B | Qwen3-8B |
| --- | --- | --- |
| without validation | 23 | 40 |
| **with validation + repair** | **88** | **91** |

**These numbers carry about ±3 of run-to-run noise, and the noise is not sampling.** Decoding is
greedy - `do_sample=False`, one question one answer - and three generations in one process are
byte-identical. Across processes they are not: the same prompt, the same model, the same card
gives a different plan than a stored run did, because which CUDA kernels are chosen depends on
what else is resident, and a different reduction order flips a near-tied argmax. The consequence
is that no single run's total should be read to the task, and a change worth one or two tasks
cannot be demonstrated by running the benchmark twice and subtracting.

What *can* be demonstrated is a change on tasks named in advance - and the discipline matters,
because naming them wrongly is easy. The goal-name resolution below was first predicted to fix
four tasks. Rebuilding every row's belief and comparing it against that row's predicted goal shows
it can only ever have affected **one**: on three of the four the belief graph and the goal use the
*same* string, because stage 1 wrote `tshirt` and the goal adapter wrote `tshirt`, so the loop was
comparing like with like. The `t_shirt` that appears in those rows' failure messages is the answer
key's spelling, printed by the simulator after grounding, not a comparison the loop ever made. Two
of the three flipped to passing in the re-run anyway - which is exactly the +/-3 noise, and a
reminder that a flip in the predicted direction is not evidence unless the mechanism is checked.

**Checking the plan is worth more than doubling the model.** An unchecked 8B solves 43; a
checked 4B solves 87. The 4B's raw output is little better than half the 8B's — 23 against 43 —
and the loop closes most of the gap, repairing 65 of the 4B's plans and 50 of the 8B's.

**A harness fault that used to cost both arms, now fixed.** `GraphMachine._resolve_goal_name`
matched goal object names by equality, and its docstring said loose matching had been removed
because the goal is "canonicalised on the way in" — but `parse_goal` canonicalises only the
*predicate* vocabulary, never object names, so that canonicalisation did not exist. The function
was also a tautology: `return name if name in self.graph.objects else name` returns `name` either
way. A goal term that never resolves can never hold, so the loop refuses every plan for five
attempts and reports whatever the last one wrote.

`_resolve_goal_name` now resolves through `object_names.same`, but only when **exactly one** object
in the graph could be the term — `cabinet` with both a top and a bottom cabinet present stays
unresolved, which is what the strict version existed to protect against, since it used to guess and
let the plan break ties.

**It is worth exactly one task in two hundred, and that is the honest figure.** Scanning all 200
rows and rebuilding each belief: four carry a goal term that fails `==` and `object_names.same`
resolves, every one of them the same word, `tv` for `standing_tv`. No term in the benchmark is
ambiguous and none is unresolvable. Of those four, one actually failed - `Pomaria_0_int-04` in the
4B, where the model's first plan is correct, drives clean, and is refused five times over
`toggled(tv)`. The other three carried the mismatch and passed regardless, because their plans
failed or succeeded on other grounds first.

The fix is still right - a term that cannot resolve makes a condition unsatisfiable, and the loop
then rejects correct work - but it is a one-task fix, not the three-to-six the first count
suggested. That first count came from grepping failing rows for a resolvable mismatch instead of
asking whether the mismatch was what refused the plan.

**Belief against reality.** The loop believed it was finished on 88 tasks (4B) and 94 (8B), and
was right on 86 and 91. The wrong beliefs are two and three respectively, and they share the
same two tasks — `Beechwood_0_int-08` and `Wainscott_0_int-07`, where stage 1 *invents* a
location ("coffee_cup ON_TOP desk", "plate ON_TOP coffee_table") that the instruction never
states. The belief then has the object already at its destination, the plan does not fetch it,
and the loop has nothing to object to: it is measuring against a world that is wrong before the
robot moves. Nothing downstream can catch that.

The mirror case happens too: `Pomaria_0_int-09` (both arms) and `Pomaria_0_int-04` (8B) were
refused on all five attempts and then met their goal when driven — the goal-name fault above.
Being conservative is the safe direction; it costs attempts, not correctness.

**The failures.** The 4B's thirteen: six goal-unmet, four placements with an empty hand, three
where the plan reaches for something it never navigated to. The 8B's seven: four goal-unmet and
three unreached objects. Four of the 4B's and one of the 8B's are downstream of a stage-1
error — an invented location or a misread relation — and three in each arm are the goal-name
fault, which leaves the genuinely planner-attributable count much smaller than the totals
suggest.

The feedback itself is correct: the complaint names the unmet relations verbatim, and the
planner is shown the goal. What separates the two models is not the quality of the complaint
but whether they act on it. The 4B still loses four tasks to a placement with an empty hand —
a refusal the machine states plainly and the model does not answer — where the 8B loses none.
Both lose the same three to objects the plan reaches for without navigating to first, and both
lose their remaining tasks to plans the machine accepts and the goal check rejects, which is
the one class neither the checker nor the repair can reach.

## Several errands in one instruction

"Put the notebook in the office bookcase, put the wineglass on the breakfast table, and take
the textbook out of the kitchen bookcase" is not a harder single task — it is several tasks
sharing a robot. Everything above still applies to each errand; what is new is that the robot
must decide **what order to do them in**, and that the decision is worth making at all only
because the robot is embodied.

A symbolic planner has no preference here: every ordering satisfies the same goal. The order
matters because the robot has to walk, and because it does not know where anything is. Stage 4
gives a distribution over rooms, not a location, so reaching an object costs an expected
*search* rather than a known distance — and sweeping the kitchen for the mug also reveals the
bowl. That is the whole lever: information gathered doing one errand changes what the others
cost.

| Stage | Module | What it does |
| --- | --- | --- |
| 5 | `gavel.decompose` | the instruction is split into one sentence per errand |
| 6 | `replan.py` | each errand is planned, validated and mended **exactly as a single task** |
| 7 | `cost_matrix.py` + `search_cost.py` + `order.py` | the errands are sequenced by expected distance |
| 8 | `gavel.compose` | the concatenation is checked — subplans valid alone can still break together |
| 9-11 | `gavel.solve` | commit one errand, execute it, update the belief with what was seen, re-optimise |

Stages 1-4 and 6 are the single-task pipeline unchanged, and that is deliberate: the multi-task
arm reuses the same extractor, the same RSN, the same `GraphMachine`, the same repair loop and
the same simulator, so a difference in the results is the new stages and not a new pipeline.

**Stage 1's answer key has three classes, and they are not interchangeable.** `stated` is
where the sentence names an object's *room* ("the kitchen countertop"); `dependent` is where it
names an object's *support* ("the wineglass from the countertop"); `uncertain` is what it says
nothing about, and only that last class is what the RSN is asked to guess. `subtasks.py`
originally computed its own key and put every name into `uncertain` with the other two empty.
Two things went wrong at once, and the second is much worse than the first:

* the extractor, which reads the sentence correctly and answers in the three classes, was
  marked wrong on every one of the 500 instructions;
* `populate` was handed those same wrong classes, so the RSN was set guessing rooms that the
  instruction had stated outright — and the belief the planner reasoned over was worse than the
  instruction warranted.

The multi-task set now gets its key the way the single-task set does: authored per sentence in
`data/subtask_extraction.json`, in the same format, read by the same `extraction_truth`
function. Merging across errands respects the disjointness — an object `uncertain` in one clause
and `stated` in another is stated in the combined instruction, because one of the clauses says
so.

The errands themselves are written the same way the single-task benchmark is — `subtasks.py`
declares them in `tasks.py`'s format and `build_multitask.py` puts every one through
`build_tasks.verify`, the same function that verifies the 100-task set, before any of them is
combined. There is one task generator, not two. `build_multitask.py` refuses to write
`data/tasks.json` at all, for the same reason `build_tasks.py` does.

Combinations are drawn under two rules. No two errands in one instruction may touch the same
object, which is what makes any ordering legal and the cost decomposition exact. And no
furniture *category* may appear in two different rooms across them.

The second rule exists because the belief is keyed by category: `stated` maps `bookcase` to one
room and `populate` creates one `bookcase` node. An instruction saying "the office bookcase" in
one clause and "the kitchen bookcase" in another therefore has no single referent for
`NAVIGATE_TO bookcase`, and one of the two clauses is wrong whatever the planner does. That is
not a hard instruction, it is an unanswerable one, and 174 of the first 500 were built that way
— 35% of the benchmark scoring the planner on an ambiguity it could not resolve. Sharing a
*destination* is still allowed; two things can go in the same bookcase, and that leaves the
referent unique. Every scene still admits 200 to 1,500 admissible combinations against the 50
it needs.

Precedence is deliberately absent: two errands that must happen in a fixed order are not two
errands, they are one.

### What the ordering costs to compute

The objective decomposes, because the errands share no object:

```
J(order) = sum internal(k)            constant — the same for every ordering
         + sum A[order(k-1), order(k)]  an asymmetric path through a pairwise matrix
```

So the matrix is built once with `N^2` belief rollouts and every ordering is then scored by
adding `N` numbers, instead of rolling the world forward `N!` times. On a 3-errand instance the
pairwise scores reproduce the full rollouts to the decimal on all six orderings. That is what
makes re-optimising at **every** errand boundary cheap enough to do online.

The matrix is asymmetric — doing the kitchen errand then the bedroom one leaves the robot in the
bedroom, and the reverse leaves it in the kitchen — so this is an asymmetric Hamiltonian *path*.
Christofides does not apply: it needs a symmetric metric obeying the triangle inequality, and
builds a spanning tree and a perfect matching, neither defined on a directed graph. At 2-5
errands exhaustive enumeration is exact and instant, so nothing is approximated. Past about
eight, Held-Karp is the exact choice; it handles asymmetry natively.

`order.ranked` returns orderings cheapest-first and the executor takes the first whose
concatenation still runs. Reordering is sound only because the errands are independent, and
with subplans a *model* wrote that is an expectation rather than a guarantee — one that opens a
cabinet in one errand and reaches into it in another has made them dependent whatever the
instruction said. Checking each candidate ordering means a bad decomposition costs distance
instead of correctness.

### What a sweep teaches

`gavel.observe` is the reason reordering beats ordering once. A sweep is not a query about one
object: the robot walks a room and sees what is in it, so every object it is still looking for
is resolved by that one visit — the ones that are there become located, and the ones that are
not lose that room from their candidates. Ruling a room out is worth as much as finding
something, and it is why the estimate for an errand elsewhere can go *up* once its likeliest
room has been eliminated.

Execution reads rooms from the **truth** graph and the estimator reads them from the belief.
Confusing the two was the one bug that made the whole stage meaningless: reading rooms off the
belief makes the robot "find" the mug in whichever room the RSN guessed, so a wrong guess is
never paid for and never corrected, and search stops costing or teaching anything.

When ruling out leaves an object with no candidate the robot can reach, the belief has been
*refuted*, not sharpened — the object exists, so it is somewhere nobody looked. The distribution
falls back to the object's own RSN ranking, and failing that to a sweep of what is left.
Without this the estimate goes infinite and every ordering ties, not because the errand is
impossible but because the guesser ran out of guesses.

### The RSN's beliefs are now complete distributions

Stage 4 used to hand the searcher a ranked list of rooms plus the winner's confidence, and the
tail shared whatever was left over. That does not sum to one, and an estimator that weighs a
candidate by its probability then makes an object the model is *unsure* about look cheap to
find. `scene_graph` now returns a full distribution over every room the scene has, normalised.

The RSN scores room *types*; a house has room *instances*. A type's mass is split equally
across its instances — the model has said nothing that tells two bedrooms apart, so they are a
genuine tie — and the tie is broken where the information to break it exists: `search_cost`
orders equal-probability rooms by distance from wherever the robot is standing, so it sweeps the
near bedroom first. That is a fact about the robot's position, which changes as it moves, so it
cannot be baked into the scene graph.

Keeping only the largest instance of each type, which is what this did before, was not merely a
mis-ranking: a sofa really in the smaller living room was **unreachable by search**, because the
room never entered the candidate list at all.

### The three arms

| arm | decomposes | orders | re-orders online |
| --- | --- | --- | --- |
| `monolithic` | no | no | no |
| `static` | yes | once, against the prior | no |
| `gavel` | yes | yes | yes |

`static` against `gavel` isolates the online information and nothing else — same decomposition,
same subplans, same cost model, same executor, so the only thing that varies is *when* the order
is decided. `monolithic` against the other two is the value of decomposing at all. All three run
on the 4B.

Every arm is judged by the 2-D simulator against the true world, exactly as the single-task
experiment is. Two distances are reported: `walked`, what the cost model charged — the number
the ordering stage actually optimised — and `driven`, what the simulator really drove. The
ordering claim is only worth making if it holds on the second.

**Results, 50 instructions sampled evenly across the ten scenes, 4B planner** (benchmark stamp
`25afcba3af59a180`):

| arm | goal reached |
| --- | --- |
| monolithic — one model, whole instruction | 17/50 |
| static — decomposed, ordered once | **46/50** |
| gavel — decomposed, reordered online | **46/50** |

**Decomposing is worth 29 tasks out of 50.** The whole-instruction read carries a stage-1 error on
36 of the 50 - the extractor was trained on single sentences naming two to four objects and these
name up to thirteen - and every dropped object is one the planner then cannot refer to. Reading the
instruction a clause at a time removes that, and the validation loop refuses only 2 errands out of
roughly 170.

The two decomposed arms differ by -0.7% of driven distance, which is *not* evidence that reordering
hurts: they choose a different order on only 7 of the 50 tasks, so this sample cannot resolve an
effect of about a percent either way. The 500-instruction reference-plan measurement below is the
one with the power; this one is consistent with it and says nothing on its own.

### What the ordering is worth

An earlier version of this section reported reordering as worth +0.6% and called the stage
marginal. That was the wrong measurement. `static` does not skip the ordering - it optimises once
on its first pass and never revises - so `static` against `gavel` was only ever the *online
increment on top of an already optimal order*. Nothing measured what ordering itself buys.

`gavel.solve(..., force=<permutation>)` supplies the missing rung: an order imposed from outside,
with no optimisation. `force=tuple(range(n))` is "do the errands in the order the instruction
names them", which is what a planner with no ordering stage does. Every one of the `n!` orderings
of every task can then be enumerated on both meters. On the current benchmark, 161 instructions
whose every ordering runs clean:

| policy | driven | vs written |
| --- | --- | --- |
| written — the instruction's own order | 10,970 m | — |
| random | 11,423 m | +4.13% |
| **static** — ordered once, against the prior | 10,300 m | **-6.10%** |
| **gavel** — re-optimised online | 10,079 m | **-8.11%** |
| oracle — the cost model's own best ordering | 9,957 m | -9.23% |
| the true floor — the simulator's best ordering | 8,807 m | -19.72% |

**Where the distance goes**, as what each step is worth on top of the one below it:

| step | points | share of the gap |
| --- | --- | --- |
| ordering at all (written -> static) | 6.10 | 30.9% |
| re-optimising online (static -> gavel) | 2.01 | 10.2% |
| optimising the cost model exactly (gavel -> oracle) | 1.11 | 5.7% |
| **a cost model that matched the simulator** (oracle -> true floor) | **10.49** | **53.2%** |

Over all 500 instructions, paired, gavel beats static by **+1.2%** of driven distance and **+4.0%**
of the cost model's - **+1.3%** and **+4.8%** once the two-errand instructions, which are a
structural zero, are set aside. An earlier measurement of the same thing gave +1.01% [+0.22%,
+1.79%] before the executor's free-teleport bug was fixed and before the errands were spread. Both are worth quoting and neither should be quoted alone - the tasks the
stricter rule drops are `Wainscott_0_int` ones where gavel does *worse*, so that rule flatters it.
The gain is also fragile and unevenly distributed: orders differ on 107 of 500 tasks, gavel drives
shorter on 66 and **longer on 38**, dropping the five largest single-task wins halves the total to
+0.53%, and gavel is net negative in two scenes of ten.

**The largest single loss is not the ordering policy.** It is that the cost model and the simulator
disagree about which ordering is cheapest. Kendall's tau between the two meters across a task's
orderings is about 0.28-0.36 - an earlier draft printed 0.53, which is inflated by two-errand tasks
where tau is +/-1 by construction and carries no information. The model's own best ordering is the
simulator's best on 37-44% of tasks. A *perfect* optimiser of the cost model recovers 0.82% of a
12.22% true ordering headroom on a full enumeration of the 246 instructions with three errands or
fewer. No amount of reordering can reach past that.

### When ordering pays, and when it cannot

**Rooms.** Only the transition term of `J = internal + transitions` depends on the order, so errands
crowded into one part of the house leave little to win. Drawing combinations uniformly averaged 3.4
rooms an instruction; `build_multitask.combine` now requires errands to reach `size + 2` rooms,
capped at what each scene can offer and relaxed per scene only when its pool runs out, which lifts
the mean to 4.5 and takes the share of instructions touching four or more rooms from 45% to 84%.
Measured against the old set, that roughly doubled the cost-model gain (1.7% -> 3.6%) and raised the
simulator gain from 0.6% to 1.0%, with the five-errand gain going 0.9% -> 2.0%.

An earlier draft of this section said the stage buys "nothing at all" below four rooms. That was a
median read of a stratum in which most tasks tie: the totals for the few-room group are -2.98%
(static) and -4.77% (gavel), not zero. The effect is a gradient, not a cliff. Any stratified table
here should print the paired total and the number of non-tied tasks beside the median, because
medians over mostly-tied strata read as categorical when they are not.

**Being wrong, not being unsure.** The obvious explanation for the small online increment is that
83% of task objects have a point-mass belief, so sweeping teaches nothing. That is not what the data
says. A point mass is not a *correct* belief: 18.5% of them name the room the object is not in. What
actually moves an ordering is discovering that a confidently stated room is wrong. Splitting all 500
instructions on whether the prior is ambiguous and whether it is wrong:

| prior | tasks | orders differ | driven saved |
| --- | --- | --- | --- |
| sharp and right everywhere | 46 | 0 | 0.0 m (+0.00%) |
| sharp but at least one **wrong** | 88 | 13 | 218.8 m (+3.10%) |
| ambiguous, none wrong | 116 | 7 | 13.8 m (+0.24%) |
| ambiguous **and** wrong | 250 | 87 | 111.1 m (+0.57%) |

64% of the entire benchmark's saving comes from thirteen tasks with *no prior ambiguity at all*.

Sweeping that axis directly confirms it. `exp_corrupt.py` takes objects the belief is *certain*
about and moves the certainty to a room the object is not in, at rates from none to all of them,
leaving the amount of ambiguity untouched:

| confident beliefs corrupted | static | gavel | gain | instructions whose order changes |
| --- | --- | --- | --- | --- |
| 0% | 7,089 m | 6,918 m | +2.41% | 18/100 |
| 25% | 7,975 m | 7,512 m | **+5.80%** | 39/100 |
| 50% | 8,268 m | 7,820 m | +5.41% | 53/100 |
| 75% | 8,850 m | 8,276 m | +6.49% | 54/100 |
| 100% | 9,284 m | 8,827 m | +4.92% | 51/100 |

Corrupting a quarter of them more than doubles the gain and triples the number of instructions
where reordering does anything at all, and the response then saturates. Deleting stated rooms -
the manipulation that adds *ambiguity* rather than error - moves the same gap only from 1.21% to
1.98%. So the benchmark does not need vaguer instructions to exercise this stage; it needs
instructions the robot can be wrong about, which is what a real RSN prior on a real house
supplies. (Measured on the cost model's meter; the simulator was not driven for the sweep.)
Where beliefs are sharp and correct the stage is provably inert - 0 of 46 tasks reorder, and exactly
0 m is saved - which is the behaviour a correct implementation should show and is the strongest
single piece of evidence that the machinery is sound. It also means the productive way to make this
benchmark harder is not to delete stated rooms, which merely adds ambiguity along the +0.24% axis,
but to *corrupt* them.

**A structural floor.** Two-errand instructions are provably identical between the two arms - there
are only two orderings and the executor commits to the first before anything can be learned - and
they are a fifth of the benchmark. They contribute a guaranteed zero to the headline. Either exclude
them from the gavel-against-static comparison or say that they are in it.

## Known limitations

- **The primitives are state changes, not motion.** Manipulation welds objects to the gripper
  and teleports them; the arm does not reach. **Navigation is real** — A* over the eroded map.
- **BEHAVIOR scenes are furniture-only**, so manipulable objects are injected. Where one is
  placed matters as much as that it exists: sampled against the back wall of a counter, it is
  reachable in the graph and not in the world.
- **Half the scenes are not one region of floor.** 24 of 51 disagree with the room graph before
  door thresholds are opened, 4 after. `Wainscott_0_int` is in two pieces.
- **"Was it heated" is expressible; "did it happen at some point" is not.** The goal describes
  the finished state, so a clause about *when* something happened cannot be scored. Two shapes
  remain: "closing it each time" — 13 tasks, where the door must end shut but nothing checks it
  was shut *between* placements; and "run it and switch it off" for a coffee maker and a sink,
  which confer no state, so those tasks cannot record that they ran. The third shape, "switch
  the lamp on and off again", was removed from the benchmark for this reason rather than left
  in as 6 unscoreable tasks.
- **`PLACE_INSIDE` on something with no inside is allowed.** The graph model has no containment
  affordance and the simulator refuses only on distance.
- **The RSN is a soft prior.** `P(toilet in kitchen)` is 0.18, not ~0; unseen names fall back to
  the text embedding. It conditions on room type, not on this scene.
- **The RSN was trained on 7 of the 10 benchmark scenes.**
- **The goal model cannot abstain.** Asked about a destination it has no relation for, it
  guesses rather than declining.

## Files

`graph_machine.py` the checker · `repair.py` the mechanical edits · `replan.py` the loop ·
`planner.py` prompts and affordances · `scene_graph.py` beliefs · `world_graph.py` typed edges ·
`sim2d.py` + `floor_world.py` the 2-D simulator · `sim_eval.py` driving a plan ·
`evaluate.py` the experiment · `build_tasks.py` + `tasks.py` + `task_shapes.py` the benchmark ·
`derive_vocab.py` the vocabulary · `finetune_extraction.py` the adapters.

Multi-task: `gavel.py` decomposition, observation and the commit-execute-reorder loop ·
`exp_ladder.py` the written/random/static/gavel/oracle ladder · `exp_spread.py` headroom against
room spread · `exp_uncertainty.py`, `exp_infogain.py`, `exp_decomposition.py`, `exp_params.py`,
`exp_adversarial.py`, `exp_corrupt.py` the diagnosis of why online reordering adds little ·
`order.py` the pairwise matrix and exhaustive enumeration · `search_cost.py` expected search and
navigation over a belief · `cost_matrix.py` A* room-to-room distance and per-room sweep cost ·
`subtasks.py` + `build_multitask.py` the 500-task set · `evaluate_multi.py` the three arms ·
`oracle_order.py` the ordering stage measured on reference subplans, with the model taken out.

Tests: `test_graph_machine.py`, `test_repair.py`, `test_pipeline.py`, `test_sim2d.py`,
`test_fallback.py`, `test_object_names.py`, `test_stance_order.py`, `test_gavel.py`.

## The datasets

`data/` holds only live inputs — nothing derived from a past run, nothing kept "just in case".
Every file here is read by something, and everything read is here.

| file | what it is | who reads it | regenerated by |
| --- | --- | --- | --- |
| `tasks.json` | **the benchmark** — 100 tasks, byte-identical to a rebuild from `tasks.py` | `evaluate.py`, `sim_eval.py`, `build_tasks.py`, tests | `build_tasks.py` (refuses to overwrite it) |
| `extraction_truth.json` | stage-1 answer key, derived from each task's structure | `build_tasks.py` | — |
| `extraction-train.json` | 8,000 generated examples, `--seed 7` | `finetune_extraction.py` | `extraction_data.py --n 8000 --seed 7` |
| `extraction-val.json` | 500 held out, `--seed 11` | `finetune_extraction.py` | `extraction_data.py --n 500 --seed 11` |
| `extraction-dev.json` | 200 for prompt comparisons, `--seed 3` | `extraction_eval.py` | `extraction_data.py --n 200 --seed 3` |
| `household_vocab.json` | 731 categories and where each belongs | `extraction_data.py` | `derive_vocab.py` |
| `room_graphs.json` | room adjacency for all 51 scenes | `scene_graph.py`, `floor_world.py`, `world_graph.py` | `room_graph.py` |
| `scene_survey.json` | what each scene is furnished with — what the tasks were written against | `tasks.py` | — |
| `reference-sim.json` | evidence every benchmark task is physically achievable | `run_reference_sim.py` | `run_reference_sim.py` |
| `subtask_extraction.json` | stage-1 answer key for the errands, authored per sentence — same format as `extraction_truth.json` | `build_multitask.py` | — |
| `subtasks.json` | 118 short errands, 10-12 per scene, each verified by `build_tasks.verify` | `build_multitask.py` | `build_multitask.py` (writes both files in one run) |
| `multitask.json` | **the multi-task benchmark** — 500 instructions, 2-5 errands each | `evaluate_multi.py`, `oracle_order.py` | `build_multitask.py` |
| `multitask-stamp.json` | which generation of the benchmark that is — content hash, seed, size mix, mean room spread | anything reporting a result | written beside it |
| `pairs_merged.csv`, `placements.csv`, `category_embeddings.npz` | the RSN's training data and embeddings | `train_rsn.py` | `build_dataset.py`, `embed_categories.py` |

**Both adapters train on the same two files.** `finetune_extraction.py --target extraction`
and `--target goal` read `extraction-train.json` and `extraction-val.json` and differ only in
which field of each row is the answer — the object list or the goal. There is no separate goal
dataset, which is deliberate: when there were two, the goal model had never seen a category the
extractor had, and the two disagreed on names the pipeline then could not reconcile.

**Every result records the benchmark it ran against.** `build_multitask.py` writes a stamp - a
hash of the instruction set plus its seed, size mix and mean room spread - and `oracle_order.py`
and `evaluate_multi.py` write the same stamp beside their own output. Results carrying different
stamps are not a comparison, however alike their task ids look. This is not hypothetical: the
combination rule changed mid-analysis once and the file was regenerated underneath a set of running
experiments, so three quarters of the ids kept their names while changing how many errands they
held, and several numbers that still looked comparable no longer were. A hash makes that loud.

Run results (`data/v*.json`) are deleted once superseded. They are the output of a run, not an
input to one, and keeping several invites quoting the wrong one.
