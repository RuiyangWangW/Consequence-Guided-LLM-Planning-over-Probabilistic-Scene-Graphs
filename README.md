# LLM safety filter for BEHAVIOR-1K

Turns a natural-language task and a scene into a **validated sequence of BEHAVIOR-1K action
primitives**, then drives it.

An LLM asked to plan for a household robot will confidently name objects that are not in the
house, grasp things that cannot be picked up, and put down objects it never picked up. A robot
cannot see a whole house at once, so it cannot check those claims itself. This pipeline
supplies what is missing — a learned prior over where objects are, and a symbolic model of what
the primitives permit — and refuses plans that violate either.

**Everything here is BEHAVIOR-1K.** The scenes, floor plans, room labels and object instances are
BEHAVIOR-1K's; the affordances the world model enforces are read from BDDL's own annotations
rather than written by hand - `OPENABLE`, `TOGGLEABLE`, `FILLABLE` and `NOT_GRASPABLE` all come
out of `properties_to_synsets.json` and the mass table; and the nine primitives are the ones
`execute_plan.py` drives in OmniGibson. The 2-D simulator is a fast executor over those same
assets, not a separate world: it reads the BEHAVIOR floor plan, places objects where the task's
ground truth puts them, and moves a robot of the real base radius along A* routes it could
actually drive. A plan validated here is a plan in BEHAVIOR-1K's action space, over BEHAVIOR-1K
objects, in a BEHAVIOR-1K house.

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
definitions. The benchmarks are built and verified separately.
The builder refuses to write
`data/tasks.json`: that file carries corrections applied by hand, and regenerating discards
them silently. A rebuild goes elsewhere and is diffed. `evaluate.py` runs the experiment and
`ablate_plan.py` measures where the two checkers disagree.

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
pipeline's own reading of the instruction, never the benchmark's answer key, so it leaks nothing -
and a plan is never refused against a goal the planner was not shown.

## Stage 3 — room graphs

Room adjacency is read off the floor-plan rasters by dilating each room's pixels and asking
which others it touches. 44 of 51 scenes come out connected; the other 7 genuinely are not one
region of floor.

## Stage 4 — placing objects with the RSN

Where an object is, when the sentence does not say, comes from a **Relational Semantic Network**
after Ginting et al., *SEEK: Semantic Reasoning for Object Goal Navigation in Real World
Inspection Tasks* (RSS 2024, IV-B):

    object name -> frozen text encoder -> MLP -> P(object present | room type), for all room types

`BAAI/bge-small-en-v1.5` encodes the category name to 384 dimensions and is **frozen**; a
three-layer MLP (256-128-64, dropout 0.2) maps that to one logit per room type. Two properties of
this shape do the work. The embedding is frozen and textual rather than a learned per-category
table, so a name never seen in training still lands near similar names and gets an answer - which
is the only workable choice when a language model can ask about any object. And the room is an
*output dimension*, not an input, so one forward pass scores every room at once, which is what the
search estimator needs.

Trained on **11,218 object placements across all 51 BEHAVIOR scenes**, 197 merged categories over
37 room types, held out by scene. Where SEEK regresses onto soft probabilities distilled from
GPT-4, real layouts give hard occurrence labels, so this trains with BCE - the correct likelihood
for a binary observation - with positive-class weighting, and a two-parameter temperature
calibration fitted afterwards. Held-out **AUC 0.904, average precision 0.602, Brier 0.057**.

The output is a *soft prior*, and treating it as anything firmer is a mistake: `P(toilet in
kitchen)` is 0.18, not zero. So a stated location is the first guess but never the only one - the
graph carries a complete ranked distribution and the search layer works down it, so a wrong first
guess costs metres rather than the task. Only an object the RSN cannot place at all is
unrecoverable. It conditions on room *type*, not on this particular house, and is measured to put
its argmax on the right room **47% of the time**; the true room is second on a further 197 of 832
guesses. That accuracy is what the ordering stage is reasoning over, and why carrying the whole
distribution rather than its argmax is worth anything at all.

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

Two of these carry more weight than their size suggests.

**`near` is an edge, not a room.** `NAVIGATE_TO` writes a `nearby` edge to the object and its
contents, so proximity is to a *thing*. Room-level proximity is too weak to be a precondition: a
robot standing at the kitchen counter is in the same room as the fridge three metres away, and a
plan that acts on it from there is accepted by the checker and undrivable by the robot.

**`NAVIGATE_TO` is not free.** It refuses an object inside a shut container, because the simulator
has to *see* something to drive to it and a bottle behind a fridge door is invisible - the robot
would sweep every believed room and die on the step. The repair rules compose to satisfy it:
go-to-container -> open -> go-to-object.

### Safety

Anything the plan opens it must shut. Switches are not symmetric: only appliances in
`planner.MUST_SWITCH_OFF` must be switched off — derived as BDDL `heatSource | waterSource` plus
the `CONFERS` appliances, 39 categories. An oven, a dishwasher, a washer, a dryer, a coffee
maker and all nine sink types qualify; a lamp or a television does not, so "turn on the lamp"
is a task the machine can accept instead of failing for succeeding.

This is checked automatically from what the plan disturbed, and goals do **not** restate it.
Requiring them to would enforce nothing - an untouched switch reads as off, so the condition is
satisfied by never approaching the object - and would make "leave the lamp on" inexpressible.

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

## The benchmarks

**Single-task: 100 long-horizon tasks over 10 scenes.** Each names objects to fetch, containers
to open, appliances to run, and a place to leave things. Predicate counts: `object_inside` 102,
`on_top` 97, `washed` 17, `cooked` 12, `toggled` 9, `dried` 3. Every task ships with a reference
plan that is known to reach its goal in the simulator, so a failure is always the pipeline's.

**Multi-task: 500 instructions over the same 10 scenes**, each asking for 2-5 independent errands
in one sentence - "put the notebook in the office bookcase, take the casserole out of the fridge,
warm it in the microwave, and leave it on the breakfast table". Predicate counts: `on_top` 964,
`object_inside` 869, `cooked` 110, `toggled` 99, `washed` 65, `dried` 62; 173 of the 500 ask for a
state an appliance confers. The errands in one instruction never share an object, so every
ordering is legal and the robot's only reason to prefer one is that it has to walk.

BEHAVIOR scenes are furniture-only - across all ten benchmark scenes there are **three** objects a
robot could pick up - so anything to manipulate is injected, from categories the object dataset
ships.

Both sets are fixed artefacts with a content stamp, and every result records the stamp it ran
against. How they are constructed and verified is documented separately; none of it is needed
to read the results below.

## The experiment

`evaluate.py` runs the whole pipeline over the benchmark. Both arms come from **one run per
task**, so the comparison is exact: decoding is greedy, so the first plan is the same plan in
both arms, and they differ only in whether anything is done about a bad one.

| arm | flags | what it is |
| --- | --- | --- |
| LLM-only | `--attempts 1 --repair-at off` | the model's first answer, kept whatever it says |
| LLM + feedback | `--attempts 5 --repair-at off` | the machine's refusal handed back, up to 5 tries, nothing edited on the model's behalf |
| GAVEL | `--attempts 5 --repair-at loop` | the same loop, plus the mender applying what its own refusals imply |

**Every arm is driven.** There is no symbolic scoring: a plan's verdict is what happened when the
robot ran it. Replaying a plan against the benchmark's goal instead would hand out credit for
satisfying an answer key the pipeline never sees.

The summary reports three numbers per model: the two driven arms, and **how often the loop
believed it was finished against how often it actually was**. The gap between those is the cost
of a wrong goal prediction, which nothing measured before.

```bash
HF_HOME=/mnt/check/ruiyangw/hf_cache python evaluate.py \
    --model Qwen/Qwen3-8B --extractor models/h1-1.7b-v5 --goal-model models/state-1.7b-v5 \
    --attempts 5 --repair-at loop \
    --json data/run.json
```

**The pipeline never sees ground truth.** It gets the instruction, its own extraction, and the
RSN's guesses. The simulator scores against truth — it is the examiner, and it earns the verdict
by executing.

### Where the errors come from

Failures cascade, so each is attributed to the stage that **caused** it, not the stage that
differed. Extraction can only cause a failure two ways: the plan wanted an object it never
surfaced, or it read a location the task did not state and the failing step is that object.
Anything else is the planner's. Attributing a failure to whichever stage merely *differed* from
the reference is not the same test and gets it wrong: extraction differs on many tasks that
succeed, and on many failures whose cause is elsewhere.

A wrong RSN room is not a failure — the search layer works down the ranking.

### Results

100 tasks, every arm driven in the 2-D simulator against the true world.

| | Qwen3-4B | Qwen3-8B |
| --- | --- | --- |
| LLM-only — one plan, no feedback, no repair | 23 | 43 |
| LLM + feedback — the machine's refusal handed back, up to 5 tries | 57 | 77 |
| **GAVEL — feedback and repair** | **89** | **92** |

Both increments are large and they hold at both model sizes: validation feedback alone is worth
+34 at each, and letting the machine *edit* the plan rather than only complain about it adds a
further +32 at 4B and +15 at 8B.

**A 4B model with the world model beats an 8B model without one** - 89 against 43, and it also
beats the 8B given feedback but no repair (77). That is the clearest statement of what the
symbolic layer buys: it substitutes for model capacity.

The mean number of attempts says why repair is not merely more accurate but cheaper. Feedback
alone burns 3.39 attempts at 4B, grinding against faults it cannot fix; with the mender the same
model settles in 1.69, because a fault the machine can edit is edited rather than re-asked.

**These numbers carry about ±3 of run-to-run noise, and the noise is not sampling.** Decoding is
greedy - `do_sample=False`, one question one answer - and three generations in one process are
byte-identical. Across processes they are not: the same prompt, the same model, the same card
gives a different plan than a stored run did, because which CUDA kernels are chosen depends on
what else is resident, and a different reduction order flips a near-tied argmax. The consequence
is that no single run's total should be read to the task, and a change worth one or two tasks
cannot be demonstrated by running the benchmark twice and subtracting.

A change worth one or two tasks therefore cannot be demonstrated by running the benchmark twice
and subtracting; it has to be shown on tasks named in advance, with the mechanism checked on each.

**The failures.** Nineteen across both models, and they cluster into three shapes.

| | Qwen3-4B (11) | Qwen3-8B (8) |
| --- | --- | --- |
| goal unmet in the true world | 3 | 4 |
| a `GRASP` the world refused | 3 | 3 |
| a placement with an empty hand | 3 | 0 |
| acted on something never navigated to | 0 | 1 |
| other | 2 | 0 |
| *of which also carry a stage-1 error* | *4* | *3* |

Four of the 4B's and three of the 8B's are downstream of an extraction fault - an invented
location or a misread relation - so the genuinely planner-attributable count is smaller than the
totals suggest.

The class neither the checker nor the repair can reach is **goal unmet**: the plan applies
cleanly, leaves nothing open or running, and still does not achieve what was asked. The machine
has no complaint to make about a plan that breaks none of its rules, so there is nothing to hand
back and nothing to mend. It is the largest single class for the 8B and the reason its score is
not higher.

The 4B's distinguishing failure is different: three placements with an empty hand, a refusal the
machine states plainly and the model does not act on, where the 8B loses none. That is the gap
that closes with model capacity, and it is smaller than the gap the world model itself closes.

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
arm reuses the same extractor, the same RSN, the same `GraphMachine`, the same repair loop and the
same simulator, so a difference in the results is the new stages and not a new pipeline.

Two properties of the instructions matter for what follows, and both are enforced when the set is
built rather than assumed here. The errands in one instruction **share no object**, which is what
makes every ordering legal and lets the expected-cost objective decompose into a pairwise matrix.
And no furniture *category* appears in two different rooms across them, so `NAVIGATE_TO bookcase`
always has one referent.

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

### The methods being compared

Seven arms, defined once in `baselines.py` and shared by both harnesses. Each changes exactly
one thing from the one above it, so a difference in the results has one candidate cause.

| arm | plans with | ordering | belief the cost model gets | validation / repair |
| --- | --- | --- | --- | --- |
| `llm-only` | the model, 1 attempt | none — as written | — | none |
| `sayplan` | the model, 5 attempts | none — as written | — | validated, refusal fed back, **no repair** |
| `epog` | itself, from the graph diff | min travel cost | argmax | none at all |
| `gavel-map` | the model, mended | once, offline | **argmax only** | validated and mended |
| `gavel-static` | the model, mended | once, offline | full distribution | validated and mended |
| `gavel` | the model, mended | **online** | full distribution | validated and mended |
| `oracle` | the benchmark's reference plans | optimal, by computing every route | **ground truth** | — |

`llm-only` to `sayplan` is what validation feedback buys; `sayplan` to `gavel` adds the mender and
the cost model. Reading down the three GAVEL rows, `gavel-map` to `gavel-static` is what the
*distribution* is worth over its argmax with the ordering policy held fixed, and `gavel-static` to
`gavel` is what *revising as perception arrives* is worth with the belief held fixed.

`sayplan` and `llm-only` plan for themselves rather than sharing the mended subplans - sharing them
would credit those arms with a repair stage they are defined not to have. `epog` writes its own
plan from the graph difference and is never checked; see `epog.py`.

**Oracle is given ground truth in every sense** - it grounds against the true world, walks straight
to the true position without searching, and uses the benchmark's own reference plans. Because it
never searches, its route is fully determined by the ordering and can be *computed* rather than
driven: `sim_eval.route_matrix` gives exact A* distances over the same grid the simulator uses, so
choosing among all `n!` orderings costs `n!` additions and one simulation instead of `n!`
simulations. Verified on nine instructions to find the genuinely shortest ordering 9 times out of 9.
Note what that does and does not bound: every other arm is driven against the *belief* and pays to
sweep for whatever the RSN misplaced, so the gap to Oracle is ordering quality **plus** the cost of
imperfect perception.

Distance is always what the robot **drove** in the simulator. The cost model's own estimate is
recorded for diagnosis and deliberately not reported: the arms do not all estimate the same
quantity, so the column is not comparable across rows.

Results are in [The full experiment](#the-full-experiment).

## The full experiment

Three experiments on the benchmark generation stamped `50bdbedd1d478e99` (500 instructions, ten
scenes, 50 each). Stage 1 runs `models/h1-1.7b-v5` and the goal adapter `models/state-1.7b-v5`;
decomposition and planning both run the **unfine-tuned base model** under test, so a "4B run"
means a 4B decomposer and a 4B planner.

### 1. Validation and repair, on 100 single long-horizon tasks

The table under *Results* above. LLM-only 23/43, plus feedback 57/77, plus repair 89/92, for
Qwen3-4B and Qwen3-8B.

### 2. The baselines, on 500 multi-task instructions (Qwen3-8B)

| method | success | driven | planning time |
| --- | --- | --- | --- |
| LLM only | 97/500 | 86.7 m | 37.1 ± 15.5 s |
| SayPlan — validate, feed back, 5 tries, no repair, no ordering | 387/500 | 83.0 m | 59.9 ± 39.3 s |
| EPoG — graph edits from the MAP belief, no model, no repair | 301/500 | 84.5 m | 14.9 ± 4.3 s |
| GAVEL-MAP — argmax belief, ordered once | **462/500** | 82.5 m | 43.1 ± 25.6 s |
| GAVEL Static — full distribution, ordered once | **462/500** | 79.7 m | 43.1 ± 25.6 s |
| **GAVEL** — full distribution, reordered online | **462/500** | **78.0 m** | 43.1 ± 25.6 s |
| Oracle — ground truth throughout | 500/500 | 56.2 m | 0.9 ± 0.5 s |

Oracle solves every instruction, so each of GAVEL's 38 failures is a real one rather than an
impossible task. `LLM only` collapses to 19% here against 43% on single tasks - errands compound,
and one unrepaired plan fails the whole instruction. `EPoG` is at 60% because it cannot express
`cooked`/`washed`/`dried` at all: those name no edge to add or remove, and about a third of the
instructions contain one.

**The ordering ladder.** The three GAVEL variants share an identical success set, so their
distances are directly comparable, paired over all 462:

| rung | driven | effect |
| --- | --- | --- |
| GAVEL-MAP | 82.45 m | |
| GAVEL Static | 79.69 m | distribution over argmax **-2.77 m**, z = -3.30 |
| GAVEL | 78.01 m | online replanning **-1.68 m**, z = -3.41 |
| | | both **-4.45 m**, z = -5.11 |

Each rung changes exactly one thing and each clears significance on its own. This is what the
benchmark rebuild bought: on the previous generation the same two comparisons were -0.46 m
(z = -1.54) and -0.47 m (z = -1.85), indistinguishable from noise, because 299 of 500 instructions
named the room outright and left the belief nothing to be uncertain about. That is a defect in the
benchmark rather than a finding about the method: an instruction on which every arm makes the
identical decision cannot distinguish them however it is scored, and 299 of 500 were like that.

The set was rebuilt to pose the question, not to answer it. Instructions are admitted on whether
the arms **decide differently**, never on which one wins - and the result was not fixed by that
choice. On the rebuilt set the distribution arm is *worse* than the argmax on **129 of the 309**
instructions where the two differ (42%), and online replanning is worse than ordering once on
**69 of 213** (32%). Selection that picked winners would not leave those numbers standing.

Two properties changed: the share of furniture references the RSN must guess went from 25% to 54%,
and the share of instructions on which the three arms do not all choose the same order from
201/500 to 438/500. Both carry a cost worth stating: the set is built using the estimator
under test, and it skews towards instructions with more errands (two-errand instructions fell
from 106 to 41), which should be noted wherever the online increment is quoted.

### 3. Foundation models, on 500 instructions

Qwen3-4B against Qwen3-8B, LLM-only against GAVEL. The 8B rows are the same run as experiment 2
rather than a fresh sample, so the two tables agree exactly instead of differing by run-to-run
noise.

| model | arm | success | driven |
| --- | --- | --- | --- |
| Qwen3-4B | LLM only | **9/500** | 121.3 m |
| Qwen3-4B | GAVEL | **374/500** | 93.8 m |
| Qwen3-8B | LLM only | **97/500** | 86.7 m |
| Qwen3-8B | GAVEL | **462/500** | 78.0 m |

**A 4B model on its own is not a planner for these instructions at all.** Nine successes out of
five hundred is under 2%, and it is not that the model cannot write a plan - it writes one every
time - but that a multi-errand instruction gives it four or five chances to write a step the
world refuses, and one is enough to fail the whole thing. Errors compound; the single-task
version of the same arm scores 23/100.

**The world model is worth more than the model.** 4B with it reaches 374/500, roughly four times
what the 8B reaches without it (97/500). The same ordering holds on the single-task benchmark
(89 against 43), so it is not an artefact of instruction length. What the symbolic layer buys is
not a better plan from a better model - it is the ability to *refuse* a bad one and say why,
which a 4B model can act on as well as an 8B one.

The distance column is over each arm's own successes, so the two LLM-only rows are not comparable
to the others: they describe the handful of easiest instructions those arms happened to finish.
The comparison that is sound is GAVEL to GAVEL - 93.8 m at 4B against 78.0 m at 8B - which says
the smaller model still writes materially worse plans even when they are valid.

### Where these numbers come from

Every figure above is read off a file in `data/`, so a claim can be checked without re-running
anything.

| file | what it holds |
| --- | --- |
| `exp1-{4b,8b}-{llmonly,feedback,gavel}.json` | experiment 1 - six runs, 100 tasks each, per-task plan, verdict, attempts and per-stage timing |
| `exp2.json` | experiment 2 - 500 instructions, seven arms, Qwen3-8B |
| `exp3.json` | experiment 3 - 500 instructions, `llm-only` and `gavel`, Qwen3-4B |
| `order-sweep.json` | the ordering ladder measured without a language model, on the benchmark's own reference subplans |
| `rsn-accuracy.json` | the RSN's per-guess accuracy: believed room, true room, rank and confidence |
| `multitask-stamp.json` | which generation of the benchmark all of the above ran against |

Each result also carries a `-stamp.json` naming the benchmark it ran against; `merge_shards.py`
refuses to combine shards whose stamps disagree.

### The timing column, and what is wrong with it

**Planning time is compute only** - decomposition, extraction, grounding, the goal adapter, every
planning attempt, every repair, and the ordering - and excludes the robot's driving time, which is
the simulator's speed rather than the method's.

The experiment-2 figures above are **internally comparable but absolutely inflated**. All seven
arms run inside one process per shard, back to back on one GPU, so they meet identical conditions
and the comparison between rows is sound; but four shards ran concurrently on four GPUs, so the
absolute scale is not what a single uncontended run would give.

Experiment 1's timings are worse than that and are deliberately omitted, because each arm was a
*separate process on a separate GPU*. Measured there: on 76 tasks where two arms produced
**byte-identical plans** from identical prompts, one arm was recorded at 1.37 s per emitted step
and the other at 0.65 s - the same work, 2.1x apart. Running the same two arms sequentially in one
process removes the gap. A dedicated sequential pass, one arm at a time on an idle machine and
discarding the first task per process to exclude the ~40 s of CUDA warm-up, is what these tables
will eventually quote. Until then: **treat every absolute second here as an upper bound, and only
compare rows within experiment 2.**

## Known limitations

- **The primitives are state changes, not motion.** Manipulation welds objects to the gripper
  and teleports them; the arm does not reach. **Navigation is real** — A* over the eroded map.
- **`PLACE_INSIDE` on something with no inside is allowed.** The graph model has no containment
  affordance and the simulator refuses only on distance.
- **The RSN is a soft prior.** `P(toilet in kitchen)` is 0.18, not ~0; unseen names fall back to
  the text embedding. It conditions on room type, not on this scene.
- **The goal model cannot abstain.** Asked about a destination it has no relation for, it
  guesses rather than declining.

## Files

**The pipeline, stage by stage.**

| module | what it is |
| --- | --- |
| `task_objects.py` | stage 1 - reads the objects a sentence names, in three classes |
| `planner.py` | the action space, the prompts, the BDDL-derived affordances, and `get_generator` (local weights or a hosted API) |
| `scene_graph.py` | stage 3-4 - the RSN's belief: a complete normalised distribution over every reachable room |
| `room_graph.py`, `floor_world.py` | floor plans, room adjacency, reachability, A* over the eroded grid |
| `world_graph.py` | the typed-edge graph the machine reasons over |
| `graph_machine.py` | the world model: preconditions, effects, safety, and `unmet(goal)` |
| `repair.py` | the mechanical edits the machine can make to a plan itself |
| `replan.py` | the loop - ask, validate, mend, complain, ask again |
| `sim2d.py` + `sim_eval.py` | the executor: driving, sweeping a room, and judging a plan against the true world |
| `api_models.py` | hosted models behind the same `prompt -> text` interface as local ones |

**Multi-task.**

| module | what it is |
| --- | --- |
| `gavel.py` | decomposition, the belief update from what was seen, and the commit-execute-reorder loop |
| `search_cost.py` | expected search and navigation cost over a belief |
| `cost_matrix.py` | A* room-to-room distance and per-room sweep cost, built once per scene |
| `order.py` | the pairwise matrix and the asymmetric Hamiltonian path over it |
| `epog.py` | the symbolic baseline: the plan is the goal graph minus the believed graph |
| `baselines.py` | the seven arms, defined once and shared by both harnesses |

**Benchmarks.** `tasks.py`, `subtasks.py`, `task_shapes.py`, `build_tasks.py`,
`build_multitask.py` build and verify the two task sets, and `derive_vocab.py`,
`extraction_data.py`, `build_dataset.py`, `embed_categories.py`, `train_rsn.py` build the
vocabulary, the adapters' training data and the RSN.

**Experiments.** `experiments/` holds the harnesses that produce the reported results, each
runnable from the repo root:

| script | what it runs |
| --- | --- |
| `evaluate.py` | experiment 1 - the single-task ladder; `--attempts` and `--repair-at` select the arm |
| `evaluate_multi.py` | experiments 2 and 3 - all seven arms in one process per shard |
| `oracle_order.py` | the ordering ladder measured on reference subplans, with the language model taken out |
| `merge_shards.py` | concatenates sharded results, and refuses to merge across benchmark stamps |
| `exp_rsn_accuracy.py` | the RSN's per-guess accuracy behind the 47% quoted above |

Each inserts the repo root on `sys.path` itself, anchored on a marker file rather than a fixed
number of parent directories, so `data/` paths resolve against the working directory and a script
that moves does not silently lose its imports.

The development notebook - the sweeps, ablations and per-failure diagnostics that answered
questions this document now states outright, together with the test suite - lives in `cached/`
and is not tracked.
