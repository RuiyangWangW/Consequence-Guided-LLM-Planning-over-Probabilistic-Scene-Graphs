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
benchmark uses only `toggled`, on 9 tasks — 3 wanting a lamp left **on**, 6 wanting a lamp or
television left **off**; no benchmark goal uses `open` at all. The goal adapter is not asked for
either.

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
| without validation | 20 | 42 |
| **with validation + repair** | **93** | **93** |
| repaired by the loop | 73 | 51 |
| made worse | 0 | 0 |
| accepted on the first attempt | 71 | 81 |
| mean distance driven, successful runs | 42.8 m | 39.9 m |

**Checking the plan is worth more than doubling the model.** An unchecked 8B solves 42; a
checked 4B solves 93. The 4B's raw output is half as good as the 8B's — 20 against 42 — and
the two finish level once the loop runs, because the loop repaired 73 of the 4B's plans and 51
of the 8B's. It made none worse in either arm; a plan it cannot repair is still refused, so the
arm can only gain.

**Belief against reality.** The loop believed it was finished on 94 tasks (4B) and 93 (8B), and
was right on 93 and 92. One wrong belief each, and both are the same task — `Wainscott_1_int-10`,
where the goal model predicts `on_top` into a trash can and the loop therefore accepts a plan
that satisfies the wrong goal on the first attempt. Nothing downstream can catch that: the
pipeline is measuring against the only goal it has.

The mirror case happens once too: the 8B's `Pomaria_1_int-10` was refused on all five attempts
and then met its goal when driven. The loop is conservative there, which is the safe direction —
it costs attempts, not correctness.

**The seven failures each.** The 4B loses three to placements with an empty hand, two to an
object left unplaced, one to never actually heating what it was told to heat, and one to the
goal model. The 8B loses six to plans that run cleanly and do the wrong thing — two degenerate
loops that fetch from a container and put the object straight back, three swaps executed as
no-ops, one shirt washed but not dried — and the same one to the goal model.

The feedback on those is correct: the complaint names the unmet relations verbatim, and the
planner is shown the goal. What separates the two models now is not the quality of the
complaint but whether they act on it — the 4B's failures are mostly refusals, which are
mechanically repairable in principle; the 8B's are plans the machine accepts and the goal check
rejects, which is the one class neither the checker nor the repair can reach.

## Known limitations

- **The primitives are state changes, not motion.** Manipulation welds objects to the gripper
  and teleports them; the arm does not reach. **Navigation is real** — A* over the eroded map.
- **BEHAVIOR scenes are furniture-only**, so manipulable objects are injected. Where one is
  placed matters as much as that it exists: sampled against the back wall of a counter, it is
  reachable in the graph and not in the world.
- **Half the scenes are not one region of floor.** 24 of 51 disagree with the room graph before
  door thresholds are opened, 4 after. `Wainscott_0_int` is in two pieces.
- **"Was it heated" is expressible; "did it happen at some point" is not.** The goal describes
  the finished state, so a clause about *when* something happened cannot be scored. Three
  shapes: "switch the lamp on and off again" — 6 tasks, where nothing now requires the switch
  to have been touched at all; "closing it each time" — 20 tasks, where the door must end shut
  but nothing checks it was shut *between* placements; and "run it and switch it off" for a
  coffee maker and a sink, which confer no state, so 2 tasks cannot record that they ran.
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

Tests: `test_graph_machine.py`, `test_repair.py`, `test_pipeline.py`, `test_sim2d.py`,
`test_fallback.py`, `test_object_names.py`, `test_stance_order.py`.

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
| `pairs_merged.csv`, `placements.csv`, `category_embeddings.npz` | the RSN's training data and embeddings | `train_rsn.py` | `build_dataset.py`, `embed_categories.py` |

**Both adapters train on the same two files.** `finetune_extraction.py --target extraction`
and `--target goal` read `extraction-train.json` and `extraction-val.json` and differ only in
which field of each row is the answer — the object list or the goal. There is no separate goal
dataset, which is deliberate: when there were two, the goal model had never seen a category the
extractor had, and the two disagreed on names the pipeline then could not reconcile.

Run results (`data/v*.json`) are deleted once superseded. They are the output of a run, not an
input to one, and keeping several invites quoting the wrong one.
