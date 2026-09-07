# LLM safety filter for BEHAVIOR-1K

Turns a natural-language task and a scene into a **validated sequence of BEHAVIOR-1K
action primitives**, then executes it in OmniGibson.

An LLM asked to plan for a household robot will confidently reference objects that are not
in the house, grasp things that cannot be picked up, and place objects it never picked up.
A robot cannot see the whole house at once, so it cannot check those claims directly. This
pipeline supplies the missing information — a learned prior over where objects are, plus a
symbolic model of what the primitives permit — and rejects plans that violate either.

```
task description ──> objects needed ──┐                     ┌─> graph machine ─> repair loop
                                      ├──> scene graph ──> LLM plan                    │
scene floor plan ──> room graph ──────┘        (RSN)         └─> 2-D simulator  <──────┘
                                                              or OmniGibson (video)
```

There are **two** checkers, and the difference between them is the point. The graph machine
replays a plan as typed graph edits in about a millisecond and asks whether every
precondition holds; the 2-D simulator drives the robot over the real floor plan and asks
whether it could have got there and reached. Measured over 480 random plans, one
precondition still separates them — see *How far apart are the two models now?*

| Stage | Module | What it does |
| --- | --- | --- |
| 1 | `task_objects.py` | task text -> the objects the task needs |
| 1' | `finetune_extraction.py` | the small model that reads the goal state out of the same sentence |
| 2 | `room_graph.py` | floor plans -> room adjacency graph |
| 3 | `scene_graph.py` | RSN places those objects in rooms |
| 4 | `planner.py` | LLM proposes primitives; `graph_machine.py` checks them |
| 4' | `repair.py` | the machine mends what its own refusals imply, inside every attempt |
| 4'' | `replan.py` | hands what it could not mend back to the LLM, up to 5 times |
| 5 | `execute_plan.py` | grounds onto scene objects, runs in OmniGibson, records video |
| 5' | `floor_world.py` + `sim2d.py` | the same nine primitives on a 2-D grid, in a second, with no Isaac |

And, to measure whether any of it helps:

| what | module | it does |
| --- | --- | --- |
| the vocabulary | `derive_vocab.py` | BEHAVIOR's activities and scenes -> the household objects, and where each belongs |
| the benchmark | `tasks.py`, `task_shapes.py`, `build_tasks.py` | 100 tasks over 10 scenes, every one proved solvable before it is written out |
| the experiment | `evaluate.py` | runs the full pipeline over the benchmark with and without the checker, and says where the errors come from |
| the ablation | `ablate_plan.py` | drops one action from a working plan and asks which model notices |

Where the plan does not know where an object is, three more modules close the gap:
`nav_controller.py` turns one `NAVIGATE_TO` into a room drive plus a frontier search,
`object_map.py` records what the camera has actually seen, and `world_graph.py` accumulates
it as typed edges that `graph_machine.py` can then check a plan against offline.

```bash
source ~/safety_filter/setup_behavior_env.sh   # behavior env, CUDA 12.8, GPU 1

python pipeline.py --scene Beechwood_0_int --task "open the fridge in the kitchen" --json plan.json
python execute_plan.py --plan plan.json --video figures/run.mp4    # OmniGibson, ~17 min
python sim2d.py --plan plan.json --scene Beechwood_0_int --gif figures/run.gif   # 2-D, ~1 s
```

```
Objects and where they are most likely to be (0-1 confidence):
  kitchen_0: fridge (0.83)

plan:
   1. NAVIGATE_TO(fridge)
   2. OPEN(fridge)
```

## The action space

Exactly the nine primitives in `SymbolicSemanticActionPrimitiveSet`:

    GRASP(obj)         PLACE_ON_TOP(obj)   PLACE_INSIDE(obj)
    OPEN(obj)          CLOSE(obj)          NAVIGATE_TO(obj)
    TOGGLE_ON(obj)     TOGGLE_OFF(obj)     RELEASE()

Arity is read from the controller method signatures, not assumed: **`RELEASE` takes no
argument** (`_release(self)`); the other eight take exactly one object. The robot is Tiago,
from `tiago_primitives.yaml`.

## Stage 1 — what the task needs

`task_objects.py` asks the LLM. One prompt, one parse, no rules.

The prompt asks for things the robot could navigate to, grasp, open or place on, and
excludes ingredients and parts. That boundary matters: "make coffee" should yield the
coffee maker and the cup, not coffee beans and water, which the robot cannot manipulate as
objects.

```
make coffee                        -> coffee_maker, cup
mop the floor                      -> mop, bucket
open the fridge in the kitchen     -> fridge
put the laptop on the coffee table -> laptop, coffee_table
```

### Most of what looked like misreading was misnaming

Extraction was once the largest remaining source of failure — 43% of what Qwen3-4B got
wrong and 61% of the 8B's. Sorting every apparently-missed object by *why* showed only two
of twenty-three were reading failures. The rest were names read correctly and written in a
form that does not ground: 65% room-qualified (`office_bottom_cabinet` for the category
`bottom_cabinet`), 26% synonyms (`apple_juice` for `bottle_of_apple_juice`).

That is a naming problem, and it is solved at the source now — see *One vocabulary* below.
`strip_room` remains as a backstop for a fused name, and it does not discard the qualifier:
a stated room is precisely what the RSN would otherwise have to guess, so it comes back as
a hint that `populate` uses with probability 1.0. A hint naming a room type the scene lacks
is ignored rather than obeyed.

**Tuning on the test set is the obvious hazard**, so `extraction_data.py` generates labelled
instructions from slot templates with no scene involved — the answer is known by
construction — and the benchmark's movable objects are always held back. A first version of
that dev set had no room-qualified phrasing, the single hardest thing in the real
instructions, and was useless at 90% against the benchmark's 69%. With that added it tracks.
Prompt ablations live in `extraction_prompts.py` and are measured by `extraction_eval.py`;
they are kept for the record but the shipped extractor is fine-tuned.

**One rule, not three flags.** It used to be `--exclude-benchmark`, `--keep-fixtures` and
`--shapes`, and the combination mattered in a way nothing recorded: the two adapters were
trained on different ones, so the goal model had never seen a container as a swap source and
answered `on_top` for a cabinet. Now the benchmark's **movables** are always held back —
a model that has seen `apple_pie` in training is being tested on memory — while its
**fixtures** are not, because knowing that a cabinet opens and a washer washes is background
knowledge every instruction assumes. Holding fixtures back left the generator unable to
build the tasks it most needs to teach: drawing a container for a swap became so rare it
never happened in 8,000 examples.

### The relation must not be the template's signature

A template that always states one relation teaches the wording, not the preposition. `swap
the {a} on the {b}` only ever meant `ON_TOP`, so the goal model answered `on_top` for a
cabinet — for a task whose own sentence said *in* — and was marked wrong for it. `UNDER` and
`NEXT_TO` were worse: each lived in a single template and appeared nowhere else, so they
were that template's signature rather than a thing the model read.

So a slot may be drawn rather than fixed. `SUPPORT` returns a surface *or* a container, and
what comes back decides the relation, the preposition and the goal predicate together:

```
swap the {a} {a_from_s} with the {b} {b_from_t}, using the {u} to set one down
  drew a table   -> "off the coffee table"   ON_TOP   goal on_top(a, t)
  drew a cabinet -> "out of the cabinet"     INSIDE   goal object_inside(a, t), open(t, False)
```

`AT` in a template's stated relations means "whatever the drawn target can host": a
container gets `INSIDE` or `NEXT_TO`, a surface gets `ON_TOP`, `UNDER` or `NEXT_TO`, a
movable gets `ON_TOP` or `NEXT_TO`. Weighted, not uniform — "on the table" is what
instructions mostly say, and a dataset where a third of objects start *under* something is
not a dataset of household tasks. Eight of the sixteen templates now draw their relation —
every one that states where something starts — so no relation is any template's signature. **Shape 11 is the one exception and is correctly
fixed**: its wording is "open the {c}, take out the {a}", so the sentence itself says INSIDE.

Goals stay `on_top`/`object_inside`, because the primitives only place on-top or inside —
nothing can be asked to end up *under* something.

Two templates that were the same task written twice, once per relation, collapsed into one.

### The vocabulary is derived, not curated

The object pools were regexes over the whole BEHAVIOR catalogue — 51 scenes including
chemistry labs, restaurants and gyms — so `SURFACE_PAT` matched anything containing
`table|bench|chair|bed` and instructions came out reading *"leave it on the periodic
table"*, *"the graduated cylinder is inside the fridge"*, *"the dishwasher in the bedroom"*.
Those are not household tasks, and a model trained on them learns that the words carry less
than they do.

`derive_vocab.py` answers it from BEHAVIOR's own data instead of a hand-written list:

| source | question it answers |
| --- | --- |
| `activity_definitions/*/problem0.bddl` (1018) | which objects real household activities use, and in which rooms |
| `combined_room_object_list.json` | what each of the 51 scenes is furnished with, room by room |
| `category_mapping.csv` | which categories realise each synset |

Four things that were judgements became derivations:

- **Which rooms a house has.** Hand-listing them let a `sauna_bench` through, because I had
  guessed a sauna was a room a house has. Read off the fifteen `_int` scenes: 19 types.
- **Which activities count.** 187 that happen only outdoors are dropped; 829 kept.
- **Fixture or movable — different tests, because different questions.** A fixture is part
  of the building, so a household fixture is one a house is actually built with: a
  `massage_bed` is installed somewhere, never in a house. A movable is brought to the task,
  so no house installs it — `mug` appears in cafe scenes and zero houses and is obviously a
  kitchen object. No crisp line exists there, so none is drawn.
- **How common each is.** Weights come from activity usage, so an object one activity
  mentions stays as rare as it really is. This caught an inflation: one synset is realised
  by several categories, and crediting each the whole activity made `massage_bed` look as
  common as `bed`. The credit is split across realisations.

731 categories, each with the rooms it is used or installed in — which is also what the room
qualifier is now drawn from, so "the dishwasher in the bedroom" is unreachable by
construction rather than filtered out afterwards.

**The RSN is deliberately not part of this.** It predicts where the *furniture* is, and is
trained on what each scene actually contains — no task-inserted objects, which is right:
a mug is brought to the task, so no scene can tell you where it lives. The extraction and
goal models are the opposite case and do train on graspable insertables, since those are
what instructions are about.

### Fine-tuning beats prompting, and a 1.7B beats an 8B

`finetune_extraction.py` LoRA-tunes a small model on generated instructions instead of
talking a large one into the task. Two things make the comparison honest:

**The training data shares no *movable* vocabulary with the test set.** The benchmark's 82
movables are held back, so the model never sees `apple_pie` or `bath_towel` and is then
asked about them. Nothing is scene-derived, so none of this is memorising the ten scenes.

**The tuned model gets a short prompt, not the long one.** Twelve worked examples exist to
demonstrate a format and an answer length; a model trained on thousands should need
neither. So it is given a ~400-character instruction against the prompted model's 3,300 —
a fair test rather than a stacked one. Answers are trained in the exact format
`task_objects.parse` already reads, so nothing downstream changes.

Exact-match on the 100 benchmark tasks:

| | model | prompt | exact |
| --- | --- | --- | --- |
| prompted, original | Qwen3-4B | 2.3k | 69% |
| prompted, original | Qwen3-8B | 2.3k | 56% |
| prompted, shipped | Qwen3-4B | 3.3k | 78% |
| prompted, shipped | Qwen3-8B | 3.3k | 82% |
| fine-tuned, 500 examples | Qwen3-1.7B | 0.4k | 74% |
| **fine-tuned, 2,000 examples** | **Qwen3-1.7B** | **0.4k** | **87%** |
| fine-tuned, 8,000 examples | Qwen3-1.7B | 0.4k | 87% |
| fine-tuned, 8,000 examples | Qwen3-4B | 0.4k | 85% |

**Five hundred generated examples make a 1.7B beat a prompted 4B**, and two thousand make
it beat the best prompted 8B by five points at a fifth of the size and an eighth of the
prompt. Training the winner took 14 minutes on one A5000, updating 17.4M LoRA parameters —
1% of the model.

Two results worth stating plainly because they decide how much effort this deserves.
**It saturates at about 2,000 examples**: 8,000 buys nothing (87% either way), so the
tedious part — generating data at volume — turns out to be unnecessary. And **size stops
helping once tuned**: the fine-tuned 4B scores 85%, *below* the 1.7B, so the remaining
errors are not capacity. Recall is 99% and the residue is relations — INSIDE versus ON_TOP,
and current position versus where the task wants the thing to end up.

```bash
python derive_vocab.py                      # BEHAVIOR -> data/household_vocab.json
python extraction_data.py --n 8000 --seed 7 --out data/extraction-train.json
python finetune_extraction.py --model Qwen/Qwen3-1.7B --out models/extract-1.7b-2000
python extraction_eval.py --on data/tasks.json --variants extract-1.7b-2000 \
                          --adapters models/extract-1.7b-2000
```


### Three classes, not two lists

Every object the task names falls into exactly one of three classes, ordered by how much
the instruction says about where it is:

```
DEPENDENT: potato INSIDE fridge      the task named its support
STATED:    fridge IN kitchen         the task named its room
UNCERTAIN: stove                     the task said nothing
```

A support beats a room and a room beats nothing, and each object takes the first that
applies. The classes being **disjoint** is what lets `populate` read the output directly
instead of reconciling overlapping lists.

The middle class did not exist at first, and losing it was expensive. 55% of instructions
name a room, the extractor read them correctly, and then the answer format had nowhere to
put the room — so the model was trained to *delete* a fact the task had given it and the
pipeline fell back to guessing a location it had been told. Adding it took room recall from
about 65% to **98%**.

Two defects fixed alongside:

**Eight of eighteen room words matched no scene room type.** An instruction says "the office
cabinet" and the scene graph says `private_office`; `pantry` is `pantry_room`, `hallway` is
`corridor`. The stated room was extracted, then silently discarded for failing an equality
test. `scene_graph.ROOM_SYNONYMS` fixes it.

**`strip_room` split 22 real categories.** `bar_soap` became soap-in-a-bar, `gym_shoe` became
shoe-in-a-gym, `garden_chair` and `kitchen_analog_scale` the same. It now refuses to split a
name that is itself a category.

With the hierarchy, on the 100 benchmark tasks:

| | exact | objects | class | relations | rooms |
| --- | --- | --- | --- | --- | --- |
| 1.7B, 2,000 examples | 89% | 98% | 95% | 96% | 98% |
| **1.7B, 8,000 examples** | **94%** | 99% | 96% | 96% | 98% |

## Stage 1' — the goal state, from the same sentence

A checker that only asks "was every action applicable" cannot ask the question that matters:
**does this plan do the task?** The benchmark's goal is the answer key and handing it to the
planner would be telling it the answer, so the goal has to be read out of the instruction by
something that has not seen the scene.

A second LoRA on Qwen3-1.7B, trained on the same scene-free generator as the extractor, does
it: task text in, goal predicates out.

```
take the apple pie out of the fridge, heat it in the oven, and leave it on the table
  -> on_top(apple_pie, breakfast_table)
     cooked(apple_pie, True)
```

It emits placements (`on_top`, `object_inside`) and the appliance-conferred states
(`cooked`, `washed`, `dried`, via `planner.CONFERS` — a dishwasher washes what is inside it).
It deliberately does **not** emit `open` or `toggled`: "put back what you disturbed" is not a
property of *this* task but of every task, and `GraphMachine` derives it from what the plan
actually opened — which also catches a cupboard the instruction never mentioned, where a
goal condition on a named object cannot.

Its output is a **prediction, never ground truth**. It is what the loop validates against, so
a wrong reading shows up as a failure rather than being hidden by one, and `evaluate.blame`
has a `goal_model` bucket to name it when it happens. Asking the *planner* for its own goal
was tried instead and was worse: its goal was right about half the time, and validating
against a wrong goal accepts plans that then miss the real one.

Both models are LoRA adapters over one shared base — loading two full copies cost 3.27 GiB
and OOM'd the 8B; sharing the base costs 0.06.

## Stage 2 — room graphs

`room_graph.py` derives room adjacency from the ground-truth floor plans in `layout/`
(`floor_insseg_0.png` for instances, `floor_semseg_0.png` for types). Two rooms are
adjacent when their pixel regions come within a few pixels of each other; dilating each
region before testing overlap bridges the wall thickness in the raster. Substantial overlap
is required rather than any contact, so rooms meeting at a single diagonal corner do not
register as traversable.

Adjacency is deliberately *not* taken from door objects: only 3 of 51 scenes come out
connected that way, because archways and open-plan boundaries carry no door object.

```bash
python room_graph.py                 # -> data/room_graphs.json
python visualize_graph.py --all      # -> figures/<scene>_graph.png
```

`visualize_graph.py` renders segmentation beside the derived graph, nodes at each room's
true centroid, colors keyed to room *type* from a fixed global palette so a kitchen reads
the same in every figure.

**44 of 51 scenes are fully connected.** The 7 that are not (`school_*`,
`restaurant_brunch`, `office_cubicles_left`) are genuine gaps in the raster, not bugs — in
`school_gym` the bathroom is physically separated from the locker rooms by unlabeled floor.

## Stage 3 — placing objects with the RSN

The room graph knows the layout but not the contents. `scene_graph.py` asks the RSN for
`P(object | room type)` and attaches each object to its most probable room **among the
types this scene actually has** — a high score for `garage` is irrelevant in a house with
no garage.

Every object is placed and its probability travels with it into the LLM prompt; there is no
confidence threshold by default. Thresholding discards real objects — `Rs_int` genuinely
contains a laptop, but the RSN scores it 0.13 for living_room, and a 0.30 cut marked it
absent, blocking a task the scene could actually support. Carrying the number instead lets
the planner weigh a doubtful object against a confident one. `--threshold` restores the old
behavior. The RSN will sometimes be wrong, and that is expected: its plan is a first guess.

### A stated location is the first guess, never the only one

`populate` resolves a location rather than guessing it wherever the task allows. A dependent
object's room is its **root's** room — follow the chain of relations to the object nothing
else hangs off, and take that one's. "the mug in the office cabinet" puts the mug in the
office without the RSN ever being asked about mugs, and a chain of any depth works the same
way.

**Everything keeps a ranked fallback, including a stated room.** `candidates` is the order
the robot searches in, and it always ends with the RSN's full ranking:

| tier | what it assumes |
| --- | --- |
| 1 | where the task says the support is |
| 2 | where the support might be instead, if that room is ruled out |
| 3 | where this object itself tends to live — the tier that saves the task when the *relation* was wrong, not just the room |

Before this a stated room produced a **one-element** list, so a wrong statement was
unrecoverable: the object was simply unfindable and the plan died. A potato reported on the
countertop but actually in the fridge is only findable because the potato's own ranking is
in the list.

And a disconfirmed belief is **retracted**, not merely overridden later.
`WorldGraph.rule_out_room` removes the `room_inside` edge and records the room, so nothing
downstream keeps asserting a room the robot has already swept, and no search returns to it.
`test_fallback.py` checks all of this, including that the same search fails with a single
candidate.

## Stage 4 — planning and validation

The LLM (Qwen2.5-7B-Instruct by default) sees the action space, the scene graph, the rules,
and worked examples, and returns one plan. `planner.generate` is **single-shot on purpose**
— it asks once and surfaces the raw failure, so the checker can be measured against an
unaided model. Closing the loop is stage 4', and it is the question this project exists to
study rather than something the planner hides.

The parser tolerates numbering, bullets, markdown fences and prose, and repairs
malformations that carry unambiguous intent (`PLACE_ON_TOP.bed()`, `GRASP: mug`). Dropping
those silently made corrected plans look incomplete.

The validator replays the plan against a symbolic world model tracking what the robot
holds, where it is, what is open and what is switched on. Every refusal it can make, and
the `(kind, object)` pair it reports alongside the sentence:

| kind | raised by | what the plan did |
| --- | --- | --- |
| `arity` | any | `RELEASE(x)`, or a primitive with no argument |
| `room` | any | named a room where an object belongs |
| `unknown` | any | named something the pipeline never produced |
| `not_near` | all but `NAVIGATE_TO` | acted on something it had not driven to |
| `holding` | `GRASP` | grasped with a full hand |
| `empty_hand` | `PLACE_*` | placed with an empty one |
| `closed` | `GRASP` | reached into a shut container |
| `not_open` | `PLACE_INSIDE` | placed into a container it never opened |
| `not_graspable` | `GRASP` | tried to pick up fixed furniture |
| `no_door` | `OPEN`/`CLOSE` | opened something with no door |
| `no_switch` | `TOGGLE_*` | switched on something with no switch |
| `unknown_action` | — | not a primitive |

Two more are checked once the whole plan has run: the goal it was given, and whether
anything was **left open or switched on**. Those arrive together in one complaint, because
a plan can be wrong in both ways at once and telling it about one at a time spent two of
five attempts on a single message.

Warnings cover the non-fatal cases: opening what is already open, closing something never
opened, `RELEASE` with an empty hand, driving between rooms the topology says are not
adjacent.

**The line the machine draws is between knowing what a *kind* of thing is and knowing what
is true of *this house*.** Whether a fridge has a door is a fact about fridges, and a robot
that recognises one knows it; where the fridge is, and whether it is shut right now, are
things the belief graph has to guess. So `planner.OPENABLE`, `NOT_GRASPABLE` and
`TOGGLEABLE` are consulted, and the guess is never second-guessed.

That fact has to cut both ways or it is not knowledge. A container with a door must be
opened before anything comes out of it or goes into it; a bowl, a sink, an open-topped bin
has nothing to open, so `PLACE_INSIDE` needs no `OPEN` first **and** `OPEN` on it is
refused outright. Excusing one while permitting the other was the incoherent middle: it
claimed the machine could not know a bin has no lid at the moment it refused, and did know
at the moment it excused.

### NAVIGATE_TO takes an object, not a room

Planners write `NAVIGATE_TO(kitchen_0)` and mean "go to the kitchen". The machine used to
admit it by inventing a node called `kitchen_0` and navigating to that, so the plan
validated and then died on the simulator's very first step — **23 of the 27** plans that
passed validation and failed when driven.

It is refused now, with a complaint that names the room so the repair loop can rewrite the
step: *"'kitchen_0' is a room, not an object; NAVIGATE_TO takes the object you are about to
act on."* The simulator refuses it in the same words, because the two have to agree about
what a plan may say or a plan passes validation and dies downstream.

## Stage 4' — repair, then replanning from the checker's complaint

`planner.generate` asks once and surfaces the raw failure, deliberately. `replan.py` closes
the loop around it. One attempt is: **the model writes a plan, the machine validates and
mends it until it has nothing left to do, and only then is the model asked again** — about
what survived the mending rather than about what it wrote.

```
per attempt:  LLM writes a plan
              -> repair() validates, edits, re-validates, iterating until the plan
                 holds or the fault that remains has no derivable edit
              -> if it holds: accept
              -> else: complain about the MENDED plan, ask again      (up to 5)
```

That ordering is the whole gain. Mending only *after* the last attempt — which is what this
did first — spends every retry on faults the machine could have removed itself, so the model
is asked five times to insert a `NAVIGATE_TO` and never once about the thing that actually
defeats it. Measured over the hundred tasks, moving the repair inside the loop is worth **+9
tasks to the 4B and +3 to the 8B**, and it roughly **halves the LLM calls** — 3.54 to 1.63,
and 2.68 to 1.30. Seventy-three of a hundred tasks are then accepted on the first attempt.

### What the machine mends, and what it hands back

`repair.py` branches on the structured `fault`, never on the English:

| fault | edit |
| --- | --- |
| `not_near` | insert `NAVIGATE_TO(x)` |
| `closed` / `not_open` | insert `OPEN(container)` |
| `room` | delete the step |
| `not_graspable` | delete the step |
| `no_door` / `no_switch` | delete it **and its matching partner**, in one edit |
| left open / left on | `NAVIGATE_TO` + `CLOSE`/`TOGGLE_OFF`, appended at the end |
| `empty_hand` | the goal says what belongs there → insert `NAVIGATE_TO(x)` + `GRASP(x)` |
| `holding` | the goal says where the held thing was going → finish that errand first |

The first six are derivable from the graph alone. The last two are not — the machine knows
the hand is wrong but not what the plan *meant* — and they are the same mistake seen from
two sides: the model plans as though the robot had two hands, so it grasps twice before
placing, or places twice having grasped once. The **goal** is the only thing in the pipeline
that states where each object is going, and over the surviving failures of both models it
named the missing object in 28 of 32. Together they were 22 of the 8B's 23 remaining
planning failures.

`arity`, `unknown_action`, `unknown` and *the goal simply not being met* always go back to
the model. The first three are malformed plans and the last is a plan that runs and does the
wrong thing; both need intent the graph does not hold.

**The rules compose, so each stays small.** A `closed` fault inserts a bare `OPEN`; the next
round raises `not_near` on the container, which rule 1 answers; the round after that raises
`not_near` on whatever the failing step was reaching for. Two rules do what one three-action
splice used to, and the splice had to guess where the robot had been standing in order to
write the drive back.

**An edit is a hypothesis, and only the result is judged.** The loop runs freely, keeps the
best plan it has seen, and compares that against the plan it was handed:

```python
(applicable, goal_met, safe, how_far_it_got, -length)
```

Requiring each *edit* to improve the outcome looked right and stalled everything: inserting a
bare `OPEN` moves the plan sideways — same failing index, one action longer — until the next
round makes it count. `-length` stops a repair winning by padding. Replayed over the plans
the 4B had actually had refused: **19 of 39 made applicable, none made worse.**

The complaint then quotes the **mended** plan and says what was filled in — a model shown
steps it did not write, with the marked step number pointing into a plan it does not
recognise, has to work out whether it misremembers its own answer.

One scene graph serves both halves — `WorldGraph.from_scene_graph` over the RSN's output is
what the LLM is shown *and* what the plan is checked against — so the checker can only
complain about things the planner was told, which is what makes its complaints repairable.

```bash
python replan.py --scene Beechwood_0_int --task "take the potato from the countertop, \
    heat it in the oven, then put it on the breakfast table" --json plan.json
```

**The prompt has to state the rules the plan is judged by.** It did not, and that alone
accounted for every failure in the first end-to-end run. The action space was nine
one-liners — `GRASP(object) - Grasp an object` — with no preconditions in sight, so the
model wrote `PLACE_INSIDE(potato)` (passing the object rather than the container),
`PLACE_ON_TOP` with an empty hand, and `GRASP(potato)` while standing at the table. Every
one is a rule the machine enforces and the prompt never mentioned. `PRIMITIVES` now carries
`requires` and `effect` per action, rendered into the prompt:

```
  PLACE_INSIDE(object)  - Put down what is held, inside something
      requires: the robot is standing at the destination, the destination is already
                open, and the robot is holding something
      then:     what was held is inside the destination; the hand is empty
```

With that, the 13-step potato/oven plan came back **valid on the first attempt**, and ran
13/13 in the 2-D simulator.

**The complaint has three parts**, and the third is what the precondition specification
buys — a refusal the model can act on rather than a rejection:

```
Your previous attempt:

   1. NAVIGATE_TO(countertop)   ok
   2. GRASP(potato)   ok
   3. NAVIGATE_TO(breakfast_table)   ok
   4. GRASP(oven)   <-- REJECTED: already holding 'potato'; place or release it first

The plan was rejected at the marked step. Everything before it is fine; fix that step
and anything after it that depended on it.

Remember what GRASP(object) needs:
  requires: the hand is empty; the robot is standing at the object; if the object is
            inside a container, that container is open
  then:     the robot is holding it, and it travels with the robot
```

The whole plan is quoted back with the line marked, not just the error: a model handed
"step 4 was wrong" has to reconstruct what step 4 was. The `requires`/`effect` block is the
same specification the machine enforced, so the complaint and the rule are never out of
step. Quoting only the one-line doc was tried first and is much weaker — "Remember: GRASP -
Pick an object up" says nothing about why this GRASP failed.

There is a second branch for a plan that applies but does not do the task, which lists the
goal edges still missing at the end.

Four tasks, five attempts each:

| task | outcome |
| --- | --- |
| heat the potato in the oven, then put it on the breakfast table | accepted, attempt 1 |
| move the book from the coffee table to the bookcase | accepted, attempt 1 |
| put the apple inside the fridge | accepted, attempt 2 |
| put the plate in the dishwasher and turn it on | accepted, attempt 3 |
| take the bowl out of the fridge and leave it on the table | **no valid plan in 5** |

The bowl task turned out not to be a planning failure at all. Chasing its refusal led to
**stage 1**: `task_objects.extract` read "take the bowl **out of** the fridge and leave it
on the breakfast table" as `bowl ON_TOP breakfast_table` — the task's *goal* rather than
the bowl's *current* location. The scene graph then asserted the bowl already sat on the
table, so a plan that opens the fridge to get it was refused, correctly, against a world in
which the task was already done. The extraction prompt distinguishes the two carefully and
even has an "off the shelf" example, but its phrase list did not cover "out of". With that
added the task is accepted on attempt 2:

    NAVIGATE_TO(fridge) OPEN(fridge) GRASP(bowl) NAVIGATE_TO(breakfast_table)
    PLACE_ON_TOP(breakfast_table) RELEASE()

It is not fixed in general — "get the mug **out of** the dishwasher" is still read as
`mug ON_TOP counter` — and that failure is worth understanding: the resulting plan is
*accepted*, because against a graph that says the mug is already on the counter it is
internally valid. **The filter can only be as sound as the graph it is given**, and a
stage-1 error is invisible to every stage after it.

## One vocabulary, and the single place names are matched

Every stage after extraction compares object names with `==`. That is possible only because
the names entering the pipeline are the dataset's own, and it is worth the effort because
the alternative is what this codebase had for months: five components each inventing their
own way to bridge a gap none of them created.

```
task text  ──▶ extract() ─┐
                          ├──▶ one vocabulary ──▶ belief graph, machine, repair, goal check
task text  ──▶ parse_goal()┘                                    (all by equality)
                                                                        │
                                                   sim_eval.ground() ◀───┘
                                                   ONE match: our name → the real instance
```

**The instruction names the dataset's category.** A task that says "the bathroom cabinet"
while the ground truth is `bottom_cabinet` forces every downstream component to guess, and
`cabinet` is as good a name for `top_cabinet` as for `bottom_cabinet`. So the text says "the
bathroom bottom cabinet", and where a container has instances in several rooms the sentence
names the room too — "the playroom bottom cabinet", "the kitchen top cabinet". Without that
the RSN guesses which room, opens the wrong cabinet, and the object is unfindable: a camera
cannot see through a cupboard door, so room-by-room search can never recover it.

**Two aliases, resolved once.** `object_names.canonical` is a fixed two-entry table —
`trash_can → public_trash_can`, `dryer → clothes_dryer` — for the words nobody says aloud.
`extract()` and `parse_goal()` both apply it, so the object list and the goal agree even
though two different models produced them. It is deliberately *not* a lookup against
`data/vocab.json`: that file lists what the **scenes** contain, and tasks inject objects the
scenes do not have — `t_shirt` and `bath_towel` are absent from it — so canonicalising
against it would rewrite half the vocabulary and leave the other half alone.

**Ambiguity is left alone rather than guessed.** `canonical("cabinet")` returns `cabinet`
unchanged. If a plan says it, the machine refuses and the loop asks which was meant. A coin
flip would be worse, because a coin flip is silent.

**`WorldGraph.resolve` is equality.** A name that is not a node is a name the pipeline never
produced — the model invented it — and it is refused (`unknown`) rather than admitted.
Admitting it used to build a node for a cupboard that does not exist and let the plan fill
it, satisfying the checker while corresponding to nothing.

### The one match: `sim_eval.ground`

Category by exact name; **instance by the task's own relations, then the room**. Both the
plan and the goal are grounded *before the first action runs* — grounding the goal afterwards
asked a question the plan had already changed the answer to. A swap moves both objects off
their sources, so the relation the binding depends on was gone by the end, the lookup fell
through to the room and chose a different table: the robot put the clock exactly where it
was asked and all four swap tasks were scored as missing it.

```python
held = [rel for rel in graph["relations"] if rel["to"] == name]
for rel in held:                       # the task says the toothbrush is in the cabinet
    for _, support in world.truth.edges_of(edge, src=rel["from"]):
        if support in instances:       # so the cabinet holding it is the one meant
            return support
return next((i for i in instances if world.room_of(i) == believed), instances[0])
```

The room is only the RSN's guess where the task did not state one, and it is wrong often
enough to matter — it put a top cabinet in the bathroom when the mug's was in the child's
room, and the plan opened a cabinet in the wrong half of the house. The relation is not a
guess: the task says where the object is, the setup put it in one particular instance, so
that instance is the one the task is about. It also distinguishes cabinets the room cannot —
Wainscott's `bedroom_0` holds four `bottom_cabinet` instances.

A *wrong* relation costs nothing: the `support in instances` guard means the rule simply
does not fire, and the room decides as before.

## Stage 5 — execution in the simulator

`execute_plan.py` follows `solve_simple_task.py` exactly: same `tiago_primitives.yaml`
config, same environment/settle/controller sequence, same restricted
`load_object_categories`, executing through `apply_ref`. Two deviations, both load-bearing:

- **`env.reset()` after `og.Environment(...)`.** Without it the articulation view can be
  uninitialized and the first `get_joint_positions()` returns None.
- **Never resize the viewer camera.** Assigning `image_width`/`image_height` on a live
  `og.sim.viewer_camera`, or overriding `render.viewer_width/height`, reallocates render
  buffers and segfaults inside the renderer with no Python traceback. Leave it at the
  config's 1280x720.

**Do not override the GPU variables from `setup_behavior_env.sh`.** `OMNIGIBSON_GPU_ID=1`
is this project's configuration. `CUDA_VISIBLE_DEVICES` in particular breaks Isaac Sim's
renderer, which enumerates physical devices independently ("No device could be created").
If a run segfaults with an OOM, check for your own leftover processes before blaming
contention — `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`. A zombie run
can hold ~19 GB across all four cards.

### The files

| | |
| --- | --- |
| `task_objects.py` | stage 1 — extraction into the three classes, and the parser |
| `extraction_data.py` | generates labelled instructions; no scene involved |
| `extraction_prompts.py` | the prompt wordings that lost, kept as ablations of the one that won |
| `extraction_eval.py` | measures extraction on its own, per class |
| `finetune_extraction.py` | LoRA-tunes a small model on generated data |
| `scene_graph.py` | stage 2/3 — rooms, and resolving a location from what the task said |
| `planner.py` | stage 4 — the prompt, the parser, the affordance tables |
| `graph_machine.py` | the validator: preconditions, effects, goal, safety |
| `replan.py` | stage 4' — the repair loop |
| `floor_world.py` | the 2-D house: floor, rooms, objects, footprints |
| `sim2d.py` | the robot: A*, the wedge camera, the primitives |
| `sim_eval.py` | drives one plan in the simulator and judges it |
| `build_tasks.py` / `task_shapes.py` | the benchmark, and the shapes it is built from |
| `evaluate.py` | the three-arm experiment over all 100 tasks |
| `run_reference_sim.py` | the ceiling: can the reference plans themselves be driven? |

`data/tasks.json` is the benchmark and `data/reference-sim.json` is the evidence every task
in it can be finished. Trained adapters, rendered figures and run output are not tracked —
`finetune_extraction.py` rebuilds an adapter in 14 minutes and `extraction_data.py` rebuilds
any split byte-for-byte from its seed.

### Testing

Everything offline, in order of cost. Nothing here needs Isaac.

```bash
python check_names.py *.py                 # <1s  - names a function uses that nothing defines
python test_graph_machine.py               # ~2s  - the world graph and the precondition model
python build_tasks.py                      # ~20s - all 100 tasks, seven checks each
python test_sim2d.py                       # ~15s - the 2-D world, camera, search and actions
python ablate_plan.py                      # ~10s - which broken plans each model catches
python ablate_plan.py --fuzz 240           # ~20s - where the two models still differ
python test_pipeline.py                    # ~1s  - the parser and the flat validator
python test_stance_order.py                # ~2s  - which stance the map picks
```

| suite | checks |
| --- | --- |
| `test_graph_machine.py` | 95 — one per line of the precondition and effect tables, the robot node, the carried-object rules, and the scene-graph seed |
| `test_sim2d.py` | 72 — the loader's coordinate transform, A\*, the camera wedge, the search, every primitive, and the belief-vs-world audit |
| `test_pipeline.py` | 20 — plan parsing and the flat validator |
| `build_tasks.py` | 100 tasks × 7 checks |

Then one simulator run, which does need Isaac:

```bash
python test_primitives.py --scene house_single_floor --search --bev --video out.mp4
```

### Recording a bird's eye view

`--bev` puts the camera directly above the robot looking straight down, tracking its x/y
every captured frame; `--bev-height` (default 6 m) and `--bev-tilt` (default 0, a true plan
view) adjust it. `--robot-camera` records from the robot's own head camera instead. The
camera is repositioned rather than parented to the robot, deliberately: parenting would
inherit the base's yaw and spin the whole room every time the robot turned.

**The ceiling has to be hidden, and cannot be excluded at load time.** In
`interactive_traversable_scene`, `... or is_building_structure` overrides even
`not_load_object_categories`, and that covers ceilings, roof, walls and doors. Measured in
`house_single_floor` the roof reaches z=3.41 m, so a camera at 6 m films the top of it.
`--bev` sets `visible = False` on the 19 ceiling and roof parts instead — the right tool
anyway, since this is a rendering concern and removing them would change the physics and
the traversability the robot navigates by.

### Results

`house_single_floor`, the potato/plate plan:

```
16/16 plan actions succeeded
NAVIGATE_TO (potato)      PASS  779      TOGGLE_ON (oven)           PASS  100
GRASP (potato)            PASS  100      TOGGLE_OFF (oven)          PASS  100
NAVIGATE_TO (plate)       PASS  278      OPEN (oven, to take out)   PASS  150
PLACE_ON_TOP (plate)      PASS  150      GRASP (plate, from oven)   PASS  100
GRASP (plate)             PASS  100      CLOSE (oven, after)        PASS  150
NAVIGATE_TO (oven)        PASS  383      NAVIGATE_TO (table)        PASS  598
OPEN (oven)               PASS  150      PLACE_ON_TOP (table)       PASS  150
PLACE_INSIDE (oven)       PASS  150
CLOSE (oven)              PASS  150
```

Read a green row carefully. **Navigation is real** — the robot drives each route itself and
arrives within 0.00 m, level. **Manipulation is not**: a green `PLACE_ON_TOP` means the
object really is `OnTop`,
established by teleporting it there, not that an arm reached for it — which is why every
manipulation lands on a flat 100–150 steps while the navigation legs run with the distance
actually driven.

## The 2-D simulator

Everything above needs Isaac, a GPU and about seventeen minutes to answer a question that
is usually settled in the first thirty seconds — did the robot get there, and did it find
the thing. `floor_world.py` and `sim2d.py` run the same nine primitives on a grid instead,
in **about a second**, so the parts of the pipeline worth iterating on can be iterated on.

```bash
python floor_world.py --scene Beechwood_0_int                    # an empty house
python floor_world.py --scene Beechwood_0_int --categories oven countertop
python sim2d.py --scene Beechwood_0_int --demo                   # 16 actions, ~1 s
python sim2d.py --scene Beechwood_0_int --plan plan.json --gif figures/sim2d.gif
python test_sim2d.py                                             # 54 checks, ~15 s
```

It is the same 16-step potato/plate/oven plan the simulator run above executes:

```
 1. NAVIGATE_TO(potato)    ok  drove 9.9 m, at 1.14 m, saw 81 new, robot -> kitchen_0
 2. GRASP(potato)          ok  at 1.14 m, -kinematic(potato), held = potato
 ...
16. PLACE_ON_TOP(counter)  ok  at 1.14 m, on_top(plate, countertop_jveutp_0), held = none

16/16 actions succeeded, 15.8 m driven
audit: saw 17 of 22 objects; 12 edges agree, 0 believed but not true
```

### The camera is the robot's, not a round number

The 2-D camera is derived from the same intrinsics OmniGibson's `VisionSensor` ships, the
way `object_map.camera_fov` derives it from a loaded sensor:

```
h_fov = 2 * atan(horizontal_aperture / (2 * focal_length))
      = 2 * atan(20.995 / 34.0)  =  1.1064 rad  =  63.4 deg
```

It was 1.2 rad — the *fallback* those functions use when no camera object is available,
hardcoded here where a camera is always implied. The 2-D robot was seeing 5.4° wider than
the one in Isaac, which makes a simulated search easier than the real one by that much arc.

Correcting it exposed a second defect. `SCAN_HEADINGS` was fixed at 4, and four headings of
a 63.4° camera cover **254° of 360**, leaving four 26.6° blind wedges on the diagonals. A
ring of markers every 15° around a stopped robot came back with exactly the four at 45°,
135°, 225° and 315° missing — an object standing there was invisible to a robot that had
stopped and turned specifically to find it. The count is derived now,
`ceil(2*pi / CAMERA_FOV)` = 6, and the ring closes. Both properties are tests: one look
while driving sees only what is ahead, and a stop sees the whole ring.

The camera runs **while the robot moves** — a look every `OBSERVE_EVERY = 1.0` m along the
route, along the direction of travel. That is where most of the map comes from. On a task
whose plan names three objects, the robot finishes knowing 89 of the 104 things in the
house: 23 visible from the start pose, **60 learned while driving**, 6 from deliberate
scans. It is also why the second navigation to a room is direct.

Range is not a sensor limit — OmniGibson's default clipping is effectively unbounded. The
5 m is the distance beyond which `object_map.observe` stops trusting a detection, matched
here so the two searches explore at the same rate.

### The world is the floor plan

`FloorWorld` is built from the same three files the room graph in `figures/` is built
from, at 10 cm per cell:

| source | gives |
| --- | --- |
| `layout/floor_trav_no_door_0.png` | which cells are floor |
| `layout/floor_insseg_0.png` | which room each cell belongs to, named as `room_graph.py` names it |
| `room_graph.py` | which rooms adjoin — the `room_connect` edges |
| `json/<scene>_best.json` | the furniture: category, position, switch state |

**Everything is two-dimensional.** A position is `[x, y]`; there is no height, no extent
and no support surface, because none of the nine primitives is decided by one. An object
on a counter *is at* the counter, and the `on_top` edge is what says it is on it.

**The house starts empty and you load what the task needs**, the load policy
`execute_plan.py` and `test_primitives.py` apply before starting Isaac — restrict
`load_object_categories` to the structure plus the categories the plan names, and inject
the small objects. Here the default is stricter still: no furniture at all.

```python
world = FloorWorld.load("Rs_int", categories=["countertop", "oven"])   # like the real one
world = FloorWorld.load("Rs_int")                                      # an empty house
world = FloorWorld.load("Rs_int", categories=None)                     # everything
```

Structure — walls, floors, ceilings, doors, windows — is never loaded as objects whatever
is asked for. A plan cannot act on a wall, and 60 wall panels in the world graph bury the
handful of objects the run is about: on `Beechwood_0_int` the demo's graph went from **123
objects seen to 17**, and the graph panel from a hairball of 281 `next_to` edges to the
four things the plan touches.

**Objects go wherever you want them**, which is the point — BEHAVIOR scenes are
furniture-only, so anything to pick up has to be injected:

```python
world.add_object("potato", "potato", on_top="countertop_jveutp_0")   # on a support
world.add_object("mug", "mug", cell=(27, 80))                        # in a grid cell
world.add_object("spoon", "spoon", room="kitchen_0")                 # on free floor there
```

**The furniture stays in the floor raster even when it is not loaded as objects**, which
is what OmniGibson does — `load_object_categories` decides which objects exist, and the
traversability map is read from the file regardless. Keeping it is also what makes the
camera worth simulating. Running on the bare floor (`no_obj`) was tried: nothing occludes,
so `kitchen_0` reads **98% covered without the potato ever being seen**, and nothing
blocks, so the stance lands **0.00 m** from the potato — in the middle of the counter.

**Ground truth is a `WorldGraph`,** the same class the robot's belief uses. That is not
thrift: `sim.audit()` is then a set difference between two objects of the same type rather
than a translation between two representations.

### The doorways the raster leaves shut

A scene ships with its rooms in several regions of standable floor, and a plan that
crosses the house cannot run. Two causes, measured across all 51 scenes:

**The doors.** There is no door-opening primitive, so a shut interior door is not an
obstacle the robot could do anything about — it is one that silently partitions the house.
With the doors in, `house_single_floor` breaks into 10 pieces and `Beechwood_0_int` into
4. `floor_trav_no_door_0.png` is the default for that reason, and it is what "10 pieces"
meant.

**The thresholds.** That leaves **24 of the 51 scenes** still in pieces, and at every
broken doorway the gap in the *un-eroded* floor map is **0.2 m to 0.3 m** — two or three
cells. That is not a wall. It is the strip of non-floor where the door frame sits, and the
room graph, built from dilated floor-plan adjacency, calls those rooms adjacent and is
right to. `open_doorways` reconciles the two, and opens nothing anywhere else:

| | |
| --- | --- |
| a **sill** | the two rooms' floors do not join, and the gap is at most `MAX_SILL` (0.5 m) |
| a **pinch** | the floor joins them but the footprint erosion took the doorway |

Both are opened to `2 · radius + 0.2 m`, the width that survives the erosion that follows.
Nothing is opened where the room graph has no edge, where the sill is wider than 0.5 m, or
where the route would need more than 40 cells — those are walls, and they stay up. Each
pair gets at most 5 attempts, because carving changes the map: unbounded,
`Wainscott_0_int` cut 34 openings over ten passes and ended in the two regions it started
in.

```
Rs_int: 5 rooms, 7 connections, 4224 free cells (3188 standable)
opened 2 doorways the raster had shut:
  bathroom_0  <-> bedroom_0   0.30 m sill
  bathroom_0  <-> kitchen_0   0.20 m sill
```

**27 of 51 scenes → 47 of 51.** What is left is honest and is reported rather than carved
through: `Wainscott_0_int`, `hall_train_station`, `hotel_gym_spa` and `restaurant_brunch`
still have a wing behind a gap of 0.8 m or more, and two rooms across the dataset are too
small for the robot to stand in at all. Both entry points say so before a run:

```
warning: from living_room_0 the robot can reach kitchen_0, corridor_0, …
         but not sauna_0, locker_room_0
```

`open_doorways=False` gets the raster exactly as the dataset ships it.

### Ablation: drop one action and see who notices

`ablate_plan.py` starts from the plan above, removes exactly one action, and runs both
models. 13 leave-one-out cases plus 9 reorderings and substitutions, in **10 seconds**.

```bash
python ablate_plan.py                    # the potato/oven plan
python ablate_plan.py --plan plan.json   # a plan from pipeline.py
```

| dropped step | graph model | 2-D simulator |
| --- | --- | --- |
| `NAVIGATE_TO(countertop)` | inapplicable @1 | inapplicable @1 |
| `GRASP(potato)` | inapplicable @4 | inapplicable @4 |
| **`NAVIGATE_TO(oven)`** | **ok** | **inapplicable @3** |
| `OPEN(oven)` | inapplicable @4 | inapplicable @4 |
| `PLACE_INSIDE(oven)` | inapplicable @9 | inapplicable @9 |
| `CLOSE` / `TOGGLE_ON` / `TOGGLE_OFF` / `CLOSE` | ok | ok |
| `OPEN(oven)` before taking it out | inapplicable @9 | inapplicable @9 |
| `GRASP(potato)` from the oven | inapplicable @12 | inapplicable @12 |
| **`NAVIGATE_TO(breakfast_table)`** | **ok** | **inapplicable @12** |
| `PLACE_ON_TOP(breakfast_table)` | goal not met | goal not met |

**Three outcomes, and the third is not a failure.** Dropping `CLOSE` or either `TOGGLE`
leaves a plan that still applies and still reaches the goal, because the goal is
`on_top(potato, breakfast_table)` and says nothing about the oven having been on. That is
a fact about the goal, not a miss — the models are right and the goal is incomplete.
Nothing in the six edge types can express "was heated", which is a limit of the
representation worth stating rather than papering over.

**Where the 2-D simulator earns its keep.** Three of the 22 cases are caught only by it,
all the same root cause: `GraphMachine._require_here` asks whether the robot is *in the
object's room*, and the oven, the counter and the breakfast table are all in `kitchen_0`.
Drop the `NAVIGATE_TO` and the graph model still believes the robot is there. The
simulator measures 3.32 m and refuses. **Room-level "here" is too coarse for a kitchen**,
and that gap is exactly the layer a symbolic model has to assume.

**One gap neither closes.** `PLACE_INSIDE(countertop)` is not a thing — a countertop has
no inside — and the graph model allows it. The simulator refuses it, but on distance
rather than on containment, so it is caught for the wrong reason. Deciding what is a
container from a category list is the same guess `planner.py` warns about for openability,
so it is recorded here rather than patched.

### How far apart are the two models now?

`ablate_plan.py --fuzz N` builds plans by walking the graph model forward — at each step
random actions are tried until one is *accepted* — so what comes out is a plan the graph
model runs end to end. Then the simulator runs the same plan, and the reverse arm does the
same the other way round. Sampling actions uniformly instead is close to useless: almost
every random plan dies on its first step in both models for the same reason, and the run
reports 100% agreement having tested nothing.

240 plans per scene, ~5 actions each, before and after `near(x)` became an edge:

| | `Beechwood_0_int` | `Rs_int` |
| --- | --- | --- |
| graph-valid plans the simulator accepts, with `near` = same room | 190/240 (79%) | 201/240 (84%) |
| graph-valid plans the simulator accepts, with `near` = the `nearby` edge | **240/240** | **240/240** |
| plans the simulator accepts that the graph model refuses | 2 | 15 |

**The `nearby` edge closed the whole gap in the dangerous direction.** Every plan the graph
model now accepts, the simulator can drive — 480 of 480 across two scenes — and the 22-case
ablation went from 4 disagreements to none. Before, about a fifth of symbolically-valid
plans were undrivable, and every single one was the same mistake: the plan drove to one
object in the kitchen and then acted on another three metres away in the same kitchen.
A room was never fine enough to catch that; what the robot last drove to is.

**What is left is the model being conservative, which is the safe direction.** The
simulator measures 1.5 m, so it knows two objects a metre apart are both in reach; the
graph model only knows what the robot drove to and what was on or inside it. So it
sometimes demands a `NAVIGATE_TO` that geometry says is unnecessary — 2 plans in
`Beechwood_0_int`, 15 in the much smaller `Rs_int`. That costs a redundant step, where the
old failure cost a robot that could not carry out the plan.

That the symbolic preconditions agree at all is structural rather than lucky: `Sim2D`
**runs a `GraphMachine`** over ground truth, and supplies its `nearby` edges from measured
distance, so both models check the same preconditions with the same code and differ only
in where the edges came from.

### What is kept, and what is dropped

Dropped: the arm, contact, dynamics. `GRASP` welds a name to the hand, `PLACE` teleports it
onto a support, `OPEN` flips a flag. Kept, because dropping them makes the answers
meaningless:

**Navigation is A\* on the eroded grid.** Free floor eroded by the robot's footprint,
8-connected, no corner-cutting — a diagonal needs both of the orthogonal cells it passes
between, because the base would clip both. (`open_doorways` is the one caller that allows
it, to recognise two floors meeting at a single diagonal cell as the doorway it is.) A stance is the nearest standable cell to the object that
is in the robot's own connected region — the same funnel `nav_controller.py` runs, minus
CuRobo's 3-D check. The erosion radius is **0.35 m, not Tiago's 0.892 m**: the real map is
eroded harder because CuRobo checks the whole body against a counter lip the base would
drive under, and there is no body here. At 0.892 m in `Rs_int` the kitchen keeps 4
standable cells and the bathroom none — a house no plan can run in.

**The camera is a wedge.** 1.2 rad, 5 m, 121 rays, each stopped at the first wall, exactly
`object_map.observe_fov`. An object is revealed when the camera *reached the floor beside
it* — within 0.8 m, on the cells this look actually covered. Testing the object's own cell
would reveal nothing ever: an object sits inside its own footprint, which is precisely what
makes that cell untraversable.

**The search is `nav_controller`'s search**, with its constants: drive into the room, scan
`SCAN_HEADINGS = 4` headings, walk to the nearest frontier at least 0.6 m away, scan again,
stop at 95% coverage or 10 moves or no frontier left. One heading instead of four is not a
smaller version of this — it stalls at 21% coverage in `Rs_int`, shuffling between two cells
half a metre apart, because a 1.2 rad look never moves the frontier it is standing on.

**An action needs the robot to have driven to it.** Within 1.5 m, and the distance it acted
at is in every result line.

### One effect model, two graphs

The primitives' preconditions and effects are not reimplemented — `GraphMachine` already
knows them, so the simulator drives **two** of them:

```
physical gate      can the robot get there, is it close enough, does the thing open at all
    -> truth machine     GraphMachine over world.truth      what actually happens
    -> belief machine    GraphMachine over sim.graph        what the robot thinks happened
```

A disagreement between the two is reported rather than patched, because it is a real
divergence. The physical layer supplies exactly what the graph machine has to assume, and
two affordance checks the graph model cannot make: `OPEN` on a countertop and `TOGGLE_ON`
on a fridge are refused here, where `GraphMachine` sets a flag on any node it is given.

`sim2d_render.py` draws the run as the two panels `world_trace.py` renders for a simulator
run: the **object semantic map** on the left and the **world graph** on the right. The map
is the robot's, not the floor plan — cells the camera never reached stay grey, what it saw
is drawn in its room's colour, and a room it has not entered is a grey hole in the middle
of the picture. Coverage is replayed from the recorded poses rather than stored, so a
frame in the trace is a dozen numbers instead of a grid; the frames therefore have to
record every heading a stop scanned, not just the one the robot ended up facing, or three
quarters of each scan goes missing and objects appear on unmapped floor.

**Both panels are drawn about the task, not about everything the robot knows.** The robot
records every object it sees — that is what makes "I searched the kitchen" mean something
— but searching a kitchen for a potato turns up eight countertops, and drawing all eight
buries the four objects the task is about. `focus` is the plan's own objects; the pictures
show those plus whatever the graph relates **directly** to one of them, so an object joins
the figure at the moment the robot relates it to the task. One hop, deliberately:
following the relations transitively walks the whole counter run back in, because each
countertop is `next_to` the next one. On the run above that is **6 of 12 objects** drawn,
and the audit still covers all 12.

### A task, end to end

`Beechwood_0_int`, *"take the potato from the countertop, heat it in the oven, then put it
on the breakfast table"*, all five stages and no Isaac:

```
stage 1  task text -> potato, countertop, oven, breakfast_table
stage 3  RSN       -> kitchen_0: countertop 0.99, oven 0.77, breakfast_table 0.56
                      potato is on top of the countertop (certain, from the task)
stage 4  Qwen2.5-7B proposes 9 steps; the validator REJECTS them:
             step 5 PLACE_INSIDE(potato): cannot place 'potato' onto itself
stage 5' the 2-D simulator runs that plan anyway and stops at the same step
```

The corrected 13-step plan, on the same world:

```
 1. NAVIGATE_TO(countertop_jveutp_0)  ok  drove 9.9 m, at 1.14 m, saw 11 new
 2. GRASP(potato)                     ok  at 1.14 m, -kinematic(potato), held = potato
 3. NAVIGATE_TO(oven_wuinhm_0)        ok  drove 3.0 m, at 0.59 m, carried potato
 4-11 OPEN / PLACE_INSIDE / CLOSE / TOGGLE_ON / TOGGLE_OFF / OPEN / GRASP / CLOSE   ok
12. NAVIGATE_TO(breakfast_table_uhrsex_0)  ok  drove 4.4 m, at 0.89 m, carried potato
13. PLACE_ON_TOP(breakfast_table_uhrsex_0) ok  on_top(potato, breakfast_table_uhrsex_0)

13/13 actions succeeded, 17.2 m driven
audit: saw 12 of 14 objects; 6 edges agree, 0 believed but not true
```

`graph_machine.check` and the simulator agree on both plans — inapplicable at the same
step for the LLM's, applicable and goal-met for the corrected one — which is the point of
having both: the millisecond check is trustworthy because the two-second one confirms it.
`figures/sim2d_potato_oven.gif` is the run.

**Grounding is a real step, not bookkeeping.** The planner names categories because that is
what the RSN predicts, and `Beechwood_0_int` has **eight countertops and four breakfast
tables**; `stage_plan` picks the instance in the room the graph predicts, exactly as
`execute_plan.ground_plan` does. So is honouring the graph's relations: the RSN graph says
`potato ON_TOP countertop`, and dropping the potato on the floor of the right room instead
put it 2.56 m from the counter the plan drives to, so `GRASP` failed on reach for a plan
whose real fault was three steps later.

### What it does not answer

Whether the arm can reach, whether CuRobo can route, whether an object sampled against the
back wall of a counter is graspable. Those need the real thing. What it does answer is
whether the plan is drivable, whether the search finds what the plan assumes, and whether
the graph the robot ends up with matches the world — which is what fails first.

### Driving the plan, not replaying it

`evaluate.py --simulate` adds a third arm: the accepted plan is **driven** in `sim_eval.py`
rather than replayed symbolically. `GraphMachine` has no camera and no floor, so a
`NAVIGATE_TO` always succeeds there and a plan that depends on finding a mug in the wrong
room scores the same as one that does not. The gap between the two arms is the cost of
perception.

**The world is ground truth and the belief is not**, and that separation is easy to lose.
`sim2d.stage_plan` injects each object into the room the *belief* names, which is right for
a demo and fatal for a measurement: a wrong belief then places the object where it expects
and can never be caught. In `sim_eval` the objects go where `task["spawn"]` says — with the
instance chosen by `build_tasks.seed_graph`, so the house is identical no matter which
belief is under test — and the belief comes from extraction and the RSN.

Running the benchmark's **own reference plans** through it found nothing about planning and
a great deal about the simulator. Every task in `data/tasks.json` was validated by the graph
machine when it was built, so symbolically all 100 succeed. Driven, 82 did. The other 18
were all the simulator's fault, and the ceiling is now **100/100** — proof, recorded in
`data/reference-sim.json`, that a successful plan exists for every task.

| what was wrong | tasks |
| --- | --- |
| objects placed at a support's geometric *centre* — the robot could set a clock on a sofa it was standing beside and never pick it up again | 12 |
| the room sweep scoped strictly to the room mask, when all eight vantage points for one television were just outside it | 2 |
| the sweep giving up while reachable floor was still unseen | 2 |
| two tasks that were physically impossible | 2 |

**Objects are points; furniture is not.** `FloorWorld.footprint` floods the untraversable
blob around an object, bounded at 1.2 m so a sofa against a wall does not annex the wall,
and `reachable_point_on` returns the edge of it that touches free floor. Placing now happens
at the cell nearest the *robot* — arm's length, where it is standing — and `distance_to` and
`can_see` measure to the near edge rather than the middle. An object at the centre of a bed
is 0.90 m from any standable floor, past the camera's 0.8 m sight margin, so it was
literally invisible however well the robot searched; at the near edge it is 0.10 m.

Two tasks were dropped rather than papered over. One needed a utility room with **0 of 298**
standable cells reachable — the washer is behind a doorway too narrow for Tiago. The other
named the scene's only armchair, 2.30 m from the nearest floor the robot can stand on,
against a 1.5 m arm. Both were replaced with tasks built from the same shapes on reachable
furniture, and `build_tasks.verify()` now **requires the reference plan to run in the
simulator**, so a task that cannot be done cannot be added.

## The benchmark: 100 tasks over 10 scenes

`data/tasks.json`, built by `build_tasks.py` from `tasks.py` and `task_shapes.py`. Ten
scenes, ten tasks each, **11 to 17 actions apiece** (mean 12.4). Each task carries what the
pipeline would otherwise have to be trusted on:

| field | what it is |
| --- | --- |
| `task` | the instruction — the only thing an LLM is given |
| `extraction` | the ground-truth stage-1 answer, split into stated and unstated locations |
| `spawn` | the small objects the task needs, and where they start |
| `goal` | what must hold at the end, as graph edges or node state |
| `plan` | a reference sequence that achieves it |

```bash
python build_tasks.py                 # verify all 100 and write data/tasks.json
python build_tasks.py --scene Rs_int --verbose
```

### A task has to be sayable, reachable, and mean one thing

Three classes of defect were found by running the benchmark rather than reading it, and each
one made tasks that no plan could complete:

**An ambiguous container is an impossible task.** Eight tasks hid an object inside a
container the *sentence* did not locate — "take the mug out of the top cabinet" where the
scene has three, in three rooms. The `rooms` metadata pinned the right one, but metadata is
not something the robot reads. The RSN then guesses, the robot opens the wrong cabinet, and
because a camera cannot see through a cupboard door the room-by-room search can never
recover: it swept all eight rooms, *including the right one*, and found nothing. The
sentences now say which — "the playroom bottom cabinet", "the bed in the child's room".

**A room with no route is an impossible task.** Beechwood_0_int's only washer and dryer are
both in a `utility_room_0` that has no path from the rest of the house, so a laundry task
cannot be set in that scene at all — no choice of cabinet helps. Pomaria_0_int's single
armchair is routable but every stance around it is beyond the 1.5 m arm. Both tasks were
replaced. The spawner had been *masking* the first one: unable to reach the utility-room
cabinet, it silently fell back to any reachable cabinet in the house and dropped the shirt in
the private office, so the failure surfaced three steps later as a confusing search error.

**A task must not contradict itself.** `swap_places` spawned both objects `ON_TOP` of their
sources, including "swap the detergent bottle **in** the utility room bottom cabinet" — the
bottle sat on the cabinet's roof and the goal demanded the soap end up there too. The shape
now reads the relation off the source: a source with a door holds its object *inside*, the
swap puts the other object inside it, and the goal and plan say so. The model's plan, which
put the soap in the cabinet, had been marked wrong for doing what the words said.

**And where an object sits on its support matters.** Placing at the footprint edge nearest
the support's *centre* keeps the object on the furniture but says nothing about the side the
robot approaches from: on a bed that was 2.20 m from the robot's stance, past the 1.5 m arm,
so `NAVIGATE_TO(bed)` could never make `GRASP(cardstock)` work. Spawns are now anchored to a
stance the robot can actually route to — 2.20 m became 0.30 m.

**Nothing is taken on trust.** Every reference plan is replayed through `GraphMachine`, and
the dataset is not written unless all of them pass seven checks. Each one caught something:

| check | what it caught |
| --- | --- |
| every action applies | `OPEN(public_trash_can)` and `OPEN(cedar_chest)` — neither is in `planner.OPENABLE` |
| the goal is reached | — |
| the goal is **not already true** | "carry the pillow from one bed to another": objects are named by category, both beds are `bed`, so the empty plan satisfied it |
| the plan leaves nothing open or on | the reshaped `stack_then_store`, which opened a cupboard it never shut |
| **the goal requires** putting it back | not enough that the reference plan tidies up — an unsafe plan must fail the goal |
| furniture exists in *this scene*, in the room claimed | `top_cabinet` claimed in `childs_room_1`; it is in `childs_room_0` |
| injected objects are loadable BEHAVIOR categories, and do not clash with the scene | **71 of 101** invented names — `pie`, `cutlery`, `cue_ball` — that OmniGibson cannot instantiate; and 3 that duplicate something the scene already has |

That last row is worth dwelling on. Asked what each scene contains that a robot could pick
up, the answer across all ten is **three objects**: `Rs_int`'s laptop, `Wainscott_0_int`'s
coffee maker, and a garden fence. BEHAVIOR scenes are furniture-only, so every task injects
what it needs — and what it may inject is limited by what the object dataset ships.

**Plans come from shapes, not from hand-writing**, so the length is a property of the shape
and cannot drift while someone edits a plan:

| shape | actions | the job |
| --- | --- | --- |
| `laundry_cycle` | 17 | wash it, take it out, dry it, leave both machines off and shut |
| `fetch_heat_serve` | 15 | out of the fridge, heated, served |
| `load_and_run` | 14 | two things into an appliance, run it, switch it off |
| `heat_and_serve` | 13 | in, run, out, set down elsewhere |
| `two_into_container`, `move_three`, `unload_two`, `swap_places` | 12 | |
| `stack_then_store` | 11 | out of a cupboard, onto a tray, and the pair put away |
| `carry_two_and_switch` | 11 | two things moved, then something switched on and off |

Two of them lean on things a flat model cannot express: `stack_then_store` needs the rider
to travel with the tray when it is grasped, and `swap_places` needs a spare surface because
the robot has one hand.

**Everything a plan opens it shuts, and everything it switches on it switches off** — and
the *goal says so*. 65 tasks open something and all 65 require it shut; 34 switch something
on and all 34 require it off. No goal anywhere asks for something to be left running.

### The extraction ground truth is derived, not written

`uncertain` is every object the plan acts on or the task injects, minus the ones whose
starting place the instruction states; `dependent` is those, and an injected object counts
as `dependent` **exactly when the instruction names the thing it starts in or on** — "take
the fruitcake out of the fridge" states a location, "put the mug away in the cabinet" does
not.

Deriving it removed a whole class of unfair scoring. Written by hand, **57 of 100** named a
worktop the plan passes over and the instruction never mentions, so an extractor would have
been marked wrong for not inventing objects nobody asked for — a false ~57% extraction
failure rate, which is a number this experiment exists to measure. `build_tasks.py` now
refuses a task whose ground truth names something the text does not.

### The relations had to be audited too

`mentions()` refuses a task whose ground truth names an object the text does not. That check
was never applied to the *relations*, and 20 of them came from the scene's spawn
configuration rather than from the sentence:

```
bring the sock, the t-shirt and the bath towel to the hamper in the bedroom
   labelled: sock ON_TOP bed        <- the instruction never says where the sock is
```

An extractor reads text. A relation that exists only in the spawn config cannot be
extracted from the sentence, and scoring against it measures clairvoyance rather than
extraction — it was understating relation recall by about 16%. Those objects are now
`uncertain`, which is the honest label: still required, location unknown.

This removes no obligation from the model. `uncertain` **is** the "I don't know where it is"
answer, and the split between the classes is itself what gets graded: omitting a relation
the task did state costs recall, and inventing one it did not costs precision.

The stated-room labels needed the same care. Two false-positive patterns had to be
excluded — "the folder **away in the** office cabinet" is not the folder being in the
office, and "in the office **bottom cabinet**" is the room qualifying the cabinet rather
than a phrase about where something is — leaving 44 room facts, every one verified adjacent
to its object in the text.

## The experiment: does checking the plan help?

`evaluate.py` runs the whole pipeline over the benchmark. **Both arms come from one run per
task**, which is what makes the comparison exact rather than approximate:

| arm | what it is |
| --- | --- |
| without validation | the model's first answer, kept whatever it says |
| with validation | `replan.py`'s loop — mend what the machine can, hand back what it cannot, ask again, up to 5 times |

`--repair-at` selects where the machine may edit: `loop` (the default — inside every
attempt), `end` (only the plan the attempts settled on), or `off`. The last two are
ablations; measured, `loop` beats `end` by 9 tasks on the 4B and 3 on the 8B while halving
the LLM calls.

Decoding is deterministic, so the first plan is the *same plan* in both arms; they differ
only in whether anything is done about a bad one. Stage 1 and stage 3 are shared.

```bash
HF_HOME=/mnt/check/ruiyangw/hf_cache python evaluate.py \
    --models Qwen/Qwen3-4B Qwen/Qwen3-8B --json data/evaluation.json
```

**The pipeline never sees ground truth.** It gets the instruction, its own extraction, and
the RSN's guesses. **Scoring uses ground truth** — a plan valid in the RSN's imagined house
and impossible in the real one is a failure, and scoring against the RSN's own guess would
hide exactly the error the filter exists to catch.

**The dataset's reference plan takes no part in scoring at all**, and its extraction answer
only labels a failure after the fact. A run whose extraction differs from the reference and
whose plan works anyway is a success, because it is one. Two leaks had to be closed to make
that true: the verdict was being overridden by the ground-truth extraction, and the scoring
world contained only the furniture the reference named — so a plan routing through a
different real cupboard failed for touching furniture that exists.

### Where the errors come from

Failures cascade — an object nobody extracted is one the RSN cannot place and the planner
cannot use — so each is attributed to the stage that **caused** it:

| bucket | what it means |
| --- | --- |
| `task_objects extraction` | the plan wanted something stage 1 never surfaced, or acted on a location stage 1 misread |
| `RSN could not place it` | absent from the graph the planner was shown, so unnameable |
| `LLM plan invalid` | inapplicable against the real world, or nothing parsed |
| `LLM plan valid but does not do the task` | every action applies, goal unmet |
| `goal model read the task wrong` | the loop accepted a plan that met the goal stage 1' predicted, and that goal was not the real one |
| `left unsafe` | goal met, something left open or switched on |

**"Caused" is the word doing the work, and it used to be wrong.** The rule was "blame stage 1
if stage 1 differed from the reference at all", which is a correlation, not a cause. Measured
on the 8B, stage 1 differed on 11 tasks and **missed an object on none of them** — every one
of those failures actually died on a planner error the plan's own words show: navigating to a
room, grasping while already holding, acting on something it had not driven to. Extraction had
differed, so extraction got the name, and the largest failure class was undercounted by a
fifth.

Extraction can only *cause* a failure two ways: the plan wanted an object stage 1 never
surfaced, so the planner could not name it; or stage 1 read a location the task did not state,
the graph took a wrong edge from it, and the step that failed is the object whose location was
misread. Anything else is the planner's.

**A wrong RSN room is not a failure.** The RSN returns a ranking and the search layer works
down it: a room searched and ruled out costs metres, not the task, and nothing re-plans
because the instruction only ever said `NAVIGATE_TO(potato)`. Only an object the RSN could
not place *at all* is unrecoverable.

### What it answers

Tasks solved out of 100, scored against ground truth. **These are the symbolic arm**, and
each row is a separate finding rather than a tuning step:

| | Qwen3-4B | Qwen3-8B |
| --- | --- | --- |
| without validation | 16 | 17 |
| with validation, preconditions only | 25 | 37 |
| with validation, preconditions + safety | 33 | 49 |
| + extraction naming fixed | 37 | 59 |
| + fine-tuned extractor | 42 | **62** |

**Checking the plan is worth more than trebling the model.** An 8B unchecked solves 17; a
4B with the checker solves 42. The loop repaired 23 tasks for the 4B and 41 for the 8B and
made none worse in either — a rejected plan that cannot be repaired is still refused, so
the arm can only gain.

Validating preconditions alone let a plan pass that satisfied every precondition and still
did not do the task, or left the oven on; feeding the goal and safety check back as a
complaint took the 8B from 37 to 49. Fixing the naming problem took it to 59, and replacing
the prompted extractor with the fine-tuned one to 62.

**The progression above predates the simulator work, the repair pass, and the benchmark
fixes, and is kept only for the shape of the argument.** It was measured before
`NAVIGATE_TO` stopped accepting rooms — alone 23 of 27 plans that validated and then failed
when driven — before the mechanical repair existed, and before the benchmark's ambiguous
containers and unreachable rooms were found. Every one of those moved the number. The
current arms are being re-measured; quoting the old driven figures would be quoting a camera
model that has changed twice and a task set that has changed since.

### Running it

Two things about the models are worth recording because either would have wrecked a run
silently. **Qwen3's chat template enables `<think>` reasoning by default** — left on, the
model spends the whole token budget reasoning, the reply reaching `parse_plan` has no
actions in it, and every task scores `unparsed` as though the model could not plan.
`planner` passes `enable_thinking=False`, withdrawn for templates that do not know it, and
strips a stray `</think>` prefix as a backstop. Decoding is greedy on every attempt. What changes between them is the
prompt: `repair_prompt` quotes the failed plan back with the offending step marked, so the
model is answering a different question each round.

Model weights live in `/mnt/check/ruiyangw/hf_cache` rather than `~/.cache`, which was at
98%. Runs need `HF_HOME` set to it. `evaluate.py --models` frees the GPU between models,
because an 8B in fp16 is ~16 GB and two will not share a card.

## Choosing a scene, and setting it up

Which furniture a run uses is decided **offline, before Isaac starts**, by
`scene_setup.py` — ranking it at runtime cost a full simulator startup just to find out the
chosen surface was unusable.

```bash
python scene_setup.py Beechwood_0_int     # one scene
python scene_setup.py --all               # every scene in the dataset
```

It reads `layout/floor_trav_0.png`, erodes it by the robot's footprint, and measures two
things per candidate object:

| measure | why it decides usability |
| --- | --- |
| **standing room** | free floor in the annulus the robot must stand in — 0.8–1.6 m for appliances, 0.5–1.3 m for surfaces. Zero means the primitive cannot run there at all. |
| **floor component** | which connected region of floor the object sits beside. Doorways narrower than the eroded robot split a scene into pieces; an apple on the far side of one can never be carried to the fridge. |

A support is accepted only if it shares the fridge's component and holds up across a sweep
of erosion radii (0.35/0.50/0.60 m), since the exact footprint needs a loaded robot.
Component labels are renumbered per radius, so the fridge's component is re-derived at each
one rather than compared across them.

The result is a fixed table in `test_primitives.py`; scenes absent from it have no usable
support, `Rs_int` among them — its fridge sits in a different floor component from the
robot's spawn.

```python
SCENE_SETUP = {
    "Beechwood_0_int": {
        "support": "countertop_tpuwys_0", "support_category": "countertop",
        "fridge": "fridge_dszchb_0", "oven": "oven_wuinhm_0",
    },
    ...
}
```

**Load only what the scene needs**, derived from that table rather than fixed:

```python
["floors", "walls", "fridge", "oven", setup["support_category"]]
```

Beechwood loads 85 objects this way, well under CuRobo's hardcoded 2048-mesh collision
cache. Overrunning it corrupts GPU memory and every primitive dies with `CUDA error: an
illegal memory access was encountered`, which is how a full load of a large scene fails.

**Objects the test needs are injected, not hunted for.** `INJECTED` spawns each object and
drops it onto the named furniture:

```python
INJECTED = [
    {"name": "apple", "category": "apple",      "model": "agveuv", "on": "support"},
    {"name": "pan",   "category": "frying_pan", "model": "cprjvq", "on": "support",
     "lateral": 0.35},
]
```

`OnTop.set_value` samples a valid resting pose, then the object is slid to the edge of the
surface that faces open floor, 15 cm in from the lip — an apple sampled against the back
wall of a counter is one the robot can stand beside but not reach across to.

### What can actually be toggled

`TOGGLE_ON` / `TOGGLE_OFF` act on the `ToggledOn` object state, and an object only has that
state if it declares the **`toggleable` ability**. This is not baked into the model asset —
grep every `misc/metadata.json` in the dataset and *no* model ships a `togglebutton`
meta-link. That meta-link only matters for *physical* toggling (fingers overlapping the
button for `CAN_TOGGLE_STEPS` = 5 consecutive steps), which is the path OmniGibson disabled
with `NotImplementedError`. Setting the state directly bypasses it.

An object is toggleable when the scene or the object config says so — the pattern
OmniGibson's own `examples/wip/heat_source_or_sink_demo.py` uses. **Objects loaded from a
scene** carry the ability if their category does: in `Beechwood_0_int` the oven and
microwave report `ToggledOn`, the fridge does not. **Objects injected into a scene** get
nothing unless asked, which is why `test_primitives.py` declares
`abilities: {"openable": {}, "toggleable": {}}` explicitly.

What `ToggledOn` is wired to downstream, from `object_states/factory.py`:

| paired ability | effect of toggling on |
| --- | --- |
| `heatSource` with `requires_toggled_on` | stove/oven becomes an active heat source — food cooks, fire visuals appear |
| `particleApplier` / `particleSource` | a sink or spray starts emitting particles |
| `particleRemover` / `particleSink` | a drain or vacuum starts removing them |

**Lights are not `ToggledOn`.** Lamps (`table_lamp`, `floor_lamp`, `light_bulb`, …) are
`DatasetObject`s whose USD carries `Light` prims and whose metadata shows only a
`base_link`; brightness lives in `LightObject.intensity`, not in the action space. To make
a lamp respond to `TOGGLE_ON`, give it `abilities: {"toggleable": {}}` **and** drive the
intensity from the state yourself — the primitive alone flips a boolean nothing reads.

### Grounding

The planner names *categories* (`laptop`) because that is what the RSN predicts; the
simulator holds *instances* (`laptop_nvulcs_0`). Room ids ground via a proxy object the
scene graph placed in that room, since the action set has no room-level navigation
primitive:

```
grounded living_room_0 -> via coffee_table_fqluyq_0 (room proxy)
grounded laptop        -> laptop_nvulcs_0 (laptop)
```

Objects with no instance are reported UNGROUNDED rather than skipped silently. That gap
between "the RSN believes it is here" and "it is actually here" is the signal worth
measuring.

## The primitive backend

`primitive_patches.build` returns a controller in which all nine primitives work.

| primitive | how |
| --- | --- |
| NAVIGATE_TO | `_navigate_near` — a real drive over the traversability map |
| GRASP / PLACE_ON_TOP / PLACE_INSIDE / RELEASE | symbolic — weld to the gripper, teleport to a chosen pose |
| OPEN / CLOSE / TOGGLE_ON / TOGGLE_OFF | symbolic — `set_value` |

**Everything is symbolic except the driving.** `SymbolicSemanticActionPrimitives`
implements all nine by changing object state. The physical path,
`StarterSemanticActionPrimitives`, was removed: four of its nine raise a bare
`NotImplementedError` (`_open_or_close`, `_toggle`), it skips its own OPEN tests with
`reason="primitives are broken"`, and the five that do exist
were unreliable enough — a sticky grasp slips off a thin object, the planner cannot route
the last few centimetres — that every attempt ended up rescued symbolically anyway, at
twenty minutes a step. What stays real is the navigation, the half of the problem a plan
can actually get wrong.

The symbolic class subclasses the starter one, so all the motion machinery is inherited. It
passes `skip_curobo_initilization=True`, leaving `_motion_generator` as `None`; `build()`
constructs one afterwards, because the stance search needs it to collision-check candidates.

**Use `primitive_set()`, never the starter enum.** The two sets declare the same names in a
different order, so a name carries a different integer in each — `NAVIGATE_TO` is 6 in the
starter set and 13 in the symbolic one, where 6 is `TOGGLE_ON`. They are `IntEnum`s, so a
member of the wrong set hashes equal to whatever shares its value and the dispatch table
returns *a* primitive rather than raising. Measured: asking for `NAVIGATE_TO` with a starter
member ran `TOGGLE_ON`, silently.

**Primitives are atomic, and a plan that skips NAVIGATE_TO fails.** The state changes do
not navigate; getting the robot there is a separate plan step, so the failing step is the
one reported. Every primitive that takes an object calls `_require_near` first:

```
PRE_CONDITION_ERROR: GRASP needs the robot at the object - no NAVIGATE_TO ran
{'object': 'apple', 'distance': 5.7, 'limit': 3.1}
```

Without it such a plan would quietly pass, since the symbolic primitives only write object
state and would work from another room. The rule is that a primitive may adjust its own
footing but may not travel: the limit is the object's half-diagonal plus 3 m.

**Carrying something does not block OPEN/CLOSE/TOGGLE.** Upstream refuses outright —
"Cannot open or close an object while holding an object" — which rules out the ordinary way
to do this task: carry the plate to the oven, open the oven, put the plate in. Tiago has
two arms and the alternative is setting the plate on the floor mid-plan, so
`_hand_allowed_full()` makes that one call see an empty hand. It is a context manager scoped
to the borrowed `_open_or_close` / `_toggle` generators and nothing else — **GRASP still
refuses a second object**, and the placement primitives, which genuinely need to know what
is held, see the true state.

### The arm never moves

**The arm has one configuration**, `tucked_default_joint_pos`, held or empty. `_hold_pose`
returns it unconditionally and every drive waypoint commands it, so the arm is folded
against the body for the whole task and the map is eroded by one radius. Nothing poses it
because nothing needs to: every primitive except NAVIGATE_TO teleports its object into the
post-condition state, so the arm's configuration has no bearing on whether a manipulation
succeeds.

**A carried object is welded level and taken out of the physics.** `_grasp` sets it to zero
roll and pitch — keeping its yaw so it is not spun on the spot — welds it to the
end-effector, and sets `visual_only`, which removes its gravity and its collisions.
`_release` puts it back. Because the arm never moves afterwards and the base only rotates
about the vertical, an object welded flat is still flat at the far end of the route.

`visual_only` is the load-bearing part, and it is what a symbolically held object should be:
the primitives teleport it into place, so nothing should ever push it. A welded object is
part of the arm's rigid chain, so any contact force on it becomes joint torque — and **these
joints are not clamped at their limits**, so once one starts turning nothing stops it.
Measured, `arm_left_5` reached +32 rad against a ±2.094 limit and the arm turned through
several revolutions.

| carrying a plate | tuck residual | plate orientation |
| --- | --- | --- |
| welded, still physical | 19.5 / 33.7 / 5.9 rad | +129°, −172°, +43° |
| welded, `visual_only` | **0.004–0.015 rad** | **flat (±2°)** |

Two traps from an earlier attempt to fix this by posing the arm instead: **commanding a
measured joint position ratchets** (read a drifting joint, command where it is, and it
walks away), and **the contact point given to `_establish_grasp` must be the object's own
position**, or the weld is anchored away from the body it welds and drives the wrist.

Two consequences, each found by a run. A `visual_only` object has **no collision mesh**, so
handing it to CuRobo as an attached body raises `'NoneType' object has no attribute
'get_trimesh_mesh'` — it is skipped instead, since something that cannot collide does not
change where the robot fits. And `OnTop` is **contact-based**, so the object has to be
physical again before the check, which is why `_place_with_predicate` is reimplemented
rather than wrapped.

### Placing

**`Inside` and `OnTop` are different tests, and the primitive has to treat them
differently.**

| | what it checks | so the object must be |
| --- | --- | --- |
| `OnTop` | contact | **physical** for the check |
| `Inside` | AABB containment + `check_points_in_volume` on a `fillable` meta-link | only in the right **place** |

Upstream's order is sample → release → teleport → settle → check, which leaves the object
physical while it is being moved, so gravity reaches it before it has arrived. Here it
stays out of the physics until it has arrived, and when it is handed back depends on the
predicate: **before** the settle for `OnTop`, which needs contact, and **after** the check
for `Inside`, which does not.

`PLACE_INSIDE` wants the object's centre inside a **fillable meta-link volume**, so
`_shelf_pose` aims there. Aiming anywhere geometrically sensible — on the open door, on the
rack, at the cavity centre — fails, because none of those is what the predicate tests.
`metadata.json` cannot tell you whether a model has such a volume (a bowl and a bucket both
list no meta links, because they are built at load time), so read the loaded model.

**A placement has to look sensible, not just satisfy the predicate.** Upstream samples by
ray-casting down onto the target and returns the *first* pose that fits, which is a uniform
draw over the whole surface — on the video the plate was set down at the far end of the
coffee table, out of reach of the robot standing at its near edge. `_near_pose` draws
`PLACE_CANDIDATES` (12) and keeps the one nearest the robot. Two things it deliberately
does not do:

| rejected | why |
| --- | --- |
| clamp the robot's position into the target's bounding box | `aabb` is world-axis-aligned, so for any rotated piece of furniture it overshoots the real top face and the near edge it would aim at can be off the surface entirely. The sampler already ray-casts the true geometry. |
| pass `near_poses` / `near_poses_threshold` | upstream has them, but they *reject* anything past the cutoff and raise when nothing survives — a tight threshold fails where taking the first pose would have worked. Ranking N draws cannot lose to taking one. |

`PLACE_INSIDE` is unaffected: a fillable volume is small enough that any point in it is as
good as any other, so it still aims at the centre.

## Opening objects: accounting for the door swing

An open door occupies floor. Park in it and the door either hits the robot or cannot open
at all, so the swept area is an obstacle **before** the robot drives there — the stance has
to be valid for the object *after* it opens. Only the target object's swing is blocked:
blocking every openable thing also blocks the room doors, which is how the robot gets
between rooms, and measured that severed all 11 doorways and left every one of the fridge's
collision-free stances unreachable. `_block_door_swing` takes `self._nav_target` and nothing
else.

**The swing is measured offline.** `door_swing.py` reads it out of each model's
`misc/metadata.json` and prints a table to paste into `DOOR_SWING`. It is a fixed property
of the model, so there is nothing to work out while the simulator runs.

```bash
python door_swing.py                 # the models the test scenes use
python door_swing.py fridge/dszchb   # a specific one
```

**How to work out a new object's swing.**
`link_bounding_boxes[<link>]["visual"]["axis_aligned"]["transform"]` places the door
panel's bounding box in the link's own frame, and the link's origin sits on the hinge. That
transform's translation is therefore the vector from the hinge to the panel's centre, and
which way it points says how the door is hung. The centre is half a panel from the hinge,
so **twice the dominant offset is how far the door reaches.**

| offset hinge -> panel centre | hinge | door reaches | blocked as |
| --- | --- | --- | --- |
| mostly **horizontal** | vertical line down one edge — side-hung, like a fridge | the panel's **width** | a **disc** of that radius: the door sweeps an arc, and the disc is its superset |
| mostly **vertical** | the panel's bottom edge, lying flat — bottom-hung, like an oven | the panel's **height** | a **rectangle** `a` deep by `b` wide: it drops straight out into the room without ever sweeping sideways, and a disc would take floor beside and behind the appliance that the door cannot reach |

`link_tags` marks the door links `openable` on most models; where it does not, every
non-base link is considered and `MIN_OFFSET` discards the ones whose centre sits on their
own origin — shelves, racks, panes of glass. A double-door model contributes one shape per
door link.

| model | offset | hung | shape | panel | object footprint |
| --- | --- | --- | --- | --- | --- |
| `fridge/dszchb` `link_0` | (+0.025, −0.281, +0.004) | side | disc **r=0.564** | 0.609 | 0.899 |
| `oven/ffitak` `door` | (−0.009, −0.000, +0.219) | bottom | box **0.438 × 0.592** | 0.469 | 0.772 |
| `fridge/xyejdx` `link_0`,`link_1` | (+0.020, −0.300, ~0) | side, double | disc **r=0.602** ×2 | 0.633 | 0.971 |
| `oven/fexqbj` `dof_rootd_aa001_r` | (+0.012, −0.000, +0.218) | bottom | box **0.435 × 0.574** | 0.446 | 0.836 |

Only the hinge's *position* is read from the scene, and that needs no frame algebra: the
door link's origin is the hinge. The outward direction comes from the hinge's position
relative to the object's centre, since a door is mounted on a face.

**Sanity-check every new value against the object's own footprint** — a door cannot sweep
further than the object is wide. Deriving the radius from the hinge *axis* instead is the
trap: `joint.axis` is expressed in the joint's own frame, offset from the child link by
`physics:localRot1`, so rotating it by the link's world orientation alone tests the wrong
vector. That misread the fridge as bottom-hung and returned its door's *height*, 1.39 m,
for an appliance 0.64 m wide, which blocked every stance in the kitchen. The footprint
check catches it immediately.

## How NAVIGATE_TO moves

| stage | what it does |
| --- | --- |
| fold | tuck the arm, so the footprint is the one the robot will drive in |
| enumerate | free cells of the eroded map, minus the target's door swing |
| filter | in the robot's own connected component, outside the object's own footprint |
| rank | **nearest to the object**, by true distance |
| check | collision-check the closest `CELL_CANDIDATES` (120) and keep those that pass |
| plan | A* on the same map, string-pulled to its corners |
| drive | rotate in place, drive straight, rotate in place — via `_execute_motion_plan` |

**The map is enumerated, not sampled.** It already knows which cells are floor, which are
eroded away by the robot's footprint, which the door swing takes, and which are in the same
connected component as the robot — so the nearest workable cell is read off directly. Being
in the robot's own component is what sampling could not check, and it dominates: measured at
the potato, **205 928 free cells, of which 5 759 — 2.8% — are reachable**. An earlier
version drew random poses from a widening ring, i.e. from the other 97% blindly, and each
unreachable draw cost a full routing attempt to discover. Enumerating found 8 usable stances
by collision-checking 18 cells, against 240 sampled poses before. Sorting is not a cost
worth avoiding: distances over all 220 000 free cells take 0.10 ms vectorised, and the sort
runs on the few thousand that survive the filters in 0.34 ms.

**One filter, not four.** Only two conditions decide whether a cell is a candidate: it is
reachable, and it is not inside the object. Three others were tried and removed, each
because it took floor away without buying anything:

| removed | what it did | why it went |
| --- | --- | --- |
| centreline arc | kept stances within ±25° of the object's open-floor direction, squaring the robot up to the door | the swing is blocked *before* eroding, so no surviving cell is in the door's way, and the stance faces the object anyway. It narrowed the oven from 6203 reachable cells to 535 and left too few to be routable. |
| room match | required the robot to stand in the object's annotated room | reachability already proves the robot can get there. It narrowed the oven further, 535 to 223, and an appliance set into a wall between two rooms can legitimately be worked at from the other one. |
| growing ring | sampled shells outwards until stances appeared | superseded by enumerating the map, which finds the nearest directly. |

**Ranking is by true distance from the object**, not by distance from a breadth-first seed.
An eight-connected search expands in grid steps, so its frontier is a square rather than a
circle and the cell it lands on is nearest in *steps*, not in metres; ranking from that cell
then carried the error into the stance.

| | ranked from a BFS seed | ranked from the object |
| --- | --- | --- |
| potato, from its centre | 0.75 m | **0.74 m** |
| plate, from its centre | 0.83 m | **0.53 m** |
| plate, from its edge | 0.68 m | **0.38 m** |

**How close the robot can stand is set by its body, not by the map.**
`test_primitives.py --probe` walks the base out from an object and asks both at each step:

```
distance   map says   CuRobo says
  0.50      blocked    COLLIDES
  0.55      free       COLLIDES     <- the map allows it, the robot does not fit
  0.70      free       clear        <- first pose the robot actually fits
```

The map models the robot's *footprint* against the floor plan; CuRobo checks the whole robot
in 3-D, and a lip overhanging the counter strikes its upper body while the base still fits
underneath. That gap is why lowering `ARM_CLEARANCE_MARGIN` does not help — it moves the
map's threshold, already the looser of the two, and at zero it fragments the free space badly
enough that `NAVIGATE_TO(oven)` fails with every stance unroutable. But 0.70 m is not a
global limit: the probe walks straight out along the counter's normal, the worst approach.
Enumerating searches every direction at once and finds better footing off to the side, which
is how the plate is now worked at from 0.53 m.

**The footprint is measured once, after tucking, and never changes.** It has to be measured
tucked, because the map is eroded by the robot's *current* bounding box — measured, the first
navigation of a run eroded by 0.99 m against 0.77 m once folded, the same robot in the same
scene getting two different maps. And it has to stay fixed, because `aabb_extent` grows when
the robot carries something (0.77 m empty, 0.79 m with a potato, 0.87 m with a plate), and
eroding by more later than when the robot parked puts a stance that was legitimately free on
arrival *inside an obstacle*, so A\* cannot even start. Measured: parked 0.53 m from the
plate, every route to the oven failed with "no route, and the direct line is blocked"; pinned
at 0.77 m, that same stance routes over 3.1 m and 3 corners and parks 1.02 m out. A carried object is
`visual_only` and cannot collide, which is why it is already kept out of CuRobo's
`attached_obj`; the map has to agree. One radius per configuration is also what lets the
eroded map and its components be **cached** — neither depends on the target or on where the
robot stands, and before this every navigation re-eroded and re-labelled a 200 000-cell map.

**The robot's region comes from the nearest free cell, not the cell it stands on.** At a
large enough erosion radius the robot's own position is eroded away; `region_of` then returns
the background label, which no free cell carries, and every candidate is discarded as being
in a different component. Measured at the oven while carrying the plate: `in_robot_region: 0`
out of 219 017 free cells, while standing in an open room. An object's region is read the
same way, since its centre is never traversable.

Every `[nav]` line prints the whole funnel — free cells, outside the footprint, in the
robot's region, checked, collision-free — so a failure says which constraint took the floor
away.

### Driving the route

**A\* is the collision guarantee, not a planner in the loop.** CuRobo plans the arm
excellently but, asked for a base trajectory, plans 0.5 m and fails at 1.0 m in this scene.
A\* runs on a map already eroded by the robot's own footprint, so the route it returns has
clearance for the base everywhere along it.

**Rotate, drive, rotate.** One heading change at a time, never both at once, and only
"forward" and "turn" are ever commanded: a Tiago is differential-drive and cannot slide
sideways, whatever OmniGibson's `HolonomicBaseJointController` will accept. Regulated Pure
Pursuit (nav2's default) tracked well but spreads each turn across the drive so the base
arcs continuously, which is harder to watch and to reason about than "turn, then go".

**Waypoint spacing is the tracking error.** `_execute_motion_plan` treats each waypoint as
a step change in target, and the base answers error with force (`command_input_limits:
null`). At 0.12 m spacing the robot arrived 7.5° out of level and on one route toppled
completely; at 0.05 m it stays level.

**Speed is regulated by clearance.** Full speed at 0.45 m of room, down to a quarter of it
in tight places, from a distance transform of the same eroded map. Leaving this out is why
the robot kept toppling between the fridge and the stove — it ran at full cruise through a
gap barely wider than itself.

**Every waypoint commands the tucked pose, and the arm is tucked before setting off** — a
person picks something up, brings it in, then walks. `_reset_robot` tucks after every
primitive, so the arm is folded by default rather than at upstream's reset pose, which
leaves the hand 0.55 m in front of the base against 0.29 m tucked. Two details here toppled
the robot while it carried a plate: the waypoints named the arm's *measured* position
rather than the tucked pose, so an unfinished tuck became the target and a swinging load
ratcheted its own sag in; and the tuck gave up silently, because `_execute_motion_plan`
spends at most `MAX_STEPS_FOR_JOINT_MOTION` (10) steps per waypoint and `ignore_failure=True`
swallows the miss. It now holds the final target until the joints are within
`TUCK_TOLERANCE`, settles, and prints the residual.

**`NAVIGATE_TO` fails if the base is no longer upright**, because the consequence is out of
all proportion to the cause: a toppled robot measures 1.14 × 0.96 m, the map is then eroded
by 1.04 m, its own position stops resolving to a valid floor region, and **every table in
`house_single_floor` becomes unreachable** — failures that look like planning or furniture
problems and are neither. Checking position alone once reported PASS for a robot at
roll=+172°, upside down. Every route now arrives at ±0.0° roll and pitch, including the
tight fridge–stove gap and an 8.1 m run.

## Searching for an object the robot has not seen

Everything above assumes the plan knows where things are. It does not. `NAVIGATE_TO(potato)`
says where the robot should end up, and the executor could only obey it by looking the
potato up in the scene registry — information the robot could not have. Three modules close
that gap, and they sit at three different levels.

| module | level | what it owns |
| --- | --- | --- |
| `exploration.py` | the first version of this | an occupancy grid and a frontier walk, kept for reference |
| `nav_controller.py` | below the primitive | turns one `NAVIGATE_TO` into a series of drives |
| `object_map.py` | below that | what the camera has actually seen, per room |
| `world_graph.py` | beside them | what is known, as typed edges |
| `graph_machine.py` | above the plan | whether a plan works, checked without a simulator |

**A wrong room costs metres, not the task.** The RSN returns a distribution and
`scene_graph.populate` now keeps the whole ranking, not just the winner. When the robot
searches the most likely room and the object is not there, that is evidence: the belief
moves to the next room down and the search runs again. **Nothing re-plans** — the plan only
ever said `NAVIGATE_TO(potato)`, and *finding* it is this layer's job, so the planner is
never asked a second question. Measured in `Beechwood_0_int` with the right room ranked
third: two rooms ruled out, the potato found in the third, 28 m driven against 10 m for a
correct first guess, and the plan untouched.

That also settles what counts as a grounding *failure* in the evaluation. A wrong first
guess is not one, because it is recoverable. The only unrecoverable case is an object the
RSN could not place at all — absent from the graph the planner was shown, so it cannot be
named in a plan.

**The search is a lower layer, not a tenth primitive.** The nine primitives stay high-level
and the action space does not change. `NavigationController.navigate_to` decides *where to
drive next*; the existing drive layer still does the A\* and the rotate-drive-rotate, so the
navigation guarantees above hold unchanged.

```
NAVIGATE_TO(potato)                     high-level, from the plan
    -> NavigationController             room sub-goal -> frontier -> frontier -> approach
    -> controller._drive_to             the simulator's navigation, untouched
```

Two paths. An object already in the world graph gets one approach sub-goal — the old
behaviour exactly. An object never seen gets driven to its room and frontier-searched until
it appears. Finding it writes it into the graph, so the *next* navigation to it is direct.
Measured on the 16-step plan: the first `NAVIGATE_TO(potato)` took 11 sub-goals and 2057
steps, and the three that followed took one sub-goal each, because the plate, oven and
coffee table had all been seen during that first sweep — the table through a doorway.

### The object semantic map

Starts entirely UNKNOWN and fills from the robot's own camera. `seg_instance` decides what
was genuinely seen; `depth_linear` stops each coverage ray at the first surface, so space
behind a counter stays unknown rather than being claimed as searched. **Masked to one room**,
because the bounding box of `kitchen_0` includes corridor and living room, and without the
mask "fully explored" is a claim about floor nobody searched and frontier selection walks the
robot out of the room it was told to look in.

Three things about the camera that cost a run each, all of them the seam between this layer
and the existing machinery rather than the search logic itself:

| symptom | cause |
| --- | --- |
| `no room was given` for the potato and plate | `in_rooms` is the scene file's annotation, and objects spawned at runtime have none — the two objects the search exists to find were the two it could not place. The room now comes from the segmentation under the object's own position. |
| `no reachable floor inside kitchen_0` | the footprint is measured once by whoever touches the navigation map first, and cached for the run. Querying it before tucking pinned it at the untucked **0.99 m** instead of 0.77 m, eroding the kitchen away entirely. Tuck first — `_navigate_near` always did, and so must this. |
| coverage climbing to 76% with **zero objects seen** | `robot.get_obs()` is keyed by *sensor*, and each entry is itself a dict of modalities. Matching modality names against the top-level keys finds nothing, and finds it silently — an empty result is indistinguishable from "the camera looked and there was nothing there". |

**The camera already looks down, and a second look adds nothing.** Tiago's
`reset_joint_pos` puts the head at **−0.45 rad**, about 26° below horizontal, so the obvious
worry — that a level camera misses everything on a low surface — is not the problem. Four
runs were spent on head control before that was measured, and the numbers are worth keeping
so nobody repeats them:

| change | objects seen | potato found |
| --- | --- | --- |
| leave the head alone (default −0.45) | **55** | yes, frontier 9 |
| force an absolute 0.0 rad "level" | 39 | no |
| pitch to −0.6 rad | 34 | no |
| two looks, −0.45 and −0.95 | 37 | no |

The last row settles it: instrumented per-look, the two looks returned **23 and 9 objects
with a union of 23** — the down-look's objects are a strict subset of what the default pose
already sees. Head aiming was removed.

Two things learned on the way, both still true and both cheap to fall into again:

- A down-look is not a floor measurement. `observe_fov` reads any depth return shorter than
  its max range as a surface, which is right for a level camera and wrong for a tilted one:
  pitched down, rays terminate on the *floor* and paint reachable cells as OCCUPIED,
  destroying the frontiers that would have led the robot on.
- Aiming the head is not a matter of setting the joint. `_hold_pose` returns only the trunk
  and arm indices, so the primitives leave the head alone — but the camera has its own
  position `JointController`, so the head has a persistent **drive target** and the motor
  pulls a teleported joint straight back. And `quat2euler(...)[1]` is not a USD camera's
  pitch; it read +0° for a head that had moved, which is what made the first diagnosis wrong.

**What this leaves.** The plate ends at 0.28 m on the coffee table and the robot does not
re-observe it after placing it, so the final graph still records it on the counter and the
predicted-vs-observed audit disagrees on that edge. That is a limitation of *when* the robot
looks, not of what it can see — the fix is a scan after the last placement, not a camera
angle.

### The world graph

Eight edge types. `room_connect` is seeded from `room_graph.py` and is all the robot knows
before it moves; the observed five are written **only once the robot has seen the objects
they connect**, which is what makes the graph a belief rather than a copy of the scene
registry.

| edge | source |
| --- | --- |
| `room_connect` | the room graph, known up front |
| `room_inside` | ground truth, when the object is seen |
| `object_inside`, `on_top`, `under`, `next_to` | simulator predicates, when both objects are seen |
| `holding` | the robot's own actions |
| `nearby` | what `NAVIGATE_TO` drove to, and what is on or inside it |

**One graph, not two.** `scene_graph.populate` produces a plain dict — `{potato:
{room: kitchen_0, probability: 1.0}}` plus any relation the task stated outright — and
that dict is what the LLM is shown and what `planner.validate` checks. It is now also what
`GraphMachine` checks: `WorldGraph.from_scene_graph` reads it in. Before that,
`from_room_graph` took `rooms` and `edges` and **dropped `objects` on the floor**, so the
graph model was validating plans against a graph with nothing in it and every `GRASP`
failed for want of an object the planner had been told about.

What comes across is a belief and is written as one — the RSN's room, the probability that
justified it, and no position at all, because the RSN predicts rooms rather than
coordinates. A stated relation ("the potato on the counter") becomes a kinematic edge,
which is what makes `GRASP(potato)` resolvable before the robot has looked at anything.

Two ways to get this wrong, both silent and both total, so both are commented at the site
and covered by tests. A room a placement names but the topology does not declare must be
**registered, not dropped**: an object in no room is an object `_require_here` finds
nothing to contradict about, so every precondition that turns on where the robot is
standing passes vacuously and a broken plan comes back clean. And the category must
survive grounding: an oven whose category reads `oven_wuinhm_0` rather than `oven` is an
oven that does not open, because the category is what every affordance is decided from.

**The robot is a node.** `ROBOT` sits in the graph like anything else, and the two facts
about it that relate it to something else are edges: `room_inside(robot, kitchen_0)` says
where it is and `holding(robot, potato)` says what it has. So "where is the robot and what
is it carrying" is answered by reading the graph rather than by asking whoever happens to
be executing the plan, and both facts are audited against the world alongside every other
edge. `GraphMachine.location` and `.held` are **views onto those edges**, not a second copy
— two records of where the robot is, is two records that can disagree.

**A carried object has no room of its own.** `GRASP` removes `room_inside` for the object
and everything riding on it, and putting it down writes the edges back. While it is in the
hand its room is the robot's, and `room_of` derives it by following `holding` — one record
of the fact instead of two kept in step by rewriting them on every drive.

```
after NAVIGATE_TO(counter)  robot=kitchen_0  held=None    potato.room=kitchen_0  stored=True
after GRASP(potato)         robot=kitchen_0  held=potato  potato.room=kitchen_0  stored=False
after PLACE_INSIDE(oven)    robot=kitchen_0  held=None    potato.room=kitchen_0  stored=True
```

Deriving a value that used to be stored has one trap, and it cost a completed plan whose
potato ended up in no room at all: `_move_to_room` compared against `room_of`, which for a
still-held object answers with the robot's room, so the restore decided the edge was
already right and wrote nothing. Edge-rewriting helpers read **edges**, not derived
answers.

Two more consequences. `holding` is deliberately *not* one of the kinematic edges, because
an object in the hand is not resting on anything: putting it there would mean re-observing
a carried object drops it. And `open` and `toggled` stay off the graph, because they are
properties of a single node rather than relations between two, and an edge is the wrong
shape for them.

**Carried objects move, and so does whatever rides on them.** Grasping the plate carries the
potato resting on it, and anything resting on the potato — `_carried_with` walks `on_top` and
`object_inside` transitively, so one `NAVIGATE_TO` rewrites `room_inside` for the whole stack.
Lifting an object also ends the relations in which it was the *supported* thing while keeping
those in which it was the support, which is exactly the edge that makes a later `GRASP(plate)`
known to carry the potato to the oven.

### The graph edit state machine

**The specification.** `near(x)` is "the robot is standing at x", and it is a graph fact:
the `nearby(robot, x)` edge `NAVIGATE_TO` writes, alongside `room_inside` and `holding`. `openable`/`switchable` are
affordances, read from tracked state where it exists and from the category lists in
`planner.py` otherwise; those three lists live in one place so the validator, the machine
and the 2-D world cannot disagree about what an object affords.

"The stack" below is the held object plus everything riding on it, which `carried_with`
walks over `on_top` and `object_inside`.

| action | preconditions | effect, in words | effect, in edges |
| --- | --- | --- | --- |
| `NAVIGATE_TO(x)` | none | the robot is now standing at x, and at whatever is on or inside it | `room_inside(robot)` := room of x; `nearby(robot)` := x and its contents, plus whatever is in the hand |
| `RELEASE()` | none | the hand is empty; what it held stays in the room the robot released it in, resting on nothing | `-holding`; `room_inside(stack)` := the robot's room |
| `GRASP(x)` | hand empty; `near(x)`; if `inside(x, c)` then `open(c)`; `graspable(x)` | the robot is holding x, and x travels with it | `+holding(robot, x)`; drop x's kinematic edges but keep its riders; drop `room_inside` for the stack |
| `TOGGLE_ON/OFF(x)` | `near(x)`; `switchable(x)` | x is now on / off | `toggled(x)` := True / False |
| `OPEN/CLOSE(x)` | `near(x)`; `openable(x)` | x is now open / shut | `open(x)` := True / False |
| `PLACE_INSIDE(x)` | `near(x)`; `open(x)` if `openable(x)`; holding something | what was held is now inside x, and is no longer held | `+object_inside(held, x)`; `-holding`; `room_inside(stack)` := room of x |
| `PLACE_ON_TOP(x)` | `near(x)`; holding something | what was held is now on top of x, and is no longer held | `+on_top(held, x)`, `+under(x, held)`; `-holding`; `room_inside(stack)` := room of x |

Two checks are made that are **not** preconditions, because they are about the plan being
well formed rather than about the world: arity (`RELEASE` takes no argument, the other
eight take one), and whether the argument names something the graph has heard of at all.
`test_graph_machine.py` has one case per line of the table.

Three of those lines are worth a sentence each.

- **`NAVIGATE_TO` has no preconditions about the floor.** Whether the robot can get there
  is a question the room graph answers too coarsely to refuse a plan over; `sim2d` runs A\*
  on the eroded map and answers it properly. It does check its *argument*, though: a room is
  refused, and so is a name the graph does not hold — that used to be admitted, which built
  a node for a cupboard that does not exist and let the plan fill it.
- **`RELEASE` has none either.** Opening an empty hand is a step that does nothing, not an
  error, and a plan is not wrong for containing one.
- **`PLACE_INSIDE` needs the container actually open**, and not knowing is not the same as
  knowing it is open — a plan that never opened it has not established what the step needs,
  so it is refused on the same grounds as one that shut it. Something with no door at all,
  a bowl or a sink, has nothing to open and the requirement does not apply.

Offline. Takes the room graph and a plan, applies each action as a **temporal graph edit** —
preconditions read the graph as it stands after every earlier action, effects rewrite it — and
checks the goal against the final graph. It runs in about a millisecond, so a plan can be
rejected before a seventeen-minute simulator run is spent proving it wrong. Two failures,
reported separately because they mean different things:

    inapplicable   some action's preconditions do not hold. The plan is wrong.
    goal not met   every action ran, and the required edges are absent. The plan is
                   executable and does not do the task.

What the graph catches that `planner.py:validate`'s flat model cannot:

| plan | flat model | graph |
| --- | --- | --- |
| `GRASP(plate)` with the plate in a closed oven | legal, the robot is at the oven | rejected: `object_inside(plate, oven)` and the oven is shut |
| the plate put back on the counter instead of the table | legal, every step applies | applicable, **goal not met**, names the missing `on_top(plate, coffee_table)` |
| `GRASP(plate)` after `PLACE_ON_TOP(plate)` | something is held | warns that the potato rides along, and keeps the edge |

`python test_graph_machine.py` covers all of this offline in about a second — 25 checks, no
simulator.

**One goal edge is unobservable by construction.** `OnTop` is contact-based, and an object
stuck to another is deliberately `visual_only`, so it touches nothing and the predicate reads
False however squarely it is sitting there. The predicted-vs-observed audit says so rather
than reporting it as a disagreement.

```bash
python test_graph_machine.py                    # ~1s, no simulator - graph and machine
python test_primitives.py --scene house_single_floor --search --bev --video out.mp4
```

## The RSN

`P(object present | room type)` for every room type in one forward pass, following Ginting
et al., *"SEEK: Semantic Reasoning for Object Goal Navigation in Real World Inspection
Tasks"* (RSS 2024), `docs/SEEK.pdf`, section IV-B.

    object name -> frozen text encoder (BGE-small) -> MLP (3 hidden layers)
                -> P(object present) in each of 37 room types

The frozen encoder is what makes the model **open-vocabulary**: an unseen name still lands
near semantically similar ones, so the model degrades instead of failing. Since an LLM can
name any object and BEHAVIOR covers 197, this matters. Emitting a vector over *all* room
types matches SEEK, whose MDP planner needs P(object) in every room.

```bash
python extract_scene_data.py     # scene JSONs -> data/placements.csv, data/vocab.json
python build_dataset.py          # -> data/pairs_merged.csv (with negatives)
python embed_categories.py       # cache frozen BGE-small embeddings
python train_rsn.py --calibrate --out models/rsn_cal.pt

python query_rsn.py --object toaster --room kitchen
python query_rsn.py --object "fire extinguisher" --top 5
```

### Divergences from the paper

- **Loss.** SEEK regresses with MSE onto GPT-4-distilled probabilities because it has no
  ground-truth layouts. We have 51 annotated scenes, so we use BCE on hard occurrence
  labels — the correct likelihood for binary observations. Brier is reported alongside for
  comparability with the paper's Table II (theirs is against soft LLM targets, so the
  numbers are not directly comparable).
- **No second output head.** SEEK also predicts P(find without careful search) to set MDP
  transition probabilities. That is search difficulty, not placement plausibility, and
  BEHAVIOR has no label for it.
- **Calibration.** `--calibrate` fits Platt scaling on training cells after training. Class
  reweighting inflates probabilities (mean 0.26 against a 0.08 base rate); calibration
  corrects this without affecting ranking, since the transform is monotonic.

### Data

Each of the 51 scenes ships one `json/<scene>_best.json`. Under `objects_info.init_info`,
every object records its `category` and an `in_rooms` list:

```json
"bookcase_njwsoa_0": {
  "args": {"category": "bookcase", "model": "njwsoa", "in_rooms": ["living_room_0"]}
}
```

The room *type* is the instance name with its trailing index stripped. The 39 types
recovered match `metadata/room_categories.txt` exactly. Structural geometry (walls,
ceilings, floors) carries no `in_rooms` and is excluded. Totals: **51 scenes, 39 room
types, 250 object categories, 11,218 placements across 359 room instances.**

**Negatives.** The raw data is positive-only, so each concrete room instance is treated as
a **closed world**: within one room we know every object present, so each absent category
is a true negative. Rooms with fewer than 3 objects are dropped (`--min-room-objects`) — a
nearly empty room reflects incomplete annotation, not genuine absence.

**Merging.** BEHAVIOR splits common nouns into variants: nine sinks, eight chairs, eight
ceiling lights. `category_merge.py` collapses 61 fine-grained categories into 17 everyday
nouns, taking the vocabulary from 250 to 197. This is a *training-label* decision, not a
query-vocabulary one — the encoder still accepts arbitrary names. It matters because
splitting one noun across rare variants fragments supervision:

| RSN trained on | AUC | AP | Brier |
| --- | --- | --- | --- |
| merged (197 categories) | **0.904** | **0.602** | **0.057** |
| fine-grained (250 categories) | 0.879 | 0.535 | 0.062 |

Merging uses an explicit table, never substring matching — names mislead in both directions
(`periodic_table` is a wall chart, `pool_table` is furniture, `hall_tree` is a coatrack),
and a substring rule would silently corrupt the prior. Distinctions carrying real placement
information are kept: `urinal` vs `toilet`, `display_fridge` vs `fridge`.

### Evaluation

The split is **grouped by scene**, so all room instances of a building land on the same
side and validation measures generalization to an unseen building. A pair-level split would
leak. Class balance is ~5% positive, so the loss uses `pos_weight` and the metrics are
ROC-AUC and average precision — accuracy is meaningless when predicting "absent" everywhere
scores 95%.

Held-out scenes (10 of 51):

| metric | model | random baseline |
| --- | --- | --- |
| ROC-AUC | 0.904 | 0.500 |
| Average precision | 0.602 | 0.083 |
| Brier (calibrated) | 0.057 | — |

AUC and AP are implemented in `evaluation.py` (sklearn is not in the `behavior` env); both
were verified against brute-force computation including tie handling.

## Known limitations

- **All nine primitives are state changes, not motion.** Manipulation is resolved by
  welding objects to the gripper and teleporting them; the arm does not reach for anything.
  **The navigation is real** — A* over the eroded traversability map, driven by the robot —
  so a video shows genuine driving and a state flip for everything else.
- **BEHAVIOR scenes are furniture-only.** Asked what each of the ten benchmark scenes holds
  that a robot could pick up, the answer across all ten is **three objects**: `Rs_int`'s
  laptop, `Wainscott_0_int`'s coffee maker, and a garden fence. Anything to pick up has to
  be injected, and it has to be a category the object dataset ships. Where it is placed
  matters as much as that it exists: an object sampled against the back wall of a counter is
  one the robot can stand beside but not reach.
- **Half the scenes are not one region of floor.** The room graph says which rooms adjoin;
  A\* says which the robot can drive between, and 24 of 51 disagree before the door
  thresholds are opened, 4 after. `Wainscott_0_int` is in two pieces, so its ten benchmark
  tasks all stay inside one of them.
- **The goal language cannot say "was heated".** Goals are graph edges plus the two node
  properties, so a plan that never switches the oven on still satisfies "heat the pie and
  put it on the table". Three ablation cases are harmless for exactly this reason - a fact
  about the goal, not a miss by the checker.
- **`PLACE_INSIDE` on something with no inside is allowed.** A countertop is not a
  container, and neither model refuses it: the graph model has no containment affordance,
  and the simulator refuses it only on distance. Deciding what is a container from a
  category list is the same guess `planner.py` warns about for openability, so it is
  recorded rather than patched.
- **Open-vocabulary generalization is imperfect.** `printer` scores 0.92 for kitchen;
  `bathrobe` favors garden. Fine-grained names the model was not trained on fall back to
  the text embedding — raw `pot_plant` scores garden 0.97 and living_room 0.05, though its
  merged form `plant` correctly scores living_room 0.98. Treat RSN output on unseen names
  as a soft prior.
- **Probabilities are softer than a lookup table.** `P(toilet in kitchen)` is 0.18, not ~0.
  Calibrate thresholds on held-out scenes rather than assuming 0.5 is meaningful.
- **The model conditions on room type, not scene.** It says how typical a dishwasher is in
  kitchens, not whether *this* kitchen has one. Per-scene data is in `placements.csv`.
- **Setup.** `sentence-transformers` is installed in the `behavior` env (torch, CUDA and
  OmniGibson verified intact afterward). `embed_categories.py` downloads BGE-small on first
  run.
