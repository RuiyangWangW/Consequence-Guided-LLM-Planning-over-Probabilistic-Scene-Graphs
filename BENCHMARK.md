# Building the benchmarks

How the two task sets are constructed, verified and regenerated. **Nothing here is needed to
read the results** - `README.md` describes what the benchmarks contain and what the pipeline
scores on them. This file is the other half: what makes a task admissible, what is checked
before one is written out, and how to rebuild either set from source.

It is kept separate deliberately. A benchmark whose construction rules are part of the method's
own documentation invites tuning the rules until the method looks good; keeping the two apart
makes it obvious which decisions belong to the benchmark and which to the pipeline. Every rule
below is a property of the *task*, not of any planner.

---

## The single-task benchmark

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

### Verification

`build_tasks.verify` gates every task. It replays the reference plan through `GraphMachine`,
drives it in the 2-D simulator, checks the furniture is really in the room the sentence names,
checks the movable is a category OmniGibson can build, and applies the in-versus-on rule. A task
whose own reference plan does not reach its goal is never written out.

`build_tasks.py` **refuses to overwrite `data/tasks.json`** unless the rebuild is byte-identical.
The benchmark is a fixed artefact; a run that silently regenerated it underneath a comparison
would make two result files incomparable while their task ids still matched.

---

## The multi-task benchmark

500 instructions over the same ten scenes, each combining 2-5 short errands drawn from a pool of
140. The errands are declared in `subtasks.py` in exactly `tasks.py`'s format and every one goes
through `build_tasks.verify` - the same function that gates the single-task set - before any of
them is combined. **There is one task generator, not two.** `build_multitask.py` refuses to write
`data/tasks.json` at all, for the same reason `build_tasks.py` does.

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

**Errands that change an object's state, not only its place.** Twenty-two of the 140 errands ask
for `cooked`, `washed` or `dried`, which correspond to no primitive: they expand into an appliance
macro - in, shut, run, off, open, out, shut - derived from `GraphMachine`'s own rule that
`TOGGLE_ON` confers the state on whatever is *inside* the appliance. These reuse the exact shape
calls the single-task benchmark already verifies (`heat_and_serve`, `fetch_heat_serve`), plus a
short `confer_in_place` form for the cases where the long one only added carrying. They are the
part of the benchmark a purely relational planner cannot express at all - `epog` scores 0 on every
instruction containing one.

**How much the sentence gives away.** An instruction that says "the *kitchen* countertop" tells the
robot where to go; one that says "the countertop" does not. Three quarters of the errands were
written the first way, which makes them easier than the task they are meant to stand for - a person
asking for a mug to be moved does not usually name the room, and a robot that is told cannot be said
to have found anything.

The room word is therefore dropped from 63 of the 140 sentences, and only where the scene holds
exactly one instance of that category, so the reference stays unambiguous while its location becomes
something the robot has to establish. That takes the share of furniture references the RSN must
guess from 25% to 54%.

This makes the benchmark harder for every method, not easier for one: the robot now searches for
objects it was previously handed, and driven distances rose across all arms (the three GAVEL
variants from 62-63 m to 72-76 m). It is also the change that makes an uncertainty-aware ordering
stage worth having *or* not - a planner that reasons badly about where things are has more room to
be wrong here, not less.

**The live quota.** `combine` fills 70% of each scene with instructions on which ordering from the
full distribution, from its argmax, and online do not all give the same answer - and tops the scene
up with ordinary draws. It is a quota rather than a filter so no scene is dropped: `Pomaria_1_int`
could produce no live instruction at all under the old sentences and keeps its fifty either way.
**The test is whether the arms *decide* differently, never which one wins.** That distinction is
the whole defence of this rule, so it is worth checking rather than asserting: on the resulting
set the full distribution produces a *worse* route than its own argmax on 129 of the 309
instructions where the two differ, and online replanning is worse than ordering once on 69 of 213.
A rule that selected for the method would not leave 42% and 32% of its own comparisons going the
other way.

What the rule buys is power, not direction. An instruction on which every arm chooses the identical
order contributes exactly zero to a paired comparison - not noise, zero - so a set full of them
dilutes the reported effect while adding nothing. It took the live share from 201/500 to 438/500.

Two costs, both real:

* **The set is built using the estimator under test.** `ordering_live` runs the same three arms
  that are later compared. The criterion does not read their outcome, but it does read their
  decisions, and a different cost model would select a different 70%.
* **It skews the size mix.** Two-errand instructions fell from 106 to 41, because with only two
  orderings and a commitment made before anything is seen, they are a structural zero for the
  online arm. Their absence should be stated wherever the online increment is quoted.

---

## How the belief is prepared

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

---

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
| `subtasks.json` | 140 short errands, 11-15 per scene, each verified by `build_tasks.verify` | `build_multitask.py` | `build_multitask.py` (writes both files in one run) |
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
