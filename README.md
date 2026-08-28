# LLM safety filter for BEHAVIOR-1K

Turns a natural-language task and a scene into a **validated sequence of BEHAVIOR-1K
action primitives**, then executes it in OmniGibson.

An LLM asked to plan for a household robot will confidently reference objects that are not
in the house, grasp things that cannot be picked up, and place objects it never picked up.
A robot cannot see the whole house at once, so it cannot check those claims directly. This
pipeline supplies the missing information — a learned prior over where objects are, plus a
symbolic model of what the primitives permit — and rejects plans that violate either.

```
task description ──> objects needed ──┐
                                      ├──> scene graph ──> LLM plan ──> validator ──> simulator
scene floor plan ──> room graph ──────┘        (RSN)                                   (video)
```

| Stage | Module | What it does |
| --- | --- | --- |
| 1 | `task_objects.py` | task text -> the objects the task needs |
| 2 | `room_graph.py` | floor plans -> room adjacency graph |
| 3 | `scene_graph.py` | RSN places those objects in rooms |
| 4 | `planner.py` | LLM proposes primitives; validator checks them |
| 5 | `execute_plan.py` | grounds onto scene objects, runs in OmniGibson, records video |

```bash
source ~/safety_filter/setup_behavior_env.sh   # behavior env, CUDA 12.8, GPU 1

python pipeline.py --scene Beechwood_0_int --task "open the fridge in the kitchen" --json plan.json
python execute_plan.py --plan plan.json --video figures/run.mp4
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

## Stage 4 — planning and validation

The LLM (Qwen2.5-7B-Instruct by default) sees the action space, the scene graph, the rules,
and worked examples, and returns one plan. **Planning is single-shot on purpose.**
Repairing a rejected plan — feeding errors back, replanning, recovering mid-execution — is
the open research question this project exists to study, so the pipeline surfaces the raw
failure rather than papering over it.

The parser tolerates numbering, bullets, markdown fences and prose, and repairs
malformations that carry unambiguous intent (`PLACE_ON_TOP.bed()`, `GRASP: mug`). Dropping
those silently made corrected plans look incomplete.

The validator replays the plan against a symbolic world model tracking what the robot
holds, where it is, and what is open. It rejects:

- placing with an empty hand, or grasping while already holding something
- acting on an object in a different room without navigating there
- objects the scene graph does not contain
- **grasping a fixed appliance** — the LLM reliably writes `GRASP(fridge)` before
  `OPEN(fridge)`; a robot cannot pick up a fridge, so this is caught at planning time
- a plan that ends still holding something — it has not finished the task, and calling it
  executable would be misleading
- wrong arity, including an argument passed to `RELEASE`

Warnings cover the non-fatal cases: placing inside a closed container, closing something
never opened, navigating between non-adjacent rooms.

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

### Testing

Five offline entry points and one simulator run, split by cost:

```bash
python check_names.py primitive_patches.py test_primitives.py   # <1s - undefined names
python test_stance_order.py                # ~2s - which stance the map picks
python test_pipeline.py                    # ~1s - parser and validator
python scene_setup.py Beechwood_0_int      # ~2s - furniture selection
python door_swing.py                       # ~1s - door swing shapes

OMNIGIBSON_GPU_ID=1 CUROBO_GPU_ID=1 python test_primitives.py \
    --scene house_single_floor --bev --video figures/plan_potato_plate_bev.mp4
```

`check_names.py` reports names a function reads that nothing it can see defines — the
NameError class of bug, found in a second rather than three minutes into a simulator run.
It follows Python's scoping rules, so a closure variable or a module-level import is not
flagged while a genuine typo is. Run it before every launch.

`test_stance_order.py` reproduces the stance choice from the navigation map alone and
`test_primitives.py --probe` measures the distance at which the robot actually fits, so the
ranking can be iterated on in seconds rather than half-hour simulator runs.

**Check that a run actually started before believing its output.** Compare the log's
timestamp against the sources it should have picked up: a background launch that silently
fails leaves the previous log in place. Editing a source *during* a run has the same
effect, since Python has already imported it.

`test_primitives.py` launches Isaac Sim and runs one atomic action plan end to end — cook a
potato on a plate and set it on the table — checking the resulting world state after every
step (`Open`, `ToggledOn`, `OnTop`, `Inside`, object-in-hand), not merely that no exception
was raised, so a step cannot pass by doing nothing.

```
NAVIGATE_TO(potato)   GRASP(potato)      NAVIGATE_TO(plate)  PLACE_ON_TOP(plate)
GRASP(plate)          NAVIGATE_TO(oven)  OPEN(oven)          PLACE_INSIDE(oven)
CLOSE(oven)           TOGGLE_ON(oven)    TOGGLE_OFF(oven)    OPEN(oven)
GRASP(plate)          CLOSE(oven)        NAVIGATE_TO(table)  PLACE_ON_TOP(table)
```

Sixteen steps covering eight of the nine primitives — every one except `RELEASE`, which
this plan never needs because each placement empties the hand. Every `NAVIGATE_TO` is its
own step, because the manipulation primitives do not travel and `_require_near` fails them
if none ran.

**The plan runs exactly as written.** Nothing is inserted between steps to tidy up after a
failure. An earlier version released whatever was in the hand before each
OPEN/CLOSE/TOGGLE, which kept later steps passing but would put the plate on the floor
halfway through this plan. If a step leaves the world in a state the next step cannot use,
that is a result about the plan and it should show.

The potato and the plate are dropped onto the counter at opposite ends of its edge
(`lateral` ∓0.45 m), so the first two `NAVIGATE_TO` steps are genuinely different drives.
The potato is welded to the plate the moment it lands there: it is a separate rigid body
resting on contact alone, so it would otherwise slide off when the plate is lifted, and the
run would measure whether a potato stays balanced rather than whether the plan is right.

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
- **BEHAVIOR scenes are furniture-only.** No loose small objects exist in any of the 51
  scenes, so anything to pick up has to be injected. Where it is placed matters as much as
  that it exists: an object sampled against the back wall of a counter is one the robot can
  stand beside but not reach.
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
