# Running the OmniGibson branch

`omnigibson_runtime/` drives a validated plan in the full Isaac simulator. Nothing in the 2-D
pipeline imports it and no reported number depends on it, so it is the one part of the repository
that the offline test suite does not cover — it is verified by running it.

Running it is not "activate the environment and go". Isaac, CuRobo and OmniGibson each need
configuration that is easy to get wrong in ways that produce **plausible wrong answers** rather
than errors. Everything below was learned by getting it wrong.

## The one rule

```bash
source ~/safety_filter/setup_behavior_env.sh
python -u omnigibson_runtime/execute_plan.py --plan plan.json --video figures/run.mp4
```

`run_sim.sh` is the reference invocation and already does both correctly:

```bash
./run_sim.sh plan.json figures/run.mp4
```

## Why each part matters

### Always source `setup_behavior_env.sh`

It is not a convenience wrapper. It sets, and then *validates*:

| what | why |
| --- | --- |
| `conda activate behavior` | the interpreter, at `/mnt/check/ruiyangw/conda/envs` |
| `CUDA_HOME`, `CUDA_PATH`, `CUDACXX`, `CPATH`, `LIBRARY_PATH`, `LD_LIBRARY_PATH` | CuRobo compiles against CUDA **12.8**; the script refuses to proceed if `nvcc` reports anything else |
| `TORCH_CUDA_ARCH_LIST=8.6` | the RTX A5000's compute capability |
| `OMNIGIBSON_GPU_ID=1` | which card renders |
| `OMNIGIBSON_HEADLESS=1` | no local X/GLFW window |
| `OMNIGIBSON_REMOTE_STREAMING=webrtc` + ports | the streaming client |
| four `grep` checks on `simulator.py` | the Isaac Sim 5.1 WebRTC patches are still applied |

**Measured consequence of skipping it:** `test_primitives` scored 14/16 with `CLOSE (oven)`
failing three attempts with `POST_CONDITION_ERROR: the object did not open or close as expected`.
With the environment sourced and nothing else changed, the same test scores **16/16** and `CLOSE`
succeeds — taking 350 physics steps where the misconfigured run gave up at 300. The plate was
placed at the identical coordinates in both runs, so the placement was never the problem: CuRobo
without its own CUDA toolchain simply plans worse.

A failure like that looks exactly like a bug in the primitive. It is not.

### Never set `CUDA_VISIBLE_DEVICES`

It overrides `OMNIGIBSON_GPU_ID` and collapses a deliberate split. Correctly configured, one run
occupies **two** cards:

```
GPU 0   ~9 GB    CuRobo (it pins cuda:0, which is expected and harmless)
GPU 1   ~7 GB    OmniGibson rendering  (OMNIGIBSON_GPU_ID=1)
```

Forcing everything onto one card is what produced the `CLOSE` failure above.

### Always run with `python -u`

OmniGibson tears down through `os._exit()`, which **skips flushing Python's stdout buffer**. When
stdout is a file or a pipe it is block-buffered, so the last few KB of output are discarded on
exit. A completed 16-step run then appears in the log to have stopped at step 7 or 8, with
Isaac's own unbuffered `Simulation App Shutting Down` printed after the last surviving line.

This costs hours if you do not know it: the log looks exactly like a hard crash mid-plan, and the
process still exits 0.

### Read Python's exit status, not a pipe's

```bash
python -u cached/tests/test_primitives.py 2>&1 | tail -6   # $? is tail's status. Always 0.
```

Redirect to a file and grep it afterwards. And note that `test_primitives` reports each step as
`PASS`/`FAIL` **without** exiting non-zero on a failed step — it raises `SystemExit` only for
setup problems. The verdict is the summary line:

```
16/16 plan actions succeeded
```

If that line is missing, the run did not finish.

### One Isaac job at a time

A run holds ~20 GB and takes roughly 20 minutes. Before launching, check for your own stale
processes — including ones left by an earlier session:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```

Note that `pgrep -f test_primitives` matches the shell you type it in. Wait on an explicit PID
with `kill -0 "$PID"` instead.

## The tests

Both need Isaac and a free GPU.

| test | what it drives |
| --- | --- |
| `cached/tests/test_primitives.py` | all nine primitives, as a 16-step cook-and-serve plan: fetch a potato, put it on a plate, carry the plate to the oven, open, place inside, close, run the oven, take it out, close, carry it to the table. Manipulation is symbolic; **navigation is real** — A* over the eroded traversability map |
| `cached/tests/test_stance_order.py` | where the robot chooses to stand relative to a target |

```bash
source ~/safety_filter/setup_behavior_env.sh
python -u cached/tests/test_primitives.py > /tmp/prim.log 2>&1
grep -E "PASS|FAIL|plan actions succeeded" /tmp/prim.log
```

`cached/` is not tracked, so these are only available in a working copy that has it.

## Known flake

`PLACE_INSIDE` chooses where to rest an object by casting rays into the container's fillable
volume and taking the surviving surface furthest from the door — a centred placement leaves the
object in the doorway, where the door fouls on it and `CLOSE` fails. The comments in
`primitive_patches.py` record the tuning: on `oven/ffitak` the current rule scored 7/8 against
8/12 for the alternative. So an occasional `CLOSE` failure after a placement is a known
limitation of that search, **but** confirm the environment is sourced before concluding you have
hit it — a misconfigured run fails the same way every time.
