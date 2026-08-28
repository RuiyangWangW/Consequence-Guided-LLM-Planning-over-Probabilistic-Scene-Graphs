"""Turn a task description plus a populated scene graph into a validated action plan.

The LLM proposes a sequence of BEHAVIOR-1K action primitives; a symbolic world model
then replays that sequence and rejects anything the real `StarterSemanticActionPrimitives`
controller would refuse. LLMs reliably produce plausible-looking but infeasible plans -
placing an object never grasped, opening a fridge from another room - so generation
alone is not enough. Validation is what makes the output executable.

The action space is exactly the nine primitives in `controller_functions`. Arity comes
from their real signatures: RELEASE takes no argument (`_execute_release(self)`), every
other primitive takes exactly one object.
"""

import json
import re

# The nine primitives, mirroring StarterSemanticActionPrimitiveSet. `takes_object` is
# read off the controller method signatures, not guessed.
PRIMITIVES = {
    "GRASP": {"takes_object": True, "doc": "Grasp an object"},
    "PLACE_ON_TOP": {"takes_object": True, "doc": "Place the held object on top of another"},
    "PLACE_INSIDE": {"takes_object": True, "doc": "Place the held object inside another"},
    "OPEN": {"takes_object": True, "doc": "Open an object"},
    "CLOSE": {"takes_object": True, "doc": "Close an object"},
    "NAVIGATE_TO": {"takes_object": True, "doc": "Navigate to an object"},
    "RELEASE": {"takes_object": False, "doc": "Release the held object, letting it fall"},
    "TOGGLE_ON": {"takes_object": True, "doc": "Toggle an object on"},
    "TOGGLE_OFF": {"takes_object": True, "doc": "Toggle an object off"},
}

# Containers a robot must open before placing inside. Used only for warnings: the real
# precondition lives in the simulator, and an object's true openability depends on its
# model, which the scene graph does not carry.
OPENABLE = {
    "fridge", "refrigerator", "freezer", "oven", "microwave", "dishwasher", "washer",
    "clothes_dryer", "dryer", "cabinet", "bottom_cabinet", "top_cabinet", "drawer",
    "trash_can", "box", "carton", "backpack", "briefcase",
}


# Large fixed appliances and furniture. These are articulated (OPEN/CLOSE act on their
# doors) or switchable (TOGGLE_*), but they are not portable: GRASP on one is a planning
# error the simulator would also refuse, since the robot cannot pick up a fridge.
NOT_GRASPABLE = {
    "fridge", "refrigerator", "freezer", "oven", "stove", "microwave", "dishwasher",
    "washer", "clothes_dryer", "dryer", "sink", "furniture_sink", "pedestal_sink",
    "bathtub", "shower", "shower_stall", "toilet", "bed", "sofa", "counter",
    "countertop", "cabinet", "bottom_cabinet", "top_cabinet", "bookcase", "door",
    "window", "openable_window", "coffee_table", "breakfast_table", "table", "desk",
    "standing_tv", "television", "fireplace", "trash_can", "public_trash_can",
}


class PlanError(Exception):
    """A plan step the controller would refuse."""


def parse_plan(text):
    """Pull `PRIMITIVE(object)` steps out of an LLM reply.

    Tolerant of the usual noise: numbering, bullets, markdown fences, prose around the
    list. Anything that is not a recognized primitive call is ignored rather than
    guessed at, so a chatty model does not inject junk steps.
    """
    steps = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", "```")):
            continue
        line = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()

        # Canonical form is PRIMITIVE(object). Models also emit near-misses that carry
        # the same unambiguous intent - PLACE_ON_TOP.bed(), GRASP: mug, NAVIGATE_TO bed
        # - and dropping those silently makes a sound plan look incomplete. Rewrite the
        # recognizable shapes rather than discarding a step the model got right.
        m = re.match(r"^([A-Z_]+)\s*[.:]\s*([A-Za-z0-9_]+)\s*\(\s*\)", line)
        if m:
            line = f"{m.group(1)}({m.group(2)})"
        else:
            m = re.match(r"^([A-Z_]+)\s*[:\s]\s*([A-Za-z0-9_]+)\s*$", line)
            if m and m.group(1) in PRIMITIVES:
                line = f"{m.group(1)}({m.group(2)})"

        m = re.match(r"^([A-Z_]+)\s*\(\s*([^)]*?)\s*\)", line)
        if not m:
            continue
        name, arg = m.group(1).upper(), m.group(2).strip()
        if name not in PRIMITIVES:
            continue
        arg = arg.strip("\"'").strip()
        # Models sometimes write NAVIGATE_TO(kitchen_0, fridge); keep the first term.
        if "," in arg:
            arg = arg.split(",")[0].strip().strip("\"'")
        steps.append({"action": name, "object": arg if arg else None})
    return steps


def validate(steps, graph, strict=True):
    """Replay `steps` against a symbolic world model, collecting errors and warnings.

    Tracks what the robot holds, where it is, and what is open - the state the real
    primitives check. Returns (errors, warnings); `errors` empty means the plan is
    internally consistent and every referenced object is one the scene graph believes
    exists.
    """
    objects = graph.get("objects", {})
    rooms = set(graph.get("rooms", {}))
    unplaced = set(graph.get("unplaced", {}))

    adjacency = {r: set() for r in rooms}
    for a, b in graph.get("edges", []):
        adjacency[a].add(b)
        adjacency[b].add(a)

    errors, warnings = [], []
    held = None
    location = None  # room the robot is in; None until the first navigation
    opened = set()

    def where(obj):
        return objects[obj]["room"] if obj in objects else None

    for i, step in enumerate(steps, 1):
        action, obj = step["action"], step.get("object")
        spec = PRIMITIVES[action]
        tag = f"step {i} {action}({obj or ''})"

        # --- arity, straight from the controller signatures ---
        if spec["takes_object"] and not obj:
            errors.append(f"{tag}: {action} requires an object argument")
            continue
        if not spec["takes_object"] and obj:
            warnings.append(f"{tag}: RELEASE takes no argument; ignoring '{obj}'")
            obj = None

        # --- the object must be something the scene believes in ---
        if obj is not None:
            if obj in unplaced:
                errors.append(f"{tag}: '{obj}' is believed NOT present in this scene")
                continue
            if obj not in objects and obj not in rooms:
                errors.append(f"{tag}: '{obj}' is not in the scene graph")
                continue

        # --- reachability: acting on an object requires being in its room ---
        target_room = obj if obj in rooms else where(obj)
        if action == "NAVIGATE_TO":
            if target_room is not None:
                if location is not None and target_room != location:
                    if target_room not in adjacency.get(location, set()):
                        # Not adjacent is not fatal - the motion planner routes through
                        # intermediate rooms - but a long hop is worth surfacing.
                        warnings.append(
                            f"{tag}: {location} and {target_room} are not directly "
                            "connected; the robot must route through other rooms"
                        )
                location = target_room
        else:
            if target_room is not None and location != target_room:
                if strict:
                    errors.append(
                        f"{tag}: robot is in {location or 'an unknown room'} but "
                        f"'{obj}' is in {target_room}; navigate there first"
                    )
                    continue
                warnings.append(f"{tag}: implicit navigation to {target_room}")
                location = target_room

        # --- per-primitive preconditions, mirroring the controller's own checks ---
        if action == "GRASP":
            if held is not None and held != obj:
                errors.append(f"{tag}: already holding '{held}'; release or place it first")
                continue
            if obj in NOT_GRASPABLE:
                errors.append(
                    f"{tag}: '{obj}' is a fixed appliance or furniture and cannot be "
                    f"picked up; use OPEN/CLOSE for its door or TOGGLE_ON/TOGGLE_OFF "
                    f"to switch it"
                )
                continue
            held = obj

        elif action in ("PLACE_ON_TOP", "PLACE_INSIDE"):
            if held is None:
                errors.append(f"{tag}: nothing in hand to place")
                continue
            if held == obj:
                errors.append(f"{tag}: cannot place '{obj}' onto itself")
                continue
            if action == "PLACE_INSIDE" and obj in OPENABLE and obj not in opened:
                warnings.append(f"{tag}: '{obj}' is usually opened before placing inside")
            held = None

        elif action == "RELEASE":
            if held is None:
                warnings.append(f"{tag}: nothing in hand to release")
            held = None

        elif action == "OPEN":
            if obj in opened:
                warnings.append(f"{tag}: '{obj}' is already open")
            opened.add(obj)

        elif action == "CLOSE":
            if obj not in opened:
                warnings.append(f"{tag}: '{obj}' was not opened by this plan")
            opened.discard(obj)

    # An unfinished plan is an error, not a warning: a sequence that ends mid-carry has
    # not completed the task, and reporting it as EXECUTABLE would be misleading. This
    # is the failure mode LLM planners hit most often - they navigate to the target and
    # forget the final placement - and it is precisely the kind of feedback the retry
    # loop can act on.
    if held is not None:
        errors.append(
            f"plan ends while still holding '{held}': finish by placing or releasing it"
        )
    for obj in sorted(opened):
        warnings.append(f"plan ends with '{obj}' left open")

    return errors, warnings


def build_prompt(task, graph):
    """The planning prompt: action space, scene graph, rules, and output format."""
    from scene_graph import format_for_llm

    actions = "\n".join(
        f"  {name}({'object' if s['takes_object'] else ''})  - {s['doc']}"
        for name, s in PRIMITIVES.items()
    )
    prompt = f"""You are a task planner for a household robot in a simulated home.

Produce a sequence of atomic actions that completes the task. You may ONLY use these
actions, exactly as written:

{actions}

{format_for_llm(graph)}

Rules:
- Use only the objects listed above. Do not invent objects.
- The number after each object is how confident we are that it is really there. A low
  number means the object may not exist in this house; plan for it only if the task
  requires it.
- NAVIGATE_TO(object) before acting on an object in a different room.
- GRASP before any PLACE. The robot has one hand: it cannot hold two objects.
- After PLACE_ON_TOP or PLACE_INSIDE the hand is empty again.
- OPEN a closed container before placing something inside it, and CLOSE it after.
- OPEN and CLOSE act on the object directly. Do NOT grasp a door, appliance or cabinet
  in order to open it: write OPEN(fridge), never GRASP(fridge).
- GRASP is only for objects the robot will carry somewhere.
- RELEASE takes no argument: write exactly RELEASE().

Examples of correct sequences:

  Task: open the microwave
  NAVIGATE_TO(microwave)
  OPEN(microwave)

  Task: turn on the light, then turn it off
  NAVIGATE_TO(light)
  TOGGLE_ON(light)
  TOGGLE_OFF(light)

  Task: put the book on the table
  NAVIGATE_TO(book)
  GRASP(book)
  NAVIGATE_TO(table)
  PLACE_ON_TOP(table)

Task: {task}

Reply with ONLY the action sequence, one action per line, no numbering, no prose."""

    return prompt


def generate(task, graph, model_name="Qwen/Qwen2.5-7B-Instruct",
             max_new_tokens=512, strict=True, verbose=True):
    """Ask the LLM for a plan once, and report what the validator makes of it.

    Deliberately single-shot. Repairing a rejected plan - feeding validator errors back,
    replanning, or recovering mid-execution - is the open research question this project
    exists to study, so the pipeline surfaces the raw failure rather than papering over
    it here.
    """
    generator = get_generator(model_name)
    reply = generator(build_prompt(task, graph), max_new_tokens)
    steps = parse_plan(reply)
    errors, warnings = validate(steps, graph, strict)
    if verbose:
        print(f"  {len(steps)} steps, {len(errors)} errors, {len(warnings)} warnings")
    return {"steps": steps, "errors": errors, "warnings": warnings, "raw": reply}


# One loaded model per name. Object extraction and planning both call the LLM, and a 7B
# load costs ~30s of GPU time, so the second caller reuses the first one's weights.
_GENERATORS = {}


def get_generator(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """Return a cached prompt->text function for this model."""
    if model_name not in _GENERATORS:
        _GENERATORS[model_name] = _local_generator(model_name)
    return _GENERATORS[model_name]


def _local_generator(model_name):
    """Lazily load a local instruct model and return a prompt->text function."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    # `device_map="auto"` needs `accelerate`, which is not in the `behavior` env and is
    # not worth installing there - the env has a verified torch/CUDA/OmniGibson stack.
    # A 7B model in fp16 is ~15GB and fits on one A5000, so load it and move it whole,
    # falling back to sharding only if accelerate happens to be available.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
    except (ValueError, ImportError):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    def run(prompt, max_new_tokens):
        messages = [{"role": "user", "content": prompt}]
        # Depending on the transformers version this returns either a bare tensor or a
        # BatchEncoding; normalize to a tensor of ids so both work.
        encoded = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)

    return run


def format_plan(result):
    """Human-readable plan report."""
    lines = []
    for i, s in enumerate(result["steps"], 1):
        arg = s["object"] or ""
        lines.append(f"  {i:2d}. {s['action']}({arg})")
    if not result["steps"]:
        lines.append("  (no valid actions parsed)")
    if result["errors"]:
        lines += ["", "ERRORS (plan is not executable):"]
        lines += [f"  ! {e}" for e in result["errors"]]
    if result["warnings"]:
        lines += ["", "warnings:"]
        lines += [f"  - {w}" for w in result["warnings"]]
    return "\n".join(lines)
