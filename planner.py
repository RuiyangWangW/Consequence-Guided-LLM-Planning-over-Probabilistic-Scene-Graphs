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
#
# `requires` and `effect` are the specification `graph_machine.GraphMachine` enforces,
# written out for the LLM. They are here rather than only in the machine because a planner
# judged against rules it was never shown will keep breaking them: measured, a 7B model
# given only the one-line docs wrote `PLACE_INSIDE(potato)` - passing the object being
# placed rather than the container - and reproduced it on every retry, because nothing in
# the prompt said the argument is the destination. Keep these in step with
# `GraphMachine.step`; `test_graph_machine.py` pins the machine's half.
PRIMITIVES = {
    "GRASP": {
        "takes_object": True, "doc": "Pick an object up",
        "requires": "the hand is empty; the robot is standing at the object; if the "
                    "object is inside a container, that container is open",
        "effect": "the robot is holding it, and it travels with the robot"},
    "PLACE_ON_TOP": {
        "takes_object": True, "doc": "Put down what is held, on top of something",
        "requires": "the robot is standing at the destination and is holding something",
        "effect": "what was held is on top of the destination; the hand is empty"},
    "PLACE_INSIDE": {
        "takes_object": True, "doc": "Put down what is held, inside something",
        "requires": "the robot is standing at the destination, the destination is "
                    "already open, and the robot is holding something",
        "effect": "what was held is inside the destination; the hand is empty"},
    "OPEN": {
        "takes_object": True, "doc": "Open a door, lid or drawer",
        "requires": "the robot is standing at the object, and the object has a door",
        "effect": "it is open"},
    "CLOSE": {
        "takes_object": True, "doc": "Shut a door, lid or drawer",
        "requires": "the robot is standing at the object, and the object has a door",
        "effect": "it is shut"},
    "NAVIGATE_TO": {
        "takes_object": True, "doc": "Drive to an object",
        "requires": "nothing",
        "effect": "the robot is then standing at that object, and at whatever is on or "
                  "inside it. It is no longer standing at what it drove away from"},
    "RELEASE": {
        "takes_object": False, "doc": "Let go of what is held",
        "requires": "nothing",
        "effect": "the hand is empty; what it held is left on the floor of this room"},
    "TOGGLE_ON": {
        "takes_object": True, "doc": "Switch an object on",
        "requires": "the robot is standing at the object, and the object has a switch",
        "effect": "it is on"},
    "TOGGLE_OFF": {
        "takes_object": True, "doc": "Switch an object off",
        "requires": "the robot is standing at the object, and the object has a switch",
        "effect": "it is off"},
}

# Containers a robot must open before placing inside. Used only for warnings: the real
# precondition lives in the simulator, and an object's true openability depends on its
# model, which the scene graph does not carry.
OPENABLE = {
    "fridge", "refrigerator", "freezer", "oven", "microwave", "dishwasher", "washer",
    "clothes_dryer", "dryer", "cabinet", "bottom_cabinet", "top_cabinet", "drawer",
    "trash_can", "box", "carton", "backpack", "briefcase",
}


# Objects with a switch. The third of the three affordance lists, and it lives here with
# the other two so that the validator, the graph machine and the 2-D world all decide what
# an object affords from one place - three copies of this list is three ways to disagree.
TOGGLEABLE = {
    "oven", "stove", "microwave", "dishwasher", "washer", "clothes_dryer", "dryer",
    "coffee_maker", "blender", "toaster", "kettle", "electric_kettle", "lamp",
    "floor_lamp", "table_lamp", "light", "ceiling_light", "television", "standing_tv",
    "shower", "sink", "furniture_sink", "fan", "electric_switch",
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

    actions = "\n\n".join(
        f"  {name}({'object' if s['takes_object'] else ''})  - {s['doc']}\n"
        f"      requires: {s['requires']}\n"
        f"      then:     {s['effect']}"
        for name, s in PRIMITIVES.items()
    )
    prompt = f"""You are a task planner for a household robot in a simulated home.

Produce a sequence of atomic actions that completes the task. You may ONLY use these
actions, exactly as written:

{actions}

{format_for_llm(graph)}

Every action above is checked against its `requires` before it runs. If one fails, the
whole plan is rejected.

Rules:
- Use only the objects listed above. Do not invent objects.
- The number after each object is how confident we are that it is really there. A low
  number means the object may not exist in this house; plan for it only if the task
  requires it.
- **NAVIGATE_TO takes an object, never a room.** `NAVIGATE_TO(kitchen)` is not a step;
  name the thing in the kitchen you are going to touch.
- **NAVIGATE_TO the object you are about to act on, every time.** Being in the same room
  is not enough, and having driven there earlier is not enough - if the robot has driven
  somewhere else since, drive back.
- For PLACE_ON_TOP and PLACE_INSIDE the argument is the **destination**: the surface or
  container being put onto or into. What is being put down is whatever the robot is
  holding, and is never named.
- The robot has one hand. GRASP before any PLACE, and after a PLACE the hand is empty
  again - to move something a second time, GRASP it again.
- OPEN and CLOSE act on the object directly. Do NOT grasp a door, appliance or cabinet
  in order to open it: write OPEN(fridge), never GRASP(fridge).
- To get something back out of a container, NAVIGATE_TO the container, OPEN it, then
  GRASP the object.
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

  Task: heat the pie in the oven, then put it on the counter
  NAVIGATE_TO(pie)
  GRASP(pie)
  NAVIGATE_TO(oven)
  OPEN(oven)
  PLACE_INSIDE(oven)
  CLOSE(oven)
  TOGGLE_ON(oven)
  TOGGLE_OFF(oven)
  OPEN(oven)
  GRASP(pie)
  CLOSE(oven)
  NAVIGATE_TO(counter)
  PLACE_ON_TOP(counter)

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


def get_generator(model_name="Qwen/Qwen2.5-7B-Instruct", adapter=None):
    """Return a cached prompt->text function for this model."""
    key = (model_name, adapter)
    if key not in _GENERATORS:
        _GENERATORS[key] = _local_generator(model_name, adapter)
    return _GENERATORS[key]


def release_generator(model_name=None):
    """Drop a loaded model and give the GPU memory back.

    Comparing models means loading several in one process, and an 8B in fp16 is ~16 GB -
    two of them will not sit on one card. Dropping the closure is not enough on its own,
    because the allocator keeps the freed blocks reserved; `empty_cache` returns them.
    """
    import gc

    # Keys are (model_name, adapter), so a bare model name drops every adapter loaded on
    # top of it too - which is what a caller freeing the card wants.
    for key in [k for k in _GENERATORS if model_name in (None, k[0])]:
        _GENERATORS.pop(key, None)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _local_generator(model_name, adapter=None):
    """Lazily load a local instruct model and return a prompt->text function.

    `adapter` points at a LoRA directory from `finetune_extraction.py`, whose base model
    it names, so a fine-tuned extractor is loaded by adapter path alone.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if adapter:
        import json

        model_name = json.load(open(f"{adapter}/training.json"))["base"]
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
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    def run(prompt, max_new_tokens, temperature=0.0):
        """`temperature` 0 is greedy, which is the right default: one question, one answer.

        Anything above it samples, which is what a *retry* needs. Greedy decoding makes a
        repair loop pointless past the second attempt - measured, a rejected plan came back
        byte-identical five times running, because the model had already given its best
        answer to a prompt it was not persuaded by.
        """
        messages = [{"role": "user", "content": prompt}]
        # Depending on the transformers version this returns either a bare tensor or a
        # BatchEncoding; normalize to a tensor of ids so both work.
        #
        # `enable_thinking=False` is for the Qwen3 family, whose chat template turns on a
        # `<think>...</think>` preamble by default. Left on, the model spends the whole
        # token budget reasoning and the reply that reaches `parse_plan` has no actions in
        # it at all - the plan is not wrong, it never arrives. Templates that do not know
        # the argument reject it, so it is offered and withdrawn.
        try:
            encoded = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt",
                enable_thinking=False,
            )
        except (TypeError, ValueError):
            encoded = tok.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        ids = ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                **({"temperature": temperature, "top_p": 0.9} if temperature > 0 else {}),
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True)
        # Belt and braces: if a thinking block comes back anyway, the answer is what
        # follows it.
        if "</think>" in reply:
            reply = reply.rsplit("</think>", 1)[1]
        return reply

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
