"""Execute a validated plan in OmniGibson and record a video of the run.

Follows `solve_simple_task.py`: same `tiago_primitives.yaml` config, same
`og.Environment` / settle / controller / `apply_ref` sequence. The controller comes from
`primitive_patches.build`: all nine primitives working, with real CuRobo navigation and
symbolic manipulation.
The only additions are grounding the plan's category names onto real scene objects and
writing frames to an mp4.

GPU selection comes from `setup_behavior_env.sh` (OMNIGIBSON_GPU_ID=1). Do not override
it here.

    source ~/safety_filter/setup_behavior_env.sh
    python execute_plan.py --plan plan.json --video figures/run.mp4
"""

import argparse
import json
import os

import yaml

import omnigibson as og

# Structural scene elements the planner may name; navigating to them is meaningless.
STRUCTURAL = {"floor", "floors", "wall", "walls", "ceiling", "ceilings", "door", "room"}


def ground_plan(plan_steps, scene, graph=None, verbose=True):
    """Map planned category names onto real objects in the loaded scene.

    The planner names *categories* (`laptop`) because that is what the RSN predicts;
    the simulator holds *instances* (`laptop_nvulcs_0`). Room ids are grounded via an
    object the scene graph placed in that room, since the action set has no room-level
    navigation primitive.
    """
    by_category = {}
    for obj in scene.objects:
        by_category.setdefault(obj.category, []).append(obj)

    def find(name):
        if name in by_category:
            return by_category[name][0]
        hits = [c for c in by_category if name in c.split("_") or c.startswith(name)]
        return by_category[sorted(hits, key=len)[0]][0] if hits else None

    rooms = set((graph or {}).get("rooms", {}))
    room_proxy = {}
    for obj_name, info in (graph or {}).get("objects", {}).items():
        room_proxy.setdefault(info["room"], []).append(obj_name)

    grounded, missing = [], []
    for step in plan_steps:
        name = step.get("object")
        if name is None:  # RELEASE takes no argument
            grounded.append({**step, "instance": None})
            continue
        if name in STRUCTURAL:
            missing.append((name, "structural element, not a graspable object"))
            continue
        if name in rooms:
            proxy = next((find(o) for o in room_proxy.get(name, []) if find(o)), None)
            if proxy is None:
                missing.append((name, "no grounded object in this room"))
                continue
            grounded.append({**step, "instance": proxy})
            if verbose:
                print(f"  grounded {name:20s} -> via {proxy.name} (room proxy)")
            continue
        obj = find(name)
        if obj is None:
            missing.append((name, "no instance of this category in the loaded scene"))
            continue
        grounded.append({**step, "instance": obj})
        if verbose:
            print(f"  grounded {name:20s} -> {obj.name} ({obj.category})")
    return grounded, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="JSON from pipeline.py --json")
    parser.add_argument("--video", default="figures/run.mp4")
    parser.add_argument("--every", type=int, default=2, help="record 1 frame per N steps")
    parser.add_argument("--curobo-batch-size", type=int, default=3,
                        help="base poses validated per sampling round. OmniGibson's "
                             "default of 3 rarely finds a pose near a wall-mounted "
                             "object; see primitive_patches")
    parser.add_argument("--explore", action="store_true",
                        help="find objects by searching, instead of looking them up in "
                             "the scene registry. The robot goes to the room the RSN "
                             "considers most likely, looks around, walks frontiers until "
                             "the room is explored, and falls back to the next most "
                             "likely room. See exploration.py")
    parser.add_argument("--robot-camera", action="store_true",
                        help="record from the robot's own camera rather than the viewer")
    parser.add_argument("--full-scene", action="store_true",
                        help="load every object category, not just the plan's. Slower to "
                             "start, but navigation can fail in a stripped scene")
    args = parser.parse_args()

    with open(args.plan) as f:
        saved = json.load(f)
    steps = saved["plan"]["steps"]
    scene_model = saved["scene"]

    # Only the categories this plan actually touches, exactly as the reference script
    # restricts its load. Loading the full scene is far heavier and is not needed.
    wanted = {s["object"] for s in steps if s.get("object")}
    graph = saved.get("graph", {})
    for room in list(wanted):
        if room in graph.get("rooms", {}):
            wanted.discard(room)
            wanted.update(
                o for o, i in graph.get("objects", {}).items() if i["room"] == room
            )
    # Restricting the load keeps startup fast, but a sparse scene can defeat navigation:
    # `_sample_pose_near_object` requires a pose that is in the object's room *and*
    # collision-free and reachable, and a room stripped of its furniture can fail that.
    # `--full-scene` loads everything.
    categories = None if args.full_scene else ["floors", "ceilings", "walls"] + sorted(wanted)

    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)
    config["scene"]["scene_model"] = scene_model
    if args.explore:
        # Active search needs the camera to actually see things, and seg_instance is what
        # names the object each pixel belongs to.
        for robot_cfg in config.get("robots", []):
            mods = list(robot_cfg.get("obs_modalities") or ["rgb"])
            for m in ("rgb", "depth_linear", "seg_instance"):
                if m not in mods:
                    mods.append(m)
            robot_cfg["obs_modalities"] = mods
    config["scene"]["load_object_categories"] = categories
    # Leave the viewer camera exactly as tiago_primitives.yaml configures it (1280x720).
    # Both changing render.viewer_width/height here and reassigning image_width /
    # image_height on the live og.sim.viewer_camera segfault inside the renderer, with
    # no Python traceback to point at the cause.

    print(f"task:  {saved.get('task', '')}")
    print(f"scene: {scene_model}")
    print(f"load:  {'entire scene' if categories is None else ', '.join(categories)}")
    print(f"plan:  {len(steps)} actions\n")

    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Initialize the articulation view before anything reads the robot's joints.
    # Without this the first get_joint_positions() inside the action primitives
    # returns None ("'NoneType' object has no attribute 'view'").
    env.reset()

    # Let the objects settle
    for _ in range(30):
        og.sim.step()

    print("grounding plan against the loaded scene:")
    grounded, missing = ground_plan(steps, scene, graph)
    for name, reason in missing:
        print(f"  UNGROUNDED {name}: {reason}")
    print()

    import imageio

    os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
    writer = imageio.get_writer(args.video, fps=30, quality=8)
    cam = og.sim.viewer_camera
    cam.add_modality("rgb")

    import primitive_patches

    PSet = primitive_patches.primitive_set()
    controller = primitive_patches.build(env, robot, args.curobo_batch_size)

    def execute_controller(ctrl_gen, env, tag):
        n = 0
        for action in ctrl_gen:
            env.step(action)
            n += 1
            if n % args.every == 0:
                writer.append_data(cam.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype("uint8"))
        return n

    results = []
    for i, step in enumerate(grounded, 1):
        primitive = getattr(PSet, step["action"])
        tag = f"{step['action']}({step['object'] or ''})"
        print(f"[{i}/{len(grounded)}] {tag}")
        prim_args = () if step["instance"] is None else (step["instance"],)
        error = None
        try:
            if step["action"] == "NAVIGATE_TO":
                # Drive to the object, nothing more. GRASP and PLACE run their own final
                # approach against the pose they actually intend to use.
                gen = controller._navigate_near(*prim_args)
            else:
                gen = controller.apply_ref(primitive, *prim_args)
            n = execute_controller(gen, env, tag)
        except Exception as e:
            # ActionPrimitiveErrorGroup wraps one error per attempt; its first line is
            # just "an error occurred during each attempt", so keep the whole message -
            # the useful reason (planning failure, unreachable pose, grasp slipped) is
            # in the nested entries.
            n, error = 0, f"{type(e).__name__}: {str(e)}"
        results.append({"step": i, "action": tag, "sim_steps": n, "error": error})
        print(f"        {'FAILED: ' + error if error else f'ok ({n} sim steps)'}")

    writer.close()
    n_ok = sum(1 for r in results if not r["error"])
    print(f"\nexecuted {n_ok}/{len(results)} actions successfully")
    print(f"wrote {args.video}")

    out_path = os.path.splitext(args.plan)[0] + "_execution.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)

    print("Shutting down OmniGibson cleanly")
    og.shutdown()


if __name__ == "__main__":
    main()
