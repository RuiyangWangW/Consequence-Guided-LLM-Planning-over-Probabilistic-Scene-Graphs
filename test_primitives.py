"""Run one atomic action plan in the simulator, step by step, and check every step.

The plan: cook a potato on a plate and set it on the table.

    NAVIGATE_TO(potato)   GRASP(potato)      NAVIGATE_TO(plate)  PLACE_ON_TOP(plate)
    GRASP(plate)          NAVIGATE_TO(oven)  OPEN(oven)          PLACE_INSIDE(oven)
    CLOSE(oven)           TOGGLE_ON(oven)    TOGGLE_OFF(oven)    OPEN(oven)
    GRASP(plate)          CLOSE(oven)        NAVIGATE_TO(table)  PLACE_ON_TOP(table)

Every primitive is symbolic - the manipulation is resolved by setting object state - but
the navigation is real: the robot plans a route over the eroded traversability map and
drives it. That is the half of the problem a plan can actually get wrong, and the half
worth watching.

Sixteen steps covering eight of the nine primitives - all but RELEASE, which this plan
never needs because each placement empties the hand.

Every step verifies the resulting world state - object in hand, `OnTop`, `Inside`, `Open`,
`ToggledOn` - not merely that no exception was raised, so a step cannot pass by silently
doing nothing. The plan runs exactly as written, with nothing inserted between steps to
tidy up after a failure.

Only what the plan needs is loaded: floors, walls, the oven, the counter and the table.
The potato and the plate are injected and dropped onto the counter, well apart, so the
first two NAVIGATE_TO steps are genuinely different drives.

    source ~/safety_filter/setup_behavior_env.sh

    OMNIGIBSON_GPU_ID=1 CUROBO_GPU_ID=1 python test_primitives.py --bev --video out.mp4

Both variables matter: CuRobo defaults to cuda:0 whatever Isaac is told, so two runs in
parallel collide on GPU 0 without `CUROBO_GPU_ID`. `--bev` records a bird's eye view that
follows the robot; `--robot-camera` records from the robot's own head camera instead.
"""

import argparse
import math
import os

import yaml

# CuRobo device pinning MUST happen before omnigibson is imported.
#
# `CuRoboMotionGenerator` defaults to device="cuda:0" (curobo.py:74) and the starter
# primitives never pass a device, so every run puts its motion planner on GPU 0 no matter
# what OMNIGIBSON_GPU_ID says - that variable only steers Isaac's renderer and physics.
# Two runs in parallel therefore both claim ~13GB of the same card and the second dies
# with CUDA out of memory.
#
# Two defaults have to move together, or tensors end up split across devices and CuRobo
# segfaults in its compiled kernels with no Python traceback:
#
#   1. CuRoboMotionGenerator(device=...)
#   2. TensorDeviceType.device, whose dataclass default is torch.device("cuda", 0)
#
# (2) is the subtle one: the generated __init__ captured that default when the class was
# created, so rebinding the class attribute does nothing - the value handed out lives in
# __init__.__defaults__. And the ordering matters as much as the patch, because
# OmniGibson constructs curobo objects while its own modules import.
def _pin_curobo(gpu):
    if gpu in (None, 0, "0"):
        return False
    import torch as th
    from curobo.types.base import TensorDeviceType

    dev = th.device("cuda", int(gpu))
    TensorDeviceType.device = dev
    TensorDeviceType.__dataclass_fields__["device"].default = dev
    TensorDeviceType.__init__.__defaults__ = tuple(
        dev if (isinstance(v, th.device) and v.type == "cuda") else v
        for v in TensorDeviceType.__init__.__defaults__
    )
    assert TensorDeviceType().device == dev, "failed to repoint TensorDeviceType"

    import omnigibson.action_primitives.curobo as curobo_mod

    original = curobo_mod.CuRoboMotionGenerator

    class PinnedCuRoboMotionGenerator(original):
        def __init__(self, *a, **kw):
            kw.setdefault("device", f"cuda:{int(gpu)}")
            super().__init__(*a, **kw)

    curobo_mod.CuRoboMotionGenerator = PinnedCuRoboMotionGenerator
    import omnigibson.action_primitives.starter_semantic_action_primitives as sap

    sap.CuRoboMotionGenerator = PinnedCuRoboMotionGenerator
    return True


_CUROBO_GPU = os.environ.get("CUROBO_GPU_ID")
if _CUROBO_GPU is not None and _pin_curobo(_CUROBO_GPU):
    print(f"pinned CuRobo to cuda:{_CUROBO_GPU}")

import omnigibson as og
import omnigibson.utils.transform_utils as T
from omnigibson import object_states


SCENE = "house_single_floor"

# Chosen offline by scene_setup.py: the surface with the most standing room that shares
# the fridge's connected component of floor, so what is picked up can actually be carried
# there. Scenes absent from this table have no usable support - Rs_int is one: its fridge
# sits in a different component from where the robot spawns, so it cannot be reached.
SCENE_SETUP = {
    # Chosen by scene_setup.py. These are the only two scenes in the dataset where a
    # Tiago can reach a fridge from where it spawns - at its measured 0.892 m footprint a
    # single narrow doorway seals the kitchen off in every other one, and 19 are rejected
    # for exactly that. Re-run `python scene_setup.py --all` to re-check.
    "house_single_floor": {
        "support": "countertop_kelker_0",
        "support_category": "countertop",
        "fridge": "fridge_dszchb_0",
        "oven": "oven_ffitak_0",
        # Where the plate ends up. Picked by measuring standing room around every table
        # and desk in the scene against the eroded map: this one has 226 free pixels in
        # the 0.5-1.6 m band with the nearest floor 0.90 m out, in the same connected
        # component as the spawn. breakfast_table_rhjoby_0 is the obvious-looking choice
        # and the wrong one - 5 pixels, nearest floor 1.53 m.
        "table": "coffee_table_rlsebe_0",
        "table_category": "coffee_table",
    },
    "Pomaria_1_int": {
        "support": "countertop_tpuwys_2",
        "support_category": "countertop",
        "fridge": "fridge_xyejdx_0",
        "oven": "oven_fexqbj_2",
        "table": None,          # not surveyed; run scene_setup.py before using this scene
        "table_category": "breakfast_table",
    },
}

# Objects we add ourselves, rather than hunting for one already in the scene. Each is
# dropped onto the scene furniture named above with OnTop.set_value, which samples a
# valid resting pose - no hardcoded coordinates to go stale when a model or scene
# changes. `on` refers to a key of the SCENE_SETUP entry.
INJECTED = [
    # Set well apart along the counter edge so the plan's first NAVIGATE_TO(potato) and
    # NAVIGATE_TO(plate) are genuinely different drives rather than the same spot twice.
    #
    # The plate is 0.209 m across, against the potato's 0.077 m.
    #
    # A 0.162 m plate was tried, to make it easier to carry level, and PLACE_ON_TOP then
    # failed every time with "Could not find a position to put this object in the desired
    # relation" - including with the potato held perfectly flat, so it was the plate being
    # too small a target and not the approach angle. 0.209 m samples fine. Carrying it is
    # no longer the problem it was: the carry pose now turns only the wrist, leaving the
    # upper arm and forearm exactly where the tucked pose puts them.
    {"name": "potato", "category": "potato", "model": "lqjear", "on": "support",
     "lateral": -0.45},
    {"name": "plate", "category": "plate", "model": "akfjxx", "on": "support",
     "lateral": 0.45},
]



def build_config(search=False, camera_size=None):
    """The tiago_primitives config, with one load policy for every scene.

    `search` adds `seg_instance`, which names the object each pixel belongs to and is
    what decides whether an object has genuinely been seen. It is off by default because
    it costs render time on every step of every drive.
    """
    path = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # The scene to load. Without this the yaml's own `scene_model: Rs_int` wins and
    # --scene silently changes nothing but the furniture names looked up afterwards,
    # which then fail to resolve.
    config["scene"]["scene_model"] = SCENE

    # One load policy for every scene. A restricted load is what keeps the scene under
    # CuRobo's hardcoded 2048-mesh collision cache (curobo.py:143); overrunning it
    # corrupts GPU memory and every primitive dies with "CUDA error: an illegal memory
    # access was encountered", which is how Wainscott_0_int (2747 meshes) fails. These
    # categories cover the appliances under test plus the structure needed to navigate.
    config["scene"].pop("not_load_object_categories", None)
    setup = SCENE_SETUP[SCENE]
    config["scene"]["load_object_categories"] = [
        "floors", "walls",                    # the structure the robot drives through
        "oven",                               # what the plan opens, fills and toggles
        setup["support_category"],            # what the injected objects rest on
        setup["table_category"],              # where the plate is finally set down
    ]
    # Spawned at an arbitrary height and dropped onto real furniture at runtime.
    config["objects"] = [
        {"type": "DatasetObject", "name": o["name"], "category": o["category"],
         "model": o["model"], "position": [0.0, 0.0, 10.0 + 0.5 * i],
         "orientation": [0, 0, 0, 1]}
        for i, o in enumerate(INJECTED)
    ]

    for robot_cfg in config.get("robots", []):
        if search:
            mods = list(robot_cfg.get("obs_modalities") or ["rgb"])
            # No depth: occlusion now comes from the traversability map, because the
            # head's default downward pitch made every depth ray end on the floor at
            # ~2.5 m. One render pass fewer on every step of every drive.
            for m in ("rgb", "seg_instance"):
                if m not in mods:
                    mods.append(m)
            robot_cfg["obs_modalities"] = mods
        # On, so a symbolically welded object is not torn loose by the per-step
        # assisted-grasp logic during settling. RELEASE turns it off for its own duration
        # (see primitive_patches._execute_release), since that same logic is what lets go
        # of a physically grasped object when the gripper opens.
        robot_cfg["disable_grasp_handling"] = True
        # tiago_primitives.yaml sets the robot camera to 128x128 - fine for a policy,
        # useless to watch. Override it here, in the config: resizing a live VisionSensor
        # rebuilds its render product and segfaults with no Python traceback.
        vision = robot_cfg.setdefault("sensor_config", {}).setdefault("VisionSensor", {})
        height, width = camera_size or (720, 1280)
        vision.setdefault("sensor_kwargs", {}).update(
            {"image_height": height, "image_width": width})

    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", help="record the whole run to this mp4")
    parser.add_argument("--every", type=int, default=3, help="record 1 frame per N steps")
    parser.add_argument("--probe", action="store_true",
                        help="do not run the plan; walk the base out from each object "
                             "along the counter's normal and report, at each distance, "
                             "what the map says and what the collision checker says")
    parser.add_argument("--scene", default="house_single_floor",
                        help="scene to test in. house_single_floor is the only one where "
                             "a Tiago can reach a fridge from its spawn - see the scene "
                             "measurements in the README")
    parser.add_argument("--bev", action="store_true",
                        help="record a bird's eye view that follows the robot, instead of "
                             "the robot's own camera or the fixed viewer")
    parser.add_argument("--bev-height", type=float, default=6.0,
                        help="how far above the robot the bird's eye camera sits, metres")
    parser.add_argument("--bev-tilt", type=float, default=0.0,
                        help="degrees from straight down. 0, the default, is a true "
                             "vertical plan view directly above the robot; larger angles "
                             "pull the camera back into a chase view")
    parser.add_argument("--robot-camera", action="store_true",
                        help="record from the robot's own camera instead of the fixed "
                             "viewer camera, so the effect of each action is visible "
                             "from where the robot is looking")
    parser.add_argument("--place-inside-trials", type=int, default=0,
                        help="do not run the plan. Navigate to the oven, open it, and "
                             "repeat PLACE_INSIDE/GRASP this many times, reporting the "
                             "pass rate. One simulator start instead of one per trial.")
    parser.add_argument("--trace", default=None,
                        help="with --search, record map and graph snapshots to this "
                             "json for world_trace.py to animate")
    parser.add_argument("--search", action="store_true",
                        help="do not look objects up in the scene registry. Route every "
                             "NAVIGATE_TO through the low-level navigation controller, "
                             "which drives to the object's room and frontier-searches it, "
                             "and build the world graph from what the camera actually "
                             "sees. Then check the plan offline against that graph.")
    args = parser.parse_args()

    global SCENE
    SCENE = args.scene

    # Searching adds two render passes - segmentation and depth - to every step of every
    # drive, and the search reads them at walking pace rather than needing detail. Drop the
    # robot camera unless it is what is being recorded.
    camera_size = (360, 640) if (args.search and not args.robot_camera) else None
    env = og.Environment(configs=build_config(search=args.search,
                                              camera_size=camera_size))

    # Open the grippers to match cuRobo's default state, then re-baseline the scene.
    # Straight from OmniGibson's own examples/wip/rs_int_primitives_example.py, which is
    # the only shipped example that runs physical primitives in a *furnished* scene - the
    # case we care about. CuRobo plans from the robot's current joint configuration, so a
    # closed gripper it does not expect can make otherwise-valid motions unplannable.
    import torch as th

    robot = env.robots[0]
    for arm_name in robot.gripper_control_idx.keys():
        idx = robot.gripper_control_idx[arm_name]
        robot.set_joint_positions(th.ones_like(idx), indices=idx, normalized=True)
    robot.keep_still()

    for _ in range(5):
        og.sim.step()

    env.scene.update_initial_file()
    env.scene.reset()

    for _ in range(30):
        og.sim.step()

    scene = env.scene

    import primitive_patches

    PSet = primitive_patches.primitive_set()
    controller = primitive_patches.build(env, robot)

    # The low-level navigation controller and the world graph it feeds. The graph starts
    # as the room graph and nothing else - which rooms exist and which connect - so every
    # object edge in it at the end was written by something the camera saw.
    navctl = world = None
    if args.search:
        from graph_machine import GraphMachine, check as graph_check
        from world_trace import Trace
        from nav_controller import NavigationController
        from world_graph import EDGE_TYPES, WorldGraph, room_of_object

        world = WorldGraph.from_scene_file(SCENE)
        print(f"\n[graph] seeded from the room graph: {world.summary()}")
        navctl = NavigationController(env, robot, controller, world)
        # The same effect model the offline checker uses, but editing the live graph as
        # each primitive succeeds. Observation says what the robot found; this says what
        # the robot did, which is why the graph knows the plate is on the table without
        # the robot having to stand there looking at it afterwards.
        live = GraphMachine(world, copy=False)
        # Snapshots for the side-by-side animation. Recording is cheap - a grid copy and
        # an edge list - and rendering happens offline, so a drawing bug cannot kill a
        # seventeen-minute run.
        trace = Trace(args.trace or "figures/world_trace.json")
        driven = {"m": 0.0, "last": None}   # metres travelled, for the frame captions

        def snapshot(event):
            # The scene-wide map, not the current room's. The room maps are masked and
            # swap as the robot walks between rooms, so drawing them makes the picture
            # jump and hides everything discovered elsewhere. The scene map accumulates.
            import omnigibson.utils.transform_utils as _T

            room_map = navctl.scene_map
            if room_map is None:
                room = navctl.room_here()
                room_map = navctl.maps.get(room) if room else None
            pos, orn = robot.get_position_orientation()
            yaw = float(_T.quat2euler(orn)[2])
            here = (float(pos[0]), float(pos[1]))
            if driven["last"] is not None:
                driven["m"] += math.dist(here, driven["last"])
            driven["last"] = here
            trace.add(event, world, omap=room_map,
                      robot_pose=(here[0], here[1], yaw), held=live.held,
                      distance=driven["m"])

        navctl.on_observe = snapshot

    writer = grab_frame = None
    if args.video:
        import imageio

        os.makedirs(os.path.dirname(args.video) or ".", exist_ok=True)
        writer = imageio.get_writer(args.video, fps=30, quality=8)

        if args.robot_camera:
            # The robot's own view, so the effect of each primitive is visible from where
            # the robot is looking - the fridge door swinging open, the apple leaving the
            # gripper - instead of from a fixed corner of the room.
            from omnigibson.sensors.vision_sensor import VisionSensor

            cams = [s for s in robot.sensors.values() if isinstance(s, VisionSensor)]
            cam = next((c for c in cams if "head" in c.name.lower()
                        or "eyes" in c.name.lower()), cams[0] if cams else None)
            if cam is None:
                raise SystemExit("robot has no vision sensor to record from")

            # The camera needs its rgb annotator attached and several render passes before
            # it returns real pixels. Without this the first frames come back a flat grey
            # (std ~12, ~600 unique colours against ~11000 for a real frame) - the render
            # product exists but has not been filled. OmniGibson's own resolution setter
            # ends with `for i in range(4): og.sim.render()` for the same reason.
            cam.add_modality("rgb")
            for _ in range(6):
                og.sim.render()

            probe, _ = cam.get_obs()
            rgb = probe.get("rgb")
            if rgb is None:
                raise SystemExit(f"{cam.name} returned no rgb")
            import torch as _th

            spread = float(_th.std(rgb[..., :3].float()))
            print(f"recording from robot camera: {cam.name} "
                  f"({rgb.shape[1]}x{rgb.shape[0]}, pixel std {spread:.1f})")
            if spread < 5.0:
                print("  WARNING: frames look blank - the camera may not be rendering")

            def grab_frame(_cam=cam):
                obs, _ = _cam.get_obs()
                rgb = obs.get("rgb")
                return None if rgb is None else rgb[..., :3].cpu().numpy().astype("uint8")
        elif args.bev:
            # Bird's eye view that follows the robot: straight down from directly
            # overhead by default, tracking the robot's x/y every captured frame.
            #
            # The camera is repositioned rather than parented to the robot, deliberately.
            # Parenting would inherit the base's yaw and spin the whole scene every time
            # the robot turned, which is unwatchable. Keeping the camera's own orientation
            # fixed means the room stays put and the robot moves within it.
            #
            # Only ever move this camera. Assigning image_width/image_height on a live
            # viewer_camera reallocates render buffers and segfaults with no traceback.
            import math as _math

            import torch as _th
            import omnigibson.utils.transform_utils as _T

            # Hide the ceiling so the camera can see in. It cannot be excluded at load
            # time: interactive_traversable_scene returns
            #     (not_blacklisted and whitelisted and ...) or is_building_structure
            # and ceilings/roof/walls/doors are building structure, so that final `or`
            # overrides even not_load_object_categories. Hiding is also the right tool -
            # this is a rendering concern, and removing the objects would change physics
            # and the traversability the robot navigates by.
            hidden = 0
            for _o in scene.objects:
                if getattr(_o, "category", None) in ("ceilings", "roof"):
                    _o.visible = False
                    hidden += 1

            cam = og.sim.viewer_camera
            cam.add_modality("rgb")
            print(f"recording bird's eye view: {args.bev_height:.1f} m up, "
                  f"{args.bev_tilt:.0f} deg from vertical, {hidden} ceiling parts hidden")

            def grab_frame(_cam=cam, _robot=robot):
                pos, _ = _robot.get_position_orientation()
                tilt = _math.radians(args.bev_tilt)
                # With no tilt the camera sits directly above the robot. Tilting pulls it
                # back along -Y by exactly the amount that keeps the robot centred.
                back = args.bev_height * _math.tan(tilt)
                _cam.set_position_orientation(
                    position=_th.tensor([float(pos[0]),
                                         float(pos[1]) - back,
                                         float(pos[2]) + args.bev_height]),
                    orientation=_T.euler2quat(_th.tensor([tilt, 0.0, 0.0])),
                )
                og.sim.render()
                obs, _ = _cam.get_obs()
                rgb = obs.get("rgb")
                return None if rgb is None else rgb[..., :3].cpu().numpy().astype("uint8")
        else:
            cam = og.sim.viewer_camera
            cam.add_modality("rgb")

            def grab_frame(_cam=cam):
                return _cam.get_obs()[0]["rgb"][..., :3].cpu().numpy().astype("uint8")

    # The furniture is chosen offline by scene_setup.py, which measures standing room
    # and floor connectivity from the shipped floor plan. Ranking it here meant paying a
    # full Isaac startup just to discover the chosen surface was unusable, and the
    # measurement needs no simulator at all. Re-run `python scene_setup.py <scene>` to
    # add a scene or to check one after a dataset change.
    setup = SCENE_SETUP.get(SCENE)
    if setup is None:
        raise SystemExit(
            f"no offline setup for {SCENE}. Run: python scene_setup.py {SCENE}")

    support = scene.object_registry("name", setup["support"])
    if support is None:
        raise SystemExit(f"{SCENE}: {setup['support']} not in the scene - the offline "
                         f"setup is stale, re-run scene_setup.py {SCENE}")

    # Where on the surface matters. OnTop.set_value samples anywhere on top, and an apple
    # sampled against the back wall of a counter is unreachable: the base sampler found
    # 28 collision-free stances and the arm could reach the apple from none of them.
    # Place objects on the edge that faces open floor instead, so the robot can stand in
    # front of the surface and reach across only a little of it.
    import numpy as _np
    import torch as _th3

    _tm = scene._trav_map
    _arr = _tm._erode_trav_map(_th3.clone(_tm.floor_map[0]), robot=robot).cpu().numpy()
    _free = _np.argwhere(_arr > 0)

    def free_side(support_obj, about=None):
        """Unit vector from `about` (default the surface's centre) to the nearest floor.

        Taking it from the centre once and reusing it for every object only works if the
        surface is a straight run. It is not: measured, the potato ended up with standable
        floor 0.58 m from it and the plate 0.86 m, because "towards open floor" from the
        counter's centre is not the direction to open floor at the plate's end of it. The
        robot then has to stand a third of a metre further back to grasp the plate than the
        potato, off the same surface.
        """
        c = support_obj.get_position_orientation()[0] if about is None else about
        cm = _tm.world_to_map(c[:2])
        d = (_free[:, 0] - int(cm[0])) ** 2 + (_free[:, 1] - int(cm[1])) ** 2
        n = _free[int(_np.argmin(d))]
        wx, wy = _tm.map_to_world(_th3.tensor([int(n[0]), int(n[1])]))
        v = _np.array([float(wx) - float(c[0]), float(wy) - float(c[1])])
        norm = float(_np.linalg.norm(v))
        return v / norm if norm > 1e-6 else _np.array([1.0, 0.0])

    for spec in INJECTED:
        obj = scene.object_registry("name", spec["name"])
        if obj is None:
            raise SystemExit(f"{spec['name']} was not spawned - check INJECTED")
        target = scene.object_registry("name", setup[spec["on"]])

        # Land it on the surface first, so the height is whatever the sampler says is
        # valid, then slide it towards the open-floor edge keeping that height.
        if not obj.states[object_states.OnTop].set_value(target, True):
            raise SystemExit(f"could not place {spec['name']} on {target.name}")
        for _ in range(10):
            og.sim.step()

        tc = target.get_position_orientation()[0]
        ext = target.aabb_extent
        keep_z = float(obj.get_position_orientation()[0][2])
        lateral = spec.get("lateral", 0.0)    # spread objects along the edge

        # Spread first, then find the open floor from there, so each object is slid
        # towards the floor nearest to where it will actually sit.
        v0 = free_side(target)
        spread = _th3.tensor([float(tc[0]) - v0[1] * lateral,
                              float(tc[1]) + v0[0] * lateral, keep_z])
        v = free_side(target, about=spread)
        reach = 0.5 * float(_np.hypot(float(ext[0]) * v[0], float(ext[1]) * v[1]))
        off = max(reach - 0.15, 0.0)          # 15 cm in from the lip, so it stays put
        pos = _th3.tensor([float(spread[0]) + v[0] * off,
                           float(spread[1]) + v[1] * off,
                           keep_z])
        obj.set_position_orientation(position=pos)
        obj.keep_still()
        for _ in range(30):
            og.sim.step()

        on_top = obj.states[object_states.OnTop].get_value(target)
        print(f"placed {spec['name']} on {target.name} "
              f"{off:.2f} m towards open floor (OnTop={on_top})")
        if not on_top:
            raise SystemExit(f"{spec['name']} slid off {target.name}")

    chosen_support = support

    def by_category(category):
        found = scene.object_registry("category", category)
        return next(iter(found)) if found else None

    # Objects by name, from the same offline table as the support, so the run uses the
    # exact instances that were checked for standing room and connectivity.
    oven = scene.object_registry("name", setup["oven"])
    table = scene.object_registry("name", setup["table"]) if setup["table"] else None
    potato = scene.object_registry("name", "potato")
    plate = scene.object_registry("name", "plate")

    if world is not None:
        # The task's own objects. Everything the robot finds is still recorded; this only
        # decides what the animation draws, since a kitchen search turns up eight
        # countertops and drawing all eight buries the four the task is about. Set here
        # rather than where the trace is built, because these are not bound until now.
        trace.focus = [o.name for o in (potato, plate, oven, table) if o is not None]

    missing = [n for n, o in (("oven", oven), ("table", table),
                              ("potato", potato), ("plate", plate)) if o is None]
    if missing:
        raise SystemExit(f"scene is missing: {', '.join(missing)}")

    stuck = []          # (child, parent, child-pose-relative-to-parent)

    def stick(child, parent):
        """Make `child` ride on `parent` from now on.

        The plan carries the plate with the potato on it. The potato is a separate rigid
        body resting on the plate by contact alone, so the moment the plate is picked up,
        tilted or swung into the oven it slides off - and the run then measures whether a
        potato stays balanced rather than whether the plan is right.

        This records the offset and `carry_stuck` reapplies it after every step, rather
        than welding the two with a physics joint. A FixedJoint looks like the obvious
        answer and does not work here: the symbolic primitives place things by teleporting
        them (`set_position_orientation`) and settling, so a constrained second body fights
        the teleport and the placement check reads the result as "probably dropped" -
        measured, that is exactly how PLACE_INSIDE failed. A kinematic follow composes with
        teleporting placement because it cannot perturb the solver at all.
        """
        c_pos, c_quat = child.get_position_orientation()
        p_pos, p_quat = parent.get_position_orientation()
        stuck.append((child, parent,
                      T.relative_pose_transform(c_pos, c_quat, p_pos, p_quat)))

        # Out of the physics too, like anything the robot is carrying.
        #
        # Riding on the plate kinematically is not enough on its own while the potato is
        # still a physical body: gravity moves it between steps and `carry_stuck` snaps it
        # back on the next one, which reads as the potato jittering on the plate. With no
        # gravity and no collisions there is nothing to snap back from - it simply goes
        # where it is put. The `OnTop` check has already run by this point, and that check
        # needs contact, so this cannot be done any earlier.
        child.visual_only = True
        print(f"    [stick] {child.name} now rides on {parent.name}, out of the physics")

    def carry_stuck():
        """Put every stuck object back where it belongs on its parent."""
        for child, parent, (rel_pos, rel_quat) in stuck:
            p_pos, p_quat = parent.get_position_orientation()
            pos, quat = T.pose_transform(p_pos, p_quat, rel_pos, rel_quat)
            child.set_position_orientation(position=pos, orientation=quat)

    # How often to map while moving, in simulation steps. Every step would mean a render
    # and a segmentation transfer for each step of a five-thousand-step plan; every tenth
    # is about 0.2 s of robot motion, fine enough that the map fills in smoothly rather
    # than in jumps.
    OBSERVE_EVERY = 10
    ticks = [0]

    def refresh_moved():
        """Re-read the position of whatever the robot is carrying, and its riders.

        The static-world rule says an object's position is recorded once, when it is first
        seen, and not re-read - the world does not move on its own. A carried object is the
        exception the rule is built around: the robot is the thing moving it, and it knows
        where it has taken it. Without this the graph keeps drawing the plate on the
        kitchen counter while the simulator has it in the living room, which makes the
        animation disagree with the run it is supposed to depict.
        """
        if navctl is None or live.held is None:
            return
        for name in live._carried_with(live.held):
            obj = scene.object_registry("name", name)
            if obj is None or name not in world.objects:
                continue
            world.objects[name]["position"] = obj.get_position_orientation()[0].tolist()

    def tick():
        """Called once per simulation step, whichever path is driving."""
        ticks[0] += 1
        carry_stuck()
        if writer is not None and ticks[0] % args.every == 0:
            frame = grab_frame()
            if frame is not None:
                writer.append_data(frame)
        if navctl is not None and ticks[0] % OBSERVE_EVERY == 0:
            navctl.observe_here()
            refresh_moved()
            snapshot("moving")

    def run(primitive, *a):
        """Execute one primitive; return (ok, error_string, sim_steps)."""
        n = 0
        if primitive == "NAVIGATE_TO" and navctl is not None:
            # The high-level action says where to end up. The low-level controller turns
            # it into the drives that get there, searching the room when the object has
            # never been seen. Only the *room* is ground truth; the object itself has to
            # come into frame before anything drives to it.
            target = a[0]
            counter = [0]

            def on_step():
                counter[0] += 1
                tick()

            navctl.step_cb = on_step
            try:
                result = navctl.navigate_to(target.name, room=room_of_object(target, scene))
            except Exception as e:
                return False, f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:500]}", counter[0]
            path = " -> ".join(kind for kind, _ in result.subgoals)
            print(f"    [search] {result.status} via {len(result.subgoals)} sub-goals: {path}")
            if not result.ok:
                return False, f"search ended {result.status}", counter[0]
            return True, None, counter[0]
        try:
            if primitive == "NAVIGATE_TO":
                gen = controller._navigate_near(*a)
            else:
                gen = controller.apply_ref(getattr(PSet, primitive), *a)
            # Hold the potato on the plate every step, placements included.
            #
            # Every step, because the potato rests on the plate by contact alone and left
            # to itself for the hundreds of steps of a drive it simply falls off, snapping
            # back only when the step ends - which is what the video showed.
            #
            # Placements included, because suspending it there was measured and made
            # things worse: PLACE_INSIDE teleports the plate into the oven, and with the
            # follow off the potato stays where it was, drops into the oven doorway and
            # blocks it. CLOSE then failed its post-state check and the run went from
            # 15/16 to 13/16. It does not fix PLACE_INSIDE either way - that step fails
            # because the plate does not stay Inside after settling, which has nothing to
            # do with the potato.
            for action in gen:
                env.step(action)
                n += 1
                tick()
            return True, None, n
        except Exception as e:
            # Unwrap ActionPrimitiveErrorGroup: its own message says nothing useful, the
            # nested attempts carry the real reason. 500 chars keeps the `Additional info`
            # dict, which is where the useful diagnostic lives.
            return False, f"{type(e).__name__}: {str(e).replace(chr(10), ' ')[:500]}", n

    # The plan under test: cook a potato on a plate and set it on the table.
    #
    # Written out as the flat list of atomic primitives a planner would emit. Every
    # NAVIGATE_TO is its own step - the manipulation primitives do not travel, and
    # `_require_near` fails them if no NAVIGATE_TO put the robot there first, so a plan
    # that forgets one is caught at the step that is actually wrong.
    #
    # Note the two steps that act on the oven while the plate is in hand: OPEN before
    # PLACE_INSIDE, and CLOSE after taking the plate back out. Upstream refuses both
    # ("Cannot open or close an object while holding an object"); Tiago has two arms and
    # the alternative is putting the plate on the floor mid-task, so
    # `_hand_allowed_full` lets them through.
    if args.place_inside_trials:
        # PLACE_INSIDE failed 2 runs in 5 with the plate landing within 0.001 m of its
        # target, which is a marginal check rather than a near miss. One full plan per
        # sample would be seventeen minutes each; this exercises the one action in a loop
        # against the same geometry, which is what a flaky check needs to be judged on.
        print("\n" + "=" * 82)
        print(f"PLACE_INSIDE trial: {args.place_inside_trials} repeats on {oven.name}")
        print("=" * 82)

        # Fetch the plate first. It starts on the counter, 3.49 m from where the robot
        # stands at the oven, which is past `_require_near`'s limit - so a loop that
        # grasps before placing fails on its first step every time and never reaches the
        # action under test. Carry it over, and let each iteration end holding it again by
        # taking it back out of the oven.
        for label, prim, target in (("NAVIGATE_TO(plate)", "NAVIGATE_TO", plate),
                                    ("GRASP(plate)", "GRASP", plate),
                                    ("NAVIGATE_TO(oven)", "NAVIGATE_TO", oven),
                                    ("OPEN(oven)", "OPEN", oven)):
            ok, err, _ = run(prim, target)
            print(f"  {label:18s} {'ok' if ok else 'FAILED ' + str(err)[:120]}")
            if not ok:
                break

        passed = 0
        for trial in range(args.place_inside_trials):
            got, err, _ = run("PLACE_INSIDE", oven)
            inside = False
            if got:
                try:
                    # Both readings: the object-state one is what the simulator reports,
                    # and it is unreliable for an object that was `visual_only` through
                    # the placement; the geometric one is BEHAVIOR's own definition of
                    # Inside computed from prim poses. Printing both is what showed they
                    # disagree.
                    # Three readings, because they answer different questions:
                    #   inside   geometry now, after the plate has been released and has
                    #            settled - does it still sit in the cavity?
                    #   reported what the simulator's own object state says
                    # If `inside` is False the plate physically left the cavity and a
                    # failure is correct; if `inside` is True while `reported` is False,
                    # the predicate is misreading a plate that is genuinely in there.
                    inside = bool(controller._inside_now(plate, oven))
                    reported = bool(plate.states[object_states.Inside].get_value(oven))
                    if reported != inside:
                        err = f"(states[Inside]={reported}, settled geometry={inside})"
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
            passed += 1 if inside else 0
            pos = plate.get_position_orientation()[0]
            lo, hi = plate.aabb
            centre = (lo + hi) / 2.0
            print(f"  trial {trial + 1:2d}: Inside={inside!s:5s}  "
                  f"pos ({float(pos[0]):+.3f}, {float(pos[1]):+.3f}, {float(pos[2]):+.3f})  "
                  f"aabb centre z {float(centre[2]):+.3f}"
                  + (f"  {err}" if err else ""))
            # Take it back out so the next iteration starts holding it again.
            if trial + 1 < args.place_inside_trials:
                back = run("GRASP", plate)
                if not back[0]:
                    print(f"  trial {trial + 1:2d}: could not retrieve the plate - "
                          f"{str(back[1])[:120]}")
                    break

        n = args.place_inside_trials
        print("=" * 82)
        print(f"{passed}/{n} PLACE_INSIDE trials satisfied Inside "
              f"({passed / max(1, n):.0%})")
        print("=" * 82)
        og.shutdown()
        return

    if args.probe:
        # Why is the floor directly in front of an object rejected?
        #
        # The map says the robot fits 0.45 m from the potato and 0.47 m from the plate,
        # essentially straight in front of each - 5 degrees off the counter's normal. The
        # robot ends up 0.75 m and 0.83 m away instead, so roughly a quarter of a metre is
        # lost after the map, in CuRobo's collision check. The map only knows the robot's
        # *footprint*; CuRobo checks the whole robot in 3-D, and a worktop with an
        # overhanging lip strikes its upper body while the base still fits underneath.
        #
        # This walks the base outward along the counter's normal and asks at each step,
        # which gives the distance at which the robot actually starts to fit.
        import math as _mp

        import numpy as _np
        import torch as _thp

        tmap_p = scene._trav_map
        eroded_p = tmap_p._erode_trav_map(
            _thp.clone(tmap_p.floor_map[0]), robot=robot).cpu().numpy()
        free_p = _np.argwhere(eroded_p > 0)
        support = scene.object_registry("name", setup["support"])
        sc = support.get_position_orientation()[0]
        scm = tmap_p.world_to_map(sc[:2])
        near_p = free_p[int(_np.argmin((free_p[:, 0] - int(scm[0])) ** 2
                                       + (free_p[:, 1] - int(scm[1])) ** 2))]
        wx, wy = tmap_p.map_to_world(_thp.tensor([int(near_p[0]), int(near_p[1])]))
        nrm = _np.array([float(wx) - float(sc[0]), float(wy) - float(sc[1])])
        nrm /= (_np.linalg.norm(nrm) or 1.0)
        print(f"\ncounter normal ({nrm[0]:+.2f}, {nrm[1]:+.2f})")

        q0 = robot.get_joint_positions().clone()
        for target_obj in (potato, plate):
            oxy = target_obj.get_position_orientation()[0][:2]
            print(f"\n{target_obj.name}: base walked out along the normal")
            print(f"{'distance':>9s}  {'map says':>9s}  {'CuRobo says':>12s}")
            for d in [round(x, 2) for x in _np.arange(0.35, 1.15, 0.05)]:
                px = float(oxy[0]) + nrm[0] * d
                py = float(oxy[1]) + nrm[1] * d
                yaw = _mp.atan2(float(oxy[1]) - py, float(oxy[0]) - px)
                m = tmap_p.world_to_map(_thp.tensor([px, py], dtype=_thp.float32))
                r_, c_ = int(m[0]), int(m[1])
                on_map = (0 <= r_ < eroded_p.shape[0] and 0 <= c_ < eroded_p.shape[1]
                          and eroded_p[r_, c_] > 0)
                j = q0.clone()
                j[robot.base_control_idx] = _thp.tensor([px, py, yaw], dtype=j.dtype)
                bad = controller._motion_generator.check_collisions(
                    j.unsqueeze(0), self_collision_check=False, attached_obj=None).cpu()
                print(f"{d:9.2f}  {'free' if on_map else 'blocked':>9s}  "
                      f"{'COLLIDES' if bool(bad[0]) else 'clear':>12s}")
        og.shutdown()
        return

    checks = [
        ("NAVIGATE_TO (potato)", "NAVIGATE_TO", (potato,), None),
        ("GRASP (potato)", "GRASP", (potato,),
         lambda: controller._get_obj_in_hand() is potato),
        ("NAVIGATE_TO (plate)", "NAVIGATE_TO", (plate,), None),
        ("PLACE_ON_TOP (plate)", "PLACE_ON_TOP", (plate,),
         lambda: potato.states[object_states.OnTop].get_value(plate)),
        ("GRASP (plate)", "GRASP", (plate,),
         lambda: controller._get_obj_in_hand() is plate),
        ("NAVIGATE_TO (oven)", "NAVIGATE_TO", (oven,), None),
        ("OPEN (oven)", "OPEN", (oven,),
         lambda: oven.states[object_states.Open].get_value()),
        ("PLACE_INSIDE (oven)", "PLACE_INSIDE", (oven,),
         lambda: plate.states[object_states.Inside].get_value(oven)),
        ("CLOSE (oven)", "CLOSE", (oven,),
         lambda: not oven.states[object_states.Open].get_value()),
        ("TOGGLE_ON (oven)", "TOGGLE_ON", (oven,),
         lambda: oven.states[object_states.ToggledOn].get_value()),
        ("TOGGLE_OFF (oven)", "TOGGLE_OFF", (oven,),
         lambda: not oven.states[object_states.ToggledOn].get_value()),
        ("OPEN (oven, to take out)", "OPEN", (oven,),
         lambda: oven.states[object_states.Open].get_value()),
        ("GRASP (plate, from oven)", "GRASP", (plate,),
         lambda: controller._get_obj_in_hand() is plate),
        ("CLOSE (oven, after)", "CLOSE", (oven,),
         lambda: not oven.states[object_states.Open].get_value()),
        ("NAVIGATE_TO (table)", "NAVIGATE_TO", (table,), None),
        ("PLACE_ON_TOP (table)", "PLACE_ON_TOP", (table,),
         lambda: plate.states[object_states.OnTop].get_value(table)),
    ]

    # Weld the potato to the plate the moment it is placed there, before the plate is
    # picked up again.
    STICK_AFTER = {"PLACE_ON_TOP (plate)": (lambda: stick(potato, plate))}

    # Observations taken mid-plan, because some edges only exist for a few steps.
    # `object_inside(plate, oven)` is true from the moment the plate is placed until the
    # plate comes back out, but the oven is only *open* - and the plate only visible -
    # between PLACE_INSIDE and CLOSE. Without a look in that window the run never
    # exercises the one edge type the final graph can never show.
    LOOK_AFTER = {"PLACE_INSIDE (oven)"}

    print("\n" + "=" * 82)
    print(f"plan: {len(checks)} atomic actions in {SCENE}")
    print(f"{'action':28s} {'result':8s} {'steps':>6s}  detail")
    print("=" * 82)

    # The plan runs exactly as written. Nothing is inserted between steps to tidy up
    # after a failure: an earlier version released whatever was in the hand before each
    # OPEN/CLOSE/TOGGLE, which kept later checks passing but would have put the plate on
    # the floor halfway through this plan. If a step leaves the world in a state the next
    # step cannot use, that is a result about the plan and it should show.
    results = []
    for label, primitive, prim_args, verify in checks:
        ok, err, n = run(primitive, *prim_args)
        detail = err or ""
        if ok and verify is not None:
            try:
                if not verify():
                    ok, detail = False, "ran without error but post-state is wrong"
            except Exception as e:
                ok, detail = False, f"state check raised {type(e).__name__}: {e}"
        if ok and world is not None:
            # The simulator has already accepted this action, so its effects are fact.
            # A precondition the model rejects is worth printing - it means the model and
            # the world disagree about what was legal - but the edits still apply, because
            # the world is the authority on what happened.
            moving = live._carried_with(live.held) if live.held else set()
            step_result = live.step(len(results), primitive,
                                    prim_args[0].name if prim_args else None)
            # After a placement `held` is None, so the set has to be taken beforehand.
            for name in moving:
                obj = scene.object_registry("name", name)
                if obj is not None and name in world.objects:
                    world.objects[name]["position"] = \
                        obj.get_position_orientation()[0].tolist()
            if not step_result.ok:
                print(f"    [graph] model disagreed: {step_result.reason}")
            elif step_result.edits:
                print(f"    [graph] {', '.join(step_result.edits)}")
            snapshot(label)
        if ok and label in STICK_AFTER:
            STICK_AFTER[label]()
        if ok and world is not None and label in LOOK_AFTER:
            # No scan: the oven door is open and swung out into the room, and turning the
            # base here would drive it into the door.
            room_map = next(iter(reversed(list(navctl.maps.values()))), None)
            if room_map is not None:
                looked = navctl._look(room_map, scan=False)
                held_edges = world.edges_of("object_inside")
                print(f"    [graph] mid-plan look after {label}: saw {len(looked)} "
                      f"objects, plate in frame = {'plate' in looked}; "
                      f"object_inside = {held_edges or 'none'}")
        carry_stuck()
        results.append((label, ok))
        print(f"{label:28s} {'PASS' if ok else 'FAIL':8s} {n:6d}  {detail}")

        if not ok:
            # Where did everything actually end up? A placement that reports "probably
            # dropped" has already put the object somewhere, and knowing where separates
            # "it fell out of the cavity" from "it never got in" - which guessing does not.
            for name, o in (("potato", potato), ("plate", plate),
                            ("oven", oven), ("table", table)):
                if o is None:
                    continue
                pos = o.get_position_orientation()[0]
                lo, hi = o.aabb
                print(f"       {name:6s} at ({float(pos[0]):+.2f}, {float(pos[1]):+.2f}, "
                      f"{float(pos[2]):+.2f})  aabb z {float(lo[2]):.2f}..{float(hi[2]):.2f}")

    n_pass = sum(1 for _, ok in results if ok)
    print("=" * 82)
    print(f"{n_pass}/{len(results)} plan actions succeeded")
    print("=" * 82)

    if world is not None:
        # One last look, so the graph reflects what the plan actually did rather than
        # what it looked like before the final placement. Without it the graph's last
        # word on the plate is wherever it was when the robot last had it in frame.
        last_map = next(iter(reversed(list(navctl.maps.values()))), None)
        if last_map is not None:
            # Deliberately without the turn-in-place scan. During a search the robot scans
            # from frontier cells, which are free floor on the eroded map by construction;
            # here it is parked 0.20 m from the table it just placed on, and turning a
            # 0.72 x 0.61 m base against furniture risks moving the very thing being
            # audited. One view, honestly reported, beats a refresh that disturbs the
            # world it is measuring.
            # Pitched down, and no scan. The plate ends at 0.28 m on the coffee table and
            # the robot stands 0.89 m from its centre; at the head's default -0.45 rad the
            # frame spans about 0.43-1.11 m, so the plate is below its bottom edge from
            # every heading. -0.80 rad drops the frame far enough to include it.
            navctl._look(last_map, scan=False, head_tilt=-0.80)

        # Does the picture match the world it claims to depict? Compare every recorded
        # position and room against the simulator. A mismatch here means the animation is
        # drawing a fiction, which is worse than not drawing it at all.
        print("\n" + "=" * 82)
        print("graph against the simulator")
        print("=" * 82)
        worst, bad_rooms = 0.0, 0
        for name, rec in sorted(world.objects.items()):
            obj = scene.object_registry("name", name)
            if obj is None or rec.get("position") is None:
                continue
            actual = obj.get_position_orientation()[0].tolist()
            drift = max(abs(a - b) for a, b in zip(actual, rec["position"]))
            true_room = room_of_object(obj, scene)
            graph_room = world.room_of(name)
            flag = ""
            if drift > 0.05:
                flag += f"  POSITION off by {drift:.2f} m"
                worst = max(worst, drift)
            if true_room and graph_room and true_room != graph_room:
                flag += f"  ROOM says {graph_room}, actually {true_room}"
                bad_rooms += 1
            if flag:
                print(f"  {name:26s}{flag}")
        print(f"  {len(world.objects)} objects checked; worst position drift "
              f"{worst:.3f} m, {bad_rooms} room mismatches")

        snapshot("final")
        print(f"\n[graph] wrote {trace.save()} ({len(trace.frames)} snapshots)")

        print("\n" + "=" * 82)
        print("world graph, built entirely from what the camera saw")
        print("=" * 82)
        print(f"  {world.summary()}")
        for edge_type in EDGE_TYPES:
            found = world.edges_of(edge_type)
            if edge_type == "room_connect":
                print(f"  {edge_type:14s} {len(found)} (from the room graph)")
                continue
            print(f"  {edge_type:14s} {len(found)}")
            for a, b in found[:12]:
                print(f"       {a} -> {b}")
            if len(found) > 12:
                print(f"       ... and {len(found) - 12} more")

        for room, omap in navctl.maps.items():
            print(f"  searched {omap.summary()}")

        # The graph edit state machine, offline, from what the robot knew before it
        # started: the room graph and nothing else.
        print("\n" + "=" * 82)
        print("graph edit state machine: the same plan, checked without a simulator")
        print("=" * 82)
        plan_pairs = [(primitive, prim_args[0].name if prim_args else None)
                      for _, primitive, prim_args, _ in checks]
        goal = [("on_top", potato.name, plate.name),
                ("on_top", plate.name, table.name)]
        outcome = graph_check(WorldGraph.from_scene_file(SCENE), plan_pairs, goal)
        print(outcome.report())

        # Does the model agree with the world? The machine predicted a final graph from
        # the room graph alone; the simulator produced one from observation. Comparing
        # the goal edges is what says whether the edit model is right.
        print("\n" + "=" * 82)
        print("predicted vs observed")
        print("=" * 82)
        for edge_type, a, b in goal:
            predicted = outcome.graph.has_edge(edge_type, a, b)
            observed = world.has_edge(edge_type, a, b)
            # What the simulator itself says, independent of whether the robot looked.
            # Without this a disagreement is ambiguous: the graph being wrong and the
            # world having changed look identical from the graph alone.
            truth = None
            sa = scene.object_registry("name", a)
            sb = scene.object_registry("name", b)
            state_cls = {"on_top": object_states.OnTop,
                         "object_inside": object_states.Inside}.get(edge_type)
            if sa is not None and sb is not None and state_cls is not None:
                try:
                    truth = bool(sa.states[state_cls].get_value(sb))
                except Exception:
                    truth = None
            mark = "agree" if predicted == observed else "DISAGREE"
            note = ""
            if predicted != observed:
                # One disagreement is not a modelling error. `OnTop` is contact-based, and
                # an object stuck to another is deliberately `visual_only` - no collisions,
                # so it touches nothing and the predicate must read False however squarely
                # it is sitting there. The edge is unobservable by construction, not wrong.
                subject = scene.object_registry("name", a)
                if subject is not None and getattr(subject, "visual_only", False):
                    note = ("  (unobservable: " + a + " is out of the physics, and "
                            "OnTop needs contact)")
                    mark = "expected"
            print(f"  {edge_type}({a}, {b}): predicted {predicted}, "
                  f"observed {observed}, simulator says {truth}  <- {mark}{note}")

    if writer is not None:
        writer.close()
        print(f"wrote {args.video}")

    og.shutdown()


if __name__ == "__main__":
    main()
