

import os as _os, sys as _sys
# The repo root, found by marker rather than by counting parents, so these run from
# wherever they are filed. They import `world_graph` and `graph_machine` from there.
_d = _os.path.dirname(_os.path.abspath(__file__))
while _d != _os.path.dirname(_d) and not _os.path.isdir(_os.path.join(_d, 'src')):
    _d = _os.path.dirname(_d)
_roots = [_d, _os.path.join(_d, 'omnigibson_runtime')]
_roots += [_f.path for _r in ('src', 'benchmark')
           for _f in _os.scandir(_os.path.join(_d, _r))
           if _f.is_dir() and not _f.name.startswith(('.', '_'))]
for _p in _roots:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import os

import yaml

import omnigibson as og
from omnigibson.action_primitives.starter_semantic_action_primitives import (
    StarterSemanticActionPrimitives,
    StarterSemanticActionPrimitiveSet,
)

# Don't use GPU dynamics for performance boost
# gm.USE_GPU_DYNAMICS = True


def execute_controller(ctrl_gen, env):
    for action in ctrl_gen:
        env.step(action)


def main():
    """
    Demonstrates how to use the action primitives to pick and place an object in an empty scene.

    It loads Rs_int with a robot, and the robot picks and places a bottle of cologne.
    """
    # Load the config
    config_filename = os.path.join(og.example_config_path, "tiago_primitives.yaml")
    config = yaml.load(open(config_filename, "r"), Loader=yaml.FullLoader)

    # Update it to create a custom environment and run some actions
    config["scene"]["scene_model"] = "Rs_int"
    config["scene"]["load_object_categories"] = ["floors", "ceilings", "walls", "coffee_table"]
    config["objects"] = [
        {
            "type": "DatasetObject",
            "name": "cologne",
            "category": "bottle_of_cologne",
            "model": "lyipur",
            "position": [-0.3, -0.8, 0.5],
            "orientation": [0, 0, 0, 1],
        },
        {
            "type": "DatasetObject",
            "name": "table",
            "category": "breakfast_table",
            "model": "rjgmmy",
            "scale": [0.3, 0.3, 0.3],
            "position": [-0.7, 0.5, 0.2],
            "orientation": [0, 0, 0, 1],
        },
    ]

    # Load the environment
    env = og.Environment(configs=config)
    scene = env.scene
    robot = env.robots[0]

    # Let the objects settle
    for _ in range(30):
        og.sim.step()

    # Allow user to move camera more easily

    controller = StarterSemanticActionPrimitives(env, robot, enable_head_tracking=False)

    # Grasp of cologne
    grasp_obj = scene.object_registry("name", "cologne")
    print("Executing controller")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.GRASP, grasp_obj), env)
    print("Finished executing grasp")

    # Place cologne on another table
    print("Executing controller")
    table = scene.object_registry("name", "table")
    execute_controller(controller.apply_ref(StarterSemanticActionPrimitiveSet.PLACE_ON_TOP, table), env)
    print("Finished executing place")
    print("Shutting down OmniGibson cleanly")
    og.shutdown()


if __name__ == "__main__":
    main()
