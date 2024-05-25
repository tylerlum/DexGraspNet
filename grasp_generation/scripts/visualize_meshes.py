"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


1080 balls of solitude
-------------------------
Demonstrates the use of collision filtering to limit collisions to actors within an environment,
simulate all collisions including between actors in different environments, or simulate no collisions between
actors - they will still collide with the ground plane.

Modes can be set via command line arguments:
    --no_collisions to have no actors colide with other actors
    --all_collisions to have all actors, even those from different environments, collide

Press 'R' to reset the  simulation
"""

import numpy as np
from isaacgym import gymutil
from isaacgym import gymapi
from math import sqrt

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(
    description="Collision Filtering: Demonstrates filtering of collisions within and between environments",
    custom_parameters=[
        {"name": "--all_collisions", "action": "store_true", "help": "Simulate all collisions"},
        {"name": "--no_collisions", "action": "store_true", "help": "Ignore all collisions"}])

# configure sim
sim_params = gymapi.SimParams()
if args.physics_engine == gymapi.SIM_FLEX:
    sim_params.flex.shape_collision_margin = 0.25
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 10
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.substeps = 1
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# load assets
obj_asset_options = gymapi.AssetOptions()
obj_asset_options.override_com = True
obj_asset_options.override_inertia = True
obj_asset_options.density = 500
obj_asset_options.vhacd_enabled = (
    # True  # Convex decomposition is better than convex hull
    False  # Convex decomposition is better than convex hull
)

import pathlib
from tqdm import tqdm
import random
meshdata_root_path = pathlib.Path("../data/rotated_meshdata_stable")
assert meshdata_root_path.exists(), f"Meshdata root path {meshdata_root_path} does not exist"
urdf_paths = sorted([x for x in tqdm(meshdata_root_path.rglob("**/coacd.urdf"), desc="Finding URDFs")])
print(f"Found {len(urdf_paths)} urdf_paths")
MAX_NUM_OBJECTS = 320
print(f"Selecting {MAX_NUM_OBJECTS} objects")
random.seed(18)
urdf_paths = random.sample(urdf_paths, MAX_NUM_OBJECTS)

assets = [
    gym.load_asset(sim, str(urdf_path.parents[0]), urdf_path.name, obj_asset_options)
    for urdf_path in tqdm(urdf_paths, desc="Loading assets")
]
num_envs = len(assets)

# set up the env grid
num_per_row = int(sqrt(num_envs))
env_spacing = 1.25
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

envs = []

# subscribe to spacebar event for reset
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_R, "reset")

# set random seed
np.random.seed(17)

for i in range(num_envs):
    asset = assets[i]

    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # generate random bright color
    c = 0.5 + 0.5 * np.random.random(3)
    color = gymapi.Vec3(c[0], c[1], c[2])

    # create ball pyramid
    pose = gymapi.Transform()
    pose.r = gymapi.Quat(0, 0, 0, 1)
    pose.p = gymapi.Vec3(0, 1.5, 0)

    # Set up collision filtering.
    if args.all_collisions:
        # Everything should collide.
        # Put all actors in the same group, with filtering mask set to 0 (no filtering).
        collision_group = 0
        collision_filter = 0

    elif args.no_collisions:
        # Nothing should collide.
        # Use identical filtering masks for all actors to filter collisions between them.
        # Group assignment doesn't matter in this case.
        # Alternative would be to put each actor in a different group.
        collision_group = 0
        collision_filter = 1

    else:
        # Balls in the same env should collide, but not with balls from different envs.
        # Use one group per env, and filtering masks set to 0.
        collision_group = i
        collision_filter = 0

    ahandle = gym.create_actor(env, asset, pose, None, collision_group, collision_filter)
    num_rbs = gym.get_actor_rigid_body_count(env, ahandle)

    for i in range(num_rbs):
        gym.set_rigid_body_color(env, ahandle, i, gymapi.MESH_VISUAL_AND_COLLISION, color)

gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(20, 5, 20), gymapi.Vec3(0, 1, 0))

# create a local copy of initial state, which we can send back for reset
initial_state = np.copy(gym.get_sim_rigid_body_states(sim, gymapi.STATE_ALL))


while not gym.query_viewer_has_closed(viewer):

    # Get input actions from the viewer and handle them appropriately
    for evt in gym.query_viewer_action_events(viewer):
        if evt.action == "reset" and evt.value > 0:
            gym.set_sim_rigid_body_states(sim, initial_state, gymapi.STATE_ALL)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
