from utils.hand_model import HandModel
from utils.object_model import ObjectModel
import torch
from typing import Union, Dict, Any, Tuple, Optional, List
import numpy as np
from collections import defaultdict

from enum import Enum, auto


class AutoName(Enum):
    # https://docs.python.org/3.9/library/enum.html#using-automatic-values
    def _generate_next_value_(name, start, count, last_values):
        return name


class OptimizationMethod(AutoName):
    DESIRED_PENETRATION_DEPTH = auto()
    DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS = auto()
    DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP = auto()


def compute_fingers_center(
    hand_model: HandModel,
) -> torch.Tensor:
    # (batch_size, num_fingers, 3)
    fingertip_positions = compute_fingertip_mean_contact_positions(
        joint_angles=hand_model.hand_pose[:, 9:], hand_model=hand_model
    )

    # (batch_size, 3)
    fingers_center = fingertip_positions.mean(dim=1)
    return fingers_center


def compute_optimized_joint_angle_targets(
    optimization_method: OptimizationMethod,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> Tuple[torch.Tensor, List[float], List[Dict[str, Any]]]:
    # TODO: Many of the parameters here are hardcoded
    original_hand_pose = hand_model.hand_pose.detach().clone()
    joint_angle_targets_to_optimize = (
        original_hand_pose.detach().clone()[:, 9:].requires_grad_(True)
    )

    losses = []
    debug_infos = []
    if optimization_method == OptimizationMethod.DESIRED_PENETRATION_DEPTH:
        N_ITERS = 100
        cached_target_points = None
        cached_contact_nearest_point_indexes = None
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_penetration_depth(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=cached_target_points,
                cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
                dist_thresh_to_move_finger=0.01,
                desired_penetration_depth=0.005,
                return_debug_info=True,
            )
            if cached_target_points is None:
                cached_target_points = debug_info["target_points"]
            if cached_contact_nearest_point_indexes is None:
                cached_contact_nearest_point_indexes = debug_info[
                    "contact_nearest_point_indexes"
                ]
            grad_step_size = 5
            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_ONE_STEP
    ):
        N_ITERS = 1
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                dist_thresh_to_move_finger=0.001,
                dist_move_link=0.001,
                return_debug_info=True,
            )
            grad_step_size = 500

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    ):
        N_ITERS = 100
        # Use cached target and indices to continue moving the same points toward the same targets for each iter
        # Otherwise, would be moving different points to different targets each iter
        cached_target_points = None
        cached_contact_nearest_point_indexes = None
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=cached_target_points,
                cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
                dist_thresh_to_move_finger=0.01,
                dist_move_link=0.01,
                return_debug_info=True,
            )
            if cached_target_points is None:
                cached_target_points = debug_info["target_points"]
            if cached_contact_nearest_point_indexes is None:
                cached_contact_nearest_point_indexes = debug_info[
                    "contact_nearest_point_indexes"
                ]
            grad_step_size = 5

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_FINGERS_CENTER_ONE_STEP
    ):
        N_ITERS = 1
        num_links = len(hand_model.mesh)
        target_points = compute_fingers_center(
            hand_model=hand_model, object_model=object_model, device=device
        )
        target_points = target_points.reshape(-1, 1, 3).repeat(1, num_links, 1)
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=target_points,
                dist_thresh_to_move_finger=0.001,
                dist_move_link=0.001,
                return_debug_info=True,
            )
            grad_step_size = 5

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    elif (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_FINGERS_CENTER_MULTIPLE_STEP
    ):
        N_ITERS = 100
        # Use cached target and indices to continue moving the same points toward the same targets for each iter
        # Otherwise, would be moving different points to different targets each iter
        num_links = len(hand_model.mesh)
        cached_target_points = compute_fingers_center(
            hand_model=hand_model, object_model=object_model, device=device
        )
        cached_target_points = cached_target_points.reshape(-1, 1, 3).repeat(
            1, num_links, 1
        )

        cached_contact_nearest_point_indexes = None
        for i in range(N_ITERS):
            loss, debug_info = compute_loss_desired_dist_move(
                joint_angle_targets_to_optimize=joint_angle_targets_to_optimize,
                hand_model=hand_model,
                object_model=object_model,
                device=device,
                cached_target_points=cached_target_points,
                cached_contact_nearest_point_indexes=cached_contact_nearest_point_indexes,
                dist_thresh_to_move_finger=0.01,
                dist_move_link=0.01,
                return_debug_info=True,
            )
            if cached_target_points is None:
                cached_target_points = debug_info["target_points"]
            if cached_contact_nearest_point_indexes is None:
                cached_contact_nearest_point_indexes = debug_info[
                    "contact_nearest_point_indexes"
                ]
            grad_step_size = 1

            loss.backward(retain_graph=True)

            with torch.no_grad():
                joint_angle_targets_to_optimize -= (
                    joint_angle_targets_to_optimize.grad * grad_step_size
                )
                joint_angle_targets_to_optimize.grad.zero_()
            losses.append(loss.item())
            debug_infos.append(debug_info)

    else:
        raise NotImplementedError

    # Update hand pose parameters
    new_hand_pose = original_hand_pose.detach().clone()
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
    hand_model.set_parameters(new_hand_pose)

    return joint_angle_targets_to_optimize, losses, debug_infos

def compute_fingertip_mean_contact_positions(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
) -> torch.Tensor:
    batch_size = joint_angles.shape[0]
    num_fingers = hand_model.num_fingers

    # Each link has a set of contact candidates
    link_name_to_contact_candidates = compute_link_name_to_contact_candidates(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )
    # Merge contact candidates for links associated with same fingertip
    fingertip_name_to_contact_candidates = compute_fingertip_name_to_contact_candidates(
        link_name_to_contact_candidates=link_name_to_contact_candidates,
    )

    # Iterate in deterministic order
    fingertip_names = sorted(list(fingertip_name_to_contact_candidates.keys()))
    fingertip_mean_positions = []
    for i, fingertip_name in enumerate(fingertip_names):
        contact_candidates = fingertip_name_to_contact_candidates[fingertip_name]
        num_contact_candidates_this_link = contact_candidates.shape[1]
        assert contact_candidates.shape == (batch_size, num_contact_candidates_this_link, 3)

        fingertip_mean_positions.append(contact_candidates.mean(dim=1))

    fingertip_mean_positions = torch.stack(fingertip_mean_positions, dim=1)
    assert fingertip_mean_positions.shape == (batch_size, num_fingers, 3)
    return fingertip_mean_positions


def compute_link_name_to_contact_candidates(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
) -> Dict[str, torch.Tensor]:
    batch_size = joint_angles.shape[0]

    current_status = hand_model.chain.forward_kinematics(joint_angles)
    link_name_to_contact_candidates = {}
    for i, link_name in enumerate(hand_model.mesh):
        # Compute contact candidates
        untransformed_contact_candidates = hand_model.mesh[link_name][
            "contact_candidates"
        ]
        if len(untransformed_contact_candidates) == 0:
            continue

        contact_candidates = (
            current_status[link_name]
            .transform_points(untransformed_contact_candidates)
            .expand(batch_size, -1, 3)
        )
        contact_candidates = contact_candidates @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)

        assert contact_candidates.shape == (
            batch_size,
            len(untransformed_contact_candidates),
            3,
        )
        link_name_to_contact_candidates[link_name] = contact_candidates

    return link_name_to_contact_candidates


def compute_fingertip_name_to_contact_candidates(
    link_name_to_contact_candidates: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    # HACK: hardcoded
    # Merge links associated with same fingertip
    fingertip_keywords = ["link_3.0", "link_7.0", "link_11.0", "link_15.0"]
    fingertip_name_to_contact_candidates = {}
    for fingertip_keyword in fingertip_keywords:
        merged_contact_candidates = []
        for link_name, contact_candidates in link_name_to_contact_candidates.items():
            batch_size, n_contact_candidates, _ = contact_candidates.shape
            assert contact_candidates.shape == (batch_size, n_contact_candidates, 3)

            if fingertip_keyword in link_name:
                merged_contact_candidates.append(contact_candidates)

        fingertip_name_to_contact_candidates[fingertip_keyword] = torch.cat(
            merged_contact_candidates, dim=1
        )
    return fingertip_name_to_contact_candidates


def compute_closest_contact_point_info(
    joint_angles: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # For each link, find the closest point on the object
    link_name_to_contact_candidates = compute_link_name_to_contact_candidates(
        joint_angles=joint_angles,
        hand_model=hand_model,
    )

    # Merge links associated with same fingertip
    fingertip_name_to_contact_candidates = compute_fingertip_name_to_contact_candidates(
        link_name_to_contact_candidates=link_name_to_contact_candidates,
    )

    # To be populated
    num_fingers = len(fingertip_name_to_contact_candidates)
    batch_size = joint_angles.shape[0]
    all_hand_contact_nearest_points = torch.zeros((batch_size, num_fingers, 3)).to(
        device
    )
    all_nearest_object_to_hand_directions = torch.zeros(
        (batch_size, num_fingers, 3)
    ).to(device)
    all_nearest_distances = torch.zeros((batch_size, num_fingers)).to(device)
    all_hand_contact_nearest_point_indices = (
        torch.zeros((batch_size, num_fingers)).long().to(device)
    )

    fingertip_names = sorted(list(fingertip_name_to_contact_candidates.keys()))
    for i, fingertip_name in enumerate(fingertip_names):
        contact_candidates = fingertip_name_to_contact_candidates[fingertip_name]
        n_contact_candidates = contact_candidates.shape[1]
        assert contact_candidates.shape == (batch_size, n_contact_candidates, 3)

        # From cal_distance, interiors are positive dist, exteriors are negative dist
        # Normals point from object to hand
        (
            distances_interior_positive,
            object_to_hand_directions,
        ) = object_model.cal_distance(contact_candidates)
        distances_interior_negative = (
            -distances_interior_positive
        )  # Large positive distance => far away
        nearest_point_index = distances_interior_negative.argmin(dim=1)
        nearest_distances = torch.gather(
            input=distances_interior_positive,
            dim=1,
            index=nearest_point_index.unsqueeze(1),
        )
        hand_contact_nearest_points = torch.gather(
            inputs=contact_candidates,
            dim=1,
            index=nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
        )
        nearest_object_to_hand_directions = torch.gather(
            inputs=object_to_hand_directions,
            dim=1,
            index=nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3),
        )

        assert hand_contact_nearest_points.shape == (batch_size, 1, 3)
        assert nearest_object_to_hand_directions.shape == (batch_size, 1, 3)
        assert nearest_distances.shape == (batch_size, 1)
        assert nearest_point_index.shape == (batch_size,)

        # Update tensors
        all_hand_contact_nearest_points[:, i : i + 1, :] = hand_contact_nearest_points
        all_nearest_object_to_hand_directions[
            :, i : i + 1, :
        ] = nearest_object_to_hand_directions
        all_nearest_distances[:, i : i + 1] = nearest_distances
        all_hand_contact_nearest_point_indices[:, i] = nearest_point_index

    return (
        all_hand_contact_nearest_points,
        all_nearest_object_to_hand_directions,
        all_nearest_distances,
        all_hand_contact_nearest_point_indices,
    )


def compute_fingertip_targets_and_hand_contact_nearest_point_indices(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Define both the fingertip targets and the indices of the contact points on the hand that should move towards those targets

    (
        hand_contact_nearest_points,
        nearest_object_to_hand_directions,
        nearest_distances,
        hand_contact_nearest_point_indices,
    ) = compute_closest_contact_point_info(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )

    optimization_method = (
        OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    )
    if (
        optimization_method
        == OptimizationMethod.DESIRED_DIST_TOWARDS_OBJECT_SURFACE_MULTIPLE_STEPS
    ):
        fingertip_targets = (
            hand_contact_nearest_points - nearest_object_to_hand_directions * 0.01
        )
    elif optimization_method == OptimizationMethod.DESIRED_PENETRATION_DEPTH:
        fingertip_targets = (
            hand_contact_nearest_points
            - nearest_object_to_hand_directions * (nearest_distances[..., None] + 0.05)
        )
    else:
        raise NotImplementedError

    return fingertip_targets, hand_contact_nearest_point_indices


def compute_grasp_orientations(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    object_model: ObjectModel,
    device: torch.device,
) -> torch.Tensor:
    # Can't just compute_grasp_dirs because we need to know the orientation of the fingers
    # Each finger has a rotation matrix [x, y, z] where x y z are column vectors
    #    * z is direction the fingertip moves
    #    * y is direction "up" along finger (from finger center to fingertip), modified to be perpendicular to z
    #    * x is direction "right" along finger (from finger center to fingertip), modified to be perpendicular to z and y
    # if y.cross(z) == 0, then need backup

    fingertip_targets, hand_
    (
        hand_contact_nearest_points,
        nearest_object_to_hand_directions,
        nearest_distances,
        hand_contact_nearest_point_indices,
    ) = compute_closest_contact_point_info(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
        object_model=object_model,
        device=device,
    )

    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)
    return grasp_orientations

def computer_fingertip_targets(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
) -> torch.Tensor:
    batch_size = joint_angles_start.shape[0]
    num_fingers = hand_model.num_fingers
    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

    grasp_directions = grasp_orientations[:, :, :, 2]
    assert grasp_directions.shape == (batch_size, num_fingers, 3)

    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (batch_size, num_fingers, 3)

    fingertip_targets = fingertip_mean_positions + grasp_directions * 0.01
    return fingertip_targets


def compute_optimized_joint_angle_targets_given_fingertip_targets(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    fingertip_targets: torch.Tensor,
) -> Tuple[torch.Tensor, defaultdict]:
    # Sanity check
    batch_size = hand_model.batch_size
    num_fingers = hand_model.num_fingers
    num_xyz = 3
    assert fingertip_targets.shape == (batch_size, num_fingers, num_xyz)

    # Store original hand pose for later
    original_hand_pose = hand_model.hand_pose.detach().clone()

    # Optimize joint angles
    joint_angle_targets_to_optimize = (
        joint_angles_start.detach().clone().requires_grad_(True)
    )
    debug_info = defaultdict(list)
    N_ITERS = 100
    for i in range(N_ITERS):
        contact_points_hand_to_optimize = compute_fingertip_mean_contact_positions(
            joint_angles=joint_angle_targets_to_optimize, hand_model=hand_model
        )
        loss = (
            (fingertip_targets.detach().clone() - contact_points_hand_to_optimize)
            .square()
            .sum()
        )
        GRAD_STEP_SIZE = 5

        loss.backward(retain_graph=True)

        with torch.no_grad():
            joint_angle_targets_to_optimize -= (
                joint_angle_targets_to_optimize.grad * GRAD_STEP_SIZE
            )
            joint_angle_targets_to_optimize.grad.zero_()
        debug_info["loss"].append(loss.item())
        debug_info["fingertip_targets"].append(fingertip_targets.detach().clone())
        debug_info["contact_points_hand"].append(
            contact_points_hand_to_optimize.detach().clone()
        )

    # Update hand pose parameters
    new_hand_pose = original_hand_pose.detach().clone()
    new_hand_pose[:, 9:] = joint_angle_targets_to_optimize
    hand_model.set_parameters(new_hand_pose)

    return joint_angle_targets_to_optimize, debug_info


def compute_optimized_joint_angle_targets_given_grasp_orientations(
    joint_angles_start: torch.Tensor,
    hand_model: HandModel,
    grasp_orientations: torch.Tensor,
) -> torch.Tensor:
    # Sanity check
    batch_size = joint_angles_start.shape[0]
    num_fingers = hand_model.num_fingers
    assert grasp_orientations.shape == (batch_size, num_fingers, 3, 3)

    # Get current positions
    fingertip_mean_positions = compute_fingertip_mean_contact_positions(
        joint_angles=joint_angles_start,
        hand_model=hand_model,
    )
    assert fingertip_mean_positions.shape == (batch_size, num_fingers, 3)

    # Get grasp directions
    grasp_directions = grasp_orientations[:, :, :, 2]
    assert grasp_directions.shape == (batch_size, num_fingers, 3)

    # Get fingertip targets
    DIST_MOVE_FINGER = 0.01
    fingertip_targets = fingertip_mean_positions + grasp_directions * DIST_MOVE_FINGER

    joint_angle_targets, _ = compute_optimized_joint_angle_targets_given_fingertip_targets(
        joint_angles_start=joint_angles_start,
        hand_model=hand_model,
        fingertip_targets=fingertip_targets,
    )
    return joint_angle_targets