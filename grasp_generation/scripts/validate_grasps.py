"""
Last modified date: 2023.02.23
Author: Ruicheng Wang
Description: validate grasps on Isaac simulator
"""

import os
import sys

sys.path.append(os.path.realpath("."))

from utils.isaac_validator import IsaacValidator
import argparse
import torch
import numpy as np
import transforms3d
from utils.hand_model import HandModel
from utils.object_model import ObjectModel
from utils.hand_model_type import (
    HandModelType,
    handmodeltype_to_joint_names,
    handmodeltype_to_hand_root_hand_file,
)
from utils.qpos_pose_conversion import qpos_to_pose, qpos_to_translation_rot_jointangles
from typing import List


def compute_joint_angle_targets(
    args: argparse.Namespace, joint_names: List[str], data_dict: np.ndarray
):
    # Read in hand state and scale tensor
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    batch_size = data_dict.shape[0]
    hand_poses = []
    scale_tensor = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        scale = data_dict[i]["scale"]
        hand_pose = qpos_to_pose(
            qpos=qpos, joint_names=joint_names, unsqueeze_batch_dim=False
        ).to(device)
        hand_poses.append(hand_pose)
        scale_tensor.append(scale)
    hand_poses = torch.stack(hand_poses).to(device).requires_grad_()
    scale_tensor = torch.tensor(scale_tensor).reshape(1, -1).to(device)

    # hand model
    hand_model = HandModel(
        hand_model_type=args.hand_model_type, n_surface_points=2000, device=device
    )
    hand_model.set_parameters(hand_poses)

    # object model
    object_model = ObjectModel(
        data_root_path=args.mesh_path,
        batch_size_each=batch_size,
        num_samples=0,
        device=device,
    )
    object_model.initialize(args.object_code)
    object_model.object_scale_tensor = scale_tensor

    # calculate contact points and contact normals
    num_links = len(hand_model.mesh)
    contact_points_hand = torch.zeros((batch_size, num_links, 3)).to(device)
    contact_normals = torch.zeros((batch_size, num_links, 3)).to(device)

    for i, link_name in enumerate(hand_model.mesh):
        if len(hand_model.mesh[link_name]["surface_points"]) == 0:
            continue
        surface_points = (
            hand_model.current_status[link_name]
            .transform_points(hand_model.mesh[link_name]["surface_points"])
            .expand(batch_size, -1, 3)
        )
        surface_points = surface_points @ hand_model.global_rotation.transpose(
            1, 2
        ) + hand_model.global_translation.unsqueeze(1)
        distances, normals = object_model.cal_distance(surface_points)
        nearest_point_index = distances.argmax(dim=1)
        nearest_distances = torch.gather(distances, 1, nearest_point_index.unsqueeze(1))
        nearest_points_hand = torch.gather(
            surface_points, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        nearest_normals = torch.gather(
            normals, 1, nearest_point_index.reshape(-1, 1, 1).expand(-1, 1, 3)
        )
        admited = -nearest_distances < args.thres_cont
        admited = admited.reshape(-1, 1, 1).expand(-1, 1, 3)
        contact_points_hand[:, i : i + 1, :] = torch.where(
            admited, nearest_points_hand, contact_points_hand[:, i : i + 1, :]
        )
        contact_normals[:, i : i + 1, :] = torch.where(
            admited, nearest_normals, contact_normals[:, i : i + 1, :]
        )

    target_points = contact_points_hand + contact_normals * args.dis_move
    loss = (target_points.detach().clone() - contact_points_hand).square().sum()
    loss.backward()
    with torch.no_grad():
        hand_poses[:, 9:] += hand_poses.grad[:, 9:] * args.grad_move
        hand_poses.grad.zero_()

    assert hand_poses.shape == (batch_size, 3 + 6 + hand_model.n_dofs)
    joint_angle_targets = hand_poses[:, 9:]

    return joint_angle_targets


def main(args):
    joint_names = handmodeltype_to_joint_names[args.hand_model_type]
    os.environ.pop("CUDA_VISIBLE_DEVICES")

    if args.index is not None:
        sim = IsaacValidator(
            hand_model_type=args.hand_model_type,
            gpu=args.gpu,
            mode="gui",
            start_with_step_mode=args.start_with_step_mode,
        )
    else:
        sim = IsaacValidator(hand_model_type=args.hand_model_type, gpu=args.gpu)

    # Read in data
    data_dict = np.load(
        os.path.join(args.grasp_path, args.object_code + ".npy"), allow_pickle=True
    )
    batch_size = data_dict.shape[0]
    translations = []
    rotations = []
    joint_angles_array = []
    scale_array = []
    E_pen_array = []
    for i in range(batch_size):
        qpos = data_dict[i]["qpos"]
        translation, rot, joint_angles = qpos_to_translation_rot_jointangles(
            qpos=qpos, joint_names=joint_names
        )
        translations.append(translation)
        rotations.append(transforms3d.euler.euler2quat(*rot))
        joint_angles_array.append(joint_angles)

        scale = data_dict[i]["scale"]
        scale_array.append(scale)

        if "E_pen" in data_dict[i]:
            E_pen_array.append(data_dict[i]["E_pen"])
        elif args.index is not None:
            print(f"Warning: E_pen not found in data_dict[{i}]")
            print(
                "This is expected behavior if you are validating already validated grasps"
            )
            E_pen_array.append(0)
        else:
            raise ValueError(f"E_pen not found in data_dict[{i}]")
    E_pen_array = np.array(E_pen_array)

    if not args.no_force:
        joint_angle_targets_array = (
            compute_joint_angle_targets(
                args=args, joint_names=joint_names, data_dict=data_dict
            )
            .detach()
            .cpu()
            .numpy()
        )
    else:
        joint_angle_targets_array = None

    hand_root, hand_file = handmodeltype_to_hand_root_hand_file[args.hand_model_type]

    if args.index is not None:
        sim.set_asset(
            hand_root=hand_root,
            hand_file=hand_file,
            obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
            obj_file="coacd.urdf",
        )
        index = args.index
        sim.add_env_single_test_rotation(
            hand_rotation=rotations[index],
            hand_translation=translations[index],
            hand_qpos=joint_angles_array[index],
            obj_scale=scale_array[index],
            target_qpos=(
                joint_angle_targets_array[index]
                if joint_angle_targets_array is not None
                else None
            ),
        )
        result = sim.run_sim()
        print(f"result = {result}")
        print(
            " = ".join(
                [
                    "E_pen < args.penetration_threshold",
                    f"{E_pen_array[index]:.7f} < {args.penetration_threshold:.7f}",
                    f"{E_pen_array[index] < args.penetration_threshold}",
                ]
            )
        )
        estimated = E_pen_array < args.penetration_threshold
    else:
        simulated = np.zeros(batch_size, dtype=np.bool8)
        offset = 0
        result = []
        for batch in range(batch_size // args.val_batch):
            offset_ = min(offset + args.val_batch, batch_size)
            sim.set_asset(
                hand_root=hand_root,
                hand_file=hand_file,
                obj_root=os.path.join(args.mesh_path, args.object_code, "coacd"),
                obj_file="coacd.urdf",
            )
            for index in range(offset, offset_):
                sim.add_env_all_test_rotations(
                    hand_rotation=rotations[index],
                    hand_translation=translations[index],
                    hand_qpos=joint_angles_array[index],
                    obj_scale=scale_array[index],
                    target_qpos=(
                        joint_angle_targets_array[index]
                        if joint_angle_targets_array is not None
                        else None
                    ),
                )
            result = [*result, *sim.run_sim()]
            sim.reset_simulator()
            offset = offset_

        num_envs_per_grasp = len(sim.test_rotations)
        for i in range(batch_size):
            simulated[i] = np.array(
                sum(result[i * num_envs_per_grasp : (i + 1) * num_envs_per_grasp])
                == num_envs_per_grasp
            )

        estimated = E_pen_array < args.penetration_threshold
        valid = simulated * estimated
        print(
            f"estimated: {estimated.sum().item()}/{batch_size}, "
            f"simulated: {simulated.sum().item()}/{batch_size}, "
            f"valid: {valid.sum().item()}/{batch_size}"
        )
        result_list = []
        for i in range(batch_size):
            if valid[i]:
                new_data_dict = {}
                new_data_dict["qpos"] = data_dict[i]["qpos"]
                new_data_dict["scale"] = data_dict[i]["scale"]
                result_list.append(new_data_dict)

        os.makedirs(args.result_path, exist_ok=True)
        np.save(
            os.path.join(args.result_path, args.object_code + ".npy"),
            result_list,
            allow_pickle=True,
        )
    sim.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hand_model_type",
        default=HandModelType.SHADOW_HAND,
        type=HandModelType.from_string,
        choices=list(HandModelType),
    )
    parser.add_argument("--gpu", default=3, type=int)
    parser.add_argument("--val_batch", default=500, type=int)
    parser.add_argument("--mesh_path", default="../data/meshdata", type=str)
    parser.add_argument("--grasp_path", default="../data/graspdata", type=str)
    parser.add_argument("--result_path", default="../data/dataset", type=str)
    parser.add_argument(
        "--object_code", default="sem-Xbox360-d0dff348985d4f8e65ca1b579a4b8d2", type=str
    )
    # if index is received, then the debug mode is on
    parser.add_argument("--index", type=int)
    parser.add_argument("--start_with_step_mode", action="store_true")
    parser.add_argument("--no_force", action="store_true")
    parser.add_argument("--thres_cont", default=0.001, type=float)
    parser.add_argument("--dis_move", default=0.001, type=float)
    parser.add_argument("--grad_move", default=500, type=float)
    parser.add_argument("--penetration_threshold", default=0.001, type=float)

    args = parser.parse_args()
    main(args)
