from tap import Tap
import pathlib
import subprocess
import datetime
import numpy as np
from typing import Optional


class ArgumentParser(Tap):
    meshdata_root_path: pathlib.Path = pathlib.Path("../data/rotated_meshdata_v2")
    nerf_meshdata_root_path: Optional[pathlib.Path] = None
    plan_using_nerf: bool = False
    eval_using_nerf: bool = False
    output_eval_results_path: pathlib.Path = pathlib.Path(
        "../data/eval_results"
    ) / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def print_and_run(command: str) -> None:
    print(f"Running {command}")
    subprocess.run(
        command,
        shell=True,
        check=True,
    )


def main() -> None:
    args = ArgumentParser().parse_args()
    print("=" * 80)
    print(f"{pathlib.Path(__file__).name} args = {args}")
    print("=" * 80 + "\n")

    assert args.meshdata_root_path.exists(), f"{args.meshdata_root_path} does not exist"
    assert (
        args.nerf_meshdata_root_path is None or args.nerf_meshdata_root_path.exists()
    ), f"{args.nerf_meshdata_root_path} does not exist"

    args.output_eval_results_path.mkdir(parents=True)

    print("~" * 80)
    print("REMINDER: Expects the scale of the nerf_meshdata to be the same as the scale of the meshdata.")
    print("~" * 80 + "\n")
    planning_meshdata_root_path = (
        args.nerf_meshdata_root_path
        if args.plan_using_nerf
        else args.meshdata_root_path
    )
    eval_meshdata_root_path = (
        args.nerf_meshdata_root_path
        if args.eval_using_nerf
        else args.meshdata_root_path
    )
    assert (
        planning_meshdata_root_path is not None and planning_meshdata_root_path.exists()
    ), f"{planning_meshdata_root_path} does not exist"
    assert (
        eval_meshdata_root_path is not None and eval_meshdata_root_path.exists()
    ), f"{eval_meshdata_root_path} does not exist"

    # Gen hand configs
    hand_gen_command = (
        f"python scripts/generate_hand_config_dicts.py --meshdata_root_path {planning_meshdata_root_path}"
        + f" --output_hand_config_dicts_path {args.output_eval_results_path / 'hand_config_dicts'}"
        + " --use_penetration_energy"
    )
    print_and_run(hand_gen_command)

    # Gen grasp configs
    grasp_gen_command = (
        f"python scripts/generate_grasp_config_dicts.py --meshdata_root_path {planning_meshdata_root_path}"
        + f" --input_hand_config_dicts_path {args.output_eval_results_path / 'hand_config_dicts'}"
        + f" --output_grasp_config_dicts_path {args.output_eval_results_path / 'grasp_config_dicts'}"
    )
    print_and_run(grasp_gen_command)

    # Eval final grasp configs.
    eval_final_grasp_command = (
        "python scripts/eval_all_grasp_config_dicts.py"
        + f" --input_grasp_config_dicts_path {args.output_eval_results_path / 'grasp_config_dicts'}"
        + f" --output_evaled_grasp_config_dicts_path {args.output_eval_results_path / 'evaled_grasp_config_dicts'}"
        + f" --meshdata_root_path {eval_meshdata_root_path}"
    )
    print_and_run(eval_final_grasp_command)

    # Look at success rate of all grasps and best k grasps.
    BEST_K = 10
    num_successes, num_total = 0, 0
    num_successes_best_k, num_total_best_k = 0, 0
    for filepath in (
        args.output_eval_results_path / "evaled_grasp_config_dicts"
    ).iterdir():
        evaled_grasp_config_dict = np.load(filepath, allow_pickle=True).item()
        passed_eval = evaled_grasp_config_dict["passed_eval"]
        assert len(passed_eval.shape) == 1

        num_total += passed_eval.shape[0]
        num_successes += passed_eval.sum()

        total_energy = evaled_grasp_config_dict["Total Energy"]
        assert total_energy.shape == passed_eval.shape

        best_k_indices = np.argsort(total_energy)[:BEST_K]
        best_energy = total_energy[best_k_indices]
        assert np.all(best_energy < total_energy[np.argsort(total_energy)[-BEST_K:]])

        num_total_best_k += passed_eval[best_k_indices].shape[0]
        num_successes_best_k += passed_eval[best_k_indices].sum()
    print(
        f"Total success rate: num_successes / num_total = {num_successes} / {num_total} = {num_successes / num_total}"
    )
    print(
        f"Total success rate of top {BEST_K} energy: num_successes_best_k / num_total_best_k = {num_successes_best_k} / {num_total_best_k} = {num_successes_best_k / num_total_best_k}"
    )


if __name__ == "__main__":
    main()
