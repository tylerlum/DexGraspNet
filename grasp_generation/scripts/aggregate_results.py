import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter


from tap import Tap
import os
import subprocess
import sys
import pathlib
from typing import Optional
from datetime import datetime

sys.path.append(os.path.realpath("."))


class ArgParser(Tap):
    """
    Assumes the following directory structure:
    <experiment_dir>
    |
    |--- <method 1>
    |   |--- evaled_grasp_config_dicts
    |       |--- <object code 1>.npy
    |       |--- <object code 2>.npy
    |            ...
    |
    |--- <method 2>
         ...
    """

    experiment_dir: pathlib.Path  # Path to the experiment directory


def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    tick_locations = np.round(np.linspace(min(bins) + bin_w / 2, max(bins) - bin_w / 2, len(bins)), 1)
    plt.xticks(tick_locations, bins, **kwargs)
    plt.xlim(bins[0], bins[-1])


def main() -> None:
    args = ArgParser().parse_args()
    assert args.experiment_dir.exists(), f"Path does not exist: {args.experiment_dir}"

    method_paths = sorted([d for d in args.experiment_dir.iterdir() if d.is_dir()])
    method_names = [d.name for d in method_paths]

    method_name_to_dict = {}
    for method_name, method_path in zip(method_names, method_paths):
        evaled_grasp_config_dicts_path = method_path / "evaled_grasp_config_dicts"
        if not evaled_grasp_config_dicts_path.exists():
            print(
                f"Skipping method {method_name} because {evaled_grasp_config_dicts_path} does not exist"
            )

        evaled_grasp_config_dict_paths = sorted(
            list(evaled_grasp_config_dicts_path.glob("*.npy"))
        )
        evaled_grasp_config_dicts = [
            np.load(path, allow_pickle=True).item()
            for path in evaled_grasp_config_dict_paths
        ]
        print(
            f"For method {method_name}: Found {len(evaled_grasp_config_dicts)} evaled_grasp_config_dicts in {evaled_grasp_config_dicts_path}"
        )

        passed_eval_means = [
            evaled_grasp_config_dict["passed_eval"].mean()
            for evaled_grasp_config_dict in evaled_grasp_config_dicts
        ]
        passed_simulation_means = [
            evaled_grasp_config_dict["passed_simulation"].mean()
            for evaled_grasp_config_dict in evaled_grasp_config_dicts
        ]
        passed_penetration_means = [
            evaled_grasp_config_dict["passed_new_penetration_test"].mean()
            for evaled_grasp_config_dict in evaled_grasp_config_dicts
        ]
        method_name_to_dict[method_name] = {
            "passed_eval_means": passed_eval_means,
            "passed_simulation_means": passed_simulation_means,
            "passed_penetration_means": passed_penetration_means,
        }

    labels = [
        "passed_eval_means",
        "passed_simulation_means",
        "passed_penetration_means",
    ]
    for label in labels:
        plt.figure(figsize=(14, 10))
        plt.rcParams.update({"font.size": 22})

        bins = np.linspace(0, 1, 11)
        plt.hist(
            [method_name_to_dict[method_name][label] for method_name in method_names],
            bins=bins,
            label=method_names,
        )
        bins_labels(bins, fontsize=20)
        plt.title(f'Histogram of {label.replace("_", " ").capitalize()}')
        plt.xlabel("Success Rate")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(axis="y")
        plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places

        img_filename = f"{label}_histogram.png"
        plt.savefig(img_filename, dpi=300)
        print(f"Saved image to {img_filename}")


if __name__ == "__main__":
    main()
