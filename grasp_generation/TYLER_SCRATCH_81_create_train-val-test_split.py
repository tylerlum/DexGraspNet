# %%
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np
import pathlib

# %%
data_folder = pathlib.Path("/juno/u/tylerlum/github_repos/DexGraspNet/data")
assert data_folder.exists()

# %%
experiment_folders = sorted(
    list(data_folder.glob("2024-05-06_rotated_stable_grasps_*"))
    + list(data_folder.glob("2024-05-26_rotated_v2_only_grasps_*"))
)
print(f"Found {len(experiment_folders)} experiment folders")

# %%
grasp_config_dict_paths = [
    experiment_folder / "raw_grasp_config_dicts"
    for experiment_folder in experiment_folders
]

for grasp_config_dict_path in grasp_config_dict_paths:
    assert grasp_config_dict_path.exists()

print(f"Found {len(grasp_config_dict_paths)} grasp config dicts")

# %%
nerfcheckpoint_paths = [
    experiment_folder / "NEW_nerfcheckpoints_100imgs_400iters"
    for experiment_folder in experiment_folders
]

for nerfcheckpoint_path in nerfcheckpoint_paths:
    assert nerfcheckpoint_path.exists()

print(f"Found {len(nerfcheckpoint_paths)} nerf checkpoints")

# %%
npy_files = sorted(
    [
        x
        for grasp_config_dict_path in grasp_config_dict_paths
        for x in grasp_config_dict_path.glob("*.npy")
    ]
)
print(f"Found {len(npy_files)} npy files")

# %%
nerfcheckpoint_files = sorted(
    [
        x
        for nerfcheckpoint_path in nerfcheckpoint_paths
        for x in nerfcheckpoint_path.rglob("**/config.yml")
    ]
)
print(f"Found {len(nerfcheckpoint_files)} nerf checkpoint files")

# %%
failure_logs = [
    experiment_folder / "NEW_nerfdata_100imgs_failures.txt"
    for experiment_folder in experiment_folders
    if (experiment_folder / "NEW_nerfdata_100imgs_failures.txt").exists()
]
print(f"Found {len(failure_logs)} failure logs")


# %%
def read_and_extract_object_names(file_path):
    object_names = []
    with open(file_path, "r") as file:
        for line in file:
            if ":" in line:
                object_name = line.split(":")[0]
                object_names.append(object_name.strip())
    return object_names


# %%
failed_object_names = []
for failure_log in failure_logs:
    object_names = read_and_extract_object_names(failure_log)
    print(f"{failure_log.name}: {object_names}")
    failed_object_names.extend(object_names)
print(f"Found {len(failed_object_names)} object names")


# %%
in_all = set([x.stem for x in npy_files]) & set(
    [x.parents[2].stem for x in nerfcheckpoint_files]
)
print(f"Found {len(in_all)} files in both npy and nerfcheckpoint")

# %%
FINAL_OBJECT_CODE_AND_SCALE_LIST = list(in_all - set(failed_object_names))
print(
    f"Filtered out {len(in_all) - len(FINAL_OBJECT_CODE_AND_SCALE_LIST)} failed object names"
)
print(f"Now have {len(FINAL_OBJECT_CODE_AND_SCALE_LIST)} object names")

print(f"FINAL_OBJECT_CODE_AND_SCALE_LIST[:5]: {FINAL_OBJECT_CODE_AND_SCALE_LIST[:5]}")

# %%
FINAL_OBJECT_CODE_TO_SCALES = defaultdict(list)
for object_code_and_scale in FINAL_OBJECT_CODE_AND_SCALE_LIST:
    idx = object_code_and_scale.index("_0_")
    object_code = object_code_and_scale[:idx]
    object_scale = float(object_code_and_scale[idx + 1 :].replace("_", "."))
    FINAL_OBJECT_CODE_TO_SCALES[object_code].append(object_scale)

# %%
UNIQUE_OBJECT_CODES = list(set(FINAL_OBJECT_CODE_TO_SCALES.keys()))
print(f"Unique object codes: {len(UNIQUE_OBJECT_CODES)}")

# %%
val_frac, test_frac = 0.05, 0.025
train_frac = 1 - val_frac - test_frac

TRAIN_OBJECT_CODES, TEST_OBJECT_CODES = train_test_split(
    list(UNIQUE_OBJECT_CODES), test_size=val_frac + test_frac, random_state=42
)
VAL_OBJECT_CODES, TEST_OBJECT_CODES = train_test_split(
    TEST_OBJECT_CODES, test_size=test_frac / (val_frac + test_frac), random_state=42
)
print(f"Train unique objects: {len(TRAIN_OBJECT_CODES)}")
print(f"Val unique objects: {len(VAL_OBJECT_CODES)}")
print(f"Test unique objects: {len(TEST_OBJECT_CODES)}")

# %%
TRAIN_OBJECT_CODE_TO_SCALES = {
    object_code: FINAL_OBJECT_CODE_TO_SCALES[object_code]
    for object_code in TRAIN_OBJECT_CODES
}
VAL_OBJECT_CODE_TO_SCALES = {
    object_code: FINAL_OBJECT_CODE_TO_SCALES[object_code]
    for object_code in VAL_OBJECT_CODES
}
TEST_OBJECT_CODE_TO_SCALES = {
    object_code: FINAL_OBJECT_CODE_TO_SCALES[object_code]
    for object_code in TEST_OBJECT_CODES
}

# %%
TRAIN_OBJECT_CODE_AND_SCALE_LIST = sorted(
    [
        f"{object_code}_{object_scale:.4f}".replace(".", "_")
        for object_code, object_scales in TRAIN_OBJECT_CODE_TO_SCALES.items()
        for object_scale in object_scales
    ]
)
VAL_OBJECT_CODE_AND_SCALE_LIST = sorted(
    [
        f"{object_code}_{object_scale:.4f}".replace(".", "_")
        for object_code, object_scales in VAL_OBJECT_CODE_TO_SCALES.items()
        for object_scale in object_scales
    ]
)
TEST_OBJECT_CODE_AND_SCALE_LIST = sorted(
    [
        f"{object_code}_{object_scale:.4f}".replace(".", "_")
        for object_code, object_scales in TEST_OBJECT_CODE_TO_SCALES.items()
        for object_scale in object_scales
    ]
)

print(f"Train: {len(TRAIN_OBJECT_CODE_AND_SCALE_LIST)}")
print(f"Val: {len(VAL_OBJECT_CODE_AND_SCALE_LIST)}")
print(f"Test: {len(TEST_OBJECT_CODE_AND_SCALE_LIST)}")

# %%
OUTPUT_DIR = pathlib.Path(
    "/juno/u/tylerlum/github_repos/DexGraspNet/2024-06-02_NEW_train_val_test_splits"
)
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
print(f"Created {OUTPUT_DIR}")

with open(OUTPUT_DIR / "train.txt", "w") as file:
    for object_code_and_scale in TRAIN_OBJECT_CODE_AND_SCALE_LIST:
        file.write(f"{object_code_and_scale}\n")

with open(OUTPUT_DIR / "val.txt", "w") as file:
    for object_code_and_scale in VAL_OBJECT_CODE_AND_SCALE_LIST:
        file.write(f"{object_code_and_scale}\n")

with open(OUTPUT_DIR / "test.txt", "w") as file:
    for object_code_and_scale in TEST_OBJECT_CODE_AND_SCALE_LIST:
        file.write(f"{object_code_and_scale}\n")

# %%
read_in_train_object_code_and_scale_list = []
with open(OUTPUT_DIR / "train.txt", "r") as file:
    for line in file:
        read_in_train_object_code_and_scale_list.append(line.strip())

read_in_val_object_code_and_scale_list = []
with open(OUTPUT_DIR / "val.txt", "r") as file:
    for line in file:
        read_in_val_object_code_and_scale_list.append(line.strip())

read_in_test_object_code_and_scale_list = []
with open(OUTPUT_DIR / "test.txt", "r") as file:
    for line in file:
        read_in_test_object_code_and_scale_list.append(line.strip())

# %%
assert read_in_train_object_code_and_scale_list == TRAIN_OBJECT_CODE_AND_SCALE_LIST
assert read_in_val_object_code_and_scale_list == VAL_OBJECT_CODE_AND_SCALE_LIST
assert read_in_test_object_code_and_scale_list == TEST_OBJECT_CODE_AND_SCALE_LIST
print(f"Successfully wrote and read train, val, and test splits to {OUTPUT_DIR}")

# %%
