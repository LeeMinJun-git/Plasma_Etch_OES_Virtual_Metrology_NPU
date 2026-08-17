import os
import numpy as np
import pandas as pd

# ==========================================================
# Setting
# ==========================================================

CSV_FILE = "Si_Oxide_etch_9_points.csv"

INPUT_DIR = "cnn_inputs_2024_08_05"

DATE = "2024-08-05"

WAFERS = [
    "Wafer_01",
    "Wafer_02",
    "Wafer_03",
    "Wafer_04",
    "Wafer_05",
]

OUTPUT_FILE = "dataset_2024_08_05.npz"


# ==========================================================
# 1. Measurement CSV load
# ==========================================================

df = pd.read_csv(CSV_FILE)


# ==========================================================
# 2. Dataset containers
# ==========================================================

X_list = []
y_list = []

experiment_keys = []
lot_numbers = []
wafer_numbers = []


# ==========================================================
# 3. Wafer별 X / y 연결
# ==========================================================

print("=" * 72)
print("BUILD X / y DATASET")
print("=" * 72)


for wafer in WAFERS:

    # ------------------------------------------------------
    # Wafer 번호
    # Wafer_01 -> 1
    # ------------------------------------------------------

    wafer_number = int(
        wafer.split("_")[1]
    )

    experiment_key = (
        f"{DATE}_{wafer_number:02d}"
    )


    print()
    print("-" * 72)
    print(experiment_key)
    print("-" * 72)


    # ======================================================
    # 3-1. CNN input X
    # ======================================================

    input_file = os.path.join(
        INPUT_DIR,
        f"2024_08_05_{wafer}_100x128x1.npy"
    )


    if not os.path.exists(
        input_file
    ):

        print(
            "FAIL: CNN input file not found:"
        )

        print(
            input_file
        )

        continue


    X = np.load(
        input_file
    )


    if X.shape != (
        100,
        128,
        1
    ):

        print(
            "FAIL: Unexpected X shape:",
            X.shape
        )

        continue


    # ======================================================
    # 3-2. CSV target rows
    # ======================================================

    target_rows = df[
        df["experiment_key"]
        == experiment_key
    ]


    print(
        "Measurement points :",
        len(target_rows)
    )


    # 9 point가 모두 있어야 함
    if len(target_rows) != 9:

        print(
            "FAIL: Expected 9 measurement points"
        )

        continue


    # ======================================================
    # 3-3. si_etch NaN 검사
    # ======================================================

    si_etch = target_rows[
        "si_etch"
    ].to_numpy(
        dtype=np.float32
    )


    if np.isnan(
        si_etch
    ).any():

        print(
            "FAIL: NaN found in si_etch"
        )

        continue


    # ======================================================
    # 3-4. Mean Si Etch Depth
    # ======================================================

    mean_si_etch = float(
        np.mean(
            si_etch
        )
    )


    lot_number = int(
        target_rows[
            "lot_number"
        ].iloc[0]
    )


    csv_wafer_number = int(
        target_rows[
            "wafer_number"
        ].iloc[0]
    )


    # ======================================================
    # 3-5. Metadata consistency
    # ======================================================

    if (
        csv_wafer_number
        != wafer_number
    ):

        print(
            "FAIL: Wafer number mismatch"
        )

        continue


    # ======================================================
    # 3-6. Save to list
    # ======================================================

    X_list.append(
        X.astype(
            np.float32
        )
    )

    y_list.append(
        mean_si_etch
    )

    experiment_keys.append(
        experiment_key
    )

    lot_numbers.append(
        lot_number
    )

    wafer_numbers.append(
        wafer_number
    )


    print(
        "X shape             :",
        X.shape
    )

    print(
        "Mean Si Etch Depth  :",
        f"{mean_si_etch:.4f}",
        "um"
    )

    print(
        "Lot                 :",
        lot_number
    )

    print(
        "RESULT              : PASS"
    )


# ==========================================================
# 4. Stack
# ==========================================================

if len(X_list) == 0:

    raise RuntimeError(
        "No valid samples generated"
    )


X_dataset = np.stack(
    X_list,
    axis=0
)

y_dataset = np.asarray(
    y_list,
    dtype=np.float32
)

lot_dataset = np.asarray(
    lot_numbers,
    dtype=np.int32
)

wafer_dataset = np.asarray(
    wafer_numbers,
    dtype=np.int32
)

experiment_dataset = np.asarray(
    experiment_keys
)


# ==========================================================
# 5. Dataset validation
# ==========================================================

print()
print("=" * 72)
print("DATASET SUMMARY")
print("=" * 72)

print(
    "X shape :",
    X_dataset.shape
)

print(
    "y shape :",
    y_dataset.shape
)

print(
    "lot shape :",
    lot_dataset.shape
)

print()
print(
    "Target values:"
)

for key, target in zip(
    experiment_dataset,
    y_dataset
):

    print(
        f"{key} -> "
        f"{target:.4f} um"
    )


print()
print(
    "Target min  :",
    y_dataset.min()
)

print(
    "Target max  :",
    y_dataset.max()
)

print(
    "Target mean :",
    y_dataset.mean()
)


# ==========================================================
# 6. Save
# ==========================================================

np.savez_compressed(
    OUTPUT_FILE,

    X=X_dataset,
    y=y_dataset,

    experiment_key=experiment_dataset,
    lot=lot_dataset,
    wafer=wafer_dataset,
)


print()
print(
    "Saved:",
    OUTPUT_FILE
)

print(
    "Generated samples:",
    len(X_dataset)
)