import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ==========================================================
# Setting
# ==========================================================

DATASET_FILE = "full_dataset_raw.npz"
RESIDUAL_FILE = "oes_residual_result.npz"


# ==========================================================
# 1. Dataset load
# ==========================================================

data = np.load(
    DATASET_FILE,
    allow_pickle=True
)

y = data["y"].astype(
    np.float32
)

lots = data["lot"].astype(
    np.int32
)

wafers = data["wafer"].astype(
    np.float32
)

experiments = data[
    "experiment_key"
]


# ==========================================================
# 2. Residual model prediction load
# ==========================================================

result = np.load(
    RESIDUAL_FILE,
    allow_pickle=True
)

residual_prediction = result[
    "prediction"
].astype(
    np.float32
)


# ==========================================================
# 3. 기본 검증
# ==========================================================

if len(residual_prediction) != len(y):

    raise RuntimeError(
        "Prediction/sample count mismatch"
    )


unique_lots = np.unique(
    lots
)


# ==========================================================
# 4. Wafer-order LOLO baseline 다시 계산
# ==========================================================

wafer_prediction = np.zeros_like(
    y
)


for test_lot in unique_lots:

    train_mask = (
        lots != test_lot
    )

    test_mask = (
        lots == test_lot
    )


    X_train = wafers[
        train_mask
    ].reshape(-1, 1)

    X_test = wafers[
        test_mask
    ].reshape(-1, 1)


    y_train = y[
        train_mask
    ]


    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )


    wafer_prediction[
        test_mask
    ] = model.predict(
        X_test
    )


# ==========================================================
# 5. Lot별 비교
# ==========================================================

print("=" * 90)
print("LOT-WISE OES RESIDUAL CONSISTENCY CHECK")
print("=" * 90)


improved_lots = 0
worse_lots = 0

lot_results = []


for lot in unique_lots:

    mask = (
        lots == lot
    )


    y_lot = y[
        mask
    ]

    wafer_pred_lot = wafer_prediction[
        mask
    ]

    residual_pred_lot = residual_prediction[
        mask
    ]


    # ------------------------------------------------------
    # MAE
    # ------------------------------------------------------

    baseline_mae = mean_absolute_error(
        y_lot,
        wafer_pred_lot
    )

    residual_mae = mean_absolute_error(
        y_lot,
        residual_pred_lot
    )


    # ------------------------------------------------------
    # RMSE
    # ------------------------------------------------------

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_lot,
            wafer_pred_lot
        )
    )

    residual_rmse = np.sqrt(
        mean_squared_error(
            y_lot,
            residual_pred_lot
        )
    )


    # ------------------------------------------------------
    # Improvement
    # ------------------------------------------------------

    improvement = (
        baseline_mae
        - residual_mae
    )


    improvement_percent = (
        improvement
        / baseline_mae
        * 100.0
    )


    if improvement > 0:

        result_text = "IMPROVED"
        improved_lots += 1

    else:

        result_text = "WORSE"
        worse_lots += 1


    print()
    print(
        f"Lot {lot}"
    )

    print(
        f"  Samples              : "
        f"{len(y_lot)}"
    )

    print(
        f"  Wafer baseline MAE   : "
        f"{baseline_mae:.4f} um"
    )

    print(
        f"  + OES residual MAE   : "
        f"{residual_mae:.4f} um"
    )

    print(
        f"  MAE improvement      : "
        f"{improvement:+.4f} um"
    )

    print(
        f"  Improvement percent  : "
        f"{improvement_percent:+.1f}%"
    )

    print(
        f"  Baseline RMSE        : "
        f"{baseline_rmse:.4f} um"
    )

    print(
        f"  OES residual RMSE    : "
        f"{residual_rmse:.4f} um"
    )

    print(
        f"  RESULT               : "
        f"{result_text}"
    )


    lot_results.append(
        (
            lot,
            len(y_lot),
            baseline_mae,
            residual_mae,
            improvement,
            improvement_percent
        )
    )


# ==========================================================
# 6. Overall comparison
# ==========================================================

overall_baseline_mae = mean_absolute_error(
    y,
    wafer_prediction
)

overall_residual_mae = mean_absolute_error(
    y,
    residual_prediction
)


overall_improvement = (
    overall_baseline_mae
    - overall_residual_mae
)

overall_percent = (
    overall_improvement
    / overall_baseline_mae
    * 100.0
)


print()
print("=" * 90)
print("OVERALL CONSISTENCY SUMMARY")
print("=" * 90)


print(
    "Total lots              :",
    len(unique_lots)
)

print(
    "Improved lots           :",
    improved_lots
)

print(
    "Worse lots              :",
    worse_lots
)


print()

print(
    f"Wafer baseline MAE      : "
    f"{overall_baseline_mae:.4f} um"
)

print(
    f"Wafer + OES MAE         : "
    f"{overall_residual_mae:.4f} um"
)

print(
    f"Overall improvement     : "
    f"{overall_improvement:.4f} um"
)

print(
    f"Overall improvement (%) : "
    f"{overall_percent:.1f}%"
)


# ==========================================================
# 7. 간단 판정
# ==========================================================

print()
print("=" * 90)
print("CONSISTENCY JUDGEMENT")
print("=" * 90)


if (
    improved_lots >= 7
    and overall_residual_mae < 0.12
):

    print(
        "STRONG PASS"
    )

    print(
        "OES improvement is consistent "
        "across most lots."
    )


elif (
    improved_lots >= 6
    and overall_residual_mae < 0.15
):

    print(
        "PASS"
    )

    print(
        "OES provides generally useful "
        "additional information."
    )


elif (
    improved_lots >= 4
):

    print(
        "CHECK"
    )

    print(
        "OES improvement is not fully "
        "consistent across lots."
    )


else:

    print(
        "FAIL"
    )

    print(
        "Overall performance may be driven "
        "by only a few lots."
    )