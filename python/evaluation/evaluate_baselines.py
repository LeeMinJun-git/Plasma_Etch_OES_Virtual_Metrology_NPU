import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATASET = "full_dataset_raw.npz"


# ==========================================================
# 1. Load
# ==========================================================

data = np.load(
    DATASET,
    allow_pickle=True
)

y = data["y"]
lots = data["lot"]
wafers = data["wafer"]

unique_lots = np.unique(
    lots
)


# ==========================================================
# 2. Prediction containers
# ==========================================================

pred_mean = np.zeros_like(
    y,
    dtype=np.float32
)

pred_linear = np.zeros_like(
    y,
    dtype=np.float32
)


print("=" * 80)
print("LEAVE-ONE-LOT-OUT BASELINE")
print("=" * 80)


# ==========================================================
# 3. Lot-wise evaluation
# ==========================================================

for test_lot in unique_lots:

    train_mask = (
        lots != test_lot
    )

    test_mask = (
        lots == test_lot
    )


    y_train = y[
        train_mask
    ]

    y_test = y[
        test_mask
    ]


    # ======================================================
    # Baseline A:
    # Train target mean
    # ======================================================

    train_mean = np.mean(
        y_train
    )

    mean_prediction = np.full(
        len(y_test),
        train_mean,
        dtype=np.float32
    )

    pred_mean[
        test_mask
    ] = mean_prediction


    # ======================================================
    # Baseline B:
    # Wafer order linear regression
    # ======================================================

    wafer_train = wafers[
        train_mask
    ].reshape(-1, 1)

    wafer_test = wafers[
        test_mask
    ].reshape(-1, 1)


    model = LinearRegression()

    model.fit(
        wafer_train,
        y_train
    )


    linear_prediction = model.predict(
        wafer_test
    )

    pred_linear[
        test_mask
    ] = linear_prediction


    # ======================================================
    # Fold metrics
    # ======================================================

    mean_mae = mean_absolute_error(
        y_test,
        mean_prediction
    )

    linear_mae = mean_absolute_error(
        y_test,
        linear_prediction
    )


    print()
    print(
        f"Test Lot {test_lot}"
    )

    print(
        f"  Samples               : "
        f"{len(y_test)}"
    )

    print(
        f"  Mean baseline MAE     : "
        f"{mean_mae:.4f} um"
    )

    print(
        f"  Wafer linear MAE      : "
        f"{linear_mae:.4f} um"
    )


# ==========================================================
# 4. Overall metrics
# ==========================================================

def print_metrics(
    name,
    target,
    prediction
):

    mae = mean_absolute_error(
        target,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            target,
            prediction
        )
    )

    r2 = r2_score(
        target,
        prediction
    )


    print()
    print(name)

    print(
        f"  MAE  : "
        f"{mae:.4f} um"
    )

    print(
        f"  RMSE : "
        f"{rmse:.4f} um"
    )

    print(
        f"  R2   : "
        f"{r2:.4f}"
    )


print()
print("=" * 80)
print("OVERALL RESULT")
print("=" * 80)


print_metrics(
    "Mean Baseline",
    y,
    pred_mean
)


print_metrics(
    "Wafer-order Linear Baseline",
    y,
    pred_linear
)