import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge
)

from sklearn.preprocessing import (
    StandardScaler
)

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

X = data["X"].astype(
    np.float32
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
# 2. OES Feature Extraction
# ==========================================================

def extract_features(X):

    # (N,100,128,1)
    # ->
    # (N,100,128)

    X = X[:, :, :, 0]

    X = np.log1p(
        X
    )


    # ------------------------------------------------------
    # A. 평균 spectrum
    # ------------------------------------------------------

    mean_spectrum = np.mean(
        X,
        axis=1
    )


    # ------------------------------------------------------
    # B. Cycle 방향 slope
    # ------------------------------------------------------

    cycle_axis = np.arange(
        X.shape[1],
        dtype=np.float32
    )

    cycle_axis = (
        cycle_axis
        - cycle_axis.mean()
    )

    denominator = np.sum(
        cycle_axis ** 2
    )


    centered = (
        X
        - np.mean(
            X,
            axis=1,
            keepdims=True
        )
    )


    slope = np.sum(
        centered
        * cycle_axis[
            None,
            :,
            None
        ],
        axis=1
    ) / denominator


    features = np.concatenate(
        [
            mean_spectrum,
            slope
        ],
        axis=1
    )


    return features.astype(
        np.float32
    )


oes_features = extract_features(
    X
)

print(
    "OES feature shape:",
    oes_features.shape
)


# ==========================================================
# 3. Setting
# ==========================================================

alphas = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0
]

unique_lots = np.unique(
    lots
)

predictions = np.zeros_like(
    y
)


print()
print("=" * 80)
print("WAFER ORDER + OES RESIDUAL MODEL")
print("=" * 80)


# ==========================================================
# 4. Outer Leave-One-Lot-Out
# ==========================================================

for test_lot in unique_lots:

    outer_train = (
        lots != test_lot
    )

    outer_test = (
        lots == test_lot
    )


    X_train = oes_features[
        outer_train
    ]

    X_test = oes_features[
        outer_test
    ]


    y_train = y[
        outer_train
    ]

    y_test = y[
        outer_test
    ]


    wafer_train = wafers[
        outer_train
    ].reshape(-1, 1)

    wafer_test = wafers[
        outer_test
    ].reshape(-1, 1)


    train_lots = lots[
        outer_train
    ]


    # ======================================================
    # 5. Inner LOLO로 Ridge alpha 선택
    # ======================================================

    best_alpha = None
    best_inner_mae = np.inf


    for alpha in alphas:

        inner_predictions = np.zeros_like(
            y_train
        )


        for val_lot in np.unique(
            train_lots
        ):

            inner_train = (
                train_lots != val_lot
            )

            inner_val = (
                train_lots == val_lot
            )


            # --------------------------------------------------
            # A. Wafer-order baseline
            # --------------------------------------------------

            baseline = LinearRegression()

            baseline.fit(
                wafer_train[
                    inner_train
                ],
                y_train[
                    inner_train
                ]
            )


            base_train_pred = baseline.predict(
                wafer_train[
                    inner_train
                ]
            )


            base_val_pred = baseline.predict(
                wafer_train[
                    inner_val
                ]
            )


            # --------------------------------------------------
            # B. Training residual
            # --------------------------------------------------

            residual_train = (
                y_train[
                    inner_train
                ]
                - base_train_pred
            )


            # --------------------------------------------------
            # C. OES feature scaling
            # --------------------------------------------------

            scaler = StandardScaler()

            X_inner_train = scaler.fit_transform(
                X_train[
                    inner_train
                ]
            )

            X_inner_val = scaler.transform(
                X_train[
                    inner_val
                ]
            )


            # --------------------------------------------------
            # D. OES residual predictor
            # --------------------------------------------------

            residual_model = Ridge(
                alpha=alpha
            )

            residual_model.fit(
                X_inner_train,
                residual_train
            )


            residual_pred = (
                residual_model.predict(
                    X_inner_val
                )
            )


            # --------------------------------------------------
            # E. Final prediction
            # --------------------------------------------------

            final_pred = (
                base_val_pred
                + residual_pred
            )


            inner_predictions[
                inner_val
            ] = final_pred


        inner_mae = mean_absolute_error(
            y_train,
            inner_predictions
        )


        if inner_mae < best_inner_mae:

            best_inner_mae = (
                inner_mae
            )

            best_alpha = alpha


    # ======================================================
    # 6. Outer Training
    # ======================================================

    baseline = LinearRegression()

    baseline.fit(
        wafer_train,
        y_train
    )


    base_train_pred = baseline.predict(
        wafer_train
    )

    base_test_pred = baseline.predict(
        wafer_test
    )


    residual_train = (
        y_train
        - base_train_pred
    )


    # ------------------------------------------------------
    # OES scaler
    # ------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test
        )
    )


    # ------------------------------------------------------
    # Residual model
    # ------------------------------------------------------

    residual_model = Ridge(
        alpha=best_alpha
    )

    residual_model.fit(
        X_train_scaled,
        residual_train
    )


    residual_prediction = (
        residual_model.predict(
            X_test_scaled
        )
    )


    final_prediction = (
        base_test_pred
        + residual_prediction
    )


    predictions[
        outer_test
    ] = final_prediction


    # ======================================================
    # Fold Metrics
    # ======================================================

    fold_mae = mean_absolute_error(
        y_test,
        final_prediction
    )


    baseline_mae = mean_absolute_error(
        y_test,
        base_test_pred
    )


    print()
    print(
        f"Test Lot {test_lot}"
    )

    print(
        f"  Alpha              : "
        f"{best_alpha}"
    )

    print(
        f"  Wafer baseline MAE : "
        f"{baseline_mae:.4f} um"
    )

    print(
        f"  + OES residual MAE : "
        f"{fold_mae:.4f} um"
    )


# ==========================================================
# 7. Overall Result
# ==========================================================

mae = mean_absolute_error(
    y,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        predictions
    )
)

r2 = r2_score(
    y,
    predictions
)


print()
print("=" * 80)
print("OVERALL WAFER + OES RESULT")
print("=" * 80)

print(
    f"MAE  : {mae:.4f} um"
)

print(
    f"RMSE : {rmse:.4f} um"
)

print(
    f"R2   : {r2:.4f}"
)


print()
print("=" * 80)
print("REFERENCE")
print("=" * 80)

print(
    "Mean baseline             : "
    "0.2984 um"
)

print(
    "OES Ridge                 : "
    "0.2319 um"
)

print(
    "Wafer-order baseline      : "
    "0.1541 um"
)


# ==========================================================
# 8. Save
# ==========================================================

np.savez_compressed(
    "oes_residual_result.npz",

    target=y,

    prediction=predictions,

    lot=lots,

    experiment_key=experiments
)


print()
print(
    "Saved: oes_residual_result.npz"
)