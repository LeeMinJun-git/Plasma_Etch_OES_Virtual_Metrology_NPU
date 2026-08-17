import numpy as np

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATASET = "full_dataset_raw.npz"


# ==========================================================
# 1. Dataset
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

experiments = data[
    "experiment_key"
]


print("X :", X.shape)
print("y :", y.shape)


# ==========================================================
# 2. OES Feature Extraction
# ==========================================================

def extract_features(X):

    # X:
    # (N, 100, 128, 1)

    X = X[:, :, :, 0]

    # dynamic range 완화
    X = np.log1p(X)


    # ------------------------------------------------------
    # A. Cycle 전체 평균 spectrum
    #
    # (N,100,128)
    # ->
    # (N,128)
    # ------------------------------------------------------

    mean_spectrum = np.mean(
        X,
        axis=1
    )


    # ------------------------------------------------------
    # B. Cycle 방향 slope
    #
    # 각 wavelength가
    # Cycle 1 -> 100 동안 어떻게 변하는지
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


    # ------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------

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


features = extract_features(
    X
)


print(
    "OES feature shape :",
    features.shape
)


# ==========================================================
# 3. Leave-One-Lot-Out
# ==========================================================

predictions = np.zeros_like(
    y
)

unique_lots = np.unique(
    lots
)


print()
print("=" * 80)
print("OES RIDGE - LEAVE-ONE-LOT-OUT")
print("=" * 80)


# Ridge strength 후보
alphas = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]


for test_lot in unique_lots:

    train_mask = (
        lots != test_lot
    )

    test_mask = (
        lots == test_lot
    )


    X_train = features[
        train_mask
    ]

    X_test = features[
        test_mask
    ]

    y_train = y[
        train_mask
    ]

    y_test = y[
        test_mask
    ]


    # ======================================================
    # Feature scaling
    #
    # 반드시 Train으로만 fit
    # ======================================================

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # ======================================================
    # Train 내부에서 alpha 선택
    #
    # 여기서는 단순하게 training MSE가 아닌
    # 내부 lot-wise validation 사용
    # ======================================================

    train_lots = lots[
        train_mask
    ]

    best_alpha = None
    best_mae = np.inf


    for alpha in alphas:

        inner_predictions = np.zeros_like(
            y_train
        )


        for inner_lot in np.unique(
            train_lots
        ):

            inner_train_mask = (
                train_lots
                != inner_lot
            )

            inner_val_mask = (
                train_lots
                == inner_lot
            )


            inner_scaler = StandardScaler()

            inner_X_train = (
                inner_scaler.fit_transform(
                    X_train[
                        inner_train_mask
                    ]
                )
            )

            inner_X_val = (
                inner_scaler.transform(
                    X_train[
                        inner_val_mask
                    ]
                )
            )


            model = Ridge(
                alpha=alpha
            )

            model.fit(
                inner_X_train,
                y_train[
                    inner_train_mask
                ]
            )


            inner_predictions[
                inner_val_mask
            ] = model.predict(
                inner_X_val
            )


        alpha_mae = (
            mean_absolute_error(
                y_train,
                inner_predictions
            )
        )


        if alpha_mae < best_mae:

            best_mae = (
                alpha_mae
            )

            best_alpha = (
                alpha
            )


    # ======================================================
    # Outer fold final model
    # ======================================================

    model = Ridge(
        alpha=best_alpha
    )

    model.fit(
        X_train_scaled,
        y_train
    )


    pred = model.predict(
        X_test_scaled
    )


    predictions[
        test_mask
    ] = pred


    fold_mae = (
        mean_absolute_error(
            y_test,
            pred
        )
    )


    print()
    print(
        f"Test Lot {test_lot}"
    )

    print(
        f"  Alpha    : "
        f"{best_alpha}"
    )

    print(
        f"  Fold MAE : "
        f"{fold_mae:.4f} um"
    )


# ==========================================================
# 4. Overall
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
print("OVERALL OES RIDGE RESULT")
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
    "Mean baseline        : "
    "0.2984 um"
)

print(
    "Wafer-order baseline : "
    "0.1541 um"
)