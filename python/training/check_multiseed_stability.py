import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# Setting
# ==========================================================

DATASET = "full_dataset_raw.npz"

SEEDS = [
    7,
    21,
    42,
    84,
    123
]

EPOCHS = 250
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# GPU reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ==========================================================
# Dataset
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


unique_lots = np.unique(
    lots
)


print("X shape :", X.shape)
print("y shape :", y.shape)
print("Lots    :", unique_lots)


# ==========================================================
# Model
# ==========================================================

class ResidualTinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                4,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            nn.Conv2d(
                4,
                8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            nn.AvgPool2d(
                kernel_size=(5, 4),
                stride=(5, 4)
            )
        )


        self.regressor = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                320,
                16
            ),

            nn.ReLU(),

            nn.Dropout(
                p=0.2
            ),

            nn.Linear(
                16,
                1
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.regressor(x)

        return x.squeeze(1)


# ==========================================================
# Baseline predictions
#
# Seed와 무관하므로 한 번만 계산
# ==========================================================

baseline_predictions = np.zeros_like(
    y
)


for test_lot in unique_lots:

    train_mask = (
        lots != test_lot
    )

    test_mask = (
        lots == test_lot
    )


    model = LinearRegression()

    model.fit(
        wafers[
            train_mask
        ].reshape(-1, 1),

        y[
            train_mask
        ]
    )


    baseline_predictions[
        test_mask
    ] = model.predict(
        wafers[
            test_mask
        ].reshape(-1, 1)
    )


baseline_mae = mean_absolute_error(
    y,
    baseline_predictions
)


print()
print(
    f"Wafer-order baseline MAE : "
    f"{baseline_mae:.4f} um"
)


# ==========================================================
# Results
# ==========================================================

seed_results = []
lot_results = []

all_seed_predictions = []


# ==========================================================
# Multi-seed
# ==========================================================

for seed in SEEDS:

    print()
    print("=" * 90)
    print(
        f"SEED {seed}"
    )
    print("=" * 90)


    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    seed_predictions = np.zeros_like(
        y
    )


    # ======================================================
    # Outer LOLO
    # ======================================================

    for test_lot in unique_lots:

        print(
            f"  Test Lot {test_lot} ... ",
            end="",
            flush=True
        )


        train_mask = (
            lots != test_lot
        )

        test_mask = (
            lots == test_lot
        )


        # --------------------------------------------------
        # Dataset split
        # --------------------------------------------------

        X_train = X[
            train_mask
        ].copy()

        X_test = X[
            test_mask
        ].copy()


        y_train = y[
            train_mask
        ].copy()

        y_test = y[
            test_mask
        ].copy()


        wafer_train = wafers[
            train_mask
        ].reshape(-1, 1)

        wafer_test = wafers[
            test_mask
        ].reshape(-1, 1)


        # ==================================================
        # Wafer-order baseline
        # ==================================================

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


        # ==================================================
        # Residual target
        # ==================================================

        residual_train = (
            y_train
            - base_train_pred
        )


        residual_mean = np.mean(
            residual_train
        )

        residual_std = np.std(
            residual_train
        )


        residual_train_norm = (
            residual_train
            - residual_mean
        ) / (
            residual_std
            + 1e-8
        )


        # ==================================================
        # OES preprocessing
        # ==================================================

        X_train = np.log1p(
            X_train
        )

        X_test = np.log1p(
            X_test
        )


        # --------------------------------------------------
        # Train fold only normalization
        # --------------------------------------------------

        x_mean = np.mean(
            X_train
        )

        x_std = np.std(
            X_train
        )


        X_train = (
            X_train
            - x_mean
        ) / (
            x_std
            + 1e-8
        )


        X_test = (
            X_test
            - x_mean
        ) / (
            x_std
            + 1e-8
        )


        # --------------------------------------------------
        # NHWC -> NCHW
        # --------------------------------------------------

        X_train = np.transpose(
            X_train,
            (0, 3, 1, 2)
        )

        X_test = np.transpose(
            X_test,
            (0, 3, 1, 2)
        )


        # ==================================================
        # Tensor
        # ==================================================

        X_train_tensor = torch.tensor(
            X_train,
            dtype=torch.float32
        )

        y_train_tensor = torch.tensor(
            residual_train_norm,
            dtype=torch.float32
        )

        X_test_tensor = torch.tensor(
            X_test,
            dtype=torch.float32
        )


        train_dataset = TensorDataset(
            X_train_tensor,
            y_train_tensor
        )


        # fold마다 deterministic shuffle
        generator = torch.Generator()

        generator.manual_seed(
            seed * 100
            + int(test_lot)
        )


        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            generator=generator
        )


        # ==================================================
        # Model
        # ==================================================

        model_seed = (
            seed * 100
            + int(test_lot)
        )


        torch.manual_seed(
            model_seed
        )

        if torch.cuda.is_available():

            torch.cuda.manual_seed_all(
                model_seed
            )


        model = ResidualTinyCNN().to(
            device
        )


        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )


        loss_fn = nn.MSELoss()


        # ==================================================
        # Training
        # ==================================================

        for epoch in range(
            EPOCHS
        ):

            model.train()


            for batch_X, batch_y in train_loader:

                batch_X = batch_X.to(
                    device
                )

                batch_y = batch_y.to(
                    device
                )


                optimizer.zero_grad()


                pred = model(
                    batch_X
                )


                loss = loss_fn(
                    pred,
                    batch_y
                )


                loss.backward()

                optimizer.step()


        # ==================================================
        # Inference
        # ==================================================

        model.eval()


        with torch.no_grad():

            pred_norm = model(
                X_test_tensor.to(
                    device
                )
            ).cpu().numpy()


        residual_pred = (
            pred_norm
            * residual_std
            + residual_mean
        )


        final_prediction = (
            base_test_pred
            + residual_pred
        )


        seed_predictions[
            test_mask
        ] = final_prediction


        # ==================================================
        # Fold metrics
        # ==================================================

        fold_baseline_mae = (
            mean_absolute_error(
                y_test,
                base_test_pred
            )
        )


        fold_cnn_mae = (
            mean_absolute_error(
                y_test,
                final_prediction
            )
        )


        improvement = (
            fold_baseline_mae
            - fold_cnn_mae
        )


        improved = (
            improvement > 0
        )


        lot_results.append(
            {
                "seed": seed,
                "test_lot": int(test_lot),
                "samples": int(
                    np.sum(test_mask)
                ),
                "baseline_mae": fold_baseline_mae,
                "cnn_mae": fold_cnn_mae,
                "improvement": improvement,
                "improved": improved
            }
        )


        print(
            f"baseline={fold_baseline_mae:.4f}, "
            f"CNN={fold_cnn_mae:.4f}, "
            f"{'IMPROVED' if improved else 'WORSE'}"
        )


    # ======================================================
    # Seed overall metric
    # ======================================================

    mae = mean_absolute_error(
        y,
        seed_predictions
    )


    rmse = np.sqrt(
        mean_squared_error(
            y,
            seed_predictions
        )
    )


    r2 = r2_score(
        y,
        seed_predictions
    )


    improvement_percent = (
        (
            baseline_mae
            - mae
        )
        / baseline_mae
        * 100.0
    )


    seed_lot_df = pd.DataFrame(
        [
            r
            for r in lot_results
            if r["seed"] == seed
        ]
    )


    improved_lots = int(
        seed_lot_df[
            "improved"
        ].sum()
    )


    seed_results.append(
        {
            "seed": seed,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "improvement_percent":
                improvement_percent,
            "improved_lots":
                improved_lots
        }
    )


    all_seed_predictions.append(
        seed_predictions.copy()
    )


    print()
    print(
        f"Seed {seed} RESULT"
    )

    print(
        f"  MAE           : "
        f"{mae:.4f} um"
    )

    print(
        f"  RMSE          : "
        f"{rmse:.4f} um"
    )

    print(
        f"  R2            : "
        f"{r2:.4f}"
    )

    print(
        f"  Improved Lots : "
        f"{improved_lots}/8"
    )


# ==========================================================
# Summary
# ==========================================================

seed_df = pd.DataFrame(
    seed_results
)

lot_df = pd.DataFrame(
    lot_results
)


print()
print("=" * 90)
print("MULTI-SEED SUMMARY")
print("=" * 90)

print(
    seed_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ==========================================================
# Stability statistics
# ==========================================================

mean_mae = seed_df[
    "mae"
].mean()

std_mae = seed_df[
    "mae"
].std(
    ddof=1
)

min_mae = seed_df[
    "mae"
].min()

max_mae = seed_df[
    "mae"
].max()


mean_r2 = seed_df[
    "r2"
].mean()

std_r2 = seed_df[
    "r2"
].std(
    ddof=1
)


print()
print("=" * 90)
print("STABILITY STATISTICS")
print("=" * 90)

print(
    f"Mean MAE        : "
    f"{mean_mae:.4f} um"
)

print(
    f"MAE std         : "
    f"{std_mae:.4f} um"
)

print(
    f"Best MAE        : "
    f"{min_mae:.4f} um"
)

print(
    f"Worst MAE       : "
    f"{max_mae:.4f} um"
)

print(
    f"Mean R2         : "
    f"{mean_r2:.4f}"
)

print(
    f"R2 std          : "
    f"{std_r2:.4f}"
)


# ==========================================================
# Lot stability
# ==========================================================

print()
print("=" * 90)
print("LOT-WISE STABILITY")
print("=" * 90)


for lot in unique_lots:

    temp = lot_df[
        lot_df["test_lot"]
        == lot
    ]


    print(
        f"Lot {lot}: "
        f"Mean CNN MAE="
        f"{temp['cnn_mae'].mean():.4f} um | "
        f"Std="
        f"{temp['cnn_mae'].std(ddof=1):.4f} | "
        f"Improved="
        f"{int(temp['improved'].sum())}"
        f"/{len(SEEDS)} seeds"
    )


# ==========================================================
# Final judgement
# ==========================================================

minimum_improved_lots = seed_df[
    "improved_lots"
].min()


print()
print("=" * 90)
print("FINAL STABILITY JUDGEMENT")
print("=" * 90)


if (
    mean_mae < 0.13
    and std_mae < 0.02
    and max_mae < 0.1541
    and minimum_improved_lots >= 6
):

    print("STRONG PASS")

    print(
        "CNN performance is stable "
        "across random seeds."
    )


elif (
    mean_mae < 0.1541
    and std_mae < 0.03
):

    print("PASS")

    print(
        "CNN consistently outperforms "
        "the wafer-order baseline."
    )


elif (
    mean_mae < 0.1541
):

    print("CHECK")

    print(
        "Average performance is good, "
        "but seed sensitivity exists."
    )


else:

    print("FAIL")

    print(
        "CNN performance is not reliably "
        "better than the baseline."
    )


# ==========================================================
# Save
# ==========================================================

seed_df.to_csv(
    "multiseed_summary.csv",
    index=False
)

lot_df.to_csv(
    "multiseed_lot_results.csv",
    index=False
)


all_seed_predictions = np.stack(
    all_seed_predictions,
    axis=0
)


np.savez_compressed(
    "multiseed_predictions.npz",

    seeds=np.asarray(
        SEEDS
    ),

    target=y,

    predictions=
        all_seed_predictions,

    baseline_prediction=
        baseline_predictions,

    lot=lots,

    experiment_key=
        experiments
)


print()
print(
    "Saved:"
)

print(
    "  multiseed_summary.csv"
)

print(
    "  multiseed_lot_results.csv"
)

print(
    "  multiseed_predictions.npz"
)