import numpy as np

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

SEED = 42

EPOCHS = 250
BATCH_SIZE = 8

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# ==========================================================
# Seed
# ==========================================================

np.random.seed(SEED)
torch.manual_seed(SEED)


# ==========================================================
# Device
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


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


print("X shape :", X.shape)
print("y shape :", y.shape)
print("Lots    :", np.unique(lots))


# ==========================================================
# Residual Tiny CNN
# ==========================================================

class ResidualTinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            # ----------------------------------------------
            # 100 × 128 × 1
            # ->
            # 100 × 128 × 4
            # ----------------------------------------------

            nn.Conv2d(
                in_channels=1,
                out_channels=4,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            # ----------------------------------------------
            # 50 × 64 × 4
            # ->
            # 50 × 64 × 8
            # ----------------------------------------------

            nn.Conv2d(
                in_channels=4,
                out_channels=8,
                kernel_size=3,
                padding=1,
                bias=True
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),


            # ----------------------------------------------
            # 25 × 32 × 8
            # ->
            # 5 × 8 × 8
            #
            # 완전 GAP가 아니라
            # coarse spatial position 유지
            # ----------------------------------------------

            nn.AvgPool2d(
                kernel_size=(5, 4),
                stride=(5, 4)
            )
        )


        self.regressor = nn.Sequential(

            nn.Flatten(),

            # 5 × 8 × 8 = 320
            nn.Linear(
                320,
                16
            ),

            nn.ReLU(),

            # 학습 시 overfitting 억제
            # FPGA inference에는 존재하지 않음
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
# Parameter count
# ==========================================================

temp_model = ResidualTinyCNN()

parameters = sum(
    p.numel()
    for p in temp_model.parameters()
)

print(
    "Trainable parameters:",
    parameters
)


# ==========================================================
# Result containers
# ==========================================================

cnn_predictions = np.zeros_like(
    y
)

baseline_predictions = np.zeros_like(
    y
)


unique_lots = np.unique(
    lots
)


print()
print("=" * 80)
print("RESIDUAL TINY CNN - LEAVE-ONE-LOT-OUT")
print("=" * 80)


# ==========================================================
# Outer LOLO
# ==========================================================

for test_lot in unique_lots:

    print()
    print("-" * 80)
    print(
        f"TEST LOT {test_lot}"
    )
    print("-" * 80)


    # ======================================================
    # Split
    # ======================================================

    train_mask = (
        lots != test_lot
    )

    test_mask = (
        lots == test_lot
    )


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


    # ======================================================
    # 1. Wafer-order baseline
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


    baseline_predictions[
        test_mask
    ] = base_test_pred


    # ======================================================
    # 2. Residual Target
    # ======================================================

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
        residual_std + 1e-8
    )


    # ======================================================
    # 3. OES preprocessing
    #
    # dynamic range 완화
    # ======================================================

    X_train = np.log1p(
        X_train
    )

    X_test = np.log1p(
        X_test
    )


    # ======================================================
    # Train fold만 이용해서 normalization
    # ======================================================

    x_mean = np.mean(
        X_train
    )

    x_std = np.std(
        X_train
    )


    X_train = (
        X_train - x_mean
    ) / (
        x_std + 1e-8
    )


    X_test = (
        X_test - x_mean
    ) / (
        x_std + 1e-8
    )


    # ======================================================
    # NHWC -> NCHW
    # ======================================================

    X_train = np.transpose(
        X_train,
        (0, 3, 1, 2)
    )

    X_test = np.transpose(
        X_test,
        (0, 3, 1, 2)
    )


    # ======================================================
    # Tensor
    # ======================================================

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    residual_tensor = torch.tensor(
        residual_train_norm,
        dtype=torch.float32
    )

    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )


    train_dataset = TensorDataset(
        X_train_tensor,
        residual_tensor
    )


    generator = torch.Generator()

    generator.manual_seed(
        SEED + int(test_lot)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )


    # ======================================================
    # Model
    # ======================================================

    torch.manual_seed(
        SEED + int(test_lot)
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


    # ======================================================
    # Training
    # ======================================================

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0


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


            total_loss += (
                loss.item()
                * len(batch_X)
            )


        total_loss /= len(
            train_dataset
        )


        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
        ):

            print(
                f"Epoch "
                f"{epoch+1:3d}/{EPOCHS}"
                f" | Loss="
                f"{total_loss:.6f}"
            )


    # ======================================================
    # Test
    # ======================================================

    model.eval()


    with torch.no_grad():

        pred_norm = model(
            X_test_tensor.to(
                device
            )
        ).cpu().numpy()


    # residual physical unit 복원
    residual_pred = (
        pred_norm
        * residual_std
        + residual_mean
    )


    # ======================================================
    # 최종 prediction
    # ======================================================

    final_prediction = (
        base_test_pred
        + residual_pred
    )


    cnn_predictions[
        test_mask
    ] = final_prediction


    # ======================================================
    # Fold metrics
    # ======================================================

    baseline_mae = mean_absolute_error(
        y_test,
        base_test_pred
    )


    cnn_mae = mean_absolute_error(
        y_test,
        final_prediction
    )


    improvement = (
        baseline_mae
        - cnn_mae
    )


    print()

    print(
        f"Samples             : "
        f"{len(y_test)}"
    )

    print(
        f"Wafer baseline MAE  : "
        f"{baseline_mae:.4f} um"
    )

    print(
        f"+ CNN residual MAE  : "
        f"{cnn_mae:.4f} um"
    )

    print(
        f"Improvement         : "
        f"{improvement:+.4f} um"
    )


# ==========================================================
# Overall metrics
# ==========================================================

baseline_mae = mean_absolute_error(
    y,
    baseline_predictions
)

cnn_mae = mean_absolute_error(
    y,
    cnn_predictions
)


cnn_rmse = np.sqrt(
    mean_squared_error(
        y,
        cnn_predictions
    )
)


cnn_r2 = r2_score(
    y,
    cnn_predictions
)


improvement_percent = (
    (
        baseline_mae
        - cnn_mae
    )
    / baseline_mae
    * 100
)


print()
print("=" * 80)
print("OVERALL RESIDUAL TINY CNN RESULT")
print("=" * 80)


print(
    f"Wafer baseline MAE : "
    f"{baseline_mae:.4f} um"
)

print(
    f"CNN residual MAE   : "
    f"{cnn_mae:.4f} um"
)

print(
    f"CNN RMSE           : "
    f"{cnn_rmse:.4f} um"
)

print(
    f"CNN R2             : "
    f"{cnn_r2:.4f}"
)

print(
    f"Improvement        : "
    f"{improvement_percent:.1f}%"
)


print()
print("=" * 80)
print("REFERENCE")
print("=" * 80)

print(
    "Mean baseline        : 0.2984 um"
)

print(
    "OES Ridge            : 0.2319 um"
)

print(
    "Wafer baseline       : 0.1541 um"
)

print(
    "Wafer + OES Ridge    : 0.0988 um"
)


# ==========================================================
# Save
# ==========================================================

np.savez_compressed(
    "residual_tiny_cnn_result.npz",

    target=y,

    prediction=cnn_predictions,

    baseline_prediction=baseline_predictions,

    lot=lots,

    experiment_key=experiments
)


print()
print(
    "Saved: residual_tiny_cnn_result.npz"
)