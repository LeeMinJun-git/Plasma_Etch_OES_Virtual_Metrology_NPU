import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import (
    TensorDataset,
    DataLoader
)

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

EPOCHS = 300
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
# Dataset Load
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


print("X shape:", X.shape)
print("y shape:", y.shape)
print("Lots:", np.unique(lots))


# ==========================================================
# Tiny CNN
# ==========================================================

class TinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

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
            )
        )


        self.gap = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        self.fc = nn.Linear(
            8,
            1
        )


    def forward(self, x):

        x = self.features(x)

        x = self.gap(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc(x)

        return x.squeeze(1)


# ==========================================================
# Parameter Count
# ==========================================================

temp_model = TinyCNN()

parameter_count = sum(
    p.numel()
    for p in temp_model.parameters()
)

print(
    "Trainable parameters:",
    parameter_count
)


# ==========================================================
# Prediction Containers
# ==========================================================

all_predictions = np.zeros_like(
    y
)

unique_lots = np.unique(
    lots
)


# ==========================================================
# Leave-One-Lot-Out
# ==========================================================

print()
print("=" * 80)
print("TINY CNN - LEAVE-ONE-LOT-OUT")
print("=" * 80)


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


    # ======================================================
    # Input preprocessing
    #
    # log1p:
    # OES의 큰 emission peak dynamic range 완화
    #
    # 반드시 TRAIN 데이터 기준으로만 mean/std 계산
    # ======================================================

    X_train = np.log1p(
        X_train
    )

    X_test = np.log1p(
        X_test
    )


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
    # Target normalization
    #
    # Train fold만 사용
    # ======================================================

    y_mean = np.mean(
        y_train
    )

    y_std = np.std(
        y_train
    )


    y_train_norm = (
        y_train - y_mean
    ) / (
        y_std + 1e-8
    )


    # ======================================================
    # NHWC -> NCHW
    #
    # (N,100,128,1)
    # ->
    # (N,1,100,128)
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

    y_train_tensor = torch.tensor(
        y_train_norm,
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

    model = TinyCNN().to(
        device
    )


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )


    loss_fn = nn.MSELoss()


    # ======================================================
    # Train
    # ======================================================

    model.train()


    for epoch in range(EPOCHS):

        epoch_loss = 0.0


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


            epoch_loss += (
                loss.item()
                * len(batch_X)
            )


        epoch_loss /= len(
            train_dataset
        )


        if (
            epoch == 0
            or (epoch + 1) % 50 == 0
        ):

            print(
                f"Epoch "
                f"{epoch+1:3d}/{EPOCHS} "
                f"| Loss = "
                f"{epoch_loss:.6f}"
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


    # physical unit 복원
    prediction = (
        pred_norm * y_std
        + y_mean
    )


    all_predictions[
        test_mask
    ] = prediction


    # ======================================================
    # Fold Metric
    # ======================================================

    fold_mae = mean_absolute_error(
        y_test,
        prediction
    )


    fold_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )


    print()
    print(
        f"Test samples : "
        f"{len(y_test)}"
    )

    print(
        f"Fold MAE     : "
        f"{fold_mae:.4f} um"
    )

    print(
        f"Fold RMSE    : "
        f"{fold_rmse:.4f} um"
    )


    print()
    print(
        "Target / Prediction"
    )


    test_indices = np.where(
        test_mask
    )[0]


    for idx, pred_value in zip(
        test_indices,
        prediction
    ):

        print(
            f"{experiments[idx]} | "
            f"Target={y[idx]:.4f} | "
            f"Pred={pred_value:.4f}"
        )


# ==========================================================
# Overall Evaluation
# ==========================================================

mae = mean_absolute_error(
    y,
    all_predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y,
        all_predictions
    )
)

r2 = r2_score(
    y,
    all_predictions
)


print()
print("=" * 80)
print("OVERALL TINY CNN RESULT")
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


# ==========================================================
# Baseline reference
# ==========================================================

print()
print("=" * 80)
print("REFERENCE")
print("=" * 80)

print(
    "Mean baseline MAE         : "
    "0.2984 um"
)

print(
    "Wafer-order baseline MAE  : "
    "0.1541 um"
)


# ==========================================================
# Save predictions
# ==========================================================

np.savez_compressed(
    "tiny_cnn_lolo_result.npz",

    target=y,

    prediction=all_predictions,

    lot=lots,

    experiment_key=experiments
)


print()
print(
    "Saved: tiny_cnn_lolo_result.npz"
)