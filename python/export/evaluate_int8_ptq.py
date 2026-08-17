import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

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

QMAX = 127


# ==========================================================
# Reproducibility
# ==========================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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

X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)

lots = data["lot"].astype(np.int32)
wafers = data["wafer"].astype(np.float32)

experiments = data["experiment_key"]

unique_lots = np.unique(lots)


print("X shape :", X.shape)
print("y shape :", y.shape)
print("Lots    :", unique_lots)


# ==========================================================
# FP32 Model
# ==========================================================

class ResidualTinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1,
            4,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            2,
            2
        )

        self.conv2 = nn.Conv2d(
            4,
            8,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            2,
            2
        )

        self.avgpool = nn.AvgPool2d(
            kernel_size=(5, 4),
            stride=(5, 4)
        )

        self.fc1 = nn.Linear(
            320,
            16
        )

        self.dropout = nn.Dropout(
            p=0.2
        )

        self.fc2 = nn.Linear(
            16,
            1
        )


    def forward(self, x):

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc1(x)
        x = torch.relu(x)

        x = self.dropout(x)

        x = self.fc2(x)

        return x.squeeze(1)


    # Calibration용 activation 반환
    def forward_calibration(self, x):

        x = self.conv1(x)
        a1 = torch.relu(x)

        x = self.pool1(a1)

        x = self.conv2(x)
        a2 = torch.relu(x)

        x = self.pool2(a2)

        x = self.avgpool(x)

        x = torch.flatten(
            x,
            1
        )

        x = self.fc1(x)
        a3 = torch.relu(x)

        return a1, a2, a3


# ==========================================================
# Quantization utility
# ==========================================================

def round_away_from_zero(x):

    return np.where(
        x >= 0,
        np.floor(x + 0.5),
        np.ceil(x - 0.5)
    )


def symmetric_scale(x):

    max_abs = np.max(
        np.abs(x)
    )

    if max_abs < 1e-12:
        return 1.0

    return float(
        max_abs / QMAX
    )


def quantize_int8(
    x,
    scale
):

    q = round_away_from_zero(
        x / scale
    )

    q = np.clip(
        q,
        -QMAX,
        QMAX
    )

    return q.astype(
        np.int8
    )


def quantize_bias(
    bias,
    input_scale,
    weight_scale
):

    scale = (
        input_scale
        * weight_scale
    )

    q = round_away_from_zero(
        bias / scale
    )

    return q.astype(
        np.int64
    )


def requantize(
    accumulator,
    input_scale,
    weight_scale,
    output_scale
):

    multiplier = (
        input_scale
        * weight_scale
        / output_scale
    )

    value = (
        accumulator.astype(np.float64)
        * multiplier
    )

    value = round_away_from_zero(
        value
    )

    value = np.clip(
        value,
        -QMAX,
        QMAX
    )

    return value.astype(
        np.int8
    )


# ==========================================================
# Integer Conv2D
#
# NCHW
# stride = 1
# padding = 1
# 3x3 only
# ==========================================================

def conv2d_int(
    x_q,
    w_q,
    b_q
):

    N, Cin, H, W = x_q.shape

    Cout, _, KH, KW = w_q.shape


    x_pad = np.pad(
        x_q,
        (
            (0, 0),
            (0, 0),
            (1, 1),
            (1, 1)
        ),
        mode="constant"
    )


    output = np.zeros(
        (
            N,
            Cout,
            H,
            W
        ),
        dtype=np.int64
    )


    for oc in range(Cout):

        acc = np.full(
            (
                N,
                H,
                W
            ),
            b_q[oc],
            dtype=np.int64
        )


        for ic in range(Cin):

            for kh in range(KH):

                for kw in range(KW):

                    patch = x_pad[
                        :,
                        ic,
                        kh:kh + H,
                        kw:kw + W
                    ].astype(
                        np.int64
                    )


                    weight = int(
                        w_q[
                            oc,
                            ic,
                            kh,
                            kw
                        ]
                    )


                    acc += (
                        patch
                        * weight
                    )


        output[
            :,
            oc,
            :,
            :
        ] = acc


    # 현재 구조에서는 충분히 INT32 범위
    if (
        output.min()
        < np.iinfo(np.int32).min
        or
        output.max()
        > np.iinfo(np.int32).max
    ):

        raise RuntimeError(
            "INT32 accumulator overflow"
        )


    return output.astype(
        np.int32
    )


# ==========================================================
# INT8 MaxPool 2x2
# ==========================================================

def maxpool2x2_int(x):

    N, C, H, W = x.shape

    x = x.reshape(
        N,
        C,
        H // 2,
        2,
        W // 2,
        2
    )

    return x.max(
        axis=(3, 5)
    ).astype(
        np.int8
    )


# ==========================================================
# Integer AvgPool 5x4
#
# Sum 20 values -> integer rounded divide
# scale remains unchanged
# ==========================================================

def avgpool5x4_int(x):

    N, C, H, W = x.shape

    x64 = x.astype(
        np.int64
    )


    x64 = x64.reshape(
        N,
        C,
        H // 5,
        5,
        W // 4,
        4
    )


    summed = x64.sum(
        axis=(3, 5)
    )


    averaged = round_away_from_zero(
        summed.astype(np.float64)
        / 20.0
    )


    averaged = np.clip(
        averaged,
        -QMAX,
        QMAX
    )


    return averaged.astype(
        np.int8
    )


# ==========================================================
# Integer Fully Connected
# ==========================================================

def linear_int(
    x_q,
    w_q,
    b_q
):

    acc = (
        x_q.astype(np.int64)
        @
        w_q.astype(np.int64).T
    )

    acc += b_q[
        None,
        :
    ]


    if (
        acc.min()
        < np.iinfo(np.int32).min
        or
        acc.max()
        > np.iinfo(np.int32).max
    ):

        raise RuntimeError(
            "FC INT32 accumulator overflow"
        )


    return acc.astype(
        np.int32
    )


# ==========================================================
# Results
# ==========================================================

fp32_predictions = np.zeros_like(
    y
)

int8_predictions = np.zeros_like(
    y
)

baseline_predictions = np.zeros_like(
    y
)


fold_results = []


print()
print("=" * 90)
print("INT8 PTQ FEASIBILITY - LEAVE-ONE-LOT-OUT")
print("=" * 90)


# ==========================================================
# LOLO
# ==========================================================

for test_lot in unique_lots:

    print()
    print("-" * 90)
    print(
        f"TEST LOT {test_lot}"
    )
    print("-" * 90)


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
    # A. Wafer baseline
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
    # B. Residual target
    # ======================================================

    residual_train = (
        y_train
        - base_train_pred
    )


    residual_mean = float(
        np.mean(residual_train)
    )

    residual_std = float(
        np.std(residual_train)
    )


    residual_train_norm = (
        residual_train
        - residual_mean
    ) / (
        residual_std
        + 1e-8
    )


    # ======================================================
    # C. Input preprocessing
    # ======================================================

    X_train = np.log1p(
        X_train
    )

    X_test = np.log1p(
        X_test
    )


    x_mean = float(
        np.mean(X_train)
    )

    x_std = float(
        np.std(X_train)
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


    # NHWC -> NCHW
    X_train = np.transpose(
        X_train,
        (0, 3, 1, 2)
    )

    X_test = np.transpose(
        X_test,
        (0, 3, 1, 2)
    )


    # ======================================================
    # D. Training tensors
    # ======================================================

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        residual_train_norm,
        dtype=torch.float32
    )


    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )


    generator = torch.Generator()

    generator.manual_seed(
        SEED * 100
        + int(test_lot)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )


    # ======================================================
    # E. Model
    # ======================================================

    model_seed = (
        SEED * 100
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


    # ======================================================
    # F. FP32 Training
    # ======================================================

    for epoch in range(EPOCHS):

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


    # ======================================================
    # G. FP32 Test
    # ======================================================

    model.eval()


    X_test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    ).to(
        device
    )


    with torch.no_grad():

        fp32_norm = model(
            X_test_tensor
        ).cpu().numpy()


    fp32_residual = (
        fp32_norm
        * residual_std
        + residual_mean
    )


    fp32_final = (
        base_test_pred
        + fp32_residual
    )


    fp32_predictions[
        test_mask
    ] = fp32_final


    # ======================================================
    # H. Calibration
    #
    # 오직 Training Lots 사용
    # ======================================================

    with torch.no_grad():

        calibration_input = torch.tensor(
            X_train,
            dtype=torch.float32
        ).to(
            device
        )


        a1, a2, a3 = (
            model.forward_calibration(
                calibration_input
            )
        )


    a1 = a1.cpu().numpy()
    a2 = a2.cpu().numpy()
    a3 = a3.cpu().numpy()


    # Activation scales
    s_input = symmetric_scale(
        X_train
    )

    s_a1 = symmetric_scale(
        a1
    )

    s_a2 = symmetric_scale(
        a2
    )

    s_a3 = symmetric_scale(
        a3
    )


    # ======================================================
    # I. Weight quantization
    # ======================================================

    w1 = model.conv1.weight.detach().cpu().numpy()
    b1 = model.conv1.bias.detach().cpu().numpy()

    w2 = model.conv2.weight.detach().cpu().numpy()
    b2 = model.conv2.bias.detach().cpu().numpy()

    w3 = model.fc1.weight.detach().cpu().numpy()
    b3 = model.fc1.bias.detach().cpu().numpy()

    w4 = model.fc2.weight.detach().cpu().numpy()
    b4 = model.fc2.bias.detach().cpu().numpy()


    s_w1 = symmetric_scale(w1)
    s_w2 = symmetric_scale(w2)
    s_w3 = symmetric_scale(w3)
    s_w4 = symmetric_scale(w4)


    qw1 = quantize_int8(
        w1,
        s_w1
    )

    qw2 = quantize_int8(
        w2,
        s_w2
    )

    qw3 = quantize_int8(
        w3,
        s_w3
    )

    qw4 = quantize_int8(
        w4,
        s_w4
    )


    qb1 = quantize_bias(
        b1,
        s_input,
        s_w1
    )

    qb2 = quantize_bias(
        b2,
        s_a1,
        s_w2
    )

    # AvgPool은 scale을 유지하므로
    # FC1 input scale = s_a2
    qb3 = quantize_bias(
        b3,
        s_a2,
        s_w3
    )

    qb4 = quantize_bias(
        b4,
        s_a3,
        s_w4
    )


    # ======================================================
    # J. INT8 Input
    # ======================================================

    xq = quantize_int8(
        X_test,
        s_input
    )


    # ======================================================
    # K. Conv1
    # ======================================================

    acc1 = conv2d_int(
        xq,
        qw1,
        qb1
    )


    q1 = requantize(
        acc1,
        s_input,
        s_w1,
        s_a1
    )


    # ReLU
    q1 = np.maximum(
        q1,
        0
    ).astype(
        np.int8
    )


    # MaxPool
    q1 = maxpool2x2_int(
        q1
    )


    # ======================================================
    # L. Conv2
    # ======================================================

    acc2 = conv2d_int(
        q1,
        qw2,
        qb2
    )


    q2 = requantize(
        acc2,
        s_a1,
        s_w2,
        s_a2
    )


    q2 = np.maximum(
        q2,
        0
    ).astype(
        np.int8
    )


    q2 = maxpool2x2_int(
        q2
    )


    # ======================================================
    # M. AvgPool 5x4
    # ======================================================

    q2 = avgpool5x4_int(
        q2
    )


    # ======================================================
    # N. Flatten
    # ======================================================

    q_flat = q2.reshape(
        q2.shape[0],
        -1
    )


    # ======================================================
    # O. FC1
    # ======================================================

    acc3 = linear_int(
        q_flat,
        qw3,
        qb3
    )


    q3 = requantize(
        acc3,
        s_a2,
        s_w3,
        s_a3
    )


    q3 = np.maximum(
        q3,
        0
    ).astype(
        np.int8
    )


    # ======================================================
    # P. FC2
    #
    # 마지막은 requant하지 않고
    # INT32 accumulator 그대로 사용
    # ======================================================

    acc4 = linear_int(
        q3,
        qw4,
        qb4
    )


    # INT32 accumulator -> FP residual_norm
    int8_norm = (
        acc4[:, 0].astype(
            np.float64
        )
        *
        (
            s_a3
            * s_w4
        )
    )


    # normalized residual -> um
    int8_residual = (
        int8_norm
        * residual_std
        + residual_mean
    )


    int8_final = (
        base_test_pred
        + int8_residual
    )


    int8_predictions[
        test_mask
    ] = int8_final


    # ======================================================
    # Fold metrics
    # ======================================================

    baseline_mae = mean_absolute_error(
        y_test,
        base_test_pred
    )

    fp32_mae = mean_absolute_error(
        y_test,
        fp32_final
    )

    int8_mae = mean_absolute_error(
        y_test,
        int8_final
    )


    delta = (
        int8_mae
        - fp32_mae
    )


    print(
        f"Baseline MAE : "
        f"{baseline_mae:.4f} um"
    )

    print(
        f"FP32 MAE     : "
        f"{fp32_mae:.4f} um"
    )

    print(
        f"INT8 MAE     : "
        f"{int8_mae:.4f} um"
    )

    print(
        f"INT8 - FP32  : "
        f"{delta:+.4f} um"
    )


    print(
        "Scales:"
    )

    print(
        f"  input={s_input:.8f}"
    )

    print(
        f"  a1   ={s_a1:.8f}"
    )

    print(
        f"  a2   ={s_a2:.8f}"
    )

    print(
        f"  a3   ={s_a3:.8f}"
    )


    fold_results.append(
        {
            "lot": int(test_lot),
            "baseline_mae":
                baseline_mae,
            "fp32_mae":
                fp32_mae,
            "int8_mae":
                int8_mae,
            "delta":
                delta
        }
    )


# ==========================================================
# Overall
# ==========================================================

baseline_mae = mean_absolute_error(
    y,
    baseline_predictions
)


fp32_mae = mean_absolute_error(
    y,
    fp32_predictions
)

fp32_rmse = np.sqrt(
    mean_squared_error(
        y,
        fp32_predictions
    )
)

fp32_r2 = r2_score(
    y,
    fp32_predictions
)


int8_mae = mean_absolute_error(
    y,
    int8_predictions
)

int8_rmse = np.sqrt(
    mean_squared_error(
        y,
        int8_predictions
    )
)

int8_r2 = r2_score(
    y,
    int8_predictions
)


mae_delta = (
    int8_mae
    - fp32_mae
)


print()
print("=" * 90)
print("OVERALL INT8 PTQ RESULT")
print("=" * 90)

print(
    f"Wafer baseline MAE : "
    f"{baseline_mae:.4f} um"
)

print()
print("FP32")

print(
    f"  MAE  : "
    f"{fp32_mae:.4f} um"
)

print(
    f"  RMSE : "
    f"{fp32_rmse:.4f} um"
)

print(
    f"  R2   : "
    f"{fp32_r2:.4f}"
)


print()
print("INT8")

print(
    f"  MAE  : "
    f"{int8_mae:.4f} um"
)

print(
    f"  RMSE : "
    f"{int8_rmse:.4f} um"
)

print(
    f"  R2   : "
    f"{int8_r2:.4f}"
)


print()
print(
    f"INT8 MAE degradation : "
    f"{mae_delta:+.4f} um"
)


# ==========================================================
# Judgement
# ==========================================================

print()
print("=" * 90)
print("INT8 FEASIBILITY JUDGEMENT")
print("=" * 90)


if (
    int8_mae < 0.13
    and mae_delta < 0.01
):

    print("STRONG PASS")

    print(
        "INT8 inference preserves "
        "FP32 accuracy."
    )


elif (
    int8_mae < 0.1541
    and mae_delta < 0.02
):

    print("PASS")

    print(
        "INT8 inference remains better "
        "than the wafer baseline."
    )


elif int8_mae < 0.1541:

    print("CHECK")

    print(
        "INT8 is useful, but quantization "
        "loss should be reduced."
    )


else:

    print("FAIL")

    print(
        "Current PTQ scheme causes "
        "excessive accuracy loss."
    )


# ==========================================================
# Save
# ==========================================================

np.savez_compressed(
    "int8_ptq_result.npz",

    target=y,

    fp32_prediction=
        fp32_predictions,

    int8_prediction=
        int8_predictions,

    baseline_prediction=
        baseline_predictions,

    lot=lots,

    experiment_key=
        experiments
)


print()
print(
    "Saved: int8_ptq_result.npz"
)