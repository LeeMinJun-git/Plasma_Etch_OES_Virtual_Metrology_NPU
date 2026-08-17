import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.linear_model import LinearRegression


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

X = data["X"].astype(
    np.float32
)

y = data["y"].astype(
    np.float32
)

wafers = data["wafer"].astype(
    np.float32
)

print("X :", X.shape)
print("y :", y.shape)


# ==========================================================
# Model
# ==========================================================

class ResidualTinyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(
            1, 4,
            kernel_size=3,
            padding=1
        )

        self.pool1 = nn.MaxPool2d(
            2, 2
        )

        self.conv2 = nn.Conv2d(
            4, 8,
            kernel_size=3,
            padding=1
        )

        self.pool2 = nn.MaxPool2d(
            2, 2
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
            0.2
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
# Quantization utilities
# ==========================================================

def round_away_from_zero(x):

    return np.where(
        x >= 0,
        np.floor(x + 0.5),
        np.ceil(x - 0.5)
    )


def symmetric_scale(x):

    max_abs = float(
        np.max(np.abs(x))
    )

    if max_abs < 1e-12:
        raise RuntimeError(
            "Dead tensor detected during quantization"
        )

    return max_abs / QMAX


def quantize_int8(
    x,
    scale
):

    q = round_away_from_zero(
        x / scale
    )

    q = np.clip(
        q,
        -127,
        127
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

    # FPGA에서는 INT32 bias 사용
    if (
        q.min()
        < np.iinfo(np.int32).min
        or
        q.max()
        > np.iinfo(np.int32).max
    ):
        raise RuntimeError(
            "Bias INT32 overflow"
        )

    return q.astype(
        np.int32
    )


# ==========================================================
# 1. Full-data wafer baseline
#
# 실제 PS에서 수행할 부분
# ==========================================================

baseline = LinearRegression()

baseline.fit(
    wafers.reshape(-1, 1),
    y
)

baseline_prediction = baseline.predict(
    wafers.reshape(-1, 1)
)


baseline_coef = float(
    baseline.coef_[0]
)

baseline_intercept = float(
    baseline.intercept_
)


print()
print("=" * 80)
print("FINAL WAFER BASELINE")
print("=" * 80)

print(
    f"Coefficient : {baseline_coef:.8f}"
)

print(
    f"Intercept   : {baseline_intercept:.8f}"
)


# ==========================================================
# 2. Residual target
# ==========================================================

residual = (
    y
    - baseline_prediction
)

residual_mean = float(
    np.mean(residual)
)

residual_std = float(
    np.std(residual)
)


residual_norm = (
    residual
    - residual_mean
) / (
    residual_std
    + 1e-8
)


print()
print(
    f"Residual mean : "
    f"{residual_mean:.8f}"
)

print(
    f"Residual std  : "
    f"{residual_std:.8f}"
)


# ==========================================================
# 3. Input preprocessing
# ==========================================================

X_pre = np.log1p(
    X
)


x_mean = float(
    np.mean(X_pre)
)

x_std = float(
    np.std(X_pre)
)


X_pre = (
    X_pre
    - x_mean
) / (
    x_std
    + 1e-8
)


# NHWC -> NCHW
X_pre = np.transpose(
    X_pre,
    (0, 3, 1, 2)
)


print()
print(
    f"Input mean : {x_mean:.8f}"
)

print(
    f"Input std  : {x_std:.8f}"
)


# ==========================================================
# 4. Tensor / DataLoader
# ==========================================================

X_tensor = torch.tensor(
    X_pre,
    dtype=torch.float32
)

y_tensor = torch.tensor(
    residual_norm,
    dtype=torch.float32
)


dataset = TensorDataset(
    X_tensor,
    y_tensor
)


generator = torch.Generator()

generator.manual_seed(
    SEED
)


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=generator
)


# ==========================================================
# 5. Train final deployment model
# ==========================================================

model = ResidualTinyCNN().to(
    device
)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


loss_fn = nn.MSELoss()


print()
print("=" * 80)
print("TRAIN FINAL DEPLOYMENT MODEL")
print("=" * 80)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0


    for bx, by in loader:

        bx = bx.to(
            device
        )

        by = by.to(
            device
        )


        optimizer.zero_grad()

        pred = model(
            bx
        )

        loss = loss_fn(
            pred,
            by
        )

        loss.backward()

        optimizer.step()


        total_loss += (
            loss.item()
            * len(bx)
        )


    total_loss /= len(
        dataset
    )


    if (
        epoch == 0
        or
        (epoch + 1) % 50 == 0
    ):

        print(
            f"Epoch {epoch+1:3d}/{EPOCHS}"
            f" | Loss="
            f"{total_loss:.6f}"
        )


# ==========================================================
# 6. Calibration
# ==========================================================

model.eval()


with torch.no_grad():

    calibration_input = (
        X_tensor.to(device)
    )

    a1, a2, a3 = (
        model.forward_calibration(
            calibration_input
        )
    )


a1 = a1.cpu().numpy()
a2 = a2.cpu().numpy()
a3 = a3.cpu().numpy()


print()
print("=" * 80)
print("ACTIVATION CHECK")
print("=" * 80)


for name, a in [
    ("A1", a1),
    ("A2", a2),
    ("A3", a3)
]:

    print(
        f"{name}: "
        f"min={a.min():.6f}, "
        f"max={a.max():.6f}, "
        f"mean={a.mean():.6f}, "
        f"nonzero="
        f"{np.count_nonzero(a)}/{a.size}"
    )


# Dead-layer 방지
if np.max(np.abs(a1)) < 1e-12:
    raise RuntimeError("A1 is dead")

if np.max(np.abs(a2)) < 1e-12:
    raise RuntimeError("A2 is dead")

if np.max(np.abs(a3)) < 1e-12:
    raise RuntimeError("A3 is dead")


# ==========================================================
# 7. Activation scales
# ==========================================================

s_input = symmetric_scale(
    X_pre
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


# ==========================================================
# 8. FP32 weights
# ==========================================================

w1 = (
    model.conv1.weight
    .detach()
    .cpu()
    .numpy()
)

b1 = (
    model.conv1.bias
    .detach()
    .cpu()
    .numpy()
)


w2 = (
    model.conv2.weight
    .detach()
    .cpu()
    .numpy()
)

b2 = (
    model.conv2.bias
    .detach()
    .cpu()
    .numpy()
)


w3 = (
    model.fc1.weight
    .detach()
    .cpu()
    .numpy()
)

b3 = (
    model.fc1.bias
    .detach()
    .cpu()
    .numpy()
)


w4 = (
    model.fc2.weight
    .detach()
    .cpu()
    .numpy()
)

b4 = (
    model.fc2.bias
    .detach()
    .cpu()
    .numpy()
)


# ==========================================================
# 9. Weight scales
# ==========================================================

s_w1 = symmetric_scale(w1)
s_w2 = symmetric_scale(w2)
s_w3 = symmetric_scale(w3)
s_w4 = symmetric_scale(w4)


# ==========================================================
# 10. INT8 weights
# ==========================================================

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


# ==========================================================
# 11. INT32 biases
# ==========================================================

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

# AvgPool이 scale 유지
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


# ==========================================================
# 12. Quantization report
# ==========================================================

print()
print("=" * 80)
print("FINAL QUANTIZATION PARAMETERS")
print("=" * 80)

print(
    f"s_input = {s_input:.10f}"
)

print(
    f"s_a1    = {s_a1:.10f}"
)

print(
    f"s_a2    = {s_a2:.10f}"
)

print(
    f"s_a3    = {s_a3:.10f}"
)

print()

print(
    f"s_w1    = {s_w1:.10f}"
)

print(
    f"s_w2    = {s_w2:.10f}"
)

print(
    f"s_w3    = {s_w3:.10f}"
)

print(
    f"s_w4    = {s_w4:.10f}"
)


# 마지막 FC INT32의 physical residual scale
output_norm_scale = (
    s_a3
    * s_w4
)

output_um_scale = (
    output_norm_scale
    * residual_std
)


print()
print(
    f"Output normalized scale : "
    f"{output_norm_scale:.12f}"
)

print(
    f"Output physical scale   : "
    f"{output_um_scale:.12f} um/count"
)


# ==========================================================
# 13. Save FP32 checkpoint
# ==========================================================

torch.save(
    model.state_dict(),
    "deployment_fp32_model.pt"
)


# ==========================================================
# 14. Save complete deployment package
# ==========================================================

np.savez_compressed(
    "deployment_int8_model.npz",

    # ----------------------------------------------
    # PS preprocessing
    # ----------------------------------------------

    x_mean=np.float32(
        x_mean
    ),

    x_std=np.float32(
        x_std
    ),


    # ----------------------------------------------
    # Wafer-order baseline
    # ----------------------------------------------

    baseline_coef=np.float32(
        baseline_coef
    ),

    baseline_intercept=np.float32(
        baseline_intercept
    ),


    # ----------------------------------------------
    # Residual de-normalization
    # ----------------------------------------------

    residual_mean=np.float32(
        residual_mean
    ),

    residual_std=np.float32(
        residual_std
    ),


    # ----------------------------------------------
    # Activation scales
    # ----------------------------------------------

    s_input=np.float64(
        s_input
    ),

    s_a1=np.float64(
        s_a1
    ),

    s_a2=np.float64(
        s_a2
    ),

    s_a3=np.float64(
        s_a3
    ),


    # ----------------------------------------------
    # Weight scales
    # ----------------------------------------------

    s_w1=np.float64(
        s_w1
    ),

    s_w2=np.float64(
        s_w2
    ),

    s_w3=np.float64(
        s_w3
    ),

    s_w4=np.float64(
        s_w4
    ),


    # ----------------------------------------------
    # Quantized parameters
    # ----------------------------------------------

    conv1_weight=qw1,
    conv1_bias=qb1,

    conv2_weight=qw2,
    conv2_bias=qb2,

    fc1_weight=qw3,
    fc1_bias=qb3,

    fc2_weight=qw4,
    fc2_bias=qb4,


    # ----------------------------------------------
    # Final output scaling
    # ----------------------------------------------

    output_norm_scale=np.float64(
        output_norm_scale
    ),

    output_um_scale=np.float64(
        output_um_scale
    )
)


print()
print("=" * 80)
print("EXPORT COMPLETE")
print("=" * 80)

print(
    "Saved: deployment_fp32_model.pt"
)

print(
    "Saved: deployment_int8_model.npz"
)