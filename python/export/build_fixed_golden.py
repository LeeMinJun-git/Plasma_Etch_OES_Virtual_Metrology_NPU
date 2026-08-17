import numpy as np


# ==========================================================
# Setting
# ==========================================================

DATASET_FILE = "full_dataset_raw.npz"
MODEL_FILE = "deployment_int8_model.npz"

QMAX = 127

# FPGA requantization fractional bits
REQUANT_SHIFT = 24


# ==========================================================
# Load
# ==========================================================

data = np.load(
    DATASET_FILE,
    allow_pickle=True
)

model = np.load(
    MODEL_FILE,
    allow_pickle=True
)


X = data["X"].astype(np.float32)
y = data["y"].astype(np.float32)

wafers = data["wafer"].astype(np.float32)

experiments = data["experiment_key"]


# ==========================================================
# Deployment parameters
# ==========================================================

x_mean = float(model["x_mean"])
x_std = float(model["x_std"])

baseline_coef = float(
    model["baseline_coef"]
)

baseline_intercept = float(
    model["baseline_intercept"]
)

residual_mean = float(
    model["residual_mean"]
)

residual_std = float(
    model["residual_std"]
)


s_input = float(model["s_input"])

s_a1 = float(model["s_a1"])
s_a2 = float(model["s_a2"])
s_a3 = float(model["s_a3"])

s_w1 = float(model["s_w1"])
s_w2 = float(model["s_w2"])
s_w3 = float(model["s_w3"])
s_w4 = float(model["s_w4"])


qw1 = model[
    "conv1_weight"
].astype(np.int8)

qb1 = model[
    "conv1_bias"
].astype(np.int32)

qw2 = model[
    "conv2_weight"
].astype(np.int8)

qb2 = model[
    "conv2_bias"
].astype(np.int32)

qw3 = model[
    "fc1_weight"
].astype(np.int8)

qb3 = model[
    "fc1_bias"
].astype(np.int32)

qw4 = model[
    "fc2_weight"
].astype(np.int8)

qb4 = model[
    "fc2_bias"
].astype(np.int32)


# ==========================================================
# Rounding
#
# 반드시 RTL에서도 같은 규칙 사용
#
# Half away from zero
# ==========================================================

def round_away_from_zero(x):

    return np.where(
        x >= 0,
        np.floor(x + 0.5),
        np.ceil(x - 0.5)
    )


# ==========================================================
# INT8 quantization
# ==========================================================

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


# ==========================================================
# Integer multiplier 생성
#
# real_multiplier
# ≈
# M / 2^SHIFT
# ==========================================================

def make_fixed_multiplier(
    real_multiplier,
    shift
):

    M = int(
        np.floor(
            real_multiplier
            * (1 << shift)
            + 0.5
        )
    )

    approx = (
        M
        / float(1 << shift)
    )

    error = (
        approx
        - real_multiplier
    )

    return M, approx, error


# ==========================================================
# Signed rounding right shift
#
# round half away from zero
#
# RTL에서도 그대로 구현해야 함
# ==========================================================

def rounded_right_shift(
    value,
    shift
):

    value = value.astype(
        np.int64
    )

    result = np.empty_like(
        value
    )

    rounding = (
        1 << (shift - 1)
    )

    positive = (
        value >= 0
    )

    result[
        positive
    ] = (
        value[positive]
        + rounding
    ) >> shift


    negative_value = (
        -value[~positive]
    )

    result[
        ~positive
    ] = -(
        (
            negative_value
            + rounding
        ) >> shift
    )

    return result


# ==========================================================
# Float multiplier reference requantization
#
# 이전 PTQ와 동일한 방식
# ==========================================================

def requantize_reference(
    accumulator,
    real_multiplier
):

    value = (
        accumulator.astype(
            np.float64
        )
        * real_multiplier
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
# FPGA Fixed-point requantization
# ==========================================================

def requantize_fixed(
    accumulator,
    multiplier,
    shift
):

    accumulator64 = (
        accumulator.astype(
            np.int64
        )
    )


    # INT32 accumulator × integer multiplier
    product = (
        accumulator64
        * np.int64(multiplier)
    )


    # INT64 overflow safety
    max_product = np.max(
        np.abs(product)
    )

    if (
        max_product
        >= np.iinfo(np.int64).max
    ):

        raise RuntimeError(
            "INT64 requantization overflow"
        )


    value = rounded_right_shift(
        product,
        shift
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
# Conv2D INT8 × INT8 -> INT32
#
# NCHW
# Kernel = 3×3
# Padding = 1
# ==========================================================

def conv2d_int(
    x_q,
    w_q,
    b_q
):

    N, Cin, H, W = (
        x_q.shape
    )

    Cout, _, KH, KW = (
        w_q.shape
    )


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
            int(b_q[oc]),
            dtype=np.int64
        )


        for ic in range(Cin):

            for kh in range(KH):

                for kw in range(KW):

                    patch = (
                        x_pad[
                            :,
                            ic,
                            kh:kh + H,
                            kw:kw + W
                        ]
                        .astype(
                            np.int64
                        )
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


    if (
        output.min()
        < np.iinfo(np.int32).min
        or
        output.max()
        > np.iinfo(np.int32).max
    ):

        raise RuntimeError(
            "Conv INT32 accumulator overflow"
        )


    return output.astype(
        np.int32
    )


# ==========================================================
# MaxPool 2×2
# ==========================================================

def maxpool2x2_int(x):

    N, C, H, W = (
        x.shape
    )

    x = x.reshape(
        N,
        C,
        H // 2,
        2,
        W // 2,
        2
    )

    return (
        x.max(
            axis=(3, 5)
        )
        .astype(
            np.int8
        )
    )


# ==========================================================
# AvgPool 5×4
#
# 20개 INT8 합산
# -> round(sum / 20)
#
# scale은 s_a2 유지
# ==========================================================

def avgpool5x4_int(x):

    N, C, H, W = (
        x.shape
    )


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
        summed.astype(
            np.float64
        )
        / 20.0
    )


    averaged = np.clip(
        averaged,
        -127,
        127
    )


    return averaged.astype(
        np.int8
    )


# ==========================================================
# Fully Connected
#
# INT8 × INT8 -> INT32
# ==========================================================

def linear_int(
    x_q,
    w_q,
    b_q
):

    acc = (
        x_q.astype(
            np.int64
        )
        @
        w_q.astype(
            np.int64
        ).T
    )


    acc += (
        b_q[
            None,
            :
        ]
    )


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
# Real requant multipliers
# ==========================================================

real_m1 = (
    s_input
    * s_w1
    / s_a1
)

real_m2 = (
    s_a1
    * s_w2
    / s_a2
)

real_m3 = (
    s_a2
    * s_w3
    / s_a3
)


# ==========================================================
# Fixed M + SHIFT
# ==========================================================

M1, approx_m1, err1 = (
    make_fixed_multiplier(
        real_m1,
        REQUANT_SHIFT
    )
)

M2, approx_m2, err2 = (
    make_fixed_multiplier(
        real_m2,
        REQUANT_SHIFT
    )
)

M3, approx_m3, err3 = (
    make_fixed_multiplier(
        real_m3,
        REQUANT_SHIFT
    )
)


print()
print("=" * 90)
print("FIXED-POINT REQUANTIZATION PARAMETERS")
print("=" * 90)


for name, real, M, approx, err in [

    (
        "CONV1",
        real_m1,
        M1,
        approx_m1,
        err1
    ),

    (
        "CONV2",
        real_m2,
        M2,
        approx_m2,
        err2
    ),

    (
        "FC1",
        real_m3,
        M3,
        approx_m3,
        err3
    )
]:

    bits = max(
        1,
        int(M).bit_length()
    )

    print()

    print(name)

    print(
        f"  Real multiplier : "
        f"{real:.12f}"
    )

    print(
        f"  Integer M       : "
        f"{M}"
    )

    print(
        f"  Shift           : "
        f"{REQUANT_SHIFT}"
    )

    print(
        f"  Approx multiplier: "
        f"{approx:.12f}"
    )

    print(
        f"  Approx error    : "
        f"{err:+.12e}"
    )

    print(
        f"  M bit width     : "
        f"{bits} bits"
    )


# ==========================================================
# Input preprocessing
#
# PS에서 수행할 영역
# ==========================================================

X_pre = np.log1p(
    X
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


x_q = quantize_int8(
    X_pre,
    s_input
)


# ==========================================================
# Inference pipeline
# ==========================================================

def run_integer_npu(
    x_q,
    fixed
):

    result = {}


    # ------------------------------------------------------
    # Conv1
    # ------------------------------------------------------

    acc1 = conv2d_int(
        x_q,
        qw1,
        qb1
    )


    if fixed:

        q1 = requantize_fixed(
            acc1,
            M1,
            REQUANT_SHIFT
        )

    else:

        q1 = requantize_reference(
            acc1,
            real_m1
        )


    # ReLU
    q1 = np.maximum(
        q1,
        0
    ).astype(
        np.int8
    )


    result[
        "conv1_relu"
    ] = q1.copy()


    q1_pool = maxpool2x2_int(
        q1
    )


    result[
        "pool1"
    ] = q1_pool.copy()


    # ------------------------------------------------------
    # Conv2
    # ------------------------------------------------------

    acc2 = conv2d_int(
        q1_pool,
        qw2,
        qb2
    )


    if fixed:

        q2 = requantize_fixed(
            acc2,
            M2,
            REQUANT_SHIFT
        )

    else:

        q2 = requantize_reference(
            acc2,
            real_m2
        )


    q2 = np.maximum(
        q2,
        0
    ).astype(
        np.int8
    )


    result[
        "conv2_relu"
    ] = q2.copy()


    q2_pool = maxpool2x2_int(
        q2
    )


    result[
        "pool2"
    ] = q2_pool.copy()


    # ------------------------------------------------------
    # AvgPool
    # ------------------------------------------------------

    q_avg = avgpool5x4_int(
        q2_pool
    )


    result[
        "avgpool"
    ] = q_avg.copy()


    # ------------------------------------------------------
    # Flatten
    # ------------------------------------------------------

    q_flat = q_avg.reshape(
        q_avg.shape[0],
        -1
    )


    # ------------------------------------------------------
    # FC1
    # ------------------------------------------------------

    acc3 = linear_int(
        q_flat,
        qw3,
        qb3
    )


    if fixed:

        q3 = requantize_fixed(
            acc3,
            M3,
            REQUANT_SHIFT
        )

    else:

        q3 = requantize_reference(
            acc3,
            real_m3
        )


    q3 = np.maximum(
        q3,
        0
    ).astype(
        np.int8
    )


    result[
        "fc1_relu"
    ] = q3.copy()


    # ------------------------------------------------------
    # FC2
    #
    # 마지막은 requantization 없음
    # ------------------------------------------------------

    acc4 = linear_int(
        q3,
        qw4,
        qb4
    )


    result[
        "fc2_acc"
    ] = acc4.copy()


    return result


# ==========================================================
# Reference PTQ vs FPGA Fixed
# ==========================================================

print()
print("=" * 90)
print("RUNNING INTEGER INFERENCE")
print("=" * 90)


reference = run_integer_npu(
    x_q,
    fixed=False
)

fixed = run_integer_npu(
    x_q,
    fixed=True
)


# ==========================================================
# Layer comparison
# ==========================================================

def compare_layer(
    name,
    reference,
    fixed
):

    ref64 = reference.astype(
        np.int64
    )

    fix64 = fixed.astype(
        np.int64
    )


    diff = (
        fix64
        - ref64
    )


    mismatches = int(
        np.count_nonzero(
            diff
        )
    )


    total = int(
        diff.size
    )


    max_diff = int(
        np.max(
            np.abs(diff)
        )
    )


    mismatch_rate = (
        mismatches
        / total
        * 100.0
    )


    print(
        f"{name:12s} | "
        f"Mismatch="
        f"{mismatches:8d}/{total:8d} "
        f"({mismatch_rate:.6f}%) | "
        f"Max diff={max_diff}"
    )


    return (
        mismatches,
        total,
        max_diff
    )


print()
print("=" * 90)
print("LAYER BIT COMPARISON")
print("=" * 90)


comparison = {}


for layer in [
    "conv1_relu",
    "pool1",
    "conv2_relu",
    "pool2",
    "avgpool",
    "fc1_relu",
    "fc2_acc"
]:

    comparison[layer] = (
        compare_layer(
            layer,
            reference[layer],
            fixed[layer]
        )
    )


# ==========================================================
# Convert FC2 accumulator to physical residual
# ==========================================================

output_norm_scale = (
    s_a3
    * s_w4
)

output_um_scale = (
    output_norm_scale
    * residual_std
)


reference_acc = (
    reference[
        "fc2_acc"
    ][
        :,
        0
    ].astype(
        np.float64
    )
)

fixed_acc = (
    fixed[
        "fc2_acc"
    ][
        :,
        0
    ].astype(
        np.float64
    )
)


reference_residual = (
    reference_acc
    * output_um_scale
    + residual_mean
)

fixed_residual = (
    fixed_acc
    * output_um_scale
    + residual_mean
)


# ==========================================================
# PS wafer baseline
# ==========================================================

baseline_prediction = (
    baseline_coef
    * wafers
    + baseline_intercept
)


reference_prediction = (
    baseline_prediction
    + reference_residual
)

fixed_prediction = (
    baseline_prediction
    + fixed_residual
)


# ==========================================================
# Output comparison
# ==========================================================

physical_difference = np.abs(
    fixed_prediction
    - reference_prediction
)


print()
print("=" * 90)
print("FINAL OUTPUT COMPARISON")
print("=" * 90)


print(
    f"Mean |Fixed - Reference| : "
    f"{physical_difference.mean():.10f} um"
)

print(
    f"Max  |Fixed - Reference| : "
    f"{physical_difference.max():.10f} um"
)


# ==========================================================
# Deployment-data fit sanity check
#
# 주의:
# 이 MAE는 일반화 성능이 아니라
# deployment 모델 기능 확인용
# ==========================================================

reference_mae = np.mean(
    np.abs(
        y
        - reference_prediction
    )
)

fixed_mae = np.mean(
    np.abs(
        y
        - fixed_prediction
    )
)


print()
print(
    "Deployment-set fit sanity only"
)

print(
    f"Reference PTQ MAE : "
    f"{reference_mae:.6f} um"
)

print(
    f"Fixed-point MAE   : "
    f"{fixed_mae:.6f} um"
)


# ==========================================================
# Save fixed parameters
# ==========================================================

np.savez_compressed(
    "fixed_requant_params.npz",

    conv1_M=np.int32(M1),
    conv1_shift=np.int32(
        REQUANT_SHIFT
    ),

    conv2_M=np.int32(M2),
    conv2_shift=np.int32(
        REQUANT_SHIFT
    ),

    fc1_M=np.int32(M3),
    fc1_shift=np.int32(
        REQUANT_SHIFT
    ),

    output_um_scale=np.float64(
        output_um_scale
    )
)


# ==========================================================
# Save whole-dataset result
# ==========================================================

np.savez_compressed(
    "fixed_golden_result.npz",

    target=y,

    reference_prediction=
        reference_prediction,

    fixed_prediction=
        fixed_prediction,

    fc2_acc=
        fixed["fc2_acc"],

    wafer=wafers,

    experiment_key=
        experiments
)


# ==========================================================
# Save one complete RTL golden sample
#
# 이후 Verilog testbench에서 사용
# ==========================================================

sample = 0


np.savez_compressed(
    "rtl_golden_sample0.npz",

    experiment_key=
        experiments[sample],

    wafer=np.float32(
        wafers[sample]
    ),

    input_q=
        x_q[sample],

    conv1_relu=
        fixed[
            "conv1_relu"
        ][sample],

    pool1=
        fixed[
            "pool1"
        ][sample],

    conv2_relu=
        fixed[
            "conv2_relu"
        ][sample],

    pool2=
        fixed[
            "pool2"
        ][sample],

    avgpool=
        fixed[
            "avgpool"
        ][sample],

    fc1_relu=
        fixed[
            "fc1_relu"
        ][sample],

    fc2_acc=
        fixed[
            "fc2_acc"
        ][sample],

    baseline_prediction=
        np.float64(
            baseline_prediction[
                sample
            ]
        ),

    residual_prediction=
        np.float64(
            fixed_residual[
                sample
            ]
        ),

    final_prediction=
        np.float64(
            fixed_prediction[
                sample
            ]
        )
)


print()
print("=" * 90)
print("EXPORT")
print("=" * 90)

print(
    "Saved: fixed_requant_params.npz"
)

print(
    "Saved: fixed_golden_result.npz"
)

print(
    "Saved: rtl_golden_sample0.npz"
)