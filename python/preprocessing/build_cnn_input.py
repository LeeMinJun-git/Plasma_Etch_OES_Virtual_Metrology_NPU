from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Setting
# ==========================================================

OES_FILE = "Day_2024_08_05.nc"
OES_DICT = "Dictionary_OES.nc"

PROCESS_FILE = "Process_data.nc"
PROCESS_DICT = "Dictionary_process.nc"

OES_WAFER = "Wafer_01"
PROCESS_WAFER = "Day_2024_08_05_Wafer_01"

TIME_OFFSET = 0.2

N_WAVELENGTH_BINS = 128


# ==========================================================
# 1. OES load
# ==========================================================

with Dataset(OES_DICT, "r") as ds:
    oes_decoder = np.asarray(ds["data"][:])

with Dataset(OES_FILE, "r") as ds:

    grp = ds.groups[OES_WAFER]

    oes_time_raw = np.asarray(grp["times"][:])
    oes_time = oes_time_raw - oes_time_raw[0]

    wavelengths = np.asarray(grp["wavelengths"][:])

    encoded = np.asarray(
        grp["data"][:],
        dtype=np.int64
    )

    oes_data = oes_decoder[encoded]


# ==========================================================
# 2. Process load
# ==========================================================

with Dataset(PROCESS_DICT, "r") as ds:
    process_decoder = np.asarray(ds["data"][:])

with Dataset(PROCESS_FILE, "r") as ds:

    grp = ds.groups[PROCESS_WAFER]

    process_time_raw = np.asarray(grp["times"][:])
    process_time = process_time_raw - process_time_raw[0]

    features = [
        f.decode() if isinstance(f, bytes) else str(f)
        for f in grp["feature"][:]
    ]

    encoded = np.asarray(
        grp["data"][:],
        dtype=np.int64
    )

    process_data = process_decoder[encoded]


gas4_idx = features.index(
    "Stat3_Etch_MV_Gas4Flow"
)

gas5_idx = features.index(
    "Stat3_Etch_MV_Gas5Flow"
)

gas4 = process_data[:, gas4_idx]
gas5 = process_data[:, gas5_idx]


# ==========================================================
# 3. Gas5 rising edge → 100 BOSCH cycles
# ==========================================================

gas5_high = gas5 > 300

rising_idx = np.where(
    (~gas5_high[:-1]) &
    (gas5_high[1:])
)[0] + 1

rise_times = process_time[rising_idx]

print("All Gas5 rising edges:", len(rise_times))

# Wafer_01:
# 첫 18 sec pulse는 BOSCH 이전 pulse
cycle_starts = rise_times[1:]

if len(cycle_starts) != 100:
    raise RuntimeError(
        f"Expected 100 BOSCH cycles, "
        f"found {len(cycle_starts)}"
    )

BOSCH_END = 628.8

print("Detected cycles :", len(cycle_starts))
print("First cycle :", cycle_starts[0])
print("Last cycle  :", cycle_starts[-1])


# ==========================================================
# 4. OES 시간 → Process 시간으로 변환
# ==========================================================

# OES가 Process보다 약 0.2 sec 늦음
oes_as_process_time = oes_time - TIME_OFFSET


# Process gas 값을 OES timestamp에 맞춤
gas4_at_oes = np.interp(
    oes_as_process_time,
    process_time,
    gas4
)

gas5_at_oes = np.interp(
    oes_as_process_time,
    process_time,
    gas5
)


# ==========================================================
# 5. Wavelength pooling function
# ==========================================================

def wavelength_max_pool(spectrum, bins=128):

    indices = np.array_split(
        np.arange(len(spectrum)),
        bins
    )

    pooled = np.array([
        np.max(spectrum[idx])
        for idx in indices
    ])

    return pooled


# ==========================================================
# 6. 100 cycle × 128 wavelength × 2 phase
# ==========================================================

# cnn_input = np.zeros(
#     (100, N_WAVELENGTH_BINS, 2),
#     dtype=np.float32
# )

cnn_input = np.zeros(
    (100, N_WAVELENGTH_BINS, 1),
    dtype=np.float32
)

# for cycle in range(100):

#     start = cycle_starts[cycle]

#     if cycle < 99:
#         end = cycle_starts[cycle + 1]
#     else:
#         end = BOSCH_END

#     # 현재 cycle에 속하는 OES
#     cycle_mask = (
#         (oes_as_process_time >= start) &
#         (oes_as_process_time < end)
#     )

#     # Gas5 phase
#     gas5_mask = (
#         cycle_mask &
#         (gas5_at_oes > 300)
#     )

#     # Gas4 phase
#     gas4_mask = (
#         cycle_mask &
#         (gas4_at_oes > 150)
#     )

#     if np.sum(gas5_mask) == 0:
#         print(
#             f"Warning: cycle {cycle+1} "
#             "has no Gas5 OES samples"
#         )
#         continue

#     if np.sum(gas4_mask) == 0:
#         print(
#             f"Warning: cycle {cycle+1} "
#             "has no Gas4 OES samples"
#         )
#         continue

#     # Phase별 평균 spectrum
#     gas5_spectrum = np.mean(
#         oes_data[gas5_mask, :],
#         axis=0
#     )

#     gas4_spectrum = np.mean(
#         oes_data[gas4_mask, :],
#         axis=0
#     )

#     # 3648 → 128 wavelength bins
#     cnn_input[cycle, :, 0] = wavelength_max_pool(
#         gas5_spectrum,
#         N_WAVELENGTH_BINS
#     )

#     cnn_input[cycle, :, 1] = wavelength_max_pool(
#         gas4_spectrum,
#         N_WAVELENGTH_BINS
#     )

for cycle in range(100):

    start = cycle_starts[cycle]

    if cycle < 99:
        end = cycle_starts[cycle + 1]
    else:
        end = BOSCH_END

    cycle_mask = (
        (oes_as_process_time >= start) &
        (oes_as_process_time < end)
    )

    gas5_mask = (
        cycle_mask &
        (gas5_at_oes > 300)
    )

    if np.sum(gas5_mask) == 0:
        raise RuntimeError(
            f"Cycle {cycle+1}: "
            "No Gas5 OES samples"
        )

    gas5_spectrum = np.mean(
        oes_data[gas5_mask, :],
        axis=0
    )

    cnn_input[cycle, :, 0] = wavelength_max_pool(
        gas5_spectrum,
        N_WAVELENGTH_BINS
    )


# ==========================================================
# 7. Save
# ==========================================================

# np.save(
#     "Wafer_01_input_100x128x2.npy",
#     cnn_input
# )

np.save(
    "Wafer_01_input_100x128x1.npy",
    cnn_input
)

print("\nCNN input shape :", cnn_input.shape)

# print(
#     "Gas5 min/max :",
#     cnn_input[:, :, 0].min(),
#     cnn_input[:, :, 0].max()
# )

# print(
#     "Gas4 min/max :",
#     cnn_input[:, :, 1].min(),
#     cnn_input[:, :, 1].max()
# )

print(
    "Gas5 min/max :",
    cnn_input[:, :, 0].min(),
    cnn_input[:, :, 0].max()
)


# ==========================================================
# 8. Visualization
# ==========================================================

# 시각화에만 log 사용
# gas5_view = np.log1p(cnn_input[:, :, 0])
# gas4_view = np.log1p(cnn_input[:, :, 1])
gas5_view = np.log1p(cnn_input[:, :, 0])


plt.figure(figsize=(12, 5))

plt.imshow(
    gas5_view,
    aspect="auto",
    origin="lower"
)

plt.xlabel("Wavelength Bin")
plt.ylabel("BOSCH Etch Cycle")
plt.title("Wafer_01 - Gas5 Etch Phase OES")

plt.colorbar(
    label="log(1 + intensity)"
)

plt.tight_layout()

plt.savefig(
    "Wafer_01_Gas5_100x128.png",
    dpi=150
)

plt.show()