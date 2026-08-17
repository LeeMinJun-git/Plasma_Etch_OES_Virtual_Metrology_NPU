from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. OES
# =========================================================

with Dataset("Dictionary_OES.nc", "r") as ds:
    oes_decoder = np.asarray(ds["data"][:])

with Dataset("Day_2024_08_05.nc", "r") as ds:
    grp = ds.groups["Wafer_01"]

    oes_time_raw = np.asarray(grp["times"][:])
    wavelengths = np.asarray(grp["wavelengths"][:])

    # 시작점을 0초로 변환
    oes_time = oes_time_raw - oes_time_raw[0]

    # 495~505 nm 영역 선택
    wave_mask = (
        (wavelengths >= 495) &
        (wavelengths <= 505)
    )

    wave_idx = np.where(wave_mask)[0]

    print("OES wavelength channels used:", len(wave_idx))

    # 해당 wavelength 영역만 읽기
    encoded = np.asarray(
        grp["data"][:, wave_idx],
        dtype=np.int64
    )

    decoded = oes_decoder[encoded]

    # 495~505 nm 평균 intensity
    oes_500 = np.mean(decoded, axis=1)


# =========================================================
# 2. PROCESS DATA
# =========================================================

with Dataset("Dictionary_process.nc", "r") as ds:
    process_decoder = np.asarray(ds["data"][:])

with Dataset("Process_data.nc", "r") as ds:

    grp = ds.groups["Day_2024_08_05_Wafer_01"]

    process_time_raw = np.asarray(grp["times"][:])

    # 시작점을 0초로 변환
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


# =========================================================
# 3. Plot
# =========================================================

fig, ax1 = plt.subplots(figsize=(14, 6))

# OES
ax1.plot(
    oes_time,
    oes_500,
    label="OES 495-505 nm",
)

ax1.set_xlabel("Relative Time (sec)")
ax1.set_ylabel("OES Intensity")


# Gas Flow를 오른쪽 Y축에 표시
ax2 = ax1.twinx()

ax2.plot(
    process_time,
    gas4,
    alpha=0.6,
    label="Gas4"
)

ax2.plot(
    process_time,
    gas5,
    alpha=0.6,
    label="Gas5"
)

ax2.set_ylabel("Gas Flow")


# 범위 제한
ax1.set_xlim(0, 100)

ax1.set_title(
    "Wafer_01 - OES vs BOSCH Gas Cycle"
)

ax1.grid()

# Legend 합치기
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right"
)

plt.tight_layout()
plt.savefig(
    "Wafer_01_oes_process_sync.png",
    dpi=150
)

plt.show()