from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# 1. OES 데이터
# ==========================================================

with Dataset("Dictionary_OES.nc", "r") as ds:
    oes_decoder = np.asarray(ds["data"][:])

with Dataset("Day_2024_08_05.nc", "r") as ds:
    grp = ds.groups["Wafer_01"]

    oes_time_raw = np.asarray(grp["times"][:])
    oes_time = oes_time_raw - oes_time_raw[0]

    wavelengths = np.asarray(grp["wavelengths"][:])

    # 반복 패턴이 잘 보였던 495~505 nm
    wave_idx = np.where(
        (wavelengths >= 495) &
        (wavelengths <= 505)
    )[0]

    encoded = np.asarray(
        grp["data"][:, wave_idx],
        dtype=np.int64
    )

    decoded = oes_decoder[encoded]

    oes_signal = np.mean(decoded, axis=1)


# ==========================================================
# 2. Process Gas5
# ==========================================================

with Dataset("Dictionary_process.nc", "r") as ds:
    process_decoder = np.asarray(ds["data"][:])

with Dataset("Process_data.nc", "r") as ds:
    grp = ds.groups["Day_2024_08_05_Wafer_01"]

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

    gas5_idx = features.index(
        "Stat3_Etch_MV_Gas5Flow"
    )

    gas5 = process_data[:, gas5_idx]


# ==========================================================
# 3. BOSCH 구간 중심으로 공통 시간축 생성
# ==========================================================

dt = 0.05    # 20 Hz

common_time = np.arange(
    25,
    630,
    dt
)

oes_interp = np.interp(
    common_time,
    oes_time,
    oes_signal
)

gas_interp = np.interp(
    common_time,
    process_time,
    gas5
)


# ==========================================================
# 4. Normalize
# ==========================================================

oes_norm = (
    oes_interp - np.mean(oes_interp)
) / np.std(oes_interp)

gas_norm = (
    gas_interp - np.mean(gas_interp)
) / np.std(gas_interp)


# ==========================================================
# 5. Cross-correlation
# ==========================================================

# corr = np.correlate(
#     oes_norm,
#     gas_norm,
#     mode="full"
# )

# lags = np.arange(
#     -len(gas_norm) + 1,
#     len(oes_norm)
# )

# # ±10초 안에서만 검색
# max_lag_samples = int(10 / dt)

# center = len(corr) // 2

# search_start = center - max_lag_samples
# search_end   = center + max_lag_samples + 1

# local_corr = corr[search_start:search_end]
# local_lags = lags[search_start:search_end]

# best_idx = np.argmax(np.abs(local_corr))

# best_lag_samples = local_lags[best_idx]
# best_lag_sec = best_lag_samples * dt

# print("Best lag :", best_lag_sec, "sec")

# ==========================================================
# 5. Cross-correlation
#    6초 주기 alias를 피하기 위해 ±3초만 탐색
# ==========================================================

max_lag_sec = 3
max_lag_samples = int(max_lag_sec / dt)

lags = np.arange(
    -max_lag_samples,
    max_lag_samples + 1
)

scores = []

for lag in lags:

    if lag < 0:
        oes_part = oes_norm[:lag]
        gas_part = gas_norm[-lag:]

    elif lag > 0:
        oes_part = oes_norm[lag:]
        gas_part = gas_norm[:-lag]

    else:
        oes_part = oes_norm
        gas_part = gas_norm

    score = np.corrcoef(
        oes_part,
        gas_part
    )[0, 1]

    scores.append(score)

scores = np.asarray(scores)

# 양의 상관 / 음의 상관 모두 고려
best_idx = np.argmax(np.abs(scores))

best_lag_samples = lags[best_idx]
best_lag_sec = best_lag_samples * dt

print("Best lag :", best_lag_sec, "sec")
print("Best correlation :", scores[best_idx])


# ==========================================================
# 6. 확인용 Plot
# ==========================================================

plt.figure(figsize=(12,5))

plt.plot(
    common_time,
    oes_norm,
    label="OES 495-505 nm"
)

plt.plot(
    common_time + best_lag_sec,
    gas_norm,
    label="Gas5 shifted"
)

plt.xlim(25, 100)

plt.xlabel("Relative Time (sec)")
plt.ylabel("Normalized Signal")

plt.title(
    f"OES / Process Alignment (lag={best_lag_sec:.2f}s)"
)

plt.grid()
plt.legend()

plt.tight_layout()

plt.savefig(
    "Wafer_01_time_alignment.png",
    dpi=150
)

plt.show()