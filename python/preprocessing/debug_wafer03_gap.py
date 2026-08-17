from netCDF4 import Dataset
import numpy as np

PROCESS_FILE = "Process_data.nc"
PROCESS_DICT = "Dictionary_process.nc"
WAFER = "Day_2024_08_05_Wafer_03"


# ==========================================================
# 1. Dictionary
# ==========================================================

with Dataset(PROCESS_DICT, "r") as ds:
    decoder = np.asarray(ds["data"][:])


# ==========================================================
# 2. Process data
# ==========================================================

with Dataset(PROCESS_FILE, "r") as ds:

    grp = ds.groups[WAFER]

    raw_time = np.asarray(grp["times"][:])
    process_time = raw_time - raw_time[0]

    features = [
        f.decode() if isinstance(f, bytes) else str(f)
        for f in grp["feature"][:]
    ]

    encoded = np.asarray(
        grp["data"][:],
        dtype=np.int64
    )

    data = decoder[encoded]


# ==========================================================
# 3. Gas4 / Gas5
# ==========================================================

gas4_idx = features.index(
    "Stat3_Etch_MV_Gas4Flow"
)

gas5_idx = features.index(
    "Stat3_Etch_MV_Gas5Flow"
)

gas4 = data[:, gas4_idx]
gas5 = data[:, gas5_idx]


# ==========================================================
# 4. Edge detection
# ==========================================================

gas5_high = gas5 > 300
gas4_high = gas4 > 150

gas5_rising_idx = np.where(
    (~gas5_high[:-1]) &
    (gas5_high[1:])
)[0] + 1

gas5_falling_idx = np.where(
    (gas5_high[:-1]) &
    (~gas5_high[1:])
)[0] + 1

gas4_rising_idx = np.where(
    (~gas4_high[:-1]) &
    (gas4_high[1:])
)[0] + 1

gas4_falling_idx = np.where(
    (gas4_high[:-1]) &
    (~gas4_high[1:])
)[0] + 1


gas5_rise = process_time[gas5_rising_idx]
gas5_fall = process_time[gas5_falling_idx]

gas4_rise = process_time[gas4_rising_idx]
gas4_fall = process_time[gas4_falling_idx]


# ==========================================================
# 5. 사전 pulse 제외 → 본 공정 100 cycles
# ==========================================================

cycles = gas5_rise[1:]

print("Selected cycles :", len(cycles))

intervals = np.diff(cycles)


# ==========================================================
# 6. 비정상 interval 찾기
# ==========================================================

abnormal_idx = np.where(
    (intervals < 4.0) |
    (intervals > 7.0)
)[0]

print("\n====================================")
print("ABNORMAL INTERVALS")
print("====================================")

if len(abnormal_idx) == 0:

    print("No abnormal intervals.")

else:

    for idx in abnormal_idx:

        print(
            f"\nCycle {idx+1} -> {idx+2}"
        )

        print(
            f"Gas5 start : "
            f"{cycles[idx]:.2f} "
            f"-> {cycles[idx+1]:.2f}"
        )

        print(
            f"Interval   : "
            f"{intervals[idx]:.2f} sec"
        )

        # 해당 구간 주변 ±3초 확장
        t_start = cycles[idx] - 3
        t_end = cycles[idx + 1] + 3

        print(
            f"Debug range: "
            f"{t_start:.2f} ~ {t_end:.2f} sec"
        )

        # 해당 범위의 Gas5 edges
        local_gas5_rise = gas5_rise[
            (gas5_rise >= t_start) &
            (gas5_rise <= t_end)
        ]

        local_gas5_fall = gas5_fall[
            (gas5_fall >= t_start) &
            (gas5_fall <= t_end)
        ]

        local_gas4_rise = gas4_rise[
            (gas4_rise >= t_start) &
            (gas4_rise <= t_end)
        ]

        local_gas4_fall = gas4_fall[
            (gas4_fall >= t_start) &
            (gas4_fall <= t_end)
        ]

        print(
            "Gas5 rising :",
            local_gas5_rise
        )

        print(
            "Gas5 falling:",
            local_gas5_fall
        )

        print(
            "Gas4 rising :",
            local_gas4_rise
        )

        print(
            "Gas4 falling:",
            local_gas4_fall
        )