from netCDF4 import Dataset
import numpy as np

# ==========================================================
# Setting
# ==========================================================

PROCESS_FILE = "Process_data.nc"
PROCESS_DICT = "Dictionary_process.nc"

PROCESS_WAFER = "Day_2024_08_05_Wafer_01"


# ==========================================================
# 1. Process dictionary load
# ==========================================================

with Dataset(PROCESS_DICT, "r") as ds:
    process_decoder = np.asarray(ds["data"][:])


# ==========================================================
# 2. Process data load
# ==========================================================

with Dataset(PROCESS_FILE, "r") as ds:

    grp = ds.groups[PROCESS_WAFER]

    process_time_raw = np.asarray(
        grp["times"][:]
    )

    # 시작점을 0초로 변환
    process_time = (
        process_time_raw
        - process_time_raw[0]
    )

    features = [
        f.decode()
        if isinstance(f, bytes)
        else str(f)
        for f in grp["feature"][:]
    ]

    encoded = np.asarray(
        grp["data"][:],
        dtype=np.int64
    )

    process_data = process_decoder[
        encoded
    ]


# ==========================================================
# 3. Gas4 / Gas5 추출
# ==========================================================

gas4_idx = features.index(
    "Stat3_Etch_MV_Gas4Flow"
)

gas5_idx = features.index(
    "Stat3_Etch_MV_Gas5Flow"
)

gas4 = process_data[:, gas4_idx]
gas5 = process_data[:, gas5_idx]


# ==========================================================
# 4. HIGH / LOW 판정
# ==========================================================

# 앞에서 사용했던 threshold 유지
gas5_high = gas5 > 300
gas4_high = gas4 > 150


# ==========================================================
# 5. Gas5 Rising / Falling Edge
# ==========================================================

gas5_rising_idx = np.where(
    (~gas5_high[:-1])
    & (gas5_high[1:])
)[0] + 1

gas5_falling_idx = np.where(
    (gas5_high[:-1])
    & (~gas5_high[1:])
)[0] + 1


gas5_rise_times = process_time[
    gas5_rising_idx
]

gas5_fall_times = process_time[
    gas5_falling_idx
]


# ==========================================================
# 6. Gas4 Rising / Falling Edge
# ==========================================================

gas4_rising_idx = np.where(
    (~gas4_high[:-1])
    & (gas4_high[1:])
)[0] + 1

gas4_falling_idx = np.where(
    (gas4_high[:-1])
    & (~gas4_high[1:])
)[0] + 1


gas4_rise_times = process_time[
    gas4_rising_idx
]

gas4_fall_times = process_time[
    gas4_falling_idx
]


# ==========================================================
# 7. 기본 Count 출력
# ==========================================================

print(
    "===================================="
)
print(
    "        GAS EDGE DEBUG"
)
print(
    "===================================="
)

print(
    "\nGas5 rising count :",
    len(gas5_rise_times)
)

print(
    "Gas5 falling count:",
    len(gas5_fall_times)
)

print(
    "Gas4 rising count :",
    len(gas4_rise_times)
)

print(
    "Gas4 falling count:",
    len(gas4_fall_times)
)


# ==========================================================
# 8. 전체 초기 Edge 확인
# ==========================================================

print(
    "\n===================================="
)
print(
    "FIRST 10 EDGES"
)
print(
    "===================================="
)

print(
    "\nFirst 10 Gas5 rising:"
)
print(
    gas5_rise_times[:10]
)

print(
    "\nFirst 10 Gas5 falling:"
)
print(
    gas5_fall_times[:10]
)

print(
    "\nFirst 10 Gas4 rising:"
)
print(
    gas4_rise_times[:10]
)

print(
    "\nFirst 10 Gas4 falling:"
)
print(
    gas4_fall_times[:10]
)


# ==========================================================
# 9. 마지막 Edge 확인
# ==========================================================

print(
    "\n===================================="
)
print(
    "LAST 10 EDGES"
)
print(
    "===================================="
)

print(
    "\nLast 10 Gas5 rising:"
)
print(
    gas5_rise_times[-10:]
)

print(
    "\nLast 10 Gas5 falling:"
)
print(
    gas5_fall_times[-10:]
)

print(
    "\nLast 10 Gas4 rising:"
)
print(
    gas4_rise_times[-10:]
)

print(
    "\nLast 10 Gas4 falling:"
)
print(
    gas4_fall_times[-10:]
)


# ==========================================================
# 10. Gas5 Rising Interval 확인
# ==========================================================

gas5_intervals = np.diff(
    gas5_rise_times
)

print(
    "\n===================================="
)
print(
    "GAS5 RISING INTERVAL"
)
print(
    "===================================="
)

print(
    "\nFirst 15 intervals:"
)
print(
    gas5_intervals[:15]
)

print(
    "\nLast 15 intervals:"
)
print(
    gas5_intervals[-15:]
)


# ==========================================================
# 11. 기존 BOSCH 후보 확인
# ==========================================================

# 앞에서 사용했던 가정:
# 첫 Gas5 rising edge 18 sec는 사전 pulse
cycle_starts = gas5_rise_times[1:]

print(
    "\n===================================="
)
print(
    "CURRENT BOSCH CANDIDATE"
)
print(
    "===================================="
)

print(
    "\nCandidate cycle count:",
    len(cycle_starts)
)

if len(cycle_starts) > 0:

    print(
        "Candidate first cycle:",
        cycle_starts[0]
    )

    print(
        "Candidate last cycle :",
        cycle_starts[-1]
    )


# ==========================================================
# 12. 마지막 Gas5 이후 Gas4 존재 여부 확인
# ==========================================================

if len(cycle_starts) > 0:

    last_cycle_start = (
        cycle_starts[-1]
    )

    candidate_gas5_falls = (
        gas5_fall_times[
            gas5_fall_times
            > last_cycle_start
        ]
    )

    print(
        "\n===================================="
    )
    print(
        "FINAL CYCLE DEBUG"
    )
    print(
        "===================================="
    )

    print(
        "\nLast candidate Gas5 start:",
        last_cycle_start
    )

    if (
        len(candidate_gas5_falls)
        > 0
    ):

        last_gas5_end = (
            candidate_gas5_falls[0]
        )

        print(
            "Last Gas5 end:",
            last_gas5_end
        )

        gas4_after_last_gas5_rise = (
            gas4_rise_times[
                gas4_rise_times
                > last_gas5_end
            ]
        )

        gas4_after_last_gas5_fall = (
            gas4_fall_times[
                gas4_fall_times
                > last_gas5_end
            ]
        )

        print(
            "\nGas4 rising edges "
            "after last Gas5 end:"
        )
        print(
            gas4_after_last_gas5_rise
        )

        print(
            "\nGas4 falling edges "
            "after last Gas5 end:"
        )
        print(
            gas4_after_last_gas5_fall
        )

    else:

        print(
            "No Gas5 falling edge "
            "after last cycle start."
        )


# ==========================================================
# 13. 마지막 실제 값 확인
# ==========================================================

print(
    "\n===================================="
)
print(
    "FINAL PROCESS VALUES"
)
print(
    "===================================="
)

print(
    "\nProcess final time:",
    process_time[-1]
)

print(
    "Final Gas4 value:",
    gas4[-1]
)

print(
    "Final Gas5 value:",
    gas5[-1]
)

print(
    "\nDEBUG COMPLETE"
)