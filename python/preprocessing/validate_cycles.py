from netCDF4 import Dataset
import numpy as np

# ==========================================================
# Setting
# ==========================================================

PROCESS_FILE = "Process_data.nc"
PROCESS_DICT = "Dictionary_process.nc"

DATE = "2024_08_05"

WAFERS = [
    "Wafer_01",
    "Wafer_02",
    "Wafer_03",
    "Wafer_04",
    "Wafer_05",
]


# ==========================================================
# 1. Process dictionary
# ==========================================================

with Dataset(PROCESS_DICT, "r") as ds:
    decoder = np.asarray(ds["data"][:])


# ==========================================================
# 2. 100-cycle 후보 자동 검출 함수
# ==========================================================

def find_100_cycles(rise_times):

    if len(rise_times) < 100:
        return None, None

    # Rising edge가 정확히 100개라면 그대로 사용
    if len(rise_times) == 100:

        cycles = rise_times.copy()
        intervals = np.diff(cycles)

        score = np.mean(
            np.abs(intervals - 6.0)
        )

        return cycles, score

    # Rising edge가 100개보다 많다면
    # 연속된 100개 후보 중
    # 6초 주기에 가장 가까운 구간 선택
    best_cycles = None
    best_score = np.inf

    for start in range(
        len(rise_times) - 100 + 1
    ):

        candidate = rise_times[
            start:start + 100
        ]

        intervals = np.diff(
            candidate
        )

        score = np.mean(
            np.abs(
                intervals - 6.0
            )
        )

        if score < best_score:

            best_score = score
            best_cycles = candidate

    return best_cycles, best_score


# ==========================================================
# 3. Wafer별 검사
# ==========================================================

print("=" * 72)
print("BOSCH CYCLE VALIDATION")
print("=" * 72)


with Dataset(PROCESS_FILE, "r") as ds:

    for wafer in WAFERS:

        group_name = (
            f"Day_{DATE}_{wafer}"
        )

        print()
        print("-" * 72)
        print(group_name)
        print("-" * 72)

        # --------------------------------------------------
        # Group 존재 여부
        # --------------------------------------------------

        if group_name not in ds.groups:

            print(
                "RESULT : FAIL "
                "(Process group not found)"
            )

            continue

        grp = ds.groups[
            group_name
        ]


        # --------------------------------------------------
        # Time
        # --------------------------------------------------

        raw_time = np.asarray(
            grp["times"][:]
        )

        process_time = (
            raw_time
            - raw_time[0]
        )


        # --------------------------------------------------
        # Feature names
        # --------------------------------------------------

        features = [
            f.decode()
            if isinstance(f, bytes)
            else str(f)
            for f in grp["feature"][:]
        ]


        # --------------------------------------------------
        # Process data decode
        # --------------------------------------------------

        encoded = np.asarray(
            grp["data"][:],
            dtype=np.int64
        )

        data = decoder[
            encoded
        ]


        # --------------------------------------------------
        # Gas4 / Gas5
        # --------------------------------------------------

        gas4_idx = features.index(
            "Stat3_Etch_MV_Gas4Flow"
        )

        gas5_idx = features.index(
            "Stat3_Etch_MV_Gas5Flow"
        )

        gas4 = data[
            :,
            gas4_idx
        ]

        gas5 = data[
            :,
            gas5_idx
        ]


        # --------------------------------------------------
        # HIGH 판정
        # --------------------------------------------------

        gas5_high = (
            gas5 > 300
        )

        gas4_high = (
            gas4 > 150
        )


        # --------------------------------------------------
        # Gas5 Rising / Falling
        # --------------------------------------------------

        gas5_rising_idx = np.where(
            (~gas5_high[:-1])
            &
            (gas5_high[1:])
        )[0] + 1

        gas5_falling_idx = np.where(
            (gas5_high[:-1])
            &
            (~gas5_high[1:])
        )[0] + 1


        gas5_rise_times = process_time[
            gas5_rising_idx
        ]

        gas5_fall_times = process_time[
            gas5_falling_idx
        ]


        # --------------------------------------------------
        # Gas4 Rising / Falling
        # --------------------------------------------------

        gas4_rising_idx = np.where(
            (~gas4_high[:-1])
            &
            (gas4_high[1:])
        )[0] + 1

        gas4_falling_idx = np.where(
            (gas4_high[:-1])
            &
            (~gas4_high[1:])
        )[0] + 1


        gas4_rise_times = process_time[
            gas4_rising_idx
        ]

        gas4_fall_times = process_time[
            gas4_falling_idx
        ]


        # --------------------------------------------------
        # Edge count 출력
        # --------------------------------------------------

        print(
            "Gas5 rising/falling :",
            len(gas5_rise_times),
            "/",
            len(gas5_fall_times)
        )

        print(
            "Gas4 rising/falling :",
            len(gas4_rise_times),
            "/",
            len(gas4_fall_times)
        )


        # --------------------------------------------------
        # 100개의 Gas5 Etch cycle 선택
        # --------------------------------------------------

        cycles, score = find_100_cycles(
            gas5_rise_times
        )

        if cycles is None:

            print(
                "RESULT : FAIL "
                "(Gas5 pulse < 100)"
            )

            continue


        intervals = np.diff(
            cycles
        )


        # --------------------------------------------------
        # 결과 출력
        # --------------------------------------------------

        print(
            "Selected cycles     :",
            len(cycles)
        )

        print(
            "First cycle start   :",
            cycles[0]
        )

        print(
            "Last cycle start    :",
            cycles[-1]
        )

        print(
            "Mean interval       :",
            np.mean(intervals)
        )

        print(
            "Min/Max interval    :",
            np.min(intervals),
            "/",
            np.max(intervals)
        )

        print(
            "6-sec deviation     :",
            score
        )


        # --------------------------------------------------
        # Startup interval
        # --------------------------------------------------

        startup_interval = (
            intervals[0]
        )

        # Cycle 2 이후의 안정적인 반복 구간
        steady_intervals = (
            intervals[1:]
        )


        # --------------------------------------------------
        # 비정상 steady interval 탐색
        # --------------------------------------------------

        abnormal_steady = np.where(
            (steady_intervals < 5.0)
            |
            (steady_intervals > 7.0)
        )[0]


        # --------------------------------------------------
        # 최종 판정
        # --------------------------------------------------

        if (
            len(cycles) == 100
            and len(abnormal_steady) == 0
        ):

            print(
                "RESULT : PASS"
            )

            # 첫 cycle은 startup 특성 때문에
            # 6초에서 벗어날 수 있으므로
            # FAIL로 처리하지 않고 NOTE만 출력
            if not (
                4.0
                <= startup_interval
                <= 7.0
            ):

                print(
                    "NOTE   : "
                    "Startup interval anomaly =",
                    startup_interval,
                    "sec"
                )

        else:

            print(
                "RESULT : CHECK"
            )

            # 이상 steady cycle이 있다면
            # 위치까지 출력
            if len(
                abnormal_steady
            ) > 0:

                print(
                    "Abnormal steady intervals:"
                )

                for idx in abnormal_steady:

                    # steady_intervals[0]
                    # = cycles[1] -> cycles[2]
                    cycle_idx = idx + 1

                    print(
                        f"  Cycle "
                        f"{cycle_idx + 1}"
                        f" -> "
                        f"{cycle_idx + 2}"
                        f" : "
                        f"{intervals[cycle_idx]:.2f}"
                        f" sec"
                    )