from netCDF4 import Dataset
import numpy as np

# ==========================================================
# Setting
# ==========================================================

OES_FILE = "Day_2024_08_05.nc"
OES_DICT = "Dictionary_OES.nc"

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

DT = 0.05
MAX_LAG_SEC = 3.0


# ==========================================================
# Dictionary load
# ==========================================================

with Dataset(OES_DICT, "r") as ds:
    oes_decoder = np.asarray(ds["data"][:])

with Dataset(PROCESS_DICT, "r") as ds:
    process_decoder = np.asarray(ds["data"][:])


# ==========================================================
# Wafer별 Offset 계산
# ==========================================================

print("=" * 72)
print("OES / PROCESS TIME OFFSET VALIDATION")
print("=" * 72)


for wafer in WAFERS:

    print()
    print("-" * 72)
    print(wafer)
    print("-" * 72)

    # ======================================================
    # 1. OES
    # ======================================================

    with Dataset(OES_FILE, "r") as ds:

        if wafer not in ds.groups:
            print("RESULT : FAIL - OES group not found")
            continue

        grp = ds.groups[wafer]

        oes_time_raw = np.asarray(
            grp["times"][:]
        )

        oes_time = (
            oes_time_raw
            - oes_time_raw[0]
        )

        wavelengths = np.asarray(
            grp["wavelengths"][:]
        )

        # 495 ~ 505 nm 영역
        wave_idx = np.where(
            (wavelengths >= 495)
            &
            (wavelengths <= 505)
        )[0]

        encoded = np.asarray(
            grp["data"][:, wave_idx],
            dtype=np.int64
        )

        decoded = oes_decoder[
            encoded
        ]

        oes_signal = np.mean(
            decoded,
            axis=1
        )


    # ======================================================
    # 2. Process
    # ======================================================

    process_group = (
        f"Day_{DATE}_{wafer}"
    )

    with Dataset(PROCESS_FILE, "r") as ds:

        if process_group not in ds.groups:
            print(
                "RESULT : FAIL - "
                "Process group not found"
            )
            continue

        grp = ds.groups[
            process_group
        ]

        process_time_raw = np.asarray(
            grp["times"][:]
        )

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

        process_data = (
            process_decoder[
                encoded
            ]
        )


    gas5_idx = features.index(
        "Stat3_Etch_MV_Gas5Flow"
    )

    gas5 = process_data[
        :,
        gas5_idx
    ]


    # ======================================================
    # 3. 공통 시간축
    # ======================================================

    common_start = 25.0

    common_end = min(
        oes_time[-1],
        process_time[-1],
        640.0
    )

    common_time = np.arange(
        common_start,
        common_end,
        DT
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


    # ======================================================
    # 4. Normalize
    # ======================================================

    oes_std = np.std(
        oes_interp
    )

    gas_std = np.std(
        gas_interp
    )

    if oes_std == 0 or gas_std == 0:

        print(
            "RESULT : FAIL - "
            "zero standard deviation"
        )

        continue


    oes_norm = (
        oes_interp
        - np.mean(oes_interp)
    ) / oes_std


    gas_norm = (
        gas_interp
        - np.mean(gas_interp)
    ) / gas_std


    # ======================================================
    # 5. ±3초 범위 Offset 탐색
    # ======================================================

    max_lag_samples = int(
        MAX_LAG_SEC / DT
    )

    lags = np.arange(
        -max_lag_samples,
        max_lag_samples + 1
    )

    scores = []


    for lag in lags:

        if lag < 0:

            oes_part = (
                oes_norm[:lag]
            )

            gas_part = (
                gas_norm[-lag:]
            )

        elif lag > 0:

            oes_part = (
                oes_norm[lag:]
            )

            gas_part = (
                gas_norm[:-lag]
            )

        else:

            oes_part = oes_norm
            gas_part = gas_norm


        correlation = np.corrcoef(
            oes_part,
            gas_part
        )[0, 1]

        scores.append(
            correlation
        )


    scores = np.asarray(
        scores
    )

    best_idx = np.argmax(
        np.abs(scores)
    )

    best_lag_samples = (
        lags[best_idx]
    )

    best_lag_sec = (
        best_lag_samples
        * DT
    )

    best_corr = (
        scores[best_idx]
    )


    # ======================================================
    # 6. 결과
    # ======================================================

    print(
        "OES duration       :",
        oes_time[-1]
    )

    print(
        "Process duration   :",
        process_time[-1]
    )

    print(
        "Best lag           :",
        best_lag_sec,
        "sec"
    )

    print(
        "Best correlation   :",
        best_corr
    )

    if abs(best_corr) >= 0.6:

        print(
            "RESULT             : PASS"
        )

    else:

        print(
            "RESULT             : CHECK"
        )