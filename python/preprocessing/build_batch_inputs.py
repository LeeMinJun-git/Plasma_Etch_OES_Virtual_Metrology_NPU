from netCDF4 import Dataset
import numpy as np
import os

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

N_WAVELENGTH_BINS = 128

DT = 0.05
MAX_LAG_SEC = 3.0

OUTPUT_DIR = "cnn_inputs_2024_08_05"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ==========================================================
# Dictionary load
# ==========================================================

with Dataset(OES_DICT, "r") as ds:
    oes_decoder = np.asarray(
        ds["data"][:]
    )

with Dataset(PROCESS_DICT, "r") as ds:
    process_decoder = np.asarray(
        ds["data"][:]
    )


# ==========================================================
# 100개의 Gas5 cycle 자동 선택
# ==========================================================

def find_100_cycles(rise_times):

    if len(rise_times) < 100:
        return None

    if len(rise_times) == 100:
        return rise_times.copy()

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

    return best_cycles


# ==========================================================
# Wavelength pooling
# ==========================================================

def wavelength_max_pool(
    spectrum,
    bins=128
):

    indices = np.array_split(
        np.arange(
            len(spectrum)
        ),
        bins
    )

    pooled = np.array(
        [
            np.max(
                spectrum[idx]
            )
            for idx in indices
        ],
        dtype=np.float32
    )

    return pooled


# ==========================================================
# OES / Process offset 계산
# ==========================================================

def calculate_offset(
    oes_time,
    wavelengths,
    encoded_oes,
    process_time,
    gas5
):

    # ------------------------------------------------------
    # 495~505 nm OES
    # ------------------------------------------------------

    wave_idx = np.where(
        (wavelengths >= 495)
        &
        (wavelengths <= 505)
    )[0]

    encoded_band = encoded_oes[
        :,
        wave_idx
    ]

    decoded_band = oes_decoder[
        encoded_band
    ]

    oes_signal = np.mean(
        decoded_band,
        axis=1
    )


    # ------------------------------------------------------
    # 공통 시간축
    # ------------------------------------------------------

    common_end = min(
        oes_time[-1],
        process_time[-1],
        640.0
    )

    common_time = np.arange(
        25.0,
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


    # ------------------------------------------------------
    # Normalize
    # ------------------------------------------------------

    oes_norm = (
        oes_interp
        - np.mean(oes_interp)
    ) / np.std(oes_interp)

    gas_norm = (
        gas_interp
        - np.mean(gas_interp)
    ) / np.std(gas_interp)


    # ------------------------------------------------------
    # ±3초 범위 correlation
    # ------------------------------------------------------

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


        corr = np.corrcoef(
            oes_part,
            gas_part
        )[0, 1]

        scores.append(
            corr
        )


    scores = np.asarray(
        scores
    )

    best_idx = np.argmax(
        np.abs(scores)
    )

    best_lag = (
        lags[best_idx]
        * DT
    )

    best_corr = (
        scores[best_idx]
    )

    return best_lag, best_corr


# ==========================================================
# Wafer별 처리
# ==========================================================

print("=" * 72)
print("BATCH CNN INPUT GENERATION")
print("=" * 72)


summary = []


for wafer in WAFERS:

    print()
    print("-" * 72)
    print(wafer)
    print("-" * 72)


    # ======================================================
    # 1. Process load
    # ======================================================

    process_group = (
        f"Day_{DATE}_{wafer}"
    )

    with Dataset(
        PROCESS_FILE,
        "r"
    ) as ds:

        if process_group not in ds.groups:

            print(
                "FAIL: Process group not found"
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

        encoded_process = np.asarray(
            grp["data"][:]
        )

        process_data = (
            process_decoder[
                encoded_process
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
    # 2. Gas5 edge 검출
    # ======================================================

    gas5_high = (
        gas5 > 300
    )

    rising_idx = np.where(
        (~gas5_high[:-1])
        &
        (gas5_high[1:])
    )[0] + 1


    falling_idx = np.where(
        (gas5_high[:-1])
        &
        (~gas5_high[1:])
    )[0] + 1


    rise_times = process_time[
        rising_idx
    ]

    fall_times = process_time[
        falling_idx
    ]


    cycles = find_100_cycles(
        rise_times
    )


    if cycles is None:

        print(
            "FAIL: 100 cycles not found"
        )

        continue


    if len(cycles) != 100:

        print(
            "FAIL: Invalid cycle count"
        )

        continue


    print(
        "Gas5 cycles :",
        len(cycles)
    )


    # ======================================================
    # 3. 마지막 Gas5 phase 종료점
    # ======================================================

    last_cycle_start = (
        cycles[-1]
    )

    candidate_falls = fall_times[
        fall_times
        > last_cycle_start
    ]

    if len(candidate_falls) == 0:

        print(
            "FAIL: Final Gas5 fall not found"
        )

        continue


    last_cycle_end = (
        candidate_falls[0]
    )


    # ======================================================
    # 4. OES load
    # ======================================================

    with Dataset(
        OES_FILE,
        "r"
    ) as ds:

        if wafer not in ds.groups:

            print(
                "FAIL: OES group not found"
            )

            continue

        grp = ds.groups[
            wafer
        ]

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

        # 원래 dtype 유지
        encoded_oes = np.asarray(
            grp["data"][:]
        )


    # ======================================================
    # 5. Offset 계산
    # ======================================================

    offset, correlation = (
        calculate_offset(
            oes_time,
            wavelengths,
            encoded_oes,
            process_time,
            gas5
        )
    )


    print(
        "Offset      :",
        offset,
        "sec"
    )

    print(
        "Correlation :",
        correlation
    )


    if abs(correlation) < 0.6:

        print(
            "FAIL: weak OES/Process correlation"
        )

        continue


    # ======================================================
    # 6. OES → Process 시간축
    # ======================================================

    oes_as_process_time = (
        oes_time - offset
    )


    gas5_at_oes = np.interp(
        oes_as_process_time,
        process_time,
        gas5
    )


    # ======================================================
    # 7. 전체 OES decode
    # ======================================================

    oes_data = oes_decoder[
        encoded_oes
    ]


    # ======================================================
    # 8. CNN input
    # ======================================================

    cnn_input = np.zeros(
        (
            100,
            N_WAVELENGTH_BINS,
            1
        ),
        dtype=np.float32
    )


    valid = True


    for cycle in range(100):

        start = cycles[
            cycle
        ]


        if cycle < 99:

            end = cycles[
                cycle + 1
            ]

        else:

            # 마지막 cycle은
            # 실제 Gas5 falling edge까지만
            end = last_cycle_end


        cycle_mask = (
            (oes_as_process_time >= start)
            &
            (oes_as_process_time < end)
        )


        gas5_mask = (
            cycle_mask
            &
            (gas5_at_oes > 300)
        )


        sample_count = np.sum(
            gas5_mask
        )


        if sample_count == 0:

            print(
                f"FAIL: Cycle "
                f"{cycle+1} "
                f"has no OES samples"
            )

            valid = False
            break


        # Gas5 phase의 평균 spectrum
        spectrum = np.mean(
            oes_data[
                gas5_mask,
                :
            ],
            axis=0
        )


        # 3648 → 128
        pooled = wavelength_max_pool(
            spectrum,
            N_WAVELENGTH_BINS
        )


        cnn_input[
            cycle,
            :,
            0
        ] = pooled


    if not valid:
        continue


    # ======================================================
    # 9. 검증
    # ======================================================

    if np.isnan(
        cnn_input
    ).any():

        print(
            "FAIL: NaN detected"
        )

        continue


    if np.isinf(
        cnn_input
    ).any():

        print(
            "FAIL: Inf detected"
        )

        continue


    # ======================================================
    # 10. 저장
    # ======================================================

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{DATE}_{wafer}_100x128x1.npy"
    )


    np.save(
        output_file,
        cnn_input
    )


    print(
        "Shape       :",
        cnn_input.shape
    )

    print(
        "Min/Max     :",
        cnn_input.min(),
        "/",
        cnn_input.max()
    )

    print(
        "Saved       :",
        output_file
    )

    print(
        "RESULT      : PASS"
    )


    summary.append(
        (
            wafer,
            offset,
            correlation,
            cnn_input.min(),
            cnn_input.max()
        )
    )


# ==========================================================
# Summary
# ==========================================================

print()
print("=" * 72)
print("SUMMARY")
print("=" * 72)


for item in summary:

    wafer = item[0]
    offset = item[1]
    corr = item[2]
    min_value = item[3]
    max_value = item[4]

    print(
        f"{wafer} | "
        f"offset={offset:+.2f}s | "
        f"corr={corr:.3f} | "
        f"min={min_value:.2f} | "
        f"max={max_value:.2f}"
    )


print()
print(
    "Generated samples:",
    len(summary),
    "/",
    len(WAFERS)
)