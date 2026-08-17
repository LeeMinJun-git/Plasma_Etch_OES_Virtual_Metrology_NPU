from netCDF4 import Dataset
import numpy as np
import pandas as pd
import os

# ==========================================================
# Setting
# ==========================================================

OES_DICT_FILE = "Dictionary_OES.nc"
PROCESS_DICT_FILE = "Dictionary_process.nc"
PROCESS_FILE = "Process_data.nc"

TARGET_CSV = "Si_Oxide_etch_9_points.csv"
USABLE_CSV = "usable_experiments.csv"

OUTPUT_FILE = "full_dataset_raw.npz"
QC_FILE = "full_dataset_qc.csv"

N_CYCLES = 100
N_WAVELENGTH_BINS = 128

GAS5_THRESHOLD = 300

DT = 0.05
MAX_LAG_SEC = 3.0
MIN_CORRELATION = 0.60


# ==========================================================
# 1. Dictionary load
# ==========================================================

print("Loading dictionaries...")

with Dataset(OES_DICT_FILE, "r") as ds:
    oes_decoder = np.asarray(ds["data"][:])

with Dataset(PROCESS_DICT_FILE, "r") as ds:
    process_decoder = np.asarray(ds["data"][:])


# ==========================================================
# 2. Metadata load
# ==========================================================

usable_df = pd.read_csv(USABLE_CSV)
target_df = pd.read_csv(TARGET_CSV)

print("Usable experiments :", len(usable_df))


# ==========================================================
# 3. 100 Gas5 cycles 선택
# ==========================================================

def find_100_cycles(rise_times):

    if len(rise_times) < N_CYCLES:
        return None, None

    best_cycles = None
    best_score = np.inf

    # 연속된 100개 rising edge 후보 중
    # 6초 반복 구조에 가장 잘 맞는 후보 선택
    for start in range(
        len(rise_times) - N_CYCLES + 1
    ):

        candidate = rise_times[
            start:start + N_CYCLES
        ]

        intervals = np.diff(candidate)

        # 첫 interval은 startup 특성이 있을 수 있으므로
        # steady state 중심으로 평가
        if len(intervals) > 1:
            steady = intervals[1:]
        else:
            steady = intervals

        score = np.mean(
            np.abs(steady - 6.0)
        )

        if score < best_score:
            best_score = score
            best_cycles = candidate

    return best_cycles, best_score


# ==========================================================
# 4. Wavelength 3648 -> 128
# ==========================================================

def wavelength_max_pool(spectrum, bins):

    groups = np.array_split(
        np.arange(len(spectrum)),
        bins
    )

    pooled = np.asarray(
        [
            np.max(spectrum[idx])
            for idx in groups
        ],
        dtype=np.float32
    )

    return pooled


# ==========================================================
# 5. OES / Process Offset
# ==========================================================

def calculate_offset(
    oes_time,
    wavelengths,
    encoded_band,
    process_time,
    gas5
):

    decoded_band = oes_decoder[
        encoded_band
    ]

    oes_signal = np.mean(
        decoded_band,
        axis=1
    )

    common_end = min(
        oes_time[-1],
        process_time[-1],
        640.0
    )

    if common_end <= 30:
        return None, None

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

    oes_std = np.std(oes_interp)
    gas_std = np.std(gas_interp)

    if oes_std == 0 or gas_std == 0:
        return None, None

    oes_norm = (
        oes_interp - np.mean(oes_interp)
    ) / oes_std

    gas_norm = (
        gas_interp - np.mean(gas_interp)
    ) / gas_std

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
            oes_part = oes_norm[:lag]
            gas_part = gas_norm[-lag:]

        elif lag > 0:
            oes_part = oes_norm[lag:]
            gas_part = gas_norm[:-lag]

        else:
            oes_part = oes_norm
            gas_part = gas_norm

        corr = np.corrcoef(
            oes_part,
            gas_part
        )[0, 1]

        scores.append(corr)

    scores = np.asarray(scores)

    best_idx = np.argmax(
        np.abs(scores)
    )

    best_lag = (
        lags[best_idx] * DT
    )

    best_corr = scores[best_idx]

    return float(best_lag), float(best_corr)


# ==========================================================
# 6. Containers
# ==========================================================

X_list = []
y_list = []

lot_list = []
wafer_list = []
experiment_list = []
date_list = []

qc_records = []


# ==========================================================
# 7. Process_data.nc 한 번 열기
# ==========================================================

process_ds = Dataset(
    PROCESS_FILE,
    "r"
)


# ==========================================================
# 8. 전체 75 Wafer 처리
# ==========================================================

print()
print("=" * 80)
print("FULL DATASET GENERATION")
print("=" * 80)


for row_idx, row in usable_df.iterrows():

    experiment_key = str(
        row["experiment_key"]
    )

    date = str(
        row["date"]
    )

    lot = int(
        row["lot"]
    )

    wafer_number = int(
        row["wafer"]
    )

    wafer_name = (
        f"Wafer_{wafer_number:02d}"
    )

    nc_date = date.replace(
        "-",
        "_"
    )

    oes_file = (
        f"Day_{nc_date}.nc"
    )

    process_group = (
        f"Day_{nc_date}_{wafer_name}"
    )

    print()
    print(
        f"[{row_idx + 1:02d}/{len(usable_df)}] "
        f"{experiment_key} | Lot {lot}"
    )


    # ------------------------------------------------------
    # QC 초기값
    # ------------------------------------------------------

    qc = {
        "experiment_key": experiment_key,
        "date": date,
        "lot": lot,
        "wafer": wafer_number,
        "status": "FAIL",
        "gas5_edges": np.nan,
        "cycle_score": np.nan,
        "offset": np.nan,
        "correlation": np.nan,
        "input_min": np.nan,
        "input_max": np.nan,
        "target": np.nan,
        "reason": "",
    }


    # ======================================================
    # A. OES 파일 존재 확인
    # ======================================================

    if not os.path.exists(
        oes_file
    ):

        qc["reason"] = (
            "OES file not found"
        )

        print("  FAIL: OES file missing")

        qc_records.append(qc)
        continue


    # ======================================================
    # B. Process group 확인
    # ======================================================

    if process_group not in process_ds.groups:

        qc["reason"] = (
            "Process group not found"
        )

        print("  FAIL: Process group missing")

        qc_records.append(qc)
        continue


    # ======================================================
    # C. Process data
    # ======================================================

    pgrp = process_ds.groups[
        process_group
    ]

    process_time_raw = np.asarray(
        pgrp["times"][:]
    )

    process_time = (
        process_time_raw
        - process_time_raw[0]
    )

    features = [
        f.decode()
        if isinstance(f, bytes)
        else str(f)
        for f in pgrp["feature"][:]
    ]

    encoded_process = np.asarray(
        pgrp["data"][:]
    )

    process_data = process_decoder[
        encoded_process
    ]

    gas5_idx = features.index(
        "Stat3_Etch_MV_Gas5Flow"
    )

    gas5 = process_data[
        :,
        gas5_idx
    ]


    # ======================================================
    # D. Gas5 edges
    # ======================================================

    gas5_high = (
        gas5 > GAS5_THRESHOLD
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

    qc["gas5_edges"] = len(
        rise_times
    )


    # ======================================================
    # E. 100 cycles
    # ======================================================

    cycles, cycle_score = (
        find_100_cycles(
            rise_times
        )
    )

    if cycles is None:

        qc["reason"] = (
            "100 Gas5 cycles not found"
        )

        print(
            "  FAIL: "
            f"only {len(rise_times)} Gas5 edges"
        )

        qc_records.append(qc)
        continue

    qc["cycle_score"] = (
        cycle_score
    )


    # 마지막 Gas5 종료
    candidate_falls = fall_times[
        fall_times > cycles[-1]
    ]

    if len(candidate_falls) == 0:

        qc["reason"] = (
            "Final Gas5 falling edge missing"
        )

        print(
            "  FAIL: final Gas5 fall missing"
        )

        qc_records.append(qc)
        continue

    final_gas5_end = (
        candidate_falls[0]
    )


    # ======================================================
    # F. OES metadata / 500 nm band
    # ======================================================

    with Dataset(
        oes_file,
        "r"
    ) as ods:

        if wafer_name not in ods.groups:

            qc["reason"] = (
                "OES wafer group not found"
            )

            print(
                "  FAIL: OES group missing"
            )

            qc_records.append(qc)
            continue

        ogrp = ods.groups[
            wafer_name
        ]

        oes_time_raw = np.asarray(
            ogrp["times"][:]
        )

        oes_time = (
            oes_time_raw
            - oes_time_raw[0]
        )

        wavelengths = np.asarray(
            ogrp["wavelengths"][:]
        )

        wave_idx = np.where(
            (wavelengths >= 495)
            &
            (wavelengths <= 505)
        )[0]

        encoded_band = np.asarray(
            ogrp["data"][:, wave_idx]
        )


        # ==================================================
        # G. Offset
        # ==================================================

        offset, corr = calculate_offset(
            oes_time,
            wavelengths,
            encoded_band,
            process_time,
            gas5
        )

        if offset is None:

            qc["reason"] = (
                "Offset calculation failed"
            )

            print(
                "  FAIL: offset calculation"
            )

            qc_records.append(qc)
            continue

        qc["offset"] = offset
        qc["correlation"] = corr


        if abs(corr) < MIN_CORRELATION:

            qc["reason"] = (
                "Weak OES/process correlation"
            )

            print(
                f"  CHECK: correlation={corr:.3f}"
            )

            qc_records.append(qc)
            continue


        # ==================================================
        # H. CNN input 생성
        # ==================================================

        oes_as_process_time = (
            oes_time - offset
        )

        gas5_at_oes = np.interp(
            oes_as_process_time,
            process_time,
            gas5
        )

        cnn_input = np.zeros(
            (
                N_CYCLES,
                N_WAVELENGTH_BINS,
                1
            ),
            dtype=np.float32
        )

        input_valid = True


        for cycle_idx in range(
            N_CYCLES
        ):

            start = cycles[
                cycle_idx
            ]

            if cycle_idx < (
                N_CYCLES - 1
            ):

                end = cycles[
                    cycle_idx + 1
                ]

            else:

                end = (
                    final_gas5_end
                )


            cycle_mask = (
                (oes_as_process_time >= start)
                &
                (oes_as_process_time < end)
            )

            phase_mask = (
                cycle_mask
                &
                (
                    gas5_at_oes
                    > GAS5_THRESHOLD
                )
            )

            oes_indices = np.where(
                phase_mask
            )[0]


            if len(oes_indices) == 0:

                print(
                    f"  FAIL: "
                    f"cycle {cycle_idx+1} "
                    f"has no OES samples"
                )

                qc["reason"] = (
                    f"No OES at cycle "
                    f"{cycle_idx+1}"
                )

                input_valid = False
                break


            # ----------------------------------------------
            # 해당 Gas5 phase 데이터만 읽어서 decode
            # ----------------------------------------------

            encoded_phase = np.asarray(
                ogrp["data"][
                    oes_indices,
                    :
                ]
            )

            decoded_phase = oes_decoder[
                encoded_phase
            ]

            spectrum = np.mean(
                decoded_phase,
                axis=0
            )

            cnn_input[
                cycle_idx,
                :,
                0
            ] = wavelength_max_pool(
                spectrum,
                N_WAVELENGTH_BINS
            )


    if not input_valid:

        qc_records.append(qc)
        continue


    # ======================================================
    # I. Input sanity check
    # ======================================================

    if np.isnan(cnn_input).any():

        qc["reason"] = (
            "NaN in CNN input"
        )

        print("  FAIL: NaN input")

        qc_records.append(qc)
        continue


    if np.isinf(cnn_input).any():

        qc["reason"] = (
            "Inf in CNN input"
        )

        print("  FAIL: Inf input")

        qc_records.append(qc)
        continue


    # ======================================================
    # J. Target
    # ======================================================

    target_rows = target_df[
        target_df["experiment_key"]
        == experiment_key
    ]

    if len(target_rows) != 9:

        qc["reason"] = (
            f"Measurement count={len(target_rows)}"
        )

        print(
            "  FAIL: target points != 9"
        )

        qc_records.append(qc)
        continue


    si_etch = target_rows[
        "si_etch"
    ].to_numpy(
        dtype=np.float32
    )

    if np.isnan(si_etch).any():

        qc["reason"] = (
            "NaN target"
        )

        print(
            "  FAIL: NaN target"
        )

        qc_records.append(qc)
        continue


    target = float(
        np.mean(si_etch)
    )


    # ======================================================
    # K. Append
    # ======================================================

    X_list.append(
        cnn_input
    )

    y_list.append(
        target
    )

    lot_list.append(
        lot
    )

    wafer_list.append(
        wafer_number
    )

    experiment_list.append(
        experiment_key
    )

    date_list.append(
        date
    )


    qc["status"] = "PASS"
    qc["reason"] = ""
    qc["input_min"] = float(
        cnn_input.min()
    )
    qc["input_max"] = float(
        cnn_input.max()
    )
    qc["target"] = target

    qc_records.append(qc)


    print(
        f"  PASS | "
        f"cycles=100 | "
        f"offset={offset:+.2f}s | "
        f"corr={corr:.3f} | "
        f"target={target:.4f}"
    )


# ==========================================================
# 9. Close Process file
# ==========================================================

process_ds.close()


# ==========================================================
# 10. QC 저장
# ==========================================================

qc_df = pd.DataFrame(
    qc_records
)

qc_df.to_csv(
    QC_FILE,
    index=False
)


# ==========================================================
# 11. Dataset 생성
# ==========================================================

if len(X_list) == 0:

    raise RuntimeError(
        "No valid samples generated."
    )


X = np.stack(
    X_list,
    axis=0
).astype(
    np.float32
)

y = np.asarray(
    y_list,
    dtype=np.float32
)

lots = np.asarray(
    lot_list,
    dtype=np.int32
)

wafers = np.asarray(
    wafer_list,
    dtype=np.int32
)

experiments = np.asarray(
    experiment_list
)

dates = np.asarray(
    date_list
)


# ==========================================================
# 12. Save NPZ
# ==========================================================

np.savez_compressed(
    OUTPUT_FILE,
    X=X,
    y=y,
    lot=lots,
    wafer=wafers,
    experiment_key=experiments,
    date=dates,
)


# ==========================================================
# 13. Final summary
# ==========================================================

print()
print("=" * 80)
print("FINAL DATASET SUMMARY")
print("=" * 80)

print(
    "Generated samples :",
    len(X),
    "/",
    len(usable_df)
)

print(
    "Failed samples    :",
    len(usable_df) - len(X)
)

print(
    "X shape           :",
    X.shape
)

print(
    "y shape           :",
    y.shape
)

print(
    "Lots              :",
    np.unique(lots)
)

print(
    "Target min/max    :",
    y.min(),
    "/",
    y.max()
)

print(
    "Target mean/std   :",
    y.mean(),
    "/",
    y.std()
)

print()
print(
    "Saved dataset :",
    OUTPUT_FILE
)

print(
    "Saved QC      :",
    QC_FILE
)


# 실패 항목 출력
failed = qc_df[
    qc_df["status"] != "PASS"
]

if len(failed) > 0:

    print()
    print("=" * 80)
    print("FAILED / CHECK SAMPLES")
    print("=" * 80)

    print(
        failed[
            [
                "experiment_key",
                "gas5_edges",
                "offset",
                "correlation",
                "reason"
            ]
        ].to_string(
            index=False
        )
    )