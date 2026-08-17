import pandas as pd
import numpy as np

CSV_FILE = "Si_Oxide_etch_9_points.csv"

# ==========================================================
# 1. CSV Load
# ==========================================================

df = pd.read_csv(CSV_FILE)

print("=" * 80)
print("FULL DATASET INSPECTION")
print("=" * 80)

print("\nTotal rows :", len(df))
print(
    "Unique experiment keys :",
    df["experiment_key"].nunique()
)


# ==========================================================
# 2. Experiment별 상태 확인
# ==========================================================

records = []

for experiment_key, group in df.groupby(
    "experiment_key"
):

    point_count = len(group)

    valid_si_count = (
        group["si_etch"]
        .notna()
        .sum()
    )

    lot_values = (
        group["lot_number"]
        .dropna()
        .unique()
    )

    wafer_values = (
        group["wafer_number"]
        .dropna()
        .unique()
    )

    lot = (
        int(lot_values[0])
        if len(lot_values) == 1
        else None
    )

    wafer = (
        int(wafer_values[0])
        if len(wafer_values) == 1
        else None
    )

    # experiment_key:
    # 2024-08-05_01
    date = experiment_key[:10]

    # 9개 측정점 + si_etch 모두 존재
    usable = (
        point_count == 9
        and valid_si_count == 9
        and lot is not None
        and wafer is not None
    )

    mean_si_etch = (
        group["si_etch"].mean()
        if usable
        else np.nan
    )

    records.append(
        {
            "experiment_key": experiment_key,
            "date": date,
            "lot": lot,
            "wafer": wafer,
            "points": point_count,
            "valid_si": valid_si_count,
            "mean_si_etch": mean_si_etch,
            "usable": usable,
        }
    )


summary = pd.DataFrame(records)


# ==========================================================
# 3. 전체 Summary
# ==========================================================

usable = summary[
    summary["usable"]
].copy()

invalid = summary[
    ~summary["usable"]
].copy()


print()
print("=" * 80)
print("USABLE SAMPLE SUMMARY")
print("=" * 80)

print(
    "\nUsable wafers :",
    len(usable)
)

print(
    "Invalid wafers:",
    len(invalid)
)

print(
    "Unique usable dates:",
    usable["date"].nunique()
)

print(
    "Unique usable lots:",
    usable["lot"].nunique()
)


# ==========================================================
# 4. Lot별 샘플 수
# ==========================================================

print()
print("=" * 80)
print("SAMPLES BY LOT")
print("=" * 80)

lot_summary = (
    usable.groupby("lot")
    .agg(
        samples=("experiment_key", "count"),
        dates=("date", "nunique"),
        target_mean=("mean_si_etch", "mean"),
        target_min=("mean_si_etch", "min"),
        target_max=("mean_si_etch", "max"),
    )
)

print(
    lot_summary.to_string()
)


# ==========================================================
# 5. Date별 샘플 수
# ==========================================================

print()
print("=" * 80)
print("SAMPLES BY DATE")
print("=" * 80)

date_summary = (
    usable.groupby("date")
    .agg(
        lot=("lot", "first"),
        samples=("experiment_key", "count"),
        target_mean=("mean_si_etch", "mean"),
    )
)

print(
    date_summary.to_string()
)


# ==========================================================
# 6. 필요한 OES 파일 목록
# ==========================================================

print()
print("=" * 80)
print("REQUIRED OES FILES")
print("=" * 80)

dates = sorted(
    usable["date"].unique()
)

for date in dates:

    nc_date = date.replace("-", "_")

    count = (
        usable["date"]
        .eq(date)
        .sum()
    )

    print(
        f"Day_{nc_date}.nc"
        f"  ({count} usable wafers)"
    )


# ==========================================================
# 7. 전체 Target 범위
# ==========================================================

print()
print("=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

print(
    "Mean Si Etch min  :",
    usable["mean_si_etch"].min()
)

print(
    "Mean Si Etch max  :",
    usable["mean_si_etch"].max()
)

print(
    "Mean Si Etch mean :",
    usable["mean_si_etch"].mean()
)

print(
    "Mean Si Etch std  :",
    usable["mean_si_etch"].std()
)


# ==========================================================
# 8. 상세 usable 목록 저장
# ==========================================================

usable.to_csv(
    "usable_experiments.csv",
    index=False
)

print()
print(
    "Saved: usable_experiments.csv"
)