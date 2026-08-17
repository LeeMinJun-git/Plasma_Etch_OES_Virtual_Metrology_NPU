import pandas as pd

CSV_FILE = "Si_Oxide_etch_9_points.csv"

# ==========================================================
# 1. CSV Load
# ==========================================================

df = pd.read_csv(CSV_FILE)

print("=" * 80)
print("CSV BASIC INFO")
print("=" * 80)

print("\nShape:")
print(df.shape)

print("\nColumns:")
for i, col in enumerate(df.columns):
    print(i, repr(col))

print("\nDtypes:")
print(df.dtypes)

print("\nFirst 10 rows:")
print(df.head(10).to_string())


# ==========================================================
# 2. 2024-08-05가 들어있는 행 찾기
# ==========================================================

print()
print("=" * 80)
print("SEARCH: 2024-08-05")
print("=" * 80)

mask = pd.Series(False, index=df.index)

for col in df.columns:

    text = df[col].astype(str)

    mask |= text.str.contains(
        "2024-08-05",
        na=False,
        regex=False
    )

matched = df[mask]

print("\nMatched rows:")
print(matched.to_string())

print("\nMatched row count:")
print(len(matched))


# ==========================================================
# 3. si / etch 관련 column 확인
# ==========================================================

print()
print("=" * 80)
print("ETCH-RELATED COLUMNS")
print("=" * 80)

for col in df.columns:

    name = str(col).lower()

    if (
        "etch" in name
        or "si" in name
        or "wafer" in name
        or "date" in name
        or "lot" in name
        or "point" in name
    ):
        print(repr(col))