from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

OES_FILE = "Day_2024_08_05.nc"
DICT_FILE = "Dictionary_OES.nc"
WAFER = "Wafer_01"

# Process에서 검출한 값
PROCESS_BOSCH_START = 30.0
PROCESS_BOSCH_END   = 628.8

# OES가 Process보다 약 0.2초 늦음
TIME_OFFSET = 0.2

OES_BOSCH_START = PROCESS_BOSCH_START + TIME_OFFSET
OES_BOSCH_END   = PROCESS_BOSCH_END + TIME_OFFSET

print("OES BOSCH start :", OES_BOSCH_START)
print("OES BOSCH end   :", OES_BOSCH_END)


# ==========================================================
# 1. OES Dictionary
# ==========================================================

with Dataset(DICT_FILE, "r") as ds:
    decoder = np.asarray(ds["data"][:])


# ==========================================================
# 2. OES load + BOSCH crop
# ==========================================================

with Dataset(OES_FILE, "r") as ds:

    grp = ds.groups[WAFER]

    raw_time = np.asarray(grp["times"][:])
    oes_time = raw_time - raw_time[0]

    wavelengths = np.asarray(grp["wavelengths"][:])

    # BOSCH 구간 선택
    mask = (
        (oes_time >= OES_BOSCH_START) &
        (oes_time <= OES_BOSCH_END)
    )

    indices = np.where(mask)[0]

    print("Selected OES samples :", len(indices))

    start_idx = indices[0]
    end_idx = indices[-1] + 1

    # 필요한 시간 영역만 NetCDF에서 읽음
    encoded = np.asarray(
        grp["data"][start_idx:end_idx, :],
        dtype=np.int64
    )

    oes_crop = decoder[encoded]

    crop_time = oes_time[start_idx:end_idx]


# Crop 시작점을 다시 0초로
crop_time = crop_time - crop_time[0]

print("Crop shape :", oes_crop.shape)
print(
    "Crop duration :",
    crop_time[-1] - crop_time[0],
    "sec"
)

print(
    "Wavelength :",
    wavelengths[0],
    "~",
    wavelengths[-1],
    "nm"
)


# ==========================================================
# 3. 확인용 Time-Wavelength Map
# ==========================================================

# 시각화만 하기 위해 시간축 일부 샘플링
num_display = 600

display_idx = np.linspace(
    0,
    len(crop_time) - 1,
    num_display
).astype(int)

display_data = oes_crop[display_idx, :]

plt.figure(figsize=(13, 6))

plt.imshow(
    display_data,
    aspect="auto",
    origin="lower",
    extent=[
        wavelengths[0],
        wavelengths[-1],
        crop_time[0],
        crop_time[-1]
    ]
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("BOSCH Process Time (sec)")
plt.title("Wafer_01 - Cropped BOSCH OES")

plt.colorbar(label="OES Intensity")

plt.tight_layout()

plt.savefig(
    "Wafer_01_BOSCH_OES_crop.png",
    dpi=150
)

plt.show()