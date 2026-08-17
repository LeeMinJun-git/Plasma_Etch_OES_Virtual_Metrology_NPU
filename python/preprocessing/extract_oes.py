# from netCDF4 import Dataset

# file_path = "Day_2024_08_05.nc"

# with Dataset(file_path, "r") as ds:
#     print("Groups:")
#     print(list(ds.groups.keys()))

##########################################

# from netCDF4 import Dataset

# with Dataset("Day_2024_08_05.nc", "r") as ds:
#     grp = ds.groups["Wafer_01"]

#     print("Variables:")
#     print(list(grp.variables.keys()))

#     for name, var in grp.variables.items():
#         print(name, var.shape)


#########################################
from netCDF4 import Dataset, num2date
import numpy as np
import matplotlib.pyplot as plt

OES_FILE = "Day_2024_08_05.nc"
DICT_FILE = "Dictionary_OES.nc"
WAFER = "Wafer_01"

# --------------------------------------------------
# 1. Dictionary 불러오기
# --------------------------------------------------
with Dataset(DICT_FILE, "r") as dict_ds:
    decoder = np.asarray(dict_ds["data"][:])

print("Dictionary size :", decoder.shape)
print("Dictionary dtype:", decoder.dtype)

# --------------------------------------------------
# 2. Wafer_01 데이터 확인
# --------------------------------------------------
with Dataset(OES_FILE, "r") as ds:

    grp = ds.groups[WAFER]

    wavelengths = np.asarray(grp["wavelengths"][:])

    time_var = grp["times"]
    raw_times = time_var[:]

    times = num2date(
        raw_times,
        time_var.units,
        time_var.calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True
    )

    n_time = grp["data"].shape[0]
    n_wave = grp["data"].shape[1]

    print()
    print("Wafer:", WAFER)
    print("OES shape:", (n_time, n_wave))
    print(
        "Wavelength:",
        wavelengths.min(),
        "~",
        wavelengths.max(),
        "nm"
    )

    duration = (times[-1] - times[0]).total_seconds()
    print("Duration:", duration, "sec")
    print("Approx sampling rate:", (n_time - 1) / duration, "Hz")

    # --------------------------------------------------
    # 3. 가운데 시점의 Spectrum 하나 Decode
    # --------------------------------------------------
    mid = n_time // 2

    encoded_spectrum = np.asarray(
        grp["data"][mid, :],
        dtype=np.int64
    )

    spectrum = decoder[encoded_spectrum]

    # --------------------------------------------------
    # 4. Heatmap용으로 256×256 Downsampling
    # --------------------------------------------------
    time_idx = np.linspace(
        0, n_time - 1, 256
    ).astype(int)

    wave_idx = np.linspace(
        0, n_wave - 1, 256
    ).astype(int)

    # 한 번에 전체 14920×3648을 읽지 않음
    encoded_time_sample = np.asarray(
        grp["data"][time_idx, :],
        dtype=np.int64
    )

    encoded_small = encoded_time_sample[:, wave_idx]
    oes_small = decoder[encoded_small]

    wave_small = wavelengths[wave_idx]

# --------------------------------------------------
# 5. Spectrum 출력
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(wavelengths, spectrum)

plt.xlabel("Wavelength (nm)")
plt.ylabel("OES Intensity")
plt.title("Wafer_01 - OES Spectrum")
plt.tight_layout()

plt.savefig("Wafer_01_spectrum.png", dpi=150)
plt.show()

# --------------------------------------------------
# 6. Time × Wavelength Heatmap 출력
# --------------------------------------------------
plt.figure(figsize=(10, 6))

plt.imshow(
    oes_small,
    aspect="auto",
    origin="lower",
    extent=[
        wave_small[0],
        wave_small[-1],
        0,
        duration
    ]
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Time (sec)")
plt.title("Wafer_01 - OES Time-Wavelength Map")

plt.colorbar(label="OES Intensity")

plt.tight_layout()
plt.savefig("Wafer_01_heatmap.png", dpi=150)
plt.show()