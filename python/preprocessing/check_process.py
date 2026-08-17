# from netCDF4 import Dataset, num2date
# import numpy as np
# import pandas as pd

# PROCESS_FILE = "Process_data.nc"
# DICT_FILE = "Dictionary_process.nc"
# WAFER = "Day_2024_08_05_Wafer_01"

# # ---------------------------------------
# # Dictionary
# # ---------------------------------------
# with Dataset(DICT_FILE, "r") as dict_ds:
#     decoder = dict_ds["data"][:]

# # ---------------------------------------
# # Process Data
# # ---------------------------------------
# with Dataset(PROCESS_FILE, "r") as ds:

#     print("Group exists:", WAFER in ds.groups)

#     grp = ds.groups[WAFER]

#     print("\nVariables:")
#     for name, var in grp.variables.items():
#         print(name, var.shape)

#     encoded = grp["data"][:]
#     process_data = decoder[encoded]

#     # Feature names
#     features = grp["feature"][:]

#     feature_names = []

#     for f in features:
#         if isinstance(f, bytes):
#             feature_names.append(f.decode())
#         else:
#             feature_names.append(str(f))

#     # Time
#     # time_var = grp["times"]

#     # times = num2date(
#     #     time_var[:],
#     #     time_var.units,
#     #     time_var.calendar,
#     #     only_use_cftime_datetimes=False,
#     #     only_use_python_datetimes=True
#     # )

#     time_var = grp["times"]

#     # 파일이 열려 있을 때 실제 값을 복사
#     raw_process_time = np.asarray(time_var[:])

#     times = num2date(
#         raw_process_time,
#         time_var.units,
#         time_var.calendar,
#         only_use_cftime_datetimes=False,
#         only_use_python_datetimes=True
#     )

#     df = pd.DataFrame(
#         process_data,
#         index=times,
#         columns=feature_names
#     )

# print("\nNumber of features:", len(feature_names))

# print("\nFeature names:")
# for i, name in enumerate(feature_names):
#     print(i, name)

# duration = (
#     df.index[-1] - df.index[0]
# ).total_seconds()

# print("\nStart :", df.index[0])
# print("End   :", df.index[-1])
# print("Duration :", duration, "sec")

# print("\nShape :", df.shape)

# import matplotlib.pyplot as plt
# import numpy as np

# # Process time을 숫자 초 단위로 사용
# # process_time = np.asarray(time_var[:])
# process_time = raw_process_time

# gas4 = df["Stat3_Etch_MV_Gas4Flow"].values
# gas5 = df["Stat3_Etch_MV_Gas5Flow"].values

# # 전체 구간
# plt.figure(figsize=(12, 5))

# plt.plot(process_time, gas4, label="Gas4Flow")
# plt.plot(process_time, gas5, label="Gas5Flow")

# plt.xlabel("Process Time (sec)")
# plt.ylabel("Gas Flow")
# plt.title("Wafer_01 - BOSCH Gas Flow")
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.savefig("Wafer_01_gas_flow.png", dpi=150)
# plt.show()


# # 반복 주기가 잘 보이도록 일부만 확대
# plt.figure(figsize=(12, 5))

# mask = (process_time >= 150) & (process_time <= 200)

# plt.plot(
#     process_time[mask],
#     gas4[mask],
#     label="Gas4Flow"
# )

# plt.plot(
#     process_time[mask],
#     gas5[mask],
#     label="Gas5Flow"
# )

# plt.xlabel("Process Time (sec)")
# plt.ylabel("Gas Flow")
# plt.title("Wafer_01 - BOSCH Cycle Zoom")
# plt.legend()
# plt.grid()

# plt.tight_layout()
# plt.savefig("Wafer_01_gas_cycle_zoom.png", dpi=150)
# plt.show()

from netCDF4 import Dataset
import numpy as np

# --------------------------
# OES raw time
# --------------------------
with Dataset("Day_2024_08_05.nc", "r") as ds:
    grp = ds.groups["Wafer_01"]
    oes_time_var = grp["times"]

    oes_time = np.asarray(oes_time_var[:])

    print("=== OES ===")
    print("units :", oes_time_var.units)
    print("start :", oes_time[0])
    print("end   :", oes_time[-1])
    print("duration :", oes_time[-1] - oes_time[0])

# --------------------------
# Process raw time
# --------------------------
with Dataset("Process_data.nc", "r") as ds:
    grp = ds.groups["Day_2024_08_05_Wafer_01"]
    process_time_var = grp["times"]

    process_time = np.asarray(process_time_var[:])

    print("\n=== PROCESS ===")
    print("units :", process_time_var.units)
    print("start :", process_time[0])
    print("end   :", process_time[-1])
    print("duration :", process_time[-1] - process_time[0])