from netCDF4 import Dataset
import numpy as np

PROCESS_FILE = "Process_data.nc"
DICT_FILE = "Dictionary_process.nc"
WAFER = "Day_2024_08_05_Wafer_01"

# -------------------------------------------------
# 1. Dictionary
# -------------------------------------------------
with Dataset(DICT_FILE, "r") as ds:
    decoder = np.asarray(ds["data"][:])

# -------------------------------------------------
# 2. Process data
# -------------------------------------------------
with Dataset(PROCESS_FILE, "r") as ds:
    grp = ds.groups[WAFER]

    raw_time = np.asarray(grp["times"][:])
    process_time = raw_time - raw_time[0]

    features = [
        f.decode() if isinstance(f, bytes) else str(f)
        for f in grp["feature"][:]
    ]

    encoded = np.asarray(grp["data"][:], dtype=np.int64)
    data = decoder[encoded]

gas5_idx = features.index("Stat3_Etch_MV_Gas5Flow")
gas5 = data[:, gas5_idx]

# -------------------------------------------------
# 3. Gas5 HIGH 구간 검출
# -------------------------------------------------
# 정상 BOSCH Gas5는 약 600이므로 threshold 300 사용
high = gas5 > 300

rising = np.where(
    (~high[:-1]) & (high[1:])
)[0] + 1

falling = np.where(
    (high[:-1]) & (~high[1:])
)[0] + 1

rise_times = process_time[rising]
fall_times = process_time[falling]

print("All Gas5 rising edges :", len(rise_times))
print(rise_times)

# -------------------------------------------------
# 4. 약 6초 간격으로 이어지는 연속 pulse 찾기
# -------------------------------------------------
# intervals = np.diff(rise_times)

# print("\nIntervals between rising edges:")
# print(intervals)

# # 5~7초 간격인 pulse만 BOSCH cycle 후보
# valid = (intervals > 5.0) & (intervals < 7.0)

# # 가장 긴 연속 구간 탐색
# best_start = 0
# best_len = 0

# current_start = 0
# current_len = 0

# for i, ok in enumerate(valid):
#     if ok:
#         if current_len == 0:
#             current_start = i
#         current_len += 1

#         if current_len > best_len:
#             best_len = current_len
#             best_start = current_start
#     else:
#         current_len = 0

# # edge 개수 = interval 개수 + 1
# cycle_rises = rise_times[
#     best_start : best_start + best_len + 1
# ]

# print("\nDetected BOSCH cycles:", len(cycle_rises))
# print("First cycle start :", cycle_rises[0], "sec")
# print("Last cycle start  :", cycle_rises[-1], "sec")

# # 마지막 Gas5 falling edge를 찾아 전체 종료 계산
# last_rise = cycle_rises[-1]

# candidate_falls = fall_times[fall_times > last_rise]

# if len(candidate_falls):
#     bosch_end = candidate_falls[0]
# else:
#     bosch_end = last_rise + 4.5

# bosch_start = cycle_rises[0]

# print("BOSCH start :", bosch_start, "sec")
# print("BOSCH end   :", bosch_end, "sec")
# print("Duration    :", bosch_end - bosch_start, "sec")

# -------------------------------------------------
# 4. Wafer_01 BOSCH 100-cycle 구간 확정
# -------------------------------------------------

intervals = np.diff(rise_times)

print("\nIntervals between rising edges:")
print(intervals)

# Wafer_01에서는 첫 rising edge(18 sec)가
# BOSCH 100-cycle 이전의 사전 pulse로 확인됨.
# 따라서 이후 100개의 rising edge를 BOSCH cycle로 사용.
cycle_rises = rise_times[1:]

print("\nDetected BOSCH cycles:", len(cycle_rises))

bosch_start = cycle_rises[0]
last_rise = cycle_rises[-1]

# 마지막 Gas5 pulse가 끝나는 시점
candidate_falls = fall_times[fall_times > last_rise]

if len(candidate_falls) > 0:
    bosch_end = candidate_falls[0]
else:
    bosch_end = last_rise + 4.5

print("First cycle start :", cycle_rises[0], "sec")
print("Last cycle start  :", cycle_rises[-1], "sec")

print("\nBOSCH start :", bosch_start, "sec")
print("BOSCH end   :", bosch_end, "sec")
print("Duration    :", bosch_end - bosch_start, "sec")