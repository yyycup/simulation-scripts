"""Diagnose the 8D4b NMPC under-band behaviour (all out-of-band is undershoot)."""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DT = 5.0
REF = 25.0
BAND = 0.65

frame = pd.read_csv(HERE / "case_T2_regd_8d4_nmpc_soc_aware_timeseries.csv")
frame = frame.sort_values("time_s")
temp = frame["battery_avg_temp_c"].to_numpy()
t = frame["time_s"].to_numpy()

below = temp < REF - BAND
above = temp > REF + BAND
print(f"below-band: {below.sum() * DT:.0f} s, above-band: {above.sum() * DT:.0f} s")
print(f"min temp {temp.min():.2f} at t={t[temp.argmin()]:.0f} s, "
      f"max temp {temp.max():.2f} at t={t[temp.argmax()]:.0f} s")
print(f"tank min {frame['tank_temp_c'].min():.2f} / mean {frame['tank_temp_c'].mean():.2f}")
print(f"supply min {frame['supply_temp_c'].min():.2f} / mean {frame['supply_temp_c'].mean():.2f}")
print(f"pump cmd min/mean/max {frame['pump_command_rpm'].min():.0f}/"
      f"{frame['pump_command_rpm'].mean():.0f}/{frame['pump_command_rpm'].max():.0f}")
print(f"comp cmd min/mean/max {frame['compressor_command_rpm'].min():.0f}/"
      f"{frame['compressor_command_rpm'].mean():.0f}/{frame['compressor_command_rpm'].max():.0f}")

# where are the under-band episodes?
runs = []
start = None
for index, flag in enumerate(below):
    if flag and start is None:
        start = index
    elif not flag and start is not None:
        runs.append((t[start], t[index - 1]))
        start = None
if start is not None:
    runs.append((t[start], t[-1]))
print(f"{len(runs)} under-band episodes:")
for begin, end in runs[:15]:
    mask = (t >= begin) & (t <= end)
    print(f"  {begin:.0f}-{end:.0f} s ({end - begin:.0f} s), "
          f"temp min {temp[mask].min():.2f}, "
          f"pump mean {frame['pump_command_rpm'][mask].mean():.0f}, "
          f"comp mean {frame['compressor_command_rpm'][mask].mean():.0f}")

# sample the trajectory at coarse intervals
print("coarse trajectory (t, temp, tank, supply, comp_cmd, pump_cmd):")
for index in range(0, len(frame), 160):
    row = frame.iloc[index]
    print(f"  {row['time_s']:6.0f} {row['battery_avg_temp_c']:6.2f} "
          f"{row['tank_temp_c']:6.2f} {row['supply_temp_c']:6.2f} "
          f"{row['compressor_command_rpm']:6.0f} {row['pump_command_rpm']:6.0f}")
