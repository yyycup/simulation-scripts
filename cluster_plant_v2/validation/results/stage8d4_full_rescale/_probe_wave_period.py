"""Probe: map compressor on/off segments to temperature swings (8D4f run).

Explains the wavy temperature: the steady-state cooling demand sits
below the compressor's minimum continuous capacity, so the thermostat
overlay cycles the machine on/off and temperature rides a limit cycle.
Kept for reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np
import pandas as pd

RES = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")
df = pd.read_csv(
    RES / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv"
).sort_values("time_s")
t = df["time_s"].to_numpy()
temp = df["battery_avg_temp_c"].to_numpy()
cmd = df["compressor_command_rpm"].to_numpy()
act = df["compressor_actual_rpm"].to_numpy()

off = act < 100.0
edges = [0] + list(np.where(np.diff(off.astype(int)) != 0)[0] + 1) + [len(off)]

print("start  end    state  dur(s)  temp_swing(K)  temp_end")
for a, b in zip(edges[:-1], edges[1:]):
    state = "OFF" if off[a] else "ON"
    seg = temp[a:b]
    print(
        f"{t[a]:6.0f} {t[b - 1]:6.0f}  {state:3s}  {t[b - 1] - t[a]:6.0f}"
        f"   {seg.max() - seg.min():6.2f}       {seg[-1]:5.2f}"
    )

# Late-segment stats split by on/off.
late_mask = t > 3600
print()
print("t>3600s: off fraction", float(off[late_mask].mean()))
print("t>3600s: ON mean temp %.3f  OFF mean temp %.3f"
      % (temp[late_mask & ~off].mean(), temp[late_mask & off].mean()))

# Dominant period of the temperature wave (late segment).
late = temp[late_mask] - temp[late_mask].mean()
f = np.fft.rfft(late)
freq = np.fft.rfftfreq(len(late), d=5.0)
k = int(np.argmax(np.abs(f[1:])) + 1)
print("dominant wave period: %.0f s  amplitude %.3f K"
      % (1.0 / freq[k], 2.0 * np.abs(f[k]) / len(late)))
