import numpy as np
import pandas as pd

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

# 1) 单 Pack 工程：720 步调频电流
p1 = (
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制/"
    "single_pack_plant/outputs/dompc_frequency/physics_p_dompc/"
    "frequency_current_720steps.csv"
)
f1 = pd.read_csv(p1)
print("720steps csv cols:", f1.columns.tolist(), "rows:", len(f1))
col = f1.columns[-1]
v = f1[col].to_numpy(float)
print(f"pack current: min={v.min():.1f} max={v.max():.1f} "
      f"mean={v.mean():.1f} mean_abs={np.abs(v).mean():.1f} rms={np.sqrt((v**2).mean()):.1f}")
print(f"per-cell (pack/4): min={v.min()/4:.1f} max={v.max()/4:.1f} "
      f"rms={np.sqrt((v**2).mean())/4:.1f} A")
print(f"per-cell C-rate @280Ah: charge={-v.min()/4/280:.2f}C discharge={v.max()/4/280:.2f}C")

# 2) cluster_plant_v2：RegD ×1120 A 的原始包络
from cluster_plant_v2.validation import validate_domp_mpc as val

_, cur = val.load_regd_profile(val.AGC_DATA_FILE, duration_s=6400.0, dt=5.0)
print("\ncluster RegD x1120: min=%.1f max=%.1f" % (cur.min(), cur.max()))
print("cluster/4 per-cell: min=%.1f max=%.1f A" % (cur.min() / 4, cur.max() / 4))

# 3) cluster 里 1120 A 是加给谁的：查 case_currents 的使用方式与 pack 容量
import inspect

src = inspect.getsource(val.case_currents)
print("\ncase_currents:\n", src[:600])
