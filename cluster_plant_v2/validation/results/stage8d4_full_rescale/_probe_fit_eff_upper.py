"""Quick probe: does the 8D4f fit roll diverge for eff > 1?"""

import sys
import time
from pathlib import Path

ROOT = Path("c:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")
sys.path.insert(0, str(ROOT))

src = Path(
    "cluster_plant_v2/validation/results/stage8d4_full_rescale/"
    "_fit_8d4f_cluster_effectiveness.py"
).read_text(encoding="utf-8")
code = src.split("best = (None")[0]
exec(compile(code, "fit_head", "exec"))

t0 = time.time()
for eff in (1.05, 1.10, 1.20, 1.40):
    try:
        err = fit_error(eff, 0.80)
        print(f"eff {eff:.2f}: rms {err:.3f}, {time.time()-t0:.0f}s", flush=True)
    except Exception as exc:
        print(f"eff {eff:.2f}: FAILED {type(exc).__name__}: {exc}", flush=True)

# Inspect a divergent roll if any.
import numpy as np

row = sweep.iloc[0]
predictor = build_predictor(1.20, 0.80)
x0 = seed_state(row)
x = np.asarray(x0, dtype=float).reshape(-1, 1)
u = np.array([float(row["cmd_rpm"]), min(BASELINE_PUMP_RPM * PV, 4800.0)]).reshape(-1, 1)
d = np.array([560.0, T_AMB_C, heat_at_sweep_end(0)]).reshape(-1, 1)
for i in range(720):
    x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
    if i in (10, 50, 200, 719) or not np.all(np.isfinite(x)):
        print(
            f"step {i}: bat {float(x[0]):8.2f} tank {float(x[1]):8.2f} "
            f"plate {float(x[2]):8.2f} qevap {float(x[5]):8.0f} "
            f"supply {float(x[7]):8.2f}",
            flush=True,
        )
        if not np.all(np.isfinite(x)):
            break
