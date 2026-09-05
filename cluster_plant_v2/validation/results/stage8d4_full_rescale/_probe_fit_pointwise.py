"""8D4f fit diagnosis: per-point residuals and roll convergence checks.

The sweep-anchor fit keeps pushing the plate-fluid anchor toward zero,
which means some other calibration is off. Print the per-point model vs
plant battery temperature, the model equilibrium details at the cmd-1900
point (near the closed-loop operating point), and confirm the 60 min rolls
actually converge. Kept for reference.
"""

import sys
from pathlib import Path

ROOT = Path("c:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")
sys.path.insert(0, str(ROOT))

import numpy as np

src = Path(
    "cluster_plant_v2/validation/results/stage8d4_full_rescale/"
    "_fit_8d4f_cluster_effectiveness.py"
).read_text(encoding="utf-8")
code = src.split("best = (None")[0]
exec(compile(code, "fit_head", "exec"))

pump_virtual = min(BASELINE_PUMP_RPM * PV, 4800.0)

for ref, scale in ((280.0, 1.00), (280.0, 0.80), (395.0, 0.80), (395.0, 0.74)):
    predictor = build_predictor(ref, scale)
    print(f"\n=== ref {ref:.0f} W/K, scale {scale:.2f} ===")
    print("cmd   plant  model  resid   qanchor  conv_last10_drift")
    for index, row in sweep.iterrows():
        x0 = seed_state(row)
        x = np.asarray(x0, dtype=float).reshape(-1, 1)
        u = np.array([float(row["cmd_rpm"]), pump_virtual]).reshape(-1, 1)
        d = np.array(
            [560.0, T_AMB_C, heat_at_sweep_end(int(index))]
        ).reshape(-1, 1)
        tail = []
        for _ in range(ROLL_STEPS):
            x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
            tail.append(float(x[0]))
        drift = tail[-1] - tail[-11]
        print(
            f"{row['cmd_rpm']:5.0f}  {row['bat_c']:6.2f}  {tail[-1]:6.2f}  "
            f"{tail[-1]-row['bat_c']:+6.2f}   {float(d[2]):7.0f}   {drift:+8.4f}"
        )

# Equilibrium decomposition at the cmd-1900 point with ref 395.
predictor = build_predictor(395.0, 0.74)
row = sweep.iloc[3]
x = np.asarray(seed_state(row), dtype=float).reshape(-1, 1)
u = np.array([float(row["cmd_rpm"]), pump_virtual]).reshape(-1, 1)
q_anchor = heat_at_sweep_end(3)
d = np.array([560.0, T_AMB_C, q_anchor]).reshape(-1, 1)
for _ in range(ROLL_STEPS):
    x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
x = x.reshape(-1)
pf = 395.0 * (4800.0 / 3140.58) ** 0.8
g = 521.86 + pf
print(f"\ncmd-1900 model equilibrium (ref 395, scale 0.74):")
print(f"  q_anchor {q_anchor:.0f} W, q_gen_eff {q_anchor*0.74:.0f} W")
print(f"  plate_fluid {pf:.0f} W/K, series G {1/(1/521.86+1/pf):.0f} W/K")
print(f"  model bat {x[0]:.2f} plate {x[2]:.2f} supply {x[7]:.2f} "
      f"tank {x[1]:.2f} return {x[10]:.2f}")
print(f"  plant bat {row['bat_c']:.2f} supply {row['supply_c']:.2f} "
      f"tank {row['tank_c']:.2f}")
amb = 7.72 * (x[0] - T_AMB_C)
qp = pf * (x[2] - x[7])
print(f"  balance: q_gen_eff {q_anchor*0.74:.0f} = plate {qp:.0f} + amb {amb:.0f}")
