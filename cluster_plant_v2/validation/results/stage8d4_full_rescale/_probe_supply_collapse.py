"""8D4f diagnosis: why does the model supply collapse in the steady rolls?

The plant sweep is healthy (q_evap/map = 1.01..1.06), yet rolling the
corrected model at the same commands drives the supply 4-5 K below the
plant value. Reconcile at each sweep point: plant q_evap vs the capacity
map vs the model's own steady evaporator load, and report where the
model's self-consistent operating point departs from the plant. Kept for
reference.
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
predictor = build_predictor(395.0, 0.74)

print(
    "cmd   plant_qevap  map@plantTank  model_qevap  model_tank  "
    "model_supply  plant_supply  evap_drop_model  evap_drop_plant"
)
for index, row in sweep.iterrows():
    x = np.asarray(seed_state(row), dtype=float).reshape(-1, 1)
    u = np.array([float(row["cmd_rpm"]), pump_virtual]).reshape(-1, 1)
    d = np.array([560.0, T_AMB_C, heat_at_sweep_end(int(index))]).reshape(-1, 1)
    for _ in range(ROLL_STEPS):
        x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
    x = x.reshape(-1)
    q_map_plant_tank = float(
        predictor._capacity(float(row["cmd_rpm"]), pump_virtual, row["tank_c"], 35.0)
    )
    evap_drop_model = x[1] - x[9]  # tank - evaporator out
    evap_drop_plant = row["q_evap_plant_w"] / (0.3899 * (4800 / 3140.58) * 3391)
    print(
        f"{row['cmd_rpm']:5.0f}  {row['q_evap_plant_w']:10.0f}  "
        f"{q_map_plant_tank:12.0f}  {x[5]:11.0f}  {x[1]:9.2f}  "
        f"{x[7]:11.2f}  {row['supply_c']:11.2f}  {evap_drop_model:13.2f}  "
        f"{evap_drop_plant:13.2f}"
    )

# Also check the map at the model's own tank temperature (self-consistent).
print("\ncmd   model_tank  map@modelTank  model_qevap  gap")
for index, row in sweep.iterrows():
    x = np.asarray(seed_state(row), dtype=float).reshape(-1, 1)
    u = np.array([float(row["cmd_rpm"]), pump_virtual]).reshape(-1, 1)
    d = np.array([560.0, T_AMB_C, heat_at_sweep_end(int(index))]).reshape(-1, 1)
    for _ in range(ROLL_STEPS):
        x = np.asarray(predictor.function(x, u, d), dtype=float).reshape(-1, 1)
    x = x.reshape(-1)
    q_map_model_tank = float(
        predictor._capacity(float(row["cmd_rpm"]), pump_virtual, x[1], 35.0)
    )
    print(
        f"{row['cmd_rpm']:5.0f}  {x[1]:9.2f}  {q_map_model_tank:12.0f}  "
        f"{x[5]:11.0f}  {q_map_model_tank - x[5]:+6.0f}"
    )
