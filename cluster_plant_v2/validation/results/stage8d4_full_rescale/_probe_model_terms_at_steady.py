"""Probe: model term audit at the settled 8D4e operating point.

Computes the frozen model's plate-fluid conductance, flow capacity and
equilibrium battery temperature at the late-steady command/pump state
of the 8D4e rerun and compares against the plant's implied per-pack
coupling, to size the plate-fluid coupling mismatch. Kept for reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import json

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = Path("single_pack_plant/model_data/physics_p_operational_8d4.json")

thermal = json.loads(ROOT.read_text(encoding="utf-8"))["thermal"]
cp = thermal["coolant_cp_j_kg_k"]
m_ref = thermal["coolant_mass_flow_ref_kg_s"]
g_bat_plate = thermal["battery_plate_conductance_w_k"]
eff = thermal["plate_fluid_effectiveness"]
pump_ref = thermal["n_pump_ref_rpm"]

df = pd.read_csv(
    HERE / "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv"
).sort_values("time_s")
late = df[df["time_s"] > 5400]

pump_actual = late["pump_command_rpm"].mean()
pump_virtual = min(pump_actual * 2.0, 4800.0)
ratio = pump_virtual / pump_ref
flow_capacity = m_ref * ratio * cp
ref_flow = m_ref * cp
plate_fluid = eff * min(flow_capacity, ref_flow * ratio**0.8)

tank = late["tank_temp_c"].mean()
supply = late["supply_temp_c"].mean()
q_evap = late["q_evap_applied_w"].mean()
bat = late["battery_avg_temp_c"].mean()
q_gen_pack = 1656.0  # per-pack late plant q_gen from 8/27 calibration
air_in = -thermal["ambient_conductance_w_k"] * (bat - 35.0)

print(f"late steady: pump_actual {pump_actual:.0f} rpm (virtual {pump_virtual:.0f})")
print(f"ratio {ratio:.3f}, flow_capacity {flow_capacity:.0f} W/K")
print(f"plate_fluid (model) = {plate_fluid:.0f} W/K")
print(f"model supply drop: q_evap/flow_cap = {q_evap / flow_capacity:.2f} K "
      f"(plant: {tank - supply:.2f} K)")

# Model equilibrium battery temp at plant supply temp.
t_plate = (g_bat_plate * bat + plate_fluid * supply) / (g_bat_plate + plate_fluid)
print(f"model plate eq at plant bat {bat:.2f}: {t_plate:.2f} C, "
      f"q_bat_plate {g_bat_plate * (bat - t_plate):.0f} W vs need {q_gen_pack + air_in:.0f} W")
# Solve model equilibrium self-consistently with the pack heat balance:
# bat - plate = q_need/g_bat_plate; plate = (g*bat + pf*supply)/(g+pf)
# => bat = supply + q_need*(1/g + 1/pf)
q_need = q_gen_pack + air_in
bat_eq_model = supply + q_need * (1.0 / g_bat_plate + 1.0 / plate_fluid)
print(f"model equilibrium battery temp at plant supply: {bat_eq_model:.2f} C "
      f"(plant bat {bat:.2f} -> gap {bat_eq_model - bat:+.2f} K)")

# Plant implied per-pack plate-fluid conductance.
per_pack_fluid_heat = q_need  # steady: all pack heat leaves via fluid
plant_return = late["return_temp_c"].mean()
flow_per_pack = flow_capacity / 5.0 / cp  # kg/s per pack
fluid_rise = per_pack_fluid_heat / (flow_per_pack * cp)
print(f"plant per-pack fluid rise {fluid_rise:.2f} K "
      f"(from return-supply {plant_return - supply:.2f} K)")
t_fluid_mean = supply + (plant_return - supply) / 2.0
plate_implied = bat - q_need / g_bat_plate
g_plant_plate_fluid = per_pack_fluid_heat / max(plate_implied - t_fluid_mean, 0.1)
print(f"plant implied plate temp {plate_implied:.2f} C, "
      f"plate-fluid effective conductance {g_plant_plate_fluid:.0f} W/K "
      f"vs model {plate_fluid:.0f} W/K")
