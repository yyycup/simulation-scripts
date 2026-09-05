"""Probe: verify the 8D4h hot-start plant initial state.

Builds the same plant as _run_peak_s0_8d4h_hot_start.py and prints the
initial battery/supply/tank temperatures and q_evap, plus the first few
steps, to confirm the loop starts hot-biased (no pre-chilled 4000 rpm
equilibrium). Kept for reference.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import numpy as np

from cluster_plant_v2.control.physics_p_nmpc_controller import (
    PhysicsPNmpcParameters,
)
from cluster_plant_v2.validation.validate_domp_mpc import (
    BASELINE_PUMP_RPM,
    build_cases,
    case_currents,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    FAN_SPEED_RPM,
)
from cluster_plant_v2.validation.validate_physics_p_nmpc import (
    KELVIN_OFFSET,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    INITIAL_SOC,
    PUMP_SPEED_RPM,
    build_engineering_pump,
    build_medium_header_network,
)
from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import CoolantTank
from cluster_plant_v2.parameters import (
    COMPRESSOR_TIME_CONSTANT_S,
    EVAPORATOR_TIME_CONSTANT_S,
)
from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.refrigeration import ClosedR134aCycle, CompressorSpeedActuator

DT = 5.0
STEPS = 1280
HOT_START_TEMP_C = 27.0
HOT_START_TEMP_K = HOT_START_TEMP_C + KELVIN_OFFSET

case = [c for c in build_cases() if c.case_id == "S0_constant"][0]
currents = case_currents(case, STEPS, DT)
parameters = PhysicsPNmpcParameters()

network = build_medium_header_network()
cluster = ReducedCluster(
    n_packs=5,
    hydraulic_mode="header_network",
    hydraulic_network=network,
    pack_config={
        "initial_soc": INITIAL_SOC,
        # ReducedBatteryPack reads "initial_temp_c" (degC); the legacy
        # kelvin keys in build_final_plant are silently ignored.
        "initial_temp_c": HOT_START_TEMP_C,
    },
)
# ReducedCluster never forwards a plate temperature, so set it directly.
for pack in cluster.packs:
    pack.cold_plate.plate_temperatures[:] = HOT_START_TEMP_K
    pack.cold_plate.initial_zone_energy_J = (
        pack.cold_plate.zone_heat_capacities
        * pack.cold_plate.plate_temperatures
    )
    pack.cold_plate.coolant_outlet_temperature = float(
        pack.cold_plate.plate_temperatures[0]
    )
    pack.cold_plate.coolant_mean_temperatures = (
        pack.cold_plate.plate_temperatures.copy()
    )
pump = build_engineering_pump(network)
tank = CoolantTank(initial_temperature_k=HOT_START_TEMP_K)
cycle = ClosedR134aCycle()
actuator = CompressorSpeedActuator(
    initial_speed_rpm=parameters.compressor_plant_lower_rpm,
    time_constant_s=COMPRESSOR_TIME_CONSTANT_S,
)
plant = ClusterPlant.from_equilibrium(
    cluster=cluster,
    hydraulic_network=network,
    pump=pump,
    tank=tank,
    refrigeration_cycle=cycle,
    compressor_actuator=actuator,
    dt_s=DT_S,
    pump_speed_rpm=PUMP_SPEED_RPM,
    fan_speed_rpm=FAN_SPEED_RPM,
    initial_cluster_current_a=float(currents[0]),
    ambient_temperature_k=AMBIENT_TEMPERATURE_K,
    direction="forward",
    evaporator_time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
)

battery_c = float(
    np.mean(
        [
            np.average(
                pack.battery.temps, weights=pack.battery.zone_heat_capacities
            )
            for pack in plant.cluster.packs
        ]
    )
    - KELVIN_OFFSET
)
print(
    "initial: battery %.2f C  tank %.2f C  evaporator q %.0f W"
    % (
        battery_c,
        float(plant.tank.temperature_k - KELVIN_OFFSET),
        float(plant.evaporator_dynamics.q_evap_applied_w),
    )
)

for i in range(6):
    result = plant.step(
        dt_s=DT,
        cluster_current_a=float(currents[i]),
        pump_speed_rpm=BASELINE_PUMP_RPM,
        compressor_speed_command_rpm=parameters.compressor_plant_lower_rpm,
        fan_speed_rpm=FAN_SPEED_RPM,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction="forward",
    )
    print(
        "t=%4.0f  battery %5.2f  supply %5.2f  tank %5.2f  q_evap %6.0f"
        % (
            (i + 1) * DT,
            float(
                np.mean(
                    result["cluster_result"][
                        "pack_battery_average_temperatures_k"
                    ]
                )
                - KELVIN_OFFSET
            ),
            float(result["cluster_supply_temperature_k"] - KELVIN_OFFSET),
            float(result["tank_temperature_after_k"] - KELVIN_OFFSET),
            float(result["q_evap_applied_w"]),
        )
    )
print("probe done")
