"""Physics-based reduced-order cluster model for do-mpc prediction.

Continuous-time dynamics (all temperatures in kelvin):

- compressor speed actuator: first-order lag, ``COMPRESSOR_TIME_CONSTANT_S``
- evaporator applied cooling: first-order lag toward a quadratic cycle
  surrogate, ``EVAPORATOR_TIME_CONSTANT_S``
- supply/return transport delays: chains of 5 s first-order lags (3 and 4
  lags reproduce the frozen 15 s / 20 s delays)
- cluster thermal chain: lumped battery capacitance -> lumped cold-plate
  capacitance -> well-mixed coolant
- tank: well-mixed mixing of the delayed return stream

The quadratic cycle surrogate coefficients come from
``cluster_plant_v2.control.chiller_surrogates``. State extraction maps real
``ClusterPlantOutputs`` onto this reduced state vector for closed-loop use.
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca
import do_mpc
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cluster_plant_v2.control.chiller_surrogates import (
    FEATURE_NAMES,
    load_surrogates,
)
from cluster_plant_v2.hppc_parameters import load_hppc_parameters
from cluster_plant_v2.parameters import (
    BATTERY_PLATE_AREA_M2,
    BATTERY_PLATE_NOMINAL_HTC_W_M2_K,
    COMPRESSOR_TIME_CONSTANT_S,
    COOLANT_DENSITY_KG_M3,
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    EVAPORATOR_TIME_CONSTANT_S,
    LEGACY_REFERENCE_POWER_W,
    LEGACY_REFERENCE_SPEED_RPM,
    LEGACY_TANK_VOLUME_L,
    RETURN_TRANSPORT_DELAY_S,
    SIMULATION_TIME_STEP_S,
    SUPPLY_TRANSPORT_DELAY_S,
)


SUPPLY_LAG_COUNT = int(round(SUPPLY_TRANSPORT_DELAY_S / SIMULATION_TIME_STEP_S))
RETURN_LAG_COUNT = int(round(RETURN_TRANSPORT_DELAY_S / SIMULATION_TIME_STEP_S))
LAG_TIME_CONSTANT_S = SIMULATION_TIME_STEP_S

STATE_NAMES = (
    ["omega_c", "q_evap"]
    + [f"t_sup_{i}" for i in range(1, SUPPLY_LAG_COUNT + 1)]
    + [f"t_ret_{i}" for i in range(1, RETURN_LAG_COUNT + 1)]
    + ["t_battery", "t_plate", "t_tank"]
)


@dataclass(frozen=True)
class ClusterModelCoefficients:
    """Frozen lumped parameters derived once from a built ClusterPlant."""

    k_flow_kg_s_per_rpm: float
    c_battery_j_k: float
    c_plate_j_k: float
    c_tank_j_k: float
    h_battery_plate_w_k: float
    q_gen_coeff_w_a2: float
    q_evap_coeffs: tuple[float, ...]
    shaft_power_coeffs: tuple[float, ...]


def _surface_expression(
    coefficients: tuple[float, ...] | list[float],
    omega,
    mass_flow,
    tank_temperature,
):
    """Evaluate the 8-feature quadratic surrogate as a CasADi expression."""
    features = [
        1.0,
        omega,
        mass_flow,
        tank_temperature,
        omega * omega,
        tank_temperature * tank_temperature,
        omega * mass_flow,
        omega * tank_temperature,
    ]
    if len(coefficients) != len(FEATURE_NAMES):
        raise ValueError("surrogate coefficients must have 8 entries")
    return sum(coefficient * feature
               for coefficient, feature in zip(coefficients, features))


def derive_coefficients(
    plant,
    *,
    nominal_pump_speed_rpm: float,
    hppc_soc: float = 0.95,
    hppc_temperature_k: float = 298.15,
    surrogates: dict[str, object] | None = None,
) -> ClusterModelCoefficients:
    """Extract lumped model constants from an already-built ClusterPlant."""
    operating_point = plant.pump.solve_operating_point(
        nominal_pump_speed_rpm, plant.hydraulic_network
    )
    total_flow = float(operating_point["total_mass_flow_kg_s"])
    if total_flow <= 0.0:
        raise ValueError("nominal pump operating point must have positive flow")
    k_flow = total_flow / float(nominal_pump_speed_rpm)

    c_battery = float(
        sum(
            float(np.sum(pack.battery.zone_heat_capacities))
            for pack in plant.cluster.packs
        )
    )
    c_plate = float(
        sum(
            float(np.sum(pack.cold_plate.zone_heat_capacities))
            for pack in plant.cluster.packs
        )
    )
    c_tank = float(
        COOLANT_DENSITY_KG_M3
        * (LEGACY_TANK_VOLUME_L * 1e-3)
        * COOLANT_SPECIFIC_HEAT_J_KG_K
    )
    h_battery_plate = float(
        plant.cluster.n_packs
        * BATTERY_PLATE_AREA_M2
        * BATTERY_PLATE_NOMINAL_HTC_W_M2_K
    )

    hppc = load_hppc_parameters()
    r0_dis = RegularGridInterpolator(
        (hppc["soc"], hppc["temp"]), hppc["r0_dis"]
    )((hppc_soc, hppc_temperature_k))
    q_gen_coeff = float(plant.cluster.n_packs) * float(r0_dis)

    if surrogates is None:
        surrogates = load_surrogates()
    q_evap_coeffs = tuple(surrogates["coefficients"]["q_evap_cycle_w"])
    shaft_power_coeffs = tuple(
        surrogates["coefficients"]["compressor_shaft_power_w"]
    )

    return ClusterModelCoefficients(
        k_flow_kg_s_per_rpm=k_flow,
        c_battery_j_k=c_battery,
        c_plate_j_k=c_plate,
        c_tank_j_k=c_tank,
        h_battery_plate_w_k=h_battery_plate,
        q_gen_coeff_w_a2=q_gen_coeff,
        q_evap_coeffs=q_evap_coeffs,
        shaft_power_coeffs=shaft_power_coeffs,
    )


def pump_power_w(pump_speed_rpm) -> ca.SX:
    """Cubic affinity-law pump power anchored at the frozen reference point."""
    ratio = pump_speed_rpm / LEGACY_REFERENCE_SPEED_RPM
    return LEGACY_REFERENCE_POWER_W * ratio**3


def build_cluster_model(
    coefficients: ClusterModelCoefficients,
) -> do_mpc.model.Model:
    """Assemble the do-mpc continuous-time model with surrogate dynamics."""
    model = do_mpc.model.Model(model_type="continuous")

    omega_c = model.set_variable("_x", "omega_c")
    q_evap = model.set_variable("_x", "q_evap")
    t_sup = [model.set_variable("_x", name)
             for name in STATE_NAMES[2:2 + SUPPLY_LAG_COUNT]]
    t_ret = [
        model.set_variable("_x", name)
        for name in STATE_NAMES[2 + SUPPLY_LAG_COUNT:-3]
    ]
    t_battery = model.set_variable("_x", "t_battery")
    t_plate = model.set_variable("_x", "t_plate")
    t_tank = model.set_variable("_x", "t_tank")

    comp_cmd = model.set_variable("_u", "comp_cmd_rpm")
    pump_rpm = model.set_variable("_u", "pump_rpm")
    current_a = model.set_variable("_tvp", "current_a")
    t_amb = model.set_variable("_tvp", "t_amb")

    mass_flow = coefficients.k_flow_kg_s_per_rpm * pump_rpm
    coolant_capacity = mass_flow * COOLANT_SPECIFIC_HEAT_J_KG_K

    q_evap_cycle = _surface_expression(
        coefficients.q_evap_coeffs, omega_c, mass_flow, t_tank
    )
    shaft_power = _surface_expression(
        coefficients.shaft_power_coeffs, omega_c, mass_flow, t_tank
    )
    model.set_expression("q_evap_cycle_w", q_evap_cycle)
    model.set_expression("compressor_shaft_power_w", shaft_power)
    model.set_expression("pump_power_w", pump_power_w(pump_rpm))
    model.set_expression("mass_flow_kg_s", mass_flow)

    # Actuator and evaporator lags.
    model.set_rhs(
        "omega_c",
        (comp_cmd - omega_c) / COMPRESSOR_TIME_CONSTANT_S,
    )
    model.set_rhs(
        "q_evap",
        (q_evap_cycle - q_evap) / EVAPORATOR_TIME_CONSTANT_S,
    )

    # Supply delay chain driven by the evaporator outlet temperature.
    t_evap_outlet = t_tank - q_evap / coolant_capacity
    previous = t_evap_outlet
    for index, lag in enumerate(t_sup):
        model.set_rhs(
            STATE_NAMES[2 + index],
            (previous - lag) / LAG_TIME_CONSTANT_S,
        )
        previous = lag
    t_supply = previous

    # Cluster thermal chain: battery -> plate -> well-mixed coolant.
    q_gen = coefficients.q_gen_coeff_w_a2 * current_a * current_a
    q_battery_to_plate = coefficients.h_battery_plate_w_k * (
        t_battery - t_plate
    )
    q_plate_to_fluid = coolant_capacity * (t_plate - t_supply)
    model.set_expression("q_gen_total_w", q_gen)
    model.set_expression("q_plate_to_fluid_w", q_plate_to_fluid)
    model.set_rhs(
        "t_battery",
        (q_gen - q_battery_to_plate) / coefficients.c_battery_j_k,
    )
    model.set_rhs(
        "t_plate",
        (q_battery_to_plate - q_plate_to_fluid)
        / coefficients.c_plate_j_k,
    )

    # Return delay chain driven by the cluster return temperature.
    previous = t_plate
    offset = 2 + SUPPLY_LAG_COUNT
    for index, lag in enumerate(t_ret):
        model.set_rhs(
            STATE_NAMES[offset + index],
            (previous - lag) / LAG_TIME_CONSTANT_S,
        )
        previous = lag
    t_return = previous

    model.set_rhs(
        "t_tank",
        coolant_capacity * (t_return - t_tank) / coefficients.c_tank_j_k,
    )

    model.set_expression("t_supply_k", t_supply)
    model.set_expression("t_return_k", t_return)
    model.set_expression("t_evap_outlet_k", t_evap_outlet)
    model.set_expression("t_amb_k", t_amb)

    model.setup()
    return model


def extract_x0(plant, result) -> np.ndarray:
    """Map one ClusterPlantOutputs snapshot onto the reduced state vector."""
    cluster_result = result["cluster_result"]
    supply_queue = np.asarray(result["supply_delay_queue_k"], dtype=float)
    return_queue = np.asarray(result["return_delay_queue_k"], dtype=float)
    if supply_queue.size != SUPPLY_LAG_COUNT:
        raise ValueError("supply delay queue length mismatch")
    if return_queue.size != RETURN_LAG_COUNT:
        raise ValueError("return delay queue length mismatch")

    t_battery = float(
        np.mean(cluster_result["pack_battery_average_temperatures_k"])
    )
    t_plate = float(np.mean(cluster_result["pack_plate_temperatures_k"]))

    values = {
        "omega_c": float(result["compressor_speed_rpm"]),
        "q_evap": float(result["q_evap_applied_w"]),
        "t_battery": t_battery,
        "t_plate": t_plate,
        "t_tank": float(result["tank_temperature_after_k"]),
    }
    # FIFO queues: index 0 is the oldest (most downstream) sample.
    for index in range(SUPPLY_LAG_COUNT):
        values[f"t_sup_{index + 1}"] = float(
            supply_queue[SUPPLY_LAG_COUNT - 1 - index]
        )
    for index in range(RETURN_LAG_COUNT):
        values[f"t_ret_{index + 1}"] = float(
            return_queue[RETURN_LAG_COUNT - 1 - index]
        )
    return np.asarray([values[name] for name in STATE_NAMES], dtype=float)
