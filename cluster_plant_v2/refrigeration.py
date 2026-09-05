"""Cluster refrigeration cycle, actuator, evaporator, and transport delay.

PyCharm navigation:
- Conservative R134a cycle: ``ClosedR134aCycle``
- Compressor dynamics: ``CompressorSpeedActuator``
- Evaporator dynamics: ``EvaporatorThermalDynamics``
- Supply/return delay: ``CoolantTransportDelay``
- Shared defaults: ``parameters.py``
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Iterable

import CoolProp.CoolProp as CP
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares

from cluster_plant_v2.parameters import (
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    EVAPORATOR_TIME_CONSTANT_S,
    MAXIMUM_COMPRESSOR_SPEED_RPM,
    MAXIMUM_FAN_SPEED_RPM,
    MINIMUM_COMPRESSOR_SPEED_RPM,
    NOMINAL_COOLANT_MASS_FLOW_KG_S,
)


REFRIGERANT = "R134a"

COMPRESSOR_SPEED_AXIS_RPM = np.array([2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
COMPRESSOR_PRESSURE_RATIO_AXIS = np.array([4.0, 5.0, 6.4, 8.0, 10.0])
COMPRESSOR_VOLUMETRIC_EFFICIENCY = np.array(
    [
        [0.88, 0.83, 0.76, 0.69, 0.54],
        [0.94, 0.91, 0.87, 0.82, 0.74],
        [0.96, 0.94, 0.91, 0.86, 0.80],
        [0.98, 0.96, 0.94, 0.89, 0.84],
        [0.98, 0.96, 0.95, 0.90, 0.87],
    ]
)
COMPRESSOR_ISENTROPIC_EFFICIENCY = np.array(
    [
        [0.63, 0.57, 0.50, 0.43, 0.37],
        [0.70, 0.63, 0.59, 0.54, 0.46],
        [0.70, 0.62, 0.62, 0.57, 0.51],
        [0.68, 0.64, 0.64, 0.58, 0.54],
        [0.68, 0.65, 0.63, 0.60, 0.56],
    ]
)

# Stage 8D4 full rescale: the RegD full-deviation charge segment produces
# ~17.4 kW cluster heat (3486 W/pack x 5, Plant-measured at -1120 A = 1C),
# which exceeded even the Stage 8D3 capacity (10.78 kW @ 6000 rpm, and only
# 9.5 kW at the spike-time tank temperature 20 C). The scan gated capacity
# at Tin=20 C >= 17.5 kW with Tcond <= 53 C and picked the 72 cc family
# (condenser 8 m2 and air 6.5 kg/s are hard gates); evaporator 5.5 m2 keeps
# Tevap margin ~1.6 K above the 7 C floor. Scan:
# validation/results/stage8d4_full_rescale/.
COMPRESSOR_DISPLACEMENT_M3_PER_REV = 72.0e-6
COMPRESSOR_MECHANICAL_EFFICIENCY = 0.9134570767518371
TARGET_SUPERHEAT_K = 5.0
TARGET_SUBCOOLING_K = 5.0
MINIMUM_EVAPORATING_SATURATION_TEMPERATURE_K = 280.15
MAXIMUM_CONDENSING_SATURATION_TEMPERATURE_K = 328.15

EVAPORATOR_AREA_M2 = 5.5
EVAPORATOR_COOLANT_H_NOMINAL_W_M2_K = 2000.0
EVAPORATOR_REFRIGERANT_H_NOMINAL_W_M2_K = 10000.0
EVAPORATOR_UA_FACTOR = 2.0
EVAPORATOR_UA_SCALE = 0.70
EVAPORATOR_COOLANT_FLOW_EXPONENT = 1.2
NOMINAL_REFRIGERANT_MASS_FLOW_KG_S = 0.02

CONDENSER_AREA_M2 = 8.0
CONDENSER_AIR_H_NOMINAL_W_M2_K = 800.0
CONDENSER_REFRIGERANT_H_NOMINAL_W_M2_K = 5000.0
AIR_SPECIFIC_HEAT_J_KG_K = 1005.0
NOMINAL_AIR_MASS_FLOW_KG_S = 6.5


_VOLUMETRIC_EFFICIENCY_INTERPOLATOR = RegularGridInterpolator(
    (COMPRESSOR_SPEED_AXIS_RPM, COMPRESSOR_PRESSURE_RATIO_AXIS),
    COMPRESSOR_VOLUMETRIC_EFFICIENCY,
    bounds_error=False,
    fill_value=None,
)
_ISENTROPIC_EFFICIENCY_INTERPOLATOR = RegularGridInterpolator(
    (COMPRESSOR_SPEED_AXIS_RPM, COMPRESSOR_PRESSURE_RATIO_AXIS),
    COMPRESSOR_ISENTROPIC_EFFICIENCY,
    bounds_error=False,
    fill_value=None,
)


@dataclass(frozen=True)
class RefrigerantState:
    pressure_pa: float
    temperature_k: float
    enthalpy_j_kg: float
    entropy_j_kg_k: float
    density_kg_m3: float
    quality: float | None
    phase: str


def _checked_property(
    output: str,
    input_1: str,
    value_1: float,
    input_2: str,
    value_2: float,
) -> float:
    try:
        value = float(
            CP.PropsSI(
                output,
                input_1,
                float(value_1),
                input_2,
                float(value_2),
                REFRIGERANT,
            )
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"invalid R134a property state for {output}") from exc
    if not np.isfinite(value):
        raise RuntimeError(f"non-finite R134a property state for {output}")
    return value


def _quality_or_none(pressure_pa: float, enthalpy_j_kg: float) -> float | None:
    quality = _checked_property(
        "Q", "P", pressure_pa, "Hmass", enthalpy_j_kg
    )
    if 0.0 <= quality <= 1.0:
        return quality
    return None


def _state_from_pressure_temperature(
    pressure_pa: float, temperature_k: float
) -> RefrigerantState:
    enthalpy = _checked_property(
        "Hmass", "P", pressure_pa, "T", temperature_k
    )
    return RefrigerantState(
        pressure_pa=float(pressure_pa),
        temperature_k=float(temperature_k),
        enthalpy_j_kg=enthalpy,
        entropy_j_kg_k=_checked_property(
            "Smass", "P", pressure_pa, "T", temperature_k
        ),
        density_kg_m3=_checked_property(
            "Dmass", "P", pressure_pa, "T", temperature_k
        ),
        quality=_quality_or_none(pressure_pa, enthalpy),
        phase=str(CP.PhaseSI("P", pressure_pa, "T", temperature_k, REFRIGERANT)),
    )


def _state_from_pressure_enthalpy(
    pressure_pa: float, enthalpy_j_kg: float
) -> RefrigerantState:
    temperature = _checked_property(
        "T", "P", pressure_pa, "Hmass", enthalpy_j_kg
    )
    return RefrigerantState(
        pressure_pa=float(pressure_pa),
        temperature_k=temperature,
        enthalpy_j_kg=float(enthalpy_j_kg),
        entropy_j_kg_k=_checked_property(
            "Smass", "P", pressure_pa, "Hmass", enthalpy_j_kg
        ),
        density_kg_m3=_checked_property(
            "Dmass", "P", pressure_pa, "Hmass", enthalpy_j_kg
        ),
        quality=_quality_or_none(pressure_pa, enthalpy_j_kg),
        phase=str(
            CP.PhaseSI("P", pressure_pa, "Hmass", enthalpy_j_kg, REFRIGERANT)
        ),
    )


def _compressor_efficiencies(
    speed_rpm: float, pressure_ratio: float
) -> tuple[float, float]:
    map_pressure_ratio = float(
        np.clip(
            pressure_ratio,
            COMPRESSOR_PRESSURE_RATIO_AXIS[0],
            COMPRESSOR_PRESSURE_RATIO_AXIS[-1],
        )
    )
    point = np.array([float(speed_rpm), map_pressure_ratio])
    eta_vol = float(_VOLUMETRIC_EFFICIENCY_INTERPOLATOR(point)[0])
    eta_is = float(_ISENTROPIC_EFFICIENCY_INTERPOLATOR(point)[0])
    return (
        float(np.clip(eta_vol, 0.25, 0.99)),
        float(np.clip(eta_is, 0.20, 0.80)),
    )


def _series_ua(
    hot_side_h_w_m2_k: float,
    cold_side_h_w_m2_k: float,
    area_m2: float,
) -> float:
    return 1.0 / (
        1.0 / (hot_side_h_w_m2_k * area_m2)
        + 1.0 / (cold_side_h_w_m2_k * area_m2)
    )


class ClosedR134aCycle:
    """Four-state, quasi-steady R134a vapor-compression cycle."""

    dynamic_state_count = 0

    def evaluate_operating_point(
        self,
        *,
        compressor_speed_rpm: float,
        fan_speed_rpm: float,
        coolant_inlet_temperature_k: float,
        coolant_mass_flow_kg_s: float,
        ambient_temperature_k: float,
        evaporating_saturation_temperature_k: float,
        condensing_saturation_temperature_k: float,
        refrigerant_mass_flow_kg_s: float | None = None,
    ) -> dict[str, object]:
        values = np.array(
            [
                compressor_speed_rpm,
                fan_speed_rpm,
                coolant_inlet_temperature_k,
                coolant_mass_flow_kg_s,
                ambient_temperature_k,
                evaporating_saturation_temperature_k,
                condensing_saturation_temperature_k,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("cycle inputs must be finite")
        if refrigerant_mass_flow_kg_s is not None and (
            not np.isfinite(refrigerant_mass_flow_kg_s)
            or refrigerant_mass_flow_kg_s <= 0.0
        ):
            raise ValueError("refrigerant_mass_flow_kg_s must be finite and positive")
        if not MINIMUM_COMPRESSOR_SPEED_RPM <= compressor_speed_rpm <= MAXIMUM_COMPRESSOR_SPEED_RPM:
            raise ValueError("compressor_speed_rpm must be between 1000 and 6000")
        if not 0.0 < fan_speed_rpm <= MAXIMUM_FAN_SPEED_RPM:
            raise ValueError("fan_speed_rpm must be between zero and 4000")
        if coolant_mass_flow_kg_s <= 0.0:
            raise ValueError("coolant_mass_flow_kg_s must be positive")
        if evaporating_saturation_temperature_k >= coolant_inlet_temperature_k:
            raise ValueError("evaporating temperature must be below coolant inlet")
        if condensing_saturation_temperature_k <= ambient_temperature_k:
            raise ValueError("condensing temperature must exceed ambient")
        if condensing_saturation_temperature_k <= evaporating_saturation_temperature_k:
            raise ValueError("condensing temperature must exceed evaporating temperature")

        evaporating_pressure = _checked_property(
            "P",
            "T",
            evaporating_saturation_temperature_k,
            "Q",
            1.0,
        )
        condensing_pressure = _checked_property(
            "P",
            "T",
            condensing_saturation_temperature_k,
            "Q",
            0.0,
        )

        state_1 = _state_from_pressure_temperature(
            evaporating_pressure,
            evaporating_saturation_temperature_k + TARGET_SUPERHEAT_K,
        )
        pressure_ratio = condensing_pressure / evaporating_pressure
        eta_vol, eta_is = _compressor_efficiencies(
            compressor_speed_rpm, pressure_ratio
        )
        compressor_mass_flow = (
            state_1.density_kg_m3
            * eta_vol
            * COMPRESSOR_DISPLACEMENT_M3_PER_REV
            * compressor_speed_rpm
            / 60.0
        )
        refrigerant_mass_flow = (
            compressor_mass_flow
            if refrigerant_mass_flow_kg_s is None
            else float(refrigerant_mass_flow_kg_s)
        )
        isentropic_outlet_enthalpy = _checked_property(
            "Hmass",
            "P",
            condensing_pressure,
            "Smass",
            state_1.entropy_j_kg_k,
        )
        state_2 = _state_from_pressure_enthalpy(
            condensing_pressure,
            state_1.enthalpy_j_kg
            + (isentropic_outlet_enthalpy - state_1.enthalpy_j_kg) / eta_is,
        )
        state_3 = _state_from_pressure_temperature(
            condensing_pressure,
            condensing_saturation_temperature_k - TARGET_SUBCOOLING_K,
        )
        state_4 = _state_from_pressure_enthalpy(
            evaporating_pressure, state_3.enthalpy_j_kg
        )

        q_evaporator_refrigerant = refrigerant_mass_flow * (
            state_1.enthalpy_j_kg - state_4.enthalpy_j_kg
        )
        compressor_refrigerant_power = refrigerant_mass_flow * (
            state_2.enthalpy_j_kg - state_1.enthalpy_j_kg
        )
        compressor_shaft_power = (
            compressor_refrigerant_power / COMPRESSOR_MECHANICAL_EFFICIENCY
        )
        q_condenser_refrigerant = refrigerant_mass_flow * (
            state_2.enthalpy_j_kg - state_3.enthalpy_j_kg
        )

        coolant_capacity_rate = (
            coolant_mass_flow_kg_s * COOLANT_SPECIFIC_HEAT_J_KG_K
        )
        evaporator_coolant_h = max(
            50.0,
            EVAPORATOR_UA_FACTOR
            * EVAPORATOR_UA_SCALE
            * EVAPORATOR_COOLANT_H_NOMINAL_W_M2_K
            * (coolant_mass_flow_kg_s / NOMINAL_COOLANT_MASS_FLOW_KG_S)
            ** EVAPORATOR_COOLANT_FLOW_EXPONENT,
        )
        evaporator_refrigerant_h = max(
            500.0,
            EVAPORATOR_UA_FACTOR
            * EVAPORATOR_UA_SCALE
            * EVAPORATOR_REFRIGERANT_H_NOMINAL_W_M2_K
            * (refrigerant_mass_flow / NOMINAL_REFRIGERANT_MASS_FLOW_KG_S)
            ** 0.8,
        )
        evaporator_ua = _series_ua(
            evaporator_coolant_h,
            evaporator_refrigerant_h,
            EVAPORATOR_AREA_M2,
        )
        evaporator_effectiveness = 1.0 - np.exp(
            -evaporator_ua / coolant_capacity_rate
        )
        q_evaporator_ntu = (
            evaporator_effectiveness
            * coolant_capacity_rate
            * (coolant_inlet_temperature_k - evaporating_saturation_temperature_k)
        )

        air_mass_flow = (
            NOMINAL_AIR_MASS_FLOW_KG_S
            * fan_speed_rpm
            / MAXIMUM_FAN_SPEED_RPM
        )
        air_capacity_rate = air_mass_flow * AIR_SPECIFIC_HEAT_J_KG_K
        condenser_air_h = max(
            5.0,
            CONDENSER_AIR_H_NOMINAL_W_M2_K
            * (air_mass_flow / NOMINAL_AIR_MASS_FLOW_KG_S) ** 0.8,
        )
        condenser_refrigerant_h = max(
            500.0,
            CONDENSER_REFRIGERANT_H_NOMINAL_W_M2_K
            * (refrigerant_mass_flow / NOMINAL_REFRIGERANT_MASS_FLOW_KG_S)
            ** 0.8,
        )
        condenser_ua = _series_ua(
            condenser_air_h,
            condenser_refrigerant_h,
            CONDENSER_AREA_M2,
        )
        condenser_effectiveness = 1.0 - np.exp(
            -condenser_ua / air_capacity_rate
        )
        q_condenser_ntu = (
            condenser_effectiveness
            * air_capacity_rate
            * (condensing_saturation_temperature_k - ambient_temperature_k)
        )

        return {
            "state_1": state_1,
            "state_2": state_2,
            "state_3": state_3,
            "state_4": state_4,
            "evaporator_outlet_state": state_1,
            "compressor_inlet_state": state_1,
            "refrigerant_mass_flow_kg_s": refrigerant_mass_flow,
            "compressor_mass_flow_kg_s": compressor_mass_flow,
            "mass_flow_residual_kg_s": (
                refrigerant_mass_flow - compressor_mass_flow
            ),
            "compressor_volumetric_efficiency": eta_vol,
            "compressor_isentropic_efficiency": eta_is,
            "compressor_refrigerant_power_w": compressor_refrigerant_power,
            "compressor_shaft_power_w": compressor_shaft_power,
            "compressor_mechanical_loss_w": (
                compressor_shaft_power - compressor_refrigerant_power
            ),
            "q_evaporator_refrigerant_w": q_evaporator_refrigerant,
            "q_evaporator_ntu_w": q_evaporator_ntu,
            "evaporator_residual_w": (
                q_evaporator_refrigerant - q_evaporator_ntu
            ),
            "q_condenser_refrigerant_w": q_condenser_refrigerant,
            "q_condenser_ntu_w": q_condenser_ntu,
            "condenser_residual_w": q_condenser_refrigerant - q_condenser_ntu,
            "cycle_energy_residual_w": (
                q_condenser_refrigerant
                - q_evaporator_refrigerant
                - compressor_refrigerant_power
            ),
            "coolant_outlet_temperature_k": (
                coolant_inlet_temperature_k
                - q_evaporator_refrigerant / coolant_capacity_rate
            ),
            "air_outlet_temperature_k": (
                ambient_temperature_k
                + q_condenser_refrigerant / air_capacity_rate
            ),
            "evaporator_ua_w_k": evaporator_ua,
            "condenser_ua_w_k": condenser_ua,
        }

    def solve(
        self,
        *,
        compressor_speed_rpm: float,
        fan_speed_rpm: float,
        coolant_inlet_temperature_k: float,
        coolant_mass_flow_kg_s: float,
        ambient_temperature_k: float,
    ) -> dict[str, object]:
        """Solve evaporating temperature, condensing temperature, and flow.

        The ideal thermostatic expansion valve meters the same mass flow as the
        compressor while preserving ``h4 == h3``.  No uncalibrated valve-flow
        coefficient is introduced.
        """

        evaporating_upper = coolant_inlet_temperature_k - 0.5
        condensing_lower = ambient_temperature_k + 0.5
        if evaporating_upper <= MINIMUM_EVAPORATING_SATURATION_TEMPERATURE_K:
            raise ValueError("coolant inlet is too cold for the audited cycle envelope")
        if condensing_lower >= MAXIMUM_CONDENSING_SATURATION_TEMPERATURE_K:
            raise ValueError("ambient is too hot for the audited cycle envelope")

        initial_evaporating_temperature = float(
            np.clip(
                coolant_inlet_temperature_k - 10.0,
                MINIMUM_EVAPORATING_SATURATION_TEMPERATURE_K,
                evaporating_upper,
            )
        )
        initial_condensing_temperature = float(
            np.clip(
                ambient_temperature_k + 15.0,
                condensing_lower,
                MAXIMUM_CONDENSING_SATURATION_TEMPERATURE_K,
            )
        )
        initial = self.evaluate_operating_point(
            compressor_speed_rpm=compressor_speed_rpm,
            fan_speed_rpm=fan_speed_rpm,
            coolant_inlet_temperature_k=coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=coolant_mass_flow_kg_s,
            ambient_temperature_k=ambient_temperature_k,
            evaporating_saturation_temperature_k=(
                initial_evaporating_temperature
            ),
            condensing_saturation_temperature_k=(
                initial_condensing_temperature
            ),
        )
        initial_mass_flow = float(initial["compressor_mass_flow_kg_s"])

        def normalized_residuals(unknowns: np.ndarray) -> np.ndarray:
            evaporating_temperature, condensing_temperature, mass_flow = unknowns
            point = self.evaluate_operating_point(
                compressor_speed_rpm=compressor_speed_rpm,
                fan_speed_rpm=fan_speed_rpm,
                coolant_inlet_temperature_k=coolant_inlet_temperature_k,
                coolant_mass_flow_kg_s=coolant_mass_flow_kg_s,
                ambient_temperature_k=ambient_temperature_k,
                evaporating_saturation_temperature_k=evaporating_temperature,
                condensing_saturation_temperature_k=condensing_temperature,
                refrigerant_mass_flow_kg_s=mass_flow,
            )
            mass_scale = max(
                abs(float(point["compressor_mass_flow_kg_s"])), 1e-3
            )
            evaporator_scale = max(
                abs(float(point["q_evaporator_refrigerant_w"])),
                abs(float(point["q_evaporator_ntu_w"])),
                100.0,
            )
            condenser_scale = max(
                abs(float(point["q_condenser_refrigerant_w"])),
                abs(float(point["q_condenser_ntu_w"])),
                100.0,
            )
            return np.array(
                [
                    float(point["mass_flow_residual_kg_s"]) / mass_scale,
                    float(point["evaporator_residual_w"]) / evaporator_scale,
                    float(point["condenser_residual_w"]) / condenser_scale,
                ]
            )

        solution = least_squares(
            normalized_residuals,
            x0=np.array(
                [
                    initial_evaporating_temperature,
                    initial_condensing_temperature,
                    initial_mass_flow,
                ]
            ),
            bounds=(
                np.array(
                    [
                        MINIMUM_EVAPORATING_SATURATION_TEMPERATURE_K,
                        condensing_lower,
                        1e-6,
                    ]
                ),
                np.array(
                    [
                        evaporating_upper,
                        MAXIMUM_CONDENSING_SATURATION_TEMPERATURE_K,
                        0.2,
                    ]
                ),
            ),
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-11,
            max_nfev=300,
        )
        evaporating_temperature, condensing_temperature, mass_flow = solution.x
        point = self.evaluate_operating_point(
            compressor_speed_rpm=compressor_speed_rpm,
            fan_speed_rpm=fan_speed_rpm,
            coolant_inlet_temperature_k=coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=coolant_mass_flow_kg_s,
            ambient_temperature_k=ambient_temperature_k,
            evaporating_saturation_temperature_k=evaporating_temperature,
            condensing_saturation_temperature_k=condensing_temperature,
            refrigerant_mass_flow_kg_s=mass_flow,
        )

        mass_relative_residual = abs(float(point["mass_flow_residual_kg_s"])) / max(
            abs(float(point["compressor_mass_flow_kg_s"])), 1e-12
        )
        evaporator_relative_residual = abs(
            float(point["evaporator_residual_w"])
        ) / max(
            abs(float(point["q_evaporator_refrigerant_w"])),
            abs(float(point["q_evaporator_ntu_w"])),
            1e-12,
        )
        condenser_relative_residual = abs(
            float(point["condenser_residual_w"])
        ) / max(
            abs(float(point["q_condenser_refrigerant_w"])),
            abs(float(point["q_condenser_ntu_w"])),
            1e-12,
        )
        cycle_energy_relative_residual = abs(
            float(point["cycle_energy_residual_w"])
        ) / max(abs(float(point["q_condenser_refrigerant_w"])), 1e-12)
        gates_pass = bool(
            solution.success
            and mass_relative_residual < 1e-3
            and evaporator_relative_residual < 5e-3
            and condenser_relative_residual < 5e-3
            and cycle_energy_relative_residual < 5e-3
        )
        solver_message = str(solution.message)
        if solution.success and not gates_pass:
            solver_message = "numerical solver converged but physical closure gates failed"

        return {
            **point,
            "solver_success": gates_pass,
            "solver_message": solver_message,
            "solver_function_evaluations": int(solution.nfev),
            "evaporating_saturation_temperature_k": float(
                evaporating_temperature
            ),
            "condensing_saturation_temperature_k": float(
                condensing_temperature
            ),
            "mass_flow_relative_residual": mass_relative_residual,
            "evaporator_relative_residual": evaporator_relative_residual,
            "condenser_relative_residual": condenser_relative_residual,
            "cycle_energy_relative_residual": cycle_energy_relative_residual,
            "q_evaporator_w": float(point["q_evaporator_refrigerant_w"]),
            "q_condenser_w": float(point["q_condenser_refrigerant_w"]),
        }


# Compressor Actuator


class CompressorSpeedActuator:
    """Advance one actual-speed state with an exact first-order response."""

    dynamic_state_count = 1

    def __init__(self, *, initial_speed_rpm: float, time_constant_s: float) -> None:
        self._validate_speed(initial_speed_rpm)
        if not math.isfinite(time_constant_s) or time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive and finite")
        self.speed_rpm = float(initial_speed_rpm)
        self.time_constant_s = float(time_constant_s)

    @staticmethod
    def _validate_speed(speed_rpm: float) -> None:
        if not math.isfinite(speed_rpm):
            raise ValueError("compressor speed must be finite")
        if speed_rpm == 0.0:
            # Stage 8D4d: an explicit 0 rpm command means "compressor off"
            # (thermostat cycling of an oversized unit at low load). The
            # first-order lag then spools the shaft down toward 0.
            return
        if (
            speed_rpm < MINIMUM_COMPRESSOR_SPEED_RPM
            or speed_rpm > MAXIMUM_COMPRESSOR_SPEED_RPM
        ):
            raise ValueError(
                "compressor speed must be 0 or between 1000 and 6000 rpm"
            )

    def step(self, *, dt_s: float, speed_command_rpm: float) -> dict[str, float]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        self._validate_speed(speed_command_rpm)

        speed_before = self.speed_rpm
        decay = math.exp(-float(dt_s) / self.time_constant_s)
        self.speed_rpm = float(
            speed_command_rpm + (speed_before - speed_command_rpm) * decay
        )
        return {
            "speed_before_rpm": speed_before,
            "speed_command_rpm": float(speed_command_rpm),
            "speed_after_rpm": self.speed_rpm,
        }


# Evaporator Thermal Dynamics


class EvaporatorThermalDynamics:
    """Lag coolant-side heat extraction behind the steady cycle target."""

    dynamic_state_count = 1
    diagnostic_state_count = 1

    def __init__(
        self,
        *,
        initial_q_evap_applied_w: float,
        time_constant_s: float = EVAPORATOR_TIME_CONSTANT_S,
    ) -> None:
        if (
            not math.isfinite(initial_q_evap_applied_w)
            or initial_q_evap_applied_w < 0.0
        ):
            raise ValueError(
                "initial_q_evap_applied_w must be finite and nonnegative"
            )
        if not math.isfinite(time_constant_s) or time_constant_s <= 0.0:
            raise ValueError("time_constant_s must be positive and finite")
        self.q_evap_applied_w = float(initial_q_evap_applied_w)
        self.time_constant_s = float(time_constant_s)
        self.evaporator_buffer_energy_j = 0.0

    def step(self, *, dt_s: float, q_evap_cycle_w: float) -> dict[str, float]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        if not math.isfinite(q_evap_cycle_w) or q_evap_cycle_w < 0.0:
            raise ValueError("q_evap_cycle_w must be finite and nonnegative")

        q_before = self.q_evap_applied_w
        buffer_before = self.evaporator_buffer_energy_j
        decay = math.exp(-float(dt_s) / self.time_constant_s)
        self.q_evap_applied_w = float(
            q_evap_cycle_w + (q_before - q_evap_cycle_w) * decay
        )
        expected_buffer_change = (
            self.q_evap_applied_w - q_evap_cycle_w
        ) * float(dt_s)
        self.evaporator_buffer_energy_j = float(
            buffer_before + expected_buffer_change
        )
        actual_buffer_change = (
            self.evaporator_buffer_energy_j - buffer_before
        )

        return {
            "q_evap_applied_before_w": q_before,
            "q_evap_cycle_w": float(q_evap_cycle_w),
            "q_evap_applied_w": self.q_evap_applied_w,
            "evaporator_buffer_energy_before_j": buffer_before,
            "evaporator_buffer_energy_change_j": actual_buffer_change,
            "evaporator_buffer_energy_j": self.evaporator_buffer_energy_j,
            "evaporator_dynamic_energy_residual_j": (
                actual_buffer_change - expected_buffer_change
            ),
        }


# Coolant Transport Delay


class CoolantTransportDelay:
    """Return ``input[k-delay_steps]`` using a fixed-length FIFO."""

    def __init__(
        self,
        *,
        delay_s: float,
        dt_s: float,
        initial_value: float | None = None,
        initial_queue_values: Iterable[float] | None = None,
    ) -> None:
        if not math.isfinite(delay_s) or delay_s <= 0.0:
            raise ValueError("delay_s must be positive and finite")
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be positive and finite")
        ratio = float(delay_s) / float(dt_s)
        steps = int(round(ratio))
        if not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("delay_s must be an integer multiple of dt_s")
        if (initial_value is None) == (initial_queue_values is None):
            raise ValueError(
                "provide exactly one of initial_value or initial_queue_values"
            )

        if initial_queue_values is None:
            values = [float(initial_value)] * steps
        else:
            values = [float(value) for value in initial_queue_values]
            if len(values) != steps:
                raise ValueError(
                    f"initial_queue_values must contain exactly {steps} values"
                )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("initial delay values must be finite")

        self.delay_s = float(delay_s)
        self.dt_s = float(dt_s)
        self.delay_steps = steps
        self.dynamic_state_count = steps
        self._queue = deque(values, maxlen=steps)

    @property
    def queue_values(self) -> tuple[float, ...]:
        return tuple(self._queue)

    def step(self, input_value: float) -> float:
        value = float(input_value)
        if not math.isfinite(value):
            raise ValueError("transport-delay input must be finite")
        output = float(self._queue.popleft())
        self._queue.append(value)
        return output
