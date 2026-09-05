"""Three-zone reduction of the existing thirteen-node cold plate."""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.parameters import (
    BATTERY_PLATE_AREA_M2 as A_bp_total,
    BATTERY_PLATE_NOMINAL_HTC_W_M2_K as h_bp_nominal,
    COLD_PLATE_REFERENCE_MASS_FLOW_KG_S as m_dot_nominal,
    COOLANT_DENSITY_KG_M3 as rho_cool,
    COOLANT_SPECIFIC_HEAT_J_KG_K as cp_cool,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
)


ZONE_COLUMN_COUNTS = np.array([4, 5, 4], dtype=int)


class ReducedColdPlate:
    """Cold plate with three serial coolant zones and three wall states."""

    dynamic_state_count = 3
    coolant_density = rho_cool
    coolant_cp = cp_cool

    def __init__(self, initial_temperature_c: float = 25.0) -> None:
        self.zone_column_counts = ZONE_COLUMN_COUNTS.copy()
        fractions = self.zone_column_counts.astype(float) / 13.0
        self.zone_heat_capacities = fractions * PLATE_NODE_HEAT_CAPACITY_TOTAL
        self.zone_areas = fractions * A_bp_total
        self.plate_temperatures = np.full(3, float(initial_temperature_c) + 273.15)
        self.initial_zone_energy_J = self.zone_heat_capacities * self.plate_temperatures
        self.coolant_outlet_temperature = float(self.plate_temperatures[0])
        self.coolant_mean_temperatures = self.plate_temperatures.copy()
        self.q_plate_to_fluid = np.zeros(3, dtype=float)
        self.q_plate_to_fluid_total = 0.0
        self.h_dynamic = 50.0

    @property
    def zone_energy_J(self) -> np.ndarray:
        return self.zone_heat_capacities * self.plate_temperatures

    @staticmethod
    def _robust_lmtd(delta_t_in: float, delta_t_out: float) -> float:
        if (
            abs(delta_t_in) < 1e-4
            or abs(delta_t_out) < 1e-4
            or abs(delta_t_in - delta_t_out) < 1e-4
            or delta_t_in * delta_t_out <= 0.0
        ):
            return 0.5 * (delta_t_in + delta_t_out)
        ratio = delta_t_in / delta_t_out
        logarithm = np.log(ratio)
        if not np.isfinite(logarithm) or abs(logarithm) < 1e-12:
            return 0.5 * (delta_t_in + delta_t_out)
        return float((delta_t_in - delta_t_out) / logarithm)

    def step(
        self,
        dt: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        q_battery_to_plate: np.ndarray,
        flow_direction: int = 1,
    ) -> dict[str, float | np.ndarray]:
        dt = float(dt)
        mass_flow = float(coolant_mass_flow)
        inlet = float(coolant_inlet_temperature)
        battery_heat = np.asarray(q_battery_to_plate, dtype=float)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if not np.isfinite(mass_flow) or mass_flow <= 0.0:
            raise ValueError("coolant_mass_flow must be positive and finite in kg/s")
        if not np.isfinite(inlet):
            raise ValueError("coolant_inlet_temperature must be finite")
        if battery_heat.shape != (3,) or not np.all(np.isfinite(battery_heat)):
            raise ValueError("q_battery_to_plate must be finite with shape (3,)")
        if flow_direction not in (-1, 1):
            raise ValueError("flow_direction must be 1 or -1")

        self.h_dynamic = max(
            50.0, h_bp_nominal * (mass_flow / m_dot_nominal) ** 0.8
        )
        traversal = range(3) if flow_direction == 1 else range(2, -1, -1)
        fluid_temperature = inlet
        fluid_means = np.empty(3, dtype=float)
        heat_to_fluid = np.empty(3, dtype=float)
        old_plate_temperatures = self.plate_temperatures.copy()

        for zone in traversal:
            wall_temperature = old_plate_temperatures[zone]
            delta_t_in = wall_temperature - fluid_temperature
            q_guess = self.h_dynamic * self.zone_areas[zone] * delta_t_in
            guessed_outlet = fluid_temperature + q_guess / (mass_flow * self.coolant_cp)
            delta_t_out = wall_temperature - guessed_outlet
            lmtd = self._robust_lmtd(delta_t_in, delta_t_out)
            zone_heat = self.h_dynamic * self.zone_areas[zone] * lmtd
            outlet = fluid_temperature + zone_heat / (mass_flow * self.coolant_cp)
            fluid_means[zone] = 0.5 * (fluid_temperature + outlet)
            heat_to_fluid[zone] = zone_heat
            fluid_temperature = outlet

        new_temperatures = old_plate_temperatures + dt * (
            battery_heat - heat_to_fluid
        ) / self.zone_heat_capacities
        if not (
            np.all(np.isfinite(new_temperatures))
            and np.all(np.isfinite(fluid_means))
            and np.all(np.isfinite(heat_to_fluid))
            and np.isfinite(fluid_temperature)
        ):
            raise FloatingPointError("ReducedColdPlate produced a non-finite state")

        self.plate_temperatures = new_temperatures
        self.coolant_outlet_temperature = float(fluid_temperature)
        self.coolant_mean_temperatures = fluid_means
        self.q_plate_to_fluid = heat_to_fluid
        self.q_plate_to_fluid_total = float(heat_to_fluid.sum())
        return {
            "plate_temperatures": self.plate_temperatures.copy(),
            "coolant_outlet_temperature": self.coolant_outlet_temperature,
            "coolant_mean_temperatures": self.coolant_mean_temperatures.copy(),
            "q_plate_to_fluid": self.q_plate_to_fluid.copy(),
            "q_plate_to_fluid_total": self.q_plate_to_fluid_total,
            "h_dynamic": self.h_dynamic,
        }
