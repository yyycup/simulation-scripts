"""Energy-conservative coupling of the frozen battery and cold-plate ROMs."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cluster_plant_v2.thermal.cold_plate_rom import ReducedColdPlate
from cluster_plant_v2.thermal.pack_rom import ReducedBatteryPack


class ReducedPack:
    """Coordinate one explicit Battery-ROM -> Cold-Plate-ROM time step."""

    def __init__(
        self,
        battery_config: Mapping[str, object] | None = None,
        initial_plate_temperature_c: float = 25.0,
    ) -> None:
        self.battery = ReducedBatteryPack(battery_config)
        self.cold_plate = ReducedColdPlate(initial_plate_temperature_c)
        self.dynamic_state_count = (
            self.battery.dynamic_state_count + self.cold_plate.dynamic_state_count
        )
        self.q_battery_to_plate_branch_zone = np.zeros((4, 3), dtype=float)
        self.q_battery_to_plate_zones = np.zeros(3, dtype=float)
        self.q_plate_gain_zones = np.zeros(3, dtype=float)
        self.coupling_energy_residual = np.zeros(3, dtype=float)
        self.whole_pack_energy_residual_J = 0.0
        self.whole_pack_energy_relative_error = 0.0

    def _stored_thermal_energy(self) -> float:
        battery_energy = np.sum(
            self.battery.zone_heat_capacities * self.battery.temps
        )
        plate_energy = np.sum(
            self.cold_plate.zone_heat_capacities
            * self.cold_plate.plate_temperatures
        )
        return float(battery_energy + plate_energy)

    def step(
        self,
        dt: float,
        total_current: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        ambient_temperature: float,
        flow_direction: int = 1,
    ) -> dict[str, float | np.ndarray]:
        """Advance the battery first and pass its exact plate heat to the plate."""
        old_energy = self._stored_thermal_energy()
        plate_temperatures_k = self.cold_plate.plate_temperatures.copy()
        self.battery.step(
            dt,
            total_current,
            plate_temperatures_k,
            ambient_temperature,
        )

        self.q_battery_to_plate_branch_zone = (
            self.battery.get_battery_to_plate_heat_branch_zone()
        )
        self.q_battery_to_plate_zones = self.battery.get_battery_to_plate_heat()
        plate_input = self.q_battery_to_plate_zones
        plate_result = self.cold_plate.step(
            dt,
            coolant_inlet_temperature,
            coolant_mass_flow,
            plate_input,
            flow_direction,
        )
        self.q_plate_gain_zones = plate_input.copy()
        self.coupling_energy_residual = (
            self.q_battery_to_plate_zones - self.q_plate_gain_zones
        )

        actual_delta_energy = self._stored_thermal_energy() - old_energy
        q_air_total = float(self.battery.get_air_heat_loss().sum())
        expected_delta_energy = float(dt) * (
            self.battery.q_gen_total
            - q_air_total
            - self.cold_plate.q_plate_to_fluid_total
        )
        self.whole_pack_energy_residual_J = (
            actual_delta_energy - expected_delta_energy
        )
        energy_scale = max(
            abs(actual_delta_energy), abs(expected_delta_energy), 1.0
        )
        self.whole_pack_energy_relative_error = (
            self.whole_pack_energy_residual_J / energy_scale
        )

        result = {
            "battery_temperature_average": self.battery.get_avg_temp(),
            "battery_temperature_max_zone": self.battery.get_max_temp(),
            "battery_temperature_min_zone": self.battery.get_min_temp(),
            "battery_temperature_delta_zone": self.battery.get_delta_temp(),
            "battery_zone_temperatures": self.battery.temps.copy(),
            "branch_currents": self.battery.get_branch_currents(),
            "soc": self.battery.get_soc_array(),
            "q_gen_total": self.battery.q_gen_total,
            "q_battery_to_plate_branch_zone": (
                self.q_battery_to_plate_branch_zone.copy()
            ),
            "q_battery_to_plate_zones": self.q_battery_to_plate_zones.copy(),
            "q_battery_to_plate_total": float(
                self.q_battery_to_plate_zones.sum()
            ),
            "q_plate_gain_zones": self.q_plate_gain_zones.copy(),
            "plate_temperatures": self.cold_plate.plate_temperatures.copy(),
            "coolant_outlet_temperature": (
                self.cold_plate.coolant_outlet_temperature
            ),
            "coolant_mean_temperatures": (
                self.cold_plate.coolant_mean_temperatures.copy()
            ),
            "q_plate_to_fluid_zones": self.cold_plate.q_plate_to_fluid.copy(),
            "q_plate_to_fluid_total": self.cold_plate.q_plate_to_fluid_total,
            "coupling_energy_residual": self.coupling_energy_residual.copy(),
            "max_abs_coupling_residual_W": float(
                np.max(np.abs(self.coupling_energy_residual))
            ),
            "whole_pack_energy_residual_J": self.whole_pack_energy_residual_J,
            "whole_pack_energy_relative_error": (
                self.whole_pack_energy_relative_error
            ),
            "q_air_total": q_air_total,
        }
        for value in result.values():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError("ReducedPack produced a non-finite output")
        return result
