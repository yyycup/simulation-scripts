"""Stage 8C2 evaporator thermal dynamics for the cluster plant."""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.validation.regression.dynamic_refrigerated_cluster_cooling_loop import (
    DynamicRefrigeratedClusterCoolingLoop,
)
from cluster_plant_v2.refrigeration import EvaporatorThermalDynamics


class ThermallyDynamicRefrigeratedClusterCoolingLoop:
    """Advance compressor, cycle target, evaporator lag, Cluster, then Tank."""

    def __init__(
        self,
        *,
        dynamic_cooling_loop: DynamicRefrigeratedClusterCoolingLoop,
        evaporator_dynamics: EvaporatorThermalDynamics,
    ) -> None:
        self.dynamic_cooling_loop = dynamic_cooling_loop
        self.evaporator_dynamics = evaporator_dynamics
        self.dynamic_state_count = (
            dynamic_cooling_loop.dynamic_state_count
            + evaporator_dynamics.dynamic_state_count
        )
        self.diagnostic_state_count = evaporator_dynamics.diagnostic_state_count

    @classmethod
    def from_equilibrium(
        cls,
        *,
        dynamic_cooling_loop: DynamicRefrigeratedClusterCoolingLoop,
        pump_speed_rpm: float,
        fan_speed_rpm: float,
        ambient_temperature_k: float,
        evaporator_time_constant_s: float = 45.0,
    ) -> "ThermallyDynamicRefrigeratedClusterCoolingLoop":
        base = dynamic_cooling_loop.cooling_loop
        operating_point = base.pump.solve_operating_point(
            pump_speed_rpm, base.cluster.hydraulic_network
        )
        cycle = base.refrigeration_cycle.solve(
            compressor_speed_rpm=(
                dynamic_cooling_loop.compressor_actuator.speed_rpm
            ),
            fan_speed_rpm=fan_speed_rpm,
            coolant_inlet_temperature_k=base.tank.temperature_k,
            coolant_mass_flow_kg_s=operating_point["total_mass_flow_kg_s"],
            ambient_temperature_k=ambient_temperature_k,
        )
        if not cycle["solver_success"]:
            raise RuntimeError(
                "equilibrium refrigeration solve failed: "
                f"{cycle['solver_message']}"
            )
        return cls(
            dynamic_cooling_loop=dynamic_cooling_loop,
            evaporator_dynamics=EvaporatorThermalDynamics(
                initial_q_evap_applied_w=cycle["q_evaporator_w"],
                time_constant_s=evaporator_time_constant_s,
            ),
        )

    def step(
        self,
        *,
        dt_s: float,
        cluster_current_a: float,
        pump_speed_rpm: float,
        compressor_speed_command_rpm: float,
        fan_speed_rpm: float,
        ambient_temperature_k: float,
        direction: str = "forward",
    ) -> dict[str, object]:
        stage8c1 = self.dynamic_cooling_loop
        base = stage8c1.cooling_loop
        stored_energy_before = base._stored_battery_plate_tank_energy_j()
        tank_temperature_before = float(base.tank.temperature_k)

        operating_point = base.pump.solve_operating_point(
            pump_speed_rpm, base.cluster.hydraulic_network
        )
        total_mass_flow = float(operating_point["total_mass_flow_kg_s"])
        actuator_result = stage8c1.compressor_actuator.step(
            dt_s=dt_s,
            speed_command_rpm=compressor_speed_command_rpm,
        )
        refrigeration_result = base.refrigeration_cycle.solve(
            compressor_speed_rpm=actuator_result["speed_after_rpm"],
            fan_speed_rpm=fan_speed_rpm,
            coolant_inlet_temperature_k=tank_temperature_before,
            coolant_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
        )
        if not refrigeration_result["solver_success"]:
            raise RuntimeError(
                "refrigeration cycle solve failed: "
                f"{refrigeration_result['solver_message']}"
            )

        q_evap_cycle = float(refrigeration_result["q_evaporator_w"])
        evaporator_result = self.evaporator_dynamics.step(
            dt_s=dt_s,
            q_evap_cycle_w=q_evap_cycle,
        )
        q_evap_applied = float(evaporator_result["q_evap_applied_w"])
        coolant_cp = base.tank.coolant_specific_heat_j_kg_k
        supply_temperature = float(
            tank_temperature_before
            - q_evap_applied / (total_mass_flow * coolant_cp)
        )
        if not np.isfinite(supply_temperature):
            raise FloatingPointError("non-finite evaporator supply temperature")

        cluster_result = base.cluster.step(
            dt_s=dt_s,
            cluster_current_a=cluster_current_a,
            supply_temperature_k=supply_temperature,
            total_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
            direction=direction,
        )
        return_temperature = float(cluster_result["return_temperature_k"])
        tank_result = base.tank.step(
            dt_s=dt_s,
            return_temperature_k=return_temperature,
            mass_flow_kg_s=total_mass_flow,
        )
        stored_energy_after = base._stored_battery_plate_tank_energy_j()

        q_evap_from_coolant = (
            total_mass_flow
            * coolant_cp
            * (tank_temperature_before - supply_temperature)
        )
        q_cluster_to_fluid = float(
            np.sum(cluster_result["pack_q_plate_to_fluid_total_w"])
        )
        q_cluster_from_temperatures = (
            total_mass_flow
            * coolant_cp
            * (return_temperature - supply_temperature)
        )
        q_tank = float(tank_result["return_to_tank_heat_w"])
        q_gen = float(np.sum(cluster_result["pack_q_gen_total_w"]))
        q_air = float(
            sum(
                np.sum(pack.battery.get_air_heat_loss())
                for pack in base.cluster.packs
            )
        )
        bpt_energy_change = stored_energy_after - stored_energy_before
        expected_bpt_energy_change = float(dt_s) * (
            q_gen - q_air - q_evap_applied
        )
        bpt_energy_residual = (
            bpt_energy_change - expected_bpt_energy_change
        )
        buffer_energy_change = float(
            evaporator_result["evaporator_buffer_energy_change_j"]
        )
        combined_energy_change = bpt_energy_change + buffer_energy_change
        expected_combined_energy_change = float(dt_s) * (
            q_gen - q_air - q_evap_cycle
        )
        combined_energy_residual = (
            combined_energy_change - expected_combined_energy_change
        )

        refrigerant_power = float(
            refrigeration_result["compressor_refrigerant_power_w"]
        )
        shaft_power = float(
            refrigeration_result["compressor_shaft_power_w"]
        )
        q_condenser = float(refrigeration_result["q_condenser_w"])
        scalar_diagnostics = np.array(
            [
                tank_temperature_before,
                supply_temperature,
                return_temperature,
                q_evap_cycle,
                q_evap_applied,
                q_condenser,
                refrigerant_power,
                shaft_power,
                evaporator_result["evaporator_buffer_energy_j"],
                bpt_energy_residual,
                combined_energy_residual,
            ],
            dtype=float,
        )

        return {
            **operating_point,
            "compressor_speed_command_rpm": float(
                actuator_result["speed_command_rpm"]
            ),
            "compressor_speed_before_rpm": float(
                actuator_result["speed_before_rpm"]
            ),
            "compressor_speed_rpm": float(
                actuator_result["speed_after_rpm"]
            ),
            "fan_speed_rpm": float(fan_speed_rpm),
            "refrigerant_mass_flow_kg_s": float(
                refrigeration_result["refrigerant_mass_flow_kg_s"]
            ),
            "q_evap_cycle_w": q_evap_cycle,
            "q_evap_applied_before_w": float(
                evaporator_result["q_evap_applied_before_w"]
            ),
            "q_evap_applied_w": q_evap_applied,
            "q_condenser_cycle_w": q_condenser,
            "refrigerant_compression_power_w": refrigerant_power,
            "compressor_shaft_power_w": shaft_power,
            "compressor_mechanical_loss_w": float(
                refrigeration_result["compressor_mechanical_loss_w"]
            ),
            "evaporating_saturation_temperature_k": float(
                refrigeration_result[
                    "evaporating_saturation_temperature_k"
                ]
            ),
            "condensing_saturation_temperature_k": float(
                refrigeration_result[
                    "condensing_saturation_temperature_k"
                ]
            ),
            "cycle_mass_relative_residual": float(
                refrigeration_result["mass_flow_relative_residual"]
            ),
            "cycle_evaporator_relative_residual": float(
                refrigeration_result["evaporator_relative_residual"]
            ),
            "cycle_condenser_relative_residual": float(
                refrigeration_result["condenser_relative_residual"]
            ),
            "cycle_energy_residual_w": float(
                refrigeration_result["cycle_energy_residual_w"]
            ),
            "refrigeration_solver_success": bool(
                refrigeration_result["solver_success"]
            ),
            "tank_temperature_before_k": tank_temperature_before,
            "supply_temperature_k": supply_temperature,
            "return_temperature_k": return_temperature,
            "tank_temperature_after_k": float(
                tank_result["tank_temperature_after_k"]
            ),
            "evaporator_coolant_heat_w": float(q_evap_from_coolant),
            "evaporator_coolant_residual_w": float(
                q_evap_applied - q_evap_from_coolant
            ),
            "evaporator_buffer_energy_before_j": float(
                evaporator_result["evaporator_buffer_energy_before_j"]
            ),
            "evaporator_buffer_energy_change_j": buffer_energy_change,
            "evaporator_buffer_energy_j": float(
                evaporator_result["evaporator_buffer_energy_j"]
            ),
            "evaporator_dynamic_energy_residual_j": float(
                evaporator_result[
                    "evaporator_dynamic_energy_residual_j"
                ]
            ),
            "q_cluster_to_fluid_w": q_cluster_to_fluid,
            "cluster_fluid_residual_w": float(
                q_cluster_to_fluid - q_cluster_from_temperatures
            ),
            "q_tank_w": q_tank,
            "thermal_chain_residual_w": float(
                q_tank - (q_cluster_to_fluid - q_evap_applied)
            ),
            "cluster_q_gen_total_w": q_gen,
            "cluster_q_air_total_w": q_air,
            "bpt_energy_change_j": float(bpt_energy_change),
            "expected_bpt_energy_change_j": float(
                expected_bpt_energy_change
            ),
            "bpt_energy_residual_j": float(bpt_energy_residual),
            "bpt_evaporator_energy_change_j": float(
                combined_energy_change
            ),
            "expected_bpt_evaporator_energy_change_j": float(
                expected_combined_energy_change
            ),
            "bpt_evaporator_energy_residual_j": float(
                combined_energy_residual
            ),
            "all_states_finite": bool(
                np.all(np.isfinite(scalar_diagnostics))
                and np.all(
                    np.isfinite(cluster_result["pack_mass_flows_kg_s"])
                )
            ),
            "refrigeration_result": refrigeration_result,
            "cluster_result": cluster_result,
        }
