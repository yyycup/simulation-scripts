"""Historical Stage 8B refrigeration integration retained for regression.

The coolant topology is Tank -> Pump -> Evaporator -> Cluster -> Tank.  The
evaporator adds no hydraulic pressure drop in Stage 8B because no validated
coolant-side pressure-drop model is available.  Compressor, refrigeration,
and transport dynamics are intentionally deferred to Stage 8C.
"""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import CoolantPump, CoolantTank
from cluster_plant_v2.refrigeration import (
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    ClosedR134aCycle,
)


class RefrigeratedClusterCoolingLoop:
    """Advance Pump, closed R134a cycle, Cluster, then mixed Tank."""

    def __init__(
        self,
        *,
        cluster: ReducedCluster,
        pump: CoolantPump,
        tank: CoolantTank,
        refrigeration_cycle: ClosedR134aCycle,
    ) -> None:
        if cluster.hydraulic_mode != "header_network":
            raise ValueError(
                "RefrigeratedClusterCoolingLoop requires header_network mode"
            )
        if cluster.hydraulic_network is None:
            raise ValueError(
                "RefrigeratedClusterCoolingLoop requires a hydraulic network"
            )
        if not np.isclose(
            pump.coolant_density_kg_m3,
            tank.coolant_density_kg_m3,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("pump and tank coolant densities must match")
        if not np.isclose(
            tank.coolant_specific_heat_j_kg_k,
            COOLANT_SPECIFIC_HEAT_J_KG_K,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("tank and refrigeration coolant heat capacities must match")
        self.cluster = cluster
        self.pump = pump
        self.tank = tank
        self.refrigeration_cycle = refrigeration_cycle
        self.dynamic_state_count = (
            cluster.dynamic_state_count + tank.dynamic_state_count
        )

    def _stored_battery_plate_tank_energy_j(self) -> float:
        pack_energy = 0.0
        for pack in self.cluster.packs:
            pack_energy += float(
                np.sum(
                    pack.battery.zone_heat_capacities
                    * pack.battery.temps
                )
            )
            pack_energy += float(
                np.sum(
                    pack.cold_plate.zone_heat_capacities
                    * pack.cold_plate.plate_temperatures
                )
            )
        tank_energy = self.tank.thermal_capacity_j_k * self.tank.temperature_k
        return float(pack_energy + tank_energy)

    def step(
        self,
        dt_s: float,
        cluster_current_a: float,
        pump_speed_rpm: float,
        compressor_speed_rpm: float,
        fan_speed_rpm: float,
        ambient_temperature_k: float,
        direction: str = "forward",
    ) -> dict[str, object]:
        """Solve one quasi-steady cycle at T_tank[k], then advance Cluster/Tank."""
        stored_energy_before = self._stored_battery_plate_tank_energy_j()
        tank_temperature_before = float(self.tank.temperature_k)

        operating_point = self.pump.solve_operating_point(
            pump_speed_rpm, self.cluster.hydraulic_network
        )
        total_mass_flow = float(operating_point["total_mass_flow_kg_s"])
        refrigeration_result = self.refrigeration_cycle.solve(
            compressor_speed_rpm=compressor_speed_rpm,
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
        supply_temperature = float(
            refrigeration_result["coolant_outlet_temperature_k"]
        )

        cluster_result = self.cluster.step(
            dt_s=dt_s,
            cluster_current_a=cluster_current_a,
            supply_temperature_k=supply_temperature,
            total_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
            direction=direction,
        )
        return_temperature = float(cluster_result["return_temperature_k"])
        tank_result = self.tank.step(
            dt_s=dt_s,
            return_temperature_k=return_temperature,
            mass_flow_kg_s=total_mass_flow,
        )
        stored_energy_after = self._stored_battery_plate_tank_energy_j()

        coolant_cp = self.tank.coolant_specific_heat_j_kg_k
        q_evaporator = float(refrigeration_result["q_evaporator_w"])
        q_evaporator_coolant = (
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
                for pack in self.cluster.packs
            )
        )
        actual_energy_change = stored_energy_after - stored_energy_before
        expected_energy_change = float(dt_s) * (
            q_gen - q_air - q_evaporator
        )
        total_energy_residual = (
            actual_energy_change - expected_energy_change
        )

        refrigerant_power = float(
            refrigeration_result["compressor_refrigerant_power_w"]
        )
        shaft_power = float(refrigeration_result["compressor_shaft_power_w"])
        pack_flows = np.asarray(
            cluster_result["pack_mass_flows_kg_s"], dtype=float
        )
        mean_flow = float(pack_flows.mean())
        scalar_diagnostics = np.array(
            [
                tank_temperature_before,
                supply_temperature,
                return_temperature,
                total_mass_flow,
                q_evaporator,
                q_cluster_to_fluid,
                q_tank,
                actual_energy_change,
                total_energy_residual,
            ],
            dtype=float,
        )

        return {
            **operating_point,
            "compressor_speed_rpm": float(compressor_speed_rpm),
            "fan_speed_rpm": float(fan_speed_rpm),
            "refrigeration_coolant_mass_flow_kg_s": total_mass_flow,
            "refrigerant_mass_flow_kg_s": float(
                refrigeration_result["refrigerant_mass_flow_kg_s"]
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
            "q_evaporator_w": q_evaporator,
            "q_condenser_w": float(refrigeration_result["q_condenser_w"]),
            "refrigerant_compression_power_w": refrigerant_power,
            "compressor_shaft_power_w": shaft_power,
            "compressor_mechanical_loss_w": float(
                refrigeration_result["compressor_mechanical_loss_w"]
            ),
            "cop_refrigerant": q_evaporator / refrigerant_power,
            "cop_shaft": q_evaporator / shaft_power,
            "cycle_mass_residual_kg_s": float(
                refrigeration_result["mass_flow_residual_kg_s"]
            ),
            "cycle_mass_relative_residual": float(
                refrigeration_result["mass_flow_relative_residual"]
            ),
            "cycle_evaporator_residual_w": float(
                refrigeration_result["evaporator_residual_w"]
            ),
            "cycle_evaporator_relative_residual": float(
                refrigeration_result["evaporator_relative_residual"]
            ),
            "cycle_condenser_residual_w": float(
                refrigeration_result["condenser_residual_w"]
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
            "refrigeration_solver_function_evaluations": int(
                refrigeration_result["solver_function_evaluations"]
            ),
            "tank_temperature_before_k": tank_temperature_before,
            "supply_temperature_k": supply_temperature,
            "return_temperature_k": return_temperature,
            "tank_temperature_after_k": float(
                tank_result["tank_temperature_after_k"]
            ),
            "q_evaporator_coolant_w": float(q_evaporator_coolant),
            "evaporator_coolant_residual_w": float(
                q_evaporator_coolant - q_evaporator
            ),
            "q_cluster_to_fluid_w": q_cluster_to_fluid,
            "q_cluster_from_temperatures_w": float(
                q_cluster_from_temperatures
            ),
            "cluster_fluid_residual_w": float(
                q_cluster_to_fluid - q_cluster_from_temperatures
            ),
            "q_tank_w": q_tank,
            "thermal_chain_residual_w": float(
                q_tank - (q_cluster_to_fluid - q_evaporator)
            ),
            "tank_energy_change_j": float(
                tank_result["tank_energy_change_j"]
            ),
            "tank_energy_residual_j": float(
                tank_result["tank_energy_residual_j"]
            ),
            "cluster_q_gen_total_w": q_gen,
            "cluster_q_air_total_w": q_air,
            "stored_energy_before_j": stored_energy_before,
            "stored_energy_after_j": stored_energy_after,
            "actual_energy_change_j": float(actual_energy_change),
            "expected_energy_change_j": float(expected_energy_change),
            "total_energy_residual_j": float(total_energy_residual),
            "pack_energy_residual_sum_j": float(
                sum(
                    pack.whole_pack_energy_residual_J
                    for pack in self.cluster.packs
                )
            ),
            "flow_cv": float(np.std(pack_flows) / mean_flow),
            "flow_nonuniformity": float(
                np.max(np.abs((pack_flows - mean_flow) / mean_flow))
            ),
            "all_states_finite": bool(
                np.all(np.isfinite(scalar_diagnostics))
                and np.all(np.isfinite(pack_flows))
            ),
            "refrigeration_result": refrigeration_result,
            "cluster_result": cluster_result,
        }
