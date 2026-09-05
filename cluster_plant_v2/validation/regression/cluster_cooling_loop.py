"""Historical Stage 8A coolant-loop integration retained for regression."""

from __future__ import annotations

import numpy as np

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import CoolantPump, CoolantTank


class ClusterCoolingLoop:
    """Advance Pump/Cluster algebra first and the mixed Tank state last."""

    def __init__(
        self,
        *,
        cluster: ReducedCluster,
        pump: CoolantPump,
        tank: CoolantTank,
    ) -> None:
        if cluster.hydraulic_mode != "header_network":
            raise ValueError("ClusterCoolingLoop requires header_network mode")
        if cluster.hydraulic_network is None:
            raise ValueError("ClusterCoolingLoop requires a hydraulic network")
        if not np.isclose(
            pump.coolant_density_kg_m3,
            tank.coolant_density_kg_m3,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("pump and tank coolant densities must match")
        self.cluster = cluster
        self.pump = pump
        self.tank = tank
        self.dynamic_state_count = (
            cluster.dynamic_state_count + tank.dynamic_state_count
        )

    def step(
        self,
        dt_s: float,
        cluster_current_a: float,
        pump_speed_rpm: float,
        ambient_temperature_k: float,
        direction: str = "forward",
    ) -> dict[str, object]:
        """Use T_tank[k] for Cluster step k, then update T_tank[k+1]."""
        supply_temperature = self.tank.temperature_k
        operating_point = self.pump.solve_operating_point(
            pump_speed_rpm, self.cluster.hydraulic_network
        )
        total_mass_flow = float(operating_point["total_mass_flow_kg_s"])
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

        coolant_cp = self.tank.coolant_specific_heat_j_kg_k
        fluid_heat_gain = (
            total_mass_flow
            * coolant_cp
            * (return_temperature - supply_temperature)
        )
        plate_to_fluid_heat = float(
            np.sum(cluster_result["pack_q_plate_to_fluid_total_w"])
        )
        pack_flows = np.asarray(
            cluster_result["pack_mass_flows_kg_s"], dtype=float
        )
        mean_flow = float(pack_flows.mean())
        return {
            **operating_point,
            "supply_temperature_k": float(supply_temperature),
            "return_temperature_k": return_temperature,
            "tank_temperature_before_k": float(
                tank_result["tank_temperature_before_k"]
            ),
            "tank_temperature_after_k": float(
                tank_result["tank_temperature_after_k"]
            ),
            "return_to_tank_heat_w": float(
                tank_result["return_to_tank_heat_w"]
            ),
            "tank_energy_change_j": float(tank_result["tank_energy_change_j"]),
            "tank_energy_residual_j": float(
                tank_result["tank_energy_residual_j"]
            ),
            "fluid_heat_gain_w": float(fluid_heat_gain),
            "plate_to_fluid_heat_w": plate_to_fluid_heat,
            "plate_to_fluid_residual_w": float(
                plate_to_fluid_heat - fluid_heat_gain
            ),
            "flow_cv": float(np.std(pack_flows) / mean_flow),
            "flow_nonuniformity": float(
                np.max(np.abs((pack_flows - mean_flow) / mean_flow))
            ),
            "cluster_result": cluster_result,
        }
