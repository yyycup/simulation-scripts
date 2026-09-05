"""Five-Pack Cluster organization, flow assignment, and thermal summaries.

PyCharm navigation:
- Pack ownership and independent state: ``ReducedCluster.__init__``
- Flow assignment and return mixing: ``allocate_mass_flows`` / ``mix_return_temperature``
- Full Cluster advance and diagnostics: ``ReducedCluster.step``
- Hydraulic equations and defaults: ``hydraulics.py`` / ``parameters.py``
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from cluster_plant_v2.hydraulics import ParallelHeaderHydraulicNetwork
from cluster_plant_v2.parameters import DEFAULT_CLUSTER_PACK_COUNT
from cluster_plant_v2.thermal.reduced_pack import ReducedPack


class ReducedCluster:
    """Broadcast one series current to parallel-cooled ReducedPack instances.

    The core interface uses kelvin, amperes, seconds, watts, and kilograms per
    second. Branch resistance factors are positive relative coefficients.
    """

    def __init__(
        self,
        n_packs: int = DEFAULT_CLUSTER_PACK_COUNT,
        branch_resistance_factors: np.ndarray | list[float] | None = None,
        pack_config: Mapping[str, object] | None = None,
        hydraulic_mode: str = "lumped",
        hydraulic_network: ParallelHeaderHydraulicNetwork | None = None,
    ) -> None:
        if (
            isinstance(n_packs, bool)
            or not isinstance(n_packs, (int, np.integer))
            or int(n_packs) < 1
        ):
            raise ValueError("n_packs must be an integer greater than or equal to one")
        self.n_packs = int(n_packs)
        if branch_resistance_factors is None:
            factors = np.ones(self.n_packs, dtype=float)
        else:
            factors = np.asarray(branch_resistance_factors, dtype=float)
        if (
            factors.shape != (self.n_packs,)
            or not np.all(np.isfinite(factors))
            or np.any(factors <= 0.0)
        ):
            raise ValueError(
                "branch_resistance_factors must be finite, positive, and shape (n_packs,)"
            )
        self.branch_resistance_factors = factors.copy()
        if hydraulic_mode not in {"lumped", "header_network"}:
            raise ValueError("hydraulic_mode must be 'lumped' or 'header_network'")
        if hydraulic_mode == "lumped" and hydraulic_network is not None:
            raise ValueError("hydraulic_network is only valid in header_network mode")
        if hydraulic_mode == "header_network":
            if hydraulic_network is None:
                hydraulic_network = ParallelHeaderHydraulicNetwork(self.n_packs)
            if hydraulic_network.n_packs != self.n_packs:
                raise ValueError("hydraulic network pack count must match n_packs")
        self.hydraulic_mode = hydraulic_mode
        self.hydraulic_network = hydraulic_network
        self.last_hydraulic_result: dict[str, object] | None = None
        self.packs = [ReducedPack(pack_config) for _ in range(self.n_packs)]
        self.dynamic_state_count = sum(
            pack.dynamic_state_count for pack in self.packs
        )

    def allocate_mass_flows(self, total_mass_flow_kg_s: float) -> np.ndarray:
        """Allocate nonnegative total flow by inverse square-root resistance."""
        total = float(total_mass_flow_kg_s)
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("total_mass_flow_kg_s must be finite and nonnegative")
        if total == 0.0:
            return np.zeros(self.n_packs, dtype=float)
        weights = 1.0 / np.sqrt(self.branch_resistance_factors)
        flows = total * weights / weights.sum()
        flows[-1] += total - float(flows.sum())
        return flows

    def mix_return_temperature(
        self,
        outlet_temperatures_k: np.ndarray,
        mass_flows_kg_s: np.ndarray,
        *,
        supply_temperature_k: float,
    ) -> float:
        """Return mass-weighted temperature; at zero flow return the supply boundary."""
        outlets = np.asarray(outlet_temperatures_k, dtype=float)
        flows = np.asarray(mass_flows_kg_s, dtype=float)
        supply = float(supply_temperature_k)
        if (
            outlets.shape != (self.n_packs,)
            or flows.shape != (self.n_packs,)
            or not np.all(np.isfinite(outlets))
            or not np.all(np.isfinite(flows))
            or np.any(flows < 0.0)
            or not np.isfinite(supply)
        ):
            raise ValueError("mixing inputs must be finite cluster-length arrays")
        total = float(flows.sum())
        if total == 0.0:
            return supply
        return float(np.sum(flows * outlets) / total)

    def step(
        self,
        dt_s: float,
        cluster_current_a: float,
        supply_temperature_k: float,
        total_mass_flow_kg_s: float,
        ambient_temperature_k: float,
        direction: str = "forward",
    ) -> dict[str, float | np.ndarray]:
        """Advance all Packs independently, then mix their parallel return flows."""
        if direction not in {"forward", "reverse"}:
            raise ValueError("direction must be 'forward' or 'reverse'")
        total_mass_flow = float(total_mass_flow_kg_s)
        if not np.isfinite(total_mass_flow) or total_mass_flow <= 0.0:
            raise ValueError(
                "ReducedCluster.step requires positive total coolant mass flow"
            )
        if self.hydraulic_mode == "lumped":
            pack_mass_flows = self.allocate_mass_flows(total_mass_flow)
            self.last_hydraulic_result = None
        else:
            self.last_hydraulic_result = self.hydraulic_network.solve(
                total_mass_flow
            )
            pack_mass_flows = np.asarray(
                self.last_hydraulic_result["pack_mass_flows"], dtype=float
            )
        flow_direction = 1 if direction == "forward" else -1
        pack_results = [
            pack.step(
                dt_s,
                cluster_current_a,
                supply_temperature_k,
                pack_mass_flows[index],
                ambient_temperature_k,
                flow_direction,
            )
            for index, pack in enumerate(self.packs)
        ]

        battery_average = np.array(
            [result["battery_temperature_average"] for result in pack_results]
        )
        battery_maximum = np.array(
            [result["battery_temperature_max_zone"] for result in pack_results]
        )
        battery_minimum = np.array(
            [result["battery_temperature_min_zone"] for result in pack_results]
        )
        battery_delta = battery_maximum - battery_minimum
        battery_zones = np.stack(
            [result["battery_zone_temperatures"] for result in pack_results]
        )
        plate_temperatures = np.stack(
            [result["plate_temperatures"] for result in pack_results]
        )
        plate_average = np.array(
            [
                np.average(
                    result["plate_temperatures"],
                    weights=pack.cold_plate.zone_heat_capacities,
                )
                for result, pack in zip(pack_results, self.packs)
            ]
        )
        coolant_outlets = np.array(
            [result["coolant_outlet_temperature"] for result in pack_results]
        )
        return_temperature = self.mix_return_temperature(
            coolant_outlets,
            pack_mass_flows,
            supply_temperature_k=supply_temperature_k,
        )
        mean_flow = float(pack_mass_flows.mean())

        result = {
            "pack_currents_a": np.full(
                self.n_packs, float(cluster_current_a), dtype=float
            ),
            "pack_mass_flows_kg_s": pack_mass_flows.copy(),
            "pack_battery_average_temperatures_k": battery_average,
            "pack_battery_max_temperatures_k": battery_maximum,
            "pack_battery_min_temperatures_k": battery_minimum,
            "pack_battery_delta_temperatures_k": battery_delta,
            "pack_battery_zone_temperatures_k": battery_zones,
            "pack_plate_average_temperatures_k": plate_average,
            "pack_plate_temperatures_k": plate_temperatures,
            "pack_coolant_outlet_temperatures_k": coolant_outlets,
            "pack_soc": np.stack([result["soc"] for result in pack_results]),
            "pack_branch_currents_a": np.stack(
                [result["branch_currents"] for result in pack_results]
            ),
            "pack_q_gen_total_w": np.array(
                [result["q_gen_total"] for result in pack_results]
            ),
            "pack_q_battery_to_plate_total_w": np.array(
                [result["q_battery_to_plate_total"] for result in pack_results]
            ),
            "pack_q_plate_to_fluid_total_w": np.array(
                [result["q_plate_to_fluid_total"] for result in pack_results]
            ),
            "supply_temperature_k": float(supply_temperature_k),
            "return_temperature_k": return_temperature,
            "total_mass_flow_kg_s": total_mass_flow,
            "mean_pack_mass_flow_kg_s": mean_flow,
            "mass_flow_conservation_residual_kg_s": float(
                pack_mass_flows.sum() - total_mass_flow
            ),
            "flow_nonuniformity": float(
                np.max(np.abs((pack_mass_flows - mean_flow) / mean_flow))
            ),
            "cluster_max_temperature_k": float(battery_maximum.max()),
            "cluster_min_temperature_k": float(battery_minimum.min()),
            "cluster_delta_temperature_k": float(
                battery_maximum.max() - battery_minimum.min()
            ),
            "inter_pack_delta_temperature_k": float(np.ptp(battery_average)),
            "max_abs_pack_coupling_residual_w": float(
                max(
                    result["max_abs_coupling_residual_W"]
                    for result in pack_results
                )
            ),
            "max_abs_pack_energy_residual_j": float(
                max(
                    abs(result["whole_pack_energy_residual_J"])
                    for result in pack_results
                )
            ),
            "flow_direction": flow_direction,
        }
        for value in result.values():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError("ReducedCluster produced a non-finite output")
        return result
