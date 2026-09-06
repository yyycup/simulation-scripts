"""Parallel heat-current Pack, Cluster, and System-Link assembly layer.

Stage 3 intent
--------------
This module introduces a **parallel** system-level implementation that
swaps the cold-plate sub-step and the evaporator outlet calculation for
the frozen heat-current interfaces (``ColdPlateHeatCurrent`` and
``EvaporatorHeatCurrent``) **without** modifying the authoritative
``ClusterPlant`` in ``plant.py``.

Contract
-------
* ``HeatCurrentReducedPack`` couples an independent ``ReducedBatteryPack``
  with a ``ColdPlateHeatCurrent`` plate. The battery side is the exact
  same model as the incumbent ``ReducedPack`` (no re‑definition of
  ``Q_bp = H_bp (T_b - T_p)``; we only read the upstream
  ``q_battery_to_plate_zones`` attribute that the battery already writes).
* ``HeatCurrentCluster`` mirrors the ``ReducedCluster`` shape with
  ``HeatCurrentReducedPack`` instances, returning the same diagnostic
  fields (with the additional heat-current plate fields that the new
  cold plate exposes).
* ``HeatCurrentSystemLink`` reuses the frozen pump, tank, refrigeration
  cycle, compressor actuator, evaporator dynamics, and transport delays
  (the same objects the legacy plant runs) and replaces only:
    - the cluster step (with a ``HeatCurrentCluster``), and
    - the evaporator outlet formula
      (with ``EvaporatorHeatCurrent.outlet_temperature_from_applied_heat``).
  Energy closure on the evaporator side is exact by construction
  (``G_c * (T_in - T_out_applied) = Q_applied``).

The legacy ``ClusterPlant`` code path is **not** modified. The
``build_parallel_system`` helper constructs both plants in lockstep so the
operator can compare them at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.parameters import DEFAULT_CLUSTER_PACK_COUNT
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
    CoolantTransportDelay,
    EvaporatorThermalDynamics,
)
from cluster_plant_v2.thermal.cold_plate_heat_current import ColdPlateHeatCurrent
from cluster_plant_v2.thermal.evaporator_heat_current import EvaporatorHeatCurrent
from cluster_plant_v2.thermal.pack_rom import ReducedBatteryPack


class HeatCurrentReducedPack:
    """Battery-ROM -> ColdPlateHeatCurrent coupling, parallel to ``ReducedPack``.

    The upstream ``q_battery_to_plate_zones`` is taken straight from the
    unchanged battery; this class only swaps the cold plate. Energy closure
    diagnostics mirror ``ReducedPack``.
    """

    def __init__(
        self,
        battery_config: Mapping[str, object] | None = None,
        initial_plate_temperature_c: float = 25.0,
        cold_plate: ColdPlateHeatCurrent | None = None,
    ) -> None:
        self.battery = ReducedBatteryPack(battery_config)
        self.cold_plate = (
            ColdPlateHeatCurrent(initial_plate_temperature_c)
            if cold_plate is None
            else cold_plate
        )
        battery_zone_count = int(self.battery.q_battery_to_plate_zones.size)
        if self.cold_plate.dynamic_state_count != battery_zone_count:
            raise ValueError(
                "ColdPlateHeatCurrent zone count"
                f" ({self.cold_plate.dynamic_state_count}) must match"
                " battery's q_battery_to_plate_zones length"
                f" ({battery_zone_count})"
            )
        self.dynamic_state_count = (
            self.battery.dynamic_state_count + self.cold_plate.dynamic_state_count
        )
        self.q_battery_to_plate_branch_zone = np.zeros((4, 3), dtype=float)
        self.q_battery_to_plate_zones = np.zeros(
            self.cold_plate.dynamic_state_count, dtype=float
        )
        self.q_plate_gain_zones = np.zeros(
            self.cold_plate.dynamic_state_count, dtype=float
        )
        self.coupling_energy_residual = np.zeros(
            self.cold_plate.dynamic_state_count, dtype=float
        )
        self.whole_pack_energy_residual_J = 0.0
        self.whole_pack_energy_relative_error = 0.0

    def _stored_thermal_energy(self) -> float:
        battery_energy = float(
            np.sum(self.battery.zone_heat_capacities * self.battery.temps)
        )
        plate_energy = float(
            np.sum(
                self.cold_plate.zone_heat_capacities
                * self.cold_plate.plate_temperatures
            )
        )
        return battery_energy + plate_energy

    def step(
        self,
        dt: float,
        total_current: float,
        coolant_inlet_temperature: float,
        coolant_mass_flow: float,
        ambient_temperature: float,
        flow_direction: int = 1,
    ) -> dict[str, float | np.ndarray]:
        """Advance battery, then plate, mirroring ``ReducedPack``."""
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
        q_pf_total = float(plate_result["q_plate_to_fluid_total"])
        expected_delta_energy = float(dt) * (
            self.battery.q_gen_total - q_air_total - q_pf_total
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
            "q_plate_to_fluid_zones": plate_result["q_plate_to_fluid"].copy(),
            "q_plate_to_fluid_total": q_pf_total,
            "coupling_energy_residual": self.coupling_energy_residual.copy(),
            "max_abs_coupling_residual_W": float(
                np.max(np.abs(self.coupling_energy_residual))
            ),
            "whole_pack_energy_residual_J": self.whole_pack_energy_residual_J,
            "whole_pack_energy_relative_error": (
                self.whole_pack_energy_relative_error
            ),
            "q_air_total": q_air_total,
            "plate_dynamic_h_w_m2_k": float(plate_result["h_dynamic"]),
            "plate_internal_steps": int(plate_result["internal_steps"]),
            "plate_internal_dt_s": float(plate_result["internal_dt_s"]),
        }
        for value in result.values():
            if not np.all(np.isfinite(value)):
                raise FloatingPointError(
                    "HeatCurrentReducedPack produced a non-finite output"
                )
        return result


class HeatCurrentCluster:
    """Five-Pack cluster composed of ``HeatCurrentReducedPack`` instances.

    The hydraulic layer (lumped or header-network) is reused unchanged so
    that flow allocation matches ``ReducedCluster`` byte-for-byte.
    """

    def __init__(
        self,
        n_packs: int = DEFAULT_CLUSTER_PACK_COUNT,
        branch_resistance_factors: np.ndarray | list[float] | None = None,
        pack_config: Mapping[str, object] | None = None,
        hydraulic_mode: str = "lumped",
        hydraulic_network: ParallelHeaderHydraulicNetwork | None = None,
        cold_plate_factory=None,
    ) -> None:
        if (
            isinstance(n_packs, bool)
            or not isinstance(n_packs, (int, np.integer))
            or int(n_packs) < 1
        ):
            raise ValueError(
                "n_packs must be an integer greater than or equal to one"
            )
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
                "branch_resistance_factors must be finite, positive, and shape"
                " (n_packs,)"
            )
        self.branch_resistance_factors = factors.copy()
        if hydraulic_mode not in {"lumped", "header_network"}:
            raise ValueError("hydraulic_mode must be 'lumped' or 'header_network'")
        if hydraulic_mode == "lumped" and hydraulic_network is not None:
            raise ValueError(
                "hydraulic_network is only valid in header_network mode"
            )
        if hydraulic_mode == "header_network":
            if hydraulic_network is None:
                hydraulic_network = ParallelHeaderHydraulicNetwork(self.n_packs)
            if hydraulic_network.n_packs != self.n_packs:
                raise ValueError(
                    "hydraulic network pack count must match n_packs"
                )
        self.hydraulic_mode = hydraulic_mode
        self.hydraulic_network = hydraulic_network
        self.last_hydraulic_result: dict[str, object] | None = None
        if cold_plate_factory is None:
            self.packs = [
                HeatCurrentReducedPack(pack_config) for _ in range(self.n_packs)
            ]
        else:
            self.packs = [
                HeatCurrentReducedPack(pack_config, cold_plate=cold_plate_factory())
                for _ in range(self.n_packs)
            ]
        self.dynamic_state_count = sum(
            pack.dynamic_state_count for pack in self.packs
        )

    def allocate_mass_flows(self, total_mass_flow_kg_s: float) -> np.ndarray:
        """Allocate nonnegative total flow by inverse square-root resistance.

        Identical math to ``ReducedCluster.allocate_mass_flows`` so the
        same pump operating point yields the same per-pack flows.
        """
        total = float(total_mass_flow_kg_s)
        if not np.isfinite(total) or total < 0.0:
            raise ValueError(
                "total_mass_flow_kg_s must be finite and nonnegative"
            )
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
        """Mass-weighted return; identical to ``ReducedCluster``."""
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
            raise ValueError(
                "mixing inputs must be finite cluster-length arrays"
            )
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
        total_mass_flow = float(total_mass_flow_kg_s)
        if not np.isfinite(total_mass_flow) or total_mass_flow <= 0.0:
            raise ValueError(
                "HeatCurrentCluster.step requires positive total coolant mass flow"
            )
        if direction not in {"forward", "reverse"}:
            raise ValueError("direction must be 'forward' or 'reverse'")
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
                float(
                    np.average(
                        result["plate_temperatures"],
                        weights=pack.cold_plate.zone_heat_capacities,
                    )
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
            "pack_plate_dynamic_h_w_m2_k": np.array(
                [result["plate_dynamic_h_w_m2_k"] for result in pack_results]
            ),
            "pack_plate_internal_steps": np.array(
                [result["plate_internal_steps"] for result in pack_results]
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
                raise FloatingPointError(
                    "HeatCurrentCluster produced a non-finite output"
                )
        return result


@dataclass(frozen=True)
class HeatCurrentStepInputs:
    """Inputs the heat-current system receives from the legacy plant step.

    The heat-current system **reuses** the frozen pump, cycle, compressor
    actuator, evaporator dynamics, transport delays, and tank of the legacy
    plant; only the cluster and the evaporator-outlet formula are replaced.
    To keep the comparison fair, the legacy plant is always stepped first
    and its results (excluding the cluster itself) are passed in here.
    """

    dt_s: float
    cluster_current_a: float
    ambient_temperature_k: float
    direction: str
    total_mass_flow_kg_s: float
    tank_temperature_before_k: float
    q_evap_applied_w: float
    q_evap_cycle_w: float
    evaporating_temperature_k: float
    evaporator_ua_w_k: float
    refrigeration_solver_success: bool
    compressor_speed_rpm: float


class HeatCurrentSystemLink:
    """Parallel system step that reuses frozen plant components.

    The constructor accepts the **same** pump, refrigeration cycle,
    compressor actuator, evaporator dynamics, transport delays, and tank
    instances as ``ClusterPlant``. State on those objects is **not** mutated
    here: this class only invokes ``step`` on its independent cluster, runs
    an independent evaporator-outlet formula, and reports its diagnostics.
    The legacy ``ClusterPlant`` keeps its own identical state trajectory.
    """

    def __init__(
        self,
        *,
        cluster: HeatCurrentCluster,
        hydraulic_network: ParallelHeaderHydraulicNetwork,
        pump: CoolantPump,
        tank: CoolantTank,
        refrigeration_cycle: ClosedR134aCycle,
        compressor_actuator: CompressorSpeedActuator,
        evaporator_dynamics: EvaporatorThermalDynamics,
        supply_delay: CoolantTransportDelay,
        return_delay: CoolantTransportDelay,
        evaporator_heat_current: EvaporatorHeatCurrent | None = None,
    ) -> None:
        if cluster.hydraulic_mode != "header_network":
            raise ValueError(
                "HeatCurrentSystemLink requires header_network mode (lumped only"
                " supported by HeatCurrentCluster; refused for parity with"
                " ClusterPlant)"
            )
        if cluster.hydraulic_network is not hydraulic_network:
            raise ValueError(
                "cluster and HeatCurrentSystemLink must share the same"
                " hydraulic network"
            )
        if not np.isclose(
            supply_delay.dt_s,
            return_delay.dt_s,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("supply and return delays must use the same dt_s")

        self.cluster = cluster
        self.hydraulic_network = hydraulic_network
        self.pump = pump
        self.tank = tank
        self.refrigeration_cycle = refrigeration_cycle
        self.compressor_actuator = compressor_actuator
        self.evaporator_dynamics = evaporator_dynamics
        self.supply_delay = supply_delay
        self.return_delay = return_delay
        self.evaporator_heat_current = (
            EvaporatorHeatCurrent()
            if evaporator_heat_current is None
            else evaporator_heat_current
        )

        self.transport_state_count = (
            supply_delay.dynamic_state_count
            + return_delay.dynamic_state_count
        )
        self.dynamic_state_count = (
            cluster.dynamic_state_count
            + tank.dynamic_state_count
            + compressor_actuator.dynamic_state_count
            + evaporator_dynamics.dynamic_state_count
            + self.transport_state_count
        )

    def step(self, inputs: HeatCurrentStepInputs) -> dict[str, object]:
        """Run only the cluster step and the heat-current evaporator outlet.

        All upstream quantities (pump operating point, compressor speed,
        cycle result, evaporator dynamics result, and tank temperature
        before the step) are supplied by the legacy plant for this step.
        No shared state is mutated.
        """
        if inputs.dt_s != self.supply_delay.dt_s:
            raise ValueError("dt_s must match transport-delay dt_s")

        # 4'. Evaporator outlet via the heat-current interface.
        # Energy closure is exact by construction:
        #     G_c * (T_in - T_out_applied) == q_evap_applied_w
        evaporator_outlet_applied = (
            self.evaporator_heat_current.outlet_temperature_from_applied_heat(
                coolant_inlet_temperature_k=inputs.tank_temperature_before_k,
                coolant_mass_flow_kg_s=inputs.total_mass_flow_kg_s,
                q_applied_w=inputs.q_evap_applied_w,
            )
        )
        # Steady-state cross-check using the cycle solver's T_e / UA_e.
        q_hc_w = 0.0
        outlet_ss_k = float(inputs.tank_temperature_before_k)
        effectiveness = 0.0
        if inputs.refrigeration_solver_success:
            hc_eval = self.evaporator_heat_current.evaluate(
                coolant_inlet_temperature_k=inputs.tank_temperature_before_k,
                coolant_mass_flow_kg_s=inputs.total_mass_flow_kg_s,
                evaporating_temperature_k=inputs.evaporating_temperature_k,
                evaporator_ua_w_k=inputs.evaporator_ua_w_k,
            )
            q_hc_w = float(hc_eval["q_hc_w"])
            outlet_ss_k = float(hc_eval["coolant_outlet_temperature_ss_k"])
            effectiveness = float(hc_eval["evaporator_effectiveness"])
        q_hc_minus_cycle = q_hc_w - inputs.q_evap_cycle_w

        # 5'. Supply transport delay (independent queue from the legacy one).
        cluster_supply = self.supply_delay.step(evaporator_outlet_applied)

        # 6'. Cluster (HeatCurrentCluster independent from the legacy one).
        cluster_result = self.cluster.step(
            dt_s=inputs.dt_s,
            cluster_current_a=inputs.cluster_current_a,
            supply_temperature_k=cluster_supply,
            total_mass_flow_kg_s=inputs.total_mass_flow_kg_s,
            ambient_temperature_k=inputs.ambient_temperature_k,
            direction=inputs.direction,
        )
        cluster_return = float(cluster_result["return_temperature_k"])

        # 7'. Return transport delay (independent queue).
        tank_return = self.return_delay.step(cluster_return)

        # 8'. Tank (independent buffer, but reads the same return flow).
        tank_result = self.tank.step(
            dt_s=inputs.dt_s,
            return_temperature_k=tank_return,
            mass_flow_kg_s=inputs.total_mass_flow_kg_s,
        )

        # 9'. Cross-check diagnostics.
        coolant_cp = self.tank.coolant_specific_heat_j_kg_k
        q_evap_from_coolant = (
            inputs.total_mass_flow_kg_s
            * coolant_cp
            * (inputs.tank_temperature_before_k - evaporator_outlet_applied)
        )
        q_cluster_to_fluid = float(
            np.sum(cluster_result["pack_q_plate_to_fluid_total_w"])
        )
        q_cluster_from_temperatures = (
            inputs.total_mass_flow_kg_s
            * coolant_cp
            * (cluster_return - cluster_supply)
        )

        return {
            "q_evap_hc_w": q_hc_w,
            "q_hc_minus_q_cycle_w": q_hc_minus_cycle,
            "evaporator_outlet_temperature_applied_k": evaporator_outlet_applied,
            "evaporator_outlet_temperature_ss_k": outlet_ss_k,
            "evaporator_effectiveness": effectiveness,
            "evaporator_coolant_residual_w": float(
                inputs.q_evap_applied_w - q_evap_from_coolant
            ),
            "cluster_supply_temperature_k": float(cluster_supply),
            "cluster_return_temperature_k": cluster_return,
            "tank_return_temperature_k": float(tank_return),
            "tank_temperature_after_k": float(
                tank_result["tank_temperature_after_k"]
            ),
            "q_cluster_to_fluid_w": q_cluster_to_fluid,
            "cluster_fluid_residual_w": float(
                q_cluster_to_fluid - q_cluster_from_temperatures
            ),
            "q_tank_w": float(tank_result["return_to_tank_heat_w"]),
            "tank_energy_residual_j": float(
                tank_result["tank_energy_residual_j"]
            ),
            "cluster_result": cluster_result,
            "tank_result": tank_result,
            "all_states_finite": bool(
                np.isfinite(evaporator_outlet_applied)
                and np.isfinite(cluster_supply)
                and np.isfinite(cluster_return)
                and np.isfinite(tank_return)
                and np.all(np.isfinite(cluster_result["pack_mass_flows_kg_s"]))
            ),
        }


def build_parallel_system(
    *,
    legacy_plant,  # ClusterPlant-like, only used to copy sub-objects
    n_packs: int = DEFAULT_CLUSTER_PACK_COUNT,
    cold_plate_factory=None,
) -> HeatCurrentSystemLink:
    """Construct a ``HeatCurrentSystemLink`` that shares state with ``legacy_plant``.

    The returned link reuses the legacy plant's pump, tank, refrigeration
    cycle, compressor actuator, evaporator dynamics, and transport delays
    (same Python objects). Only the cluster is freshly built; it uses its
    own independent battery state and own independent cold-plate state.
    """
    parallel_cluster = HeatCurrentCluster(
        n_packs=n_packs,
        branch_resistance_factors=np.asarray(
            legacy_plant.cluster.branch_resistance_factors, dtype=float
        ),
        pack_config=None,
        hydraulic_mode=legacy_plant.cluster.hydraulic_mode,
        hydraulic_network=legacy_plant.cluster.hydraulic_network,
        cold_plate_factory=cold_plate_factory,
    )
    return HeatCurrentSystemLink(
        cluster=parallel_cluster,
        hydraulic_network=legacy_plant.hydraulic_network,
        pump=legacy_plant.pump,
        tank=CoolantTank(initial_temperature_k=legacy_plant.tank.temperature_k),
        refrigeration_cycle=legacy_plant.refrigeration_cycle,
        compressor_actuator=legacy_plant.compressor_actuator,
        evaporator_dynamics=legacy_plant.evaporator_dynamics,
        supply_delay=CoolantTransportDelay(
            delay_s=legacy_plant.supply_delay.delay_s,
            dt_s=legacy_plant.supply_delay.dt_s,
            initial_value=legacy_plant.supply_delay.queue_values[-1],
        ),
        return_delay=CoolantTransportDelay(
            delay_s=legacy_plant.return_delay.delay_s,
            dt_s=legacy_plant.return_delay.dt_s,
            initial_value=legacy_plant.return_delay.queue_values[-1],
        ),
    )


__all__ = [
    "HeatCurrentReducedPack",
    "HeatCurrentCluster",
    "HeatCurrentSystemLink",
    "HeatCurrentStepInputs",
    "build_parallel_system",
]