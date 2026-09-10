"""Stage 4.5 — Independent Heat-Current Plant.

Why this module exists
----------------------
The Stage 3 ``HeatCurrentSystemLink`` (see ``heat_current_system.py``) is
a **shadow-comparison** tool: it reads ``tank_temperature_before_k``,
``q_evap_applied_w``, ``q_evap_cycle_w``, ``evaporating_temperature_k``,
``evaporator_ua_w_k``, ``refrigeration_solver_success``, and
``compressor_speed_used_rpm`` from the legacy ``ClusterPlant.step()``
output and feeds them into the heat-current cluster + evaporator outlet
formula. That makes it unsuitable for Stage 5's "independent full-system
validation" requirement.

This module introduces ``HeatCurrentPlant``: a **truly independently
propagating** heat-current system. It owns its own instances of

    CoolantPump
    CompressorSpeedActuator
    ClosedR134aCycle
    EvaporatorThermalDynamics
    EvaporatorHeatCurrent
    CoolantTransportDelay (supply)
    CoolantTransportDelay (return)
    CoolantTank
    HeatCurrentCluster

and runs the full 9-step physical chain inside its own ``step()``:

    pump -> compressor_actuator -> cycle -> evap_dynamics
    -> HC_evap_outlet -> supply_delay -> HC_cluster -> return_delay -> tank

**No frozen module is modified.** The cycle solver, actuator/evaporator
lag, tank, pump characteristic and cold-plate laws are reused. The default
retains fixed-time delays; an explicit reference flow in the builder opts
into fixed-inventory mass transport for supply and return. External inputs
are limited to

    I(t), N_pump_cmd(t), N_comp_cmd(t), T_amb(t), fan_rpm(t),
    d_flow(t), dt_s

— the same external inputs that drive ``ClusterPlant``. No shared state,
no shared mutable objects, no shared internal solution results.

Scope guard
-----------
This module is heat-current modeling only. It does **not** touch MPC,
TD3, controllers, or any predictive brain.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.parameters import (
    EVAPORATOR_TIME_CONSTANT_S,
    MINIMUM_COMPRESSOR_SPEED_RPM,
    RETURN_TRANSPORT_DELAY_S,
    SUPPLY_TRANSPORT_DELAY_S,
)
from cluster_plant_v2.plant import COMPRESSOR_SPINUP_SPEED_RPM
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
    CoolantTransportDelay,
    EvaporatorThermalDynamics,
)
from cluster_plant_v2.thermal.evaporator_heat_current import EvaporatorHeatCurrent
from cluster_plant_v2.thermal.coolant_mass_transport import CoolantMassTransport
from cluster_plant_v2.thermal.heat_current_system import HeatCurrentCluster


@dataclass(frozen=True)
class HeatCurrentPlantInputs:
    """External-only inputs for one ``HeatCurrentPlant.step()``.

    Mirrors ``ClusterPlantInputs`` field-for-field. The HC plant reads
    nothing else.
    """

    cluster_current_a: float
    compressor_command_rpm: float
    pump_rpm: float
    fan_rpm: float
    ambient_temperature_k: float
    flow_direction: str = "forward"


class HeatCurrentPlant:
    """Independent heat-current plant; runs the full physical chain in-house.

    Construct with the same physical parameters as the legacy plant. Each
    ``step()`` advances only this object's internal state; legacy plant
    state is never read or mutated.
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
        supply_delay: CoolantTransportDelay | CoolantMassTransport,
        return_delay: CoolantTransportDelay | CoolantMassTransport,
        evaporator_heat_current: EvaporatorHeatCurrent | None = None,
    ) -> None:
        if cluster.hydraulic_mode != "header_network":
            raise ValueError(
                "HeatCurrentPlant requires the HC cluster to be in"
                " header_network mode"
            )
        if cluster.hydraulic_network is not hydraulic_network:
            raise ValueError(
                "cluster and HeatCurrentPlant must share the same"
                " hydraulic network"
            )
        if not np.isclose(
            pump.coolant_density_kg_m3,
            tank.coolant_density_kg_m3,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("pump and tank coolant densities must match")
        if not np.isclose(
            supply_delay.dt_s,
            return_delay.dt_s,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("supply and return delays must use the same dt_s")

        # Independent ownership: every object here is referenced by this
        # plant only. The legacy plant may own *different* instances of
        # the same classes (built via ``ClusterPlant.from_equilibrium``)
        # but it never shares these Python objects with this plant.
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

    @property
    def transport_state_count(self) -> int:
        return (
            self.supply_delay.dynamic_state_count
            + self.return_delay.dynamic_state_count
        )

    @property
    def dynamic_state_count(self) -> int:
        return (
            self.cluster.dynamic_state_count
            + self.tank.dynamic_state_count
            + self.compressor_actuator.dynamic_state_count
            + self.evaporator_dynamics.dynamic_state_count
            + self.transport_state_count
        )

    @classmethod
    def from_equilibrium(
        cls,
        *,
        cluster: HeatCurrentCluster,
        hydraulic_network: ParallelHeaderHydraulicNetwork,
        pump: CoolantPump,
        tank: CoolantTank,
        refrigeration_cycle: ClosedR134aCycle,
        compressor_actuator: CompressorSpeedActuator,
        dt_s: float,
        pump_speed_rpm: float,
        fan_speed_rpm: float,
        initial_cluster_current_a: float,
        ambient_temperature_k: float,
        direction: str = "forward",
        evaporator_time_constant_s: float = EVAPORATOR_TIME_CONSTANT_S,
    ) -> "HeatCurrentPlant":
        """Build an HC plant whose initial state is the same equilibrium
        the legacy plant sees at construction time.

        The procedure mirrors ``ClusterPlant.from_equilibrium`` step for
        step: pump operating point, cycle solve, evaporator dynamics
        initial value, supply delay initial queue, HC cluster one-shot,
        return delay initial queue. No legacy state is referenced.
        """
        operating_point = pump.solve_operating_point(
            pump_speed_rpm, hydraulic_network
        )
        total_mass_flow = float(operating_point["total_mass_flow_kg_s"])
        tank_temperature = float(tank.temperature_k)
        cycle = refrigeration_cycle.solve(
            compressor_speed_rpm=compressor_actuator.speed_rpm,
            fan_speed_rpm=fan_speed_rpm,
            coolant_inlet_temperature_k=tank_temperature,
            coolant_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
        )
        if not cycle["solver_success"]:
            raise RuntimeError(
                "equilibrium refrigeration solve failed: "
                f"{cycle['solver_message']}"
            )
        evaporator_dynamics = EvaporatorThermalDynamics(
            initial_q_evap_applied_w=cycle["q_evaporator_w"],
            time_constant_s=evaporator_time_constant_s,
        )
        evaporator_heat_current = EvaporatorHeatCurrent()
        evaporator_outlet = evaporator_heat_current.outlet_temperature_from_applied_heat(
            coolant_inlet_temperature_k=tank_temperature,
            coolant_mass_flow_kg_s=total_mass_flow,
            q_applied_w=evaporator_dynamics.q_evap_applied_w,
        )
        initial_cluster_result = copy.deepcopy(cluster).step(
            dt_s=dt_s,
            cluster_current_a=initial_cluster_current_a,
            supply_temperature_k=evaporator_outlet,
            total_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
            direction=direction,
        )
        initial_return_temperature = float(
            initial_cluster_result["return_temperature_k"]
        )
        return cls(
            cluster=cluster,
            hydraulic_network=hydraulic_network,
            pump=pump,
            tank=tank,
            refrigeration_cycle=refrigeration_cycle,
            compressor_actuator=compressor_actuator,
            evaporator_dynamics=evaporator_dynamics,
            supply_delay=CoolantTransportDelay(
                delay_s=SUPPLY_TRANSPORT_DELAY_S,
                dt_s=dt_s,
                initial_value=evaporator_outlet,
            ),
            return_delay=CoolantTransportDelay(
                delay_s=RETURN_TRANSPORT_DELAY_S,
                dt_s=dt_s,
                initial_value=initial_return_temperature,
            ),
            evaporator_heat_current=evaporator_heat_current,
        )

    def step(
        self,
        inputs: HeatCurrentPlantInputs,
        *,
        dt_s: float,
    ) -> dict[str, object]:
        """Run the full 9-step physical chain on this plant's own state.

        The only inputs are external controls and disturbances. Every
        upstream quantity (``pump operating point``, ``compressor speed``,
        ``cycle q_evap_cycle``, ``T_e``, ``UA_e``, ``evaporator
        dynamics``, ``tank temperature``) is derived from this object's
        own state and the external inputs — never from any legacy plant.
        """
        if not (
            np.isclose(dt_s, self.supply_delay.dt_s, rtol=0.0, atol=1e-12)
            and np.isclose(dt_s, self.return_delay.dt_s, rtol=0.0, atol=1e-12)
        ):
            raise ValueError("dt_s must match transport-delay dt_s")

        cluster_current_a = float(inputs.cluster_current_a)
        pump_speed_rpm = float(inputs.pump_rpm)
        compressor_speed_command_rpm = float(inputs.compressor_command_rpm)
        fan_speed_rpm = float(inputs.fan_rpm)
        ambient_temperature_k = float(inputs.ambient_temperature_k)
        direction = inputs.flow_direction
        if direction not in {"forward", "reverse"}:
            raise ValueError("flow_direction must be 'forward' or 'reverse'")

        tank_temperature_before = float(self.tank.temperature_k)

        # 1. Pump + hydraulic network (own pump, own network)
        operating_point = self.pump.solve_operating_point(
            pump_speed_rpm, self.hydraulic_network
        )
        total_mass_flow = float(operating_point["total_mass_flow_kg_s"])

        # 2. Compressor actuator (own actuator state)
        actuator_result = self.compressor_actuator.step(
            dt_s=dt_s,
            speed_command_rpm=compressor_speed_command_rpm,
        )

        # 3. Conservative R134a cycle (own cycle instance)
        compressor_speed_used_rpm = float(actuator_result["speed_after_rpm"])
        compressor_off = (
            compressor_speed_command_rpm <= 0.0
            or compressor_speed_used_rpm < COMPRESSOR_SPINUP_SPEED_RPM
        )
        if compressor_off:
            refrigeration_result = {
                "solver_success": True,
                "q_evaporator_w": 0.0,
                "q_condenser_w": 0.0,
                "compressor_refrigerant_power_w": 0.0,
                "compressor_shaft_power_w": 0.0,
                "mass_flow_relative_residual": 0.0,
                "evaporator_relative_residual": 0.0,
                "condenser_relative_residual": 0.0,
                "cycle_energy_residual_w": 0.0,
            }
        else:
            refrigeration_result = self.refrigeration_cycle.solve(
                compressor_speed_rpm=max(
                    MINIMUM_COMPRESSOR_SPEED_RPM,
                    compressor_speed_used_rpm,
                ),
                fan_speed_rpm=fan_speed_rpm,
                coolant_inlet_temperature_k=tank_temperature_before,
                coolant_mass_flow_kg_s=total_mass_flow,
                ambient_temperature_k=ambient_temperature_k,
            )
            if not refrigeration_result["solver_success"]:
                unloaded = False
                for factor in (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
                    candidate = max(1000.0, compressor_speed_used_rpm * factor)
                    if candidate >= compressor_speed_used_rpm:
                        continue
                    trial = self.refrigeration_cycle.solve(
                        compressor_speed_rpm=candidate,
                        fan_speed_rpm=fan_speed_rpm,
                        coolant_inlet_temperature_k=tank_temperature_before,
                        coolant_mass_flow_kg_s=total_mass_flow,
                        ambient_temperature_k=ambient_temperature_k,
                    )
                    if trial["solver_success"]:
                        refrigeration_result = trial
                        compressor_speed_used_rpm = candidate
                        unloaded = True
                        break
                if not unloaded:
                    raise RuntimeError(
                        "refrigeration cycle solve failed: "
                        f"{refrigeration_result['solver_message']}"
                    )

        # 4. Evaporator thermal dynamics (own state)
        q_evap_cycle = float(refrigeration_result["q_evaporator_w"])
        evaporator_result = self.evaporator_dynamics.step(
            dt_s=dt_s,
            q_evap_cycle_w=q_evap_cycle,
        )
        q_evap_applied = float(evaporator_result["q_evap_applied_w"])
        coolant_cp = self.tank.coolant_specific_heat_j_kg_k
        evaporator_outlet = self.evaporator_heat_current.outlet_temperature_from_applied_heat(
            coolant_inlet_temperature_k=tank_temperature_before,
            coolant_mass_flow_kg_s=total_mass_flow,
            q_applied_w=q_evap_applied,
        )

        # 5. Supply transport delay (own queue, advanced by step())
        cluster_supply = float(
            self.supply_delay.step(evaporator_outlet, mass_flow_kg_s=total_mass_flow)
            if isinstance(self.supply_delay, CoolantMassTransport)
            else self.supply_delay.step(evaporator_outlet)
        )

        # 6. HC cluster (own cluster state)
        cluster_result = self.cluster.step(
            dt_s=dt_s,
            cluster_current_a=cluster_current_a,
            supply_temperature_k=cluster_supply,
            total_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
            direction=direction,
        )
        cluster_return = float(cluster_result["return_temperature_k"])

        # 7. Return transport delay (own queue)
        tank_return = float(
            self.return_delay.step(cluster_return, mass_flow_kg_s=total_mass_flow)
            if isinstance(self.return_delay, CoolantMassTransport)
            else self.return_delay.step(cluster_return)
        )

        # 8. Coolant tank (own state)
        tank_result = self.tank.step(
            dt_s=dt_s,
            return_temperature_k=tank_return,
            mass_flow_kg_s=total_mass_flow,
        )

        # 9. Diagnostics — no state mutation here
        q_evap_from_coolant = (
            total_mass_flow
            * coolant_cp
            * (tank_temperature_before - evaporator_outlet)
        )
        q_cluster_to_fluid = float(
            np.sum(cluster_result["pack_q_plate_to_fluid_total_w"])
        )
        q_cluster_from_temperatures = (
            total_mass_flow
            * coolant_cp
            * (cluster_return - cluster_supply)
        )
        q_gen = float(np.sum(cluster_result["pack_q_gen_total_w"]))
        refrigerant_power = float(
            refrigeration_result["compressor_refrigerant_power_w"]
        )
        shaft_power = float(
            refrigeration_result["compressor_shaft_power_w"]
        )

        return {
            **operating_point,
            "compressor_speed_command_rpm": float(
                actuator_result["speed_command_rpm"]
            ),
            "compressor_speed_before_rpm": float(
                actuator_result["speed_before_rpm"]
            ),
            "compressor_speed_used_rpm": compressor_speed_used_rpm,
            "compressor_off": bool(compressor_off),
            "tank_temperature_before_k": tank_temperature_before,
            "tank_temperature_after_k": float(
                tank_result["tank_temperature_after_k"]
            ),
            "evaporator_outlet_temperature_k": evaporator_outlet,
            "cluster_supply_temperature_k": cluster_supply,
            "cluster_return_temperature_k": cluster_return,
            "tank_return_temperature_k": tank_return,
            "q_evap_cycle_w": q_evap_cycle,
            "q_evap_applied_w": q_evap_applied,
            "q_evap_from_coolant_w": q_evap_from_coolant,
            "q_cluster_to_fluid_w": q_cluster_to_fluid,
            "q_cluster_from_temperatures_w": q_cluster_from_temperatures,
            "q_gen_w": q_gen,
            "compressor_refrigerant_power_w": refrigerant_power,
            "compressor_shaft_power_w": shaft_power,
            "evaporator_buffer_energy_j": float(
                evaporator_result["evaporator_buffer_energy_j"]
            ),
            "refrigeration_solver_success": bool(
                refrigeration_result["solver_success"]
            ),
            "supply_delay_queue_k": tuple(self.supply_delay.queue_values),
            "return_delay_queue_k": tuple(self.return_delay.queue_values),
            "cluster_result": cluster_result,
            "tank_result": tank_result,
            "all_states_finite": bool(
                np.isfinite(tank_temperature_before)
                and np.isfinite(evaporator_outlet)
                and np.isfinite(cluster_supply)
                and np.isfinite(cluster_return)
                and np.isfinite(tank_return)
                and np.all(np.isfinite(cluster_result["pack_mass_flows_kg_s"]))
                and np.all(np.isfinite(tuple(self.supply_delay.queue_values)))
                and np.all(np.isfinite(tuple(self.return_delay.queue_values)))
            ),
        }


def build_independent_hc_plant(
    *,
    legacy_plant,
    n_packs: int | None = None,
    cold_plate_factory=None,
    transport_reference_mass_flow_kg_s: float | None = None,
) -> HeatCurrentPlant:
    """Construct a ``HeatCurrentPlant`` that is **independent** of the
    given legacy plant.

    Copy every dynamic state and battery configuration, including nonuniform
    delay histories. Only the cold-plate heat-transfer law is replaced.
    Each component is copied so later configuration edits cannot leak across
    plants either. A custom cold-plate factory retains its own parameters;
    its zone layout must match the source to transfer wall temperatures.

    Passing transport_reference_mass_flow_kg_s opts into fixed-inventory
    mass transport. Otherwise the original fixed-time delays are retained.
    Supply/return mass is reference flow times the source delay duration.
    """
    legacy_cluster = legacy_plant.cluster
    if n_packs is None:
        n_packs = legacy_cluster.n_packs
    if n_packs != legacy_cluster.n_packs:
        raise ValueError("n_packs must match the source cluster for state transfer")
    network = copy.deepcopy(legacy_plant.hydraulic_network)
    parallel_cluster = HeatCurrentCluster(
        n_packs=n_packs,
        branch_resistance_factors=np.asarray(
            legacy_cluster.branch_resistance_factors, dtype=float
        ),
        pack_config=None,
        hydraulic_mode=legacy_cluster.hydraulic_mode,
        hydraulic_network=network,
        cold_plate_factory=cold_plate_factory,
    )
    for source, target in zip(legacy_cluster.packs, parallel_cluster.packs):
        target.battery = copy.deepcopy(source.battery)
        old_plate, new_plate = source.cold_plate, target.cold_plate
        if not np.array_equal(old_plate.zone_column_counts, new_plate.zone_column_counts):
            raise ValueError("cold-plate zone layout must match for state transfer")
        # Wall temperatures are states; instantaneous HC fluxes are recomputed
        # on the first step rather than copied from the legacy LMTD law.
        new_plate.plate_temperatures = old_plate.plate_temperatures.copy()
        new_plate.initial_zone_energy_J = (
            new_plate.zone_heat_capacities * new_plate.plate_temperatures
        )
        if cold_plate_factory is None:
            new_plate.zone_heat_capacities = old_plate.zone_heat_capacities.copy()
            new_plate.zone_areas = old_plate.zone_areas.copy()
            new_plate.initial_zone_energy_J = old_plate.initial_zone_energy_J.copy()
    fresh_tank = CoolantTank(
        initial_temperature_k=legacy_plant.tank.temperature_k,
        volume_l=legacy_plant.tank.volume_m3 * 1000.0,
        coolant_density_kg_m3=legacy_plant.tank.coolant_density_kg_m3,
        coolant_specific_heat_j_kg_k=(
            legacy_plant.tank.coolant_specific_heat_j_kg_k
        ),
    )
    fresh_actuator = CompressorSpeedActuator(
        initial_speed_rpm=legacy_plant.compressor_actuator.speed_rpm,
        time_constant_s=legacy_plant.compressor_actuator.time_constant_s,
    )
    fresh_evap_dynamics = EvaporatorThermalDynamics(
        initial_q_evap_applied_w=(
            legacy_plant.evaporator_dynamics.q_evap_applied_w
        ),
        time_constant_s=legacy_plant.evaporator_dynamics.time_constant_s,
    )
    fresh_evap_dynamics.evaporator_buffer_energy_j = (
        legacy_plant.evaporator_dynamics.evaporator_buffer_energy_j
    )
    fresh_supply_delay = CoolantTransportDelay(
        delay_s=legacy_plant.supply_delay.delay_s,
        dt_s=legacy_plant.supply_delay.dt_s,
        initial_queue_values=legacy_plant.supply_delay.queue_values,
    )
    fresh_return_delay = CoolantTransportDelay(
        delay_s=legacy_plant.return_delay.delay_s,
        dt_s=legacy_plant.return_delay.dt_s,
        initial_queue_values=legacy_plant.return_delay.queue_values,
    )
    if transport_reference_mass_flow_kg_s is not None:
        fresh_supply_delay = CoolantMassTransport.from_fixed_delay(
            fresh_supply_delay, reference_mass_flow_kg_s=transport_reference_mass_flow_kg_s,
            coolant_specific_heat_j_kg_k=fresh_tank.coolant_specific_heat_j_kg_k,
        )
        fresh_return_delay = CoolantMassTransport.from_fixed_delay(
            fresh_return_delay, reference_mass_flow_kg_s=transport_reference_mass_flow_kg_s,
            coolant_specific_heat_j_kg_k=fresh_tank.coolant_specific_heat_j_kg_k,
        )
    return HeatCurrentPlant(
        cluster=parallel_cluster,
        hydraulic_network=network,
        pump=copy.deepcopy(legacy_plant.pump),
        tank=fresh_tank,
        refrigeration_cycle=copy.deepcopy(legacy_plant.refrigeration_cycle),
        compressor_actuator=fresh_actuator,
        evaporator_dynamics=fresh_evap_dynamics,
        supply_delay=fresh_supply_delay,
        return_delay=fresh_return_delay,
    )


__all__ = [
    "HeatCurrentPlant",
    "HeatCurrentPlantInputs",
    "build_independent_hc_plant",
]
