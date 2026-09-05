"""Authoritative five-Pack Cluster Plant.

Cluster Plant V2 data flow for each step:

Tank -> Pump -> Hydraulic Network -> Compressor Actuator -> R134a Cycle
     -> Evaporator Dynamics -> 15 s Supply Delay -> 5-Pack Cluster
     -> 20 s Return Delay -> Tank -> Diagnostics

The complete advance order is intentionally visible in ``ClusterPlant.step``.
Component equations live in ``cluster.py``, ``hydraulics.py``, and
``refrigeration.py``; shared defaults live in ``parameters.py``.

Electrical accounting (``electrical_power_w`` and friends) is
reporting-only: it adds the motor/VFD, condenser-fan, and coolant-pump
draw on top of the compressor shaft power without feeding back into the
dynamics (see ``COP_REAL_ANCHOR_AUDIT_20260901.md``).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.parameters import (
    CONDENSER_FAN_POWER_FULL_SPEED_W,
    CONDENSER_FAN_REFERENCE_SPEED_RPM,
    DRIVE_TRAIN_EFFICIENCY,
    EVAPORATOR_TIME_CONSTANT_S,
    RETURN_TRANSPORT_DELAY_S,
    SIMULATION_TIME_STEP_S,
    SUPPLY_TRANSPORT_DELAY_S,
)
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
    CoolantTransportDelay,
    EvaporatorThermalDynamics,
    MINIMUM_COMPRESSOR_SPEED_RPM,
)

# Stage 8D4d: while restarting from standstill the first-order actuator
# lag sits below the 1000 rpm cycle floor. The R134a cycle only closes
# at/above the floor, so extraction starts once the spooling shaft
# reaches this threshold (the restart lag then takes over).
COMPRESSOR_SPINUP_SPEED_RPM = 900.0


def condenser_fan_power_w(fan_speed_rpm: float) -> float:
    """Cubic-law condenser fan power.

    ``refrigeration.py`` models the condenser air side thermally but has no
    fan power term, so an air mover of this class (8 m2 coil, up to 6.5 kg/s)
    would otherwise draw nothing. Reporting-only: the value is never fed back
    into the refrigerant or coolant equations.
    """
    ratio = float(fan_speed_rpm) / CONDENSER_FAN_REFERENCE_SPEED_RPM
    if not np.isfinite(ratio) or ratio < 0.0:
        raise ValueError("fan_speed_rpm must be finite and nonnegative")
    return CONDENSER_FAN_POWER_FULL_SPEED_W * ratio**3


@dataclass(frozen=True)
class ClusterPlantInputs:
    """Typed control and disturbance inputs for one Plant step."""

    cluster_current_a: float
    compressor_command_rpm: float
    pump_rpm: float
    fan_rpm: float
    ambient_temperature_k: float
    flow_direction: str = "forward"


class ClusterPlantOutputs(dict[str, object]):
    """Legacy-compatible result mapping with typed MPC-facing properties."""

    @property
    def battery_avg_temp_c(self) -> float:
        cluster = self["cluster_result"]
        return float(
            np.mean(cluster["pack_battery_average_temperatures_k"])
            - 273.15
        )

    @property
    def battery_max_temp_c(self) -> float:
        return float(self["cluster_result"]["cluster_max_temperature_k"] - 273.15)

    @property
    def inter_pack_delta_t_k(self) -> float:
        return float(self["cluster_result"]["inter_pack_delta_temperature_k"])

    @property
    def cluster_delta_t_k(self) -> float:
        return float(self["cluster_result"]["cluster_delta_temperature_k"])

    @property
    def tank_temp_c(self) -> float:
        return float(self["tank_temperature_after_k"] - 273.15)

    @property
    def supply_temp_c(self) -> float:
        return float(self["cluster_supply_temperature_k"] - 273.15)

    @property
    def return_temp_c(self) -> float:
        return float(self["cluster_return_temperature_k"] - 273.15)

    @property
    def compressor_actual_rpm(self) -> float:
        return float(self["compressor_speed_rpm"])

    @property
    def q_evap_cycle_w(self) -> float:
        return float(self["q_evap_cycle_w"])

    @property
    def q_evap_applied_w(self) -> float:
        return float(self["q_evap_applied_w"])

    @property
    def total_mass_flow_kg_s(self) -> float:
        return float(self["total_mass_flow_kg_s"])

    @property
    def pack_mass_flows_kg_s(self) -> np.ndarray:
        return np.asarray(
            self["cluster_result"]["pack_mass_flows_kg_s"], dtype=float
        )

    @property
    def pump_power_w(self) -> float:
        return float(self["pump_power_w"])

    @property
    def compressor_power_w(self) -> float:
        """Compressor SHAFT power only - not what the grid sees."""
        return float(self["compressor_shaft_power_w"])

    @property
    def electrical_power_w(self) -> float:
        """Grid-side electrical draw: shaft power plus drive, fan and pump.

        Use this - not :attr:`compressor_power_w` - for any energy, cost or
        COP statement. The frozen cycle stops at the compressor shaft; the
        drive train, the condenser fan and the coolant pump are missing there.
        """
        return float(self["electrical_power_w"])

    @property
    def electrical_cop(self) -> float:
        """Applied evaporator cooling per unit of grid-side electrical power."""
        return float(self["electrical_cop"])

    @property
    def soc(self) -> np.ndarray:
        return np.asarray(self["cluster_result"]["pack_soc"], dtype=float)


class ClusterPlant:
    """Directly compose and advance all authoritative Cluster components."""

    def __init__(
        self,
        *,
        cluster: ReducedCluster,
        hydraulic_network: ParallelHeaderHydraulicNetwork,
        pump: CoolantPump,
        tank: CoolantTank,
        refrigeration_cycle: ClosedR134aCycle,
        compressor_actuator: CompressorSpeedActuator,
        evaporator_dynamics: EvaporatorThermalDynamics,
        supply_delay: CoolantTransportDelay,
        return_delay: CoolantTransportDelay,
    ) -> None:
        if cluster.hydraulic_mode != "header_network":
            raise ValueError("ClusterPlant requires header_network mode")
        if cluster.hydraulic_network is not hydraulic_network:
            raise ValueError(
                "cluster and plant must share the same hydraulic network"
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

        self.cluster = cluster
        self.hydraulic_network = hydraulic_network
        self.pump = pump
        self.tank = tank
        self.refrigeration_cycle = refrigeration_cycle
        self.compressor_actuator = compressor_actuator
        self.evaporator_dynamics = evaporator_dynamics
        self.supply_delay = supply_delay
        self.return_delay = return_delay

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
        self.diagnostic_state_count = evaporator_dynamics.diagnostic_state_count

    @classmethod
    def from_equilibrium(
        cls,
        *,
        cluster: ReducedCluster,
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
    ) -> "ClusterPlant":
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
        evaporator_outlet = float(
            tank_temperature
            - evaporator_dynamics.q_evap_applied_w
            / (
                total_mass_flow
                * tank.coolant_specific_heat_j_kg_k
            )
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
        )

    def step(
        self,
        inputs: ClusterPlantInputs | None = None,
        *,
        dt_s: float = SIMULATION_TIME_STEP_S,
        cluster_current_a: float | None = None,
        pump_speed_rpm: float | None = None,
        compressor_speed_command_rpm: float | None = None,
        fan_speed_rpm: float | None = None,
        ambient_temperature_k: float | None = None,
        direction: str = "forward",
    ) -> ClusterPlantOutputs:
        """Advance the complete Cluster Plant in the declared physical order."""
        legacy_values = (
            cluster_current_a,
            pump_speed_rpm,
            compressor_speed_command_rpm,
            fan_speed_rpm,
            ambient_temperature_k,
        )
        if inputs is not None:
            if any(value is not None for value in legacy_values):
                raise TypeError(
                    "pass either ClusterPlantInputs or legacy keyword inputs, not both"
                )
            cluster_current_a = inputs.cluster_current_a
            pump_speed_rpm = inputs.pump_rpm
            compressor_speed_command_rpm = inputs.compressor_command_rpm
            fan_speed_rpm = inputs.fan_rpm
            ambient_temperature_k = inputs.ambient_temperature_k
            direction = inputs.flow_direction
        elif any(value is None for value in legacy_values):
            raise TypeError("all Cluster Plant keyword inputs are required")

        if not (
            np.isclose(dt_s, self.supply_delay.dt_s, rtol=0.0, atol=1e-12)
            and np.isclose(
                dt_s, self.return_delay.dt_s, rtol=0.0, atol=1e-12
            )
        ):
            raise ValueError("dt_s must match transport-delay dt_s")

        tank_temperature_before = float(self.tank.temperature_k)

        # 1. Pump + hydraulic network
        operating_point = self.pump.solve_operating_point(
            pump_speed_rpm, self.hydraulic_network
        )
        total_mass_flow = float(operating_point["total_mass_flow_kg_s"])

        # 2. Compressor actuator
        actuator_result = self.compressor_actuator.step(
            dt_s=dt_s,
            speed_command_rpm=compressor_speed_command_rpm,
        )

        # 3. Conservative R134a cycle
        compressor_speed_used_rpm = float(actuator_result["speed_after_rpm"])
        # Stage 8D4d: a 0 rpm command stops the compressor (thermostat
        # cycling of an oversized unit at low load). While the shaft is
        # stopped or still spooling up below the spin-up threshold there
        # is no compression: no evaporator extraction and no shaft power.
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
                # Low-pressure protection: with an overcooled return (low load vs.
                # capacity) the commanded speed would pull the evaporating
                # temperature below its 7 C floor and the cycle has no physical
                # solution there. A real unit unloads the compressor instead;
                # mirror that by reducing speed geometrically until the cycle
                # closes again (floor = the 1000 rpm compressor minimum).
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

        # 4. Evaporator thermal dynamics
        q_evap_cycle = float(refrigeration_result["q_evaporator_w"])
        evaporator_result = self.evaporator_dynamics.step(
            dt_s=dt_s,
            q_evap_cycle_w=q_evap_cycle,
        )
        q_evap_applied = float(evaporator_result["q_evap_applied_w"])
        coolant_cp = self.tank.coolant_specific_heat_j_kg_k
        evaporator_outlet = float(
            tank_temperature_before
            - q_evap_applied / (total_mass_flow * coolant_cp)
        )

        # 5. Supply transport delay
        cluster_supply = self.supply_delay.step(evaporator_outlet)

        # 6. Five-Pack Cluster
        cluster_result = self.cluster.step(
            dt_s=dt_s,
            cluster_current_a=cluster_current_a,
            supply_temperature_k=cluster_supply,
            total_mass_flow_kg_s=total_mass_flow,
            ambient_temperature_k=ambient_temperature_k,
            direction=direction,
        )
        cluster_return = float(cluster_result["return_temperature_k"])

        # 7. Return transport delay
        tank_return = self.return_delay.step(cluster_return)

        # 8. Coolant tank
        tank_result = self.tank.step(
            dt_s=dt_s,
            return_temperature_k=tank_return,
            mass_flow_kg_s=total_mass_flow,
        )

        # 9. Diagnostics
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

        # 9b. Electrical input accounting. Reporting only: none of these
        # three terms re-enter the refrigerant or coolant equations, so every
        # frozen physics result is bit-for-bit unchanged.
        fan_power = condenser_fan_power_w(fan_speed_rpm)
        pump_power = float(operating_point["pump_power_w"])
        compressor_electrical_power = shaft_power / DRIVE_TRAIN_EFFICIENCY
        electrical_power = compressor_electrical_power + fan_power + pump_power
        electrical_cop = (
            q_evap_applied / electrical_power if electrical_power > 0.0 else 0.0
        )

        scalar_diagnostics = np.asarray(
            [
                tank_temperature_before,
                evaporator_outlet,
                cluster_supply,
                cluster_return,
                tank_return,
                tank_result["tank_temperature_after_k"],
                q_evap_cycle,
                q_evap_applied,
                q_cluster_to_fluid,
                q_gen,
                refrigerant_power,
                shaft_power,
                evaporator_result["evaporator_buffer_energy_j"],
            ],
            dtype=float,
        )
        queue_values = np.asarray(
            self.supply_delay.queue_values + self.return_delay.queue_values,
            dtype=float,
        )

        return ClusterPlantOutputs({
            **operating_point,
            "compressor_speed_command_rpm": float(
                actuator_result["speed_command_rpm"]
            ),
            "compressor_speed_before_rpm": float(
                actuator_result["speed_before_rpm"]
            ),
            "compressor_speed_rpm": compressor_speed_used_rpm,
            "low_pressure_unload_factor": (
                compressor_speed_used_rpm
                / float(actuator_result["speed_after_rpm"])
            ),
            "fan_speed_rpm": float(fan_speed_rpm),
            "q_evap_cycle_w": q_evap_cycle,
            "q_evap_applied_w": q_evap_applied,
            "q_condenser_cycle_w": float(
                refrigeration_result["q_condenser_w"]
            ),
            "refrigerant_compression_power_w": refrigerant_power,
            "compressor_shaft_power_w": shaft_power,
            "condenser_fan_power_w": fan_power,
            "compressor_electrical_power_w": compressor_electrical_power,
            "electrical_power_w": electrical_power,
            "electrical_cop": electrical_cop,
            "evaporator_buffer_energy_j": float(
                evaporator_result["evaporator_buffer_energy_j"]
            ),
            "tank_temperature_before_k": tank_temperature_before,
            "evaporator_outlet_temperature_k": evaporator_outlet,
            "cluster_supply_temperature_k": float(cluster_supply),
            "cluster_return_temperature_k": cluster_return,
            "tank_return_temperature_k": float(tank_return),
            "tank_temperature_after_k": float(
                tank_result["tank_temperature_after_k"]
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
            "evaporator_coolant_residual_w": float(
                q_evap_applied - q_evap_from_coolant
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
            "q_tank_w": float(tank_result["return_to_tank_heat_w"]),
            "tank_energy_residual_j": float(
                tank_result["tank_energy_residual_j"]
            ),
            "cluster_q_gen_total_w": q_gen,
            "cross_delay_energy_balance_is_modeled": False,
            "refrigeration_solver_success": bool(
                refrigeration_result["solver_success"]
            ),
            "all_states_finite": bool(
                np.all(np.isfinite(scalar_diagnostics))
                and np.all(np.isfinite(queue_values))
                and np.all(
                    np.isfinite(cluster_result["pack_mass_flows_kg_s"])
                )
            ),
            "supply_delay_queue_k": self.supply_delay.queue_values,
            "return_delay_queue_k": self.return_delay.queue_values,
            "refrigeration_result": refrigeration_result,
            "cluster_result": cluster_result,
            "tank_result": tank_result,
        })


ClusterPlantV2 = ClusterPlant

__all__ = [
    "ClusterPlant",
    "ClusterPlantInputs",
    "ClusterPlantOutputs",
    "ClusterPlantV2",
]
