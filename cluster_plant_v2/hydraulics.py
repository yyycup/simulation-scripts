"""Cluster coolant hydraulics, pump, and tank models.

PyCharm navigation:
- Hydraulic network: ``ParallelHeaderHydraulicNetwork``
- Coolant pump: ``CoolantPump``
- Coolant tank: ``CoolantTank``
- Shared defaults: ``parameters.py``
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, least_squares

from cluster_plant_v2.parameters import (
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    BRANCH_DESIGN_DELTA_P_PA,
    COOLANT_DENSITY_KG_M3,
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    DEFAULT_CLUSTER_PACK_COUNT,
    DEFAULT_SHUTOFF_TO_OPERATING_PRESSURE_RATIO,
    DESIGN_PACK_MASS_FLOW_KG_S,
    LEGACY_REFERENCE_FLOW_L_MIN,
    LEGACY_REFERENCE_POWER_W,
    LEGACY_REFERENCE_SPEED_RPM,
    LEGACY_TANK_VOLUME_L,
    MAX_PUMP_SPEED_RPM,
    MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO,
    MIN_PUMP_SPEED_RPM,
    SMALL_HEADER_TO_BRANCH_RESISTANCE_RATIO,
)


# Hydraulic Network

class ParallelHeaderHydraulicNetwork:
    """Solve Pack flows in a reversed-return (homoverse) parallel-header network.

    Supply and return headers run in the same direction and the common outlet
    sits after Pack N, so every Pack path traverses the same number of header
    segments (supply segments 1..i plus return segments i..N).

    Packs are numbered 1..N along both header flow directions. Supply segment
    j ends at Pack j and carries the flows for Packs j..N. Return segment j
    starts at Pack j and carries the flows from Packs 1..j toward the common
    outlet after Pack N. Pack i traverses supply segments 1..i, branch i, and
    return segments i..N.

    All resistance coefficients use Pa/(kg/s)^2. The default coefficients are
    Stage 7B engineering assumptions for structural validation, not measured
    pipe data.
    """

    dynamic_state_count = 0

    def __init__(
        self,
        n_packs: int = DEFAULT_CLUSTER_PACK_COUNT,
        branch_resistances: float | np.ndarray | list[float] | None = None,
        supply_segment_resistances: float | np.ndarray | list[float] | None = None,
        return_segment_resistances: float | np.ndarray | list[float] | None = None,
    ) -> None:
        if (
            isinstance(n_packs, bool)
            or not isinstance(n_packs, (int, np.integer))
            or int(n_packs) < 1
        ):
            raise ValueError("n_packs must be an integer greater than or equal to one")
        self.n_packs = int(n_packs)
        default_header = (
            BASE_BRANCH_RESISTANCE_PA_PER_KG_S2
            * SMALL_HEADER_TO_BRANCH_RESISTANCE_RATIO
        )
        self.branch_resistances = self._resistance_array(
            branch_resistances,
            BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
            "branch_resistances",
            allow_zero=False,
        )
        self.supply_segment_resistances = self._resistance_array(
            supply_segment_resistances,
            default_header,
            "supply_segment_resistances",
            allow_zero=True,
        )
        self.return_segment_resistances = self._resistance_array(
            return_segment_resistances,
            default_header,
            "return_segment_resistances",
            allow_zero=True,
        )

    def _resistance_array(
        self,
        value: float | np.ndarray | list[float] | None,
        default: float,
        name: str,
        *,
        allow_zero: bool,
    ) -> np.ndarray:
        if value is None:
            array = np.full(self.n_packs, default, dtype=float)
        else:
            array = np.asarray(value, dtype=float)
            if array.ndim == 0:
                array = np.full(self.n_packs, float(array), dtype=float)
        invalid_sign = np.any(array < 0.0) if allow_zero else np.any(array <= 0.0)
        if (
            array.shape != (self.n_packs,)
            or not np.all(np.isfinite(array))
            or invalid_sign
        ):
            qualifier = "nonnegative" if allow_zero else "positive"
            raise ValueError(
                f"{name} must be finite, {qualifier}, and shape (n_packs,)"
            )
        return array.copy()

    def _quantities(self, pack_mass_flows: np.ndarray) -> dict[str, np.ndarray]:
        flows = np.asarray(pack_mass_flows, dtype=float).reshape(self.n_packs)
        supply_flows = np.cumsum(flows[::-1])[::-1]
        return_flows = np.cumsum(flows)
        supply_delta_p = self.supply_segment_resistances * supply_flows**2
        return_delta_p = self.return_segment_resistances * return_flows**2
        branch_delta_p = self.branch_resistances * flows**2
        supply_path_delta_p = np.cumsum(supply_delta_p)
        return_path_delta_p = np.cumsum(return_delta_p[::-1])[::-1]
        path_delta_p = (
            supply_path_delta_p + branch_delta_p + return_path_delta_p
        )

        supply_nodes = np.empty(self.n_packs, dtype=float)
        return_nodes = np.empty(self.n_packs, dtype=float)
        for index in range(self.n_packs):
            supply_downstream = (
                supply_flows[index + 1] if index + 1 < self.n_packs else 0.0
            )
            return_upstream = return_flows[index - 1] if index > 0 else 0.0
            supply_nodes[index] = (
                supply_flows[index] - flows[index] - supply_downstream
            )
            return_nodes[index] = (
                return_upstream + flows[index] - return_flows[index]
            )
        return {
            "supply_segment_flows": supply_flows,
            "return_segment_flows": return_flows,
            "supply_segment_delta_p": supply_delta_p,
            "return_segment_delta_p": return_delta_p,
            "branch_delta_p": branch_delta_p,
            "pack_path_delta_p": path_delta_p,
            "node_mass_balance_residuals": np.concatenate(
                (supply_nodes, return_nodes)
            ),
        }

    def solve(self, total_mass_flow_kg_s: float) -> dict[str, object]:
        """Solve N positive branch flows and one common network pressure drop."""
        total = float(total_mass_flow_kg_s)
        if not np.isfinite(total) or total < 0.0:
            raise ValueError("total_mass_flow_kg_s must be finite and nonnegative")
        if total == 0.0:
            zeros_n = np.zeros(self.n_packs, dtype=float)
            return {
                "pack_mass_flows": zeros_n.copy(),
                "network_delta_p": 0.0,
                "supply_segment_flows": zeros_n.copy(),
                "return_segment_flows": zeros_n.copy(),
                "supply_segment_delta_p": zeros_n.copy(),
                "return_segment_delta_p": zeros_n.copy(),
                "branch_delta_p": zeros_n.copy(),
                "pack_path_delta_p": zeros_n.copy(),
                "total_mass_balance_residual": 0.0,
                "node_mass_balance_residuals": np.zeros(2 * self.n_packs),
                "path_pressure_residuals": zeros_n.copy(),
                "solver_success": True,
                "solver_iterations": 0,
                "solver_message": "zero-flow algebraic solution",
            }

        initial_flows = np.full(self.n_packs, total / self.n_packs)
        initial_quantities = self._quantities(initial_flows)
        initial_delta_p = float(initial_quantities["pack_path_delta_p"].mean())
        pressure_scale = max(initial_delta_p, 1.0)
        initial = np.concatenate((initial_flows, [initial_delta_p]))
        lower_flow = max(total * 1e-12, np.finfo(float).tiny)
        lower = np.concatenate(
            (np.full(self.n_packs, lower_flow), [0.0])
        )
        upper = np.concatenate((np.full(self.n_packs, total), [np.inf]))

        def residuals(unknowns: np.ndarray) -> np.ndarray:
            flows = unknowns[: self.n_packs]
            common_delta_p = unknowns[-1]
            path_delta_p = self._quantities(flows)["pack_path_delta_p"]
            return np.concatenate(
                (
                    (path_delta_p - common_delta_p) / pressure_scale,
                    [(flows.sum() - total) / total],
                )
            )

        solution = least_squares(
            residuals,
            initial,
            bounds=(lower, upper),
            x_scale="jac",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=2000,
        )
        flows = np.asarray(solution.x[: self.n_packs], dtype=float)
        common_delta_p = float(solution.x[-1])
        quantities = self._quantities(flows)
        total_residual = float(flows.sum() - total)
        path_residuals = quantities["pack_path_delta_p"] - common_delta_p
        if not (
            solution.success
            and np.all(np.isfinite(solution.x))
            and abs(total_residual) <= 1e-10
            and np.max(np.abs(path_residuals)) <= 1e-3
        ):
            raise RuntimeError(
                "parallel-header hydraulic solve failed: "
                f"{solution.message}; mass residual={total_residual:.6g} kg/s; "
                f"path residual={np.max(np.abs(path_residuals)):.6g} Pa"
            )
        return {
            "pack_mass_flows": flows,
            "network_delta_p": common_delta_p,
            **quantities,
            "total_mass_balance_residual": total_residual,
            "path_pressure_residuals": path_residuals,
            "solver_success": bool(solution.success),
            "solver_iterations": int(solution.nfev),
            "solver_message": str(solution.message),
        }


# Coolant Pump

LEGACY_COOLANT_DENSITY_KG_M3 = COOLANT_DENSITY_KG_M3


class CoolantPump:
    """Parabolic pump curve scaled with the pump affinity laws.

    The reference speed, flow, and power come from the legacy model in
    ``thermal_system.py``. That model has no head curve, so the reference
    pressure and shutoff-pressure ratio are explicit Stage 8A engineering
    assumptions rather than measured pump-map data.
    """

    dynamic_state_count = 0

    def __init__(
        self,
        reference_operating_delta_p_pa: float,
        *,
        reference_speed_rpm: float = LEGACY_REFERENCE_SPEED_RPM,
        reference_volume_flow_l_min: float = LEGACY_REFERENCE_FLOW_L_MIN,
        reference_power_w: float = LEGACY_REFERENCE_POWER_W,
        coolant_density_kg_m3: float = LEGACY_COOLANT_DENSITY_KG_M3,
        shutoff_to_operating_pressure_ratio: float = (
            DEFAULT_SHUTOFF_TO_OPERATING_PRESSURE_RATIO
        ),
        minimum_speed_rpm: float = MIN_PUMP_SPEED_RPM,
        maximum_speed_rpm: float = MAX_PUMP_SPEED_RPM,
    ) -> None:
        values = np.array(
            [
                reference_operating_delta_p_pa,
                reference_speed_rpm,
                reference_volume_flow_l_min,
                reference_power_w,
                coolant_density_kg_m3,
                minimum_speed_rpm,
                maximum_speed_rpm,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("pump reference parameters must be finite and positive")
        if maximum_speed_rpm <= minimum_speed_rpm:
            raise ValueError("maximum_speed_rpm must exceed minimum_speed_rpm")
        ratio = float(shutoff_to_operating_pressure_ratio)
        if not np.isfinite(ratio) or ratio <= 1.0:
            raise ValueError(
                "shutoff_to_operating_pressure_ratio must be finite and greater than one"
            )

        self.reference_operating_delta_p_pa = float(
            reference_operating_delta_p_pa
        )
        self.reference_speed_rpm = float(reference_speed_rpm)
        self.reference_volume_flow_l_min = float(reference_volume_flow_l_min)
        self.reference_power_w = float(reference_power_w)
        self.coolant_density_kg_m3 = float(coolant_density_kg_m3)
        self.shutoff_to_operating_pressure_ratio = ratio
        self.minimum_speed_rpm = float(minimum_speed_rpm)
        self.maximum_speed_rpm = float(maximum_speed_rpm)
        self.reference_mass_flow_kg_s = (
            self.reference_volume_flow_l_min
            / 1000.0
            / 60.0
            * self.coolant_density_kg_m3
        )
        self.reference_shutoff_delta_p_pa = (
            ratio * self.reference_operating_delta_p_pa
        )
        self.reference_free_flow_kg_s = self.reference_mass_flow_kg_s / math.sqrt(
            1.0 - 1.0 / ratio
        )

    def _speed_ratio(self, speed_rpm: float) -> float:
        speed = float(speed_rpm)
        if (
            not np.isfinite(speed)
            or speed < self.minimum_speed_rpm
            or speed > self.maximum_speed_rpm
        ):
            raise ValueError(
                "speed_rpm must be finite and within the audited pump range"
            )
        return speed / self.reference_speed_rpm

    def pressure_rise_pa(self, mass_flow_kg_s: float, speed_rpm: float) -> float:
        """Return pump pressure rise from the affinity-scaled parabola."""
        flow = float(mass_flow_kg_s)
        if not np.isfinite(flow) or flow < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and nonnegative")
        speed_ratio = self._speed_ratio(speed_rpm)
        free_flow = self.reference_free_flow_kg_s * speed_ratio
        return float(
            self.reference_shutoff_delta_p_pa
            * speed_ratio**2
            * (1.0 - (flow / free_flow) ** 2)
        )

    def power_w(self, speed_rpm: float) -> float:
        """Return shaft/electrical power diagnostic using P proportional to N cubed."""
        return float(self.reference_power_w * self._speed_ratio(speed_rpm) ** 3)

    def solve_operating_point(self, speed_rpm: float, hydraulic_network) -> dict:
        """Find positive flow where pump and Stage 7B network pressures match."""
        speed_ratio = self._speed_ratio(speed_rpm)
        upper_flow = self.reference_free_flow_kg_s * speed_ratio

        def residual(flow: float) -> float:
            network_delta_p = float(
                hydraulic_network.solve(flow)["network_delta_p"]
            )
            return self.pressure_rise_pa(flow, speed_rpm) - network_delta_p

        lower_residual = residual(0.0)
        upper_residual = residual(upper_flow)
        if not (
            np.isfinite(lower_residual)
            and np.isfinite(upper_residual)
            and lower_residual > 0.0
            and upper_residual < 0.0
        ):
            raise RuntimeError(
                "no pump-network operating point within the positive pump curve"
            )

        mass_flow, root_result = brentq(
            residual,
            0.0,
            upper_flow,
            xtol=1e-12,
            rtol=1e-12,
            maxiter=100,
            full_output=True,
            disp=False,
        )
        network_result = hydraulic_network.solve(mass_flow)
        network_delta_p = float(network_result["network_delta_p"])
        pump_delta_p = self.pressure_rise_pa(mass_flow, speed_rpm)
        pressure_residual = pump_delta_p - network_delta_p
        if not root_result.converged or abs(pressure_residual) > 1e-5:
            raise RuntimeError(
                "pump-network operating-point solve failed: "
                f"pressure residual={pressure_residual:.6g} Pa"
            )
        volume_flow_l_min = (
            mass_flow / self.coolant_density_kg_m3 * 1000.0 * 60.0
        )
        return {
            "pump_speed_rpm": float(speed_rpm),
            "total_mass_flow_kg_s": float(mass_flow),
            "total_volume_flow_l_min": float(volume_flow_l_min),
            "pump_delta_p_pa": float(pump_delta_p),
            "network_delta_p_pa": network_delta_p,
            "pressure_balance_residual_pa": float(pressure_residual),
            "pump_power_w": self.power_w(speed_rpm),
            "solver_success": bool(root_result.converged),
            "solver_iterations": int(root_result.iterations),
        }


# Coolant Tank

LEGACY_COOLANT_SPECIFIC_HEAT_J_KG_K = COOLANT_SPECIFIC_HEAT_J_KG_K


class CoolantTank:
    """Advance one well-mixed coolant temperature with explicit Euler."""

    dynamic_state_count = 1

    def __init__(
        self,
        initial_temperature_k: float,
        *,
        volume_l: float = LEGACY_TANK_VOLUME_L,
        coolant_density_kg_m3: float = LEGACY_COOLANT_DENSITY_KG_M3,
        coolant_specific_heat_j_kg_k: float = (
            LEGACY_COOLANT_SPECIFIC_HEAT_J_KG_K
        ),
    ) -> None:
        values = np.array(
            [
                initial_temperature_k,
                volume_l,
                coolant_density_kg_m3,
                coolant_specific_heat_j_kg_k,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("tank temperature and physical parameters must be positive")
        self.temperature_k = float(initial_temperature_k)
        self.volume_m3 = float(volume_l) / 1000.0
        self.coolant_density_kg_m3 = float(coolant_density_kg_m3)
        self.coolant_specific_heat_j_kg_k = float(
            coolant_specific_heat_j_kg_k
        )
        self.thermal_capacity_j_k = (
            self.volume_m3
            * self.coolant_density_kg_m3
            * self.coolant_specific_heat_j_kg_k
        )

    def step(
        self,
        dt_s: float,
        return_temperature_k: float,
        mass_flow_kg_s: float,
    ) -> dict[str, float]:
        """Use the old tank state and current return enthalpy for one step."""
        dt = float(dt_s)
        return_temperature = float(return_temperature_k)
        mass_flow = float(mass_flow_kg_s)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        if not np.isfinite(return_temperature) or return_temperature <= 0.0:
            raise ValueError("return_temperature_k must be finite and positive")
        if not np.isfinite(mass_flow) or mass_flow < 0.0:
            raise ValueError("mass_flow_kg_s must be finite and nonnegative")

        temperature_before = self.temperature_k
        heat_rate = (
            mass_flow
            * self.coolant_specific_heat_j_kg_k
            * (return_temperature - temperature_before)
        )
        advected_energy = heat_rate * dt
        temperature_after = (
            temperature_before + advected_energy / self.thermal_capacity_j_k
        )
        tank_energy_change = (
            self.thermal_capacity_j_k
            * (temperature_after - temperature_before)
        )
        energy_residual = tank_energy_change - advected_energy
        self.temperature_k = float(temperature_after)
        return {
            "tank_temperature_before_k": float(temperature_before),
            "tank_temperature_after_k": float(temperature_after),
            "supply_temperature_k": float(temperature_before),
            "return_temperature_k": return_temperature,
            "mass_flow_kg_s": mass_flow,
            "return_to_tank_heat_w": float(heat_rate),
            "tank_energy_change_j": float(tank_energy_change),
            "advected_energy_j": float(advected_energy),
            "tank_energy_residual_j": float(energy_residual),
        }
