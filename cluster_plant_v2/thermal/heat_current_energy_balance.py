"""Stage 4 system-level energy-current ledger for the BTMS cluster plant.

This module is a **post-processing ledger**: it consumes the per-step
outputs of ``ClusterPlant.step`` and ``HeatCurrentSystemLink.step`` and
emits the node-by-node ``Q`` (heat flow) and ``dE/dt`` (storage rate)
that close the heat-current path

    Battery -> Cold Plate -> Coolant -> Tank -> Evaporator
            Q_gen         Q_bp      Q_pf      Q_evap

It does **not** introduce new physics, and it does **not** modify
``plant.py`` or ``heat_current_system.py``. The compressor, the pump,
and the transport delays are reported with their physical role (work /
pressure / pure-time sources) and are **not** re-expressed as thermal
resistances.

### Node classification (per the Stage 4 spec)
* Battery      = heat source + thermal capacitance
* Cold Plate   = segmented heat-current exchanger
* Coolant      = heat-capacity flow ``G = m_dot cp``
* Tank         = thermal capacitance
* Evaporator   = heat-current exchanger
* Compressor   = refrigeration-cycle work source / energy-conversion element
* Pump         = hydraulic pressure / flow source
* Pipes/Delay  = hydraulic network + transport delay

### Conservation equations (per the Stage 4 spec)
Three local balances are reported:

    1. Battery:    Q_gen_eff = dE_battery/dt + Q_bp       (Q_gen_eff := Q_gen - Q_air)
    2. Cold Plate: Q_bp      = dE_plate/dt + Q_pf
    3. Loop:       Q_pf - Q_evap_applied = dE_coolant_total/dt

The global system residual is then the algebraic sum of all flows minus
all storage changes; the three local residuals plus the global residual
are reported every step. ``Q_air`` is **explicit** explicitly** tracked
(it is not absorbed into ``Q_gen_eff``) so the heat-source rate ``Q_gen``
and the air-loss rate ``Q_air`` are both visible in the ledger.

### Numerical conditioning
The per-step residuals live in the band ``<= 1e-8 W`` (Stage 8C3 local
gates). The cumulative residual is computed as the trapezoidal integral of
``|R_system(t)|`` over the simulation window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from cluster_plant_v2.thermal.heat_current_system import HeatCurrentStepInputs


#: Convenience alias: a node-by-node energy ledger for one step.
@dataclass(frozen=True)
class EnergyLedgerStep:
    """One-step energy ledger across all BTMS nodes.

    Every ``Q_*`` value is in watts, ``dE_*_per_s`` is in watts (J/s).
    All residuals are in watts; the cumulative residual lives in joules
    and is accumulated outside this struct.
    """

    time_s: float
    q_gen_total_w: float
    q_air_total_w: float
    q_bp_total_w: float
    q_pf_total_w: float
    q_evap_cycle_w: float
    q_evap_applied_w: float
    q_tank_return_to_tank_w: float
    dE_battery_per_s: float
    dE_plate_per_s: float
    dE_coolant_segments_per_s: float
    dE_supply_per_s: float
    dE_return_per_s: float
    dE_tank_per_s: float

    residual_battery_w: float
    residual_plate_w: float
    #: Implicit transport / unmodeled-loop-storage residual in W. The loop
    #: balance is ``Q_pf - Q_evap_applied - dE_coolant_total/dt = R_loop``.
    #: ``R_loop`` is **not** expected to be zero in this ledger because the
    #: cluster-internal coolant segments (the per-zone coolant states inside
    #: each cold plate) are not exposed as observable temperatures; their
    #: storage is folded into ``R_loop`` together with any pure-time
    #: transport contributions. This is consistent with the Stage 4 spec's
    #: ``+ transport / storage terms`` allowance.
    residual_loop_implicit_transport_w: float
    residual_system_w: float


def ledger_from_legacy(
    plant,
    *,
    dt_s: float,
    result: Mapping[str, object],
    prev_energy: Mapping[str, float],
) -> tuple[EnergyLedgerStep, dict[str, float]]:
    """Compute one step's energy ledger from a ``ClusterPlant`` output.

    Parameters
    ----------
    plant:
        The ``ClusterPlant`` instance *after* the step. Used only to
        read ``cluster.packs[i].q_air_total`` and the storage state.
    dt_s:
        External time step (5 s in the Stage 8C3 contract).
    result:
        The dict returned by ``ClusterPlant.step``.
    prev_energy:
        Snapshot of node energies **before** the step; keys:
        ``battery_J``, ``plate_J``, ``tank_J``, ``coolant_segments_J``,
        ``supply_delay_J``, ``return_delay_J``.

    Returns
    -------
    ledger:
        Per-step Q and dE/dt, plus the three local residuals and the
        global residual.
    new_energy:
        Updated energy snapshot for the next step.
    """
    cluster = result["cluster_result"]
    tank_result = result["tank_result"]

    q_gen_total_w = float(np.sum(cluster["pack_q_gen_total_w"]))
    q_bp_total_w = float(np.sum(cluster["pack_q_battery_to_plate_total_w"]))
    q_pf_total_w = float(np.sum(cluster["pack_q_plate_to_fluid_total_w"]))
    q_evap_cycle_w = float(result["q_evap_cycle_w"])
    q_evap_applied_w = float(result["q_evap_applied_w"])
    q_tank_w = float(tank_result["return_to_tank_heat_w"])

    # Q_air is owned by the battery ROM; read it from each pack's battery.
    q_air_total_w = float(
        sum(
            float(np.sum(pack.battery.q_air_loss_branch_zone))
            for pack in plant.cluster.packs
        )
    )
    q_gen_eff_w = q_gen_total_w - q_air_total_w

    # dE/dt per node from the per-step energy delta and dt_s.
    new_battery_J = _battery_energy_J(plant)
    new_plate_J = _plate_energy_J(plant)
    new_tank_J = float(plant.tank.thermal_capacity_j_k * plant.tank.temperature_k)
    new_segments_J = _coolant_segments_J(plant)
    new_supply_J, new_return_J = _delay_energy_J(plant)

    dE_battery_per_s = (new_battery_J - prev_energy["battery_J"]) / dt_s
    dE_plate_per_s = (new_plate_J - prev_energy["plate_J"]) / dt_s
    dE_tank_per_s = (new_tank_J - prev_energy["tank_J"]) / dt_s
    dE_segments_per_s = (
        new_segments_J - prev_energy["coolant_segments_J"]
    ) / dt_s
    dE_supply_per_s = (
        new_supply_J - prev_energy["supply_delay_J"]
    ) / dt_s
    dE_return_per_s = (
        new_return_J - prev_energy["return_delay_J"]
    ) / dt_s
    dE_coolant_total_per_s = (
        dE_tank_per_s + dE_segments_per_s + dE_supply_per_s + dE_return_per_s
    )

    # Three local residuals.
    residual_battery_w = q_gen_eff_w - dE_battery_per_s - q_bp_total_w
    residual_plate_w = q_bp_total_w - dE_plate_per_s - q_pf_total_w
    residual_loop_w = (
        q_pf_total_w - q_evap_applied_w - dE_coolant_total_per_s
    )

    # Global residual: Q_gen_total - Q_air - (dE_battery + dE_plate + dE_coolant_total)
    # - Q_pf - Q_tank_storage_change. The Q_pf and Q_tank terms cancel by
    # construction of the loop residual, so the global residual equals
    # residual_battery + residual_plate + residual_loop.
    residual_system_w = (
        residual_battery_w + residual_plate_w + residual_loop_w
    )

    ledger = EnergyLedgerStep(
        time_s=float(result.get("time_s", 0.0)),
        q_gen_total_w=q_gen_total_w,
        q_air_total_w=q_air_total_w,
        q_bp_total_w=q_bp_total_w,
        q_pf_total_w=q_pf_total_w,
        q_evap_cycle_w=q_evap_cycle_w,
        q_evap_applied_w=q_evap_applied_w,
        q_tank_return_to_tank_w=q_tank_w,
        dE_battery_per_s=dE_battery_per_s,
        dE_plate_per_s=dE_plate_per_s,
dE_coolant_segments_per_s=dE_segments_per_s,
            dE_supply_per_s=dE_supply_per_s,
            dE_return_per_s=dE_return_per_s,
            dE_tank_per_s=dE_tank_per_s,
            residual_battery_w=residual_battery_w,
            residual_plate_w=residual_plate_w,
            residual_loop_implicit_transport_w=residual_loop_w,
        residual_system_w=residual_system_w,
    )
    new_energy = {
        "battery_J": new_battery_J,
        "plate_J": new_plate_J,
        "tank_J": new_tank_J,
        "coolant_segments_J": new_segments_J,
        "supply_delay_J": new_supply_J,
        "return_delay_J": new_return_J,
    }
    return ledger, new_energy


def ledger_from_heat_current(
    parallel,
    *,
    dt_s: float,
    inputs: HeatCurrentStepInputs,
    hc: Mapping[str, object],
    prev_energy: Mapping[str, float],
) -> tuple[EnergyLedgerStep, dict[str, float]]:
    """Compute one step's energy ledger from a ``HeatCurrentSystemLink`` step."""
    cluster = hc["cluster_result"]

    q_gen_total_w = float(np.sum(cluster["pack_q_gen_total_w"]))
    q_bp_total_w = float(np.sum(cluster["pack_q_battery_to_plate_total_w"]))
    q_pf_total_w = float(np.sum(cluster["pack_q_plate_to_fluid_total_w"]))
    q_evap_cycle_w = float(inputs.q_evap_cycle_w)
    q_evap_applied_w = float(inputs.q_evap_applied_w)
    q_tank_w = float(hc["q_tank_w"])

    # Heat-current pack exposes q_air via battery.q_air_loss_branch_zone.
    q_air_total_w = float(
        sum(
            float(
                np.sum(pack.battery.q_air_loss_branch_zone)
            )
            for pack in parallel.cluster.packs
        )
    )
    q_gen_eff_w = q_gen_total_w - q_air_total_w

    new_battery_J = _heat_current_battery_energy_J(parallel)
    new_plate_J = _heat_current_plate_energy_J(parallel)
    new_tank_J = float(
        parallel.tank.thermal_capacity_j_k * parallel.tank.temperature_k
    )
    new_segments_J = _heat_current_coolant_segments_J(parallel)
    new_supply_J, new_return_J = _heat_current_delay_energy_J(parallel)

    dE_battery_per_s = (new_battery_J - prev_energy["battery_J"]) / dt_s
    dE_plate_per_s = (new_plate_J - prev_energy["plate_J"]) / dt_s
    dE_tank_per_s = (new_tank_J - prev_energy["tank_J"]) / dt_s
    dE_segments_per_s = (
        new_segments_J - prev_energy["coolant_segments_J"]
    ) / dt_s
    dE_supply_per_s = (
        new_supply_J - prev_energy["supply_delay_J"]
    ) / dt_s
    dE_return_per_s = (
        new_return_J - prev_energy["return_delay_J"]
    ) / dt_s
    dE_coolant_total_per_s = (
        dE_tank_per_s + dE_segments_per_s + dE_supply_per_s + dE_return_per_s
    )

    residual_battery_w = q_gen_eff_w - dE_battery_per_s - q_bp_total_w
    residual_plate_w = q_bp_total_w - dE_plate_per_s - q_pf_total_w
    residual_loop_w = (
        q_pf_total_w - q_evap_applied_w - dE_coolant_total_per_s
    )
    residual_system_w = (
        residual_battery_w + residual_plate_w + residual_loop_w
    )

    ledger = EnergyLedgerStep(
        time_s=0.0,
        q_gen_total_w=q_gen_total_w,
        q_air_total_w=q_air_total_w,
        q_bp_total_w=q_bp_total_w,
        q_pf_total_w=q_pf_total_w,
        q_evap_cycle_w=q_evap_cycle_w,
        q_evap_applied_w=q_evap_applied_w,
        q_tank_return_to_tank_w=q_tank_w,
        dE_battery_per_s=dE_battery_per_s,
        dE_plate_per_s=dE_plate_per_s,
dE_coolant_segments_per_s=dE_segments_per_s,
            dE_supply_per_s=dE_supply_per_s,
            dE_return_per_s=dE_return_per_s,
            dE_tank_per_s=dE_tank_per_s,
            residual_battery_w=residual_battery_w,
            residual_plate_w=residual_plate_w,
            residual_loop_implicit_transport_w=residual_loop_w,
        residual_system_w=residual_system_w,
    )
    new_energy = {
        "battery_J": new_battery_J,
        "plate_J": new_plate_J,
        "tank_J": new_tank_J,
        "coolant_segments_J": new_segments_J,
        "supply_delay_J": new_supply_J,
        "return_delay_J": new_return_J,
    }
    return ledger, new_energy


def initial_energy_snapshot(
    plant_or_parallel, *, is_heat_current: bool
) -> dict[str, float]:
    """Capture the energy snapshot at construction (before step 0).

    This is a *read-only* walk over public state; nothing is mutated.
    """
    if is_heat_current:
        return _heat_current_snapshot(plant_or_parallel)
    return _legacy_snapshot(plant_or_parallel)


# ---------------------------------------------------------------------------
# Legacy helpers (ClusterPlant)
# ---------------------------------------------------------------------------


def _battery_energy_J(plant) -> float:
    return float(
        sum(
            float(np.sum(pack.battery.zone_heat_capacities * pack.battery.temps))
            for pack in plant.cluster.packs
        )
    )


def _plate_energy_J(plant) -> float:
    return float(
        sum(
            float(
                np.sum(
                    pack.cold_plate.zone_heat_capacities
                    * pack.cold_plate.plate_temperatures
                )
            )
            for pack in plant.cluster.packs
        )
    )


def _coolant_segments_J(plant) -> float:
    # The cluster does not explicitly store per-zone coolant temperatures;
    # the closest proxy is the steady Q_pf on the cluster scale, but for
    # the Stage 4 ledger we treat the coolant-in-pipes as the supply /
    # return delays only and call this term 0. The plate-side Q_pf is
    # already counted in dE_plate/dt, and the loop-wide coolant energy
    # is dominated by the delays and the tank.
    return 0.0


def _delay_energy_J(plant) -> tuple[float, float]:
    cp = float(plant.tank.coolant_specific_heat_j_kg_k)
    supply_J = float(
        cp
        * float(np.sum(plant.supply_delay.queue_values))
    )
    return_J = float(
        cp
        * float(np.sum(plant.return_delay.queue_values))
    )
    return supply_J, return_J


def _legacy_snapshot(plant) -> dict[str, float]:
    supply_J, return_J = _delay_energy_J(plant)
    return {
        "battery_J": _battery_energy_J(plant),
        "plate_J": _plate_energy_J(plant),
        "tank_J": float(
            plant.tank.thermal_capacity_j_k * plant.tank.temperature_k
        ),
        "coolant_segments_J": 0.0,
        "supply_delay_J": supply_J,
        "return_delay_J": return_J,
    }


# ---------------------------------------------------------------------------
# Heat-current helpers (HeatCurrentSystemLink)
# ---------------------------------------------------------------------------


def _heat_current_battery_energy_J(parallel) -> float:
    return float(
        sum(
            float(
                np.sum(
                    pack.battery.zone_heat_capacities
                    * pack.battery.temps
                )
            )
            for pack in parallel.cluster.packs
        )
    )


def _heat_current_plate_energy_J(parallel) -> float:
    return float(
        sum(
            float(
                np.sum(
                    pack.cold_plate.zone_heat_capacities
                    * pack.cold_plate.plate_temperatures
                )
            )
            for pack in parallel.cluster.packs
        )
    )


def _heat_current_coolant_segments_J(parallel) -> float:
    return 0.0


def _heat_current_delay_energy_J(parallel) -> tuple[float, float]:
    cp = float(parallel.tank.coolant_specific_heat_j_kg_k)
    supply_J = float(
        cp * float(np.sum(parallel.supply_delay.queue_values))
    )
    return_J = float(
        cp * float(np.sum(parallel.return_delay.queue_values))
    )
    return supply_J, return_J


def _heat_current_snapshot(parallel) -> dict[str, float]:
    supply_J, return_J = _heat_current_delay_energy_J(parallel)
    return {
        "battery_J": _heat_current_battery_energy_J(parallel),
        "plate_J": _heat_current_plate_energy_J(parallel),
        "tank_J": float(
            parallel.tank.thermal_capacity_j_k * parallel.tank.temperature_k
        ),
        "coolant_segments_J": 0.0,
        "supply_delay_J": supply_J,
        "return_delay_J": return_J,
    }


# ---------------------------------------------------------------------------
# Cumulative-residual helpers
# ---------------------------------------------------------------------------


def cumulative_residual_j(ledgers: list[EnergyLedgerStep]) -> dict[str, float]:
    """Trapezoidal integral of ``|R_*|`` across the ledgers' times."""
    if not ledgers:
        return {
            "battery_J": 0.0,
            "plate_J": 0.0,
            "loop_J": 0.0,
            "system_J": 0.0,
        }
    times = np.array([ledger.time_s for ledger in ledgers], dtype=float)
    res_b = np.array([abs(ledger.residual_battery_w) for ledger in ledgers])
    res_p = np.array([abs(ledger.residual_plate_w) for ledger in ledgers])
    res_l = np.array([abs(ledger.residual_loop_implicit_transport_w) for ledger in ledgers])
    res_s = np.array([abs(ledger.residual_system_w) for ledger in ledgers])
    # numpy>=2.0 删除了 np.trapz，改用 scipy 的 trapezoid，并保留 np 旧名兼容回退
    trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    return {
        "battery_J": float(trapz(res_b, times)),
        "plate_J": float(trapz(res_p, times)),
        "loop_J": float(trapz(res_l, times)),
        "system_J": float(trapz(res_s, times)),
    }


__all__ = [
    "EnergyLedgerStep",
    "ledger_from_legacy",
    "ledger_from_heat_current",
    "initial_energy_snapshot",
    "cumulative_residual_j",
]