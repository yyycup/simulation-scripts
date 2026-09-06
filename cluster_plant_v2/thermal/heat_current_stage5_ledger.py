"""Stage 5 ledger — thin wrapper around Stage 4 energy-balance for HeatCurrentPlant.

The Stage 4 ``ledger_from_heat_current`` accepts a
``HeatCurrentSystemLink``-shaped ``parallel`` object plus a
``HeatCurrentStepInputs`` snapshot. The Stage 4.5
``HeatCurrentPlant`` exposes the same per-step fields in a different
shape (no separate ``inputs`` object — the q_evap_* and q_tank fields
sit directly on the step return value).

This module bridges the two: it builds a synthetic
``HeatCurrentStepInputs`` from the ``HeatCurrentPlant.step()`` return
dict so the Stage 4 ledger can be reused without rewriting it. The
synthetic inputs carry exactly the same numbers the plant computed;
they do not represent any "second source" of truth.
"""

from __future__ import annotations

from typing import Mapping

from cluster_plant_v2.thermal.heat_current_energy_balance import (
    EnergyLedgerStep,
    ledger_from_heat_current,
)
from cluster_plant_v2.thermal.heat_current_system import (
    HeatCurrentStepInputs,
)


def ledger_from_heat_current_plant(
    hc_plant,
    *,
    dt_s: float,
    step_result: Mapping[str, object],
    prev_energy: Mapping[str, float],
    compressor_speed_used_rpm: float,
    tank_temperature_before_k: float,
    refrigeration_solver_success: bool,
) -> tuple[EnergyLedgerStep, dict[str, float]]:
    """Stage 5 thin bridge: wrap ``HeatCurrentPlant.step()`` output into a
    shape that ``ledger_from_heat_current`` accepts.

    Parameters
    ----------
    hc_plant:
        The ``HeatCurrentPlant`` whose state was advanced this step.
    dt_s:
        Outer step length.
    step_result:
        The dict returned by ``HeatCurrentPlant.step(inputs, dt_s=dt_s)``.
    prev_energy:
        Per-node thermal-energy snapshot from the previous step.
    compressor_speed_used_rpm:
        Final compressor speed used by the cycle solver this step.
    tank_temperature_before_k:
        Tank temperature read at the start of this step.
    refrigeration_solver_success:
        Whether the cycle solver closed this step.

    Returns
    -------
    EnergyLedgerStep, new_energy_dict
    """
    cluster_result = step_result["cluster_result"]
    inputs_synth = HeatCurrentStepInputs(
        dt_s=dt_s,
        cluster_current_a=0.0,
        ambient_temperature_k=0.0,
        direction="forward",
        total_mass_flow_kg_s=0.0,
        tank_temperature_before_k=tank_temperature_before_k,
        q_evap_applied_w=float(step_result["q_evap_applied_w"]),
        q_evap_cycle_w=float(step_result["q_evap_cycle_w"]),
        evaporating_temperature_k=0.0,
        evaporator_ua_w_k=0.0,
        refrigeration_solver_success=refrigeration_solver_success,
        compressor_speed_rpm=compressor_speed_used_rpm,
    )
    # ledger_from_heat_current reads:
    #   q_evap_cycle_w = inputs.q_evap_cycle_w     (line 228)
    #   q_evap_applied_w = inputs.q_evap_applied_w (line 229)
    #   q_tank_w = hc["q_tank_w"]                  (line 230)
    # It also reads hc["cluster_result"] for Q_gen / Q_bp / Q_pf.
    # Build a synthetic dict that adds "q_tank_w" from the plant's
    # tank_result dict. The other keys (cluster_result) come directly
    # from the step result.
    hc_compat = {
        "cluster_result": cluster_result,
        "q_tank_w": float(step_result["tank_result"]["return_to_tank_heat_w"]),
    }
    # Override the legacy-style q_evap fields by also exposing them on the
    # dict (defensive — ledger reads from inputs only, but be safe):
    hc_compat["q_evap_applied_w"] = float(step_result["q_evap_applied_w"])
    hc_compat["q_evap_cycle_w"] = float(step_result["q_evap_cycle_w"])
    # Note: cluster_result is exposed via the real step_result dict so
    # pack_q_* arrays are read from the actual HC cluster, not from the
    # synthetic inputs.
    ledger, new_energy = ledger_from_heat_current(
        hc_plant,
        dt_s=dt_s,
        inputs=inputs_synth,
        hc=hc_compat,
        prev_energy=prev_energy,
    )
    return ledger, new_energy


__all__ = ["ledger_from_heat_current_plant"]