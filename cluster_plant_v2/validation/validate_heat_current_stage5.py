"""Stage 5 — System-Level Heat-Current Final Validation.

Stage 4.5 (commit 7dc6c9f, tag heat-current-independent-v1) introduced
``HeatCurrentPlant``: a fully independently-propagating heat-current
system. Stage 5 finally answers the question the user spec asks:

    "Does the heat-current model, when given the **same external
    inputs** as the frozen ``ClusterPlant``, reproduce the same
    **system-level temperature / heat-current dynamics**?"

Stage 5 walks 9 canonical cases (V1-V9) on both systems. The two plants
are constructed from the same engineering parameters, given the same
external input at every step, and their trajectories are recorded
side-by-side per step. No internal state is shared between the two
plants at any point.

Outputs land in ``validation/results/heat_current_stage5_final_<DATE>/``:

* ``{case_id}_legacy.csv`` — per-step diagnostics from ClusterPlant
* ``{case_id}_heat_current.csv`` — per-step diagnostics from HeatCurrentPlant
* ``stage5_summary.json`` — per-case aggregate metrics
* ``STAGE5_FINAL_VALIDATION.md`` — auto-generated overview
* ``figures/figure{1..4}.png`` — representative plots

Scope discipline
----------------
This module only validates; it does NOT modify any frozen module.
If a result exposes a structural error, the validation script must
report it as-is — the discipline (per user spec) is to never tune
model parameters to make RMSE look better.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from cluster_plant_v2.plant import (
    ClusterPlant,
    ClusterPlantInputs,
)
from cluster_plant_v2.profiles import AGC_DATA_FILE
from cluster_plant_v2.thermal.heat_current_energy_balance import (
    EnergyLedgerStep,
    initial_energy_snapshot,
    ledger_from_legacy,
)
from cluster_plant_v2.thermal.heat_current_plant import (
    HeatCurrentPlant,
    HeatCurrentPlantInputs,
    build_independent_hc_plant,
)
from cluster_plant_v2.thermal.heat_current_stage5_ledger import (
    ledger_from_heat_current_plant,
)
from cluster_plant_v2.validation.compare_legacy_reference import (
    load_regd_profile,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    FAN_SPEED_RPM,
    PUMP_SPEED_RPM,
    build_final_plant,
)


DURATION_S = 600.0
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _today_dir() -> Path:
    from datetime import datetime, timezone, timedelta

    tz = timezone(timedelta(hours=8))
    return RESULTS_ROOT / f"heat_current_stage5_final_{datetime.now(tz).strftime('%Y%m%d')}"


@dataclass
class CaseSpec:
    case_id: str
    description: str
    initial_compressor_rpm: float
    pump_rpm: float
    direction: str  # 'forward' | 'reverse' | 'switch'
    current_kind: str  # 'constant' | 'regd' | 'step'
    constant_current_a: float | None = None
    step_current_before_a: float | None = None
    step_current_after_a: float | None = None
    step_time_s: float = 200.0
    compressor_step_before_rpm: float | None = None
    compressor_step_after_rpm: float | None = None
    pump_step_before_rpm: float | None = None
    pump_step_after_rpm: float | None = None
    pump_step_time_s: float = 200.0


def _build_case_specs() -> list[CaseSpec]:
    return [
        # V1 — constant load, nominal flow, forward
        CaseSpec(
            case_id="V1_constant_nominal_forward",
            description="560 A, 4000 rpm compressor, nominal pump,"
            " forward flow.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V2 — current step
        CaseSpec(
            case_id="V2_current_step",
            description="Current step 560 -> 800 A at 200 s.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="step",
            step_current_before_a=560.0,
            step_current_after_a=800.0,
            step_time_s=200.0,
        ),
        # V3 — compressor-speed step
        CaseSpec(
            case_id="V3_compressor_step",
            description="Compressor step 2000 -> 4000 rpm at 200 s.",
            initial_compressor_rpm=2000.0,
            compressor_step_before_rpm=2000.0,
            compressor_step_after_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V4 — pump-speed step
        CaseSpec(
            case_id="V4_pump_step",
            description="Pump step 3000 -> 4500 rpm at 200 s.",
            initial_compressor_rpm=4000.0,
            pump_rpm=3000.0,
            pump_step_before_rpm=3000.0,
            pump_step_after_rpm=4500.0,
            pump_step_time_s=200.0,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V5 — low-flow condition (low pump speed)
        CaseSpec(
            case_id="V5_low_flow",
            description="Low flow (pump 2400 rpm).",
            initial_compressor_rpm=4000.0,
            pump_rpm=2400.0,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V6 — high-flow condition
        CaseSpec(
            case_id="V6_high_flow",
            description="High flow (pump 4500 rpm).",
            initial_compressor_rpm=4000.0,
            pump_rpm=4500.0,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V7 — reverse flow
        CaseSpec(
            case_id="V7_reverse_flow",
            description="Reverse flow throughout (560 A, 4000 rpm).",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="reverse",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V8 — flow-direction switching (forward 0-300 s, reverse 300-600 s)
        CaseSpec(
            case_id="V8_flow_switch",
            description="Forward 0-300 s, reverse 300-600 s.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        # V9 — RegD dynamic profile
        CaseSpec(
            case_id="V9_regd",
            description="AGC RegD current profile, forward flow.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="regd",
        ),
    ]


def _resolve_currents(spec: CaseSpec) -> np.ndarray:
    steps = int(round(DURATION_S / DT_S))
    if spec.current_kind == "constant":
        return np.full(steps, float(spec.constant_current_a), dtype=float)
    if spec.current_kind == "regd":
        _, currents = load_regd_profile(
            AGC_DATA_FILE, duration_s=DURATION_S, dt=DT_S,
        )
        return currents
    if spec.current_kind == "step":
        before = float(spec.step_current_before_a)
        after = float(spec.step_current_after_a)
        cutoff_step = int(round(spec.step_time_s / DT_S))
        currents = np.full(steps, before, dtype=float)
        currents[cutoff_step:] = after
        return currents
    raise ValueError(f"unknown current_kind: {spec.current_kind}")


def _direction_at(spec: CaseSpec, time_s: float) -> str:
    if spec.case_id == "V8_flow_switch":
        return "forward" if time_s < 300.0 else "reverse"
    return spec.direction


def _compressor_cmd_at(spec: CaseSpec, time_s: float) -> float:
    if spec.compressor_step_before_rpm is None:
        return spec.initial_compressor_rpm
    return (
        spec.compressor_step_before_rpm
        if time_s < spec.step_time_s
        else spec.compressor_step_after_rpm
    )


def _pump_cmd_at(spec: CaseSpec, time_s: float) -> float:
    if spec.pump_step_before_rpm is None:
        return spec.pump_rpm
    return (
        spec.pump_step_before_rpm
        if time_s < spec.pump_step_time_s
        else spec.pump_step_after_rpm
    )


def _build_hc_plant(legacy: ClusterPlant, *, conservative_transport=False) -> HeatCurrentPlant:
    """Build an independent HC plant matching the legacy's initial state."""
    # A single nominal inventory across V1-V9, independent of case pump speed.
    reference_flow = (legacy.pump.solve_operating_point(
        PUMP_SPEED_RPM, legacy.hydraulic_network
    )["total_mass_flow_kg_s"] if conservative_transport else None)
    return build_independent_hc_plant(
        legacy_plant=legacy, transport_reference_mass_flow_kg_s=reference_flow,
    )


def _ledger_row(
    spec: CaseSpec, step_index: int, ledger: EnergyLedgerStep
) -> dict[str, object]:
    """One Stage-4 energy-ledger row (system-level conservation)."""
    return {
        "case_id": spec.case_id,
        "step": step_index,
        "time_s": ledger.time_s,
        "q_gen_total_w": ledger.q_gen_total_w,
        "q_air_total_w": ledger.q_air_total_w,
        "q_bp_total_w": ledger.q_bp_total_w,
        "q_pf_total_w": ledger.q_pf_total_w,
        "q_evap_cycle_w": ledger.q_evap_cycle_w,
        "q_evap_applied_w": ledger.q_evap_applied_w,
        "q_tank_return_to_tank_w": ledger.q_tank_return_to_tank_w,
        "dE_battery_per_s": ledger.dE_battery_per_s,
        "dE_plate_per_s": ledger.dE_plate_per_s,
        "dE_coolant_segments_per_s": ledger.dE_coolant_segments_per_s,
        "dE_supply_per_s": ledger.dE_supply_per_s,
        "dE_return_per_s": ledger.dE_return_per_s,
        "dE_tank_per_s": ledger.dE_tank_per_s,
        "residual_battery_w": ledger.residual_battery_w,
        "residual_plate_w": ledger.residual_plate_w,
        "residual_loop_implicit_transport_w": (
            ledger.residual_loop_implicit_transport_w
        ),
        "residual_system_w": ledger.residual_system_w,
        "transport_flow_mismatch_w": ledger.transport_flow_mismatch_w,
    }


def _legacy_row(
    spec: CaseSpec, step_index: int, out: dict[str, object]
) -> dict[str, object]:
    cluster = out["cluster_result"]
    return {
        "case_id": spec.case_id,
        "step": step_index,
        "time_s": (step_index + 1) * DT_S,
        "tank_temperature_k": float(out["tank_temperature_after_k"]),
        "t_b_avg_k": float(np.mean(cluster["pack_battery_average_temperatures_k"])),
        "t_b_max_k": float(cluster["cluster_max_temperature_k"]),
        "t_p_avg_k": float(np.mean(cluster["pack_plate_average_temperatures_k"])),
        "t_p_max_k": float(np.max(cluster["pack_plate_temperatures_k"])),
        "t_supply_k": float(out["cluster_supply_temperature_k"]),
        "t_return_k": float(out["cluster_return_temperature_k"]),
        "t_evap_out_k": float(out["evaporator_outlet_temperature_k"]),
        "q_gen_w": float(np.sum(cluster["pack_q_gen_total_w"])),
        "q_bp_w": float(np.sum(cluster["pack_q_battery_to_plate_total_w"])),
        "q_pf_w": float(np.sum(cluster["pack_q_plate_to_fluid_total_w"])),
        "q_evap_cycle_w": float(out["q_evap_cycle_w"]),
        "q_evap_applied_w": float(out["q_evap_applied_w"]),
        "compressor_speed_used_rpm": float(out["compressor_speed_rpm"]),
        "refrigeration_solver_success": bool(out["refrigeration_solver_success"]),
        "all_states_finite": bool(out["all_states_finite"]),
    }


def _hc_row(
    spec: CaseSpec, step_index: int, out: dict[str, object]
) -> dict[str, object]:
    cluster = out["cluster_result"]
    return {
        "case_id": spec.case_id,
        "step": step_index,
        "time_s": (step_index + 1) * DT_S,
        "tank_temperature_k": float(out["tank_temperature_after_k"]),
        "t_b_avg_k": float(np.mean(cluster["pack_battery_average_temperatures_k"])),
        "t_b_max_k": float(cluster["cluster_max_temperature_k"]),
        "t_p_avg_k": float(np.mean(cluster["pack_plate_average_temperatures_k"])),
        "t_p_max_k": float(np.max(cluster["pack_plate_temperatures_k"])),
        "t_supply_k": float(out["cluster_supply_temperature_k"]),
        "t_return_k": float(out["cluster_return_temperature_k"]),
        "t_evap_out_k": float(out["evaporator_outlet_temperature_k"]),
        "q_gen_w": float(np.sum(cluster["pack_q_gen_total_w"])),
        "q_bp_w": float(np.sum(cluster["pack_q_battery_to_plate_total_w"])),
        "q_pf_w": float(np.sum(cluster["pack_q_plate_to_fluid_total_w"])),
        "q_evap_cycle_w": float(out["q_evap_cycle_w"]),
        "q_evap_applied_w": float(out["q_evap_applied_w"]),
        "compressor_speed_used_rpm": float(out["compressor_speed_used_rpm"]),
        "refrigeration_solver_success": bool(out["refrigeration_solver_success"]),
        "all_states_finite": bool(out["all_states_finite"]),
    }


def _run_one(
    spec: CaseSpec, currents: np.ndarray, *, conservative_transport=False
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    float,
    list[dict[str, object]],
    list[dict[str, object]],
]:
    steps = int(round(DURATION_S / DT_S))
    legacy = build_final_plant(
        spec.initial_compressor_rpm,
        initial_cluster_current_a=float(currents[0]),
        direction=_direction_at(spec, 0.0),
    )
    hc = _build_hc_plant(legacy, conservative_transport=conservative_transport)

    reference_flow = legacy.pump.solve_operating_point(
        _pump_cmd_at(spec, 0.0), legacy.hydraulic_network
    )["total_mass_flow_kg_s"]
    prev_leg = initial_energy_snapshot(
        legacy, is_heat_current=False,
        transport_reference_mass_flow_kg_s=reference_flow,
    )
    prev_hc = initial_energy_snapshot(
        hc, is_heat_current=True,
        transport_reference_mass_flow_kg_s=reference_flow,
    )

    leg_rows: list[dict[str, object]] = []
    hc_rows: list[dict[str, object]] = []
    leg_ledger_rows: list[dict[str, object]] = []
    hc_ledger_rows: list[dict[str, object]] = []
    start = perf_counter()
    for step_index in range(steps):
        time_before = step_index * DT_S
        direction = _direction_at(spec, time_before)
        comp_cmd = _compressor_cmd_at(spec, time_before)
        pump_cmd = _pump_cmd_at(spec, time_before)
        current_a = float(currents[step_index])

        leg_out = legacy.step(
            inputs=ClusterPlantInputs(
                cluster_current_a=current_a,
                compressor_command_rpm=comp_cmd,
                pump_rpm=pump_cmd,
                fan_rpm=FAN_SPEED_RPM,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                flow_direction=direction,
            ),
            dt_s=DT_S,
        )
        hc_out = hc.step(
            inputs=HeatCurrentPlantInputs(
                cluster_current_a=current_a,
                compressor_command_rpm=comp_cmd,
                pump_rpm=pump_cmd,
                fan_rpm=FAN_SPEED_RPM,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                flow_direction=direction,
            ),
            dt_s=DT_S,
        )

        leg_rows.append(_legacy_row(spec, step_index, leg_out))
        hc_rows.append(_hc_row(spec, step_index, hc_out))

        # Stage 4 energy ledger (system-level conservation residuals).
        leg_ledger, prev_leg = ledger_from_legacy(
            legacy, dt_s=DT_S, result=leg_out, prev_energy=prev_leg,
        )
        hc_ledger, prev_hc = ledger_from_heat_current_plant(
            hc,
            dt_s=DT_S,
            step_result=hc_out,
            prev_energy=prev_hc,
            compressor_speed_used_rpm=float(hc_out["compressor_speed_used_rpm"]),
            tank_temperature_before_k=float(
                hc_out.get("tank_temperature_before_k", float("nan"))
            ),
            refrigeration_solver_success=bool(
                hc_out["refrigeration_solver_success"]
            ),
        )
        from dataclasses import replace
        leg_ledger = replace(leg_ledger, time_s=(step_index + 1) * DT_S)
        hc_ledger = replace(hc_ledger, time_s=(step_index + 1) * DT_S)
        leg_ledger_rows.append(_ledger_row(spec, step_index, leg_ledger))
        hc_row = _ledger_row(spec, step_index, hc_ledger)
        if conservative_transport:
            hc_row.update({
                "supply_mass_kg": hc.supply_delay.stored_mass_kg,
                "return_mass_kg": hc.return_delay.stored_mass_kg,
                "supply_energy_j": hc.supply_delay.stored_energy_j,
                "return_energy_j": hc.return_delay.stored_energy_j,
            })
        hc_ledger_rows.append(hc_row)

    runtime_s = perf_counter() - start
    return leg_rows, hc_rows, runtime_s, leg_ledger_rows, hc_ledger_rows


def _summarize(
    spec: CaseSpec,
    leg_rows: list[dict[str, object]],
    hc_rows: list[dict[str, object]],
    runtime_s: float,
    leg_ledger_rows: list[dict[str, object]],
    hc_ledger_rows: list[dict[str, object]],
) -> dict[str, object]:
    temp_keys = [
        "t_b_avg_k", "t_b_max_k", "t_p_avg_k", "t_p_max_k",
        "tank_temperature_k", "t_supply_k", "t_return_k", "t_evap_out_k",
    ]
    q_keys = ["q_gen_w", "q_bp_w", "q_pf_w", "q_evap_cycle_w", "q_evap_applied_w"]

    leg_t = {key: np.array([float(r[key]) for r in leg_rows]) for key in temp_keys}
    hc_t = {key: np.array([float(r[key]) for r in hc_rows]) for key in temp_keys}
    leg_q = {key: np.array([float(r[key]) for r in leg_rows]) for key in q_keys}
    hc_q = {key: np.array([float(r[key]) for r in hc_rows]) for key in q_keys}

    temp_errors: dict[str, dict[str, float]] = {}
    for key in temp_keys:
        diff = hc_t[key] - leg_t[key]
        temp_errors[key] = {
            "rmse_k": float(np.sqrt(np.mean(diff * diff))),
            "mae_k": float(np.mean(np.abs(diff))),
            "max_abs_k": float(np.max(np.abs(diff))),
        }

    times = np.array(
        [float(r["time_s"]) for r in leg_rows], dtype=float
    )
    cumulative: dict[str, dict[str, float]] = {}
    for key in q_keys:
        e_leg = float(np.trapezoid(leg_q[key], times))
        e_hc = float(np.trapezoid(hc_q[key], times))
        rel = (
            abs(e_hc - e_leg) / abs(e_leg) * 100.0 if abs(e_leg) > 0.0 else 0.0
        )
        cumulative[key] = {
            "e_legacy_j": e_leg,
            "e_hc_j": e_hc,
            "rel_diff_pct": rel,
        }

    # Domain-validity flags (heuristic, conservative)
    leg_finite = all(bool(r["all_states_finite"]) for r in leg_rows)
    hc_finite = all(bool(r["all_states_finite"]) for r in hc_rows)
    leg_solver = all(bool(r["refrigeration_solver_success"]) for r in leg_rows)
    hc_solver = all(bool(r["refrigeration_solver_success"]) for r in hc_rows)

    def _max_abs(rows, key) -> float:
        return float(
            np.max(np.abs(np.array([float(r[key]) for r in rows], dtype=float)))
        )

    ledger = {
        "legacy": {
            "max_abs_residual_battery_w": _max_abs(
                leg_ledger_rows, "residual_battery_w"
            ),
            "max_abs_residual_plate_w": _max_abs(
                leg_ledger_rows, "residual_plate_w"
            ),
            "max_abs_residual_loop_transport_w": _max_abs(
                leg_ledger_rows, "residual_loop_implicit_transport_w"
            ),
            "max_abs_residual_system_w": _max_abs(
                leg_ledger_rows, "residual_system_w"
            ),
            "max_abs_transport_flow_mismatch_w": _max_abs(
                leg_ledger_rows, "transport_flow_mismatch_w"
            ),
        },
        "heat_current": {
            "max_abs_residual_battery_w": _max_abs(
                hc_ledger_rows, "residual_battery_w"
            ),
            "max_abs_residual_plate_w": _max_abs(
                hc_ledger_rows, "residual_plate_w"
            ),
            "max_abs_residual_loop_transport_w": _max_abs(
                hc_ledger_rows, "residual_loop_implicit_transport_w"
            ),
            "max_abs_residual_system_w": _max_abs(
                hc_ledger_rows, "residual_system_w"
            ),
            "max_abs_transport_flow_mismatch_w": _max_abs(
                hc_ledger_rows, "transport_flow_mismatch_w"
            ),
        },
    }

    return {
        "case_id": spec.case_id,
        "description": spec.description,
        "runtime_s": runtime_s,
        "steps": len(leg_rows),
        "temp_errors": temp_errors,
        "cumulative": cumulative,
        "all_states_finite": {"legacy": leg_finite, "heat_current": hc_finite},
        "solver_success_all_steps": {
            "legacy": leg_solver,
            "heat_current": hc_solver,
        },
        "domain_invalid": {
            "legacy": (not leg_finite) or (not leg_solver),
            "heat_current": (not hc_finite) or (not hc_solver),
        },
        "ledger": ledger,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _read_all_rows(out_dir: Path) -> dict[str, dict[str, list]]:
    """Re-load per-case CSV rows from disk for figures-only renders."""
    rows: dict[str, dict[str, list]] = {}
    for csv_path in sorted(out_dir.glob("*_legacy.csv")):
        case_id = csv_path.stem[: -len("_legacy")]
        rows[case_id] = {
            "legacy": _read_csv(csv_path),
            "heat_current": _read_csv(out_dir / f"{case_id}_heat_current.csv"),
        }
    return rows


def _render_markdown(summaries: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Stage 5 Final Validation — per-case summary")
    lines.append("")
    lines.append(
        "Each case reports the legacy-vs-HC temperature RMSE/MAE/max-error"
        " for 8 fields, the cumulative-energy relative difference for"
        " 5 heat-current quantities, the solver/finite-state flags, and"
        " the runtime. Domain-invalid means at least one step lost"
        " finite-state or refrigeration solver success — not a frozen-"
        " model failure but a model-boundary condition."
    )
    lines.append("")
    lines.append("| case | runtime (s) | steps | legacy finite | HC finite | legacy solver | HC solver | domain-invalid (legacy / HC) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in summaries:
        leg = s["all_states_finite"]["legacy"]
        hcf = s["all_states_finite"]["heat_current"]
        legs = s["solver_success_all_steps"]["legacy"]
        hcs = s["solver_success_all_steps"]["heat_current"]
        dvi = s["domain_invalid"]
        lines.append(
            f"| {s['case_id']} | {s['runtime_s']:.1f} | {s['steps']} |"
            f" {leg} | {hcf} | {legs} | {hcs} |"
            f" {dvi['legacy']} / {dvi['heat_current']} |"
        )
    lines.append("")
    lines.append("## Temperature errors (legacy vs HC)")
    lines.append("")
    lines.append(
        "| case | T_b_avg RMSE (K) | T_b_max RMSE (K) | T_p_avg RMSE (K) |"
        " T_p_max RMSE (K) | T_tank RMSE (K) | T_supply RMSE (K) |"
        " T_return RMSE (K) | T_evap_out RMSE (K) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        te = s["temp_errors"]
        lines.append(
            f"| {s['case_id']} |"
            f" {te['t_b_avg_k']['rmse_k']:.4f} |"
            f" {te['t_b_max_k']['rmse_k']:.4f} |"
            f" {te['t_p_avg_k']['rmse_k']:.4f} |"
            f" {te['t_p_max_k']['rmse_k']:.4f} |"
            f" {te['tank_temperature_k']['rmse_k']:.4f} |"
            f" {te['t_supply_k']['rmse_k']:.4f} |"
            f" {te['t_return_k']['rmse_k']:.4f} |"
            f" {te['t_evap_out_k']['rmse_k']:.4f} |"
        )
    lines.append("")
    lines.append("## Cumulative-energy relative difference (%)")
    lines.append("")
    lines.append(
        "| case | E_gen (%) | E_bp (%) | E_pf (%) | E_evap_cycle (%) |"
        " E_evap_applied (%) |"
    )
    lines.append("|---|---|---|---|---|---|")
    for s in summaries:
        c = s["cumulative"]
        lines.append(
            f"| {s['case_id']} |"
            f" {c['q_gen_w']['rel_diff_pct']:.4f} |"
            f" {c['q_bp_w']['rel_diff_pct']:.4f} |"
            f" {c['q_pf_w']['rel_diff_pct']:.4f} |"
            f" {c['q_evap_cycle_w']['rel_diff_pct']:.4f} |"
            f" {c['q_evap_applied_w']['rel_diff_pct']:.4f} |"
        )
    lines.append("")
    lines.append("## Energy-conservation residuals (max |·| over 120 steps)")
    lines.append("")
    lines.append("Time-FIFO storage uses equivalent mass defined by the reference "
                 "flow and delay; mass-transport storage reads actual parcel energies. "
                 "R_loop (= Q_pf − Q_evap_applied − "
                 "dE_coolant_total/dt) and R_system are raw energy residuals. "
                 "At reference flow they should close numerically. Off-reference "
                 "flow exposes the fixed-time FIFO model discrepancy, reported "
                 "separately without subtracting it from the raw residuals. "
                 "Mass transport must close the raw residual at every flow.")
    lines.append("")
    lines.append("| case | backend | R_battery (W) | R_plate (W) |"
                 " R_loop (W) | R_system (W) | FIFO flow mismatch (W) |")
    lines.append("|---|---|---|---|---|---|---|")
    for s in summaries:
        for backend in ("legacy", "heat_current"):
            a = s["ledger"][backend]
            lines.append(
                f"| {s['case_id']} | {backend} |"
                f" {a['max_abs_residual_battery_w']:.3e} |"
                f" {a['max_abs_residual_plate_w']:.3e} |"
                f" {a['max_abs_residual_loop_transport_w']:.3e} |"
                f" {a['max_abs_residual_system_w']:.3e} |"
                f" {a['max_abs_transport_flow_mismatch_w']:.3e} |"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


def _make_figures(out_dir: Path, all_rows: dict[str, dict[str, list]]) -> None:
    """Render four representative figures from per-case data.

    ``all_rows`` maps case_id -> {'legacy': [...], 'heat_current': [...]}.
    Figures are PNG and are gitignored.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    def _clean_axes(ax, x_label: bool):
        """Keep tick counts low and label precision readable."""
        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.tick_params(axis="both", labelsize=8)
        if x_label:
            ax.set_xlabel("time (s)", fontsize=9)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    def _series(backend_rows: list[dict], key: str) -> tuple[list[float], list[float]]:
        # CSV DictReader returns strings; cast to float so matplotlib
        # actually uses them as numeric (otherwise it treats them categorical
        # and the x-axis spans string-index lengths, not real seconds).
        times = [float(r["time_s"]) for r in backend_rows]
        vals = [float(r[key]) for r in backend_rows]
        return times, vals

    def _plot_pair(ax, rows_l, rows_h, key):
        xl, yl = _series(rows_l, key)
        xh, yh = _series(rows_h, key)
        ax.plot(xl, yl, label="legacy", color="C0")
        ax.plot(xh, yh, "--", label="heat_current", color="C1")

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: representative temperature dynamics — V1
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
    case_id = "V1_constant_nominal_forward"
    rows_l = all_rows[case_id]["legacy"]
    rows_h = all_rows[case_id]["heat_current"]
    for ax, key, label in [
        (axes[0], "t_b_avg_k", "T_b avg (K)"),
        (axes[1], "t_p_avg_k", "T_p avg (K)"),
        (axes[2], "tank_temperature_k", "T_tank (K)"),
    ]:
        _plot_pair(ax, rows_l, rows_h, key)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _clean_axes(ax, x_label=(ax is axes[-1]))
    fig.suptitle(f"Figure 1: representative temperature dynamics ({case_id})", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure1_temperature_dynamics.png", dpi=120)
    plt.close(fig)

    # Figure 2: step-response thermal propagation — V3 (compressor step)
    fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    case_id = "V3_compressor_step"
    rows_l = all_rows[case_id]["legacy"]
    rows_h = all_rows[case_id]["heat_current"]
    for ax, key, label in [
        (axes[0], "t_evap_out_k", "T_evap_out (K)"),
        (axes[1], "t_supply_k", "T_supply (K)"),
        (axes[2], "t_p_avg_k", "T_p avg (K)"),
        (axes[3], "t_b_avg_k", "T_b avg (K)"),
    ]:
        _plot_pair(ax, rows_l, rows_h, key)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _clean_axes(ax, x_label=(ax is axes[-1]))
    fig.suptitle(f"Figure 2: step-response propagation ({case_id})", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure2_step_response.png", dpi=120)
    plt.close(fig)

    # Figure 3: forward/reverse spatial temperatures — V7 vs V8
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    cases = ["V7_reverse_flow", "V8_flow_switch"]
    titles = ["V7 reverse flow", "V8 forward → reverse @ 300 s"]
    for col, (cid, title) in enumerate(zip(cases, titles)):
        rows_l = all_rows[cid]["legacy"]
        rows_h = all_rows[cid]["heat_current"]
        for row, key, label in [
            (0, "t_b_avg_k", "T_b avg (K)"),
            (1, "t_p_avg_k", "T_p avg (K)"),
        ]:
            ax = axes[row, col]
            _plot_pair(ax, rows_l, rows_h, key)
            ax.set_title(f"{title}: {label}", fontsize=9)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _clean_axes(ax, x_label=(row == 1))
    fig.suptitle("Figure 3: forward/reverse spatial temperature comparison", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure3_forward_reverse.png", dpi=120)
    plt.close(fig)

    # Figure 4: heat-current path Q_gen -> Q_bp -> Q_pf -> Q_evap — V1
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)
    case_id = "V1_constant_nominal_forward"
    rows_l = all_rows[case_id]["legacy"]
    rows_h = all_rows[case_id]["heat_current"]
    for ax, key, label in [
        (axes[0, 0], "q_gen_w", "Q_gen (W)"),
        (axes[0, 1], "q_bp_w", "Q_bp (W)"),
        (axes[1, 0], "q_pf_w", "Q_pf (W)"),
        (axes[1, 1], "q_evap_applied_w", "Q_evap_applied (W)"),
    ]:
        _plot_pair(ax, rows_l, rows_h, key)
        ax.set_title(label, fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        _clean_axes(ax, x_label=(ax in (axes[1, 0], axes[1, 1])))
    fig.suptitle(f"Figure 4: heat-current path ({case_id})", fontsize=10)
    fig.tight_layout()
    fig.savefig(figures_dir / "figure4_heat_current_path.png", dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=None, help="override output directory",
    )
    parser.add_argument(
        "--no-figures", action="store_true",
        help="skip matplotlib figure rendering",
    )
    parser.add_argument(
        "--conservative-transport", action="store_true",
        help="use fixed-inventory mass transport in HC; retain legacy as reference",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="skip simulation; only re-render figures from existing CSVs in --output-dir",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir) if args.output_dir else _today_dir()
    if args.conservative_transport and args.output_dir is None:
        out_dir = out_dir.with_name(out_dir.name + "_mass_transport")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        all_rows = _read_all_rows(out_dir)
        _make_figures(out_dir, all_rows)
        print(f"figures: {out_dir / 'figures'}")
        return 0

    summaries: list[dict[str, object]] = []
    all_rows: dict[str, dict[str, list]] = {}
    for spec in _build_case_specs():
        currents = _resolve_currents(spec)
        leg_rows, hc_rows, runtime_s, leg_ledger_rows, hc_ledger_rows = _run_one(
            spec, currents, conservative_transport=args.conservative_transport
        )
        _write_csv(
            out_dir / f"{spec.case_id}_legacy.csv", leg_rows
        )
        _write_csv(
            out_dir / f"{spec.case_id}_heat_current.csv", hc_rows
        )
        _write_csv(
            out_dir / f"{spec.case_id}_legacy_ledger.csv", leg_ledger_rows
        )
        _write_csv(
            out_dir / f"{spec.case_id}_heat_current_ledger.csv", hc_ledger_rows
        )
        all_rows[spec.case_id] = {
            "legacy": leg_rows,
            "heat_current": hc_rows,
        }
        summaries.append(
            _summarize(spec, leg_rows, hc_rows, runtime_s, leg_ledger_rows, hc_ledger_rows)
        )
        summaries[-1]["hc_transport_model"] = (
            "fixed_inventory_mass_transport" if args.conservative_transport else "fixed_time_fifo"
        )
        if args.conservative_transport:
            summaries[-1]["hc_raw_energy_gate_passed"] = bool(
                summaries[-1]["ledger"]["heat_current"]["max_abs_residual_system_w"] < 1e-6
                and summaries[-1]["all_states_finite"]["heat_current"]
                and summaries[-1]["solver_success_all_steps"]["heat_current"]
            )
            summaries[-1]["pipe_inventory_kg"] = {
                "supply": float(hc_ledger_rows[0]["supply_mass_kg"]),
                "return": float(hc_ledger_rows[0]["return_mass_kg"]),
            }
        print(
            f"[{spec.case_id}] runtime={runtime_s:.1f}s"
            f"  T_b_avg_RMSE={summaries[-1]['temp_errors']['t_b_avg_k']['rmse_k']:.4f} K"
            f"  E_gen_relDiff={summaries[-1]['cumulative']['q_gen_w']['rel_diff_pct']:.4f}%"
            f"  R_bat_max={summaries[-1]['ledger']['legacy']['max_abs_residual_battery_w']:.2e} W"
            f"  R_transport_max={summaries[-1]['ledger']['heat_current']['max_abs_residual_loop_transport_w']:.2e} W"
        )

    summary_path = out_dir / "stage5_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)
    summary_table = out_dir / "STAGE5_FINAL_VALIDATION.md"
    transport_note = (
        "HC uses fixed-inventory mass transport; all cases share inventory anchored "
        "at the nominal pump speed. Legacy retains fixed-time FIFO. HC passes only "
        "if its raw system residual is below 1e-6 W with finite states and successful "
        "cycle solves. Temperature differences from legacy are model differences.\n\n"
        if args.conservative_transport else ""
    )
    summary_table.write_text(transport_note + _render_markdown(summaries), encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"summary table: {summary_table}")

    if not args.no_figures:
        try:
            _make_figures(out_dir, all_rows)
            print(f"figures: {out_dir / 'figures'}")
        except Exception as exc:
            print(f"figure rendering failed: {exc}")

    if args.conservative_transport and not all(s["hc_raw_energy_gate_passed"] for s in summaries):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
