"""Stage 4 system-level heat-current closure validation.

Runs six canonical cases (constant load, current step, dynamic RegD,
reverse flow, flow switch, low/nominal/high coolant flow) on both the
legacy ``ClusterPlant`` and the parallel ``HeatCurrentSystemLink``, and
records the per-step energy-current ledger for each. Each ledger
reports the three conservation equations of the Stage 4 spec plus the
implicit-transport residual on the loop equation.

Outputs land in
``validation/results/heat_current_stage4_closure_<DATE>/`` as per-case
CSVs (one per legacy/heat_current) plus an aggregate
``STAGE4_CLOSURE.md`` summary.

No MPC, no controller, no TD3: this module only walks the existing
plant components.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np

from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.thermal import (
    HeatCurrentStepInputs,
    build_parallel_system,
)
from cluster_plant_v2.thermal.heat_current_energy_balance import (
    EnergyLedgerStep,
    cumulative_residual_j,
    initial_energy_snapshot,
    ledger_from_heat_current,
    ledger_from_legacy,
)
from cluster_plant_v2.validation.compare_legacy_reference import (
    load_regd_profile,
)
from cluster_plant_v2.profiles import AGC_DATA_FILE
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
    return RESULTS_ROOT / f"heat_current_stage4_closure_{datetime.now(tz).strftime('%Y%m%d')}"


@dataclass
class CaseSpec:
    case_id: str
    description: str
    initial_compressor_rpm: float
    pump_rpm: float
    direction: str
    current_kind: str  # 'constant' | 'regd' | 'step'
    constant_current_a: float | None = None
    step_current_before_a: float | None = None
    step_current_after_a: float | None = None
    step_time_s: float = 200.0


def _build_case_specs() -> list[CaseSpec]:
    return [
        CaseSpec(
            case_id="C0_constant_load",
            description="Constant load, forward flow, nominal mass flow.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        CaseSpec(
            case_id="C1_current_step",
            description="Current step 560 -> 800 A at 200 s.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="step",
            step_current_before_a=560.0,
            step_current_after_a=800.0,
            step_time_s=200.0,
        ),
        CaseSpec(
            case_id="C2_dynamic_regd",
            description="AGC RegD current profile, forward flow.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="regd",
        ),
        CaseSpec(
            case_id="C3_reverse_flow",
            description="Reverse coolant flow throughout.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="reverse",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        CaseSpec(
            case_id="C4_flow_switch",
            description="Forward 0-300 s, reverse 300-600 s.",
            initial_compressor_rpm=4000.0,
            pump_rpm=PUMP_SPEED_RPM,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
        ),
        CaseSpec(
            case_id="C5_low_nominal_high_flow",
            description="High coolant flow (pump 4500 rpm).",
            initial_compressor_rpm=4000.0,
            pump_rpm=4500.0,
            direction="forward",
            current_kind="constant",
            constant_current_a=560.0,
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
    if spec.case_id == "C4_flow_switch":
        return "forward" if time_s < 300.0 else "reverse"
    return spec.direction


def _ledger_row(
    spec: CaseSpec, step_index: int, ledger: EnergyLedgerStep
) -> dict[str, object]:
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


def _run_one(
    spec: CaseSpec, currents: np.ndarray
) -> tuple[list[dict[str, object]], list[dict[str, object]], float]:
    steps = int(round(DURATION_S / DT_S))
    plant = build_final_plant(
        spec.initial_compressor_rpm,
        initial_cluster_current_a=float(currents[0]),
        direction=spec.direction,
    )
    parallel = build_parallel_system(legacy_plant=plant)

    reference_flow = plant.pump.solve_operating_point(
        spec.pump_rpm, plant.hydraulic_network
    )["total_mass_flow_kg_s"]
    prev_leg = initial_energy_snapshot(
        plant, is_heat_current=False,
        transport_reference_mass_flow_kg_s=reference_flow,
    )
    prev_hc = initial_energy_snapshot(
        parallel, is_heat_current=True,
        transport_reference_mass_flow_kg_s=reference_flow,
    )

    leg_rows: list[dict[str, object]] = []
    hc_rows: list[dict[str, object]] = []
    start = perf_counter()
    for step_index in range(steps):
        time_before = step_index * DT_S
        direction = _direction_at(spec, time_before)
        out = plant.step(
            dt_s=DT_S,
            cluster_current_a=float(currents[step_index]),
            pump_speed_rpm=spec.pump_rpm,
            compressor_speed_command_rpm=spec.initial_compressor_rpm,
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction=direction,
        )
        ref = out["refrigeration_result"]
        inp = HeatCurrentStepInputs(
            dt_s=DT_S,
            cluster_current_a=float(currents[step_index]),
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction=direction,
            total_mass_flow_kg_s=out["total_mass_flow_kg_s"],
            tank_temperature_before_k=out["tank_temperature_before_k"],
            q_evap_applied_w=out["q_evap_applied_w"],
            q_evap_cycle_w=out["q_evap_cycle_w"],
            evaporating_temperature_k=ref["evaporating_saturation_temperature_k"],
            evaporator_ua_w_k=ref["evaporator_ua_w_k"],
            refrigeration_solver_success=out["refrigeration_solver_success"],
            compressor_speed_rpm=out["compressor_speed_rpm"],
        )
        hc = parallel.step(inp)
        leg, prev_leg = ledger_from_legacy(
            plant, dt_s=DT_S, result=out, prev_energy=prev_leg,
        )
        hc_ledger, prev_hc = ledger_from_heat_current(
            parallel, dt_s=DT_S, inputs=inp, hc=hc, prev_energy=prev_hc,
        )
        from dataclasses import replace
        leg = replace(leg, time_s=(step_index + 1) * DT_S)
        hc_ledger = replace(hc_ledger, time_s=(step_index + 1) * DT_S)
        leg_rows.append(_ledger_row(spec, step_index, leg))
        hc_rows.append(_ledger_row(spec, step_index, hc_ledger))
    runtime_s = perf_counter() - start
    return leg_rows, hc_rows, runtime_s


def _summarize(
    spec: CaseSpec, leg_rows: list[dict[str, object]],
    hc_rows: list[dict[str, object]], runtime_s: float,
) -> dict[str, object]:
    def _max_abs(rows, key):
        return float(
            np.max(np.abs(np.array([float(r[key]) for r in rows], dtype=float)))
        )

    summary = {
        "case_id": spec.case_id,
        "description": spec.description,
        "runtime_s": runtime_s,
        "legacy": {
            "steps": len(leg_rows),
            "max_abs_residual_battery_w": _max_abs(
                leg_rows, "residual_battery_w"
            ),
            "max_abs_residual_plate_w": _max_abs(
                leg_rows, "residual_plate_w"
            ),
            "max_abs_residual_loop_implicit_transport_w": _max_abs(
                leg_rows, "residual_loop_implicit_transport_w"
            ),
            "max_abs_residual_system_w": _max_abs(
                leg_rows, "residual_system_w"
            ),
        },
        "heat_current": {
            "steps": len(hc_rows),
            "max_abs_residual_battery_w": _max_abs(
                hc_rows, "residual_battery_w"
            ),
            "max_abs_residual_plate_w": _max_abs(
                hc_rows, "residual_plate_w"
            ),
            "max_abs_residual_loop_implicit_transport_w": _max_abs(
                hc_rows, "residual_loop_implicit_transport_w"
            ),
            "max_abs_residual_system_w": _max_abs(
                hc_rows, "residual_system_w"
            ),
        },
    }
    return summary


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(summaries: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Stage 4 Closure Report — per-case residuals")
    lines.append("")
    lines.append("| case | runtime (s) | legacy max|R_bat| (W) | legacy max|R_plate| (W) | legacy max|R_loop| (W) | legacy max|R_sys| (W) | HC max|R_bat| (W) | HC max|R_plate| (W) | HC max|R_loop| (W) | HC max|R_sys| (W) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in summaries:
        leg = s["legacy"]
        hc = s["heat_current"]
        lines.append(
            f"| {s['case_id']} | {s['runtime_s']:.2f} |"
            f" {leg['max_abs_residual_battery_w']:.3e} |"
            f" {leg['max_abs_residual_plate_w']:.3e} |"
            f" {leg['max_abs_residual_loop_implicit_transport_w']:.3e} |"
            f" {leg['max_abs_residual_system_w']:.3e} |"
            f" {hc['max_abs_residual_battery_w']:.3e} |"
            f" {hc['max_abs_residual_plate_w']:.3e} |"
            f" {hc['max_abs_residual_loop_implicit_transport_w']:.3e} |"
            f" {hc['max_abs_residual_system_w']:.3e} |"
        )
    lines.append("")
    lines.append("Per-case CSVs live next to this report; see the schema for")
    lines.append("field semantics.")
    lines.append("")
    lines.append(
        "Reading the residuals: ``R_bat`` and ``R_plate`` are reported"
    )
    lines.append(
        "against the Stage 8C3 local-gate threshold of 1e-8 W (battery)"
    )
    lines.append(
        "and 1e-9 W (plate). ``R_loop`` is the loop-balance residual"
    )
    lines.append(
        "``Q_pf - Q_evap_applied - dE_coolant_total/dt``. Pipe storage uses"
    )
    lines.append(
        "fixed equivalent mass from the explicit reference flow and delay."
    )
    lines.append(
        "At reference flow it should close numerically. At other flows,"
    )
    lines.append(
        "the raw residual retains the fixed-time FIFO model discrepancy;"
    )
    lines.append("``transport_flow_mismatch_w`` in the CSV reports that term separately.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=None, help="override output directory",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir) if args.output_dir else _today_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    for spec in _build_case_specs():
        currents = _resolve_currents(spec)
        leg_rows, hc_rows, runtime_s = _run_one(spec, currents)
        _write_csv(
            out_dir / f"{spec.case_id}_legacy.csv", leg_rows
        )
        _write_csv(
            out_dir / f"{spec.case_id}_heat_current.csv", hc_rows
        )
        summaries.append(_summarize(spec, leg_rows, hc_rows, runtime_s))
        print(
            f"[{spec.case_id}] runtime={runtime_s:.1f}s"
            f"  leg |R_loop|_max={summaries[-1]['legacy']['max_abs_residual_loop_implicit_transport_w']:.2e} W"
            f"  HC |R_loop|_max={summaries[-1]['heat_current']['max_abs_residual_loop_implicit_transport_w']:.2e} W"
        )

    summary_path = out_dir / "stage4_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)
    summary_table = out_dir / "STAGE4_CLOSURE_SUMMARY.md"
    summary_table.write_text(_render_markdown(summaries), encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"summary table: {summary_table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
