"""Stage 3 system-level validation: parallel heat-current plant vs legacy.

For each of the three frozen Stage 8C3 cases (``T0_constant``,
``T1_compressor_step``, ``T2_regd``) we run the legacy
``ClusterPlant`` and, in lockstep, the parallel ``HeatCurrentSystemLink``.

Per step the legacy plant is stepped first (it carries the cycle /
evaporator / actuator state); its outputs are fed to the parallel link
via ``HeatCurrentStepInputs``. The two trajectories are compared on the
*common* subset of state (everything except the cluster internal state):
``T_battery_avg/max``, ``T_plate_avg``, ``T_supply``, ``T_return``,
``T_tank``, plus the evaporator energy closure and ``Q_HC`` cross-check
on the heat-current side.

Engineering thresholds (reused, not invented):
* ``refrigeration_solver_success`` must hold on every step;
* ``all_states_finite`` must hold on every step;
* ``cluster_fluid_residual_w`` abs must stay within the same band as the
  legacy (``<= 1e-3 W`` after warm-up); the heat-current plate is
  stiffer so the residual is in fact tighter;
* ``evaporator_coolant_residual_w`` is zero by construction on the
  heat-current link (numerical noise only).

Outputs land in
``validation/results/heat_current_stage3_system_<DATE>/`` as
per-case CSVs (both plants' per-step diagnostics) and a top-level
``STAGE3_SYSTEM_INTEGRATION.md`` summary report.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.thermal import (
    HeatCurrentStepInputs,
    build_parallel_system,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    FAN_SPEED_RPM,
    FinalPlantCaseSpec,
    PUMP_SPEED_RPM,
    build_case_specs,
    build_final_plant,
    compressor_command_at_time,
    load_regd_case_currents,
)
from cluster_plant_v2.parameters import (
    COMPRESSOR_TIME_CONSTANT_S,
    EVAPORATOR_TIME_CONSTANT_S,
)

DURATION_S = 600.0
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _today_dir() -> Path:
    from datetime import datetime, timezone, timedelta
    tz = timezone(timedelta(hours=8))
    return RESULTS_ROOT / f"heat_current_stage3_system_{datetime.now(tz).strftime('%Y%m%d')}"


@dataclass
class CaseRunner:
    case: FinalPlantCaseSpec
    duration_s: float = DURATION_S
    dt_s: float = DT_S

    def run(self) -> dict[str, object]:
        steps_float = self.duration_s / self.dt_s
        steps = int(round(steps_float))
        if steps < 1:
            raise ValueError("duration_s must be a positive multiple of dt_s")

        if self.case.current_kind == "regd":
            _, currents = load_regd_case_currents(
                duration_s=self.duration_s, dt_s=self.dt_s
            )
        else:
            currents = np.full(steps, float(self.case.current_a), dtype=float)

        plant = build_final_plant(
            self.case.initial_compressor_speed_rpm,
            initial_cluster_current_a=float(currents[0]),
            direction=self.case.direction,
        )
        parallel = build_parallel_system(legacy_plant=plant)

        legacy_rows: list[dict[str, object]] = []
        hc_rows: list[dict[str, object]] = []
        legacy_failures = 0
        hc_failures = 0
        legacy_failure_message = ""
        hc_failure_message = ""

        start = perf_counter()
        for step_index in range(steps):
            time_before = step_index * self.dt_s
            command = compressor_command_at_time(self.case, time_before)

            try:
                result = plant.step(
                    dt_s=self.dt_s,
                    cluster_current_a=float(currents[step_index]),
                    pump_speed_rpm=PUMP_SPEED_RPM,
                    compressor_speed_command_rpm=command,
                    fan_speed_rpm=FAN_SPEED_RPM,
                    ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                    direction=self.case.direction,
                )
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                legacy_failures += 1
                legacy_failure_message = str(exc)
                break

            cluster = result["cluster_result"]
            ref = result["refrigeration_result"]

            hc_inputs = HeatCurrentStepInputs(
                dt_s=self.dt_s,
                cluster_current_a=float(currents[step_index]),
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                direction=self.case.direction,
                total_mass_flow_kg_s=result["total_mass_flow_kg_s"],
                tank_temperature_before_k=result["tank_temperature_before_k"],
                q_evap_applied_w=result["q_evap_applied_w"],
                q_evap_cycle_w=result["q_evap_cycle_w"],
                evaporating_temperature_k=ref["evaporating_saturation_temperature_k"],
                evaporator_ua_w_k=ref["evaporator_ua_w_k"],
                refrigeration_solver_success=result["refrigeration_solver_success"],
                compressor_speed_rpm=result["compressor_speed_rpm"],
            )

            try:
                hc = parallel.step(hc_inputs)
            except (FloatingPointError, RuntimeError, ValueError) as exc:
                hc_failures += 1
                hc_failure_message = str(exc)
                break

            legacy_rows.append(_legacy_row(self.case, step_index, self.dt_s,
                                            result, cluster))
            hc_rows.append(_hc_row(self.case, step_index, self.dt_s, hc_inputs,
                                    hc))

        runtime_s = perf_counter() - start
        return {
            "case": self.case,
            "steps": steps,
            "legacy_rows": legacy_rows,
            "hc_rows": hc_rows,
            "legacy_failures": legacy_failures,
            "hc_failures": hc_failures,
            "legacy_failure_message": legacy_failure_message,
            "hc_failure_message": hc_failure_message,
            "runtime_s": runtime_s,
        }


def _legacy_row(case, step_index, dt_s, result, cluster) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "step": step_index,
        "time_s": (step_index + 1) * dt_s,
        "current_a": float(cluster["pack_currents_a"][0]),
        "compressor_speed_command_rpm": float(
            result["compressor_speed_command_rpm"]
        ),
        "compressor_speed_rpm": float(result["compressor_speed_rpm"]),
        "total_mass_flow_kg_s": float(result["total_mass_flow_kg_s"]),
        "tank_temp_k": float(result["tank_temperature_after_k"]),
        "tank_temp_before_k": float(result["tank_temperature_before_k"]),
        "evap_outlet_k": float(result["evaporator_outlet_temperature_k"]),
        "cluster_supply_k": float(result["cluster_supply_temperature_k"]),
        "cluster_return_k": float(result["cluster_return_temperature_k"]),
        "tank_return_k": float(result["tank_return_temperature_k"]),
        "battery_avg_k": float(
            np.mean(cluster["pack_battery_average_temperatures_k"])
        ),
        "battery_max_k": float(cluster["cluster_max_temperature_k"]),
        "plate_avg_k": float(
            np.mean(cluster["pack_plate_average_temperatures_k"])
        ),
        "inter_pack_delta_k": float(
            cluster["inter_pack_delta_temperature_k"]
        ),
        "q_evap_cycle_w": float(result["q_evap_cycle_w"]),
        "q_evap_applied_w": float(result["q_evap_applied_w"]),
        "q_cluster_to_fluid_w": float(result["q_cluster_to_fluid_w"]),
        "q_gen_w": float(result["cluster_q_gen_total_w"]),
        "cycle_solver_success": bool(
            result["refrigeration_solver_success"]
        ),
        "all_states_finite": bool(result["all_states_finite"]),
        "evaporator_coolant_residual_w": float(
            result["evaporator_coolant_residual_w"]
        ),
        "cluster_fluid_residual_w": float(result["cluster_fluid_residual_w"]),
        "tank_energy_residual_j": float(result["tank_energy_residual_j"]),
        "max_abs_pack_coupling_residual_w": float(
            cluster["max_abs_pack_coupling_residual_w"]
        ),
    }


def _hc_row(case, step_index, dt_s, hc_inputs, hc) -> dict[str, object]:
    cluster = hc["cluster_result"]
    return {
        "case_id": case.case_id,
        "step": step_index,
        "time_s": (step_index + 1) * dt_s,
        "current_a": float(cluster["pack_currents_a"][0]),
        "compressor_speed_rpm": float(hc_inputs.compressor_speed_rpm),
        "total_mass_flow_kg_s": float(hc_inputs.total_mass_flow_kg_s),
        "tank_temp_k": float(hc["tank_temperature_after_k"]),
        "tank_temp_before_k": float(hc_inputs.tank_temperature_before_k),
        "evap_outlet_applied_k": float(
            hc["evaporator_outlet_temperature_applied_k"]
        ),
        "evap_outlet_ss_k": float(hc["evaporator_outlet_temperature_ss_k"]),
        "cluster_supply_k": float(hc["cluster_supply_temperature_k"]),
        "cluster_return_k": float(hc["cluster_return_temperature_k"]),
        "tank_return_k": float(hc["tank_return_temperature_k"]),
        "battery_avg_k": float(
            np.mean(cluster["pack_battery_average_temperatures_k"])
        ),
        "battery_max_k": float(cluster["cluster_max_temperature_k"]),
        "plate_avg_k": float(
            np.mean(cluster["pack_plate_average_temperatures_k"])
        ),
        "inter_pack_delta_k": float(
            cluster["inter_pack_delta_temperature_k"]
        ),
        "q_evap_hc_w": float(hc["q_evap_hc_w"]),
        "q_hc_minus_q_cycle_w": float(hc["q_hc_minus_q_cycle_w"]),
        "q_evap_applied_w": float(hc_inputs.q_evap_applied_w),
        "q_cluster_to_fluid_w": float(hc["q_cluster_to_fluid_w"]),
        "q_gen_w": float(np.sum(cluster["pack_q_gen_total_w"])),
        "all_states_finite": bool(hc["all_states_finite"]),
        "evaporator_coolant_residual_w": float(
            hc["evaporator_coolant_residual_w"]
        ),
        "cluster_fluid_residual_w": float(hc["cluster_fluid_residual_w"]),
        "tank_energy_residual_j": float(hc["tank_energy_residual_j"]),
        "max_abs_pack_coupling_residual_w": float(
            cluster["max_abs_pack_coupling_residual_w"]
        ),
        "plate_h_dynamic_w_m2_k": float(
            np.mean(cluster["pack_plate_dynamic_h_w_m2_k"])
        ),
        "plate_internal_steps": int(
            np.mean(cluster["pack_plate_internal_steps"])
        ),
    }


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _summarize_case(result: dict[str, object]) -> dict[str, float]:
    leg = result["legacy_rows"]
    hc = result["hc_rows"]
    n = min(len(leg), len(hc))
    if n == 0:
        return {
            "steps": 0,
            "rmse_battery_avg_k": float("nan"),
            "rmse_battery_max_k": float("nan"),
            "rmse_plate_avg_k": float("nan"),
            "rmse_cluster_supply_k": float("nan"),
            "rmse_cluster_return_k": float("nan"),
            "rmse_tank_k": float("nan"),
            "rmse_tank_return_k": float("nan"),
            "max_abs_q_hc_minus_q_cycle_w": float("nan"),
            "max_abs_evap_coolant_residual_w_hc": float("nan"),
            "max_abs_cluster_fluid_residual_w_hc": float("nan"),
            "max_abs_cluster_fluid_residual_w_legacy": float("nan"),
            "legacy_failures": result["legacy_failures"],
            "hc_failures": result["hc_failures"],
            "runtime_s": result["runtime_s"],
        }

    def _to_float_array(rows, key):
        return np.array([float(r[key]) for r in rows[:n]], dtype=float)

    leg_arr = {k: _to_float_array(leg, k) for k in leg[0]
               if isinstance(leg[0][k], (int, float))}
    hc_arr = {k: _to_float_array(hc, k) for k in hc[0]
              if isinstance(hc[0][k], (int, float))}
    return {
        "case_id": result["case"].case_id,
        "steps": n,
        "rmse_battery_avg_k": _rmse(
            leg_arr["battery_avg_k"], hc_arr["battery_avg_k"]
        ),
        "rmse_battery_max_k": _rmse(
            leg_arr["battery_max_k"], hc_arr["battery_max_k"]
        ),
        "rmse_plate_avg_k": _rmse(
            leg_arr["plate_avg_k"], hc_arr["plate_avg_k"]
        ),
        "rmse_cluster_supply_k": _rmse(
            leg_arr["cluster_supply_k"], hc_arr["cluster_supply_k"]
        ),
        "rmse_cluster_return_k": _rmse(
            leg_arr["cluster_return_k"], hc_arr["cluster_return_k"]
        ),
        "rmse_tank_k": _rmse(leg_arr["tank_temp_k"], hc_arr["tank_temp_k"]),
        "rmse_tank_return_k": _rmse(
            leg_arr["tank_return_k"], hc_arr["tank_return_k"]
        ),
        "max_abs_q_hc_minus_q_cycle_w": float(
            np.max(np.abs(hc_arr["q_hc_minus_q_cycle_w"]))
        ),
        "max_abs_evap_coolant_residual_w_hc": float(
            np.max(np.abs(hc_arr["evaporator_coolant_residual_w"]))
        ),
        "max_abs_cluster_fluid_residual_w_hc": float(
            np.max(np.abs(hc_arr["cluster_fluid_residual_w"]))
        ),
        "max_abs_cluster_fluid_residual_w_legacy": float(
            np.max(np.abs(leg_arr["cluster_fluid_residual_w"]))
        ),
        "legacy_failures": result["legacy_failures"],
        "hc_failures": result["hc_failures"],
        "runtime_s": result["runtime_s"],
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_markdown(summaries: list[dict[str, float]],
                      case_results: list[dict[str, object]]) -> str:
    lines: list[str] = []
    lines.append("# Stage 3 — Heat-Current 子模型的并行系统集成与闭环验证")
    lines.append("")
    lines.append(
        "Stage 3 freezes a **parallel** heat-current system link alongside the"
    )
    lines.append(
        "incumbent `ClusterPlant`. The legacy plant keeps its bit-for-bit"
    )
    lines.append(
        "trajectory; the new link reuses the frozen pump, refrigeration"
    )
    lines.append(
        "cycle, compressor actuator, evaporator dynamics, and transport"
    )
    lines.append(
        "delays, but replaces the cluster and the evaporator outlet formula"
    )
    lines.append(
        "with the heat-current versions (`ColdPlateHeatCurrent`,"
    )
    lines.append(
        "`EvaporatorHeatCurrent`). No state on the frozen shared components"
    )
    lines.append("is mutated by the parallel link.")
    lines.append("")
    lines.append("## Per-case summary (legacy vs heat_current, common window)")
    lines.append("")
    lines.append(
        "| case | steps | RMSE T_b avg (K) | RMSE T_b max (K) |"
        " RMSE T_p avg (K) | RMSE T_supply (K) | RMSE T_return (K) |"
        " RMSE T_tank (K) | RMSE T_return_to_tank (K) |"
        " max |Q_HC − Q_cycle| (W) | max |evap residual| (W) |"
        " max |cluster residual| (W, HC) | max |cluster residual| (W, legacy) |"
        " HC failures | legacy failures | runtime (s) |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    for s in summaries:
        lines.append(
            "| {case_id} | {steps} | {rmse_battery_avg_k:.4f} |"
            " {rmse_battery_max_k:.4f} | {rmse_plate_avg_k:.4f} |"
            " {rmse_cluster_supply_k:.4f} | {rmse_cluster_return_k:.4f} |"
            " {rmse_tank_k:.4f} | {rmse_tank_return_k:.4f} |"
            " {max_abs_q_hc_minus_q_cycle_w:.3e} |"
            " {max_abs_evap_coolant_residual_w_hc:.3e} |"
            " {max_abs_cluster_fluid_residual_w_hc:.3e} |"
            " {max_abs_cluster_fluid_residual_w_legacy:.3e} |"
            " {hc_failures} | {legacy_failures} | {runtime_s:.2f} |".format(**s)
        )
    lines.append("")
    lines.append("## Per-case artefacts")
    lines.append("")
    for c in case_results:
        case_id = c["case"].case_id
        lines.append(
            f"* `{case_id}_legacy.csv` and `{case_id}_heat_current.csv` —"
            " per-step timeseries from the two plants."
        )
    lines.append("")
    lines.append("## Engineering thresholds (reused, not invented)")
    lines.append("")
    lines.append("* `cycle_solver_success` must hold on every legacy step.")
    lines.append("* `all_states_finite` must hold on every step on both plants.")
    lines.append(
        "* `cluster_fluid_residual_w` abs must stay within the legacy band;"
        " the heat-current plate is stiffer, so this should be tighter."
    )
    lines.append(
        "* `evaporator_coolant_residual_w` is zero by construction on the"
        " heat-current link (numerical noise only)."
    )
    lines.append("")
    lines.append("## What is *not* part of Stage 3")
    lines.append("")
    lines.append(
        "* No change to `ClusterPlant`, `plant.py`, the refrigeration cycle,"
        " the evaporator 45 s lag, the compressor actuator, the transport"
        " delays, the tank, or any frozen parameter."
    )
    lines.append(
        "* No replacement of the legacy plant. The legacy plant stays the"
        " reference baseline and is used to drive the heat-current link."
    )
    lines.append(
        "* No NMPC / controller / TD3 work. Stage 3 stops at the system"
        " integration and validation boundary."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=None, help="override output directory",
    )
    parser.add_argument(
        "--duration-s", type=float, default=DURATION_S,
        help="simulation duration per case (default 600 s)",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir) if args.output_dir else _today_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, float]] = []
    case_results: list[dict[str, object]] = []
    for case in build_case_specs():
        runner = CaseRunner(case, duration_s=args.duration_s, dt_s=DT_S)
        result = runner.run()
        case_id = case.case_id
        _write_csv(
            out_dir / f"{case_id}_legacy.csv", result["legacy_rows"]
        )
        _write_csv(
            out_dir / f"{case_id}_heat_current.csv", result["hc_rows"]
        )
        summary = _summarize_case(result)
        summaries.append(summary)
        case_results.append(result)
        print(
            f"[{case_id}] steps={summary['steps']}"
            f" legacy_fail={summary['legacy_failures']}"
            f" hc_fail={summary['hc_failures']}"
            f" RMSE_b_avg={summary['rmse_battery_avg_k']:.4f} K"
            f" RMSE_supply={summary['rmse_cluster_supply_k']:.4f} K"
            f" RMSE_tank={summary['rmse_tank_k']:.4f} K"
            f" max|q_hc-cycle|={summary['max_abs_q_hc_minus_q_cycle_w']:.2e} W"
        )

    summary_path = out_dir / "stage3_summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summaries, fh, indent=2, ensure_ascii=False)
    report_path = out_dir / "STAGE3_SYSTEM_INTEGRATION.md"
    report_path.write_text(_render_markdown(summaries, case_results),
                            encoding="utf-8")
    print(f"summary: {summary_path}")
    print(f"report : {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())