"""Formal 600 s benchmark: frozen State-Space QP-MPC versus GEKKO MPC.

The runner keeps the Plant, Physics-P artifact, controller constraints, initial
conditions, and disturbance profile identical within each controller pair.
It writes raw CSVs, compact summaries, a progress file, and a freeze decision.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import sys
from typing import Callable

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    # Allow PyCharm's "Run file" action as well as ``python -m``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from single_pack_plant.controllers.gekko.mpc import runtime_mpc_params_for_scene
    from single_pack_plant.controllers.fixed_qp.mpc import QPMPCWeights
    from single_pack_plant.simulation.config import (
        AGC_DATA_FILE,
        AMBIENT_TEMP_C,
        N_COMP_MAX_RPM,
        N_COMP_OFF_RPM,
        N_PUMP_MAX_RPM,
        N_PUMP_MIN_RPM,
        SIM_DT,
        TARGET_TEMP_C,
    )
    from single_pack_plant.simulation.case import load_current_profile, simulate_case
else:
    from ..controllers.gekko.mpc import runtime_mpc_params_for_scene
    from ..controllers.fixed_qp.mpc import QPMPCWeights
    from ..simulation.config import (
        AGC_DATA_FILE,
        AMBIENT_TEMP_C,
        N_COMP_MAX_RPM,
        N_COMP_OFF_RPM,
        N_PUMP_MAX_RPM,
        N_PUMP_MIN_RPM,
        SIM_DT,
        TARGET_TEMP_C,
    )
    from ..simulation.case import load_current_profile, simulate_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT = ROOT / "model_data" / "physics_p_operational_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "state_space_qp_600s_comparison"
QP_ONLY_OUTPUT_ROOT = ROOT / "outputs" / "state_space_qp_600s_only"
PROFILE_NAMES = ("constant", "current_step", "regd")
CONTROLLER_NAMES = ("qp", "gekko")
PREVIEW_STEPS = 60
# This validation experiment starts both the battery and thermal system at the
# MPC target.  The global ambient-temperature parameter remains unchanged.
INITIAL_VALIDATION_TEMP_C = 25.0

_FROZEN_QP_WEIGHTS = QPMPCWeights(
    # Temperature-priority setting verified on the 600 s peak and RegD cases.
    q_y=10.0,
    r_comp=1.0,
    r_pump=1.0e-3,
    r_delta_comp=1.0e-2,
    r_delta_pump=1.0e-2,
    p_f=1.0,
)


@dataclass(frozen=True)
class ComparisonCase:
    profile: str
    controller: str
    control: str
    scene: str
    flow: str
    dt: float
    duration_s: float
    predictor: str
    current_profile: np.ndarray


def frozen_qp_weights() -> QPMPCWeights:
    """Return the frozen QP-MPC weights used by the formal benchmark."""

    return _FROZEN_QP_WEIGHTS


def build_case(
    profile: str,
    controller: str,
    *,
    duration_s: float = 600.0,
    dt: float = SIM_DT,
) -> ComparisonCase:
    """Build one controller case with a shared disturbance-preview contract."""

    profile_key = str(profile).strip().lower()
    controller_key = str(controller).strip().lower()
    if profile_key not in PROFILE_NAMES:
        raise ValueError(f"unknown profile: {profile!r}")
    if controller_key not in CONTROLLER_NAMES:
        raise ValueError(f"unknown controller: {controller!r}")
    duration = float(duration_s)
    step = float(dt)
    if duration <= 0.0 or step <= 0.0 or duration % step:
        raise ValueError("duration_s must be a positive multiple of dt")

    episode_steps = int(round(duration / step))
    times = np.arange(episode_steps + PREVIEW_STEPS, dtype=float) * step
    if profile_key == "constant":
        scene = "调峰"
        current = np.full(times.size, 560.0, dtype=float)
    elif profile_key == "current_step":
        scene = "调峰"
        current = np.full(times.size, 840.0, dtype=float)
        current[times < duration / 2.0] = 280.0
    else:
        scene = "调频"
        current = load_current_profile("freq", times, agc_data_file=AGC_DATA_FILE)

    return ComparisonCase(
        profile=profile_key,
        controller=controller_key,
        control="state_space_qp" if controller_key == "qp" else "mpc",
        scene=scene,
        flow="单向",
        dt=step,
        duration_s=duration,
        predictor="physics_p",
        current_profile=np.asarray(current, dtype=float),
    )


def _bool_values(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
        .to_numpy(dtype=bool)
    )


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _energy_kwh(frame: pd.DataFrame, column: str, dt: float) -> float:
    values = np.nan_to_num(_numeric(frame, column), nan=0.0)
    return float(np.sum(values) * float(dt) / 3600.0)


def _maximum_constraint_violation(
    commands: np.ndarray, *, scene: str
) -> tuple[float, float, float]:
    params = runtime_mpc_params_for_scene(scene)
    lower = np.array([N_COMP_OFF_RPM, N_PUMP_MIN_RPM], dtype=float)
    upper = np.array([N_COMP_MAX_RPM, N_PUMP_MAX_RPM], dtype=float)
    previous = np.array([1500.0, 3000.0], dtype=float)
    moves = np.diff(np.vstack([previous, commands]), axis=0)
    dmax = np.array([params.dmax_comp, params.dmax_pump], dtype=float)
    bound_violation = float(
        max(
            0.0,
            np.max(lower - commands),
            np.max(commands - upper),
        )
    )
    rate_violation = float(max(0.0, np.max(np.abs(moves) - dmax)))
    return bound_violation, rate_violation, max(bound_violation, rate_violation)


def summarize_frame(
    frame: pd.DataFrame,
    *,
    profile: str,
    controller: str,
    dt: float,
) -> dict:
    """Summarize physical, actuator, domain, and solver results from one CSV."""

    commands = np.column_stack(
        [_numeric(frame, "Compressor command"), _numeric(frame, "Pump command")]
    )
    actual = np.column_stack(
        [_numeric(frame, "Compressor Speed"), _numeric(frame, "Pump Speed (RPM)")]
    )
    t_avg = _numeric(frame, "Average temperature")
    t_max = _numeric(frame, "T_cell_max_C")
    delta_t = _numeric(frame, "Delta_T_cell_C")
    solve_times = _numeric(frame, "MPC solve time")
    finite_solve_times = solve_times[np.isfinite(solve_times)]
    solved = _bool_values(frame["MPC_Solved"])
    fallback = _bool_values(frame["MPC solve recovery used"])
    domain = _bool_values(frame["MPC prediction domain valid"])
    domain_available = frame["MPC prediction domain valid"].notna().to_numpy()
    scene = "调频" if str(profile).lower() == "regd" else "调峰"
    bound_v, rate_v, constraint_v = _maximum_constraint_violation(
        commands, scene=scene
    )
    core = np.column_stack([t_avg, t_max, delta_t, commands, actual])
    total_energy = float(_numeric(frame, "Cumulative energy consumption")[-1])
    return {
        "profile": str(profile),
        "controller": str(controller),
        "steps": int(len(frame)),
        "all_core_values_finite": bool(np.all(np.isfinite(core))),
        "t_avg_mean_c": float(np.nanmean(t_avg)),
        "t_avg_max_c": float(np.nanmax(t_avg)),
        "t_max_max_c": float(np.nanmax(t_max)),
        "delta_t_mean_c": float(np.nanmean(delta_t)),
        "delta_t_max_c": float(np.nanmax(delta_t)),
        "compressor_energy_kwh": _energy_kwh(
            frame, "Compressor power (kW)", dt
        ),
        "pump_energy_kwh": _energy_kwh(frame, "Pump power (kW)", dt),
        "total_energy_kwh": total_energy,
        "n_comp_command_mean_rpm": float(np.nanmean(commands[:, 0])),
        "n_comp_command_min_rpm": float(np.nanmin(commands[:, 0])),
        "n_comp_command_max_rpm": float(np.nanmax(commands[:, 0])),
        "n_comp_command_above_5800_rate": float(np.mean(commands[:, 0] > 5800.0)),
        "n_comp_actual_mean_rpm": float(np.nanmean(actual[:, 0])),
        "n_comp_actual_min_rpm": float(np.nanmin(actual[:, 0])),
        "n_comp_actual_max_rpm": float(np.nanmax(actual[:, 0])),
        "n_pump_command_mean_rpm": float(np.nanmean(commands[:, 1])),
        "n_pump_command_min_rpm": float(np.nanmin(commands[:, 1])),
        "n_pump_command_max_rpm": float(np.nanmax(commands[:, 1])),
        "n_pump_actual_mean_rpm": float(np.nanmean(actual[:, 1])),
        "n_pump_actual_min_rpm": float(np.nanmin(actual[:, 1])),
        "n_pump_actual_max_rpm": float(np.nanmax(actual[:, 1])),
        "solved_rate": float(np.mean(solved)),
        "fallback_count": int(np.count_nonzero(fallback)),
        "domain_valid_rate": float(np.mean(domain)),
        "domain_available_rate": float(np.mean(domain_available)),
        "solve_time_median_s": float(np.nanmedian(finite_solve_times)),
        "solve_time_p95_s": float(np.nanpercentile(finite_solve_times, 95.0)),
        "solve_time_p99_s": float(np.nanpercentile(finite_solve_times, 99.0)),
        "solve_time_max_s": float(np.nanmax(finite_solve_times)),
        "input_bound_violation_rpm": bound_v,
        "rate_constraint_violation_rpm": rate_v,
        "constraint_max_violation_rpm": constraint_v,
    }


def _comparison_rows(summary: pd.DataFrame) -> list[dict]:
    rows = []
    for profile in PROFILE_NAMES:
        indexed = summary.loc[summary["profile"] == profile].set_index("controller")
        qp = indexed.loc["qp"]
        gekko = indexed.loc["gekko"]
        rows.append(
            {
                "profile": profile,
                "qp_minus_gekko_t_avg_max_c": qp["t_avg_max_c"] - gekko["t_avg_max_c"],
                "qp_minus_gekko_t_max_max_c": qp["t_max_max_c"] - gekko["t_max_max_c"],
                "qp_minus_gekko_delta_t_max_c": qp["delta_t_max_c"] - gekko["delta_t_max_c"],
                "qp_total_energy_change_rate": (
                    qp["total_energy_kwh"] / gekko["total_energy_kwh"] - 1.0
                ),
                "solve_time_median_speedup": (
                    gekko["solve_time_median_s"] / qp["solve_time_median_s"]
                ),
            }
        )
    return rows


def evaluate_freeze(summary: pd.DataFrame, comparison: pd.DataFrame) -> dict:
    """Apply the acceptance gates frozen in STATE_SPACE_MPC_DESIGN.md."""

    qp = summary.loc[summary["controller"] == "qp"]
    checks = {
        "all_cases_complete_and_finite": bool(
            len(summary) == 6
            and np.all(summary["steps"] == 120)
            and np.all(summary["all_core_values_finite"])
        ),
        "qp_solved_rate_at_least_99pct": bool(np.all(qp["solved_rate"] >= 0.99)),
        "qp_constraint_violation_at_most_1e_6_rpm": bool(
            np.all(qp["constraint_max_violation_rpm"] <= 1.0e-6)
        ),
        "qp_domain_valid_rate_at_least_99pct": bool(
            np.all(qp["domain_valid_rate"] >= 0.99)
        ),
        "temperature_max_degradation_at_most_0p1_c": bool(
            np.all(comparison["qp_minus_gekko_t_avg_max_c"] <= 0.1)
            and np.all(comparison["qp_minus_gekko_t_max_max_c"] <= 0.1)
        ),
        "delta_t_max_degradation_at_most_0p05_c": bool(
            np.all(comparison["qp_minus_gekko_delta_t_max_c"] <= 0.05)
        ),
        "total_energy_increase_at_most_5pct": bool(
            np.all(comparison["qp_total_energy_change_rate"] <= 0.05)
        ),
        "qp_median_solve_at_least_10x_faster": bool(
            np.all(comparison["solve_time_median_speedup"] >= 10.0)
        ),
        "qp_p99_solve_below_5_s": bool(np.all(qp["solve_time_p99_s"] < SIM_DT)),
    }
    return {"checks": checks, "freeze_qp_baseline": bool(all(checks.values()))}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, record in frame[columns].iterrows():
        values = []
        for value in record:
            values.append(f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    output_root: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    decision: dict,
) -> Path:
    metric_columns = [
        "profile",
        "controller",
        "t_avg_max_c",
        "t_max_max_c",
        "delta_t_max_c",
        "compressor_energy_kwh",
        "pump_energy_kwh",
        "total_energy_kwh",
        "solved_rate",
        "fallback_count",
        "domain_valid_rate",
        "solve_time_median_s",
        "solve_time_p95_s",
        "solve_time_max_s",
    ]
    speed_columns = [
        "profile",
        "controller",
        "n_comp_command_mean_rpm",
        "n_comp_actual_mean_rpm",
        "n_pump_command_mean_rpm",
        "n_pump_actual_mean_rpm",
        "constraint_max_violation_rpm",
    ]
    comparison_columns = list(comparison.columns)
    check_lines = [
        f"- {'PASS' if passed else 'FAIL'}: `{name}`"
        for name, passed in decision["checks"].items()
    ]
    verdict = "可以冻结" if decision["freeze_qp_baseline"] else "暂不能冻结"
    report = f"""# State-Space QP-MPC 600 s 正式对比

## 冻结运行合同

- Plant：现有 `thermal_case_simulator -> thermal_loop -> thermal_system -> pack`
- Predictor：`model_data/physics_p_operational_v1.json`
- QP weights：`{json.dumps(asdict(frozen_qp_weights()), ensure_ascii=False)}`
- `dt=5 s`，每个工况 `600 s / 120 steps`
- 工况：Constant、Current step、RegD
- 控制器：State-Space QP-MPC、现有 GEKKO Physics-P MPC
- 流向：单向 standard
- 初始电池温度 25 degC；初始冷却系统温度 25 degC；目标 25 degC

## 温度、能耗与求解

{_markdown_table(summary, metric_columns)}

## 转速与约束

{_markdown_table(summary, speed_columns)}

## QP 相对 GEKKO

{_markdown_table(comparison, comparison_columns)}

## 验收门槛

{chr(10).join(check_lines)}

## 结论

**{verdict} State-Space QP-MPC baseline。**
"""
    path = output_root / "STATE_SPACE_QP_600S_COMPARISON_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_case(
    case: ComparisonCase,
    output_root: Path,
    *,
    force: bool,
    linear_model_bank: Path | None = None,
    qp_weights: QPMPCWeights | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict:
    label = f"{case.profile}_600s_{case.controller}"
    result = simulate_case(
        control=case.control,
        scene=case.scene,
        flow=case.flow,
        source_csv=ROOT / "_state_space_qp_comparison_time_axis.csv",
        main_name=f"{label}.csv",
        snap_name=f"{label}_snapshots.csv",
        output_root=output_root,
        dt=case.dt,
        target_temp_c=TARGET_TEMP_C,
        initial_thermal_temp_c=INITIAL_VALIDATION_TEMP_C,
        duration_s=case.duration_s,
        result_tag="frozen_qp_vs_gekko_600s",
        current_profile_override=case.current_profile,
        mpc_flow_mode="standard",
        mpc_predictor=case.predictor,
        mpc_predictor_artifact=DEFAULT_ARTIFACT,
        state_space_qp_weights=(
            (qp_weights or frozen_qp_weights()) if case.controller == "qp" else None
        ),
        state_space_qp_linear_model_bank=(
            linear_model_bank if case.controller == "qp" else None
        ),
        mpc_forecast_profile_steps=case.current_profile.size,
        force=force,
        progress_interval_steps=20,
        log_func=progress_callback,
    )
    frame = pd.read_csv(result["out_csv"], encoding="utf-8-sig")
    expected_steps = int(round(case.duration_s / case.dt))
    if len(frame) != expected_steps:
        raise RuntimeError(
            f"{label} has {len(frame)} rows; expected {expected_steps}"
        )
    summary = summarize_frame(
        frame,
        profile=case.profile,
        controller=case.controller,
        dt=case.dt,
    )
    summary["csv"] = str(result["out_csv"])
    return summary


def run_formal_comparison(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
    linear_model_bank: Path | None = None,
) -> dict:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    summaries: list[dict] = []
    total_cases = len(PROFILE_NAMES) * len(CONTROLLER_NAMES)

    contract = {
        "duration_s": 600.0,
        "dt_s": SIM_DT,
        "profiles": list(PROFILE_NAMES),
        "controllers": list(CONTROLLER_NAMES),
        "physics_p_artifact": str(DEFAULT_ARTIFACT),
        "qp_weights": asdict(frozen_qp_weights()),
        "plant_initial_temperature_c": 25.0,
        "thermal_system_initial_temperature_c": INITIAL_VALIDATION_TEMP_C,
        "target_temperature_c": TARGET_TEMP_C,
        "flow": "单向",
        "flow_mode": "standard",
    }
    _write_json(output_root / "run_contract.json", contract)

    case_number = 0
    for profile in PROFILE_NAMES:
        for controller in CONTROLLER_NAMES:
            case_number += 1
            case = build_case(profile, controller)
            progress = {
                "status": "running",
                "case_number": case_number,
                "total_cases": total_cases,
                "profile": profile,
                "controller": controller,
                "last_message": "starting",
                "completed_cases": len(summaries),
            }
            _write_json(progress_path, progress)

            def update_progress(message: str) -> None:
                if message.startswith("PROGRESS"):
                    progress["last_message"] = message
                    _write_json(progress_path, progress)

            summary = run_case(
                case,
                output_root,
                force=force,
                linear_model_bank=linear_model_bank,
                progress_callback=update_progress,
            )
            summaries.append(summary)
            progress.update(
                {
                    "last_message": "completed",
                    "completed_cases": len(summaries),
                }
            )
            _write_json(progress_path, progress)
            print(
                f"DONE {case_number}/{total_cases} {profile} {controller}: "
                f"solved={summary['solved_rate']:.1%}, "
                f"Tmax={summary['t_max_max_c']:.4f}C, "
                f"E={summary['total_energy_kwh']:.6f}kWh",
                flush=True,
            )

    summary_frame = pd.DataFrame(summaries)
    comparison_frame = pd.DataFrame(_comparison_rows(summary_frame))
    decision = evaluate_freeze(summary_frame, comparison_frame)
    summary_path = output_root / "summary.csv"
    comparison_path = output_root / "pairwise_comparison.csv"
    decision_path = output_root / "freeze_decision.json"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    comparison_frame.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    _write_json(decision_path, decision)
    report_path = _write_report(
        output_root, summary_frame, comparison_frame, decision
    )
    _write_json(
        progress_path,
        {
            "status": "completed",
            "completed_cases": total_cases,
            "total_cases": total_cases,
            "freeze_qp_baseline": decision["freeze_qp_baseline"],
            "report": str(report_path),
        },
    )
    return {
        "summary_csv": str(summary_path),
        "comparison_csv": str(comparison_path),
        "decision_json": str(decision_path),
        "report": str(report_path),
        "freeze_qp_baseline": decision["freeze_qp_baseline"],
    }


def _write_qp_only_report(
    output_root: Path,
    summary: pd.DataFrame,
    weights: QPMPCWeights,
) -> Path:
    """Write the compact report for the three-case QP-only validation."""

    check_values = {
        "all_cases_complete_and_finite": bool(
            len(summary) == len(PROFILE_NAMES)
            and np.all(summary["steps"] == 120)
            and np.all(summary["all_core_values_finite"])
        ),
        "solved_rate_at_least_99pct": bool(np.all(summary["solved_rate"] >= 0.99)),
        "fallback_count_zero": bool(np.all(summary["fallback_count"] == 0)),
        "constraint_violation_at_most_1e_6_rpm": bool(
            np.all(summary["constraint_max_violation_rpm"] <= 1.0e-6)
        ),
        "domain_valid_rate_at_least_99pct": bool(
            np.all(summary["domain_valid_rate"] >= 0.99)
        ),
        "qp_p99_solve_below_5_s": bool(np.all(summary["solve_time_p99_s"] < SIM_DT)),
    }
    passed = all(check_values.values())
    metric_columns = [
        "profile",
        "t_avg_max_c",
        "t_max_max_c",
        "delta_t_max_c",
        "compressor_energy_kwh",
        "pump_energy_kwh",
        "total_energy_kwh",
        "n_comp_command_mean_rpm",
        "n_comp_command_max_rpm",
        "n_pump_command_mean_rpm",
        "n_pump_command_max_rpm",
        "solved_rate",
        "fallback_count",
        "domain_valid_rate",
        "solve_time_median_s",
        "solve_time_p99_s",
        "constraint_max_violation_rpm",
    ]
    check_lines = [
        f"- {'PASS' if value else 'FAIL'}: `{name}`"
        for name, value in check_values.items()
    ]
    report = f"""# State-Space QP-MPC 600 s 完整测试

本报告只包含 State-Space QP-MPC，不启动或比较 GEKKO。

- Plant、Physics-P artifact、初始条件、约束和 `dt=5 s` 保持不变。
- 工况：Constant、Current step、RegD；每个工况 `600 s / 120 steps`。
- QP 权重：`{json.dumps(asdict(weights), ensure_ascii=False)}`
- 离线模型库：由运行命令的 `--linear-model-bank` 指定时启用。

## 结果

{_markdown_table(summary, metric_columns)}

## 验收检查

{chr(10).join(check_lines)}

## 结论

**{'QP-MPC 完整测试通过' if passed else 'QP-MPC 完整测试未通过全部门槛'}。**
"""
    path = output_root / "STATE_SPACE_QP_600S_ONLY_REPORT.md"
    path.write_text(report, encoding="utf-8")
    return path


def run_qp_only_600s(
    output_root: Path = QP_ONLY_OUTPUT_ROOT,
    *,
    force: bool = False,
    linear_model_bank: Path | None = None,
    qp_weights: QPMPCWeights | None = None,
) -> dict:
    """Run the complete Constant/Current-step/RegD QP-only validation."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    selected_weights = (qp_weights or frozen_qp_weights()).validated()
    summaries: list[dict] = []
    total_cases = len(PROFILE_NAMES)
    contract = {
        "duration_s": 600.0,
        "dt_s": SIM_DT,
        "profiles": list(PROFILE_NAMES),
        "controllers": ["qp"],
        "physics_p_artifact": str(DEFAULT_ARTIFACT),
        "qp_weights": asdict(selected_weights),
        "linear_model_bank": str(linear_model_bank) if linear_model_bank else None,
        "plant_initial_temperature_c": 25.0,
        "thermal_system_initial_temperature_c": INITIAL_VALIDATION_TEMP_C,
        "target_temperature_c": TARGET_TEMP_C,
        "flow": "单向",
        "flow_mode": "standard",
    }
    _write_json(output_root / "run_contract.json", contract)

    for case_number, profile in enumerate(PROFILE_NAMES, start=1):
        case = build_case(profile, "qp")
        progress = {
            "status": "running",
            "case_number": case_number,
            "total_cases": total_cases,
            "profile": profile,
            "controller": "qp",
            "last_message": "starting",
            "completed_cases": len(summaries),
        }
        _write_json(progress_path, progress)

        def update_progress(message: str) -> None:
            if message.startswith("PROGRESS"):
                progress["last_message"] = message
                _write_json(progress_path, progress)

        summary = run_case(
            case,
            output_root,
            force=force,
            linear_model_bank=linear_model_bank,
            qp_weights=selected_weights,
            progress_callback=update_progress,
        )
        summaries.append(summary)
        progress.update(
            {"last_message": "completed", "completed_cases": len(summaries)}
        )
        _write_json(progress_path, progress)
        print(
            f"DONE {case_number}/{total_cases} {profile} qp: "
            f"solved={summary['solved_rate']:.1%}, "
            f"Tmax={summary['t_max_max_c']:.4f}C, "
            f"E={summary['total_energy_kwh']:.6f}kWh",
            flush=True,
        )

    summary_frame = pd.DataFrame(summaries)
    summary_path = output_root / "summary.csv"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report_path = _write_qp_only_report(
        output_root, summary_frame, selected_weights
    )
    complete = bool(
        len(summary_frame) == total_cases
        and np.all(summary_frame["steps"] == 120)
        and np.all(summary_frame["all_core_values_finite"])
        and np.all(summary_frame["solved_rate"] >= 0.99)
        and np.all(summary_frame["fallback_count"] == 0)
    )
    _write_json(
        progress_path,
        {
            "status": "completed",
            "completed_cases": total_cases,
            "total_cases": total_cases,
            "all_primary_checks_passed": complete,
            "summary_csv": str(summary_path),
            "report": str(report_path),
        },
    )
    return {
        "summary_csv": str(summary_path),
        "report": str(report_path),
        "output_root": str(output_root),
        "all_primary_checks_passed": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="结果目录；--qp-only 未指定时默认为 state_space_qp_600s_only",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--linear-model-bank", type=Path, default=None)
    parser.add_argument(
        "--q-y",
        type=float,
        default=None,
        help="仅覆盖 QP 温度跟踪权重；未指定时保持冻结值 1.0",
    )
    parser.add_argument(
        "--qp-only",
        action="store_true",
        help="只跑 Constant、Current step、RegD 三个 600 s QP-MPC 工况，不启动 GEKKO（默认）",
    )
    parser.add_argument(
        "--gekko-compare",
        action="store_true",
        help="显式运行原来的 QP 与 GEKKO 六任务对比模式",
    )
    args = parser.parse_args()
    run_qp_only = bool(args.qp_only or not args.gekko_compare)
    output_root = args.output_root or (
        QP_ONLY_OUTPUT_ROOT if run_qp_only else DEFAULT_OUTPUT_ROOT
    )
    selected_qp_weights = (
        replace(frozen_qp_weights(), q_y=float(args.q_y)).validated()
        if args.q_y is not None
        else frozen_qp_weights()
    )
    if run_qp_only:
        result = run_qp_only_600s(
            output_root,
            force=args.force,
            linear_model_bank=args.linear_model_bank,
            qp_weights=selected_qp_weights,
        )
    else:
        result = run_formal_comparison(
            output_root,
            force=args.force,
            linear_model_bank=args.linear_model_bank,
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
