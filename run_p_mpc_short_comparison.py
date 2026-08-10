"""Run a deliberately short Candidate-B versus Physics-P MPC comparison.

This is an experimental controller-usefulness check.  Both predictors use the
same detailed plant, current profile, MPC weights, rate limits, and shortened
prediction horizon.  Candidate B remains the default production predictor.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from mpc_predictor_selection import CANDIDATE_B, PHYSICS_P
from thermal_batch_config import CASES
from thermal_case_simulator import simulate_case


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_P_ARTIFACT = (
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "thermal_refinement_v1"
    / "artifacts"
    / "physics_p_cubic20_direct45_constant_kbp.json"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "outputs" / "p_mpc_short_comparison_v1" / "closed_loop"
)
SCENES = {
    "peak": "调峰",
    "freq": "调频",
}


def source_csv_for_scene(scene):
    matches = [
        case.source_csv
        for case in CASES
        if case.control == "mpc" and case.scene == scene and case.flow == "单向"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one MPC single-flow source for {scene}, got {len(matches)}")
    return matches[0]


def _column(frame, *names):
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"Missing required columns: {names}")


def summarize_run(frame, *, scene_key, predictor, horizon, artifact_path, out_csv):
    temperature = _column(
        frame,
        "Average temperature",
        "Battery Temp (C)",
    ).astype(float)
    energy = _column(
        frame,
        "Cumulative energy consumption",
        "Cumulative Energy (kWh)",
    ).astype(float)
    power = _column(frame, "Total power", "Total Power (kW)").astype(float)
    solve_time = _column(frame, "MPC solve time", "MPC_Solve_Time_S").astype(float)
    solved = _column(frame, "MPC_Solved").astype(str).str.lower().isin(("true", "1"))
    n_comp = _column(
        frame,
        "Compressor command",
        "Compressor Command (RPM)",
    ).astype(float)
    n_pump = _column(frame, "Pump command", "Pump Command (RPM)").astype(float)
    target_error = temperature - 25.0
    finite_solve_time = solve_time[np.isfinite(solve_time)]
    if "MPC solve error" in frame.columns:
        solve_errors = frame["MPC solve error"].fillna("").astype(str).str.strip()
    elif "MPC_Solve_Error" in frame.columns:
        solve_errors = frame["MPC_Solve_Error"].fillna("").astype(str).str.strip()
    else:
        solve_errors = pd.Series([""] * len(frame), index=frame.index, dtype=str)
    unique_solve_errors = list(dict.fromkeys(solve_errors[solve_errors != ""]))
    recovery_used_column = next(
        (
            name
            for name in ("MPC solve recovery used", "MPC_Solve_Recovery_Used")
            if name in frame.columns
        ),
        None,
    )
    if recovery_used_column is not None:
        recovery_used = (
            frame[recovery_used_column]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(("true", "1"))
        )
    else:
        recovery_used = pd.Series(False, index=frame.index, dtype=bool)
    recovery_reason_column = next(
        (
            name
            for name in ("MPC solve recovery reason", "MPC_Solve_Recovery_Reason")
            if name in frame.columns
        ),
        None,
    )
    if recovery_reason_column is not None:
        recovery_reasons = (
            frame[recovery_reason_column].fillna("").astype(str).str.strip()
        )
    else:
        recovery_reasons = pd.Series([""] * len(frame), index=frame.index, dtype=str)
    unique_recovery_reasons = list(
        dict.fromkeys(recovery_reasons[recovery_reasons != ""])
    )
    if "MPC_Strict_Predictor_Ablation" in frame.columns:
        strict_predictor_ablation = bool(
            frame["MPC_Strict_Predictor_Ablation"]
            .fillna(False)
            .astype(str)
            .str.lower()
            .isin(("true", "1"))
            .all()
        )
    else:
        strict_predictor_ablation = False
    return {
        "scene": scene_key,
        "predictor": predictor,
        "strict_predictor_ablation": strict_predictor_ablation,
        "steps": int(len(frame)),
        "horizon_steps": int(horizon),
        "temperature_mae_c": float(np.mean(np.abs(target_error))),
        "temperature_max_c": float(np.max(temperature)),
        "temperature_final_c": float(temperature.iloc[-1]),
        "energy_kwh": float(energy.iloc[-1]),
        "mean_power_kw": float(np.mean(power)),
        "mean_n_comp_rpm": float(np.mean(n_comp)),
        "mean_n_pump_rpm": float(np.mean(n_pump)),
        "solve_success_rate": float(np.mean(solved)),
        "solve_time_mean_s": (
            float(np.mean(finite_solve_time)) if len(finite_solve_time) else np.nan
        ),
        "solve_time_p95_s": (
            float(np.percentile(finite_solve_time, 95))
            if len(finite_solve_time)
            else np.nan
        ),
        "solve_recovery_count": int(np.count_nonzero(recovery_used)),
        "solve_recovery_reasons": " | ".join(unique_recovery_reasons),
        "solve_error_count": int(np.count_nonzero(solve_errors != "")),
        "solve_error_messages": " | ".join(unique_solve_errors),
        "artifact": str(artifact_path) if artifact_path is not None else "",
        "out_csv": str(out_csv),
    }


def build_pairwise_summary(summary):
    rows = []
    for scene_key, group in summary.groupby("scene", sort=False):
        indexed = group.set_index("predictor")
        if CANDIDATE_B not in indexed.index or PHYSICS_P not in indexed.index:
            continue
        candidate_b = indexed.loc[CANDIDATE_B]
        physics_p = indexed.loc[PHYSICS_P]
        rows.append(
            {
                "scene": scene_key,
                "p_minus_b_temperature_mae_c": (
                    physics_p["temperature_mae_c"]
                    - candidate_b["temperature_mae_c"]
                ),
                "p_minus_b_temperature_final_c": (
                    physics_p["temperature_final_c"]
                    - candidate_b["temperature_final_c"]
                ),
                "p_minus_b_energy_kwh": (
                    physics_p["energy_kwh"] - candidate_b["energy_kwh"]
                ),
                "p_over_b_solve_time_ratio": (
                    physics_p["solve_time_mean_s"]
                    / candidate_b["solve_time_mean_s"]
                    if candidate_b["solve_time_mean_s"] > 0.0
                    else np.nan
                ),
                "candidate_b_solve_success_rate": candidate_b["solve_success_rate"],
                "physics_p_solve_success_rate": physics_p["solve_success_rate"],
            }
        )
    return pd.DataFrame(rows)


def run_comparison(
    *,
    scenes,
    predictors=(CANDIDATE_B, PHYSICS_P),
    steps,
    horizon,
    artifact_path=DEFAULT_P_ARTIFACT,
    output_root=DEFAULT_OUTPUT_ROOT,
    force=False,
    progress_interval_steps=100,
    forecast_profile_steps=None,
    strict_predictor_ablation=False,
    log_func=None,
):
    artifact_path = Path(artifact_path)
    output_root = Path(output_root)
    predictors = tuple(predictors)
    if forecast_profile_steps is not None:
        forecast_profile_steps = int(forecast_profile_steps)
        if forecast_profile_steps < 1:
            raise ValueError("forecast_profile_steps must be at least 1")

    if PHYSICS_P in predictors and not artifact_path.exists():
        raise FileNotFoundError(f"Physics-P artifact not found: {artifact_path}")
    output_root.mkdir(parents=True, exist_ok=True)
    if log_func is None:
        log_func = lambda message: print(message, flush=True)


    rows = []
    for scene_key in scenes:
        scene = SCENES[scene_key]
        source_csv = source_csv_for_scene(scene)
        for predictor in predictors:
            scope_tag = "_strict_ablation" if strict_predictor_ablation else ""
            stem = (
                f"{scene_key}_{predictor}{scope_tag}_steps{steps}_horizon{horizon}"
            )
            log_func(
                f"RUN scene={scene_key} predictor={predictor} "
                f"steps={steps} horizon={horizon}"
            )
            result = simulate_case(
                "mpc",
                scene,
                "单向",
                source_csv,
                f"{stem}.csv",
                f"{stem}_snapshot.csv",
                output_root=output_root,
                force=force,
                max_steps=steps,
                target_temp_c=25.0,
                mpc_flow_mode="standard",
                result_tag=stem,
                mpc_predictor=predictor,
                mpc_predictor_artifact=(
                    artifact_path if predictor == PHYSICS_P else None
                ),
                mpc_horizon_override=horizon,
                strict_predictor_ablation=strict_predictor_ablation,
                mpc_forecast_profile_steps=forecast_profile_steps,
                progress_interval_steps=progress_interval_steps,
                log_func=log_func,
            )
            out_csv = Path(result["out_csv"])
            frame = pd.read_csv(out_csv, encoding="utf-8-sig")
            rows.append(
                summarize_run(
                    frame,
                    scene_key=scene_key,
                    predictor=predictor,
                    horizon=horizon,
                    artifact_path=(
                        artifact_path if predictor == PHYSICS_P else None
                    ),
                    out_csv=out_csv,
                )
            )
            pd.DataFrame(rows).to_csv(
                output_root / "短闭环指标汇总.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame(rows)
    pairwise = build_pairwise_summary(summary)
    summary_path = output_root / "短闭环指标汇总.csv"
    pairwise_path = output_root / "P相对CandidateB差值.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    return summary_path, pairwise_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Short closed-loop Candidate-B versus Physics-P comparison"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENES),
        default=list(SCENES),
    )
    parser.add_argument(
        "--predictors",
        nargs="+",
        choices=(CANDIDATE_B, PHYSICS_P),
        default=[CANDIDATE_B, PHYSICS_P],
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--strict-predictor-ablation",
        action="store_true",
        help=(
            "align actuator bounds, solver tolerances, control blocking, and "
            "recovery so only the internal predictor package differs"
        ),
    )
    parser.add_argument(
        "--forecast-profile-steps",
        type=int,
        default=None,
        help="retain this many future load samples even when --steps is shorter",
    )
    parser.add_argument("--artifact", type=Path, default=DEFAULT_P_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite matching per-run CSV files instead of reusing them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    if args.horizon < 2:
        raise SystemExit("--horizon must be at least 2")
    if args.forecast_profile_steps is not None and args.forecast_profile_steps < 1:
        raise SystemExit("--forecast-profile-steps must be at least 1")
    summary_path, pairwise_path = run_comparison(
        scenes=args.scenes,
        predictors=args.predictors,
        steps=args.steps,
        horizon=args.horizon,
        artifact_path=args.artifact,
        output_root=args.output_root,
        force=args.force,
        forecast_profile_steps=args.forecast_profile_steps,
        strict_predictor_ablation=args.strict_predictor_ablation,
    )
    print(f"SUMMARY {summary_path}", flush=True)
    print(f"PAIRWISE {pairwise_path}", flush=True)


if __name__ == "__main__":
    main()
