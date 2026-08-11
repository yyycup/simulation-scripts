"""Run one explicit MPC predictor locally with the formal scene horizons.

Peak shaving is run first with a 60-step prediction horizon, followed by
frequency regulation with a 45-step horizon.  Formal scene DMAX values and
weights are inherited because no predictor-specific controller overrides are
supplied. By default each candidate retains its native numerical settings.
The explicit strict-ablation option instead aligns the external optimization
contract so only the internal predictor package differs.
"""

from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from mpc_predictor_selection import (
    CANDIDATE_B,
    PHYSICS_P,
    normalize_predictor_name,
)
from mpc_flow_direction_strategies import (
    PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S,
    runtime_mpc_params_for_scene,
)
from thermal_batch_config import (
    INITIAL_TEMP_C,
    AGC_DATA_FILE,
    MPC_N_COMP_MIN_RPM,
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    SIM_DT,
)
from p_mpc_run_support import PROJECT_ROOT
from .run_p_mpc_short_comparison import (
    DEFAULT_P_ARTIFACT,
    run_comparison,
)


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "p_mpc_local_formal_v1"
FORMAL_LOCAL_CASES = (
    {"scene": "peak", "horizon": 60, "steps": 1280},
    {"scene": "freq", "horizon": 45, "steps": 720},
)
FORMAL_PROGRESS_INTERVAL_STEPS = 10
EXPERIMENT_SCOPE = "native_controller_candidate"
FORMAL_SOURCE_FILES = (
    "experiments/physics_p/tuning/run_p_mpc_local_formal.py",
    "experiments/physics_p/tuning/run_p_mpc_short_comparison.py",
    "p_mpc_run_support.py",
    "mpc_flow_direction_strategies.py",
    "mpc_physics_predictor.py",
    "thermal_case_simulator.py",
    "thermal_loop.py",
    "thermal_batch_config.py",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False, "sha256": None}
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
    }


def _git_start_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def build_run_provenance(
    *, predictor, artifact_path, strict_predictor_ablation=False
):
    artifact_record = (
        _file_record(artifact_path) if predictor == PHYSICS_P else None
    )
    source_records = {
        relative_path: _file_record(PROJECT_ROOT / relative_path)
        for relative_path in FORMAL_SOURCE_FILES
    }
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "python_executable": str(Path(sys.executable).resolve()),
        "predictor": predictor,
        "strict_predictor_ablation": bool(strict_predictor_ablation),
        "physics_p_artifact": artifact_record,
        "git_start": _git_start_state(),
        "source_files": source_records,
        "input_profiles": {
            "peak": {
                "definition": "constant_total_current_a",
                "total_current_a": 560.0,
            },
            "freq": {
                "definition": "AGC_RegD_times_1120A_interpolated",
                "agc_file": _file_record(AGC_DATA_FILE),
            },
        },
        "initialization_and_solver_semantics": {
            "all_thermal_initial_states_c": float(INITIAL_TEMP_C),
            "physics_p_first_cycle": (
                "held-input physical-state feasible trajectory; cv_temp uses only "
                "the scalar measurement; FSTATUS restored to measured states"
            ),
            "physics_p_each_cycle_time_shift": 0,
            "physics_p_recovery_order": (
                "IPOPT retry -> APOPT -> fixed-control IPOPT -> optimized IPOPT"
            ),
            "physics_p_cycle_wall_clock_budget_s": float(
                PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S
            ),
            "candidate_b_solver_behavior": (
                "common strict-ablation recovery contract"
                if strict_predictor_ablation
                else "unchanged legacy recovery path"
            ),
            "strict_ablation_recovery_order": (
                "IPOPT retry -> APOPT -> fixed-control IPOPT -> optimized IPOPT"
                if strict_predictor_ablation
                else None
            ),
            "strict_ablation_external_contract": (
                "1000-6000 rpm compressor command domain; 1600-4800 rpm "
                "pump command domain; peak MV_STEP_HOR=1 and frequency "
                "MV_STEP_HOR=3; RTOL=OTOL=1e-6; "
                "common retry and fallback policy; model-native state "
                "roll-forward (B TIME_SHIFT=1, P measured-state reanchor "
                "with TIME_SHIFT=0)"
                if strict_predictor_ablation
                else None
            ),
            "physics_p_domain_acceptance": (
                "reject solved command unless full coolant trajectory is within 15-35 C"
            ),
        },
    }


def formal_local_cases(*, peak_steps=1280, freq_steps=720):
    return (
        {"scene": "peak", "horizon": 60, "steps": int(peak_steps)},
        {"scene": "freq", "horizon": 45, "steps": int(freq_steps)},
    )


def parse_progress_message(message):
    match = re.search(r"\s(\d+)/(\d+)\s+t=", str(message))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))

def _best_effort_print(message):
    try:
        print(message, flush=True)
    except (OSError, ValueError):
        return False
    return True


def formal_scene_control_profiles(
    *,
    peak_steps=1280,
    freq_steps=720,
    strict_predictor_ablation=False,
):
    profiles = {}
    for case in formal_local_cases(
        peak_steps=peak_steps,
        freq_steps=freq_steps,
    ):
        scene = case["scene"]
        params = runtime_mpc_params_for_scene(scene)
        profiles[scene] = {
            "simulation_steps": int(case["steps"]),
            "prediction_horizon_steps": int(case["horizon"]),
            "controller_forecast_profile_steps": int(
                max(case["steps"], case["horizon"] + 1)
            ),
            "control_interval_s": float(SIM_DT),
            "initial_thermal_state_c": float(INITIAL_TEMP_C),
            "target_temperature_c": float(params.dynamic_target_min),
            "cv_band_half_width_c": float(params.cv_band_half_width),
            "compressor_dmax_rpm_per_step": float(params.dmax_comp),
            "pump_dmax_rpm_per_step": float(params.dmax_pump),
            "w_high_temp": float(params.w_high_temp),
            "w_cold_temp": float(params.w_cold_temp),
            "w_energy_comp": float(params.w_energy_comp),
            "w_energy_pump": float(params.w_energy_pump),
            "w_dcomp": float(params.w_dcomp),
            "w_dpump": float(params.w_dpump),
            "terminal_cost_enabled": bool(params.terminal_cost_enabled),
            "terminal_cost_type": str(params.terminal_cost_type),
            "w_terminal_temp": float(params.w_terminal_temp),
            "terminal_temp_scale_c": float(params.terminal_temp_scale_c),
            "mpc_flow_mode": "standard",
            "predictor_specific_mpc_overrides": False,
            "strict_predictor_ablation": bool(strict_predictor_ablation),
        }
    return profiles


def _resolved_compressor_command_bounds(
    predictor, *, strict_predictor_ablation=False
):
    if predictor is None:
        return None, None
    normalized = normalize_predictor_name(predictor)
    lower_rpm = MPC_N_COMP_MIN_RPM
    if normalized == PHYSICS_P and not strict_predictor_ablation:
        lower_rpm = N_COMP_OFF_RPM
    return float(lower_rpm), float(N_COMP_MAX_RPM)


def _write_status(
    path,
    *,
    status,
    predictor=None,
    formal_scene_profiles=None,
    run_provenance=None,
    active_scene=None,
    formal_summary_csv=None,
    active_horizon_steps=None,
    active_steps_completed=None,
    active_steps_total=None,
    completed_scenes=(),
    last_progress="",
    error="",
    strict_predictor_ablation=False,
):
    command_lower_rpm, command_upper_rpm = (
        _resolved_compressor_command_bounds(
            predictor,
            strict_predictor_ablation=strict_predictor_ablation,
        )
    )
    payload = {
        "status": status,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "predictor": predictor,
        "experiment_scope": (
            "strict_predictor_ablation"
            if strict_predictor_ablation
            else EXPERIMENT_SCOPE
        ),
        "strict_predictor_ablation": bool(strict_predictor_ablation),
        "resolved_compressor_command_lower_rpm": command_lower_rpm,
        "resolved_compressor_command_upper_rpm": command_upper_rpm,
        "formal_scene_profiles": formal_scene_profiles,
        "run_provenance": run_provenance,
        "active_scene": active_scene,
        "formal_summary_csv": formal_summary_csv,
        "active_horizon_steps": active_horizon_steps,
        "active_steps_completed": active_steps_completed,
        "active_steps_total": active_steps_total,
        "completed_scenes": list(completed_scenes),
        "last_progress": last_progress,
        "error": error,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_local_formal(
    *,
    output_root=DEFAULT_OUTPUT_ROOT,
    artifact_path=DEFAULT_P_ARTIFACT,
    predictor=None,
    peak_steps=1280,
    freq_steps=720,
    force=False,
    progress_interval_steps=FORMAL_PROGRESS_INTERVAL_STEPS,
    strict_predictor_ablation=False,
):
    output_root = Path(output_root)
    artifact_path = Path(artifact_path)
    if predictor is None or not str(predictor).strip():
        raise ValueError(
            "formal local runner requires an explicit predictor selection"
        )
    predictor = normalize_predictor_name(predictor)
    if predictor not in {CANDIDATE_B, PHYSICS_P}:
        raise ValueError(
            f"formal local runner does not support predictor: {predictor}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    progress_interval_steps = int(progress_interval_steps)
    if progress_interval_steps < 1:
        raise ValueError("progress_interval_steps must be at least 1")
    control_profiles = formal_scene_control_profiles(
        peak_steps=peak_steps,
        freq_steps=freq_steps,
        strict_predictor_ablation=strict_predictor_ablation,
    )
    if predictor == PHYSICS_P and not artifact_path.is_file():
        raise FileNotFoundError(f"Physics-P artifact not found: {artifact_path}")
    run_provenance = build_run_provenance(
        predictor=predictor,
        artifact_path=artifact_path,
        strict_predictor_ablation=strict_predictor_ablation,
    )
    status_path = output_root / "local_formal_status.json"
    formal_summary_path = output_root / "formal_run_summary.csv"
    formal_summary_csv = str(formal_summary_path.resolve())
    scene_summary_frames = []
    completed = []
    _write_status(
        status_path,
        status="RUNNING",
        predictor=predictor,
        formal_scene_profiles=control_profiles,
        run_provenance=run_provenance,
        formal_summary_csv=formal_summary_csv,
        completed_scenes=completed,
        strict_predictor_ablation=strict_predictor_ablation,
    )
    active_case = None
    last_progress = ""
    try:
        for case in formal_local_cases(
            peak_steps=peak_steps,
            freq_steps=freq_steps,
        ):
            scene = case["scene"]
            active_case = case
            last_progress = ""
            _write_status(
                status_path,
                status="RUNNING",
                predictor=predictor,
                formal_scene_profiles=control_profiles,
                run_provenance=run_provenance,
                formal_summary_csv=formal_summary_csv,
                active_scene=scene,
                active_horizon_steps=case["horizon"],
                active_steps_completed=0,
                active_steps_total=case["steps"],
                completed_scenes=completed,
                strict_predictor_ablation=strict_predictor_ablation,
            )

            def scene_log(message):
                nonlocal last_progress
                progress = parse_progress_message(message)
                if progress is not None:
                    steps_completed, steps_total = progress
                    last_progress = str(message)
                    _write_status(
                        status_path,
                        status="RUNNING",
                        predictor=predictor,
                        formal_scene_profiles=control_profiles,
                        run_provenance=run_provenance,
                        formal_summary_csv=formal_summary_csv,
                        active_scene=scene,
                        active_horizon_steps=case["horizon"],
                        active_steps_completed=steps_completed,
                        active_steps_total=steps_total,
                        completed_scenes=completed,
                        last_progress=last_progress,
                        strict_predictor_ablation=strict_predictor_ablation,
                    )
                _best_effort_print(message)
            _best_effort_print(
                f"FORMAL_LOCAL_START predictor={predictor} scene={scene} "
                f"horizon={case['horizon']} "
                f"steps={case['steps']}"
            )
            comparison_outputs = run_comparison(
                scenes=(scene,),
                predictors=(predictor,),
                steps=case["steps"],
                horizon=case["horizon"],
                artifact_path=artifact_path,
                output_root=output_root / scene,
                force=force,
                progress_interval_steps=progress_interval_steps,
                forecast_profile_steps=max(
                    case["steps"], case["horizon"] + 1
                ),
                log_func=scene_log,
                strict_predictor_ablation=strict_predictor_ablation,
            )
            completed.append(scene)
            if comparison_outputs:
                scene_summary_path = Path(comparison_outputs[0])
                if scene_summary_path.is_file():
                    scene_summary_frames.append(
                        pd.read_csv(scene_summary_path, encoding="utf-8-sig")
                    )
                    pd.concat(scene_summary_frames, ignore_index=True).to_csv(
                        formal_summary_path, index=False, encoding="utf-8-sig"
                    )
            _best_effort_print(f"FORMAL_LOCAL_DONE scene={scene}")
        _write_status(
            status_path,
            status="COMPLETE",
            predictor=predictor,
            formal_scene_profiles=control_profiles,
            run_provenance=run_provenance,
            completed_scenes=completed,
            formal_summary_csv=formal_summary_csv,
            strict_predictor_ablation=strict_predictor_ablation,
        )
    except Exception as exc:
        failed_progress = parse_progress_message(last_progress) or (None, None)
        _write_status(
            status_path,
            status="FAILED",
            predictor=predictor,
            formal_scene_profiles=control_profiles,
            run_provenance=run_provenance,
            formal_summary_csv=formal_summary_csv,
            active_scene=(
                active_case["scene"] if active_case is not None else None
            ),
            active_horizon_steps=(
                active_case["horizon"] if active_case is not None else None
            ),
            active_steps_completed=failed_progress[0],
            active_steps_total=(
                active_case["steps"] if active_case is not None else None
            ),
            completed_scenes=completed,
            last_progress=last_progress,
            error=f"{type(exc).__name__}: {exc}",
            strict_predictor_ablation=strict_predictor_ablation,
        )
        raise
    return status_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_P_ARTIFACT)
    parser.add_argument(
        "--predictor",
        choices=(PHYSICS_P, CANDIDATE_B),
        required=True,
        help="explicit internal MPC predictor for both formal scenes",
    )
    parser.add_argument("--peak-steps", type=int, default=1280)
    parser.add_argument("--freq-steps", type=int, default=720)
    parser.add_argument(
        "--progress-interval-steps",
        type=int,
        default=FORMAL_PROGRESS_INTERVAL_STEPS,
    )

    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--strict-predictor-ablation",
        action="store_true",
        help=(
            "align external MPC and solver settings between Candidate B and P"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.peak_steps < 2 or args.freq_steps < 2:
        raise SystemExit("--peak-steps and --freq-steps must both be at least 2")
    if args.progress_interval_steps < 1:
        raise SystemExit("--progress-interval-steps must be at least 1")
    status_path = run_local_formal(
        output_root=args.output_root,
        artifact_path=args.artifact,
        predictor=args.predictor,
        peak_steps=args.peak_steps,
        freq_steps=args.freq_steps,
        force=args.force,
        progress_interval_steps=args.progress_interval_steps,
        strict_predictor_ablation=args.strict_predictor_ablation,
    )
    print(f"FORMAL_LOCAL_STATUS {status_path}", flush=True)


if __name__ == "__main__":
    main()
