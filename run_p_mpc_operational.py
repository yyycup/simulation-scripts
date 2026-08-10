"""Run the explicitly enabled Physics-P MPC operational profile.

The profile deliberately uses locally validated short horizons (fourteen steps
for peak shaving and twelve for frequency regulation).  Peak Physics-P removes
the tiny linear compressor move cost that caused stick-release cycling.  DMAX
and energy weights remain inherited from the formal scene parameters. Candidate
B remains the repository-wide default and results are written separately.
"""

from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import thermal_system

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from mpc_predictor_selection import PHYSICS_P
from mpc_flow_direction_strategies import runtime_mpc_params_for_scene
from run_p_mpc_short_comparison import (
    SCENES,
    source_csv_for_scene,
    summarize_run,
)
from thermal_case_simulator import simulate_case


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "p_mpc_operational_v1"
DEFAULT_OPERATIONAL_P_ARTIFACT = (
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "thermal_bias_correction_v3"
    / "physics_p_heat_generation_corrected.json"
)
OPERATIONAL_HORIZON_STEPS_BY_SCENE = {
    "peak": 14,
    "freq": 12,
}


def displacement_scale_from_artifact(artifact_path):
    artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    scale = float(
        artifact.get("thermal", {}).get("compressor_displacement_scale", 1.0)
    )
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("compressor_displacement_scale must be positive and finite")
    return scale


@contextmanager
def patched_plant_compressor_displacement(scale):
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("compressor displacement scale must be positive and finite")
    original = float(thermal_system.V_disp_m3_per_rev)
    if scale == 1.0:
        yield
        return
    try:
        thermal_system.V_disp_m3_per_rev = original * scale
        thermal_system.clear_refrigeration_cycle_cache()
        yield
    finally:
        thermal_system.V_disp_m3_per_rev = original
        thermal_system.clear_refrigeration_cycle_cache()


def _simulate_case_with_displacement(displacement_scale, *args, **kwargs):
    with patched_plant_compressor_displacement(displacement_scale):
        return simulate_case(*args, **kwargs)


def operational_profile_for_scene(
    scene_key,
    *,
    w_dcomp=None,
    w_dcomp_quadratic=None,
    comp_command_filter_alpha=None,
    w_terminal_temp=None,
    physics_p_temp_bias_gain=None,
    horizon_steps_override=None,
):
    if scene_key not in OPERATIONAL_HORIZON_STEPS_BY_SCENE:
        raise ValueError(f"unsupported scene for Physics-P profile: {scene_key}")
    scene_params = runtime_mpc_params_for_scene(scene_key)
    resolved_horizon_steps = OPERATIONAL_HORIZON_STEPS_BY_SCENE[scene_key]
    if horizon_steps_override is not None:
        resolved_horizon_steps = int(horizon_steps_override)
        if resolved_horizon_steps < 2:
            raise ValueError("horizon_steps_override must be at least 2")
    resolved_w_dcomp = (
        0.0 if scene_key == "peak" else float(scene_params.w_dcomp)
    )
    resolved_physics_p_temp_bias_gain = 2.0 if scene_key == "peak" else 0.0
    mpc_overrides = (
        {
            "w_dcomp": resolved_w_dcomp,
            "physics_p_temp_bias_gain": resolved_physics_p_temp_bias_gain,
        }
        if scene_key == "peak"
        else None
    )
    resolved_w_dcomp_quadratic = float(scene_params.w_dcomp_quadratic)
    resolved_comp_command_filter_alpha = float(
        scene_params.comp_command_filter_alpha
    )
    resolved_w_terminal_temp = float(scene_params.w_terminal_temp)
    if w_dcomp is not None:
        resolved_w_dcomp = float(w_dcomp)
        if not np.isfinite(resolved_w_dcomp) or resolved_w_dcomp < 0.0:
            raise ValueError("w_dcomp must be finite and nonnegative")
        if mpc_overrides is None:
            mpc_overrides = {}
        mpc_overrides["w_dcomp"] = resolved_w_dcomp
    if w_dcomp_quadratic is not None:
        resolved_w_dcomp_quadratic = float(w_dcomp_quadratic)
        if (
            not np.isfinite(resolved_w_dcomp_quadratic)
            or resolved_w_dcomp_quadratic < 0.0
        ):
            raise ValueError(
                "w_dcomp_quadratic must be finite and nonnegative"
            )
        if mpc_overrides is None:
            mpc_overrides = {}
        mpc_overrides["w_dcomp_quadratic"] = (
            resolved_w_dcomp_quadratic
        )
    if comp_command_filter_alpha is not None:
        resolved_comp_command_filter_alpha = float(
            comp_command_filter_alpha
        )
        if (
            not np.isfinite(resolved_comp_command_filter_alpha)
            or resolved_comp_command_filter_alpha <= 0.0
            or resolved_comp_command_filter_alpha > 1.0
        ):
            raise ValueError(
                "comp_command_filter_alpha must be finite and in (0, 1]"
            )
        if mpc_overrides is None:
            mpc_overrides = {}
        mpc_overrides["comp_command_filter_alpha"] = (
            resolved_comp_command_filter_alpha
        )
    if w_terminal_temp is not None:
        resolved_w_terminal_temp = float(w_terminal_temp)
        if (
            not np.isfinite(resolved_w_terminal_temp)
            or resolved_w_terminal_temp < 0.0
        ):
            raise ValueError(
                "w_terminal_temp must be finite and nonnegative"
            )
        if mpc_overrides is None:
            mpc_overrides = {}
        mpc_overrides["w_terminal_temp"] = resolved_w_terminal_temp
    if physics_p_temp_bias_gain is not None:
        resolved_physics_p_temp_bias_gain = float(
            physics_p_temp_bias_gain
        )
        if (
            not np.isfinite(resolved_physics_p_temp_bias_gain)
            or resolved_physics_p_temp_bias_gain < 0.0
        ):
            raise ValueError(
                "physics_p_temp_bias_gain must be finite and nonnegative"
            )
        if mpc_overrides is None:
            mpc_overrides = {}
        mpc_overrides["physics_p_temp_bias_gain"] = (
            resolved_physics_p_temp_bias_gain
        )
    return {
        "horizon_steps": resolved_horizon_steps,
        "mpc_overrides": mpc_overrides,
        "dmax_comp_rpm_per_step": float(scene_params.dmax_comp),
        "w_energy_comp": float(scene_params.w_energy_comp),
        "w_dcomp": resolved_w_dcomp,
        "w_dcomp_quadratic": resolved_w_dcomp_quadratic,
        "comp_command_filter_alpha": (
            resolved_comp_command_filter_alpha
        ),
        "w_terminal_temp": resolved_w_terminal_temp,
        "physics_p_temp_bias_gain": resolved_physics_p_temp_bias_gain,
    }


def _column(frame, *names):
    for name in names:
        if name in frame.columns:
            return frame[name]
    raise KeyError(f"Missing required columns: {names}")


def _as_bool(series):
    return series.fillna(False).astype(str).str.lower().isin(("true", "1"))


def summarize_operational_run(frame, *, scene_key, profile, artifact_path, out_csv):
    row = summarize_run(
        frame,
        scene_key=scene_key,
        predictor=PHYSICS_P,
        horizon=profile["horizon_steps"],
        artifact_path=artifact_path,
        out_csv=out_csv,
    )
    domain_valid = _as_bool(
        _column(
            frame,
            "MPC prediction domain valid",
            "MPC_Prediction_Domain_Valid",
        )
    )
    domain_violation = _column(
        frame,
        "MPC coolant prediction domain violation",
        "MPC_T_Cool_Domain_Violation_C",
    ).astype(float)
    finite_violation = domain_violation[np.isfinite(domain_violation)]
    row.update(
        {
            "profile": "physics_p_operational_v1",
            "dmax_comp_rpm_per_step": profile["dmax_comp_rpm_per_step"],
            "w_energy_comp": profile["w_energy_comp"],
            "w_dcomp": profile["w_dcomp"],
            "w_dcomp_quadratic": profile["w_dcomp_quadratic"],
            "comp_command_filter_alpha": profile[
                "comp_command_filter_alpha"
            ],
            "w_terminal_temp": profile["w_terminal_temp"],
            "physics_p_temp_bias_gain": profile[
                "physics_p_temp_bias_gain"
            ],
            "prediction_domain_valid_rate": float(np.mean(domain_valid)),
            "prediction_domain_violation_max_c": (
                float(np.max(finite_violation))
                if len(finite_violation)
                else np.nan
            ),
        }
    )
    row["qualified"] = bool(
        row["solve_success_rate"] == 1.0
        and row["prediction_domain_valid_rate"] == 1.0
    )
    return row


def run_operational(
    *,
    scenes,
    steps,
    artifact_path=DEFAULT_OPERATIONAL_P_ARTIFACT,
    output_root=DEFAULT_OUTPUT_ROOT,
    w_dcomp=None,
    w_dcomp_quadratic=None,
    comp_command_filter_alpha=None,
    w_terminal_temp=None,
    physics_p_temp_bias_gain=None,
    horizon_steps=None,
    force=False,
):
    artifact_path = Path(artifact_path)
    output_root = Path(output_root)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Physics-P artifact not found: {artifact_path}")
    displacement_scale = displacement_scale_from_artifact(artifact_path)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for scene_key in scenes:
        profile = operational_profile_for_scene(
            scene_key,
            w_dcomp=w_dcomp,
            w_dcomp_quadratic=w_dcomp_quadratic,
            comp_command_filter_alpha=comp_command_filter_alpha,
            w_terminal_temp=w_terminal_temp,
            physics_p_temp_bias_gain=physics_p_temp_bias_gain,
            horizon_steps_override=horizon_steps,
        )
        stem = (
            f"{scene_key}_{PHYSICS_P}_operational_steps{steps}_"
            f"horizon{profile['horizon_steps']}"
        )
        print(
            f"RUN scene={scene_key} predictor={PHYSICS_P} steps={steps} "
            f"horizon={profile['horizon_steps']} "
            f"displacement_scale={displacement_scale:g} "
            f"dmax={profile['dmax_comp_rpm_per_step']:g}rpm/step "
            f"w_energy_comp={profile['w_energy_comp']:g} "
            f"w_dcomp={profile['w_dcomp']:g} "
            f"w_dcomp_quadratic={profile['w_dcomp_quadratic']:g} "
            f"comp_filter_alpha={profile['comp_command_filter_alpha']:g} "
            f"w_terminal_temp={profile['w_terminal_temp']:g} "
            f"temp_bias_gain={profile['physics_p_temp_bias_gain']:g}",
            flush=True,
        )
        result = _simulate_case_with_displacement(
            displacement_scale,
            "mpc",
            SCENES[scene_key],
            "单向",
            source_csv_for_scene(SCENES[scene_key]),
            f"{stem}.csv",
            f"{stem}_snapshot.csv",
            output_root=output_root,
            force=force,
            max_steps=steps,
            target_temp_c=25.0,
            mpc_flow_mode="standard",
            result_tag=stem,
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact_path,
            mpc_horizon_override=profile["horizon_steps"],
            physics_p_mpc_overrides=profile["mpc_overrides"],
            log_func=lambda message: print(message, flush=True),
        )
        out_csv = Path(result["out_csv"])
        frame = pd.read_csv(out_csv, encoding="utf-8-sig")
        row = summarize_operational_run(
            frame,
            scene_key=scene_key,
            profile=profile,
            artifact_path=artifact_path,
            out_csv=out_csv,
        )
        row["compressor_displacement_scale"] = displacement_scale
        row["compressor_displacement_cm3_per_rev"] = (
            5.525 * displacement_scale
        )
        rows.append(row)
        print(
            f"RESULT scene={scene_key} solved={row['solve_success_rate']:.1%} "
            f"domain_valid={row['prediction_domain_valid_rate']:.1%} "
            f"mean_comp={row['mean_n_comp_rpm']:.1f}rpm "
            f"qualified={row['qualified']}",
            flush=True,
        )

    summary = pd.DataFrame(rows)
    summary_path = output_root / "operational_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    return summary_path, summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the guarded Physics-P MPC operational profile"
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENES),
        default=list(SCENES),
    )
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_OPERATIONAL_P_ARTIFACT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--w-dcomp",
        type=float,
        default=None,
        help="Physics-P-only compressor move cost; omitted keeps the frozen runtime value",
    )
    parser.add_argument(
        "--w-dcomp-quadratic",
        type=float,
        default=None,
        help=(
            "Physics-P-only first-move quadratic compressor cost; it is "
            "added to the configured linear DCOST"
        ),
    )
    parser.add_argument(
        "--comp-command-filter-alpha",
        type=float,
        default=None,
        help=(
            "Physics-P-only executed compressor command filter in (0, 1]; "
            "1 keeps the raw MPC command"
        ),
    )
    parser.add_argument(
        "--w-terminal-temp",
        type=float,
        default=None,
        help=(
            "Physics-P-only terminal battery-temperature weight; omitted "
            "keeps the scene runtime value"
        ),
    )
    parser.add_argument(
        "--temp-bias-gain",
        type=float,
        default=None,
        help=(
            "Physics-P-only online temperature prediction-bias gain; "
            "omitted keeps the operational profile value"
        ),
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=None,
        help=(
            "Physics-P local diagnostic horizon override; omitted keeps "
            "the validated scene profile"
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps < 2:
        raise SystemExit("--steps must be at least 2")
    summary_path, summary = run_operational(
        scenes=args.scenes,
        steps=args.steps,
        artifact_path=args.artifact,
        output_root=args.output_root,
        w_dcomp=args.w_dcomp,
        w_dcomp_quadratic=args.w_dcomp_quadratic,
        comp_command_filter_alpha=args.comp_command_filter_alpha,
        w_terminal_temp=args.w_terminal_temp,
        physics_p_temp_bias_gain=args.temp_bias_gain,
        horizon_steps=args.horizon_steps,
        force=args.force,
    )
    print(f"SUMMARY {summary_path}", flush=True)
    if not bool(summary["qualified"].all()):
        failed = ", ".join(summary.loc[~summary["qualified"], "scene"])
        raise SystemExit(f"UNQUALIFIED Physics-P operational run(s): {failed}")


if __name__ == "__main__":
    main()
