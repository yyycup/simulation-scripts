"""Sequential, Physics-P-only MPC weight tuning.

This experiment is intentionally separate from the fair Candidate-B versus P
predictor replacement comparison.  It never changes the formal MPC defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from mpc_flow_direction_strategies import runtime_mpc_params_for_scene
from mpc_physics_predictor import load_physics_artifact
from mpc_predictor_selection import PHYSICS_P
from p_mpc_run_support import (
    PROJECT_ROOT,
    SCENES,
    source_csv_for_scene,
    summarize_run,
)
from .run_p_mpc_short_comparison import DEFAULT_P_ARTIFACT
from thermal_case_simulator import simulate_case


DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "p_mpc_controller_tuning_v1"
DEFAULT_HORIZONS = {"peak": 60, "freq": 45}


def _factor_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _candidate_overrides(
    *,
    stage: str,
    base_params,
    factor: float,
    selected_comp_weight: float | None,
) -> dict[str, float]:
    if stage == "comp":
        return {
            "w_energy_comp": float(base_params.w_energy_comp) * float(factor),
            "w_terminal_temp": float(base_params.w_terminal_temp),
        }
    if stage == "terminal":
        if selected_comp_weight is None:
            raise ValueError("terminal stage requires a selected compressor weight")
        return {
            "w_energy_comp": float(selected_comp_weight),
            "w_terminal_temp": float(base_params.w_terminal_temp) * float(factor),
        }
    if stage == "dmax":
        if selected_comp_weight is None:
            raise ValueError("dmax stage requires a selected compressor weight")
        return {
            "w_energy_comp": float(selected_comp_weight),
            "w_terminal_temp": float(base_params.w_terminal_temp),
            "dmax_comp": float(factor),
        }
    raise ValueError(f"unsupported tuning stage: {stage!r}")


def _scan_metrics(
    frame: pd.DataFrame,
    *,
    target_temp_c: float,
    band_half_width_c: float,
    coolant_domain_min_c: float,
):
    comp = pd.to_numeric(frame["Compressor command"], errors="raise").to_numpy(dtype=float)
    terminal = pd.to_numeric(
        frame["MPC predicted battery temperature horizon end"],
        errors="raise",
    ).to_numpy(dtype=float)
    coolant_terminal = pd.to_numeric(
        frame["MPC predicted coolant temperature horizon end"],
        errors="raise",
    ).to_numpy(dtype=float)
    upper = float(target_temp_c) + float(band_half_width_c)
    return {
        "compressor_saturation_rate": float(np.mean(comp >= 5990.0)),
        "compressor_minimum_rate": float(np.mean(comp <= 1010.0)),
        "pred_terminal_mean_c": float(np.mean(terminal)),
        "pred_terminal_max_c": float(np.max(terminal)),
        "pred_terminal_upper_violation_max_c": float(
            np.max(np.maximum(terminal - upper, 0.0))
        ),
        "pred_coolant_terminal_min_c": float(np.min(coolant_terminal)),
        "pred_coolant_domain_under_max_c": float(
            np.max(np.maximum(float(coolant_domain_min_c) - coolant_terminal, 0.0))
        ),
    }


def _select_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    selections = []
    for scene_key, group in summary.groupby("scene", sort=False):
        group = group.copy()
        baseline = group.loc[group["is_baseline"]]
        if baseline.empty:
            raise ValueError(f"{scene_key} scan must include its formal baseline")
        baseline = baseline.iloc[0]
        eligible = group.loc[
            (group["solve_success_rate"] >= 1.0)
            & (group["pred_coolant_domain_under_max_c"] <= 1e-6)
            & (
                group["pred_terminal_mean_c"]
                <= float(baseline["pred_terminal_mean_c"]) + 0.10
            )
            & (
                group["temperature_mae_c"]
                <= float(baseline["temperature_mae_c"]) + 0.01
            )
        ].copy()
        selection_qualified = not eligible.empty
        if eligible.empty:
            eligible = group.loc[group["solve_success_rate"] >= 1.0].copy()
        if eligible.empty:
            eligible = group.copy()
        selected = eligible.sort_values(
            [
                "compressor_saturation_rate",
                "energy_kwh",
                "pred_terminal_mean_c",
                "solve_time_mean_s",
            ],
            ascending=[True, True, True, True],
        ).iloc[0]
        row = selected.to_dict()
        row["selection_qualified"] = bool(selection_qualified)
        if selection_qualified:
            row["selection_rule"] = (
                "success=100%; coolant prediction stays in validated domain; "
                "terminal_mean<=baseline+0.10C; "
                "temperature_mae<=baseline+0.01C; then minimum saturation and energy"
            )
        else:
            row["selection_rule"] = (
                "diagnostic fallback only; no candidate satisfied all qualification criteria"
            )
        selections.append(row)
    return pd.DataFrame(selections)


def run_scan(
    *,
    stage: str,
    scenes: tuple[str, ...],
    factors: tuple[float, ...],
    steps: int,
    horizons: dict[str, int],
    artifact_path: Path,
    output_root: Path,
    selected_comp_weights: dict[str, float | None],
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Physics-P artifact not found: {artifact_path}")
    artifact = load_physics_artifact(artifact_path, require_validated=True)
    coolant_domain_min_c = float(artifact["input_domain"]["t_cool_c"][0])
    if not factors or any(not np.isfinite(value) or value <= 0.0 for value in factors):
        raise ValueError("scan factors must be finite and positive")
    if stage != "dmax" and not any(np.isclose(value, 1.0) for value in factors):
        raise ValueError("scan factors must include the 1.0 baseline")

    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene_key in scenes:
        scene = SCENES[scene_key]
        horizon = int(horizons[scene_key])
        base_params = runtime_mpc_params_for_scene(scene_key)
        for factor in factors:
            overrides = _candidate_overrides(
                stage=stage,
                base_params=base_params,
                factor=factor,
                selected_comp_weight=selected_comp_weights.get(scene_key),
            )
            stem = (
                f"{scene_key}_physics_p_{stage}_factor{_factor_tag(factor)}"
                f"_steps{steps}_horizon{horizon}"
            )
            print(
                f"RUN scene={scene_key} stage={stage} factor={factor:g} "
                f"w_comp={overrides['w_energy_comp']:g} "
                f"w_terminal={overrides['w_terminal_temp']:g} "
                f"dmax_comp={overrides.get('dmax_comp', base_params.dmax_comp):g}",
                flush=True,
            )
            result = simulate_case(
                "mpc",
                scene,
                "单向",
                source_csv_for_scene(scene),
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
                mpc_horizon_override=horizon,
                physics_p_mpc_overrides=overrides,
                log_func=lambda message: print(message, flush=True),
            )
            out_csv = Path(result["out_csv"])
            frame = pd.read_csv(out_csv, encoding="utf-8-sig")
            row = summarize_run(
                frame,
                scene_key=scene_key,
                predictor=PHYSICS_P,
                horizon=horizon,
                artifact_path=artifact_path,
                out_csv=out_csv,
            )
            row.update(
                {
                    "stage": stage,
                    "factor": float(factor),
                    "w_energy_comp": overrides["w_energy_comp"],
                    "w_terminal_temp": overrides["w_terminal_temp"],
                    "dmax_comp": overrides.get("dmax_comp", base_params.dmax_comp),
                    "is_baseline": bool(
                        np.isclose(
                            overrides.get("dmax_comp", base_params.dmax_comp),
                            base_params.dmax_comp,
                        )
                        if stage == "dmax"
                        else np.isclose(factor, 1.0)
                    ),
                    **_scan_metrics(
                        frame,
                        target_temp_c=25.0,
                        band_half_width_c=base_params.cv_band_half_width,
                        coolant_domain_min_c=coolant_domain_min_c,
                    ),
                }
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                output_root / f"{stage}_scan_summary.partial.csv",
                index=False,
                encoding="utf-8-sig",
            )

    summary = pd.DataFrame(rows)
    selected = _select_candidates(summary)
    summary.to_csv(
        output_root / f"{stage}_scan_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selected.to_csv(
        output_root / f"{stage}_selected_by_scene.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary, selected


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("comp", "terminal", "dmax"), required=True)
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=tuple(SCENES),
        default=("peak", "freq"),
    )
    parser.add_argument("--factors", nargs="+", type=float, required=True)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--peak-horizon", type=int, default=60)
    parser.add_argument("--freq-horizon", type=int, default=45)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_P_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--selected-comp-peak", type=float)
    parser.add_argument("--selected-comp-freq", type=float)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    summary, selected = run_scan(
        stage=args.stage,
        scenes=tuple(args.scenes),
        factors=tuple(args.factors),
        steps=args.steps,
        horizons={"peak": args.peak_horizon, "freq": args.freq_horizon},
        artifact_path=args.artifact,
        output_root=args.output_root,
        selected_comp_weights={
            "peak": args.selected_comp_peak,
            "freq": args.selected_comp_freq,
        },
        force=args.force,
    )
    print("\nSCAN SUMMARY", flush=True)
    print(
        summary[
            [
                "scene",
                "factor",
                "w_energy_comp",
                "w_terminal_temp",
                "dmax_comp",
                "temperature_mae_c",
                "energy_kwh",
                "mean_n_comp_rpm",
                "compressor_saturation_rate",
                "pred_terminal_mean_c",
                "pred_coolant_terminal_min_c",
                "pred_coolant_domain_under_max_c",
                "solve_success_rate",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nSELECTED", flush=True)
    print(
        selected[
            [
                "scene",
                "factor",
                "w_energy_comp",
                "w_terminal_temp",
                "dmax_comp",
                "mean_n_comp_rpm",
                "compressor_saturation_rate",
                "pred_terminal_mean_c",
                "pred_coolant_terminal_min_c",
                "pred_coolant_domain_under_max_c",
                "selection_qualified",
            ]
        ].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
