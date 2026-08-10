"""Build and validate a runtime-neutral battery-heat correction for predictor P."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import pandas as pd

from evaluate_p_shadow_actual_replay import evaluate_actual_command_replay
from mpc_physics_predictor import load_physics_artifact, validate_physics_artifact


DEFAULT_SCALE = 0.74
DEFAULT_HORIZONS_S = (5.0, 25.0, 50.0, 100.0, 225.0, 300.0)


def build_corrected_artifact(
    base_artifact: dict,
    *,
    scale: float,
    coolant_mass_flow_ref_kg_s: float | None = None,
    calibration_source: str,
) -> dict:
    """Return a copy whose only model change is the battery heat-input scale."""
    validate_physics_artifact(base_artifact)
    selected_scale = float(scale)
    if not math.isfinite(selected_scale) or selected_scale <= 0.0:
        raise ValueError("scale must be a positive finite number")
    if not str(calibration_source).strip():
        raise ValueError("calibration_source must not be empty")

    corrected = copy.deepcopy(base_artifact)
    corrected["thermal"]["battery_heat_generation_scale"] = selected_scale
    corrected_parameters = ["battery_heat_generation_scale"]
    if coolant_mass_flow_ref_kg_s is not None:
        selected_mass_flow = float(coolant_mass_flow_ref_kg_s)
        if not math.isfinite(selected_mass_flow) or selected_mass_flow <= 0.0:
            raise ValueError(
                "coolant_mass_flow_ref_kg_s must be a positive finite number"
            )
        corrected["thermal"]["coolant_mass_flow_ref_kg_s"] = selected_mass_flow
        corrected_parameters.append("coolant_mass_flow_ref_kg_s")
    corrected.setdefault("fit", {})["battery_heat_generation_correction"] = {
        "fit_status": "validated",
        "method": "actual_future_command_replay",
        "calibration_source": str(calibration_source),
        "battery_heat_generation_scale": selected_scale,
        "capacity_model_frozen": True,
        "dynamic_model_frozen": True,
        "other_thermal_parameters_frozen": coolant_mass_flow_ref_kg_s is None,
        "corrected_thermal_parameters": corrected_parameters,
        "added_states": 0,
        "added_equations": 0,
        "runtime_change": "one_constant_multiplication",
    }
    validate_physics_artifact(corrected)
    return corrected


def _parse_horizons(raw: str) -> tuple[float, ...]:
    horizons = tuple(float(value.strip()) for value in raw.split(",") if value.strip())
    if not horizons:
        raise ValueError("horizons must not be empty")
    return horizons


def evaluate_before_after(
    frames: list[tuple[str, pd.DataFrame]],
    base_artifact: dict,
    corrected_artifact: dict,
    *,
    dt_s: float,
    horizons_s: tuple[float, ...],
    origin_stride: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frames = []
    summary_frames = []
    for model_name, artifact in (("base", base_artifact), ("corrected", corrected_artifact)):
        for case_name, frame in frames:
            details, summary = evaluate_actual_command_replay(
                frame,
                artifact,
                dt_s=dt_s,
                horizons_s=horizons_s,
                origin_stride=origin_stride,
            )
            details.insert(0, "case", case_name)
            details.insert(0, "model", model_name)
            summary.insert(0, "case", case_name)
            summary.insert(0, "model", model_name)
            detail_frames.append(details)
            summary_frames.append(summary)
    return (
        pd.concat(detail_frames, ignore_index=True),
        pd.concat(summary_frames, ignore_index=True),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为P模型加入电池发热比例修正，并用实际未来指令回放验证。"
    )
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--coolant-mass-flow-ref-kg-s", type=float)
    parser.add_argument("--dt-s", type=float, default=5.0)
    parser.add_argument(
        "--horizons-s",
        default=",".join(str(value) for value in DEFAULT_HORIZONS_S),
    )
    parser.add_argument("--origin-stride", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    artifact_path = output_dir / "physics_p_heat_generation_corrected.json"
    report_json = output_dir / "heat_generation_correction_report.json"
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_physics_artifact(args.base_artifact, require_validated=True)
    source_names = [path.resolve().as_posix() for path in args.input_csv]
    corrected = build_corrected_artifact(
        base,
        scale=args.scale,
        coolant_mass_flow_ref_kg_s=args.coolant_mass_flow_ref_kg_s,
        calibration_source=";".join(source_names),
    )
    frames = [
        (path.stem, pd.read_csv(path, encoding="utf-8-sig"))
        for path in args.input_csv
    ]
    horizons = _parse_horizons(args.horizons_s)
    details, summary = evaluate_before_after(
        frames,
        base,
        corrected,
        dt_s=args.dt_s,
        horizons_s=horizons,
        origin_stride=args.origin_stride,
    )

    battery_summary = summary.loc[summary["state"] == "T_Batt"].copy()
    corrected["fit"]["battery_heat_generation_correction"].update(
        {
            "dt_s": float(args.dt_s),
            "horizons_s": list(horizons),
            "origin_stride": int(args.origin_stride),
            "validation_battery_summary": battery_summary.to_dict(orient="records"),
        }
    )
    validate_physics_artifact(corrected)
    artifact_path.write_text(
        json.dumps(corrected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    details.to_csv(
        output_dir / "same_command_replay_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "same_command_replay_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    report = {
        "artifact": str(artifact_path),
        "base_artifact": str(args.base_artifact.resolve()),
        "battery_heat_generation_scale": float(args.scale),
        "coolant_mass_flow_ref_kg_s": (
            float(corrected["thermal"]["coolant_mass_flow_ref_kg_s"])
        ),
        "input_csvs": source_names,
        "dt_s": float(args.dt_s),
        "horizons_s": list(horizons),
        "origin_stride": int(args.origin_stride),
        "battery_summary": battery_summary.to_dict(orient="records"),
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"artifact={artifact_path}")
    print(battery_summary.to_string(index=False))


if __name__ == "__main__":
    main()
