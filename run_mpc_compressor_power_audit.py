"""Audit the lightweight MPC compressor-power surrogate against plant data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mpc_flow_direction_strategies import (
    COMPRESSOR_POWER_COEFF,
    COMPRESSOR_POWER_DOMAIN,
    compressor_power_normalization_w,
    compressor_power_value,
)


DEFAULT_STEADY_CSV = (
    Path("outputs")
    / "mpc_predictor_low_speed_retrain_v1"
    / "data"
    / "steady_full.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path("outputs")
    / "mpc_predictor_low_speed_retrain_v1"
    / "compressor_power_audit"
)


def old_compressor_power_value(n_comp_rpm):
    speed = float(n_comp_rpm)
    return 3.57e-6 * speed**2 + 0.442 * speed + 34.0


def _metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    relative = np.abs(error) / np.maximum(np.abs(actual), 1e-9) * 100.0
    return {
        "MAE_W": float(np.mean(np.abs(error))),
        "RMSE_W": float(np.sqrt(np.mean(np.square(error)))),
        "MAPE_percent": float(np.mean(relative)),
        "max_relative_error_percent": float(np.max(relative)),
        "bias_W": float(np.mean(error)),
    }


def build_audit(steady_frame):
    required = {
        "scenario_id",
        "split",
        "n_comp_eff_rpm",
        "n_pump_eff_rpm",
        "t_cool_c",
        "t_ambient_c",
        "w_comp_w",
    }
    missing = sorted(required - set(steady_frame.columns))
    if missing:
        raise ValueError(f"Steady power audit data missing columns: {missing}")
    frame = steady_frame.copy()
    frame["old_power_surrogate_w"] = [
        old_compressor_power_value(speed)
        for speed in frame["n_comp_eff_rpm"]
    ]
    frame["selected_power_surrogate_w"] = [
        compressor_power_value(speed, coolant, ambient)
        for speed, coolant, ambient in zip(
            frame["n_comp_eff_rpm"],
            frame["t_cool_c"],
            frame["t_ambient_c"],
        )
    ]

    summaries = []
    for split in ("train", "validation", "test", "all"):
        subset = frame if split == "all" else frame.loc[frame["split"] == split]
        for model_name, prediction_column in (
            ("old_speed_quadratic", "old_power_surrogate_w"),
            ("selected_speed_temperature", "selected_power_surrogate_w"),
        ):
            summaries.append(
                {
                    "split": split,
                    "model": model_name,
                    "row_count": len(subset),
                    **_metrics(
                        subset["w_comp_w"],
                        subset[prediction_column],
                    ),
                }
            )

    domain_speeds = np.linspace(*COMPRESSOR_POWER_DOMAIN["n_comp_rpm"], 101)
    corner_profiles = []
    for coolant in COMPRESSOR_POWER_DOMAIN["t_cool_c"]:
        for ambient in COMPRESSOR_POWER_DOMAIN["t_ambient_c"]:
            values = np.array(
                [
                    compressor_power_value(speed, coolant, ambient)
                    for speed in domain_speeds
                ]
            )
            corner_profiles.append(values)
    corner_values = np.concatenate(corner_profiles)
    physical_checks = {
        "minimum_domain_power_w": float(np.min(corner_values)),
        "maximum_domain_power_w": float(np.max(corner_values)),
        "normalization_power_w": compressor_power_normalization_w(),
        "negative_power_count": int(np.sum(corner_values < 0.0)),
        "monotonic_violation_count": int(
            sum(np.sum(np.diff(values) < -1e-9) for values in corner_profiles)
        ),
    }
    return frame, pd.DataFrame(summaries), physical_checks


def write_audit(steady_csv=DEFAULT_STEADY_CSV, output_dir=DEFAULT_OUTPUT_DIR):
    steady_csv = Path(steady_csv)
    output_dir = Path(output_dir)
    frame = pd.read_csv(steady_csv, encoding="utf-8-sig")
    points, summary, physical_checks = build_audit(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    points.to_csv(
        output_dir / "compressor_power_audit_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "compressor_power_audit_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "status": "offline_surrogate_audit",
        "steady_csv": str(steady_csv),
        "selected_formula": (
            "N_comp*(a0+a1*N_comp+a2*(T_cool_C-25)+a3*(T_ambient_C-35))"
        ),
        "coefficients": list(COMPRESSOR_POWER_COEFF),
        "fit_split": "train",
        "selection_split": "validation",
        "selection_rule": (
            "reject pump term unless validation MAPE improves at least 10 percent"
        ),
        "physical_checks": physical_checks,
    }
    (output_dir / "compressor_power_audit_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary, physical_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steady-csv", type=Path, default=DEFAULT_STEADY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary, checks = write_audit(args.steady_csv, args.output_dir)
    print(summary.to_string(index=False))
    print(json.dumps(checks, indent=2))
    if checks["negative_power_count"] or checks["monotonic_violation_count"]:
        raise SystemExit("Selected compressor-power surrogate failed physical checks")


if __name__ == "__main__":
    main()
