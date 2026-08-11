"""Compare bounded low-speed compressor-efficiency extrapolation assumptions.

This runner does not modify the canonical plant or Candidate-B artifacts.  It
varies only the loss slope below the measured compressor-map boundary and
writes a separate evidence package for later model-selection decisions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import btms_runtime

btms_runtime.ensure_env_library_bin_on_path()

import numpy as np
import pandas as pd

from thermal_batch_config import AMBIENT_TEMP_K, COMPRESSOR_MAP_MIN_RPM
from thermal_loop import staged_fan_speed
from thermal_system import (
    clear_refrigeration_cycle_cache,
    pump_model,
    run_refrigeration_cycle,
)


DEFAULT_OUTPUT_DIR = Path("outputs") / "refrigeration_low_speed_sensitivity"
PROFILES = {
    "conservative": {
        "label_cn": "保守（低速效率损失更陡）",
        "loss_scale": 1.5,
    },
    "baseline": {
        "label_cn": "基准（当前正式假设）",
        "loss_scale": 1.0,
    },
    "optimistic": {
        "label_cn": "乐观（低速效率损失更缓）",
        "loss_scale": 0.5,
    },
}
N_COMP_RPM = (1000.0, 1250.0, 1500.0, 1750.0, 1999.0, 2000.0)
N_PUMP_RPM = (1600.0, 3000.0, 4800.0)
T_COOL_IN_C = (15.0, 25.0, 35.0)


def build_grid():
    rows = []
    for profile, profile_config in PROFILES.items():
        loss_scale = profile_config["loss_scale"]
        for n_pump in N_PUMP_RPM:
            m_dot_cool, _ = pump_model(n_pump)
            for t_cool_c in T_COOL_IN_C:
                for n_comp in N_COMP_RPM:
                    # Avoid 50-rpm production cache quantization masking the
                    # dedicated 1999 -> 2000 rpm continuity check.
                    clear_refrigeration_cycle_cache()
                    result = run_refrigeration_cycle(
                        n_comp,
                        staged_fan_speed(n_comp),
                        t_cool_c + 273.15,
                        m_dot_cool,
                        AMBIENT_TEMP_K,
                        low_speed_efficiency_loss_scale=loss_scale,
                    )
                    numeric_values = [
                        result[name]
                        for name in (
                            "Q_evap",
                            "W_comp",
                            "COP_system",
                            "exergy_efficiency",
                            "compressor_eta_vol",
                            "compressor_eta_is",
                        )
                    ]
                    rows.append(
                        {
                            "profile": profile,
                            "profile_cn": profile_config["label_cn"],
                            "low_speed_efficiency_loss_scale": loss_scale,
                            "N_comp_rpm": n_comp,
                            "N_pump_rpm": n_pump,
                            "T_cool_in_C": t_cool_c,
                            "Q_evap_W": result["Q_evap"],
                            "W_comp_W": result["W_comp"],
                            "COP_system": result["COP_system"],
                            "COP_Carnot": result["COP_Carnot"],
                            "exergy_efficiency": result["exergy_efficiency"],
                            "eta_vol": result["compressor_eta_vol"],
                            "eta_is": result["compressor_eta_is"],
                            "risk_liquid_slugging": result[
                                "risk_liquid_slugging"
                            ],
                            "all_key_values_finite": bool(
                                np.all(np.isfinite(numeric_values))
                            ),
                            "compressor_efficiency_source": result[
                                "compressor_efficiency_source"
                            ],
                        }
                    )
    return pd.DataFrame(rows)


def _relative_percent(value, reference):
    return abs(value - reference) / max(abs(reference), 1e-9) * 100.0


def summarize_grid(frame):
    key_columns = ["N_comp_rpm", "N_pump_rpm", "T_cool_in_C"]
    baseline = (
        frame.loc[frame["profile"] == "baseline", key_columns + [
            "Q_evap_W",
            "W_comp_W",
            "COP_system",
        ]]
        .rename(
            columns={
                "Q_evap_W": "baseline_Q_evap_W",
                "W_comp_W": "baseline_W_comp_W",
                "COP_system": "baseline_COP_system",
            }
        )
    )
    compared = frame.merge(baseline, on=key_columns, how="left", validate="many_to_one")
    for name in ("Q_evap_W", "W_comp_W", "COP_system"):
        compared[f"{name}_relative_to_baseline_percent"] = [
            _relative_percent(value, reference)
            for value, reference in zip(
                compared[name], compared[f"baseline_{name}"]
            )
        ]

    summary_rows = []
    for profile, group in compared.groupby("profile", sort=False):
        low_speed = group.loc[group["N_comp_rpm"] < COMPRESSOR_MAP_MIN_RPM]
        monotonic_q_violations = 0
        monotonic_w_violations = 0
        boundary_q_jumps = []
        boundary_w_jumps = []
        for _, operating_line in group.groupby(["N_pump_rpm", "T_cool_in_C"]):
            operating_line = operating_line.sort_values("N_comp_rpm")
            monotonic_q_violations += int(
                np.sum(np.diff(operating_line["Q_evap_W"]) < -1e-9)
            )
            monotonic_w_violations += int(
                np.sum(np.diff(operating_line["W_comp_W"]) < -1e-9)
            )
            below = operating_line.loc[operating_line["N_comp_rpm"] == 1999.0].iloc[0]
            edge = operating_line.loc[operating_line["N_comp_rpm"] == 2000.0].iloc[0]
            boundary_q_jumps.append(
                _relative_percent(below["Q_evap_W"], edge["Q_evap_W"])
            )
            boundary_w_jumps.append(
                _relative_percent(below["W_comp_W"], edge["W_comp_W"])
            )

        invalid_state_count = int((~group["all_key_values_finite"]).sum())
        slugging_risk_count = int(group["risk_liquid_slugging"].sum())
        exergy_out_of_bounds_count = int(
            ((group["exergy_efficiency"] < 0.0) | (group["exergy_efficiency"] > 1.0)).sum()
        )
        max_q_boundary_jump = float(max(boundary_q_jumps))
        max_w_boundary_jump = float(max(boundary_w_jumps))
        acceptance_passed = (
            invalid_state_count == 0
            and slugging_risk_count == 0
            and exergy_out_of_bounds_count == 0
            and monotonic_q_violations == 0
            and monotonic_w_violations == 0
            and max_q_boundary_jump < 1.0
            and max_w_boundary_jump < 1.0
        )
        summary_rows.append(
            {
                "profile": profile,
                "profile_cn": group["profile_cn"].iloc[0],
                "low_speed_efficiency_loss_scale": group[
                    "low_speed_efficiency_loss_scale"
                ].iloc[0],
                "Q_evap_min_W": float(group["Q_evap_W"].min()),
                "Q_evap_max_W": float(group["Q_evap_W"].max()),
                "W_comp_min_W": float(group["W_comp_W"].min()),
                "W_comp_max_W": float(group["W_comp_W"].max()),
                "COP_min": float(group["COP_system"].min()),
                "COP_max": float(group["COP_system"].max()),
                "max_low_speed_Q_deviation_percent": float(
                    low_speed["Q_evap_W_relative_to_baseline_percent"].max()
                ),
                "max_low_speed_W_deviation_percent": float(
                    low_speed["W_comp_W_relative_to_baseline_percent"].max()
                ),
                "max_low_speed_COP_deviation_percent": float(
                    low_speed["COP_system_relative_to_baseline_percent"].max()
                ),
                "max_1999_to_2000_Q_jump_percent": max_q_boundary_jump,
                "max_1999_to_2000_W_jump_percent": max_w_boundary_jump,
                "Q_monotonic_violation_count": monotonic_q_violations,
                "W_monotonic_violation_count": monotonic_w_violations,
                "invalid_state_count": invalid_state_count,
                "slugging_risk_count": slugging_risk_count,
                "exergy_out_of_bounds_count": exergy_out_of_bounds_count,
                "open_loop_acceptance_passed": acceptance_passed,
            }
        )
    return compared, pd.DataFrame(summary_rows)


def write_results(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = build_grid()
    comparison, summary = summarize_grid(frame)
    comparison.to_csv(
        output_dir / "low_speed_sensitivity_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        output_dir / "low_speed_sensitivity_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "status": "open_loop_sensitivity_only",
        "canonical_baseline_modified": False,
        "candidate_b_artifact_modified": False,
        "evidence_status": "physics_constrained_extrapolation_without_measurements",
        "measured_map_min_rpm": COMPRESSOR_MAP_MIN_RPM,
        "profiles": PROFILES,
        "grid": {
            "N_comp_rpm": list(N_COMP_RPM),
            "N_pump_rpm": list(N_PUMP_RPM),
            "T_cool_in_C": list(T_COOL_IN_C),
            "ambient_temperature_K": AMBIENT_TEMP_K,
        },
        "acceptance": {
            "finite_states": True,
            "no_liquid_slugging_flag": True,
            "exergy_efficiency_range": [0.0, 1.0],
            "Q_and_W_non_decreasing_with_speed": True,
            "max_1999_to_2000_relative_jump_percent": 1.0,
        },
    }
    (output_dir / "low_speed_sensitivity_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return comparison, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    _, summary = write_results(args.output_dir)
    print(summary.to_string(index=False))
    if not bool(summary["open_loop_acceptance_passed"].all()):
        raise SystemExit("Low-speed sensitivity open-loop acceptance failed")


if __name__ == "__main__":
    main()
