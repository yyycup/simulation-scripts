"""Validate local runtime Q_y / R_comp multipliers for the frozen QP baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .validate_fixed_qp_600s import (
    DEFAULT_ARTIFACT,
    PROFILE_NAMES,
    build_case,
    summarize_frame,
)
from ..controllers.fixed_qp.mpc import QPMPCWeights
from ..simulation.config import AMBIENT_TEMP_C, TARGET_TEMP_C
from ..simulation.case import simulate_case


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "state_space_qp_runtime_sensitivity"
ALPHA_PUMP = 1.0
SENSITIVITY_CASES = (
    ("alpha_t_0p7", 0.7, 1.0),
    ("baseline", 1.0, 1.0),
    ("alpha_t_1p3", 1.3, 1.0),
    ("alpha_comp_0p7", 1.0, 0.7),
    ("alpha_comp_1p3", 1.0, 1.3),
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def evaluate_sensitivity(summary: pd.DataFrame) -> dict:
    """Check expected monotonic tradeoffs on every formal profile."""

    trend_rows = []
    alpha_t_monotonic = True
    alpha_comp_monotonic = True
    for profile in PROFILE_NAMES:
        rows = summary.loc[summary["profile"] == profile]
        t_axis = rows.loc[rows["alpha_comp"] == 1.0].sort_values("alpha_t")
        comp_axis = rows.loc[rows["alpha_t"] == 1.0].sort_values("alpha_comp")
        t_tmax = t_axis["t_max_max_c"].to_numpy(dtype=float)
        t_energy = t_axis["total_energy_kwh"].to_numpy(dtype=float)
        comp_tmax = comp_axis["t_max_max_c"].to_numpy(dtype=float)
        comp_energy = comp_axis["total_energy_kwh"].to_numpy(dtype=float)
        t_ok = bool(
            len(t_axis) == 3
            and np.all(np.diff(t_tmax) <= 1.0e-6)
            and np.all(np.diff(t_energy) >= -1.0e-9)
        )
        comp_ok = bool(
            len(comp_axis) == 3
            and np.all(np.diff(comp_tmax) >= -1.0e-6)
            and np.all(np.diff(comp_energy) <= 1.0e-9)
        )
        alpha_t_monotonic &= t_ok
        alpha_comp_monotonic &= comp_ok
        trend_rows.append(
            {
                "profile": profile,
                "alpha_t_monotonic": t_ok,
                "alpha_comp_monotonic": comp_ok,
                "alpha_t_tmax_span_c": (
                    float(t_tmax[0] - t_tmax[-1]) if len(t_tmax) == 3 else np.nan
                ),
                "alpha_t_energy_span_kwh": (
                    float(t_energy[-1] - t_energy[0]) if len(t_energy) == 3 else np.nan
                ),
                "alpha_comp_tmax_span_c": (
                    float(comp_tmax[-1] - comp_tmax[0])
                    if len(comp_tmax) == 3
                    else np.nan
                ),
                "alpha_comp_energy_span_kwh": (
                    float(comp_energy[0] - comp_energy[-1])
                    if len(comp_energy) == 3
                    else np.nan
                ),
            }
        )
    reliability = bool(
        len(summary) == len(PROFILE_NAMES) * len(SENSITIVITY_CASES)
        and (summary["solved_rate"] == 1.0).all()
        and (summary["domain_valid_rate"] == 1.0).all()
        and (summary["constraint_max_violation_rpm"] <= 1.0e-6).all()
        and summary["all_core_values_finite"].all()
    )
    return {
        "alpha_t_monotonic": bool(alpha_t_monotonic),
        "alpha_comp_monotonic": bool(alpha_comp_monotonic),
        "reliability_passed": reliability,
        "td3_action_usable": bool(
            alpha_t_monotonic and alpha_comp_monotonic and reliability
        ),
        "trends": trend_rows,
    }


def _run_case(
    *,
    profile: str,
    label: str,
    alpha_t: float,
    alpha_comp: float,
    output_root: Path,
    force: bool,
    progress_path: Path,
) -> dict:
    case = build_case(profile, "qp", duration_s=600.0)
    main_name = f"{profile}_600s_{label}.csv"
    progress = {
        "status": "running",
        "profile": profile,
        "label": label,
        "alpha_t": alpha_t,
        "alpha_comp": alpha_comp,
        "alpha_pump": ALPHA_PUMP,
        "last_message": "starting",
    }
    _write_json(progress_path, progress)

    def update_progress(message: str) -> None:
        if message.startswith("PROGRESS"):
            progress["last_message"] = message
            _write_json(progress_path, progress)

    result = simulate_case(
        control="state_space_qp",
        scene=case.scene,
        flow=case.flow,
        source_csv=ROOT / "_state_space_qp_runtime_sensitivity_time_axis.csv",
        main_name=main_name,
        snap_name=f"{Path(main_name).stem}_snapshots.csv",
        output_root=output_root,
        dt=case.dt,
        target_temp_c=TARGET_TEMP_C,
        initial_thermal_temp_c=AMBIENT_TEMP_C,
        duration_s=600.0,
        result_tag="qp_runtime_weight_sensitivity",
        current_profile_override=case.current_profile,
        mpc_flow_mode="standard",
        mpc_predictor=case.predictor,
        mpc_predictor_artifact=DEFAULT_ARTIFACT,
        mpc_runtime_weight_multipliers=(alpha_t, alpha_comp, ALPHA_PUMP),
        mpc_forecast_profile_steps=case.current_profile.size,
        force=force,
        progress_interval_steps=20,
        log_func=update_progress,
    )
    frame = pd.read_csv(result["out_csv"], encoding="utf-8-sig")
    if len(frame) != 120:
        raise RuntimeError(f"{main_name} has {len(frame)} rows; expected 120")
    for column, expected in (
        ("MPC_Alpha_Temp", alpha_t),
        ("MPC_Alpha_Comp", alpha_comp),
        ("MPC_Alpha_Pump", ALPHA_PUMP),
    ):
        actual = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.allclose(actual, expected, rtol=0.0, atol=0.0):
            raise RuntimeError(f"{column} does not match requested value {expected}")
    summary = summarize_frame(
        frame,
        profile=profile,
        controller="qp",
        dt=case.dt,
    )
    summary.update(
        {
            "label": label,
            "alpha_t": alpha_t,
            "alpha_comp": alpha_comp,
            "alpha_pump": ALPHA_PUMP,
            "csv": str(result["out_csv"]),
        }
    )
    return summary


def run_sensitivity(
    output_root: Path = DEFAULT_OUTPUT_ROOT, *, force: bool = False
) -> dict:
    """Run five one-factor runtime settings over all three 600 s profiles."""

    default_weights = QPMPCWeights()
    if default_weights.q_y != 1.0 or default_weights.r_comp != 1.0:
        raise RuntimeError("runtime sensitivity requires the frozen (1, 1) baseline")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    summaries = []
    total = len(PROFILE_NAMES) * len(SENSITIVITY_CASES)
    completed = 0
    for profile in PROFILE_NAMES:
        for label, alpha_t, alpha_comp in SENSITIVITY_CASES:
            summary = _run_case(
                profile=profile,
                label=label,
                alpha_t=alpha_t,
                alpha_comp=alpha_comp,
                output_root=root,
                force=force,
                progress_path=progress_path,
            )
            summaries.append(summary)
            completed += 1
            print(
                f"DONE {completed}/{total} {profile} {label}: "
                f"Tmax={summary['t_max_max_c']:.4f}C, "
                f"E={summary['total_energy_kwh']:.6f}kWh, "
                f"solved={summary['solved_rate']:.1%}",
                flush=True,
            )

    frame = pd.DataFrame(summaries)
    decision = evaluate_sensitivity(frame)
    summary_path = root / "runtime_sensitivity_summary.csv"
    trends_path = root / "runtime_sensitivity_trends.csv"
    decision_path = root / "runtime_sensitivity_decision.json"
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(decision["trends"]).to_csv(
        trends_path, index=False, encoding="utf-8-sig"
    )
    _write_json(decision_path, decision)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "completed_cases": total,
            "td3_action_usable": decision["td3_action_usable"],
            "summary_csv": str(summary_path),
            "trends_csv": str(trends_path),
        },
    )
    return {
        "summary_csv": str(summary_path),
        "trends_csv": str(trends_path),
        "decision_json": str(decision_path),
        "td3_action_usable": decision["td3_action_usable"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = run_sensitivity(args.output_root, force=args.force)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
