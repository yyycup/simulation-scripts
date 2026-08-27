"""Small Q_y / R_comp calibration for State-Space QP-MPC.

Only the battery-temperature tracking weight and compressor-effort weight vary.
All Plant, Physics-P, constraint, horizon, and remaining weight settings stay at
their current frozen values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

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
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "state_space_qp_qy_rcomp_calibration"
GEKKO_REFERENCE = (
    ROOT / "outputs" / "state_space_qp_600s_comparison" / "summary.csv"
)
SCREEN_CASES = (
    ("baseline", 1.0, 3.0),
    ("qy1_rcomp2", 1.0, 2.0),
    ("qy1_rcomp2p5", 1.0, 2.5),
    ("qy1p5_rcomp3", 1.5, 3.0),
    ("qy2_rcomp3", 2.0, 3.0),
)
FOLLOWUP_CASES = (
    ("qy1_rcomp1p5", 1.0, 1.5),
    ("qy1_rcomp1", 1.0, 1.0),
)


def calibration_weights(q_y: float, r_comp: float) -> QPMPCWeights:
    """Build one candidate while freezing every non-calibrated weight."""

    return QPMPCWeights(
        q_y=float(q_y),
        r_comp=float(r_comp),
        r_pump=1.0e-3,
        r_delta_comp=1.0e-2,
        r_delta_pump=1.0e-2,
        p_f=1.0,
    ).validated()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_qp_case(
    *,
    profile: str,
    duration_s: float,
    label: str,
    weights: QPMPCWeights,
    output_root: Path,
    force: bool,
    progress_path: Path,
) -> dict:
    case = build_case(profile, "qp", duration_s=duration_s)
    main_name = f"{profile}_{int(duration_s)}s_{label}.csv"
    progress = {
        "status": "running",
        "profile": profile,
        "duration_s": duration_s,
        "label": label,
        "weights": asdict(weights),
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
        source_csv=ROOT / "_state_space_qp_calibration_time_axis.csv",
        main_name=main_name,
        snap_name=f"{Path(main_name).stem}_snapshots.csv",
        output_root=output_root,
        dt=case.dt,
        target_temp_c=TARGET_TEMP_C,
        initial_thermal_temp_c=AMBIENT_TEMP_C,
        duration_s=duration_s,
        result_tag="qp_qy_rcomp_calibration",
        current_profile_override=case.current_profile,
        mpc_flow_mode="standard",
        mpc_predictor=case.predictor,
        mpc_predictor_artifact=DEFAULT_ARTIFACT,
        state_space_qp_weights=weights,
        mpc_forecast_profile_steps=case.current_profile.size,
        force=force,
        progress_interval_steps=20,
        log_func=update_progress,
    )
    frame = pd.read_csv(result["out_csv"], encoding="utf-8-sig")
    expected_steps = int(round(duration_s / case.dt))
    if len(frame) != expected_steps:
        raise RuntimeError(
            f"{main_name} has {len(frame)} rows; expected {expected_steps}"
        )
    summary = summarize_frame(
        frame,
        profile=profile,
        controller="qp",
        dt=case.dt,
    )
    summary.update(
        {
            "label": label,
            "q_y": weights.q_y,
            "r_comp": weights.r_comp,
            "csv": str(result["out_csv"]),
        }
    )
    return summary


def run_screen(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
    cases=SCREEN_CASES,
    folder: str = "screen",
) -> dict:
    """Run the baseline and four requested candidates on a 300 s load step."""

    root = Path(output_root) / folder
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    summaries = []
    for index, (label, q_y, r_comp) in enumerate(cases, start=1):
        summary = _run_qp_case(
            profile="current_step",
            duration_s=300.0,
            label=label,
            weights=calibration_weights(q_y, r_comp),
            output_root=root,
            force=force,
            progress_path=progress_path,
        )
        summaries.append(summary)
        print(
            f"DONE {folder} {index}/{len(cases)} {label}: "
            f"Tmax={summary['t_max_max_c']:.4f}C, "
            f"E={summary['total_energy_kwh']:.6f}kWh, "
            f"solved={summary['solved_rate']:.1%}",
            flush=True,
        )

    frame = pd.DataFrame(summaries)
    if "baseline" in set(frame["label"]):
        baseline = frame.loc[frame["label"] == "baseline"].iloc[0]
    else:
        baseline_path = Path(output_root) / "screen" / "screen_summary.csv"
        baseline_frame = pd.read_csv(baseline_path, encoding="utf-8-sig")
        baseline = baseline_frame.loc[baseline_frame["label"] == "baseline"].iloc[0]
    frame["tmax_change_vs_baseline_c"] = frame["t_max_max_c"] - baseline["t_max_max_c"]
    frame["energy_change_vs_baseline_rate"] = (
        frame["total_energy_kwh"] / baseline["total_energy_kwh"] - 1.0
    )
    summary_path = root / f"{folder}_summary.csv"
    frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    _write_json(
        progress_path,
        {
            "status": "completed",
            "completed_cases": len(frame),
            "summary_csv": str(summary_path),
        },
    )
    return {"summary_csv": str(summary_path), "cases": len(frame)}


def evaluate_confirmation(
    candidate: pd.DataFrame, gekko_reference: pd.DataFrame
) -> dict:
    """Evaluate the three-profile candidate against the formal GEKKO reference."""

    candidate_rows = candidate.set_index("profile")
    gekko_rows = gekko_reference.set_index("profile")
    comparisons = []
    for profile in PROFILE_NAMES:
        qp = candidate_rows.loc[profile]
        gekko = gekko_rows.loc[profile]
        comparisons.append(
            {
                "profile": profile,
                "qp_tmax_c": float(qp["t_max_max_c"]),
                "gekko_tmax_c": float(gekko["t_max_max_c"]),
                "tmax_degradation_c": float(
                    qp["t_max_max_c"] - gekko["t_max_max_c"]
                ),
                "qp_total_energy_kwh": float(qp["total_energy_kwh"]),
                "gekko_total_energy_kwh": float(gekko["total_energy_kwh"]),
                "energy_change_rate": float(
                    qp["total_energy_kwh"] / gekko["total_energy_kwh"] - 1.0
                ),
            }
        )
    comparison = pd.DataFrame(comparisons)
    checks = {
        "three_profiles_present": bool(
            set(candidate_rows.index) == set(PROFILE_NAMES)
        ),
        "tmax_degradation_at_most_0p1_c": bool(
            (comparison["tmax_degradation_c"] <= 0.1).all()
        ),
        "energy_advantage_retained": bool(
            (comparison["energy_change_rate"] < 0.0).all()
        ),
        "solved_rate_100pct": bool((candidate["solved_rate"] == 1.0).all()),
        "domain_valid_rate_100pct": bool(
            (candidate["domain_valid_rate"] == 1.0).all()
        ),
        "constraints_satisfied": bool(
            (candidate["constraint_max_violation_rpm"] <= 1.0e-6).all()
        ),
    }
    return {
        "accepted": bool(all(checks.values())),
        "checks": checks,
        "comparisons": comparisons,
    }


def run_confirmation(
    q_y: float,
    r_comp: float,
    label: str,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    *,
    force: bool = False,
) -> dict:
    """Run one selected candidate over all three formal 600 s profiles."""

    root = Path(output_root) / "confirmation" / str(label)
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    weights = calibration_weights(q_y, r_comp)
    summaries = []
    for index, profile in enumerate(PROFILE_NAMES, start=1):
        summary = _run_qp_case(
            profile=profile,
            duration_s=600.0,
            label=label,
            weights=weights,
            output_root=root,
            force=force,
            progress_path=progress_path,
        )
        summaries.append(summary)
        print(
            f"DONE confirm {index}/{len(PROFILE_NAMES)} {profile}: "
            f"Tmax={summary['t_max_max_c']:.4f}C, "
            f"E={summary['total_energy_kwh']:.6f}kWh, "
            f"solved={summary['solved_rate']:.1%}",
            flush=True,
        )

    candidate = pd.DataFrame(summaries)
    reference = pd.read_csv(GEKKO_REFERENCE, encoding="utf-8-sig")
    reference = reference.loc[reference["controller"] == "gekko"].copy()
    decision = evaluate_confirmation(candidate, reference)
    summary_path = root / "confirmation_summary.csv"
    comparison_path = root / "comparison_to_gekko.csv"
    decision_path = root / "decision.json"
    candidate.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(decision["comparisons"]).to_csv(
        comparison_path, index=False, encoding="utf-8-sig"
    )
    _write_json(decision_path, decision)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "candidate": {"label": label, "q_y": q_y, "r_comp": r_comp},
            "accepted": decision["accepted"],
            "summary_csv": str(summary_path),
            "comparison_csv": str(comparison_path),
        },
    )
    return {
        "accepted": decision["accepted"],
        "summary_csv": str(summary_path),
        "comparison_csv": str(comparison_path),
        "decision_json": str(decision_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("screen", "followup", "confirm"))
    parser.add_argument("--q-y", type=float)
    parser.add_argument("--r-comp", type=float)
    parser.add_argument("--label")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.phase == "screen":
        result = run_screen(args.output_root, force=args.force)
    elif args.phase == "followup":
        result = run_screen(
            args.output_root,
            force=args.force,
            cases=FOLLOWUP_CASES,
            folder="followup",
        )
    else:
        if args.q_y is None or args.r_comp is None or not args.label:
            parser.error("confirm requires --q-y, --r-comp, and --label")
        result = run_confirmation(
            args.q_y,
            args.r_comp,
            args.label,
            args.output_root,
            force=args.force,
        )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
