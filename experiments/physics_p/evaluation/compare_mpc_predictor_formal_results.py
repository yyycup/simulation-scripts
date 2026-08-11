"""Validate, summarize, and plot completed formal MPC predictor runs.

The current Physics-P and Candidate-B runs are operational controller
candidates, not a strict predictor-only comparison: Physics-P retains its
native 300 rpm off command and recovery chain, while Candidate B retains its
native 1000 rpm command floor. Historical fee2f79-family results are plotted
only as context because their plant initialization and controls differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from p_mpc_run_support import PROJECT_ROOT

PHYSICS_P_ID = "physics_p"
CURRENT_CANDIDATE_B_ID = "current_candidate_b"
ARCHIVE_ID = "archive_candidate_b_fee2f79_family"
CURRENT_TIER = "current_operational_candidate"
HISTORICAL_TIER = "historical_context_only"
COMPARISON_SCOPE = "operational_candidate_not_predictor_only"
TARGET_TEMP_C = 25.0
CONTROL_INTERVAL_S = 5.0
COMPRESSOR_OFF_MAX_RPM = 301.0
COMPRESSOR_CONTINUOUS_MIN_RPM = 1000.0
COMPRESSOR_LOW_SPEED_MAX_RPM = 2000.0
COMPRESSOR_SATURATION_MIN_RPM = 5990.0

SCENES = {
    "peak": {"steps": 1280, "horizon": 60, "archive_name": "\u8c03\u5cf0"},
    "freq": {"steps": 720, "horizon": 45, "archive_name": "\u8c03\u9891"},
}
MODEL_STYLE = {
    PHYSICS_P_ID: {
        "label": "Physics-P (current, native)",
        "color": "#D55E00",
        "linestyle": "-",
        "linewidth": 1.7,
    },
    CURRENT_CANDIDATE_B_ID: {
        "label": "Candidate B + simplified dynamics (current)",
        "color": "#0072B2",
        "linestyle": "-",
        "linewidth": 1.55,
    },
    ARCHIVE_ID: {
        "label": "Candidate B archive (context only)",
        "color": "#777777",
        "linestyle": "--",
        "linewidth": 1.25,
    },
}

DEFAULT_P_ROOT = (
    PROJECT_ROOT / "outputs" / "p_mpc_formal_candidate_v2" / "full_20260726"
)
DEFAULT_CANDIDATE_B_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "p_mpc_formal_candidate_v2"
    / "candidate_b_full_20260726"
)
DEFAULT_ARCHIVE_ROOT = (
    Path.home()
    / "Desktop"
    / "\u79d1\u7814"
    / "\u8bba\u6587"
    / "\u5c0f\u8bba\u6587"
    / "\u4eff\u771f\u6570\u636e\u8f93\u51fa"
    / "\u626b\u63cf\u4eff\u771f\u53c2\u6570\u5b8c\u6574\u6570\u636e"
    / "01_\u6700\u7ec8\u91c7\u7528_12\u7ec4\u5b8c\u6574\u4eff\u771f\u4e0e\u7ed8\u56fe"
    / "\u670d\u52a1\u5668_12\u7ec4_\u7ec8\u7aef\u4ee3\u4ef7_candidateB_20260712_134913"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "p_mpc_formal_candidate_v2"
    / "comparison_20260726"
)


def build_result_specs(p_root, candidate_b_root, archive_root):
    p_root = Path(p_root)
    candidate_b_root = Path(candidate_b_root)
    archive_root = Path(archive_root)
    specs = []
    for scene, config in SCENES.items():
        steps = config["steps"]
        horizon = config["horizon"]
        specs.extend(
            [
                {
                    "scene": scene,
                    "model_id": PHYSICS_P_ID,
                    "comparison_tier": CURRENT_TIER,
                    "csv_path": p_root
                    / scene
                    / "mpc"
                    / f"{scene}_physics_p_steps{steps}_horizon{horizon}.csv",
                    "expected_rows": steps,
                    "horizon_steps": horizon,
                },
                {
                    "scene": scene,
                    "model_id": CURRENT_CANDIDATE_B_ID,
                    "comparison_tier": CURRENT_TIER,
                    "csv_path": candidate_b_root
                    / scene
                    / "mpc"
                    / f"{scene}_candidate_b_steps{steps}_horizon{horizon}.csv",
                    "expected_rows": steps,
                    "horizon_steps": horizon,
                },
                {
                    "scene": scene,
                    "model_id": ARCHIVE_ID,
                    "comparison_tier": HISTORICAL_TIER,
                    "csv_path": archive_root
                    / "mpc"
                    / (
                        config["archive_name"]
                        + "_\u5355\u5411_mpc_\u5b8c\u6574.csv"
                    ),
                    "expected_rows": steps,
                    "horizon_steps": horizon,
                },
            ]
        )
    return specs


def _required_numeric(frame, column):
    if column not in frame.columns:
        raise KeyError(f"missing required result column: {column}")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"required column is empty or non-finite: {column}")
    return values


def _optional_numeric(frame, column):
    if column not in frame.columns:
        return np.full(len(frame), np.nan, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _required_boolean(frame, column):
    if column not in frame.columns:
        raise KeyError(f"missing required result column: {column}")
    return frame[column].fillna(False).astype(str).str.lower().isin(("true", "1"))


def _optional_boolean(frame, column):
    if column not in frame.columns:
        return None
    raw = frame[column]
    text = raw.fillna("").astype(str).str.strip().str.lower()
    present = text.ne("") & text.ne("nan")
    if not bool(present.any()):
        return None
    return text.isin(("true", "1"))


def _finite_min(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.min(finite)) if finite.size else float("nan")


def _finite_max(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else float("nan")


def summarize_case(
    frame,
    *,
    scene,
    model_id,
    comparison_tier,
    source_csv,
):
    time_s = _required_numeric(frame, "Time")
    temperature = _required_numeric(frame, "Average temperature")
    delta_t = _required_numeric(frame, "Maximum temperature difference")
    coolant = _required_numeric(frame, "Coolant temperature")
    compressor = _required_numeric(frame, "Compressor command")
    pump = _required_numeric(frame, "Pump command")
    power = _required_numeric(frame, "Total power")
    energy = _required_numeric(frame, "Cumulative energy consumption")
    solve_time = _required_numeric(frame, "MPC solve time")
    solved = _required_boolean(frame, "MPC_Solved")
    recovery = _optional_boolean(frame, "MPC solve recovery used")
    domain_valid = _optional_boolean(frame, "MPC prediction domain valid")
    predicted_coolant = _optional_numeric(
        frame, "MPC predicted minimum coolant temperature"
    )
    domain_violation = _optional_numeric(
        frame, "MPC coolant prediction domain violation"
    )
    error = temperature - TARGET_TEMP_C
    return {
        "scene": scene,
        "model_id": model_id,
        "comparison_tier": comparison_tier,
        "comparison_scope": (
            COMPARISON_SCOPE if comparison_tier == CURRENT_TIER else HISTORICAL_TIER
        ),
        "source_csv": str(Path(source_csv)),
        "rows": int(len(frame)),
        "duration_s": float(time_s[-1] - time_s[0]),
        "temperature_mae_c": float(np.mean(np.abs(error))),
        "temperature_rmse_c": float(np.sqrt(np.mean(np.square(error)))),
        "temperature_min_c": float(np.min(temperature)),
        "temperature_max_c": float(np.max(temperature)),
        "temperature_final_c": float(temperature[-1]),
        "max_delta_t_mean_c": float(np.mean(delta_t)),
        "max_delta_t_max_c": float(np.max(delta_t)),
        "initial_coolant_c": float(coolant[0]),
        "coolant_min_c": float(np.min(coolant)),
        "coolant_max_c": float(np.max(coolant)),
        "energy_kwh": float(energy[-1]),
        "mean_total_power_kw": float(np.mean(power)),
        "mean_compressor_rpm": float(np.mean(compressor)),
        "mean_pump_rpm": float(np.mean(pump)),
        "compressor_observed_min_rpm": float(np.min(compressor)),
        "compressor_observed_max_rpm": float(np.max(compressor)),
        "compressor_off_rate": float(
            np.mean(compressor <= COMPRESSOR_OFF_MAX_RPM)
        ),
        "compressor_startup_transition_rate": float(
            np.mean(
                (compressor > COMPRESSOR_OFF_MAX_RPM)
                & (compressor < COMPRESSOR_CONTINUOUS_MIN_RPM)
            )
        ),
        "compressor_low_speed_rate": float(
            np.mean(
                (compressor >= COMPRESSOR_CONTINUOUS_MIN_RPM)
                & (compressor < COMPRESSOR_LOW_SPEED_MAX_RPM)
            )
        ),
        "compressor_saturation_rate": float(
            np.mean(compressor >= COMPRESSOR_SATURATION_MIN_RPM)
        ),
        "solve_success_rate": float(np.mean(solved)),
        "solve_recovery_count": (
            int(np.count_nonzero(recovery)) if recovery is not None else 0
        ),
        "prediction_domain_valid_rate": (
            float(np.mean(domain_valid))
            if domain_valid is not None
            else float("nan")
        ),
        "predicted_coolant_min_c": _finite_min(predicted_coolant),
        "coolant_domain_violation_max_c": _finite_max(domain_violation),
        "solve_time_mean_s": float(np.mean(solve_time)),
        "solve_time_p95_s": float(np.percentile(solve_time, 95)),
        "solve_time_max_s": float(np.max(solve_time)),
        "deadline_exceed_rate": float(np.mean(solve_time > CONTROL_INTERVAL_S)),
    }


def build_current_operational_pairwise_summary(summary):
    current = summary.loc[summary["comparison_tier"] == CURRENT_TIER]
    rows = []
    for scene in SCENES:
        scene_rows = current.loc[current["scene"] == scene].set_index("model_id")
        if PHYSICS_P_ID not in scene_rows.index:
            continue
        if CURRENT_CANDIDATE_B_ID not in scene_rows.index:
            continue
        p_row = scene_rows.loc[PHYSICS_P_ID]
        b_row = scene_rows.loc[CURRENT_CANDIDATE_B_ID]
        b_solve_time = float(b_row["solve_time_mean_s"])
        rows.append(
            {
                "scene": scene,
                "comparison_scope": COMPARISON_SCOPE,
                "p_minus_b_temperature_mae_c": float(
                    p_row["temperature_mae_c"] - b_row["temperature_mae_c"]
                ),
                "p_minus_b_temperature_final_c": float(
                    p_row["temperature_final_c"]
                    - b_row["temperature_final_c"]
                ),
                "p_minus_b_energy_kwh": float(
                    p_row["energy_kwh"] - b_row["energy_kwh"]
                ),
                "p_over_b_solve_time_ratio": (
                    float(p_row["solve_time_mean_s"] / b_solve_time)
                    if b_solve_time > 0.0
                    else float("nan")
                ),
                "physics_p_solve_success_rate": float(
                    p_row["solve_success_rate"]
                ),
                "candidate_b_solve_success_rate": float(
                    b_row["solve_success_rate"]
                ),
            }
        )
    return pd.DataFrame(rows)


def historical_comparability_reasons():
    return [
        "Historical battery temperature began near 25 C, but coolant and cold plates began at 35 C.",
        "The historical detailed plant forced cooling and compressor power to zero below 2000 rpm.",
        "The selected archive used compressor DMAX 1200 rpm/step for peak and 2400 rpm/step for frequency, not the current settings.",
        "Capacity artifacts, compressor power, low-speed refrigeration, initialization, and control configuration differ.",
        "The archive is historical context only and must not be interpreted as a fair current-model baseline.",
    ]


def validate_formal_status_payload(payload, *, expected_predictor):
    status = str(payload.get("status", "")).upper()
    if status != "COMPLETE":
        raise RuntimeError(f"formal run is not complete: status={status or 'MISSING'}")
    actual = str(payload.get("predictor", ""))
    if actual != expected_predictor:
        raise RuntimeError(
            f"formal status predictor mismatch: expected={expected_predictor}, actual={actual or 'MISSING'}"
        )
    completed = payload.get("completed_scenes")
    if not isinstance(completed, (list, tuple)) or set(completed) != set(SCENES):
        raise RuntimeError(f"formal status scenes are incomplete: {completed}")
    return payload


def _read_and_validate_status(root, expected_predictor):
    path = Path(root) / "local_formal_status.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing formal status: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_formal_status_payload(payload, expected_predictor=expected_predictor)
    expected_lower = 300.0 if expected_predictor == PHYSICS_P_ID else 1000.0
    if payload.get("experiment_scope") != "native_controller_candidate":
        raise RuntimeError(f"missing native experiment scope in status: {path}")
    if float(payload.get("resolved_compressor_command_lower_rpm", np.nan)) != expected_lower:
        raise RuntimeError(f"unexpected compressor lower bound in status: {path}")
    if float(payload.get("resolved_compressor_command_upper_rpm", np.nan)) != 6000.0:
        raise RuntimeError(f"unexpected compressor upper bound in status: {path}")
    return path, payload


def validate_current_control_profiles(p_status, candidate_b_status):
    p_profiles = p_status.get("formal_scene_profiles")
    b_profiles = candidate_b_status.get("formal_scene_profiles")
    if not isinstance(p_profiles, dict) or not isinstance(b_profiles, dict):
        raise RuntimeError("formal status is missing scene control profiles")
    if p_profiles != b_profiles:
        raise RuntimeError(
            "current P and Candidate-B formal control profiles do not match"
        )
    for scene, expected in SCENES.items():
        profile = p_profiles.get(scene)
        if not isinstance(profile, dict):
            raise RuntimeError(f"missing formal control profile for {scene}")
        if int(profile.get("simulation_steps", -1)) != expected["steps"]:
            raise RuntimeError(f"unexpected formal simulation length for {scene}")
        if int(profile.get("prediction_horizon_steps", -1)) != expected["horizon"]:
            raise RuntimeError(f"unexpected formal prediction horizon for {scene}")
        if float(profile.get("initial_thermal_state_c", np.nan)) != TARGET_TEMP_C:
            raise RuntimeError(f"unexpected initial thermal state for {scene}")
        if profile.get("predictor_specific_mpc_overrides") is not False:
            raise RuntimeError(f"predictor-specific MPC override active for {scene}")
    return p_profiles


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def _configure_plot_style():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 8.7,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "savefig.dpi": 350,
            "savefig.bbox": "tight",
        }
    )


def plot_comparison(frames, output_root):
    _configure_plot_style()
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 7.0), constrained_layout=True)
    scene_labels = {"peak": "\u8c03\u5cf0", "freq": "\u8c03\u9891"}
    titles = [
        "\u7535\u6c60\u5e73\u5747\u6e29\u5ea6",
        "\u7d2f\u8ba1\u80fd\u8017",
        "\u538b\u7f29\u673a\u6307\u4ee4",
        "MPC\u6c42\u89e3\u65f6\u95f4",
    ]
    for row, scene in enumerate(("peak", "freq")):
        for model_id in (PHYSICS_P_ID, CURRENT_CANDIDATE_B_ID, ARCHIVE_ID):
            frame = frames[(scene, model_id)]
            style = MODEL_STYLE[model_id]
            minutes = _required_numeric(frame, "Time") / 60.0
            series = (
                _required_numeric(frame, "Average temperature"),
                _required_numeric(frame, "Cumulative energy consumption"),
                _required_numeric(frame, "Compressor command"),
                _required_numeric(frame, "MPC solve time"),
            )
            for col, values in enumerate(series):
                axes[row, col].plot(minutes, values, **style)
        axes[row, 0].axhline(
            TARGET_TEMP_C, color="#000000", linestyle=":", linewidth=1.0
        )
        axes[row, 3].axhline(
            CONTROL_INTERVAL_S, color="#CC0000", linestyle=":", linewidth=1.0
        )
        for col, title in enumerate(titles):
            axes[row, col].set_title(f"{scene_labels[scene]} - {title}")
            axes[row, col].set_xlabel("\u65f6\u95f4 (min)")
        axes[row, 0].set_ylabel("\u6e29\u5ea6 (degC)")
        axes[row, 1].set_ylabel("\u80fd\u8017 (kWh)")
        axes[row, 2].set_ylabel("\u8f6c\u901f (rpm)")
        axes[row, 3].set_ylabel("\u65f6\u95f4 (s)")
    handles = []
    labels = []
    for model_id in (PHYSICS_P_ID, CURRENT_CANDIDATE_B_ID, ARCHIVE_ID):
        style = MODEL_STYLE[model_id]
        handle, = axes[0, 0].plot([], [], **style)
        handles.append(handle)
        labels.append(style["label"])
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle(
        "Current operational candidates; archive is context only",
        fontsize=10.5,
        fontweight="bold",
    )
    png_path = Path(output_root) / "fig_formal_predictor_comparison.png"
    pdf_path = Path(output_root) / "fig_formal_predictor_comparison.pdf"
    fig.savefig(png_path, dpi=350)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def generate_comparison(
    *,
    p_root=DEFAULT_P_ROOT,
    candidate_b_root=DEFAULT_CANDIDATE_B_ROOT,
    archive_root=DEFAULT_ARCHIVE_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    force=False,
):
    p_root = Path(p_root)
    candidate_b_root = Path(candidate_b_root)
    archive_root = Path(archive_root)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()) and not force:
        raise FileExistsError(
            f"comparison output already exists; choose a new root or use --force: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    p_status_path, p_status = _read_and_validate_status(p_root, PHYSICS_P_ID)
    b_status_path, b_status = _read_and_validate_status(
        candidate_b_root, "candidate_b"
    )
    validated_control_profiles = validate_current_control_profiles(
        p_status, b_status
    )
    frames = {}
    summaries = []
    artifacts = []
    for spec in build_result_specs(p_root, candidate_b_root, archive_root):
        path = spec["csv_path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing completed result CSV: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if len(frame) != spec["expected_rows"]:
            raise ValueError(
                f"{path} has {len(frame)} rows; expected {spec['expected_rows']}"
            )
        if spec["model_id"] != ARCHIVE_ID:
            if "MPC_Predictor" not in frame.columns:
                raise KeyError(f"missing current-run predictor column: {path}")
            expected = (
                PHYSICS_P_ID
                if spec["model_id"] == PHYSICS_P_ID
                else "candidate_b"
            )
            actual = set(frame["MPC_Predictor"].dropna().astype(str))
            if actual != {expected}:
                raise RuntimeError(
                    f"CSV predictor mismatch for {path}: expected {expected}, got {sorted(actual)}"
                )
        frames[(spec["scene"], spec["model_id"])] = frame
        summaries.append(
            summarize_case(
                frame,
                scene=spec["scene"],
                model_id=spec["model_id"],
                comparison_tier=spec["comparison_tier"],
                source_csv=path,
            )
        )
        artifacts.append(
            {
                "scene": spec["scene"],
                "model_id": spec["model_id"],
                "path": str(path.resolve()),
                "rows": len(frame),
                "sha256": _sha256(path),
            }
        )
    summary = pd.DataFrame(summaries)
    pairwise = build_current_operational_pairwise_summary(summary)
    summary_path = output_root / "formal_predictor_summary.csv"
    pairwise_path = output_root / "current_operational_pairwise_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pairwise.to_csv(pairwise_path, index=False, encoding="utf-8-sig")
    png_path, pdf_path = plot_comparison(frames, output_root)
    provenance = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "comparison_scope": COMPARISON_SCOPE,
        "interpretation": (
            "Current P/B results compare native controller candidates, not only predictor formulas."
        ),
        "current_common_conditions": [
            "current detailed plant",
            "25 C battery/coolant/plate/pipe initialization",
            "same scene inputs, target, weights, DMAX, horizons, and plant",
        ],
        "current_non_common_conditions": [
            "Physics-P command domain is 300-6000 rpm; Candidate B is 1000-6000 rpm.",
            "Physics-P and Candidate B retain predictor-specific solve recovery behavior.",
        ],
        "physics_p_status_path": str(p_status_path.resolve()),
        "physics_p_status": p_status,
        "candidate_b_status_path": str(b_status_path.resolve()),
        "candidate_b_status": b_status,
        "validated_current_control_profiles": validated_control_profiles,
        "historical_root": str(archive_root.resolve()),
        "historical_tier": HISTORICAL_TIER,
        "historical_comparability_reasons": historical_comparability_reasons(),
        "git": _git_state(),
        "artifacts": artifacts,
        "outputs": [
            str(summary_path.resolve()),
            str(pairwise_path.resolve()),
            str(png_path.resolve()),
            str(pdf_path.resolve()),
        ],
    }
    provenance_path = output_root / "formal_predictor_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_path, pairwise_path, provenance_path, png_path, pdf_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p-root", type=Path, default=DEFAULT_P_ROOT)
    parser.add_argument(
        "--candidate-b-root", type=Path, default=DEFAULT_CANDIDATE_B_ROOT
    )
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = generate_comparison(
        p_root=args.p_root,
        candidate_b_root=args.candidate_b_root,
        archive_root=args.archive_root,
        output_root=args.output_root,
        force=args.force,
    )
    for path in outputs:
        print(f"COMPARISON_OUTPUT {path}", flush=True)


if __name__ == "__main__":
    main()
