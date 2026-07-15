"""Compare steady evaporator capacity in the plant and a selected MPC predictor."""

import argparse
import json
import math
from numbers import Real
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpc_evaporator_capacity_model import (
    DEFAULT_CALIBRATION_PATH,
    evaluate_capacity,
    load_capacity_calibration,
)
from mpc_predictor_selection import (
    CANDIDATE_B,
    LPV_L,
    PHYSICS_P,
    SUPPORTED_PREDICTORS,
    normalize_predictor_name,
)
from thermal_batch_config import AMBIENT_TEMP_K, MPC_U_NPUMP_INIT
from thermal_loop import staged_fan_speed
from thermal_system import pump_model, run_refrigeration_cycle


DEFAULT_SPEEDS_RPM = (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)


def _validate_candidate_b_calibration(calibration, artifact_path):
    artifact_path = Path(artifact_path)

    def reject(message):
        raise ValueError(f"Candidate B artifact {artifact_path}: {message}")

    if not isinstance(calibration, dict):
        reject("root must be a JSON object")
    model_type = calibration.get("model_type")
    if model_type != CANDIDATE_B:
        reject(f"model_type={model_type!r}; expected {CANDIDATE_B!r}")

    coefficients = calibration.get("coefficients")
    if not isinstance(coefficients, dict):
        reject("coefficients must be a JSON object")

    def finite_number(container, field):
        value = container.get(field)
        if isinstance(value, bool) or not isinstance(value, Real):
            reject(f"{field} must be a non-boolean finite number")
        value = float(value)
        if not math.isfinite(value):
            reject(f"{field} must be a finite number")
        return value

    for coefficient in ("b0", "b1", "b2"):
        finite_number(coefficients, coefficient)
    if finite_number(calibration, "n_pump_ref_rpm") <= 0.0:
        reject("n_pump_ref_rpm must be greater than zero")
    if finite_number(calibration, "q_evap_upper_bound_w") < 0.0:
        reject("q_evap_upper_bound_w must be nonnegative")
    return calibration


def build_comparison(
    n_comp_values,
    n_pump_rpm,
    coolant_temp_c,
    outdoor_temp_k=AMBIENT_TEMP_K,
    predictor=CANDIDATE_B,
    predictor_artifact=None,
):
    predictor = normalize_predictor_name(predictor)
    if predictor in (PHYSICS_P, LPV_L):
        raise ValueError(
            f"Predictor {predictor} uses a provisional artifact that is not promoted; "
            "provide an explicit promoted artifact in a later task"
        )

    calibration_path = (
        Path(predictor_artifact) if predictor_artifact is not None else DEFAULT_CALIBRATION_PATH
    )
    try:
        calibration = load_capacity_calibration(path=calibration_path)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"Candidate B artifact {calibration_path}: unable to load: {exc}"
        ) from exc
    calibration = _validate_candidate_b_calibration(calibration, calibration_path)
    m_dot_cool, _pump_power = pump_model(float(n_pump_rpm))
    rows = []
    for n_comp_rpm in n_comp_values:
        fan_rpm = staged_fan_speed(float(n_comp_rpm))
        plant = run_refrigeration_cycle(
            float(n_comp_rpm),
            fan_rpm,
            float(coolant_temp_c) + 273.15,
            m_dot_cool,
            float(outdoor_temp_k),
        )
        q_plant_w = float(plant["Q_evap"])
        q_mpc_w = evaluate_capacity(
            calibration,
            float(n_comp_rpm),
            float(n_pump_rpm),
            float(coolant_temp_c),
        )
        error_w = q_mpc_w - q_plant_w
        relative_error_percent = np.nan if abs(q_plant_w) < 1e-9 else 100.0 * error_w / q_plant_w
        rows.append(
            {
                "N_comp": float(n_comp_rpm),
                "N_pump": float(n_pump_rpm),
                "Q_evap_plant_W": q_plant_w,
                "Q_evap_mpc_W": q_mpc_w,
                "error_W": error_w,
                "relative_error_percent": relative_error_percent,
            }
        )
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        predictor=predictor,
        calibration_path=str(calibration_path),
        model_type=calibration.get("model_type", "unknown"),
    )
    return frame


def save_plot(frame, output_path):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=160)
    ax.plot(frame["N_comp"], frame["Q_evap_plant_W"], marker="o", label="完整模型")
    ax.plot(frame["N_comp"], frame["Q_evap_mpc_W"], marker="s", label="MPC 稳态预测")
    ax.set_xlabel("压缩机转速 (rpm)")
    ax.set_ylabel("蒸发器制冷量 (W)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plant/MPC evaporator steady-capacity diagnostic")
    parser.add_argument("--n-pump", type=float, default=MPC_U_NPUMP_INIT)
    parser.add_argument("--coolant-temp-c", type=float, default=25.0)
    parser.add_argument(
        "--predictor",
        choices=SUPPORTED_PREDICTORS,
        default=CANDIDATE_B,
    )
    parser.add_argument("--predictor-artifact", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "evaporator_mpc_capacity_diagnostic")
    args = parser.parse_args()

    try:
        frame = build_comparison(
            DEFAULT_SPEEDS_RPM,
            args.n_pump,
            args.coolant_temp_c,
            predictor=args.predictor,
            predictor_artifact=args.predictor_artifact,
        )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "evaporator_capacity_comparison.csv"
    figure_path = args.output_root / "evaporator_capacity_comparison.png"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_plot(frame, figure_path)
    print(
        f"predictor={frame.attrs['predictor']}; "
        f"calibration={frame.attrs['calibration_path']}; "
        f"model_type={frame.attrs['model_type']}"
    )
    print(frame.to_string(index=False))
    print(f"CSV: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
