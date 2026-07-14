"""Compare steady evaporator capacity in the plant and the reduced MPC model."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpc_flow_direction_strategies import load_reduced_model_calibration
from thermal_batch_config import AMBIENT_TEMP_K, MPC_U_NPUMP_INIT
from thermal_loop import staged_fan_speed
from thermal_system import pump_model, run_refrigeration_cycle


DEFAULT_SPEEDS_RPM = (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
BASE_KQ_W_PER_RPM = 1.0


def mpc_q_evap_steady_w(n_comp_rpm, kq_w_per_rpm):
    return float(n_comp_rpm) * float(kq_w_per_rpm)


def calibrated_kq_w_per_rpm():
    theta = load_reduced_model_calibration()
    return BASE_KQ_W_PER_RPM * float(theta.get("kq_scale", 1.0)), theta


def build_comparison(n_comp_values, n_pump_rpm, coolant_temp_c, outdoor_temp_k=AMBIENT_TEMP_K):
    kq_w_per_rpm, _theta = calibrated_kq_w_per_rpm()
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
        q_mpc_w = mpc_q_evap_steady_w(n_comp_rpm, kq_w_per_rpm)
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
    return pd.DataFrame(rows)


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
    parser.add_argument("--output-root", type=Path, default=Path("outputs") / "evaporator_mpc_capacity_diagnostic")
    args = parser.parse_args()

    frame = build_comparison(DEFAULT_SPEEDS_RPM, args.n_pump, args.coolant_temp_c)
    args.output_root.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_root / "evaporator_capacity_comparison.csv"
    figure_path = args.output_root / "evaporator_capacity_comparison.png"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_plot(frame, figure_path)
    kq_w_per_rpm, theta = calibrated_kq_w_per_rpm()
    print(f"kq_w_per_rpm={kq_w_per_rpm:g}; calibration={theta}")
    print(frame.to_string(index=False))
    print(f"CSV: {csv_path}")
    print(f"Figure: {figure_path}")


if __name__ == "__main__":
    main()
