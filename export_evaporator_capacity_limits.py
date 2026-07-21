"""Export the detailed evaporator limits over the Candidate-B 15-35 C grid."""

from pathlib import Path

import btms_runtime

btms_runtime.ensure_env_library_bin_on_path()

import pandas as pd

from run_mpc_evaporator_capacity_calibration import N_COMP, N_PUMP, T_COOL_ALL
from thermal_batch_config import AMBIENT_TEMP_K
from thermal_loop import staged_fan_speed
from thermal_system import pump_model, run_refrigeration_cycle


def build_limits_grid():
    rows = []
    for n_comp in N_COMP:
        for n_pump in N_PUMP:
            m_dot, _ = pump_model(n_pump)
            for t_cool in T_COOL_ALL:
                result = run_refrigeration_cycle(
                    n_comp,
                    staged_fan_speed(n_comp),
                    t_cool + 273.15,
                    m_dot,
                    AMBIENT_TEMP_K,
                )
                rows.append(
                    {
                        "N_comp_rpm": n_comp,
                        "N_pump_rpm": n_pump,
                        "T_cool_in_C": t_cool,
                        "Q_evap_plant_W": result["Q_evap"],
                        "Q_hx_potential_W": result["Q_hx_potential"],
                        "Q_ref_max_W": result["Q_ref_max"],
                        "limit_type": (
                            "Q_hx"
                            if result["Q_hx_potential"] <= result["Q_ref_max"]
                            else "Q_ref_max"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main():
    out = (
        Path("outputs")
        / "mpc_evaporator_capacity_candidate_b_15c"
        / "candidate_b_capacity_limits_grid.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    build_limits_grid().to_csv(out, index=False, encoding="utf-8-sig")
    print(out)


if __name__ == "__main__":
    main()
