"""Stage 8D3: chiller sizing scan targeting the RegD frequency-regulation load.

Extends the Stage 8D2 scan axes upward: the RegD case shows instantaneous
cluster heat spikes of ~10 kW against the frozen V2.1 capacity of 8.24 kW.
Target window: Qevap@6000rpm >= 11 kW with Tcond <= 52 C (3 C margin to the
55 C model bound). Never multiplies solved cycle outputs.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import pandas as pd

import cluster_plant_v2.refrigeration as refrigeration
from cluster_plant_v2.refrigeration import ClosedR134aCycle

SPEEDS_RPM = (4000.0, 6000.0)
DISPLACEMENTS_CC_REV = (22.0, 26.0, 28.0, 30.0, 32.0)
EVAPORATOR_AREAS_M2 = (2.5, 3.0, 3.5)
CONDENSER_AREAS_M2 = (4.0,)
NOMINAL_AIR_MASS_FLOWS_KG_S = (3.5, 4.0, 4.5)

COOLANT_INLET_TEMPERATURE_K = 298.15
COOLANT_MASS_FLOW_KG_S = 0.44982
AMBIENT_TEMPERATURE_K = 308.15
FAN_SPEED_RPM = 1200.0

QEVP6000_MIN_W = 11000.0
TCOND6000_MAX_C = 52.0


def _evaluate(displacement, evap_area, cond_area, air_flow) -> dict:
    original = (
        refrigeration.COMPRESSOR_DISPLACEMENT_M3_PER_REV,
        refrigeration.EVAPORATOR_AREA_M2,
        refrigeration.CONDENSER_AREA_M2,
        refrigeration.NOMINAL_AIR_MASS_FLOW_KG_S,
    )
    try:
        refrigeration.COMPRESSOR_DISPLACEMENT_M3_PER_REV = displacement * 1e-6
        refrigeration.EVAPORATOR_AREA_M2 = evap_area
        refrigeration.CONDENSER_AREA_M2 = cond_area
        refrigeration.NOMINAL_AIR_MASS_FLOW_KG_S = air_flow

        row = {
            "displacement_cc_rev": displacement,
            "evaporator_area_m2": evap_area,
            "condenser_area_m2": cond_area,
            "nominal_air_mass_flow_kg_s": air_flow,
        }
        all_closed = True
        for speed in SPEEDS_RPM:
            result = ClosedR134aCycle().solve(
                compressor_speed_rpm=speed,
                fan_speed_rpm=FAN_SPEED_RPM,
                coolant_inlet_temperature_k=COOLANT_INLET_TEMPERATURE_K,
                coolant_mass_flow_kg_s=COOLANT_MASS_FLOW_KG_S,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            )
            label = int(speed)
            q_evap = float(result["q_evaporator_w"])
            shaft = float(result["compressor_shaft_power_w"])
            row.update(
                {
                    f"qevap_{label}_w": q_evap,
                    f"qcond_{label}_w": float(result["q_condenser_w"]),
                    f"wshaft_{label}_w": shaft,
                    f"cop_{label}": q_evap / shaft,
                    f"tevap_{label}_c": float(
                        result["evaporating_saturation_temperature_k"] - 273.15
                    ),
                    f"tcond_{label}_c": float(
                        result["condensing_saturation_temperature_k"] - 273.15
                    ),
                    f"mass_flow_{label}_kg_s": float(
                        result["refrigerant_mass_flow_kg_s"]
                    ),
                    f"residuals_ok_{label}": bool(
                        result["solver_success"]
                        and float(result["mass_flow_relative_residual"]) < 1e-6
                        and float(result["cycle_energy_relative_residual"]) < 1e-9
                    ),
                }
            )
            all_closed = all_closed and row[f"residuals_ok_{label}"]

        row["all_closure_gates_pass"] = all_closed
        row["regd_target_pass"] = bool(
            row["qevap_6000_w"] >= QEVP6000_MIN_W
            and row["tcond_6000_c"] <= TCOND6000_MAX_C
        )
        # 越小越好：够用的前提下取最小排量/面积/风量
        row["oversizing_score"] = (
            row["displacement_cc_rev"]
            + 4.0 * row["evaporator_area_m2"]
            + 2.0 * row["nominal_air_mass_flow_kg_s"]
        )
        return row
    except Exception as exc:
        return {
            "displacement_cc_rev": displacement,
            "evaporator_area_m2": evap_area,
            "condenser_area_m2": cond_area,
            "nominal_air_mass_flow_kg_s": air_flow,
            "all_closure_gates_pass": False,
            "regd_target_pass": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc)[:120],
        }
    finally:
        (
            refrigeration.COMPRESSOR_DISPLACEMENT_M3_PER_REV,
            refrigeration.EVAPORATOR_AREA_M2,
            refrigeration.CONDENSER_AREA_M2,
            refrigeration.NOMINAL_AIR_MASS_FLOW_KG_S,
        ) = original


def main() -> None:
    rows = [
        _evaluate(*cand)
        for cand in product(
            DISPLACEMENTS_CC_REV,
            EVAPORATOR_AREAS_M2,
            CONDENSER_AREAS_M2,
            NOMINAL_AIR_MASS_FLOWS_KG_S,
        )
    ]
    frame = pd.DataFrame(rows)
    out_dir = Path(
        "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制/"
        "cluster_plant_v2/validation/results/stage8d3_regd_rescale"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_dir / "chiller_sizing_scan_8d3.csv", index=False)

    cols = [
        "displacement_cc_rev", "evaporator_area_m2",
        "nominal_air_mass_flow_kg_s", "qevap_4000_w", "qevap_6000_w",
        "tcond_6000_c", "tevap_6000_c", "cop_6000", "cop_4000",
        "regd_target_pass",
    ]
    passing = frame[frame["regd_target_pass"] & frame["all_closure_gates_pass"]]
    passing = passing.sort_values("oversizing_score")
    print(f"total {len(frame)} candidates, {len(passing)} pass target")
    print()
    print("=== PASS (sorted by oversizing score, smallest first) ===")
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(passing[cols].head(10).to_string(index=False))
    print()
    print("=== current V2.1 baseline for reference ===")
    base = frame[
        (frame["displacement_cc_rev"] == 22.0)
        & (frame["evaporator_area_m2"] == 2.5)
        & (frame["nominal_air_mass_flow_kg_s"] == 3.5)
    ]
    with pd.option_context("display.width", 200, "display.float_format", "{:.1f}".format):
        print(base[cols].to_string(index=False))
    print()
    if len(passing):
        best = passing.iloc[0]
        print(
            "RECOMMENDED: displacement={:.0f} cc/rev, evap={:.1f} m2, cond=4.0 m2, "
            "air={:.1f} kg/s -> Qevap6000={:.0f} W, Tcond6000={:.1f} C, COP6000={:.2f}".format(
                best["displacement_cc_rev"], best["evaporator_area_m2"],
                best["nominal_air_mass_flow_kg_s"], best["qevap_6000_w"],
                best["tcond_6000_c"], best["cop_6000"],
            )
        )
    print("csv:", out_dir / "chiller_sizing_scan_8d3.csv")


if __name__ == "__main__":
    main()
