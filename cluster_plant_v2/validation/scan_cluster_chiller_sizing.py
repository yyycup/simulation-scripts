"""Physical sizing scan for the Stage 8D2 Cluster chiller.

The scan changes compressor displacement, both heat-exchanger areas, and
condenser-air capacity.  It never multiplies solved cycle outputs.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import cluster_plant_v2.refrigeration as refrigeration
from cluster_plant_v2.refrigeration import ClosedR134aCycle


SPEEDS_RPM = (4000.0, 6000.0)
DISPLACEMENTS_CC_REV = (14.0, 16.0, 18.0, 20.0, 22.0)
EVAPORATOR_AREAS_M2 = (2.0, 2.5, 3.0, 3.5, 4.0)
CONDENSER_AREAS_M2 = (2.0, 2.5, 3.0, 3.5, 4.0)
NOMINAL_AIR_MASS_FLOWS_KG_S = (2.0, 2.5, 3.0, 3.5)

COOLANT_INLET_TEMPERATURE_K = 298.15
COOLANT_MASS_FLOW_KG_S = 0.44982
AMBIENT_TEMPERATURE_K = 308.15
FAN_SPEED_RPM = 1200.0


def _evaluate_candidate(
    displacement_cc_rev: float,
    evaporator_area_m2: float,
    condenser_area_m2: float,
    nominal_air_mass_flow_kg_s: float,
) -> dict[str, object]:
    original = (
        refrigeration.COMPRESSOR_DISPLACEMENT_M3_PER_REV,
        refrigeration.EVAPORATOR_AREA_M2,
        refrigeration.CONDENSER_AREA_M2,
        refrigeration.NOMINAL_AIR_MASS_FLOW_KG_S,
    )
    try:
        refrigeration.COMPRESSOR_DISPLACEMENT_M3_PER_REV = (
            displacement_cc_rev * 1e-6
        )
        refrigeration.EVAPORATOR_AREA_M2 = evaporator_area_m2
        refrigeration.CONDENSER_AREA_M2 = condenser_area_m2
        refrigeration.NOMINAL_AIR_MASS_FLOW_KG_S = nominal_air_mass_flow_kg_s
        row: dict[str, object] = {
            "displacement_cc_rev": displacement_cc_rev,
            "evaporator_area_m2": evaporator_area_m2,
            "condenser_area_m2": condenser_area_m2,
            "nominal_air_mass_flow_kg_s": nominal_air_mass_flow_kg_s,
        }
        all_closed = True
        for speed_rpm in SPEEDS_RPM:
            result = ClosedR134aCycle().solve(
                compressor_speed_rpm=speed_rpm,
                fan_speed_rpm=FAN_SPEED_RPM,
                coolant_inlet_temperature_k=COOLANT_INLET_TEMPERATURE_K,
                coolant_mass_flow_kg_s=COOLANT_MASS_FLOW_KG_S,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            )
            label = int(speed_rpm)
            q_evap = float(result["q_evaporator_w"])
            shaft_power = float(result["compressor_shaft_power_w"])
            row.update(
                {
                    f"qevap_{label}_w": q_evap,
                    f"qcond_{label}_w": float(result["q_condenser_w"]),
                    f"wref_{label}_w": float(
                        result["compressor_refrigerant_power_w"]
                    ),
                    f"wshaft_{label}_w": shaft_power,
                    f"cop_{label}": q_evap / shaft_power,
                    f"refrigerant_mass_flow_{label}_kg_s": float(
                        result["refrigerant_mass_flow_kg_s"]
                    ),
                    f"tevap_{label}_c": float(
                        result["evaporating_saturation_temperature_k"] - 273.15
                    ),
                    f"tcond_{label}_c": float(
                        result["condensing_saturation_temperature_k"] - 273.15
                    ),
                    f"evaporator_ua_{label}_w_k": float(
                        result["evaporator_ua_w_k"]
                    ),
                    f"condenser_ua_{label}_w_k": float(
                        result["condenser_ua_w_k"]
                    ),
                    f"mass_residual_{label}": float(
                        result["mass_flow_relative_residual"]
                    ),
                    f"evaporator_residual_{label}": float(
                        result["evaporator_relative_residual"]
                    ),
                    f"condenser_residual_{label}": float(
                        result["condenser_relative_residual"]
                    ),
                    f"energy_residual_{label}": float(
                        result["cycle_energy_relative_residual"]
                    ),
                    f"solver_success_{label}": bool(result["solver_success"]),
                }
            )
            all_closed = all_closed and bool(result["solver_success"])
        row["all_closure_gates_pass"] = all_closed
        row["capacity_targets_pass"] = bool(
            5500.0 <= float(row["qevap_4000_w"]) <= 6000.0
            and 7500.0 <= float(row["qevap_6000_w"]) <= 8500.0
        )
        row["target_error_w"] = float(
            abs(float(row["qevap_4000_w"]) - 5750.0)
            + abs(float(row["qevap_6000_w"]) - 8000.0)
        )
        return row
    except Exception as exc:
        return {
            "displacement_cc_rev": displacement_cc_rev,
            "evaporator_area_m2": evaporator_area_m2,
            "condenser_area_m2": condenser_area_m2,
            "nominal_air_mass_flow_kg_s": nominal_air_mass_flow_kg_s,
            "all_closure_gates_pass": False,
            "capacity_targets_pass": False,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
    finally:
        (
            refrigeration.COMPRESSOR_DISPLACEMENT_M3_PER_REV,
            refrigeration.EVAPORATOR_AREA_M2,
            refrigeration.CONDENSER_AREA_M2,
            refrigeration.NOMINAL_AIR_MASS_FLOW_KG_S,
        ) = original


def run_scan() -> pd.DataFrame:
    rows = [
        _evaluate_candidate(*candidate)
        for candidate in product(
            DISPLACEMENTS_CC_REV,
            EVAPORATOR_AREAS_M2,
            CONDENSER_AREAS_M2,
            NOMINAL_AIR_MASS_FLOWS_KG_S,
        )
    ]
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["capacity_targets_pass", "all_closure_gates_pass", "target_error_w"],
        ascending=[False, False, True],
        na_position="last",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "stage8d2_cluster_plant_rescale"
            / "chiller_sizing_scan.csv"
        ),
    )
    args = parser.parse_args()
    frame = run_scan()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.head(20).to_string(index=False))
    print(f"target candidates: {int(frame['capacity_targets_pass'].sum())}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
