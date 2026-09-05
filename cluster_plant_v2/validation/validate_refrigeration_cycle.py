"""Validate the independent A+B physically closed R134a cycle."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.refrigeration import ClosedR134aCycle


COOLANT_DENSITY_KG_M3 = 1071.0
COOLANT_INLET_TEMPERATURE_K = 298.15
AMBIENT_TEMPERATURE_K = 308.15


@dataclass(frozen=True)
class RefrigerationCycleCaseSpec:
    case_id: str
    compressor_speed_rpm: float
    fan_speed_rpm: float
    coolant_volume_flow_l_min: float


def fan_speed_for_compressor(compressor_speed_rpm: float) -> float:
    if compressor_speed_rpm <= 3500.0:
        return 800.0
    if compressor_speed_rpm <= 5000.0:
        return 1200.0
    return 1800.0


def build_cycle_case_specs() -> list[RefrigerationCycleCaseSpec]:
    cases = []
    for speed in (1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0):
        for flow in (16.8, 25.2, 33.6):
            flow_label = str(flow).replace(".", "p")
            cases.append(
                RefrigerationCycleCaseSpec(
                    case_id=f"R{int(speed)}_F{flow_label}",
                    compressor_speed_rpm=speed,
                    fan_speed_rpm=fan_speed_for_compressor(speed),
                    coolant_volume_flow_l_min=flow,
                )
            )
    return cases


def run_cycle_case(case: RefrigerationCycleCaseSpec) -> dict[str, object]:
    coolant_mass_flow = (
        case.coolant_volume_flow_l_min
        / 1000.0
        / 60.0
        * COOLANT_DENSITY_KG_M3
    )
    result = ClosedR134aCycle().solve(
        compressor_speed_rpm=case.compressor_speed_rpm,
        fan_speed_rpm=case.fan_speed_rpm,
        coolant_inlet_temperature_k=COOLANT_INLET_TEMPERATURE_K,
        coolant_mass_flow_kg_s=coolant_mass_flow,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
    )
    states = [result[f"state_{index}"] for index in range(1, 5)]
    state_values = np.array(
        [
            value
            for state in states
            for value in (
                state.pressure_pa,
                state.temperature_k,
                state.enthalpy_j_kg,
                state.entropy_j_kg_k,
                state.density_kg_m3,
            )
        ],
        dtype=float,
    )
    all_states_finite = bool(np.all(np.isfinite(state_values)))
    state_connections_pass = bool(
        result["evaporator_outlet_state"]
        is result["compressor_inlet_state"]
        and result["compressor_inlet_state"] is result["state_1"]
        and abs(states[3].enthalpy_j_kg - states[2].enthalpy_j_kg) <= 1e-9
    )
    coefficient_of_performance = float(
        result["q_evaporator_w"] / result["compressor_shaft_power_w"]
    )
    return {
        "case_id": case.case_id,
        "compressor_speed_rpm": case.compressor_speed_rpm,
        "fan_speed_rpm": case.fan_speed_rpm,
        "coolant_volume_flow_l_min": case.coolant_volume_flow_l_min,
        "coolant_mass_flow_kg_s": coolant_mass_flow,
        "coolant_inlet_temperature_c": (
            COOLANT_INLET_TEMPERATURE_K - 273.15
        ),
        "ambient_temperature_c": AMBIENT_TEMPERATURE_K - 273.15,
        "evaporating_saturation_temperature_c": (
            result["evaporating_saturation_temperature_k"] - 273.15
        ),
        "condensing_saturation_temperature_c": (
            result["condensing_saturation_temperature_k"] - 273.15
        ),
        "refrigerant_mass_flow_kg_s": result[
            "refrigerant_mass_flow_kg_s"
        ],
        "q_evaporator_w": result["q_evaporator_w"],
        "q_condenser_w": result["q_condenser_w"],
        "compressor_refrigerant_power_w": result[
            "compressor_refrigerant_power_w"
        ],
        "compressor_shaft_power_w": result["compressor_shaft_power_w"],
        "compressor_mechanical_loss_w": result[
            "compressor_mechanical_loss_w"
        ],
        "coefficient_of_performance": coefficient_of_performance,
        "coolant_outlet_temperature_c": (
            result["coolant_outlet_temperature_k"] - 273.15
        ),
        "air_outlet_temperature_c": (
            result["air_outlet_temperature_k"] - 273.15
        ),
        "mass_flow_relative_residual": result[
            "mass_flow_relative_residual"
        ],
        "evaporator_relative_residual": result[
            "evaporator_relative_residual"
        ],
        "condenser_relative_residual": result[
            "condenser_relative_residual"
        ],
        "cycle_energy_relative_residual": result[
            "cycle_energy_relative_residual"
        ],
        "solver_function_evaluations": result[
            "solver_function_evaluations"
        ],
        "solver_success": result["solver_success"],
        "solver_message": result["solver_message"],
        "all_states_finite": all_states_finite,
        "state_connections_pass": state_connections_pass,
        "all_gates_pass": bool(
            result["solver_success"]
            and all_states_finite
            and state_connections_pass
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the physically closed steady R134a cycle."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "cluster_plant_v2/validation/results/"
            "r134a_cycle_ab_validation_20260818"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    cases = build_cycle_case_specs()
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.case_id}")
        row = run_cycle_case(case)
        rows.append(row)
        print(
            f"  Qevap={row['q_evaporator_w']:.3f} W, "
            f"COP={row['coefficient_of_performance']:.6f}, "
            f"pass={row['all_gates_pass']}"
        )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output_dir / "r134a_cycle_ab_summary.csv", index=False)
    print(f"R134a A+B results: {args.output_dir}")


if __name__ == "__main__":
    main()
