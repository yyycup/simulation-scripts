"""Refine Physics-P thermal capacities against current-plant evidence.

The refinement is deliberately limited to two interpretable capacity hypotheses:

* restore the battery heat capacity used by the detailed 52-cell plant;
* restore the detailed plant cold-plate heat capacity through ``plate_tau_s``.

All capacity, actuator, delay, heat-transfer, and MPC parameters remain frozen.
The old artifact is never overwritten.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ..evaluation.evaluate_p_shadow_actual_replay import (
    evaluate_actual_command_replay,
)
from ..identification.fit_mpc_physics_predictor import (
    _dynamic_validation_metric,
)
from mpc_physics_predictor import (
    DEFAULT_THERMAL_PARAMETERS,
    initialize_physics_state,
    load_physics_artifact,
    step_physics_predictor,
    validate_physics_artifact,
)
from mpc_physics_shadow import battery_heat_generation_w
from ..tuning.run_p_mpc_short_comparison import (
    DEFAULT_P_ARTIFACT,
)
from p_mpc_run_support import PROJECT_ROOT
from thermal_batch_config import AMBIENT_TEMP_C, INITIAL_TEMP_C


DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "thermal_capacity_refinement_v2"
)
DEFAULT_DYNAMIC_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "data"
    / "dynamic_full.csv"
)
DEFAULT_FIXED_ROOT = (
    PROJECT_ROOT / "outputs" / "p_fixed_command_root_cause_v1" / "mpc"
)
DEFAULT_REPLAY_CSVS = (
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "thermal_refinement_v1"
    / "current_plant_replay"
    / "pid"
    / "peak_current_plant_replay90.csv",
    PROJECT_ROOT
    / "outputs"
    / "mpc_predictor_low_speed_retrain_v1"
    / "thermal_refinement_v1"
    / "current_plant_replay"
    / "pid"
    / "freq_current_plant_replay90.csv",
)
DEFAULT_ARTIFACT_NAME = (
    "physics_p_cubic20_direct45_constant_kbp_physical_cbatt.json"
)
FIXED_CASES = (
    ("peak", 60, 1000),
    ("peak", 60, 6000),
    ("freq", 45, 1000),
    ("freq", 45, 6000),
)
REPLAY_BATTERY_HORIZONS_S = (50.0, 100.0, 300.0)
DETAILED_PLANT_PLATE_HEAT_CAPACITY_J_K = 6000.0


def build_capacity_variants(base_artifact: Mapping) -> dict[str, dict]:
    """Return isolated battery/plate capacity hypotheses."""
    base = validate_physics_artifact(copy.deepcopy(dict(base_artifact)))
    thermal = base["thermal"]
    physical_battery_capacity = float(
        DEFAULT_THERMAL_PARAMETERS["battery_heat_capacity_j_k"]
    )
    reference_flow_capacity = (
        float(thermal["coolant_mass_flow_ref_kg_s"])
        * float(thermal["coolant_cp_j_kg_k"])
    )
    physical_plate_tau = (
        DETAILED_PLANT_PLATE_HEAT_CAPACITY_J_K / reference_flow_capacity
    )

    variants = {}
    for name, battery_capacity, plate_tau in (
        (
            "base",
            thermal["battery_heat_capacity_j_k"],
            thermal["plate_tau_s"],
        ),
        (
            "battery_physical",
            physical_battery_capacity,
            thermal["plate_tau_s"],
        ),
        (
            "plate_physical",
            thermal["battery_heat_capacity_j_k"],
            physical_plate_tau,
        ),
        (
            "both_physical",
            physical_battery_capacity,
            physical_plate_tau,
        ),
    ):
        artifact = copy.deepcopy(base)
        artifact["thermal"]["battery_heat_capacity_j_k"] = float(
            battery_capacity
        )
        artifact["thermal"]["plate_tau_s"] = float(plate_tau)
        variants[name] = validate_physics_artifact(artifact)
    return variants


def _fixed_csv_path(fixed_root: Path, scene: str, steps: int, comp_rpm: int):
    return fixed_root / f"{scene}_fixed_comp{comp_rpm}_steps{steps}.csv"


def evaluate_fixed_endpoints(
    artifact: Mapping,
    *,
    fixed_root: Path,
) -> pd.DataFrame:
    """Evaluate P against fixed-command detailed-plant endpoint temperatures."""
    rows = []
    for scene, steps, comp_rpm in FIXED_CASES:
        input_csv = _fixed_csv_path(fixed_root, scene, steps, comp_rpm)
        frame = pd.read_csv(input_csv, encoding="utf-8-sig")
        state = initialize_physics_state(
            n_comp_eff_rpm=1000.0,
            n_pump_eff_rpm=1600.0,
            q_cond_w=0.0,
            q_evap_w=0.0,
            t_supply_c=AMBIENT_TEMP_C,
            t_plate_c=AMBIENT_TEMP_C,
            t_return_c=AMBIENT_TEMP_C,
            t_batt_c=INITIAL_TEMP_C,
            t_cool_c=AMBIENT_TEMP_C,
        )
        for current_a in frame["Total current"].to_numpy(dtype=float):
            state = step_physics_predictor(
                state,
                n_comp_cmd_rpm=float(comp_rpm),
                n_pump_cmd_rpm=1600.0,
                q_gen_w=battery_heat_generation_w(float(current_a)),
                t_ambient_c=AMBIENT_TEMP_C,
                dt_s=5.0,
                artifact=artifact,
            )
        actual = float(frame["Average temperature"].iloc[-1])
        error = float(state.t_batt_c - actual)
        rows.append(
            {
                "scene": scene,
                "seconds": int(steps * 5),
                "compressor_rpm": int(comp_rpm),
                "pump_rpm": 1600,
                "prediction_c": float(state.t_batt_c),
                "actual_c": actual,
                "error_c": error,
                "abs_error_c": abs(error),
                "input_csv": str(input_csv),
            }
        )
    return pd.DataFrame(rows)


def evaluate_replay_battery(
    artifact: Mapping,
    *,
    replay_csvs: tuple[Path, ...],
) -> pd.DataFrame:
    """Evaluate battery MAE with real future commands from current-plant replay."""
    rows = []
    for input_csv in replay_csvs:
        frame = pd.read_csv(input_csv, encoding="utf-8-sig")
        _details, summary = evaluate_actual_command_replay(
            frame,
            artifact,
            dt_s=5.0,
            horizons_s=REPLAY_BATTERY_HORIZONS_S,
        )
        selected = summary.loc[summary["state"] == "T_Batt"]
        for record in selected.to_dict(orient="records"):
            rows.append(
                {
                    "case": input_csv.stem,
                    "horizon_s": float(record["horizon_s"]),
                    "mae_c": float(record["mae"]),
                    "bias_c": float(record["bias"]),
                    "input_csv": str(input_csv),
                }
            )
    return pd.DataFrame(rows)


def summarize_variant(
    *,
    name: str,
    fixed_details: pd.DataFrame,
    replay_details: pd.DataFrame,
    identification_validation_metric: float,
) -> dict[str, float | str]:
    row: dict[str, float | str] = {
        "variant": name,
        "fixed_endpoint_mae_c": float(fixed_details["abs_error_c"].mean()),
        "fixed_endpoint_bias_c": float(fixed_details["error_c"].mean()),
        "identification_validation_weighted_mae": float(
            identification_validation_metric
        ),
    }
    for horizon_s in REPLAY_BATTERY_HORIZONS_S:
        values = replay_details.loc[
            np.isclose(replay_details["horizon_s"], horizon_s), "mae_c"
        ]
        row[f"replay_battery_{int(horizon_s)}s_mae_c"] = float(values.mean())
    return row


def select_capacity_variant(summary: pd.DataFrame) -> str:
    """Select the lowest fixed-endpoint MAE without worsening replay battery MAE."""
    indexed = summary.set_index("variant")
    if "base" not in indexed.index:
        raise ValueError("capacity summary must contain the base variant")
    base = indexed.loc["base"]
    replay_columns = [
        f"replay_battery_{int(horizon_s)}s_mae_c"
        for horizon_s in REPLAY_BATTERY_HORIZONS_S
    ]
    eligible = []
    for name, row in indexed.iterrows():
        if any(float(row[column]) > float(base[column]) + 1e-12 for column in replay_columns):
            continue
        if float(row["fixed_endpoint_mae_c"]) > float(base["fixed_endpoint_mae_c"]) + 1e-12:
            continue
        eligible.append((float(row["fixed_endpoint_mae_c"]), str(name)))
    if not eligible:
        return "base"
    return min(eligible)[1]


def _write_selected_artifact(
    *,
    selected_artifact: dict,
    selected_variant: str,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    artifact = copy.deepcopy(selected_artifact)
    indexed = summary.set_index("variant")
    base = indexed.loc["base"]
    selected = indexed.loc[selected_variant]
    base_fixed = float(base["fixed_endpoint_mae_c"])
    selected_fixed = float(selected["fixed_endpoint_mae_c"])
    artifact.setdefault("fit", {})["thermal_capacity_refinement"] = {
        "fit_status": "validated_current_plant_fixed_and_replay",
        "selection_source": "current_plant_fixed_commands_and_actual_command_replay",
        "selected_variant": selected_variant,
        "physical_battery_heat_capacity_j_k": float(
            DEFAULT_THERMAL_PARAMETERS["battery_heat_capacity_j_k"]
        ),
        "detailed_plant_plate_heat_capacity_j_k": (
            DETAILED_PLANT_PLATE_HEAT_CAPACITY_J_K
        ),
        "all_other_parameters_frozen": True,
        "base_fixed_endpoint_mae_c": base_fixed,
        "selected_fixed_endpoint_mae_c": selected_fixed,
        "fixed_endpoint_improvement_fraction": (
            (base_fixed - selected_fixed) / base_fixed if base_fixed > 0.0 else 0.0
        ),
        "metrics": {
            name: {
                key: float(value)
                for key, value in row.items()
                if key != "variant"
            }
            for name, row in summary.set_index("variant").reset_index().set_index(
                "variant"
            ).iterrows()
        },
    }
    validate_physics_artifact(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, default=DEFAULT_P_ARTIFACT)
    parser.add_argument("--dynamic-csv", type=Path, default=DEFAULT_DYNAMIC_CSV)
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument(
        "--replay-csv",
        type=Path,
        action="append",
        dest="replay_csvs",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    return parser.parse_args()


def main():
    args = parse_args()
    replay_csvs = tuple(args.replay_csvs or DEFAULT_REPLAY_CSVS)
    base_artifact = load_physics_artifact(
        args.base_artifact,
        require_validated=True,
    )
    dynamic_frame = pd.read_csv(args.dynamic_csv, encoding="utf-8-sig")
    validation = dynamic_frame.loc[dynamic_frame["split"] == "validation"].copy()
    variants = build_capacity_variants(base_artifact)

    summary_rows = []
    fixed_frames = []
    replay_frames = []
    for name, artifact in variants.items():
        fixed = evaluate_fixed_endpoints(artifact, fixed_root=args.fixed_root)
        fixed.insert(0, "variant", name)
        replay = evaluate_replay_battery(
            artifact,
            replay_csvs=replay_csvs,
        )
        replay.insert(0, "variant", name)
        summary_rows.append(
            summarize_variant(
                name=name,
                fixed_details=fixed,
                replay_details=replay,
                identification_validation_metric=_dynamic_validation_metric(
                    artifact,
                    validation,
                ),
            )
        )
        fixed_frames.append(fixed)
        replay_frames.append(replay)

    summary = pd.DataFrame(summary_rows)
    selected_variant = select_capacity_variant(summary)
    output_root = args.output_root.resolve()
    artifact_path = output_root / "artifacts" / args.artifact_name
    _write_selected_artifact(
        selected_artifact=variants[selected_variant],
        selected_variant=selected_variant,
        summary=summary,
        output_path=artifact_path,
    )

    reports = output_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        reports / "thermal_capacity_variant_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(fixed_frames, ignore_index=True).to_csv(
        reports / "fixed_command_endpoint_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat(replay_frames, ignore_index=True).to_csv(
        reports / "actual_command_replay_battery_details.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(summary.to_string(index=False))
    print(f"selected_variant={selected_variant}")
    print(f"artifact={artifact_path}")


if __name__ == "__main__":
    main()
