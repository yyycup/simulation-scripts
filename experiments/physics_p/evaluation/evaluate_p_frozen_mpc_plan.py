"""Validate P against a cloned detailed plant under one frozen MPC plan."""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

import btms_runtime

btms_runtime.ensure_env_library_bin_on_path()

import numpy as np
import pandas as pd

import thermal_case_simulator as sim
from mpc_physics_shadow import PhysicsPShadowPredictor, battery_heat_generation_w
from thermal_batch_config import AMBIENT_TEMP_C


DT_S = 5.0
HORIZONS_S = (50.0, 100.0, 300.0)
MAX_STEPS = 60
CAPTURE_LOOKAHEAD_STEPS = 90
ACTIVE_COMP_MARGIN_RPM = 1.0
FLOW = "单向"
OUTPUT_ROOT = Path("outputs/p_shadow_frozen_mpc_plan_active")
SOURCE_ROOT = (
    Path.home()
    / "Desktop"
    / "科研"
    / "论文"
    / "小论文"
    / "仿真数据输出"
    / "归一化代价函数结果"
    / "mpc"
)
SOURCES = {
    "peak": ("调峰", SOURCE_ROOT / "调峰输出单向mpc.csv"),
    "freq": ("调频", SOURCE_ROOT / "调频输出单向mpc.csv"),
}
ARTIFACTS = {
    "Old": Path(
        "outputs/mpc_predictor_accuracy_correction_v2/artifacts/"
        "physics_p_dynamic_supply_cp_fixed_v1.json"
    ),
    "New": Path(
        "outputs/p_shadow_actual_command_replay/calibration/"
        "physics_p_actual_replay_corrected_v3_physical.json"
    ),
}
ACTUAL_FIELDS = (
    ("T_Batt", "Average temperature", "C", 1.0),
    ("T_Cool", "Coolant temperature", "C", 1.0),
    ("T_Plate", "Plate solid temperature", "C", 1.0),
    ("T_Supply", "Supply pipe coolant temperature", "C", 1.0),
    ("T_Return", "Return pipe coolant temperature", "C", 1.0),
    ("Q_Evap", "Evaporator cooling rate (kW)", "W", 1000.0),
)


class _FrozenPlanCaptured(RuntimeError):
    pass


def _extend_plan(values, n_steps=MAX_STEPS):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("MPC plan must contain at least one finite command")
    if array.size >= n_steps:
        return array[:n_steps].copy()
    return np.pad(array, (0, n_steps - array.size), constant_values=array[-1])


def _capture_first_active_plan(case, scene, source_csv):
    """Run normal receding MPC only until its first plan containing active cooling."""
    captured = {}
    original_create_controller = sim.create_controller

    def capturing_create_controller(*args, **kwargs):
        controller = original_create_controller(*args, **kwargs)
        current_profile = np.asarray(args[1], dtype=float)
        original_command = controller.command
        n_comp_min = float(controller.mpc_params.n_comp_min)

        def capturing_command(
            step_no,
            pack,
            t_tank_k,
            t_cabinet,
            **command_kwargs,
        ):
            command = original_command(
                step_no,
                pack,
                t_tank_k,
                t_cabinet,
                **command_kwargs,
            )
            flow_info = dict(getattr(controller, "last_flow_info", {}))
            comp_raw = flow_info.get("n_comp_plan_rpm")
            pump_raw = flow_info.get("n_pump_plan_rpm")
            if (
                bool(flow_info.get("solved", False))
                and comp_raw is not None
                and pump_raw is not None
                and max(comp_raw) > n_comp_min + ACTIVE_COMP_MARGIN_RPM
            ):
                origin = int(step_no)
                captured.update(
                    {
                        "pack": copy.deepcopy(pack),
                        "t_tank_k": float(t_tank_k),
                        "t_plate_k_array": np.asarray(
                            command_kwargs["plate_temps"], dtype=float
                        ).copy(),
                        "dynamic_state": copy.deepcopy(command_kwargs["thermal_state"]),
                        "is_reversed": bool(command_kwargs["is_reversed"]),
                        "current_profile": _extend_plan(
                            current_profile[origin : origin + MAX_STEPS]
                        ),
                        "origin_step": origin,
                        "origin_time_s": float(command_kwargs["current_time"]),
                        "flow_info": flow_info,
                        "n_comp_plan_rpm": _extend_plan(comp_raw),
                        "n_pump_plan_rpm": _extend_plan(pump_raw),
                        "raw_plan_steps": int(min(len(comp_raw), len(pump_raw))),
                    }
                )
                raise _FrozenPlanCaptured
            return command

        controller.command = capturing_command
        return controller

    sim.create_controller = capturing_create_controller
    try:
        try:
            sim.simulate_case(
                "mpc",
                scene,
                FLOW,
                source_csv,
                f"{case}_capture_probe.csv",
                f"{case}_capture_probe_snap.csv",
                output_root=OUTPUT_ROOT / "capture_probe",
                mpc_flow_mode="supervised",
                force=True,
                max_steps=CAPTURE_LOOKAHEAD_STEPS,
                result_tag="capture_first_active_mpc_plan",
            )
        except _FrozenPlanCaptured:
            pass
    finally:
        sim.create_controller = original_create_controller

    if not captured:
        raise RuntimeError(f"no solved active MPC plan was found for {case}")
    return captured


def _plan_metadata(case, scene, captured):
    flow_info = captured["flow_info"]
    raw_steps = captured["raw_plan_steps"]
    return {
        "case": case,
        "scene": scene,
        "origin_step": captured["origin_step"],
        "origin_time_s": captured["origin_time_s"],
        "solved": True,
        "solve_time_s": float(flow_info.get("solve_time_s", np.nan)),
        "raw_plan_steps": raw_steps,
        "executed_plan_steps": MAX_STEPS,
        "hold_last_steps": int(max(0, MAX_STEPS - raw_steps)),
        "initial_n_comp_eff_rpm": float(
            captured["dynamic_state"].get("N_comp_eff", captured["n_comp_plan_rpm"][0])
        ),
        "initial_n_pump_eff_rpm": float(
            captured["dynamic_state"].get("N_pump_eff", captured["n_pump_plan_rpm"][0])
        ),
        "n_comp_plan_rpm": captured["n_comp_plan_rpm"].tolist(),
        "n_pump_plan_rpm": captured["n_pump_plan_rpm"].tolist(),
    }


def _rollout_detailed_plant(captured):
    # Roll out a clone so the saved origin remains unchanged for P initialization.
    pack = copy.deepcopy(captured["pack"])
    t_tank_k = captured["t_tank_k"]
    t_plate_k_array = captured["t_plate_k_array"].copy()
    dynamic_state = copy.deepcopy(captured["dynamic_state"])
    is_reversed = captured["is_reversed"]
    current_profile = captured["current_profile"]
    comp_plan = captured["n_comp_plan_rpm"]
    pump_plan = captured["n_pump_plan_rpm"]
    c_plate_node = sim.PLATE_NODE_HEAT_CAPACITY_TOTAL / pack.cols
    rows = []

    for index in range(MAX_STEPS):
        pack.total_current = float(current_profile[index])
        thermal_step = sim.simulate_thermal_loop_step(
            pack=pack,
            T_tank_K=t_tank_k,
            T_plate_K_array=t_plate_k_array,
            N_comp_cmd=float(comp_plan[index]),
            N_pump_cmd=float(pump_plan[index]),
            T_outdoor=sim.AMBIENT_TEMP_K,
            dt=DT_S,
            is_reversed=is_reversed,
            C_plate_node=c_plate_node,
            compressor_power_scale=sim.compressor_power_scale_for("mpc"),
            dynamic_state=dynamic_state,
        )
        dynamic_state = thermal_step.get("dynamic_state", dynamic_state)
        t_tank_k = thermal_step["T_tank_K"]
        t_plate_k_array = thermal_step["T_plate_K_array"]
        pack.step(DT_S, T_plate=t_plate_k_array, T_cabinet=sim.AMBIENT_TEMP_K)
        pack.history = [[] for _ in range(pack.Ns)]
        rows.append(
            {
                "Time": captured["origin_time_s"] + (index + 1) * DT_S,
                "Forecast elapsed time (s)": (index + 1) * DT_S,
                "Total current": pack.total_current,
                "Compressor command": float(comp_plan[index]),
                "Pump command": float(pump_plan[index]),
                "Compressor Speed": thermal_step.get("N_comp_eff", np.nan),
                "Pump Speed (RPM)": thermal_step.get("N_pump_eff", np.nan),
                "Average temperature": pack.get_avg_temp() - 273.15,
                "Coolant temperature": t_tank_k - 273.15,
                "Plate solid temperature": float(np.mean(t_plate_k_array)) - 273.15,
                "Supply pipe coolant temperature": thermal_step.get(
                    "T_pipe_supply_K", np.nan
                )
                - 273.15,
                "Return pipe coolant temperature": thermal_step.get(
                    "T_pipe_return_K", np.nan
                )
                - 273.15,
                "Evaporator cooling rate (kW)": thermal_step.get(
                    "Q_dot_evap", np.nan
                )
                / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def _forecast_models(captured):
    dynamic_state = captured["dynamic_state"]
    pack = captured["pack"]
    t_tank_c = captured["t_tank_k"] - 273.15
    observed_state = {
        "n_comp_eff_rpm": dynamic_state.get(
            "N_comp_eff", captured["n_comp_plan_rpm"][0]
        ),
        "n_pump_eff_rpm": dynamic_state.get(
            "N_pump_eff", captured["n_pump_plan_rpm"][0]
        ),
        "q_cond_w": dynamic_state.get("Q_cond_eff", 0.0),
        "q_evap_w": dynamic_state.get("Q_evap_eff", 0.0),
        "t_supply_c": dynamic_state.get("T_pipe_supply_K", captured["t_tank_k"])
        - 273.15,
        "t_plate_c": float(np.mean(captured["t_plate_k_array"])) - 273.15,
        "t_return_c": dynamic_state.get("T_pipe_return_K", captured["t_tank_k"])
        - 273.15,
        "t_batt_c": pack.get_avg_temp() - 273.15,
        "t_cool_c": t_tank_c,
    }
    qgen = tuple(
        battery_heat_generation_w(value) for value in captured["current_profile"]
    )
    forecasts = {}
    for label, artifact in ARTIFACTS.items():
        predictor = PhysicsPShadowPredictor(
            artifact,
            dt_s=DT_S,
            horizons_s=HORIZONS_S,
        )
        forecasts[label] = predictor.forecast(
            observed_state=observed_state,
            n_comp_cmd_rpm=captured["n_comp_plan_rpm"][0],
            n_pump_cmd_rpm=captured["n_pump_plan_rpm"][0],
            n_comp_cmd_preview_rpm=captured["n_comp_plan_rpm"],
            n_pump_cmd_preview_rpm=captured["n_pump_plan_rpm"],
            q_gen_preview_w=qgen,
            t_ambient_c=AMBIENT_TEMP_C,
        )
    return forecasts


def evaluate_frozen_forecasts(case, physical_frame, forecasts):
    rows = []
    for label, forecast in forecasts.items():
        if forecast.get("P_Shadow_Command_Assumption") != "provided_plan_hold_last":
            raise ValueError(f"{label} forecast did not use the supplied frozen plan")
        for horizon_s in HORIZONS_S:
            steps = int(round(horizon_s / DT_S))
            target_index = steps - 1
            if target_index >= len(physical_frame):
                raise ValueError("physical rollout is shorter than the requested horizon")
            horizon_label = int(horizon_s)
            for state, actual_column, unit, scale in ACTUAL_FIELDS:
                prediction_key = f"P_Shadow_{state}_Pred_{horizon_label}s_{unit}"
                prediction = float(forecast[prediction_key])
                actual = float(physical_frame.iloc[target_index][actual_column]) * scale
                error = prediction - actual
                rows.append(
                    {
                        "case": case,
                        "model": label,
                        "horizon_s": horizon_s,
                        "target_index": target_index,
                        "state": state,
                        "unit": unit,
                        "prediction": prediction,
                        "actual": actual,
                        "error": error,
                        "abs_error": abs(error),
                    }
                )
    return pd.DataFrame(rows)


def _summary(details):
    rows = []
    group_columns = ["case", "model", "horizon_s", "state", "unit"]
    for keys, group in details.groupby(group_columns, sort=True):
        case, model, horizon_s, state, unit = keys
        error = group["error"].to_numpy(float)
        rows.append(
            {
                "case": case,
                "model": model,
                "horizon_s": horizon_s,
                "state": state,
                "unit": unit,
                "n": len(group),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
            }
        )
    return pd.DataFrame(rows)


def run_case(case, scene, source_csv):
    print(f"CAPTURE {case} first active MPC plan", flush=True)
    captured = _capture_first_active_plan(case, scene, source_csv)
    metadata = _plan_metadata(case, scene, captured)
    plan_path = OUTPUT_ROOT / "plans" / f"{case}_frozen_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        f"ROLLOUT {case} origin={metadata['origin_time_s']:.0f}s "
        f"raw_plan={metadata['raw_plan_steps']} hold_last={metadata['hold_last_steps']}",
        flush=True,
    )
    physical_frame = _rollout_detailed_plant(captured)
    if physical_frame["Evaporator cooling rate (kW)"].max() <= 0.0:
        raise RuntimeError(f"frozen rollout for {case} never entered active cooling")
    physical_path = OUTPUT_ROOT / "physical" / f"{case}_frozen_plan_physical.csv"
    physical_path.parent.mkdir(parents=True, exist_ok=True)
    physical_frame.to_csv(physical_path, index=False, encoding="utf-8-sig")
    forecasts = _forecast_models(captured)
    return metadata, evaluate_frozen_forecasts(case, physical_frame, forecasts)


def main(output_root=None):
    global OUTPUT_ROOT
    if output_root is not None:
        OUTPUT_ROOT = Path(output_root)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    tmp = (OUTPUT_ROOT / "gekko_tmp").resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tmp)
    os.environ["TMP"] = str(tmp)
    os.environ["TEMP"] = str(tmp)

    details_by_case = []
    metadata = {}
    for case, (scene, source_csv) in SOURCES.items():
        plan_metadata, details = run_case(case, scene, source_csv)
        metadata[case] = {
            key: value
            for key, value in plan_metadata.items()
            if key not in {"n_comp_plan_rpm", "n_pump_plan_rpm"}
        }
        details_by_case.append(details)

    details = pd.concat(details_by_case, ignore_index=True)
    combined = details.copy()
    combined["case"] = "combined"
    summary = _summary(pd.concat([details, combined], ignore_index=True))
    details.to_csv(OUTPUT_ROOT / "frozen_plan_details.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_ROOT / "frozen_plan_summary.csv", index=False, encoding="utf-8-sig")
    (OUTPUT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"SUMMARY_PATH={OUTPUT_ROOT / 'frozen_plan_summary.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate P against a cloned plant under one frozen MPC plan"
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    main(args.output_root)
