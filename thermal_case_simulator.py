import time
from pathlib import Path

import numpy as np
import pandas as pd

from age_model import AgingModel280Ah
from thermal_control_strategies import create_controller
from pack import BatteryPack
from thermal_batch_config import (
    AGC_DATA_FILE,
    AMBIENT_TEMP_K,
    INITIAL_TEMP_C,
    NEW_ROOT,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
    SIM_DT,
    T_DIFF_LIMIT_C,
    TARGET_TEMP_C,
    compressor_power_scale_for,
    initial_soc_for,
    pid_params_for_scene,
    pid_target_temp_for_scene,
)
from thermal_loop import (
    SupervisoryFlowController,
    build_pack_config,
    initialize_refrigeration_dynamic_state,
    maybe_reverse_flow,
    simulate_thermal_loop_step,
)


def _emit(log_func, message):
    if log_func is not None:
        log_func(message)
    else:
        print(message, flush=True)


def load_current_profile(scene, times, agc_data_file=AGC_DATA_FILE):
    scene_key = str(scene).strip().lower()
    if scene_key in ("peak", "调峰"):
        return np.full(len(times), 560.0)
    if scene_key not in ("freq", "frequency", "reg", "调频"):
        raise ValueError(f"Unknown current-profile scene: {scene!r}")

    agc_df = pd.read_csv(agc_data_file)
    agc_time = agc_df["Seconds"].to_numpy() if "Seconds" in agc_df.columns else agc_df.iloc[:, 0].to_numpy()
    agc_signal = agc_df["RegD"].to_numpy() if "RegD" in agc_df.columns else agc_df.iloc[:, 1].to_numpy()
    return np.interp(times, agc_time, agc_signal * 1120.0, left=0.0, right=0.0)


def make_snapshot_file(snapshots, out_path):
    matrix_output = []
    for snapshot in snapshots:
        temp_grid = snapshot["temps"].reshape(snapshot["rows"], snapshot["cols"])
        matrix_output.append([f"Time: {snapshot['time']:.1f}s"] + [""] * snapshot["cols"])
        matrix_output.append([""] + [f"C{c + 1}" for c in range(snapshot["cols"])])
        for r in range(snapshot["rows"]):
            matrix_output.append([f"R{r + 1}"] + list(temp_grid[r]))
        matrix_output.append([""] * (snapshot["cols"] + 1))
    pd.DataFrame(matrix_output).to_csv(out_path, index=False, header=False, encoding="utf-8-sig")


def _should_skip_existing(out_csv):
    if not out_csv.exists():
        return False
    try:
        return len(pd.read_csv(out_csv, encoding="utf-8-sig")) > 0
    except Exception:
        return False


def _time_column(df):
    for name in ("Time (s)", "Time"):
        if name in df.columns:
            return name
    raise KeyError("source CSV must contain either 'Time (s)' or 'Time'")


def flow_reversal_enabled(flow):
    return str(flow).strip().lower() in {"双向", "反向", "double", "bidirectional", "reversed"}



def _fallback_time_frame(scene, dt):
    stop_time = 3600.0 if scene == "freq" else 6400.0
    return pd.DataFrame({"Time": np.arange(0.0, stop_time, dt)})


OUTPUT_COLUMN_RENAMES = {
    "Time (s)": "Time",
    "Battery Temp (C)": "Average temperature",
    "Max_Delta_T": "Maximum temperature difference",
    "Coolant Temp (C)": "Coolant temperature",
    "T_Evap_Out_C": "Evaporator outlet coolant temperature",
    "T_Pipe_Supply_C": "Supply pipe coolant temperature",
    "T_Plate_In_C": "Cold plate inlet coolant temperature",
    "T_Plate_Out_C": "Cold plate outlet coolant temperature",
    "T_Pipe_Return_C": "Return pipe coolant temperature",
    "T_Tank_In_C": "Tank inlet coolant temperature",
    "Total Current (A)": "Total current",
    "Compressor Command (RPM)": "Compressor command",
    "Compressor Speed (RPM)": "Compressor Speed",
    "Fan Speed (RPM)": "Fan Speed",
    "Pump Command (RPM)": "Pump command",
    "Pump Speed (RPM)": "Pump Speed (RPM)",
    "OnOff_State": "On-off state",
    "OnOff_T_Avg_C": "On-off average temperature (C)",
    "OnOff_N_Comp_Cmd_RPM": "On-off compressor command (RPM)",
    "OnOff_N_Pump_Cmd_RPM": "On-off pump command (RPM)",
    "Total Power (kW)": "Total power",
    "Cumulative Energy (kWh)": "Cumulative energy consumption",
    "SOH Avg (%)": "SOH",
    "Flow Direction d": "Flow direction d",
    "Current_Range (A)": "Branch current range",
    "Current_Std_Dev (A)": "Branch current standard deviation",
    "MPC_Flow_Mode": "MPC flow mode",
    "Delta_T_Pred_Max": "Predicted maximum temperature difference",
    "H_Down_Pred_Max": "Predicted downstream heat accumulation",
    "Gamma_Hot_Pred_Max": "Predicted hotspot distribution coefficient",
    "J_Rev_Pred_Max": "Predicted reversal index",
    "Flow_Switched": "Flow switched",
    "Result_Tag": "Result tag",
    "MPC_Direction_d": "MPC direction d",
    "Pre_Switch_Delta_T_Pred_Max": "Pre-switch predicted maximum temperature difference",
    "Pre_Switch_Delta_T_Pred_Buffer_Max": "Pre-switch predicted maximum temperature difference in buffer",
    "Pre_Switch_Delta_T_Pred_Cross_S": "Pre-switch predicted delta-T crossing time",
    "Pre_Switch_J_Rev_Pred_Max": "Pre-switch predicted reversal index",
    "Candidate_Direction_d": "Candidate direction d",
    "Candidate_Delta_T_Pred_Max": "Candidate predicted maximum temperature difference",
    "Candidate_H_Down_Pred_Max": "Candidate predicted downstream heat accumulation",
    "Candidate_Gamma_Hot_Pred_Max": "Candidate predicted hotspot distribution coefficient",
    "Candidate_J_Rev_Pred_Max": "Candidate predicted reversal index",
    "Candidate_Reverse_Benefits": "Candidate reverse benefits",
    "Candidate_Predictive_Switch_Gate": "Candidate predictive switch gate",
    "Actual_Delta_T_C": "Actual maximum temperature difference",
    "Actual_Switch_Gate": "Actual switch gate",
    "Predictive_Switch_Gate": "Predictive switch gate",
    "Hold_Time_Satisfied": "Minimum hold time satisfied",
    "Min_Actual_Delta_T_Switch_C": "Minimum actual delta-T for reversal",
    "Pred_Delta_T_Switch_C": "Predictive delta-T switch threshold",
    "Pred_Switch_Buffer_S": "Predictive switch buffer",
    "Min_Delta_T_Improvement_C": "Minimum reversal delta-T improvement",
    "Max_Delta_T_Worsening_C": "Maximum allowed reversal delta-T worsening",
    "Min_J_Rev_Improvement": "Minimum reversal index improvement",
    "MPC_Target_Temp_C": "MPC dynamic target temperature",
    "MPC_Solve_Time_S": "MPC solve time",
    "MPC_T_Bat_Pred_1_C": "MPC predicted battery temperature +1 step",
    "MPC_T_Bat_Pred_5_C": "MPC predicted battery temperature +5 steps",
    "MPC_T_Bat_Pred_10_C": "MPC predicted battery temperature +10 steps",
    "MPC_T_Bat_Pred_20_C": "MPC predicted battery temperature +20 steps",
    "MPC_T_Bat_Pred_End_C": "MPC predicted battery temperature horizon end",
    "MPC_T_Cool_Pred_End_C": "MPC predicted coolant temperature horizon end",
    "MPC_T_Plate_Pred_End_C": "MPC predicted plate temperature horizon end",
    "MPC_Q_Evap_Eff_Pred_Mean_W": "MPC predicted mean evaporator cooling power",
    "J_track": "J_track",
    "J_spread": "J_spread",
    "J_comp": "J_comp",
    "J_pump": "J_pump",
    "J_switch": "J_switch",
    "J_total": "J_total",
    "Raw_Track_Error_Sq": "Raw tracking error square",
    "Raw_Sigma_T_Sq": "Raw temperature variance",
    "Raw_W_Comp": "Raw compressor power",
    "Raw_W_Pump": "Raw pump power",
    "Raw_J_Switch": "Raw switching penalty",
    "Weighted_J_track": "Weighted J_track",
    "Weighted_J_spread": "Weighted J_spread",
    "Weighted_J_comp": "Weighted J_comp",
    "Weighted_J_pump": "Weighted J_pump",
    "Weighted_J_switch": "Weighted J_switch",
    "COP_System": "System COP",
    "Exergy_Efficiency": "Exergy efficiency",
    "E_D_Comp": "Compressor exergy destruction rate",
    "E_D_Evap": "Evaporator exergy destruction rate",
    "E_D_Cond": "Condenser exergy destruction rate",
    "E_D_Exp": "Expansion valve exergy destruction rate",
    "E_D_Total": "Total exergy destruction rate",
    "E_D_Comp_Ratio": "Compressor exergy destruction ratio",
    "E_D_Evap_Ratio": "Evaporator exergy destruction ratio",
    "E_D_Cond_Ratio": "Condenser exergy destruction ratio",
    "E_D_Exp_Ratio": "Expansion valve exergy destruction ratio",
    "Q_Dot_Bat": "Battery heat removal rate (kW)",
    "Q_Dot_Evap": "Evaporator cooling rate (kW)",
    "Q_Dot_Cond": "Condenser heat rejection rate (kW)",
    "T_Evap_Sat": "Evaporating saturation temperature (C)",
    "T_Cond_Sat": "Condensing saturation temperature (C)",
    "P_Evap": "Evaporating pressure (kPa)",
    "P_Cond": "Condensing pressure (kPa)",
    "h_1": "State 1 enthalpy (kJ/kg)",
    "h_2": "State 2 enthalpy (kJ/kg)",
    "h_3": "State 3 enthalpy (kJ/kg)",
    "h_4": "State 4 enthalpy (kJ/kg)",
    "p_1": "State 1 pressure (kPa)",
    "p_2": "State 2 pressure (kPa)",
    "p_3": "State 3 pressure (kPa)",
    "p_4": "State 4 pressure (kPa)",
    "P_Comp": "Compressor power (kW)",
    "P_Pump": "Pump power (kW)",
    "P_Fan": "Fan power (kW)",
    "M_Dot_Cool": "Coolant mass flow rate (kg/s)",
    "Coolant_Flow_L_Min": "Coolant flow rate (L/min)",
    "Ex_Dot_In": "Exergy input rate (kW)",
    "Ex_Dot_Useful": "Useful cooling exergy rate (kW)",
    "Ex_Dot_Dest_Comp": "Compressor exergy destruction rate (kW)",
    "Ex_Dot_Dest_Cond": "Condenser exergy destruction rate (kW)",
    "Ex_Dot_Dest_Evap": "Evaporator exergy destruction rate (kW)",
    "Ex_Dot_Dest_Expansion": "Expansion valve exergy destruction rate (kW)",
    "Ex_Dot_Dest_Aux": "Auxiliary exergy destruction rate (kW)",
    "Ex_Dot_Loss_Ambient": "Exergy loss to ambient rate (kW)",
    "Current_1 (A)": "Branch current 1",
    "Current_2 (A)": "Branch current 2",
    "Current_3 (A)": "Branch current 3",
    "Current_4 (A)": "Branch current 4",
}


OUTPUT_COLUMNS_TO_DROP = {
    "J_track",
    "J_spread",
    "J_comp",
    "J_pump",
    "J_switch",
    "J_total",
    "Raw_Track_Error_Sq",
    "Raw_Sigma_T_Sq",
    "Raw_W_Comp",
    "Raw_W_Pump",
    "Raw_J_Switch",
    "Weighted_J_track",
    "Weighted_J_spread",
    "Weighted_J_comp",
    "Weighted_J_pump",
    "Weighted_J_switch",
}


def _output_dataframe(history):
    return (
        pd.DataFrame(history)
        .drop(columns=list(OUTPUT_COLUMNS_TO_DROP), errors="ignore")
        .rename(columns=OUTPUT_COLUMN_RENAMES)
    )



def simulate_case(
    control,
    scene,
    flow,
    source_csv,
    main_name,
    snap_name,
    *,
    output_root=NEW_ROOT,
    agc_data_file=AGC_DATA_FILE,
    dt=SIM_DT,
    target_temp_c=TARGET_TEMP_C,
    temp_diff_limit_c=T_DIFF_LIMIT_C,
    mpc_flow_mode="switching",
    force=False,
    max_steps=None,
    start_time_s=None,
    duration_s=None,
    result_tag=None,
    current_profile_override=None,
    pid_params=None,
    log_func=None,
):
    output_root = Path(output_root)
    out_dir = output_root / control
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / main_name
    snap_csv = out_dir / snap_name
    partial_csv = out_dir / f"{Path(main_name).stem}.partial.csv"

    if not force and _should_skip_existing(out_csv):
        _emit(log_func, f"SKIP existing {out_csv}")
        return {"out_csv": out_csv, "snap_csv": snap_csv, "skipped": True}
    if partial_csv.exists():
        partial_csv.unlink()

    source_csv = Path(source_csv)
    if source_csv.exists():
        old_df = pd.read_csv(source_csv, encoding="utf-8-sig")
    else:
        _emit(log_func, f"WARNING missing source CSV; generated fallback time axis: {source_csv}")
        old_df = _fallback_time_frame(scene, dt)
    time_col = _time_column(old_df)
    if scene == "freq":
        old_df = old_df.loc[old_df[time_col] < 3600.0].copy()
    if start_time_s is not None:
        old_df = old_df.loc[old_df[time_col] >= float(start_time_s)].copy()
    if duration_s is not None:
        if start_time_s is None:
            window_start_s = float(old_df[time_col].iloc[0]) if len(old_df) else 0.0
        else:
            window_start_s = float(start_time_s)
        old_df = old_df.loc[old_df[time_col] < window_start_s + float(duration_s)].copy()
    if max_steps is not None:
        old_df = old_df.head(max_steps).copy()

    n_steps = len(old_df)
    if n_steps == 0:
        raise ValueError(f"No rows to simulate for {control} {scene} {flow}: {source_csv}")

    times = old_df[time_col].to_numpy(dtype=float)
    if current_profile_override is None:
        current_profile = load_current_profile(scene, times, agc_data_file=agc_data_file)
    else:
        current_profile = np.asarray(current_profile_override, dtype=float)
        if current_profile.size != n_steps:
            raise ValueError("current_profile_override length must match simulation steps")
    snapshot_indices = {0, int(n_steps / 3), int(2 * n_steps / 3), n_steps - 1}

    pack_config = build_pack_config(
        total_current=560.0,
        initial_soc=initial_soc_for(scene),
        initial_temp_c=INITIAL_TEMP_C,
    )
    pack = BatteryPack(pack_config)
    aging_observer = AgingModel280Ah(initial_cycle_age=np.zeros(pack.Ns), capacity_ah=pack_config["capacity"])
    controller_target_temp_c = target_temp_c
    controller_pid_params = pid_params
    if control == "pid":
        controller_target_temp_c = pid_target_temp_for_scene(scene) if target_temp_c == TARGET_TEMP_C else target_temp_c
        controller_pid_params = pid_params_for_scene(scene) if pid_params is None else pid_params
    controller = create_controller(
        control,
        current_profile,
        dt=dt,
        target_temp_c=controller_target_temp_c,
        mpc_flow_mode=mpc_flow_mode,
        case_name=f"{scene} {flow} {main_name}",
        pid_params=controller_pid_params,
    )
    mpc_params = getattr(controller, "mpc_params", None)
    if mpc_params is not None:
        _emit(log_func, f"MPC mode: {mpc_params.name}")
        _emit(log_func, f"dynamic_target_min: {mpc_params.dynamic_target_min}")
        _emit(log_func, f"precool_max: {mpc_params.precool_max}")
        _emit(log_func, f"w_cold_temp: {mpc_params.w_cold_temp}")
        _emit(log_func, f"J_rev_on: {mpc_params.j_rev_on}")
        _emit(log_func, f"dmax_comp: {mpc_params.dmax_comp}")
        _emit(log_func, f"dmax_pump: {mpc_params.dmax_pump}")
    flow_supervisor = None
    if mpc_flow_mode == "supervised" and not getattr(controller, "owns_flow_direction", False):
        flow_supervisor = SupervisoryFlowController(dt=dt)

    t_tank_k = AMBIENT_TEMP_K
    t_plate_k_array = np.full(pack.cols, AMBIENT_TEMP_K)
    c_plate_node = PLATE_NODE_HEAT_CAPACITY_TOTAL / pack.cols
    is_reversed = False
    last_reverse_t = -999.0
    reverse_enabled = flow_reversal_enabled(flow)
    total_accumulated_kwh = 0.0
    if mpc_params is not None:
        initial_n_comp = getattr(controller, "last_n_comp", 0.0)
        initial_n_pump = getattr(controller, "last_n_pump", 0.0)
        refrigeration_dynamic_state = initialize_refrigeration_dynamic_state(initial_n_comp, initial_n_pump)
    else:
        refrigeration_dynamic_state = None
    history = []
    snapshots = []
    start = time.time()

    _emit(log_func, f"START {control} {scene} {flow} steps={n_steps} soc0={pack_config['initial_soc']}")
    for step_no, _row in enumerate(old_df.itertuples(index=False)):
        t = times[step_no]
        t_outdoor = AMBIENT_TEMP_K
        t_cabinet = AMBIENT_TEMP_K
        pack.total_current = float(current_profile[step_no])

        controller_owns_flow = getattr(controller, "owns_flow_direction", False)
        if controller_owns_flow:
            delta_t = float(np.max(pack.temps) - np.min(pack.temps))
        elif flow_supervisor is not None:
            temps_grid_c = pack.temps.reshape(pack.rows, pack.cols) - 273.15
            flow_supervisor.update(
                temps_c=temps_grid_c,
                current_time=t,
                flow_enabled=reverse_enabled,
            )
            is_reversed = flow_supervisor.is_reversed
            last_reverse_t = flow_supervisor.last_switch_time
            delta_t = float(flow_supervisor.last_flow_info["delta_t_pred_max"])
        else:
            delta_t, is_reversed, last_reverse_t = maybe_reverse_flow(
                temps=pack.temps,
                is_reversed=is_reversed,
                current_time=t,
                last_reverse_time=last_reverse_t,
                reverse_flow_enabled=reverse_enabled,
                temp_diff_limit=temp_diff_limit_c,
            )

        n_comp_used, n_pump_used = controller.command(
            step_no,
            pack,
            t_tank_k,
            t_cabinet,
            current_time=t,
            flow_enabled=reverse_enabled,
            is_reversed=is_reversed,
            thermal_state=refrigeration_dynamic_state,
            plate_temps=t_plate_k_array,
        )
        if controller_owns_flow:
            is_reversed = controller.is_reversed
            last_reverse_t = getattr(controller, "last_switch_time", last_reverse_t)
        onoff_status = getattr(controller, "last_status", {}) if control == "on-off" else {}

        thermal_step = simulate_thermal_loop_step(
            pack=pack,
            T_tank_K=t_tank_k,
            T_plate_K_array=t_plate_k_array,
            N_comp_cmd=n_comp_used,
            N_pump_cmd=n_pump_used,
            T_outdoor=t_outdoor,
            dt=dt,
            is_reversed=is_reversed,
            C_plate_node=c_plate_node,
            compressor_power_scale=compressor_power_scale_for(control),            dynamic_state=refrigeration_dynamic_state,
        )
        refrigeration_dynamic_state = thermal_step.get("dynamic_state", refrigeration_dynamic_state)
        t_tank_k = thermal_step["T_tank_K"]
        t_plate_k_array = thermal_step["T_plate_K_array"]
        pack.step(dt, T_plate=t_plate_k_array, T_cabinet=t_cabinet)
        pack.history = [[] for _ in range(pack.Ns)]
        controller.update_after_step(pack)
        if flow_supervisor is not None:
            flow_supervisor.update_after_step(pack.temps.reshape(pack.rows, pack.cols) - 273.15)

        branch_currents = [pack.branch_currents_history[k][-1] for k in range(pack.rows)]
        t_batt_curr_k = pack.get_avg_temp()
        temps_cell_c = pack.temps - 273.15
        t_cell_mean_c = float(np.mean(temps_cell_c))
        t_cell_max_c = float(np.max(temps_cell_c))
        t_cell_min_c = float(np.min(temps_cell_c))
        delta_t_cell_c = t_cell_max_c - t_cell_min_c
        soh_percent = aging_observer.step(np.repeat(branch_currents, pack.cols), pack.temps - 273.15, dt)
        current_power_kw = (
            thermal_step["W_comp_real"] + thermal_step["W_pump_val"] + thermal_step["W_fan_real"]
        ) / 1000.0
        total_accumulated_kwh += current_power_kw * (dt / 3600.0)

        record = {
            "Time (s)": t,
            "Result_Tag": result_tag,
            # Pack-average temperature, equivalent to T_cell_mean_C; not the hottest cell.
            "Battery Temp (C)": t_batt_curr_k - 273.15,
            "T_cell_mean_C": t_cell_mean_c,
            "T_cell_max_C": t_cell_max_c,
            "T_cell_min_C": t_cell_min_c,
            "Delta_T_cell_C": delta_t_cell_c,
            "Max_Delta_T": delta_t,
            "Coolant Temp (C)": t_tank_k - 273.15,
            "T_Evap_Out_C": thermal_step.get("T_evap_out_K", np.nan) - 273.15,
            "T_Pipe_Supply_C": thermal_step.get("T_pipe_supply_K", np.nan) - 273.15,
            "T_Plate_In_C": thermal_step.get("T_plate_in_K", np.nan) - 273.15,
            "T_Plate_Out_C": thermal_step.get("T_plate_out_K", np.nan) - 273.15,
            "T_Pipe_Return_C": thermal_step.get("T_pipe_return_K", np.nan) - 273.15,
            "T_Tank_In_C": thermal_step.get("T_tank_in_K", np.nan) - 273.15,
            "Total Current (A)": pack.total_current,
            "Compressor Command (RPM)": n_comp_used,
            "Compressor Speed (RPM)": thermal_step.get("N_comp_eff", n_comp_used),
            "Fan Speed (RPM)": thermal_step.get("N_fan_eff", np.nan),
            "Pump Command (RPM)": n_pump_used,
            "Pump Speed (RPM)": thermal_step.get("N_pump_eff", n_pump_used),
            "Total Power (kW)": current_power_kw,
            "Cumulative Energy (kWh)": total_accumulated_kwh,
            "SOH Avg (%)": float(np.mean(soh_percent)),
            "Flow Direction d": -1 if is_reversed else 1,
            "Current_Range (A)": float(np.max(branch_currents) - np.min(branch_currents)),
            "Current_Std_Dev (A)": float(np.std(branch_currents)),
            "COP_System": thermal_step.get("COP_system", np.nan),
            "Exergy_Efficiency": thermal_step.get("exergy_efficiency", np.nan),
            "E_D_Comp": thermal_step.get("E_D_comp", np.nan),
            "E_D_Evap": thermal_step.get("E_D_evap", np.nan),
            "E_D_Cond": thermal_step.get("E_D_cond", np.nan),
            "E_D_Exp": thermal_step.get("E_D_exp", np.nan),
            "E_D_Total": thermal_step.get("E_D_total", np.nan),
            "E_D_Comp_Ratio": thermal_step.get("E_D_comp_ratio", np.nan),
            "E_D_Evap_Ratio": thermal_step.get("E_D_evap_ratio", np.nan),
            "E_D_Cond_Ratio": thermal_step.get("E_D_cond_ratio", np.nan),
            "E_D_Exp_Ratio": thermal_step.get("E_D_exp_ratio", np.nan),
            "Q_Dot_Bat": thermal_step.get("Q_dot_bat", np.nan) / 1000.0,
            "Q_Dot_Evap": thermal_step.get("Q_dot_evap", np.nan) / 1000.0,
            "Q_Dot_Cond": thermal_step.get("Q_dot_cond", np.nan) / 1000.0,
            "T_Evap_Sat": thermal_step.get("T_evap_sat", np.nan) - 273.15,
            "T_Cond_Sat": thermal_step.get("T_cond_sat", np.nan) - 273.15,
            "P_Evap": thermal_step.get("p_evap", np.nan) / 1000.0,
            "P_Cond": thermal_step.get("p_cond", np.nan) / 1000.0,
            "h_1": thermal_step.get("h_1", np.nan) / 1000.0,
            "h_2": thermal_step.get("h_2", np.nan) / 1000.0,
            "h_3": thermal_step.get("h_3", np.nan) / 1000.0,
            "h_4": thermal_step.get("h_4", np.nan) / 1000.0,
            "p_1": thermal_step.get("p_1", np.nan) / 1000.0,
            "p_2": thermal_step.get("p_2", np.nan) / 1000.0,
            "p_3": thermal_step.get("p_3", np.nan) / 1000.0,
            "p_4": thermal_step.get("p_4", np.nan) / 1000.0,
            "P_Comp": thermal_step.get("W_comp_real", np.nan) / 1000.0,
            "P_Pump": thermal_step.get("W_pump_val", np.nan) / 1000.0,
            "P_Fan": thermal_step.get("W_fan_real", np.nan) / 1000.0,
            "M_Dot_Cool": thermal_step.get("m_dot_cool", np.nan),
            "Coolant_Flow_L_Min": thermal_step.get("coolant_flow_L_min", np.nan),
            "Ex_Dot_In": thermal_step.get("Ex_dot_in", np.nan) / 1000.0,
            "Ex_Dot_Useful": thermal_step.get("Ex_dot_useful", np.nan) / 1000.0,
            "Ex_Dot_Dest_Comp": thermal_step.get("E_D_comp", np.nan) / 1000.0,
            "Ex_Dot_Dest_Cond": thermal_step.get("E_D_cond", np.nan) / 1000.0,
            "Ex_Dot_Dest_Evap": thermal_step.get("E_D_evap", np.nan) / 1000.0,
            "Ex_Dot_Dest_Expansion": thermal_step.get("E_D_exp", np.nan) / 1000.0,
            "Ex_Dot_Dest_Aux": thermal_step.get("Ex_dot_dest_aux", np.nan) / 1000.0,
            "Ex_Dot_Loss_Ambient": thermal_step.get("Ex_dot_loss_ambient", np.nan) / 1000.0,
        }
        if onoff_status:
            record["OnOff_State"] = onoff_status.get("state", np.nan)
            record["OnOff_T_Avg_C"] = onoff_status.get("T_avg_C", np.nan)
            record["OnOff_N_Comp_Cmd_RPM"] = onoff_status.get("N_comp_cmd_rpm", np.nan)
            record["OnOff_N_Pump_Cmd_RPM"] = onoff_status.get("N_pump_cmd_rpm", np.nan)
        controller_flow_info = getattr(controller, "last_flow_info", {})
        flow_info = controller_flow_info
        if flow_supervisor is not None:
            flow_info = flow_supervisor.last_flow_info
        if flow_info:
            record["MPC_Flow_Mode"] = flow_info.get("mode", "")
            record["Delta_T_Pred_Max"] = flow_info.get("delta_t_pred_max", np.nan)
            record["H_Down_Pred_Max"] = flow_info.get("h_down_pred_max", np.nan)
            record["Gamma_Hot_Pred_Max"] = flow_info.get("gamma_hot_pred_max", np.nan)
            record["J_Rev_Pred_Max"] = flow_info.get("j_rev_pred_max", np.nan)
            record["Flow_Switched"] = int(flow_info.get("switched", False))
            record["MPC_Direction_d"] = flow_info.get("flow_direction_d", record["Flow Direction d"])
            record["Pre_Switch_Delta_T_Pred_Max"] = flow_info.get("pre_switch_delta_t_pred_max", np.nan)
            record["Pre_Switch_Delta_T_Pred_Buffer_Max"] = flow_info.get("pre_switch_delta_t_pred_buffer_max", np.nan)
            record["Pre_Switch_Delta_T_Pred_Cross_S"] = flow_info.get("pre_switch_delta_t_pred_cross_s", np.nan)
            record["Pre_Switch_J_Rev_Pred_Max"] = flow_info.get("pre_switch_j_rev_pred_max", np.nan)
            record["Candidate_Direction_d"] = flow_info.get("candidate_direction", np.nan)
            record["Candidate_Delta_T_Pred_Max"] = flow_info.get("candidate_delta_t_pred_max", np.nan)
            record["Candidate_H_Down_Pred_Max"] = flow_info.get("candidate_h_down_pred_max", np.nan)
            record["Candidate_Gamma_Hot_Pred_Max"] = flow_info.get("candidate_gamma_hot_pred_max", np.nan)
            record["Candidate_J_Rev_Pred_Max"] = flow_info.get("candidate_j_rev_pred_max", np.nan)
            record["Candidate_Reverse_Benefits"] = flow_info.get("candidate_reverse_benefits", np.nan)
            record["Candidate_Predictive_Switch_Gate"] = flow_info.get("candidate_predictive_switch_gate", np.nan)
            record["Actual_Delta_T_C"] = flow_info.get("actual_delta_t_c", np.nan)
            record["Actual_Switch_Gate"] = flow_info.get("actual_switch_gate", np.nan)
            record["Predictive_Switch_Gate"] = flow_info.get("predictive_switch_gate", np.nan)
            record["Hold_Time_Satisfied"] = flow_info.get("hold_time_satisfied", np.nan)
            record["Min_Actual_Delta_T_Switch_C"] = flow_info.get("min_actual_delta_t_switch_c", np.nan)
            record["Pred_Delta_T_Switch_C"] = flow_info.get("pred_delta_t_switch_c", np.nan)
            record["Pred_Switch_Buffer_S"] = flow_info.get("pred_switch_buffer_s", np.nan)
            record["Min_Delta_T_Improvement_C"] = flow_info.get("min_delta_t_improvement_c", np.nan)
            record["Max_Delta_T_Worsening_C"] = flow_info.get("max_delta_t_worsening_c", np.nan)
            record["Min_J_Rev_Improvement"] = flow_info.get("min_j_rev_improvement", np.nan)
            record["MPC_Target_Temp_C"] = flow_info.get("target_temp_c", np.nan)
            record["MPC_Solve_Time_S"] = flow_info.get("solve_time_s", np.nan)
            record["MPC_Solved"] = bool(flow_info.get("solved", False))
            record["MPC_T_Bat_Pred_1_C"] = flow_info.get("t_batt_pred_1_c", np.nan)
            record["MPC_T_Bat_Pred_5_C"] = flow_info.get("t_batt_pred_5_c", np.nan)
            record["MPC_T_Bat_Pred_10_C"] = flow_info.get("t_batt_pred_10_c", np.nan)
            record["MPC_T_Bat_Pred_20_C"] = flow_info.get("t_batt_pred_20_c", np.nan)
            record["MPC_T_Bat_Pred_End_C"] = flow_info.get("t_batt_pred_end_c", np.nan)
            record["MPC_T_Cool_Pred_End_C"] = flow_info.get("t_cool_pred_end_c", np.nan)
            record["MPC_T_Plate_Pred_End_C"] = flow_info.get("t_plate_pred_end_c", np.nan)
            record["MPC_Q_Evap_Eff_Pred_Mean_W"] = flow_info.get("qevap_eff_pred_mean_w", np.nan)
            record["MPC_Q_Evap_Cmd_Pred_1_W"] = controller_flow_info.get("qevap_cmd_pred_1_w", np.nan)
            record["MPC_Q_Cond_Pred_1_W"] = controller_flow_info.get("qcond_pred_1_w", np.nan)
            record["MPC_Q_Evap_Pred_1_W"] = controller_flow_info.get("qevap_pred_1_w", np.nan)
            record["Terminal_Cost_Enabled"] = bool(flow_info.get("Terminal_Cost_Enabled", flow_info.get("terminal_cost_enabled", False)))
            record["Terminal_Cost_Type"] = flow_info.get("Terminal_Cost_Type", flow_info.get("terminal_cost_type", ""))
            record["W_Terminal_Temp"] = flow_info.get("W_Terminal_Temp", flow_info.get("w_terminal_temp", np.nan))
            record["Terminal_Temp_Scale_C"] = flow_info.get("Terminal_Temp_Scale_C", flow_info.get("terminal_temp_scale_c", np.nan))
            record["T_pred_end_C"] = flow_info.get("T_pred_end_C", flow_info.get("t_pred_end_c", np.nan))
            record["J_terminal"] = flow_info.get("J_terminal", flow_info.get("j_terminal", np.nan))
            record["Weighted_J_terminal"] = flow_info.get("Weighted_J_terminal", flow_info.get("weighted_j_terminal", np.nan))
            record["J_track"] = flow_info.get("j_track", np.nan)
            record["J_spread"] = flow_info.get("j_spread", np.nan)
            record["J_comp"] = flow_info.get("j_comp", np.nan)
            record["J_pump"] = flow_info.get("j_pump", np.nan)
            record["J_switch"] = flow_info.get("j_switch", np.nan)
            record["J_total"] = flow_info.get("j_total", np.nan)
            record["Raw_Track_Error_Sq"] = flow_info.get("raw_track_error_sq", np.nan)
            record["Raw_Sigma_T_Sq"] = flow_info.get("raw_sigma_t_sq", np.nan)
            record["Raw_W_Comp"] = flow_info.get("raw_w_comp", np.nan)
            record["Raw_W_Pump"] = flow_info.get("raw_w_pump", np.nan)
            record["Raw_J_Switch"] = flow_info.get("raw_j_switch", np.nan)
            record["Weighted_J_track"] = flow_info.get("weighted_j_track", np.nan)
            record["Weighted_J_spread"] = flow_info.get("weighted_j_spread", np.nan)
            record["Weighted_J_comp"] = flow_info.get("weighted_j_comp", np.nan)
            record["Weighted_J_pump"] = flow_info.get("weighted_j_pump", np.nan)
            record["Weighted_J_switch"] = flow_info.get("weighted_j_switch", np.nan)
        for j, current in enumerate(branch_currents):
            record[f"Current_{j + 1} (A)"] = current
        history.append(record)

        if step_no in snapshot_indices:
            snapshots.append(
                {
                    "time": t,
                    "temps": (pack.temps - 273.15).copy(),
                    "rows": pack.rows,
                    "cols": pack.cols,
                }
            )

        if (step_no + 1) % 100 == 0 or step_no == n_steps - 1:
            _output_dataframe(history).to_csv(partial_csv, index=False, encoding="utf-8-sig")
            elapsed = time.time() - start
            onoff_log = ""
            if onoff_status:
                onoff_log = (
                    f" onoff_state={onoff_status.get('state', np.nan)}"
                    f" T_avg={onoff_status.get('T_avg_C', np.nan):.3f}C"
                    f" Ncomp={onoff_status.get('N_comp_cmd_rpm', np.nan):.0f}rpm"
                    f" Npump={onoff_status.get('N_pump_cmd_rpm', np.nan):.0f}rpm"
                )
            _emit(
                log_func,
                f"PROGRESS {control} {scene} {flow} {step_no + 1}/{n_steps} "
                f"t={t:.0f}s elapsed={elapsed:.1f}s T={t_batt_curr_k - 273.15:.3f}C "
                f"E={total_accumulated_kwh:.4f}kWh{onoff_log}",
            )

    _output_dataframe(history).to_csv(out_csv, index=False, encoding="utf-8-sig")
    make_snapshot_file(snapshots, snap_csv)
    if partial_csv.exists():
        partial_csv.unlink()
    _emit(log_func, f"DONE {control} {scene} {flow}: {out_csv}")
    return {"out_csv": out_csv, "snap_csv": snap_csv, "skipped": False}




