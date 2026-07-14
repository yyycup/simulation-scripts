import os
from pathlib import Path
from typing import NamedTuple

import platformdirs


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

WORKSPACE = Path(__file__).resolve().parent
platformdirs.user_config_dir = lambda *args, **kwargs: str(WORKSPACE / "outputs" / "pybamm_config")

OLD_ROOT = Path(
    os.environ.get(
        "THERMAL_OLD_ROOT",
        r"C:\Users\24776\Desktop\科研\论文\小论文\仿真数据输出\控制工况数据",
    )
)
NEW_ROOT = Path(
    os.environ.get(
        "THERMAL_NEW_ROOT",
        r"C:\Users\24776\Desktop\科研\论文\小论文\仿真数据输出\归一化代价函数结果",
    )
)
AGC_DATA_FILE = Path(
    os.environ.get(
        "THERMAL_AGC_DATA_FILE",
        r"C:\Users\24776\Desktop\科研\论文\小论文\数据\PJM_RegD_MaxLoad_2h.csv",
    )
)

SIM_DT = 5.0
TARGET_TEMP_C = 25.0
PEAK_PID_TARGET_TEMP_C = 25.0
FREQ_PID_TARGET_TEMP_C = 25.1
AMBIENT_TEMP_C = 35.0
AMBIENT_TEMP_K = AMBIENT_TEMP_C + 273.15
T_DIFF_LIMIT_C = 0.5
N_COMP_MIN_RPM = 2000.0
PID_N_COMP_MIN_RPM = 1000.0
N_COMP_MAX_RPM = 6000.0
N_COMP_OFF_RPM = 300.0
MPC_N_COMP_MIN_RPM = N_COMP_MIN_RPM
N_PUMP_MIN_RPM = 1600.0
N_PUMP_MAX_RPM = 4800.0
# Evaporator baseline for comparison: UA=1.0, flow exponent=0.8, cap=1.0.
EVAP_UA_FACTOR = 2.0
EVAP_FLOW_EXP = 1.2
EVAP_CAP_FACTOR = 1.5
MPC_FLOW_MODE = "switching"
MPC_DELTA_T_ON_C = 0.5
MPC_DELTA_T_OFF_C = 0.3
MPC_MIN_HOLD_TIME_S = 200.0
MPC_REV_DELTA_T_LIM_C = MPC_DELTA_T_ON_C
MPC_REV_H_DOWN_CELL_SCALE_C = 0.80
MPC_REV_GAMMA_HOT_LIM = 0.25
MPC_REV_J_ON = 1.85
MPC_REV_J_OFF = 1.50
MPC_REV_W_DELTA_T = 1.0
MPC_REV_W_H_DOWN = 0.8
MPC_REV_W_GAMMA_HOT = 0.2
MPC_PRED_DELTA_T_SWITCH_C = 0.45
MPC_PRED_SWITCH_BUFFER_S = 100.0
MPC_T_TRACK_SCALE_C = 1.0
MPC_T_SPREAD_SCALE_C = 0.3
MPC_CV_WSP = 5e8
MPC_CV_TAU = 10.0
MPC_CV_BAND_C = 0.5
MPC_Q_TRACK = 100000.0
MPC_Q_SPREAD = 5.0
MPC_R_COMP = 300.0
MPC_R_PUMP = 10000.0
MPC_R_SWITCH = 0.5
MPC_U_NCOMP_INIT = 1500.0
MPC_U_NPUMP_INIT = 3000.0
MPC_U_NCOMP_DMAX = 300.0
MPC_U_NPUMP_DMAX = 300.0
MPC_U_NCOMP_DCOST = 0.001
MPC_U_NPUMP_DCOST = 0.001
MPC_SWITCH_PENALTY = MPC_R_SWITCH
MPC_HEAT_PREVIEW_WINDOW_S = 750.0
MPC_HEAT_PREVIEW_ENABLED = True
MPC_HEAT_PREVIEW_NOMINAL_W = ((560.0 / 4.0) ** 2) * 0.001 * 52.0
MPC_HEAT_PREVIEW_QUANTILE = 0.90
MPC_HEAT_PREVIEW_QUANTILE_WEIGHT = 0.70
MPC_PRECOOL_MAX_C = 0.4
MPC_WARM_RELIEF_MAX_C = 0.2
MPC_DYNAMIC_TARGET_MIN_C = 24.6
MPC_DYNAMIC_TARGET_MAX_C = 25.2
MPC_COOLANT_RESERVE_ENABLED = False
MPC_W_PLATE_RESERVE = 50000.0
MPC_W_TANK_RESERVE = 10000.0
MPC_W_QEVAP_RESERVE = 50000.0
MPC_W_BAT_SAFETY = 200000.0
MPC_RESERVE_T_PLATE_IN_REF_C = 25.0
MPC_RESERVE_T_TANK_REF_C = 26.0
MPC_RESERVE_QEVAP_REF_W = 2000.0
MPC_RESERVE_T_SCALE_C = 3.0
MPC_RESERVE_QEVAP_SCALE_W = 1000.0
MPC_T_BAT_MAX_C = 26.0
MPC_T_BAT_MARGIN_C = 0.2

INITIAL_TEMP_C = 25.0
INITIAL_SOC_PEAK = 0.95
INITIAL_SOC_REG = 0.55
PEAK_PID_PARAMS = (1.7, 0.002, 0.04)
FREQ_PID_PARAMS = (2, 0.0, 0.0)

PLATE_NODE_HEAT_CAPACITY_TOTAL = 6000.0
COMPRESSOR_POWER_SCALE = {
    "on-off": 1.0,
    "pid": 1.0,
    "mpc": 1.0,
}


class SimulationCase(NamedTuple):
    control: str
    scene: str
    flow: str
    source_csv: Path
    main_name: str
    snap_name: str


CASES = [
    SimulationCase("on-off", "调峰", "单向", OLD_ROOT / "on-off" / "调峰输出单向on-off.csv", "调峰输出单向on-off.csv", "调峰温度快照单向on-off.csv"),
    SimulationCase("on-off", "调峰", "双向", OLD_ROOT / "on-off" / "调峰输出双向on-off.csv", "调峰输出双向on-off.csv", "调峰温度快照双向on-off.csv"),
    SimulationCase("on-off", "调频", "单向", OLD_ROOT / "on-off" / "调频输出单向on-off.csv", "调频输出单向on-off.csv", "调频温度快照单向on-off.csv"),
    SimulationCase("on-off", "调频", "双向", OLD_ROOT / "on-off" / "调频输出双向on-off..csv", "调频输出双向on-off.csv", "调频温度快照双向on-off.csv"),
    SimulationCase("pid", "调峰", "单向", OLD_ROOT / "pid" / "调峰输出单向pid.csv", "调峰输出单向pid.csv", "调峰温度快照单向pid.csv"),
    SimulationCase("pid", "调峰", "双向", OLD_ROOT / "pid" / "调峰输出双向pid.csv", "调峰输出双向pid.csv", "调峰温度快照双向pid.csv"),
    SimulationCase("pid", "调频", "单向", OLD_ROOT / "pid" / "调频输出单向pid.csv", "调频输出单向pid.csv", "调频温度快照单向pid.csv"),
    SimulationCase("pid", "调频", "双向", OLD_ROOT / "pid" / "调频输出双向pid.csv", "调频输出双向pid.csv", "调频温度快照双向pid.csv"),
    SimulationCase("mpc", "调峰", "单向", OLD_ROOT / "mpc" / "调峰输出单向mpc.csv", "调峰输出单向mpc.csv", "调峰温度快照单向mpc.csv"),
    SimulationCase("mpc", "调峰", "双向", OLD_ROOT / "mpc" / "调峰输出双向mpc.csv", "调峰输出双向mpc.csv", "调峰温度快照双向mpc.csv"),
    SimulationCase("mpc", "调频", "单向", OLD_ROOT / "mpc" / "调频输出单向mpc.csv", "调频输出单向mpc.csv", "调频温度快照单向mpc.csv"),
    SimulationCase("mpc", "调频", "双向", OLD_ROOT / "mpc" / "调频输出双向mpc.csv", "调频输出双向mpc.csv", "调频温度快照双向mpc.csv"),
]


def initial_soc_for(scene):
    return INITIAL_SOC_REG if scene == "调频" else INITIAL_SOC_PEAK


def is_frequency_scene(scene):
    return scene == "调频"


def pid_target_temp_for_scene(scene):
    return FREQ_PID_TARGET_TEMP_C if is_frequency_scene(scene) else PEAK_PID_TARGET_TEMP_C


def pid_params_for_scene(scene):
    return FREQ_PID_PARAMS if is_frequency_scene(scene) else PEAK_PID_PARAMS


def compressor_power_scale_for(control):
    return COMPRESSOR_POWER_SCALE.get(control, 1.0)





