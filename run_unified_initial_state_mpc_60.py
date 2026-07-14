from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import thermal_case_simulator as sim
from thermal_batch_config import EVAP_CAP_FACTOR, EVAP_FLOW_EXP, EVAP_UA_FACTOR


INIT_TEMP_C = 25.0
INIT_TEMP_K = INIT_TEMP_C + 273.15
MAX_STEPS = 60
FLOW = "单向"
OUT_ROOT = Path("outputs/unified_initial_state_mpc_60steps")
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
TMP = Path(os.environ.get("BTMS_GEKKO_TMP", "outputs/btms_gekko_unified_initial")).resolve()
TMP.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(TMP)
os.environ["TMP"] = str(TMP)
os.environ["TEMP"] = str(TMP)

SOURCE = {
    "peak": ("调峰", SOURCE_ROOT / "调峰输出单向mpc.csv"),
    "freq": ("调频", SOURCE_ROOT / "调频输出单向mpc.csv"),
}


def make_simulate_case_with_unified_initial_state():
    source = inspect.getsource(sim.simulate_case)
    replacements = {
        "initial_temp_c=INITIAL_TEMP_C,": "initial_temp_c=UNIFIED_INIT_TEMP_C,",
        "    t_tank_k = AMBIENT_TEMP_K\n": "    t_tank_k = UNIFIED_INIT_TEMP_K\n",
        "    t_plate_k_array = np.full(pack.cols, AMBIENT_TEMP_K)\n": (
            "    t_plate_k_array = np.full(pack.cols, UNIFIED_INIT_TEMP_K)\n"
        ),
        "        t_cabinet = AMBIENT_TEMP_K\n": "        t_cabinet = UNIFIED_INIT_TEMP_K\n",
    }
    for old, new in replacements.items():
        if old not in source:
            raise RuntimeError(f"simulate_case source pattern not found: {old!r}")
        source = source.replace(old, new, 1)

    init_block = (
        "        refrigeration_dynamic_state = initialize_refrigeration_dynamic_state(initial_n_comp, initial_n_pump)\n"
    )
    init_replacement = init_block + (
        "        refrigeration_dynamic_state['T_pipe_supply_K'] = UNIFIED_INIT_TEMP_K\n"
        "        refrigeration_dynamic_state['T_pipe_return_K'] = UNIFIED_INIT_TEMP_K\n"
        "        refrigeration_dynamic_state['T_pipe_supply_history_K'] = [UNIFIED_INIT_TEMP_K] * 8\n"
        "        refrigeration_dynamic_state['T_pipe_return_history_K'] = [UNIFIED_INIT_TEMP_K] * 8\n"
    )
    if init_block not in source:
        raise RuntimeError("simulate_case refrigeration dynamic-state init block not found")
    source = source.replace(init_block, init_replacement, 1)

    namespace = dict(sim.__dict__)
    namespace["UNIFIED_INIT_TEMP_C"] = INIT_TEMP_C
    namespace["UNIFIED_INIT_TEMP_K"] = INIT_TEMP_K
    exec(source, namespace)
    return namespace["simulate_case"]


def pick(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def s(df: pd.DataFrame, names: list[str]) -> pd.Series:
    col = pick(df, names)
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def integrate_kwh(power: pd.Series, time: pd.Series) -> float:
    t = time.to_numpy(float)
    p = power.to_numpy(float)
    if len(t) < 2:
        return 0.0
    return float(np.nansum(p * np.diff(t, prepend=t[0]) / 3600.0))


def duration_s(mask, time: pd.Series) -> float:
    t = time.to_numpy(float)
    if len(t) < 2:
        return 0.0
    step = np.diff(t, append=t[-1])
    good = step[np.isfinite(step) & (step > 0)]
    fallback = float(np.median(good)) if good.size else 10.0
    step[-1] = fallback
    step = np.where(np.isfinite(step) & (step > 0), step, fallback)
    return float(np.sum(step[np.asarray(mask, dtype=bool)]))


def summarize(csv_path: Path, case: str) -> dict[str, object]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    time = s(df, ["Time (s)", "Time"])
    temp = s(df, ["Battery Temp (C)", "Average temperature"])
    p_comp = s(df, ["P_Comp", "Compressor power (kW)"])
    p_pump = s(df, ["P_Pump", "Pump power (kW)"])
    p_fan = s(df, ["P_Fan", "Fan power (kW)"])
    return {
        "case": case,
        "init_temp_c": INIT_TEMP_C,
        "EVAP_UA_FACTOR": EVAP_UA_FACTOR,
        "EVAP_FLOW_EXP": EVAP_FLOW_EXP,
        "EVAP_CAP_FACTOR": EVAP_CAP_FACTOR,
        "rows": int(len(df)),
        "Tmin": float(temp.min()),
        "Tmean": float(temp.mean()),
        "Tmax": float(temp.max()),
        "Tfinal": float(temp.iloc[-1]),
        "above_25p5_s": duration_s(temp > 25.5, time),
        "below_24p5_s": duration_s(temp < 24.5, time),
        "total_kWh": float(s(df, ["Cumulative Energy (kWh)", "Cumulative energy consumption"]).iloc[-1]),
        "comp_kWh": integrate_kwh(p_comp, time),
        "pump_kWh": integrate_kwh(p_pump, time),
        "fan_kWh": integrate_kwh(p_fan, time),
        "mean_comp_rpm": float(s(df, ["Compressor Speed (RPM)", "Compressor Speed"]).mean()),
        "mean_pump_rpm": float(s(df, ["Pump Speed (RPM)"]).mean()),
        "T_supply_mean": float(s(df, ["T_Pipe_Supply_C", "Supply pipe coolant temperature"]).mean()),
        "T_tank_mean": float(s(df, ["Coolant Temp (C)", "Coolant temperature"]).mean()),
        "Q_evap_mean": float(s(df, ["Q_Dot_Evap", "Evaporator cooling rate (kW)"]).mean()),
        "Q_batt_plate_mean": float(s(df, ["Q_Dot_Bat", "Battery heat removal rate (kW)"]).mean()),
        "csv": str(csv_path),
    }


def log_to(path: Path):
    def log(message):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")
    return log


def run_one(case: str, simulate_case, log_path: Path) -> dict[str, object]:
    scene, source_csv = SOURCE[case]
    tag = f"mpc_{case}_unified_init_{INIT_TEMP_C:g}C"
    result = simulate_case(
        "mpc",
        scene,
        FLOW,
        source_csv,
        f"{tag}.csv",
        f"{tag}_snap.csv",
        output_root=OUT_ROOT,
        mpc_flow_mode="supervised",
        force=True,
        max_steps=MAX_STEPS,
        log_func=log_to(log_path),
    )
    return summarize(Path(result["out_csv"]), case)


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / "run_log.txt"
    if log_path.exists():
        log_path.unlink()
    simulate_case = make_simulate_case_with_unified_initial_state()
    rows = []
    for case in ["peak", "freq"]:
        print(f"RUN {case} init={INIT_TEMP_C:g}C", flush=True)
        row = run_one(case, simulate_case, log_path)
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT_ROOT / "unified_initial_state_mpc_metrics.csv", index=False, encoding="utf-8-sig")
        (OUT_ROOT / "unified_initial_state_mpc_metrics.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"DONE {case} Tmax={row['Tmax']:.4f} Tfinal={row['Tfinal']:.4f} "
            f"Tsup={row['T_supply_mean']:.3f} Ttank={row['T_tank_mean']:.3f}",
            flush=True,
        )
    print("METRICS_PATH=outputs/unified_initial_state_mpc_60steps/unified_initial_state_mpc_metrics.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

