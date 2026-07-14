from __future__ import annotations
import argparse
import os
import shutil
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import mpc_flow_direction_strategies as flow_mpc
import thermal_system
from mpc_flow_direction_strategies import freq_mpc_params, peak_mpc_params
from run_unified_initial_state_mpc_60 import (
    FLOW,
    INIT_TEMP_C,
    MAX_STEPS,
    SOURCE,
    duration_s,
    integrate_kwh,
    log_to,
    make_simulate_case_with_unified_initial_state,
    s,
)
from thermal_batch_config import MPC_CV_BAND_C, N_COMP_MAX_RPM, N_PUMP_MAX_RPM, SIM_DT
DEFAULT_OUTPUT_ROOT = Path("outputs/mpc_sensitivity_60steps")
FINAL_ADAPTIVE_OUTPUT_ROOT = Path("outputs/final_adaptive_mpc_candidate_verify")
OLD_FINAL_ADAPTIVE_OUTPUT_ROOTS = (Path("outputs/mpc_final_adaptive"),)
FINAL_CV_BAND_OUTPUT_ROOT = Path("outputs/mpc_final_cv_band_scan_120steps_narrow")
FINAL_PUMP_WEIGHT_OUTPUT_ROOT = Path("outputs/mpc_final_pump_weight_scan_120steps")
FINAL_PUMP_DMAX_OUTPUT_ROOT = Path("outputs/mpc_final_pump_dmax_scan_120steps")
FREQ_COMP_DISP_SCALE_OUTPUT_ROOT = Path("outputs/freq_compressor_displacement_scale_scan")
FREQ_COMP_MIN_RPM_OUTPUT_ROOT = Path("outputs/freq_compressor_min_rpm_scan")
HARDWARE_ABLATION_OUTPUT_ROOT_120 = Path("outputs/peak_freq_minrpm_dispscale_ablation_120steps")
HARDWARE_ABLATION_OUTPUT_ROOT_60 = Path("outputs/peak_freq_minrpm_dispscale_ablation_60steps")
HARDWARE_ABLATION_SWEEP_TYPE = "peak_freq_minrpm_dispscale_ablation"
PARAMSET_CROSS_OUTPUT_ROOT = Path("outputs/mpc_paramset_case_2x2_cross_validation")
PARAMSET_CROSS_SWEEP_TYPE = "mpc_paramset_case_2x2_cross_validation"
TERMINAL_COST_OUTPUT_ROOT = Path("outputs/terminal_cost_scan")
TERMINAL_COST_REFINED_OUTPUT_ROOT = Path("outputs/terminal_cost_refined")
TERMINAL_COST_SELECTED_FULL_OUTPUT_ROOT = Path("outputs/terminal_cost_selected_full")
BASELINE_PARITY_OUTPUT_ROOT = Path("outputs/baseline_parity_check")
THREE_WAY_OUTPUT_ROOT = Path("outputs/terminal_cost_three_way_compare")
THREE_WAY_FULL_OUTPUT_ROOT = Path("outputs/terminal_cost_three_way_full")
TERMINAL_COST_SWEEP_TYPE = "terminal_cost_scan"
TERMINAL_COST_REFINED_SWEEP_TYPE = "terminal_cost_refined"
TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE = "terminal_cost_selected_full"
BASELINE_PARITY_SWEEP_TYPE = "baseline_parity_check"
THREE_WAY_SWEEP_TYPE = "terminal_cost_three_way_compare"
THREE_WAY_FULL_SWEEP_TYPE = "terminal_cost_three_way_full"
TERMINAL_COST_SWEEP_TYPES = (TERMINAL_COST_SWEEP_TYPE, TERMINAL_COST_REFINED_SWEEP_TYPE, TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE)
TERMINAL_REPORT_SWEEP_TYPES = TERMINAL_COST_SWEEP_TYPES + (BASELINE_PARITY_SWEEP_TYPE, THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE)
SUPPORTED_SWEEP_TYPES = ("wsphi", "w_energy_comp", "comp_dmax_scan", "actuator_rate_ablation", "comp_rate_tuning", "comp_rate_timeseries", "adaptive_dmax_w_energy_comp", "final_cv_weight_scan", "stable_cv_weight_scan", "final_adaptive_mpc", "final_cv_band_scan", "final_pump_weight_scan", "final_pump_dmax_scan", "freq_compressor_displacement_scale_scan", "freq_compressor_min_rpm_scan", HARDWARE_ABLATION_SWEEP_TYPE, PARAMSET_CROSS_SWEEP_TYPE, TERMINAL_COST_SWEEP_TYPE, TERMINAL_COST_REFINED_SWEEP_TYPE, TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE, BASELINE_PARITY_SWEEP_TYPE, THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE)
RATE_LIMIT_MODES = (
    "baseline",
    "baseline_rate_on",
    "dcost_off",
    "dmax_2x",
    "dmax_4x",
    "dmax_8x",
    "dmax_240rpmps",
    "dmax_480rpmps",
    "comp_rate_off",
    "pump_rate_off",
    "both_rate_off",
)
ACTUATOR_RATE_MODES = ("baseline_rate_on", "comp_rate_off", "pump_rate_off", "both_rate_off")
COMP_RATE_TUNING_MODES = ("baseline", "dcost_off", "dmax_2x", "dmax_4x", "dmax_8x", "comp_rate_off")
COMP_RATE_TIMESERIES_MODES = ("baseline", "dmax_240rpmps", "dmax_480rpmps", "comp_rate_off")
ADAPTIVE_DMAX_W_ENERGY_COMP_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
FINAL_CV_WEIGHT_VALUES = (5e2, 5e4, 5e6, 5e7, 5e8, 5e9, 5e10, 5e11)
STABLE_CV_WEIGHT_VALUES = (5e6, 1e7, 5e7, 1e8, 5e8, 1e9, 5e9, 1e10)
FINAL_CV_BAND_VALUES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
FINAL_CV_BAND_STEPS = 120
FINAL_PUMP_WEIGHT_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
FINAL_PUMP_WEIGHT_STEPS = 120
FINAL_PUMP_DMAX_FACTORS = (0.5, 1.0, 2.0, 4.0, 8.0, "off")
FINAL_PUMP_DMAX_STEPS = 120
FREQ_COMP_DISP_SCALE_VALUES = (1.0, 0.9, 0.8, 0.7)
FREQ_COMP_MIN_RPM_VALUES = (1000.0, 800.0, 600.0, 500.0)
HARDWARE_ABLATION_MIN_COMP_RPM_VALUES = (1000.0, 500.0)
HARDWARE_ABLATION_DISP_SCALE_VALUES = (1.0, 0.9)
HARDWARE_ABLATION_STEPS = 120
PARAMSET_CROSS_PARAM_SETS = ("peak_params", "freq_params")
TERMINAL_COST_WEIGHTS = ("off", 0.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6)
TERMINAL_COST_REFINED_WEIGHTS = {
    "peak": ("off", 0.0, 3e4, 1e5, 3e5, 1e6, 3e6),
    "freq": ("off", 0.0, 3e4, 7e4, 1e5, 2e5, 3e5, 5e5),
}
TERMINAL_COST_SELECTED_FULL_WEIGHTS = {
    "peak": ("off", 3e5, 1e6),
    "freq": ("off", 1e5, 3e5, 5e5),
}
BASELINE_PARITY_VARIANTS = ("tuned_baseline", "terminal_off_w0", "terminal_on_w0")
THREE_WAY_FACTORS = {
    "peak": ("baseline_off", "empirical_w3e5", "empirical_w1e6"),
    "freq": ("baseline_off", "empirical_w1e5", "empirical_w3e5", "empirical_w5e5"),
}
THREE_WAY_FULL_FACTORS = {
    "peak": ("baseline_off", "empirical_w3e5", "empirical_w1e6"),
    "freq": ("baseline_off", "empirical_w1e5", "empirical_w3e5", "empirical_w5e5"),
}
THREE_WAY_FULL_SKIP_NOTES: list[str] = []
DEFAULT_FACTORS = {
    "comp_dmax_scan": (300.0, 600.0, 1200.0, 2400.0, 6000.0, 1e6),
    "wsphi": (1.0, 2.0, 5.0, 10.0),
    "w_energy_comp": (1.0, 0.5, 0.25, 0.1, 0.0),
    "adaptive_dmax_w_energy_comp": ADAPTIVE_DMAX_W_ENERGY_COMP_FACTORS,
    "final_cv_weight_scan": FINAL_CV_WEIGHT_VALUES,
    "stable_cv_weight_scan": STABLE_CV_WEIGHT_VALUES,
    "final_cv_band_scan": FINAL_CV_BAND_VALUES,
    "final_pump_weight_scan": FINAL_PUMP_WEIGHT_FACTORS,
    "final_pump_dmax_scan": FINAL_PUMP_DMAX_FACTORS,
    "freq_compressor_displacement_scale_scan": FREQ_COMP_DISP_SCALE_VALUES,
    "freq_compressor_min_rpm_scan": FREQ_COMP_MIN_RPM_VALUES,
    TERMINAL_COST_SWEEP_TYPE: TERMINAL_COST_WEIGHTS,
}


def default_three_way_factors(case: str) -> tuple[str, ...]:
    return THREE_WAY_FACTORS[case]

def default_three_way_full_factors(case: str) -> tuple[str, ...]:
    return THREE_WAY_FULL_FACTORS[case]
def parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())
def parse_float_list(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())

def parse_pump_dmax_list(raw: str) -> tuple[float | str, ...]:
    values: list[float | str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "off":
            values.append("off")
        else:
            values.append(float(token))
    return tuple(values)


def parse_terminal_cost_list(raw: str) -> tuple[float | str, ...]:
    values: list[float | str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() == "off":
            values.append("off")
        else:
            values.append(float(token))
    return tuple(values)


def normalize_terminal_cost_factor(factor: object) -> tuple[bool, float]:
    if isinstance(factor, str) and factor.lower() == "off":
        return False, 0.0
    return True, float(factor)


def format_terminal_weight_label(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"

def final_w_energy_comp_for_case(case: str) -> float:
    if case == "peak":
        return 600.0
    if case == "freq":
        return 300.0
    raise ValueError(f"Unknown case: {case}")


def final_cv_weight_for_case(case: str) -> float:
    if case == "peak":
        return 5e6
    if case == "freq":
        return 5e7
    raise ValueError(f"Unknown case: {case}")


def final_cv_band_for_case(case: str) -> float:
    if case == "peak":
        return 0.30
    if case == "freq":
        return 0.45
    raise ValueError(f"Unknown case: {case}")


def final_w_energy_pump_for_case(case: str) -> float:
    if case == "peak":
        return 10000.0
    if case == "freq":
        return 15000.0
    raise ValueError(f"Unknown case: {case}")


def final_pump_dmax_base_for_case(case: str) -> float:
    if case in ("peak", "freq"):
        return 300.0
    raise ValueError(f"Unknown case: {case}")



def planned_simulation_steps(case: str, max_steps: int | None) -> int | float:
    if max_steps is not None:
        return int(max_steps)
    try:
        _scene, source_csv = SOURCE[case]
        return int(len(pd.read_csv(source_csv, encoding="utf-8-sig")))
    except Exception:
        return np.nan
def print_param_extra(case: str, params, mpc_flow_mode: str = "switching", simulation_steps: int | float | None = None) -> None:
    controller_class = {"switching": "SwitchingSystemMPC", "supervised": "SupervisoryEventMPC", "standard": "StandardMPC", "mixed_integer": "MixedIntegerFlowMPC"}.get(mpc_flow_mode, mpc_flow_mode)
    sim_steps = simulation_steps if simulation_steps is not None else np.nan
    sim_duration_s = float(sim_steps) * SIM_DT if np.isfinite(float(sim_steps)) else np.nan
    sim_duration_min = sim_duration_s / 60.0 if np.isfinite(sim_duration_s) else np.nan
    print(
        "PARAM_EXTRA "
        f"case={case} MPC_FLOW_MODE={mpc_flow_mode} controller_class={controller_class} "
        f"simulation_steps={sim_steps} simulation_dt={SIM_DT:g} simulation_duration_s={sim_duration_s:g} simulation_duration_min={sim_duration_min:g} "
        f"prediction_horizon={params.mpc_horizon} control_horizon={params.mpc_horizon} dt={SIM_DT:g} "
        f"target_temp_c=25 tracking_term_on={abs(float(params.w_temp_obj)) > 0.0} "
        f"w_temp_obj={params.w_temp_obj:g} w_bat_safety={params.w_bat_safety:g} w_term_peak={params.w_term_peak:g} "
        f"reserve_enabled={getattr(flow_mpc, 'MPC_COOLANT_RESERVE_ENABLED', False)} coolant_reserve_enabled={getattr(flow_mpc, 'MPC_COOLANT_RESERVE_ENABLED', False)} "
        f"min_comp_rpm={params.n_comp_min:g} min_pump_rpm={params.n_pump_min:g} "
        f"n_comp_initial={params.n_comp_min:g} n_pump_initial={params.n_pump_min:g} "
        f"initial_effective_comp_rpm={params.n_comp_min:g} initial_effective_pump_rpm={params.n_pump_min:g} "
        f"flow_direction_mode={mpc_flow_mode} warm_start=False cold_start=True "
        f"terminal_cost_enabled_default={params.terminal_cost_enabled}",
        flush=True,
    )



def format_terminal_cost_label(factor: object) -> str:
    if isinstance(factor, str) and factor in BASELINE_PARITY_VARIANTS:
        return factor
    if isinstance(factor, str) and factor == "baseline_off":
        return str(factor).strip()
    if isinstance(factor, str) and factor.startswith("empirical_w"):
        return format_three_way_label(factor)
    enabled, weight = normalize_terminal_cost_factor(factor)
    state = "on" if enabled else "off"
    return f"terminal_{state}_w{format_terminal_weight_label(weight)}"


def terminal_cost_case_label(case: str, factor: object) -> str:
    if isinstance(factor, str) and factor == "baseline_off":
        return f"{case}_{factor}"
    if isinstance(factor, str) and factor.startswith("empirical_w"):
        return f"{case}_{format_three_way_label(factor)}"
    return f"{case}_{format_terminal_cost_label(factor)}"





def format_cv_weight_label(value: float) -> str:
    mantissa, exponent = f"{float(value):.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_band_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def format_factor_label(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def hardware_ablation_factor(min_comp_rpm: float, disp_scale: float) -> tuple[float, float]:
    return (float(min_comp_rpm), float(disp_scale))


def parse_hardware_ablation_factor(factor: object) -> tuple[float, float]:
    if isinstance(factor, tuple) and len(factor) == 2:
        return float(factor[0]), float(factor[1])
    raise ValueError(f"Expected (min_comp_rpm, disp_scale), got {factor!r}")


def format_hardware_ablation_label(min_comp_rpm: float, disp_scale: float) -> str:
    return f"min{format_factor_label(min_comp_rpm)}rpm_disp{format_factor_label(disp_scale)}"


def hardware_ablation_output_root(max_steps: int | None) -> Path:
    if max_steps == 60:
        return HARDWARE_ABLATION_OUTPUT_ROOT_60
    return HARDWARE_ABLATION_OUTPUT_ROOT_120


def normalize_param_set(param_set: object) -> str:
    value = str(param_set).strip()
    if value in PARAMSET_CROSS_PARAM_SETS:
        return value
    raise ValueError(f"Unsupported param_set: {param_set!r}")


def param_set_case_label(case: str, param_set: object) -> str:
    return f"{case}_{normalize_param_set(param_set)}"


def params_for_param_set(param_set: object):
    value = normalize_param_set(param_set)
    if value == "peak_params":
        return get_final_adaptive_mpc_params("peak")
    if value == "freq_params":
        return get_final_adaptive_mpc_params("freq")
    raise ValueError(f"Unsupported param_set: {param_set!r}")


def format_pump_dmax_label(value: float | str) -> str:
    if isinstance(value, str) and value.lower() == "off":
        return "off"
    return f"x{float(value):g}"


def format_pump_dmax_file_label(value: float | str) -> str:
    return format_pump_dmax_label(value).replace(".", "p")


def task_label_for_filter(sweep_type: str, factor: object, is_cv_only: bool, rate_limit_mode: str) -> str:
    if sweep_type in TERMINAL_COST_SWEEP_TYPES:
        return format_terminal_cost_label(factor)
    if sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        return normalize_param_set(factor)
    if sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        min_comp_rpm, disp_scale = parse_hardware_ablation_factor(factor)
        return format_hardware_ablation_label(min_comp_rpm, disp_scale)
    if sweep_type == BASELINE_PARITY_SWEEP_TYPE:
        return str(factor)
    if sweep_type == "final_pump_dmax_scan":
        return format_pump_dmax_label(factor)
    if sweep_type in ("actuator_rate_ablation", "comp_rate_tuning", "comp_rate_timeseries"):
        return str(rate_limit_mode)
    if is_cv_only:
        return "cv_only"
    return str(factor)


def base_params_for_case(case: str):
    if case == "peak":
        return peak_mpc_params
    if case == "freq":
        return freq_mpc_params
    raise ValueError(f"Unknown case: {case}")


def terminal_cost_off_params(params, name: str | None = None):
    return replace(
        params,
        name=params.name if name is None else name,
        terminal_cost_enabled=False,
        terminal_cost_type="empirical_temp",
        w_terminal_temp=0.0,
        terminal_temp_scale_c=1.0,
    )


def get_final_adaptive_mpc_params(case: str):
    base_params = base_params_for_case(case)
    if case == "peak":
        params = replace(
            base_params,
            name="final_adaptive_mpc_peak",
            cv_band_half_width=0.30,
            w_high_temp=5e6,
            w_cold_temp=5e6,
            w_energy_comp=600.0,
            w_energy_pump=10000.0,
            dmax_comp=6000.0,
            dmax_pump=300.0,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    if case == "freq":
        params = replace(
            base_params,
            name="final_adaptive_mpc_freq",
            cv_band_half_width=0.45,
            w_high_temp=5e7,
            w_cold_temp=5e7,
            w_energy_comp=300.0,
            w_energy_pump=15000.0,
            dmax_comp=6000.0,
            dmax_pump=600.0,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    raise ValueError(f"Unknown case: {case}")


TERMINAL_OVERLAY_FIELDS = {
    "terminal_cost_enabled",
    "terminal_cost_type",
    "w_terminal_temp",
    "terminal_temp_scale_c",
}


def build_tuned_mpc_baseline(scene: str):
    return terminal_cost_off_params(deepcopy(get_final_adaptive_mpc_params(scene)))


def apply_final_selected_terminal_cost_params(params, scene: str):
    scene_key = str(scene).strip().lower()
    if scene_key == "peak":
        weight = 1e6
    elif scene_key == "freq":
        weight = 5e5
    else:
        raise ValueError(f"Unknown scene: {scene}")
    overlay = {
        "terminal_cost_enabled": True,
        "terminal_cost_type": "empirical_temp",
        "w_terminal_temp": weight,
        "terminal_temp_scale_c": 1.0,
    }
    if isinstance(params, dict):
        out = deepcopy(params)
        out.update(overlay)
        return out
    return replace(deepcopy(params), **overlay)


def apply_terminal_cost_overlay(params, overlay: dict[str, object], name: str | None = None):
    illegal = sorted(set(overlay) - TERMINAL_OVERLAY_FIELDS)
    if illegal:
        raise ValueError(f"Terminal overlay tried to modify non-terminal fields: {illegal}")
    updated = replace(deepcopy(params), **overlay)
    if name is not None:
        updated = replace(updated, name=name)
    return updated


TERMINAL_ALLOWED_DIFF_FIELDS = TERMINAL_OVERLAY_FIELDS | {"name"}


def terminal_param_diff(case: str, params) -> dict[str, tuple[object, object]]:
    baseline = build_tuned_mpc_baseline(case)
    diff: dict[str, tuple[object, object]] = {}
    for key, base_value in vars(baseline).items():
        value = getattr(params, key)
        same = value == base_value
        if isinstance(value, float) and isinstance(base_value, float) and np.isnan(value) and np.isnan(base_value):
            same = True
        if not same:
            diff[key] = (base_value, value)
    illegal = sorted(set(diff) - TERMINAL_ALLOWED_DIFF_FIELDS)
    if illegal:
        raise RuntimeError(f"Terminal overlay changed non-terminal baseline fields: {illegal}")
    return diff


def print_terminal_param_diff(case: str, params) -> None:
    diff = terminal_param_diff(case, params)
    if not diff:
        print("CONFIG_DIFF none", flush=True)
        return
    text = "; ".join(f"{key}:{old!r}->{new!r}" for key, (old, new) in diff.items())
    print(f"CONFIG_DIFF {text}", flush=True)


def build_baseline_parity_params(case: str, variant: str):
    baseline = build_tuned_mpc_baseline(case)
    if variant in ("tuned_baseline", "terminal_off_w0"):
        return replace(deepcopy(baseline), name=f"{baseline.name}_{variant}")
    if variant == "terminal_on_w0":
        return apply_terminal_cost_overlay(
            baseline,
            {
                "terminal_cost_enabled": True,
                "terminal_cost_type": "empirical_temp",
                "w_terminal_temp": 0.0,
                "terminal_temp_scale_c": 1.0,
            },
            name=f"{baseline.name}_{variant}",
        )
    raise ValueError(f"Unknown baseline parity variant: {variant}")


def parse_three_way_label(label: object) -> tuple[str, float | None, float | None]:
    text = str(label).strip()
    if text == "baseline_off":
        return "off", None, None
    if text.startswith("empirical_w"):
        return "empirical_temp", float(text.removeprefix("empirical_w")), None
    raise ValueError(f"Unsupported terminal_cost_three_way_compare label: {label!r}")


def format_three_way_label(label: object) -> str:
    parse_three_way_label(label)
    return str(label).strip()








def build_three_way_compare_params(case: str, label: object):
    terminal_label = format_three_way_label(label)
    kind, q_or_weight, _unused = parse_three_way_label(terminal_label)
    baseline = build_tuned_mpc_baseline(case)
    if kind == "off":
        return replace(deepcopy(baseline), name=f"{baseline.name}_{terminal_label}")
    if kind == "empirical_temp":
        return apply_terminal_cost_overlay(
            baseline,
            {
                "terminal_cost_enabled": True,
                "terminal_cost_type": "empirical_temp",
                "w_terminal_temp": float(q_or_weight),
                "terminal_temp_scale_c": 1.0,
            },
            name=f"{baseline.name}_{terminal_label}",
        )
    raise ValueError(f"Unsupported terminal cost label: {label!r}")
def build_params(base_params, sweep_type: str, factor: float, rate_limit_mode: str, is_cv_only: bool = False, case: str | None = None):
    if is_cv_only:
        return apply_rate_limit_mode(
            replace(
                base_params,
                name=f"{base_params.name}_cv_only",
                w_energy_comp=0.0,
                w_energy_pump=0.0,
            ),
            rate_limit_mode,
        )
    if sweep_type == "wsphi":
        return apply_rate_limit_mode(
            replace(
                base_params,
                name=f"{base_params.name}_wsphi_x{factor:g}",
                w_high_temp=base_params.w_high_temp * float(factor),
            ),
            rate_limit_mode,
        )
    if sweep_type == "w_energy_comp":
        if case is None:
            raise ValueError("case is required for w_energy_comp")
        params = replace(
            base_params,
            name=f"{base_params.name}_wcomp_x{factor:g}",
            w_energy_comp=base_params.w_energy_comp * float(factor),
            dmax_comp=6000.0,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    if sweep_type == "adaptive_dmax_w_energy_comp":
        if case is None:
            raise ValueError("case is required for adaptive_dmax_w_energy_comp")
        dmax_rpm_per_s = adaptive_comp_dmax_rpm_per_s(case)
        return replace(
            base_params,
            name=f"{base_params.name}_adaptive_dmax_wcomp_x{factor:g}",
            w_energy_comp=base_params.w_energy_comp * float(factor),
            dmax_comp=dmax_rpm_per_s * SIM_DT,
        )
    if sweep_type in ("final_cv_weight_scan", "stable_cv_weight_scan"):
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        cv_weight = float(factor)
        params = replace(
            base_params,
            name=f"{base_params.name}_{sweep_type}_{format_cv_weight_label(factor)}",
            w_high_temp=cv_weight,
            w_cold_temp=cv_weight,
            w_energy_comp=final_w_energy_comp_for_case(case),
            w_energy_pump=final_w_energy_pump_for_case(case),
            dmax_comp=6000.0,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    if sweep_type == "final_pump_weight_scan":
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        params = replace(
            base_params,
            name=f"{base_params.name}_pump_x{factor:g}",
            cv_band_half_width=final_cv_band_for_case(case),
            w_high_temp=final_cv_weight_for_case(case),
            w_cold_temp=final_cv_weight_for_case(case),
            w_energy_comp=final_w_energy_comp_for_case(case),
            w_energy_pump=final_w_energy_pump_for_case(case) * float(factor),
            dmax_comp=6000.0,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    if sweep_type == "freq_compressor_displacement_scale_scan":
        if case != "freq":
            raise ValueError(f"{sweep_type} is only defined for freq")
        return get_final_adaptive_mpc_params("freq")
    if sweep_type == "freq_compressor_min_rpm_scan":
        if case != "freq":
            raise ValueError(f"{sweep_type} is only defined for freq")
        return replace(get_final_adaptive_mpc_params("freq"), n_comp_min=float(factor))
    if sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        min_comp_rpm, _disp_scale = parse_hardware_ablation_factor(factor)
        return replace(get_final_adaptive_mpc_params(case), n_comp_min=min_comp_rpm)
    if sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        return params_for_param_set(factor)
    if sweep_type in (THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        return build_three_way_compare_params(case, factor)
    if sweep_type in TERMINAL_COST_SWEEP_TYPES:
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        terminal_enabled, w_terminal = normalize_terminal_cost_factor(factor)
        base_final_params = build_tuned_mpc_baseline(case)
        if not terminal_enabled:
            return replace(deepcopy(base_final_params), name=f"{base_final_params.name}_{format_terminal_cost_label(factor)}")
        return apply_terminal_cost_overlay(
            base_final_params,
            {
                "terminal_cost_enabled": True,
                "terminal_cost_type": "empirical_temp",
                "w_terminal_temp": w_terminal,
                "terminal_temp_scale_c": 1.0,
            },
            name=f"{base_final_params.name}_{format_terminal_cost_label(factor)}",
        )
    if sweep_type == BASELINE_PARITY_SWEEP_TYPE:
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        return build_baseline_parity_params(case, str(factor))
    if sweep_type == "final_pump_dmax_scan":
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        if isinstance(factor, str) and factor.lower() == "off":
            pump_dmax = float(N_PUMP_MAX_RPM)
        else:
            pump_dmax = final_pump_dmax_base_for_case(case) * float(factor)
        params = replace(
            base_params,
            name=f"{base_params.name}_pump_dmax_{format_pump_dmax_file_label(factor)}",
            cv_band_half_width=final_cv_band_for_case(case),
            w_high_temp=final_cv_weight_for_case(case),
            w_cold_temp=final_cv_weight_for_case(case),
            w_energy_comp=final_w_energy_comp_for_case(case),
            w_energy_pump=final_w_energy_pump_for_case(case),
            dmax_comp=6000.0,
            dmax_pump=pump_dmax,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    if sweep_type == "final_adaptive_mpc":
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        return get_final_adaptive_mpc_params(case)
    if sweep_type == "final_cv_band_scan":
        if case is None:
            raise ValueError(f"case is required for {sweep_type}")
        params = replace(
            base_params,
            name=f"{base_params.name}_cv_band_{format_band_label(factor)}",
            cv_band_half_width=float(factor),
            w_high_temp=final_cv_weight_for_case(case),
            w_cold_temp=final_cv_weight_for_case(case),
            w_energy_comp=final_w_energy_comp_for_case(case),
            w_energy_pump=final_w_energy_pump_for_case(case),
            dmax_comp=6000.0,
        )
        return apply_final_selected_terminal_cost_params(params, case)
    if sweep_type == "comp_dmax_scan":
        dmax = float(factor)
        label = "off" if dmax >= 1e6 else f"{dmax:g}"
        params = replace(base_params, name=f"{base_params.name}_comp_dmax_{label}", dmax_comp=dmax)
        return apply_final_selected_terminal_cost_params(params, case)
    if sweep_type in ("actuator_rate_ablation", "comp_rate_tuning", "comp_rate_timeseries"):
        return apply_rate_limit_mode(base_params, rate_limit_mode)
    raise ValueError(f"Unsupported sweep type: {sweep_type}")


@contextmanager
def patched_params(case: str, params, cv_band_half_width: float | None = None):
    original_peak = flow_mpc.peak_mpc_params
    original_freq = flow_mpc.freq_mpc_params
    original_cv_band = flow_mpc.MPC_CV_BAND_C
    try:
        if cv_band_half_width is not None:
            flow_mpc.MPC_CV_BAND_C = float(cv_band_half_width)
        if case == "peak":
            flow_mpc.peak_mpc_params = params
        elif case == "freq":
            flow_mpc.freq_mpc_params = params
        else:
            raise ValueError(f"Unknown case: {case}")
        yield
    finally:
        flow_mpc.peak_mpc_params = original_peak
        flow_mpc.freq_mpc_params = original_freq
        flow_mpc.MPC_CV_BAND_C = original_cv_band


def cleanup_detail(result: dict[str, object], output_root: Path) -> None:
    for key in ("out_csv", "snap_csv"):
        path = Path(result[key])
        try:
            if path.is_file() and output_root in path.resolve().parents:
                path.unlink()
        except OSError:
            pass
    scratch = output_root / "_scratch_detail"
    if scratch.exists():
        shutil.rmtree(scratch, ignore_errors=True)


@contextmanager
def patched_compressor_displacement(scale: float | None = None):
    original_v_disp = thermal_system.V_disp_m3_per_rev
    try:
        if scale is not None:
            thermal_system.V_disp_m3_per_rev = float(original_v_disp) * float(scale)
            if hasattr(thermal_system, "clear_refrigeration_cycle_cache"):
                thermal_system.clear_refrigeration_cycle_cache()
        yield
    finally:
        thermal_system.V_disp_m3_per_rev = original_v_disp
        if hasattr(thermal_system, "clear_refrigeration_cycle_cache"):
            thermal_system.clear_refrigeration_cycle_cache()
@contextmanager
def patched_compressor_min_rpm(min_rpm: float | None = None):
    if min_rpm is None:
        yield
        return
    original_speed_axis = thermal_system.speed_axis.copy()
    original_eta_vol_interpolator = thermal_system.eta_vol_interpolator
    original_eta_is_interpolator = thermal_system.eta_is_interpolator
    try:
        patched_axis = original_speed_axis.copy()
        patched_axis[0] = float(min_rpm)
        thermal_system.speed_axis = patched_axis
        thermal_system.eta_vol_interpolator = thermal_system.RegularGridInterpolator(
            (thermal_system.speed_axis, thermal_system.pr_axis),
            thermal_system.eta_vol_map,
            bounds_error=False,
            fill_value=None,
        )
        thermal_system.eta_is_interpolator = thermal_system.RegularGridInterpolator(
            (thermal_system.speed_axis, thermal_system.pr_axis),
            thermal_system.eta_is_map,
            bounds_error=False,
            fill_value=None,
        )
        if hasattr(thermal_system, "clear_refrigeration_cycle_cache"):
            thermal_system.clear_refrigeration_cycle_cache()
        yield
    finally:
        thermal_system.speed_axis = original_speed_axis
        thermal_system.eta_vol_interpolator = original_eta_vol_interpolator
        thermal_system.eta_is_interpolator = original_eta_is_interpolator
        if hasattr(thermal_system, "clear_refrigeration_cycle_cache"):
            thermal_system.clear_refrigeration_cycle_cache()
def time_steps_s(time_s: pd.Series) -> np.ndarray:
    t = time_s.to_numpy(float)
    if len(t) == 0:
        return np.array([], dtype=float)
    if len(t) == 1:
        return np.array([0.0], dtype=float)
    step = np.diff(t, append=t[-1])
    good = step[np.isfinite(step) & (step > 0)]
    fallback = float(np.median(good)) if good.size else 5.0
    step[-1] = fallback
    return np.where(np.isfinite(step) & (step > 0), step, fallback)


def mean_abs_delta(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if values.size < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(values))))


def max_abs_delta(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if values.size < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(values))))


def upper_ratio(series: pd.Series, upper: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values >= upper - 1.0))


def lower_ratio(series: pd.Series, lower: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
    if values.size == 0:
        return float("nan")
    return float(np.mean(values <= lower + 1.0))


def summarize_csv(
    csv_path: Path,
    case: str,
    sweep_type: str,
    factor: float,
    is_cv_only: bool,
    rate_limit_mode: str,
    params,
    cv_band_half_width: float | None = None,
    n_steps: int | None = None,
) -> dict[str, object]:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    def optional_series(names: list[str], fallback: pd.Series) -> pd.Series:
        for name in names:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce")
        return fallback
    def last_valid_value(series: pd.Series, default: float = np.nan) -> float:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
            return float(default)
        return float(values.iloc[-1])

    def raw_series(name: str, default: object = "") -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(default, index=df.index)

    def last_valid_object(series: pd.Series, default: object = "") -> object:
        values = series.dropna()
        if values.empty:
            return default
        return values.iloc[-1]


    time_s = s(df, ["Time (s)", "Time"])
    temp_mean_c = s(df, ["T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
    temp_c = temp_mean_c
    temp_cell_max_c = optional_series(["T_cell_max_C"], temp_mean_c)
    temp_cell_min_c = optional_series(["T_cell_min_C"], temp_mean_c)
    delta_t_cell_c = optional_series(["Delta_T_cell_C"], temp_cell_max_c - temp_cell_min_c)
    t_ref = 25.0
    band = float(MPC_CV_BAND_C if cv_band_half_width is None else cv_band_half_width)
    hot_threshold = t_ref + band
    cold_threshold = t_ref - band
    temp_error = temp_c.to_numpy(float) - t_ref
    hot_over = np.maximum(0.0, temp_c.to_numpy(float) - hot_threshold)
    hot_25p5_over = np.maximum(0.0, temp_c.to_numpy(float) - 25.5)
    hot_25p5_cellmax_over = np.maximum(0.0, temp_cell_max_c.to_numpy(float) - 25.5)
    cold_over = np.maximum(0.0, cold_threshold - temp_c.to_numpy(float))
    steps_s = time_steps_s(time_s)
    p_comp = s(df, ["P_Comp", "Compressor power (kW)"])
    p_pump = s(df, ["P_Pump", "Pump power (kW)"])
    p_fan = s(df, ["P_Fan", "Fan power (kW)"])
    comp_speed = s(df, ["Compressor Speed (RPM)", "Compressor Speed"])
    pump_speed = s(df, ["Pump Speed (RPM)", "Pump Speed"])
    comp_command = s(df, ["Compressor Command (RPM)", "Compressor command", "Compressor Speed (RPM)", "Compressor Speed"])
    pump_command = s(df, ["Pump Command (RPM)", "Pump command", "Pump Speed (RPM)", "Pump Speed"])
    nan_series = pd.Series(np.nan, index=df.index)
    t_pred_end_series = optional_series(["T_pred_end_C"], nan_series)
    j_terminal_series = optional_series(["J_terminal"], nan_series)
    weighted_j_terminal_series = optional_series(["Weighted_J_terminal"], nan_series)
    solve_time_series = optional_series(["MPC_Solve_Time_S", "MPC solve time"], nan_series)
    solve_success_series = optional_series(["MPC_Solved"], pd.Series(1.0, index=df.index))
    try:
        factor_numeric = float(factor)
    except (TypeError, ValueError):
        factor_numeric = float("nan")
    disp_scale_value = 1.0
    min_comp_rpm_value = float(params.n_comp_min)
    if sweep_type == "freq_compressor_displacement_scale_scan":
        disp_scale_value = factor_numeric
    elif sweep_type == "freq_compressor_min_rpm_scan":
        min_comp_rpm_value = factor_numeric
    elif sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        min_comp_rpm_value, disp_scale_value = parse_hardware_ablation_factor(factor)
    if sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        param_set_value = normalize_param_set(factor)
    else:
        param_set_value = ""
    if sweep_type == "final_pump_dmax_scan":
        factor_label = format_pump_dmax_label(factor)
    elif sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        factor_label = format_hardware_ablation_label(min_comp_rpm_value, disp_scale_value)
    elif sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        factor_label = param_set_value
    elif sweep_type == BASELINE_PARITY_SWEEP_TYPE:
        factor_label = str(factor)
    elif sweep_type in (THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        factor_label = format_three_way_label(factor)
    elif sweep_type in TERMINAL_COST_SWEEP_TYPES:
        factor_label = format_terminal_cost_label(factor)
    else:
        factor_label = f"x{factor_numeric:g}"
    pump_dmax_base = final_pump_dmax_base_for_case(case) if sweep_type == "final_pump_dmax_scan" else float(params.dmax_pump)
    pump_dmax_label = format_pump_dmax_label(factor) if sweep_type == "final_pump_dmax_scan" else "baseline"
    pump_dmax_factor = "off" if pump_dmax_label == "off" else factor_numeric
    terminal_cost_enabled_value = bool(getattr(params, "terminal_cost_enabled", False))
    terminal_cost_type_value = str(getattr(params, "terminal_cost_type", "empirical_temp"))
    w_terminal_temp_value = float(getattr(params, "w_terminal_temp", 0.0))
    terminal_temp_scale_value = float(getattr(params, "terminal_temp_scale_c", 1.0))
    weighted_j_terminal_value = last_valid_value(weighted_j_terminal_series)
    if (not terminal_cost_enabled_value) or w_terminal_temp_value <= 0.0:
        weighted_j_terminal_value = 0.0
    return {
        "case": case,
        "scene": case,
        "param_set": param_set_value,
        "sweep_type": sweep_type,
        "run_type": sweep_type,
        "n_steps": int(n_steps) if n_steps is not None else int(len(df)),
        "simulation_steps": int(n_steps) if n_steps is not None else int(len(df)),
        "dt_s": float(SIM_DT),
        "simulation_dt": float(SIM_DT),
        "sim_duration_s": float((int(n_steps) if n_steps is not None else int(len(df))) * SIM_DT),
        "simulation_duration_s": float((int(n_steps) if n_steps is not None else int(len(df))) * SIM_DT),
        "simulation_duration_min": float((int(n_steps) if n_steps is not None else int(len(df))) * SIM_DT / 60.0),
        "controller": sweep_type if sweep_type in ("final_adaptive_mpc", "final_cv_band_scan", "final_pump_weight_scan", "final_pump_dmax_scan") else sweep_type,
        "cv_band_half_width": band,
        "CV_band_low": cold_threshold,
        "CV_band_high": hot_threshold,
        "T_ref": t_ref,
        "w_energy_comp_factor": factor_numeric,
        "is_cv_only": bool(is_cv_only),
        "rate_limit_mode": rate_limit_mode,
        "w_energy_comp_label": factor_label,
        "w_energy_pump_factor": float(factor) if sweep_type == "final_pump_weight_scan" else 1.0,
        "w_energy_pump_label": f"x{float(factor):g}" if sweep_type == "final_pump_weight_scan" else "baseline",
        "pump_dmax_factor": pump_dmax_factor,
        "pump_dmax_label": pump_dmax_label,
        "pump_dmax_base_rpm_per_step": pump_dmax_base,
        "pump_dmax_rpm_per_step": float(params.dmax_pump),
        "pump_dmax_rpm_per_s": float(params.dmax_pump) / SIM_DT,
        "disp_scale": disp_scale_value,
        "compressor_displacement_scale": disp_scale_value,
        "min_comp_rpm": min_comp_rpm_value,
        "n_comp_min_used": float(params.n_comp_min),
        "V_disp_m3_per_rev_base": float(thermal_system.V_disp_m3_per_rev) / disp_scale_value if disp_scale_value > 0 else float(thermal_system.V_disp_m3_per_rev),
        "V_disp_m3_per_rev_used": float(thermal_system.V_disp_m3_per_rev),
        "cv_weight_label": format_cv_weight_label(factor) if sweep_type in ("final_cv_weight_scan", "stable_cv_weight_scan") else factor_label,
        "cv_weight_factor": factor_numeric,
        "cv_weight_value": factor_numeric,
        "WSPHI_used": float(params.w_high_temp),
        "WSPLO_used": float(params.w_cold_temp),
        "comp_dmax_rpm_per_step": float(params.dmax_comp),
        "comp_dmax_rpm_per_s": float(params.dmax_comp) / SIM_DT,
        "DMAX_comp": float(params.dmax_comp),
        "DMAX_pump": float(params.dmax_pump),
        "dmax_comp_used": float(params.dmax_comp),
        "dmax_pump_used": float(params.dmax_pump),
        "terminal_cost_on": bool(terminal_cost_enabled_value and w_terminal_temp_value > 0.0) or bool(abs(float(params.w_term_peak)) > 0.0),
        "Terminal_Cost_Enabled": terminal_cost_enabled_value,
        "Terminal_Cost_Type": terminal_cost_type_value,
        "W_Terminal_Temp": w_terminal_temp_value,
        "Terminal_Temp_Scale_C": terminal_temp_scale_value,
        "terminal_cost_label": factor_label if sweep_type in TERMINAL_REPORT_SWEEP_TYPES else "",
        "terminal_weight": w_terminal_temp_value,
        "J_hot_on": False,
        "tracking_term_on": bool(abs(float(params.w_temp_obj)) > 0.0),
        "w_dcomp_used": float(params.w_dcomp),
        "w_dpump_used": float(params.w_dpump),
        "w_energy_comp": float(params.w_energy_comp),
        "w_energy_pump": float(params.w_energy_pump),
        "temperature_metric_basis": "T_ref tracking metrics use T_cell_mean_C, falling back to Battery Temp (C); cell max is safety-only",
        "Tmin": float(temp_c.min()),
        "Tmean": float(temp_c.mean()),
        "Tmax": float(temp_c.max()),
        "Tfinal": float(temp_c.iloc[-1]),
        "T_cell_mean_max": float(temp_mean_c.max()),
        "Tmean_max": float(temp_mean_c.max()),
        "T_cell_mean_min": float(temp_mean_c.min()),
        "Tmean_min": float(temp_mean_c.min()),
        "T_cell_mean_final": float(temp_mean_c.iloc[-1]),
        "Tmean_final": float(temp_mean_c.iloc[-1]),
        "T_cell_max_over_time": float(temp_cell_max_c.max()),
        "T_cell_min_over_time": float(temp_cell_min_c.min()),
        "DeltaT_cell_max": float(delta_t_cell_c.max()),
        "DeltaT_cell_mean": float(delta_t_cell_c.mean()),
        "temp_mae_to_ref": float(np.mean(np.abs(temp_error))),
        "MAE": float(np.mean(np.abs(temp_error))),
        "temp_rmse_to_ref": float(np.sqrt(np.mean(temp_error**2))),
        "RMSE": float(np.sqrt(np.mean(temp_error**2))),
        "temp_std": float(np.std(temp_error)),
        "max_abs_error_to_ref": float(np.max(np.abs(temp_error))),
        "hot_duration_s": duration_s(temp_c > hot_threshold, time_s),
        "cold_duration_s": duration_s(temp_c < cold_threshold, time_s),
        "hot_max_overshoot": float(max(0.0, temp_c.max() - hot_threshold)),
        "cold_max_undershoot": float(max(0.0, cold_threshold - temp_c.min())),
        "hot_degree_seconds": float(np.sum(hot_over * steps_s)),
        "hot_degree_seconds_band": float(np.sum(hot_over * steps_s)),
        "hot_degree_seconds_band_mean": float(np.sum(hot_over * steps_s)),
        "hot_degree_seconds_25p5": float(np.sum(hot_25p5_over * steps_s)),
        "hot_degree_seconds_25p5_cellmax": float(np.sum(hot_25p5_cellmax_over * steps_s)),
        "cold_degree_seconds": float(np.sum(cold_over * steps_s)),
        "hot_rms": float(np.sqrt(np.mean(hot_over**2))),
        "Total_kWh": float(s(df, ["Cumulative Energy (kWh)", "Cumulative energy consumption"]).iloc[-1]),
        "energy_kwh": float(s(df, ["Cumulative Energy (kWh)", "Cumulative energy consumption"]).iloc[-1]),
        "Comp_kWh": integrate_kwh(p_comp, time_s),
        "Pump_kWh": integrate_kwh(p_pump, time_s),
        "Fan_kWh": integrate_kwh(p_fan, time_s),
        "mean_comp_rpm": float(comp_speed.mean()),
        "mean_pump_rpm": float(pump_speed.mean()),
        "max_comp_rpm": float(comp_speed.max()),
        "max_pump_rpm": float(pump_speed.max()),
        "mean_dcomp": mean_abs_delta(comp_command),
        "mean_dpump": mean_abs_delta(pump_command),
        "max_dpump": max_abs_delta(pump_command),
        "solve_success_rate": float(pd.to_numeric(solve_success_series, errors="coerce").mean()),
        "mean_solve_time": float(pd.to_numeric(solve_time_series, errors="coerce").mean()),
        "comp_upper_ratio": upper_ratio(comp_command, N_COMP_MAX_RPM),
        "comp_lower_ratio": lower_ratio(comp_command, params.n_comp_min),
        "pump_upper_ratio": upper_ratio(pump_command, N_PUMP_MAX_RPM),
        "pump_lower_ratio": lower_ratio(pump_command, params.n_pump_min),
        "T_supply_mean": float(s(df, ["T_Pipe_Supply_C", "Supply pipe coolant temperature"]).mean()),
        "mean_T_supply": float(s(df, ["T_Pipe_Supply_C", "Supply pipe coolant temperature"]).mean()),
        "T_tank_mean": float(s(df, ["Coolant Temp (C)", "Coolant temperature"]).mean()),
        "mean_T_tank": float(s(df, ["Coolant Temp (C)", "Coolant temperature"]).mean()),
        "Q_evap_mean": float(s(df, ["Q_Dot_Evap", "Evaporator cooling rate (kW)"]).mean()),
        "Q_batt_plate_mean": float(s(df, ["Q_Dot_Bat", "Battery heat removal rate (kW)"]).mean()),
    }


def export_time_series(csv_path: Path, out_path: Path, case: str, rate_limit_mode: str, params) -> None:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    batt_temp = s(df, ["T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
    cell_max = pd.to_numeric(df["T_cell_max_C"], errors="coerce") if "T_cell_max_C" in df.columns else batt_temp
    cell_min = pd.to_numeric(df["T_cell_min_C"], errors="coerce") if "T_cell_min_C" in df.columns else batt_temp
    delta_t_cell = pd.to_numeric(df["Delta_T_cell_C"], errors="coerce") if "Delta_T_cell_C" in df.columns else cell_max - cell_min
    out = pd.DataFrame(
        {
            "case": case,
            "rate_limit_mode": rate_limit_mode,
            "dmax_comp_used": float(params.dmax_comp),
            "w_dcomp_used": float(params.w_dcomp),
            "time_s": s(df, ["Time (s)", "Time"]),
            "T_batt_max": cell_max,
            "T_batt_mean": batt_temp,
            "T_cell_mean_C": batt_temp,
            "T_cell_max_C": cell_max,
            "T_cell_min_C": cell_min,
            "Delta_T_cell_C": delta_t_cell,
            "comp_rpm": s(df, ["Compressor Speed (RPM)", "Compressor Speed"]),
            "pump_rpm": s(df, ["Pump Speed (RPM)", "Pump Speed"]),
            "T_supply": s(df, ["T_Pipe_Supply_C", "Supply pipe coolant temperature"]),
            "T_tank": s(df, ["Coolant Temp (C)", "Coolant temperature"]),
            "Q_evap": s(df, ["Q_Dot_Evap", "Evaporator cooling rate (kW)"]),
            "Q_batt_plate": s(df, ["Q_Dot_Bat", "Battery heat removal rate (kW)"]),
            "current_A": s(df, ["Total Current (A)", "Current (A)"]),
            "Terminal_Cost_Enabled": pd.to_numeric(df["Terminal_Cost_Enabled"], errors="coerce") if "Terminal_Cost_Enabled" in df.columns else np.nan,
            "W_Terminal_Temp": pd.to_numeric(df["W_Terminal_Temp"], errors="coerce") if "W_Terminal_Temp" in df.columns else np.nan,
            "Terminal_Temp_Scale_C": pd.to_numeric(df["Terminal_Temp_Scale_C"], errors="coerce") if "Terminal_Temp_Scale_C" in df.columns else np.nan,
            "T_pred_end_C": pd.to_numeric(df["T_pred_end_C"], errors="coerce") if "T_pred_end_C" in df.columns else np.nan,
            "J_terminal": pd.to_numeric(df["J_terminal"], errors="coerce") if "J_terminal" in df.columns else np.nan,
            "Weighted_J_terminal": pd.to_numeric(df["Weighted_J_terminal"], errors="coerce") if "Weighted_J_terminal" in df.columns else np.nan,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

def resolve_output_csv_path(path_text: object, output_root: Path) -> Path | None:
    if pd.isna(path_text):
        return None
    path = Path(str(path_text))
    if path.is_absolute() or path.exists():
        return path
    candidate = output_root / path
    return candidate if candidate.exists() else path


def compute_hot_degree_seconds_from_series(series_path: Path, threshold: float) -> float:
    df = pd.read_csv(series_path, encoding="utf-8-sig")
    time_s = s(df, ["time_s", "Time (s)", "Time"])
    temp_c = s(df, ["T_batt_mean", "T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
    over = np.maximum(0.0, temp_c.to_numpy(float) - threshold)
    return float(np.sum(over * time_steps_s(time_s)))



def _first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def _relative_pct(value: object, baseline: object) -> float:
    value_num = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    baseline_num = pd.to_numeric(pd.Series([baseline]), errors="coerce").iloc[0]
    if not np.isfinite(value_num) or not np.isfinite(baseline_num) or baseline_num == 0.0:
        return np.nan
    return float((value_num - baseline_num) / baseline_num * 100.0)


def add_terminal_baseline_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "case" not in summary_df.columns:
        return summary_df
    out = summary_df.copy()
    hot_col = _first_existing_column(out, ("hot_degree_seconds_band", "hot_degree_seconds_band_mean", "hot_degree_seconds"))
    metric_map = {
        "dTmax_pct": "T_cell_mean_max" if "T_cell_mean_max" in out.columns else "Tmax",
        "dMAE_pct": "MAE",
        "dRMSE_pct": "RMSE",
        "dhot_band_pct": hot_col,
        "dEnergy_pct": "Total_kWh",
    }
    for new_col in metric_map:
        out[new_col] = np.nan
    out["score_terminal_tradeoff"] = np.nan
    for case, group in out.groupby("case", sort=False):
        baseline_mask = (
            (group.get("Terminal_Cost_Enabled", pd.Series(False, index=group.index)).astype(bool) == False)
            & (pd.to_numeric(group.get("W_Terminal_Temp", pd.Series(np.nan, index=group.index)), errors="coerce") == 0.0)
        )
        if not baseline_mask.any() and "terminal_cost_label" in group.columns:
            baseline_mask = group["terminal_cost_label"].astype(str).eq("terminal_off_w0")
        if not baseline_mask.any():
            continue
        baseline = group.loc[baseline_mask].iloc[0]
        for row_index in group.index:
            for new_col, metric_col in metric_map.items():
                if metric_col is None or metric_col not in out.columns:
                    continue
                out.at[row_index, new_col] = _relative_pct(out.at[row_index, metric_col], baseline[metric_col])
            rmse_base = pd.to_numeric(pd.Series([baseline.get("RMSE", np.nan)]), errors="coerce").iloc[0]
            energy_base = pd.to_numeric(pd.Series([baseline.get("Total_kWh", np.nan)]), errors="coerce").iloc[0]
            hot_base = pd.to_numeric(pd.Series([baseline.get(hot_col, np.nan) if hot_col else np.nan]), errors="coerce").iloc[0]
            rmse_value = pd.to_numeric(pd.Series([out.at[row_index, "RMSE"] if "RMSE" in out.columns else np.nan]), errors="coerce").iloc[0]
            energy_value = pd.to_numeric(pd.Series([out.at[row_index, "Total_kWh"] if "Total_kWh" in out.columns else np.nan]), errors="coerce").iloc[0]
            hot_value = pd.to_numeric(pd.Series([out.at[row_index, hot_col] if hot_col else np.nan]), errors="coerce").iloc[0]
            rmse_norm = rmse_value / rmse_base if np.isfinite(rmse_value) and np.isfinite(rmse_base) and rmse_base != 0.0 else np.nan
            energy_norm = energy_value / energy_base if np.isfinite(energy_value) and np.isfinite(energy_base) and energy_base != 0.0 else np.nan
            if np.isfinite(hot_base) and hot_base != 0.0 and np.isfinite(hot_value):
                hot_norm = hot_value / hot_base
            else:
                hot_norm = 1.0
            if np.isfinite(rmse_norm) and np.isfinite(energy_norm):
                out.at[row_index, "score_terminal_tradeoff"] = float(rmse_norm + 0.5 * hot_norm + 0.3 * energy_norm)
    return out



def _numeric_cell(row: pd.Series, col: str) -> float:
    if col not in row.index:
        return np.nan
    return pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]


def _metric_delta(row: pd.Series, baseline: pd.Series, metric: str) -> float:
    value = _numeric_cell(row, metric)
    base = _numeric_cell(baseline, metric)
    if not np.isfinite(value) or not np.isfinite(base):
        return np.nan
    return float(value - base)


def _metric_improve_pct(row: pd.Series, baseline: pd.Series, metric: str) -> float:
    value = _numeric_cell(row, metric)
    base = _numeric_cell(baseline, metric)
    if not np.isfinite(value) or not np.isfinite(base) or base == 0.0:
        return np.nan
    return float((base - value) / base * 100.0)


def add_three_way_baseline_metrics(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "case" not in summary_df.columns:
        return summary_df
    out = summary_df.copy()
    hot_col = _first_existing_column(out, ("hot_degree_seconds_band", "hot_degree_seconds_band_mean", "hot_degree_seconds"))
    cold_col = _first_existing_column(out, ("cold_degree_seconds",))
    if hot_col:
        out["hot_band"] = pd.to_numeric(out[hot_col], errors="coerce")
    else:
        out["hot_band"] = np.nan
    if cold_col:
        out["cold_band"] = pd.to_numeric(out[cold_col], errors="coerce")
    else:
        out["cold_band"] = np.nan
    out["baseline_version"] = "current_shared_mpc_baseline"
    for col in (
        "delta_Tmax", "delta_MAE", "delta_RMSE", "delta_hot_band", "delta_E",
        "MAE_improve_pct", "RMSE_improve_pct", "hot_band_improve_pct", "energy_change_pct",
    ):
        out[col] = np.nan
    for case, group in out.groupby("case", sort=False):
        if "terminal_cost_label" not in group.columns:
            continue
        baseline_mask = group["terminal_cost_label"].astype(str).eq("baseline_off")
        if not baseline_mask.any():
            continue
        baseline = group.loc[baseline_mask].iloc[0]
        for row_index in group.index:
            row = out.loc[row_index]
            out.at[row_index, "delta_Tmax"] = _metric_delta(row, baseline, "Tmean_max")
            out.at[row_index, "delta_MAE"] = _metric_delta(row, baseline, "MAE")
            out.at[row_index, "delta_RMSE"] = _metric_delta(row, baseline, "RMSE")
            out.at[row_index, "delta_hot_band"] = _metric_delta(row, baseline, "hot_band")
            out.at[row_index, "delta_E"] = _metric_delta(row, baseline, "Total_kWh")
            out.at[row_index, "MAE_improve_pct"] = _metric_improve_pct(row, baseline, "MAE")
            out.at[row_index, "RMSE_improve_pct"] = _metric_improve_pct(row, baseline, "RMSE")
            out.at[row_index, "hot_band_improve_pct"] = _metric_improve_pct(row, baseline, "hot_band")
            energy_value = _numeric_cell(row, "Total_kWh")
            energy_base = _numeric_cell(baseline, "Total_kWh")
            if np.isfinite(energy_value) and np.isfinite(energy_base) and energy_base != 0.0:
                out.at[row_index, "energy_change_pct"] = float((energy_value - energy_base) / energy_base * 100.0)
    return out
def plot_terminal_cost_outputs(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN plot skipped: matplotlib unavailable ({exc})", flush=True)
        return
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty:
        return
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    hot_col = _first_existing_column(summary, ("hot_degree_seconds_band", "hot_degree_seconds_band_mean", "hot_degree_seconds"))
    metric_specs = [
        ("MAE", "MAE", "MAE (C)"),
        ("RMSE", "RMSE", "RMSE (C)"),
        ("hot_band", hot_col, "Hot degree seconds over CV band"),
        ("Total_kWh", "Total_kWh", "Total energy (kWh)"),
        ("T_pred_end_C", "T_pred_end_C", "Predicted terminal temperature (C)"),
        ("score_terminal_tradeoff", "score_terminal_tradeoff", "Tradeoff score"),
    ]
    for case, group in summary.groupby("case", sort=False):
        group = group.copy()
        group["terminal_weight_plot"] = pd.to_numeric(group.get("W_Terminal_Temp", np.nan), errors="coerce").fillna(0.0)
        group = group.sort_values(["terminal_weight_plot", "Terminal_Cost_Enabled"], kind="mergesort")
        labels = group.get("terminal_cost_label", pd.Series([""] * len(group), index=group.index)).astype(str)
        for slug, metric_col, ylabel in metric_specs:
            if metric_col is None or metric_col not in group.columns:
                continue
            y = pd.to_numeric(group[metric_col], errors="coerce")
            fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=140)
            x = group["terminal_weight_plot"].to_numpy(float)
            enabled = group.get("Terminal_Cost_Enabled", pd.Series(False, index=group.index)).astype(bool).to_numpy()
            ax.plot(x[enabled], y[enabled], marker="o", linewidth=1.6, label="terminal on")
            if np.any(~enabled):
                ax.scatter(x[~enabled], y[~enabled], marker="s", s=60, label="terminal off")
            for xi, yi, label in zip(x, y, labels):
                if np.isfinite(yi):
                    ax.annotate(label.replace("terminal_", ""), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7)
            ax.set_xscale("symlog", linthresh=1.0)
            ax.set_xlabel("w_terminal_temp (symlog; off/on_w0 at x=0)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{case}: terminal weight vs {slug}")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_root / f"{case}_terminal_weight_vs_{slug}.png")
            plt.close(fig)


def plot_three_way_compare_outputs(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN three-way plot skipped: matplotlib unavailable ({exc})", flush=True)
        return
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty or "time_series_csv" not in summary.columns:
        return
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("Tmean_max", "Tmax (C)"),
        ("MAE", "MAE (C)"),
        ("RMSE", "RMSE (C)"),
        ("hot_band", "Hot degree seconds"),
        ("Total_kWh", "Total energy (kWh)"),
    ]
    for case, group in summary.groupby("case", sort=False):
        group = group.copy()
        if "terminal_cost_label" in group.columns:
            order_map = {label: idx for idx, label in enumerate(THREE_WAY_FACTORS.get(case, tuple(group["terminal_cost_label"].astype(str))))}
            group["_plot_order"] = group["terminal_cost_label"].astype(str).map(order_map).fillna(len(order_map))
            group = group.sort_values("_plot_order", kind="mergesort")
        series_items: list[tuple[str, pd.DataFrame]] = []
        for _, row in group.iterrows():
            label = str(row.get("terminal_cost_label", "case"))
            series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
            if series_path is None or not series_path.exists():
                continue
            try:
                series_items.append((label, pd.read_csv(series_path, encoding="utf-8-sig")))
            except Exception as exc:
                print(f"WARN three-way plot skipped series {series_path}: {exc}", flush=True)
        if series_items:
            fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
            for label, df in series_items:
                time_s = s(df, ["time_s", "Time (s)", "Time"])
                temp_c = s(df, ["T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
                ax.plot(time_s, temp_c, linewidth=1.6, label=label)
            first = group.iloc[0]
            t_ref = float(first.get("T_ref", 25.0))
            splo = float(first.get("CV_band_low", t_ref - 0.5))
            sphi = float(first.get("CV_band_high", t_ref + 0.5))
            ax.axhline(t_ref, color="black", linestyle="-", linewidth=1.0, label="T_ref=25C")
            ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
            ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
            ax.set_title(f"{case} terminal cost three-way temperature")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("T_cell_mean_C (C)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_root / f"{case}_three_way_temperature.png")
            plt.close(fig)

            fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), dpi=140, sharex=True)
            for label, df in series_items:
                time_s = s(df, ["time_s", "Time (s)", "Time"])
                comp_rpm = s(df, ["Compressor Speed (RPM)", "Compressor Speed", "comp_rpm"])
                pump_rpm = s(df, ["Pump Speed (RPM)", "Pump Speed", "pump_rpm"])
                axes[0].plot(time_s, comp_rpm, linewidth=1.5, label=label)
                axes[1].plot(time_s, pump_rpm, linewidth=1.5, label=label)
            axes[0].set_ylabel("Compressor RPM")
            axes[1].set_ylabel("Pump RPM")
            axes[1].set_xlabel("Time (s)")
            axes[0].set_title(f"{case} terminal cost three-way actuators")
            for ax in axes:
                ax.grid(True, alpha=0.25)
            axes[0].legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_root / f"{case}_three_way_actuators.png")
            plt.close(fig)

        labels = group.get("terminal_cost_label", pd.Series(group.index.astype(str), index=group.index)).astype(str).tolist()
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(15.0, 4.6), dpi=140)
        for ax, (metric_col, ylabel) in zip(axes, metric_specs):
            if metric_col not in group.columns:
                ax.axis("off")
                continue
            values = pd.to_numeric(group[metric_col], errors="coerce")
            ax.bar(labels, values)
            ax.set_title(metric_col)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(f"{case} terminal cost three-way metrics")
        fig.tight_layout()
        fig.savefig(figure_root / f"{case}_three_way_key_metrics.png")
        plt.close(fig)

def plot_three_way_full_outputs(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN three-way full plot skipped: matplotlib unavailable ({exc})", flush=True)
        return
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty or "time_series_csv" not in summary.columns:
        return
    metric_specs = [
        ("Tmean_max", "Tmax (C)"),
        ("MAE", "MAE (C)"),
        ("RMSE", "RMSE (C)"),
        ("hot_band", "Hot degree seconds"),
        ("Total_kWh", "Total energy (kWh)"),
    ]
    for case, group in summary.groupby("case", sort=False):
        group = group.copy()
        figure_root = output_root / "figures" / str(case)
        figure_root.mkdir(parents=True, exist_ok=True)
        if "terminal_cost_label" in group.columns:
            order_map = {label: idx for idx, label in enumerate(THREE_WAY_FULL_FACTORS.get(case, tuple(group["terminal_cost_label"].astype(str))))}
            group["_plot_order"] = group["terminal_cost_label"].astype(str).map(order_map).fillna(len(order_map))
            group = group.sort_values("_plot_order", kind="mergesort")
        series_items: list[tuple[str, pd.DataFrame]] = []
        for _, row in group.iterrows():
            label = str(row.get("terminal_cost_label", "case"))
            series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
            if series_path is None or not series_path.exists():
                continue
            try:
                series_items.append((label, pd.read_csv(series_path, encoding="utf-8-sig")))
            except Exception as exc:
                print(f"WARN three-way full plot skipped series {series_path}: {exc}", flush=True)
        if series_items:
            first = group.iloc[0]
            t_ref = float(first.get("T_ref", 25.0))
            splo = float(first.get("CV_band_low", t_ref - 0.5))
            sphi = float(first.get("CV_band_high", t_ref + 0.5))
            fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=140)
            for label, df in series_items:
                time_s = s(df, ["time_s", "Time (s)", "Time"])
                temp_c = s(df, ["T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
                ax.plot(time_s, temp_c, linewidth=1.5, label=label)
            ax.axhline(t_ref, color="black", linestyle="-", linewidth=1.0, label="T_ref=25C")
            ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
            ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
            ax.set_title(f"{case} full terminal cost comparison: battery temperature")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("T_cell_mean_C (C)")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_root / "battery_temperature_vs_time.png")
            plt.close(fig)

            for slug, names, ylabel in [
                ("compressor_rpm_vs_time", ["Compressor Speed (RPM)", "Compressor Speed", "comp_rpm"], "Compressor RPM"),
                ("pump_rpm_vs_time", ["Pump Speed (RPM)", "Pump Speed", "pump_rpm"], "Pump RPM"),
                ("cumulative_energy_vs_time", ["Cumulative Energy (kWh)", "Cumulative energy consumption"], "Cumulative energy (kWh)"),
            ]:
                fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=140)
                for label, df in series_items:
                    time_s = s(df, ["time_s", "Time (s)", "Time"])
                    ax.plot(time_s, s(df, names), linewidth=1.5, label=label)
                ax.set_title(f"{case} full terminal cost comparison: {ylabel}")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel(ylabel)
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                fig.savefig(figure_root / f"{slug}.png")
                plt.close(fig)

            if str(case) == "freq":
                fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=140)
                for label, df in series_items:
                    time_s = s(df, ["time_s", "Time (s)", "Time"])
                    temp_c = s(df, ["T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
                    if len(time_s) > 4:
                        t0 = float(time_s.quantile(0.35))
                        t1 = float(time_s.quantile(0.70))
                        mask = (time_s >= t0) & (time_s <= t1)
                        ax.plot(time_s[mask], temp_c[mask], linewidth=1.5, label=label)
                    else:
                        ax.plot(time_s, temp_c, linewidth=1.5, label=label)
                ax.axhline(t_ref, color="black", linestyle="-", linewidth=1.0, label="T_ref=25C")
                ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
                ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
                ax.set_title("freq full terminal cost comparison: temperature zoom")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("T_cell_mean_C (C)")
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best", fontsize=8)
                fig.tight_layout()
                fig.savefig(figure_root / "battery_temperature_zoom.png")
                plt.close(fig)

        labels = group.get("terminal_cost_label", pd.Series(group.index.astype(str), index=group.index)).astype(str).tolist()
        fig, axes = plt.subplots(1, len(metric_specs), figsize=(15.0, 4.6), dpi=140)
        for ax, (metric_col, ylabel) in zip(axes, metric_specs):
            if metric_col not in group.columns:
                ax.axis("off")
                continue
            values = pd.to_numeric(group[metric_col], errors="coerce")
            ax.bar(labels, values)
            ax.set_title(metric_col)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", labelrotation=45, labelsize=8)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(f"{case} full terminal cost key metrics")
        fig.tight_layout()
        fig.savefig(figure_root / "key_metrics_bar.png")
        plt.close(fig)
def plot_final_adaptive_outputs(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    for _, row in summary.iterrows():
        case = str(row.get("case", "case"))
        series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
        if series_path is None or not series_path.exists():
            continue
        df = pd.read_csv(series_path, encoding="utf-8-sig")
        time_s = s(df, ["time_s", "Time (s)", "Time"])
        temp_c = s(df, ["T_batt_mean", "T_cell_mean_C", "Battery Temp (C)", "Average temperature"])
        t_ref = float(row.get("T_ref", 25.0))
        band = float(row.get("cv_band_half_width", 0.5))
        splo = float(row.get("CV_band_low", t_ref - band))
        sphi = float(row.get("CV_band_high", t_ref + band))

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
        ax.plot(time_s, temp_c, label="T_batt", linewidth=1.8)
        ax.axhline(t_ref, color="black", linestyle="-", linewidth=1.0, label="T_ref=25C")
        ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
        ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
        ax.axhline(25.5, color="tab:orange", linestyle=":", linewidth=1.2, label="25.5C")
        ax.set_title(f"{case} final adaptive MPC temperature")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (C)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_root / f"{case}_temperature_final_adaptive_mpc.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
        ax.plot(time_s, s(df, ["comp_rpm", "Compressor Speed (RPM)", "Compressor Speed"]), label="compressor rpm", linewidth=1.6)
        ax.plot(time_s, s(df, ["pump_rpm", "Pump Speed (RPM)", "Pump Speed"]), label="pump rpm", linewidth=1.6)
        ax.set_title(f"{case} final adaptive MPC actuators")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("RPM")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_root / f"{case}_actuators_final_adaptive_mpc.png")
        plt.close(fig)

    old_summary_path = None
    for old_root in OLD_FINAL_ADAPTIVE_OUTPUT_ROOTS:
        candidate = old_root / "summary_final_adaptive_mpc.csv"
        if candidate.exists() and candidate.resolve() != summary_path.resolve():
            old_summary_path = candidate
            break
    if old_summary_path is None:
        print("INFO old final_adaptive_mpc baseline not found; comparison plot skipped", flush=True)
        return

    old_summary = pd.read_csv(old_summary_path, encoding="utf-8-sig")
    compare_metrics = [
        "T_cell_mean_max",
        "T_cell_mean_final",
        "MAE",
        "RMSE",
        "hot_degree_seconds_25p5",
        "Total_kWh",
        "Comp_kWh",
        "Pump_kWh",
    ]
    records: list[dict[str, object]] = []
    for case in ("peak", "freq"):
        old_rows = old_summary[old_summary["case"] == case]
        new_rows = summary[summary["case"] == case]
        if old_rows.empty or new_rows.empty:
            continue
        for label, src, root in (
            ("old_cv_band_0p50", old_rows.iloc[0], old_summary_path.parent),
            ("final_candidate", new_rows.iloc[0], output_root),
        ):
            record: dict[str, object] = {"case": case, "variant": label}
            series_path = resolve_output_csv_path(src.get("time_series_csv"), root)
            for metric in compare_metrics:
                if metric in src and not pd.isna(src[metric]):
                    record[metric] = float(src[metric])
                elif metric == "MAE" and "temp_mae_to_ref" in src:
                    record[metric] = float(src["temp_mae_to_ref"])
                elif metric == "RMSE" and "temp_rmse_to_ref" in src:
                    record[metric] = float(src["temp_rmse_to_ref"])
                elif metric == "hot_degree_seconds_25p5" and series_path is not None and series_path.exists():
                    record[metric] = compute_hot_degree_seconds_from_series(series_path, 25.5)
                else:
                    record[metric] = float("nan")
            records.append(record)
    if not records:
        return

    compare = pd.DataFrame(records)
    compare_path = output_root / "summary_old_vs_final_candidate.csv"
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=140)
    axes = axes.ravel()
    xlabels = [f"{row.case}\n{row.variant.replace('_', ' ')}" for row in compare.itertuples()]
    x = np.arange(len(compare))
    for ax, metric in zip(axes, compare_metrics):
        ax.bar(x, compare[metric].to_numpy(float), color=["#4C78A8" if "old" in v else "#F58518" for v in compare["variant"]])
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "old_vs_final_candidate_metrics.png")
    plt.close(fig)
def plot_freq_compressor_displacement_scale_scan(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty:
        return
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
    for _, row in summary.iterrows():
        series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
        if series_path is None or not series_path.exists():
            continue
        df = pd.read_csv(series_path, encoding="utf-8-sig")
        ax.plot(
            s(df, ["time_s", "Time (s)", "Time"]),
            s(df, ["T_batt_mean", "T_cell_mean_C", "Battery Temp (C)", "Average temperature"]),
            label=f"disp_scale={float(row['disp_scale']):g}",
            linewidth=1.6,
        )
    t_ref = 25.0
    splo = float(summary["CV_band_low"].iloc[0])
    sphi = float(summary["CV_band_high"].iloc[0])
    ax.axhline(t_ref, color="black", linewidth=1.0, label="T_ref=25C")
    ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
    ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
    ax.axhline(25.5, color="tab:orange", linestyle=":", linewidth=1.2, label="25.5C")
    ax.set_title("freq compressor displacement scale temperature")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_root / "freq_disp_scale_temperature_compare.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
    for _, row in summary.iterrows():
        series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
        if series_path is None or not series_path.exists():
            continue
        df = pd.read_csv(series_path, encoding="utf-8-sig")
        ax.plot(
            s(df, ["time_s", "Time (s)", "Time"]),
            s(df, ["comp_rpm", "Compressor Speed (RPM)", "Compressor Speed"]),
            label=f"disp_scale={float(row['disp_scale']):g}",
            linewidth=1.6,
        )
    ax.set_title("freq compressor displacement scale compressor rpm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compressor rpm")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_root / "freq_disp_scale_comp_rpm_compare.png")
    plt.close(fig)

    metrics = ["T_cell_mean_max", "T_cell_max_over_time", "DeltaT_cell_max", "MAE", "RMSE", "Total_kWh", "comp_lower_ratio", "comp_upper_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=140)
    axes = axes.ravel()
    labels = [f"{float(v):g}" for v in summary["disp_scale"]]
    x = np.arange(len(summary))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, summary[metric].to_numpy(float), color="#4C78A8")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("disp_scale")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_root / "freq_disp_scale_metrics.png")
    plt.close(fig)
def plot_freq_compressor_min_rpm_scan(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty:
        return
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
    for _, row in summary.iterrows():
        series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
        if series_path is None or not series_path.exists():
            continue
        df = pd.read_csv(series_path, encoding="utf-8-sig")
        ax.plot(
            s(df, ["time_s", "Time (s)", "Time"]),
            s(df, ["T_batt_mean", "T_cell_mean_C", "Battery Temp (C)", "Average temperature"]),
            label=f"min_comp={float(row['min_comp_rpm']):g} rpm",
            linewidth=1.6,
        )
    t_ref = 25.0
    splo = float(summary["CV_band_low"].iloc[0])
    sphi = float(summary["CV_band_high"].iloc[0])
    ax.axhline(t_ref, color="black", linewidth=1.0, label="T_ref=25C")
    ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
    ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
    ax.axhline(25.5, color="tab:orange", linestyle=":", linewidth=1.2, label="25.5C")
    ax.set_title("freq compressor min rpm temperature")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (C)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_root / "freq_min_comp_rpm_temperature_compare.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
    for _, row in summary.iterrows():
        series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
        if series_path is None or not series_path.exists():
            continue
        df = pd.read_csv(series_path, encoding="utf-8-sig")
        ax.plot(
            s(df, ["time_s", "Time (s)", "Time"]),
            s(df, ["comp_rpm", "Compressor Speed (RPM)", "Compressor Speed"]),
            label=f"min_comp={float(row['min_comp_rpm']):g} rpm",
            linewidth=1.6,
        )
    ax.set_title("freq compressor min rpm compressor rpm")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Compressor rpm")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_root / "freq_min_comp_rpm_comp_rpm_compare.png")
    plt.close(fig)

    metrics = ["T_cell_mean_max", "T_cell_max_over_time", "DeltaT_cell_max", "MAE", "RMSE", "Total_kWh", "comp_lower_ratio", "comp_upper_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), dpi=140)
    axes = axes.ravel()
    labels = [f"{float(v):g}" for v in summary["min_comp_rpm"]]
    x = np.arange(len(summary))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, summary[metric].to_numpy(float), color="#4C78A8")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("min_comp_rpm")
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_root / "freq_min_comp_rpm_metrics.png")
    plt.close(fig)


def _ablation_label(row) -> str:
    return f"{row['case']} / {float(row['min_comp_rpm']):g}rpm / disp{float(row['disp_scale']):g}"


def write_hardware_ablation_conclusions(output_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "Short-horizon compressor hardware ablation conclusions",
        "Scope: compare only within the same case and 120-step (or user-selected) horizon.",
        "Rule of thumb: benefit means Total_kWh decreases without increasing MAE or RMSE.",
        "",
    ]
    for case in ("peak", "freq"):
        case_df = summary[summary["case"] == case]
        if case_df.empty:
            continue
        base = case_df[(case_df["min_comp_rpm"] == 1000.0) & (case_df["disp_scale"] == 1.0)]
        if base.empty:
            lines.append(f"{case}: baseline min=1000, disp=1.0 missing; skip case conclusions.")
            continue
        base_row = base.iloc[0]
        lines.append(f"{case} baseline: E={base_row['Total_kWh']:.6f}, MAE={base_row['MAE']:.6f}, RMSE={base_row['RMSE']:.6f}, T_cell_mean_max={base_row['T_cell_mean_max']:.6f}, T_cell_max_over_time={base_row['T_cell_max_over_time']:.6f}")
        for min_rpm, disp_scale, name in (
            (500.0, 1.0, "min_comp_rpm 1000 -> 500"),
            (1000.0, 0.9, "disp_scale 1.00 -> 0.90"),
            (500.0, 0.9, "combined min_comp_rpm=500 and disp_scale=0.90"),
        ):
            rows = case_df[(case_df["min_comp_rpm"] == min_rpm) & (case_df["disp_scale"] == disp_scale)]
            if rows.empty:
                lines.append(f"{case} {name}: missing result.")
                continue
            row = rows.iloc[0]
            d_e = row["Total_kWh"] - base_row["Total_kWh"]
            d_mae = row["MAE"] - base_row["MAE"]
            d_rmse = row["RMSE"] - base_row["RMSE"]
            benefit = d_e < 0 and d_mae <= 0 and d_rmse <= 0
            risk_flags = []
            if row["T_cell_max_over_time"] > base_row["T_cell_max_over_time"]:
                risk_flags.append("T_cell_max_over_time_up")
            if row["hot_degree_seconds_25p5_cellmax"] > 0:
                risk_flags.append("hot_25p5_cellmax")
            if row["comp_upper_ratio"] > base_row["comp_upper_ratio"]:
                risk_flags.append("comp_upper_ratio_up")
            verdict = "benefit" if benefit else "no_clear_benefit"
            if risk_flags:
                verdict += " with_risk=" + ",".join(risk_flags)
            lines.append(
                f"{case} {name}: dE={d_e:+.6f}, dMAE={d_mae:+.6f}, dRMSE={d_rmse:+.6f}, "
                f"dT_cell_mean_max={row['T_cell_mean_max'] - base_row['T_cell_mean_max']:+.6f}, dT_cell_max_over_time={row['T_cell_max_over_time'] - base_row['T_cell_max_over_time']:+.6f}, verdict={verdict}"
            )
        combo = case_df[(case_df["min_comp_rpm"] == 500.0) & (case_df["disp_scale"] == 0.9)]
        min_only = case_df[(case_df["min_comp_rpm"] == 500.0) & (case_df["disp_scale"] == 1.0)]
        disp_only = case_df[(case_df["min_comp_rpm"] == 1000.0) & (case_df["disp_scale"] == 0.9)]
        if not combo.empty and not min_only.empty and not disp_only.empty:
            coupling_e = (combo.iloc[0]["Total_kWh"] - base_row["Total_kWh"]) - (min_only.iloc[0]["Total_kWh"] - base_row["Total_kWh"]) - (disp_only.iloc[0]["Total_kWh"] - base_row["Total_kWh"])
            coupling_mae = (combo.iloc[0]["MAE"] - base_row["MAE"]) - (min_only.iloc[0]["MAE"] - base_row["MAE"]) - (disp_only.iloc[0]["MAE"] - base_row["MAE"])
            lines.append(f"{case} coupling residual: Total_kWh={coupling_e:+.6f}, MAE={coupling_mae:+.6f}")
        safe_benefits = case_df[(case_df["Total_kWh"] <= base_row["Total_kWh"]) & (case_df["MAE"] <= base_row["MAE"]) & (case_df["RMSE"] <= base_row["RMSE"])]
        safe_benefits = safe_benefits[~((safe_benefits["min_comp_rpm"] == 1000.0) & (safe_benefits["disp_scale"] == 1.0))]
        if safe_benefits.empty:
            lines.append(f"{case}: no non-baseline combination simultaneously lowers energy without worsening MAE/RMSE.")
        else:
            labels = "; ".join(_ablation_label(row) for _, row in safe_benefits.iterrows())
            lines.append(f"{case}: candidates for full-horizon verification: {labels}")
        lines.append("")
    (output_root / "hardware_ablation_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_hardware_ablation_outputs(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty:
        return
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    for case in ("peak", "freq"):
        case_df = summary[summary["case"] == case]
        if case_df.empty:
            continue
        for field, ylabel, filename in (
            ("T_batt_mean", "Mean temperature (C)", f"{case}_temperature_compare.png"),
            ("comp_rpm", "Compressor rpm", f"{case}_compressor_rpm_compare.png"),
        ):
            fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
            for _, row in case_df.iterrows():
                series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
                if series_path is None or not series_path.exists():
                    continue
                df = pd.read_csv(series_path, encoding="utf-8-sig")
                y = s(df, [field, "T_cell_mean_C", "Battery Temp (C)", "Average temperature"] if field == "T_batt_mean" else [field, "Compressor Speed (RPM)", "Compressor Speed"])
                label = f"min={float(row['min_comp_rpm']):g} rpm, disp={float(row['disp_scale']):g}"
                ax.plot(s(df, ["time_s", "Time (s)", "Time"]), y, label=label, linewidth=1.6)
            if field == "T_batt_mean":
                t_ref = 25.0
                splo = float(case_df["CV_band_low"].iloc[0])
                sphi = float(case_df["CV_band_high"].iloc[0])
                ax.axhline(t_ref, color="black", linewidth=1.0, label="T_ref=25C")
                ax.axhline(sphi, color="tab:red", linestyle="--", linewidth=1.0, label="SPHI")
                ax.axhline(splo, color="tab:blue", linestyle="--", linewidth=1.0, label="SPLO")
                ax.axhline(25.5, color="tab:orange", linestyle=":", linewidth=1.2, label="25.5C")
            ax.set_title(f"{case} hardware ablation {ylabel.lower()}")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_root / filename)
            plt.close(fig)

    metrics = ["T_cell_mean_max", "T_cell_max_over_time", "DeltaT_cell_max", "MAE", "RMSE", "hot_degree_seconds_25p5_cellmax", "Total_kWh", "comp_lower_ratio"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), dpi=140)
    axes = axes.ravel()
    labels = [_ablation_label(row) for _, row in summary.iterrows()]
    x = np.arange(len(summary))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, summary[metric].to_numpy(float), color="#4C78A8")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[len(metrics):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_root / "hardware_ablation_key_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    for _, row in summary.iterrows():
        ax.scatter(float(row["MAE"]), float(row["Total_kWh"]), s=48)
        ax.annotate(_ablation_label(row), (float(row["MAE"]), float(row["Total_kWh"])), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("MAE (C)")
    ax.set_ylabel("Total_kWh")
    ax.set_title("Total_kWh vs MAE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "hardware_ablation_total_kwh_vs_mae.png")
    plt.close(fig)

    write_hardware_ablation_conclusions(output_root, summary)


def _paramset_cross_label(row) -> str:
    return f"{row['case']} / {row['param_set']}"


def write_paramset_cross_conclusions(output_root: Path, summary: pd.DataFrame) -> None:
    lines = [
        "MPC param-set x case full-horizon cross-validation conclusions",
        "Benefit rule: lower Total_kWh without worsening T_cell_mean_max, MAE, RMSE, or cell-max hot_25p5 within the same case.",
        "",
    ]
    for case, matched_param, mismatched_param in (("peak", "peak_params", "freq_params"), ("freq", "freq_params", "peak_params")):
        case_df = summary[summary["case"] == case]
        matched = case_df[case_df["param_set"] == matched_param]
        mismatched = case_df[case_df["param_set"] == mismatched_param]
        if matched.empty or mismatched.empty:
            lines.append(f"{case}: missing matched or mismatched result; cannot judge.")
            continue
        good = matched.iloc[0]
        bad = mismatched.iloc[0]
        deltas = {
            "T_cell_mean_max": good["T_cell_mean_max"] - bad["T_cell_mean_max"],
            "MAE": good["MAE"] - bad["MAE"],
            "RMSE": good["RMSE"] - bad["RMSE"],
            "hot_25p5_cellmax": good["hot_degree_seconds_25p5_cellmax"] - bad["hot_degree_seconds_25p5_cellmax"],
            "Total_kWh": good["Total_kWh"] - bad["Total_kWh"],
        }
        matched_better = (
            deltas["T_cell_mean_max"] <= 0.0
            and deltas["MAE"] <= 0.0
            and deltas["RMSE"] <= 0.0
            and deltas["hot_25p5_cellmax"] <= 0.0
            and deltas["Total_kWh"] <= 0.0
        )
        worsened = [name for name, value in deltas.items() if value > 0.0]
        verdict = "matched_param_better" if matched_better else "mixed_or_no_clear_advantage"
        lines.append(
            f"{case}: matched={matched_param}, mismatched={mismatched_param}, "
            f"dT_cell_mean_max={deltas['T_cell_mean_max']:+.6f}, dMAE={deltas['MAE']:+.6f}, dRMSE={deltas['RMSE']:+.6f}, "
            f"dhot_25p5_cellmax={deltas['hot_25p5_cellmax']:+.6f}, dTotal_kWh={deltas['Total_kWh']:+.6f}, verdict={verdict}"
        )
        if worsened:
            lines.append(f"{case}: matched parameter set worsened these metrics versus mismatched: {', '.join(worsened)}")
        else:
            lines.append(f"{case}: mismatched parameter set did not beat matched set on the guarded metrics.")
    lines.append("")
    peak_rows = summary[summary["case"] == "peak"]
    freq_rows = summary[summary["case"] == "freq"]
    if len(peak_rows) == 2 and len(freq_rows) == 2:
        peak_delta_mae = abs(float(peak_rows.iloc[0]["MAE"] - peak_rows.iloc[1]["MAE"]))
        freq_delta_mae = abs(float(freq_rows.iloc[0]["MAE"] - freq_rows.iloc[1]["MAE"]))
        peak_delta_e = abs(float(peak_rows.iloc[0]["Total_kWh"] - peak_rows.iloc[1]["Total_kWh"]))
        freq_delta_e = abs(float(freq_rows.iloc[0]["Total_kWh"] - freq_rows.iloc[1]["Total_kWh"]))
        if peak_delta_mae < 1e-3 and freq_delta_mae < 1e-3 and peak_delta_e < 1e-4 and freq_delta_e < 1e-4:
            lines.append("Overall: the four results are very close; practical meaning of keeping two parameter sets may be limited.")
        else:
            lines.append("Overall: non-trivial differences exist; use matched-vs-mismatched signs above to decide whether case-adaptive parameters are supported.")
    (output_root / "mpc_paramset_case_2x2_conclusions.txt").write_text("\n".join(lines), encoding="utf-8")


def plot_paramset_cross_outputs(output_root: Path, summary_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARN plot skipped: matplotlib unavailable ({exc})", flush=True)
        return

    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path, encoding="utf-8-sig")
    if summary.empty:
        return
    figure_root = output_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)

    for case in ("peak", "freq"):
        case_df = summary[summary["case"] == case]
        if case_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5.0), dpi=140)
        for _, row in case_df.iterrows():
            series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
            if series_path is None or not series_path.exists():
                continue
            df = pd.read_csv(series_path, encoding="utf-8-sig")
            ax.plot(s(df, ["time_s", "Time (s)", "Time"]), s(df, ["T_batt_mean", "T_cell_mean_C", "Battery Temp (C)", "Average temperature"]), label=str(row["param_set"]), linewidth=1.7)
        t_ref = 25.0
        ax.axhline(t_ref, color="black", linewidth=1.0, label="T_ref=25C")
        ax.axhline(float(case_df["CV_band_high"].iloc[0]), color="tab:red", linestyle="--", linewidth=1.0, label="matched SPHI")
        ax.axhline(float(case_df["CV_band_low"].iloc[0]), color="tab:blue", linestyle="--", linewidth=1.0, label="matched SPLO")
        ax.axhline(25.5, color="tab:orange", linestyle=":", linewidth=1.2, label="25.5C")
        ax.set_title(f"{case} param-set mean temperature comparison")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Temperature (C)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(figure_root / f"{case}_paramset_temperature_compare.png")
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=140, sharex=True)
        for _, row in case_df.iterrows():
            series_path = resolve_output_csv_path(row.get("time_series_csv"), output_root)
            if series_path is None or not series_path.exists():
                continue
            df = pd.read_csv(series_path, encoding="utf-8-sig")
            time_s = s(df, ["time_s", "Time (s)", "Time"])
            label = str(row["param_set"])
            axes[0].plot(time_s, s(df, ["comp_rpm", "Compressor Speed (RPM)", "Compressor Speed"]), label=label, linewidth=1.5)
            axes[1].plot(time_s, s(df, ["pump_rpm", "Pump Speed (RPM)", "Pump Speed"]), label=label, linewidth=1.5)
        axes[0].set_ylabel("Compressor rpm")
        axes[1].set_ylabel("Pump rpm")
        axes[1].set_xlabel("Time (s)")
        for ax in axes:
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
        fig.suptitle(f"{case} param-set actuator comparison")
        fig.tight_layout()
        fig.savefig(figure_root / f"{case}_paramset_actuators_compare.png")
        plt.close(fig)

    metrics = ["T_cell_mean_max", "T_cell_max_over_time", "DeltaT_cell_max", "MAE", "RMSE", "hot_degree_seconds_25p5_cellmax", "hot_degree_seconds_band_mean", "Total_kWh", "mean_dcomp", "mean_dpump"]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.5), dpi=140)
    axes = axes.ravel()
    labels = [_paramset_cross_label(row) for _, row in summary.iterrows()]
    x = np.arange(len(summary))
    for ax, metric in zip(axes, metrics):
        ax.bar(x, summary[metric].to_numpy(float), color="#4C78A8")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "mpc_paramset_case_2x2_key_metrics.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=140)
    for _, row in summary.iterrows():
        ax.scatter(float(row["MAE"]), float(row["Total_kWh"]), s=55)
        ax.annotate(_paramset_cross_label(row), (float(row["MAE"]), float(row["Total_kWh"])), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("MAE (C)")
    ax.set_ylabel("Total_kWh")
    ax.set_title("Total_kWh vs MAE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_root / "mpc_paramset_case_2x2_total_kwh_vs_mae.png")
    plt.close(fig)

    write_paramset_cross_conclusions(output_root, summary)


def run_one(
    args: argparse.Namespace,
    case: str,
    factor: float,
    is_cv_only: bool,
    simulate_case,
    log_path: Path,
    rate_limit_mode: str | None = None,
) -> dict[str, object]:
    scene, source_csv = SOURCE[case]
    selected_rate_limit_mode = args.rate_limit_mode if rate_limit_mode is None else rate_limit_mode
    params = build_params(base_params_for_case(case), args.sweep_type, factor, selected_rate_limit_mode, is_cv_only, case=case)
    cv_band_half_width = float(params.cv_band_half_width)
    if args.sweep_type == "final_adaptive_mpc":
        suffix = "final_adaptive_mpc"
    elif args.sweep_type == "final_cv_band_scan":
        suffix = f"band_{format_band_label(cv_band_half_width)}_final_cv_band_scan"
    elif args.sweep_type == "final_pump_weight_scan":
        suffix = f"pump_x{format_factor_label(factor)}_final_pump_weight_scan"
    elif args.sweep_type == "final_pump_dmax_scan":
        suffix = f"pump_dmax_{format_pump_dmax_file_label(factor)}_final_pump_dmax_scan"
    elif args.sweep_type == "freq_compressor_displacement_scale_scan":
        suffix = f"disp_scale_{format_factor_label(factor)}"
    elif args.sweep_type == "freq_compressor_min_rpm_scan":
        suffix = f"min_comp_{format_factor_label(factor)}rpm"
    elif args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        min_comp_rpm, disp_scale = parse_hardware_ablation_factor(factor)
        suffix = format_hardware_ablation_label(min_comp_rpm, disp_scale)
    elif args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        suffix = normalize_param_set(factor)
    elif args.sweep_type in (THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        suffix = format_three_way_label(factor)
    elif args.sweep_type in TERMINAL_COST_SWEEP_TYPES:
        suffix = format_terminal_cost_label(factor)
    else:
        suffix = selected_rate_limit_mode if args.sweep_type in ("actuator_rate_ablation", "comp_rate_tuning", "comp_rate_timeseries") else ("cv_only" if is_cv_only else f"x{factor:g}")
    tag = f"mpc_{case}_{args.sweep_type}_{suffix}_unified_init_{INIT_TEMP_C:g}C"
    detail_root = args.output_root if args.save_detail else args.output_root / "_scratch_detail"
    cv_band_patch = cv_band_half_width if args.sweep_type in ("final_adaptive_mpc", "final_cv_band_scan", "final_pump_weight_scan", "final_pump_dmax_scan", "freq_compressor_displacement_scale_scan", "freq_compressor_min_rpm_scan", HARDWARE_ABLATION_SWEEP_TYPE, PARAMSET_CROSS_SWEEP_TYPE, TERMINAL_COST_SWEEP_TYPE, TERMINAL_COST_REFINED_SWEEP_TYPE, TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE, BASELINE_PARITY_SWEEP_TYPE, THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE) else None
    if args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        min_comp_patch, disp_scale_patch = parse_hardware_ablation_factor(factor)
    else:
        disp_scale_patch = float(factor) if args.sweep_type == "freq_compressor_displacement_scale_scan" else None
        min_comp_patch = float(factor) if args.sweep_type == "freq_compressor_min_rpm_scan" else None
    with patched_params(case, params, cv_band_patch), patched_compressor_displacement(disp_scale_patch), patched_compressor_min_rpm(min_comp_patch):
        result = simulate_case(
            "mpc",
            scene,
            FLOW,
            source_csv,
            f"{tag}.csv",
            f"{tag}_snap.csv",
            output_root=detail_root,
            mpc_flow_mode=args.mpc_flow_mode,
            force=True,
            max_steps=args.max_steps,
            log_func=log_to(log_path),
        )
        out_csv = Path(result["out_csv"])
        row = summarize_csv(out_csv, case, args.sweep_type, factor, is_cv_only, selected_rate_limit_mode, params, cv_band_half_width, args.max_steps)
    row["solve_status"] = result.get("solve_status", result.get("status", "ok"))
    if args.sweep_type == "final_adaptive_mpc":
        series_path = args.output_root / "time_series" / f"{case}_final_adaptive_mpc.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == "final_cv_band_scan":
        series_path = args.output_root / "time_series" / f"{case}_band_{format_band_label(cv_band_half_width)}_final_cv_band_scan.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == "final_pump_weight_scan":
        series_path = args.output_root / "time_series" / f"{case}_pump_x{format_factor_label(factor)}_final_pump_weight_scan.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == "final_pump_dmax_scan":
        series_path = args.output_root / "time_series" / f"{case}_pump_dmax_{format_pump_dmax_file_label(factor)}_final_pump_dmax_scan.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == "freq_compressor_displacement_scale_scan":
        series_path = args.output_root / "time_series" / f"freq_disp_scale_{format_factor_label(factor)}.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == "freq_compressor_min_rpm_scan":
        series_path = args.output_root / "time_series" / f"freq_min_comp_{format_factor_label(factor)}rpm.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        min_comp_rpm, disp_scale = parse_hardware_ablation_factor(factor)
        series_path = args.output_root / "time_series" / f"{case}_{format_hardware_ablation_label(min_comp_rpm, disp_scale)}.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        series_path = args.output_root / "time_series" / f"{param_set_case_label(case, factor)}.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type in TERMINAL_REPORT_SWEEP_TYPES:
        series_path = args.output_root / "time_series" / f"{terminal_cost_case_label(case, factor)}.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if args.sweep_type == "comp_rate_timeseries":
        series_path = args.output_root / "time_series_comp_rate_tuning" / f"{case}_{selected_rate_limit_mode}.csv"
        export_time_series(out_csv, series_path, case, selected_rate_limit_mode, params)
        row["time_series_csv"] = str(series_path)
    if not args.save_detail:
        cleanup_detail(result, args.output_root)
    return row



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPC sensitivity runner and final adaptive MPC runner.")
    parser.add_argument("--sweep-type", "--sweep_type", choices=SUPPORTED_SWEEP_TYPES, default=PARAMSET_CROSS_SWEEP_TYPE)
    parser.add_argument("--factors", default=None, help="Comma-separated sweep factors. Defaults depend on sweep type.")
    parser.add_argument("--cases", default="peak,freq", help="Comma-separated cases: peak,freq")
    parser.add_argument("--only-case", "--only_case", default=None, help="Filter generated task labels, e.g. baseline_off or empirical_w1e6.")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit simulation rows.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rate-limit-mode", "--rate_limit_mode", choices=RATE_LIMIT_MODES, default="comp_rate_off")
    parser.add_argument("--mpc-flow-mode", "--mpc_flow_mode", choices=("switching", "supervised", "standard", "mixed_integer"), default="switching")
    parser.add_argument("--save-detail", action="store_true", help="Keep full time-series CSV and snapshots.")
    parser.add_argument("--no-cv-only", action="store_true", help="Skip the CV-only endpoint for w_energy_comp.")
    return parser.parse_args()
def main() -> None:
    args = parse_args()
    if args.sweep_type == "final_adaptive_mpc" and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = FINAL_ADAPTIVE_OUTPUT_ROOT
    if args.sweep_type == "final_cv_band_scan" and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = FINAL_CV_BAND_OUTPUT_ROOT
    if args.sweep_type == "final_pump_weight_scan" and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = FINAL_PUMP_WEIGHT_OUTPUT_ROOT
    if args.sweep_type == "final_pump_dmax_scan" and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = FINAL_PUMP_DMAX_OUTPUT_ROOT
    if args.sweep_type == "freq_compressor_displacement_scale_scan" and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = FREQ_COMP_DISP_SCALE_OUTPUT_ROOT
    if args.sweep_type == "freq_compressor_min_rpm_scan" and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = FREQ_COMP_MIN_RPM_OUTPUT_ROOT
    if args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = PARAMSET_CROSS_OUTPUT_ROOT
    if args.sweep_type == TERMINAL_COST_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = TERMINAL_COST_OUTPUT_ROOT
    if args.sweep_type == TERMINAL_COST_REFINED_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = TERMINAL_COST_REFINED_OUTPUT_ROOT
    if args.sweep_type == TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = TERMINAL_COST_SELECTED_FULL_OUTPUT_ROOT
    if args.sweep_type == BASELINE_PARITY_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = BASELINE_PARITY_OUTPUT_ROOT
    if args.sweep_type == THREE_WAY_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = THREE_WAY_OUTPUT_ROOT
    if args.sweep_type == THREE_WAY_FULL_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = THREE_WAY_FULL_OUTPUT_ROOT
    if args.max_steps is None and args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        args.max_steps = HARDWARE_ABLATION_STEPS
    if args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE and args.output_root == DEFAULT_OUTPUT_ROOT:
        args.output_root = hardware_ablation_output_root(args.max_steps)
    if args.max_steps is None and args.sweep_type == "final_cv_band_scan":
        args.max_steps = FINAL_CV_BAND_STEPS
    if args.max_steps is None and args.sweep_type == "final_pump_weight_scan":
        args.max_steps = FINAL_PUMP_WEIGHT_STEPS
    if args.max_steps is None and args.sweep_type == "final_pump_dmax_scan":
        args.max_steps = FINAL_PUMP_DMAX_STEPS
    if args.max_steps is None and args.sweep_type not in ("final_adaptive_mpc", "freq_compressor_displacement_scale_scan", "freq_compressor_min_rpm_scan", PARAMSET_CROSS_SWEEP_TYPE, TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        args.max_steps = MAX_STEPS
    if args.sweep_type in ("actuator_rate_ablation", "comp_rate_tuning", "comp_rate_timeseries", "final_adaptive_mpc", HARDWARE_ABLATION_SWEEP_TYPE, PARAMSET_CROSS_SWEEP_TYPE):
        factors = ()
    elif args.sweep_type == "final_pump_dmax_scan":
        factors = parse_pump_dmax_list(args.factors) if args.factors else DEFAULT_FACTORS[args.sweep_type]
    elif args.sweep_type == BASELINE_PARITY_SWEEP_TYPE:
        factors = parse_csv_list(args.factors) if args.factors else BASELINE_PARITY_VARIANTS
    elif args.sweep_type in (THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        factors = parse_csv_list(args.factors) if args.factors else None
    elif args.sweep_type in TERMINAL_COST_SWEEP_TYPES:
        factors = parse_terminal_cost_list(args.factors) if args.factors else None
    else:
        factors = parse_float_list(args.factors) if args.factors else DEFAULT_FACTORS[args.sweep_type]
    cases = parse_csv_list(args.cases)
    args.output_root.mkdir(parents=True, exist_ok=True)

    log_path = args.output_root / "run.log"
    if log_path.exists():
        log_path.unlink()
    summary_names = {
        "actuator_rate_ablation": "summary_actuator_rate_ablation.csv",
        "comp_rate_tuning": "summary_comp_rate_tuning.csv",
        "comp_rate_timeseries": "summary_comp_rate_timeseries.csv",
        "adaptive_dmax_w_energy_comp": "summary_adaptive_dmax_w_energy_comp.csv",
        "final_cv_weight_scan": "summary_final_cv_weight_scan.csv",
        "stable_cv_weight_scan": "summary_stable_cv_weight_scan.csv",
        "final_adaptive_mpc": "summary_final_adaptive_mpc.csv",
        "final_cv_band_scan": "summary_final_cv_band_scan.csv",
        "final_pump_weight_scan": "summary_final_pump_weight_scan.csv",
        "final_pump_dmax_scan": "summary_final_pump_dmax_scan.csv",
        "freq_compressor_displacement_scale_scan": "summary_freq_compressor_displacement_scale_scan.csv",
        "freq_compressor_min_rpm_scan": "summary_freq_compressor_min_rpm_scan.csv",
        HARDWARE_ABLATION_SWEEP_TYPE: "summary_peak_freq_minrpm_dispscale_ablation.csv",
        PARAMSET_CROSS_SWEEP_TYPE: "summary_mpc_paramset_case_2x2_cross_validation.csv",
        TERMINAL_COST_SWEEP_TYPE: "summary_terminal_cost_scan.csv",
        TERMINAL_COST_REFINED_SWEEP_TYPE: "summary_terminal_cost_refined.csv",
        TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE: "summary_terminal_cost_selected_full.csv",
        BASELINE_PARITY_SWEEP_TYPE: "summary_baseline_parity_check.csv",
        THREE_WAY_SWEEP_TYPE: "summary_terminal_cost_three_way_compare.csv",
        THREE_WAY_FULL_SWEEP_TYPE: "summary_terminal_cost_three_way_full.csv",
    }
    summary_path = args.output_root / summary_names.get(args.sweep_type, "summary_metrics.csv")

    if args.sweep_type in ("final_adaptive_mpc", "final_cv_band_scan", "final_pump_weight_scan", "final_pump_dmax_scan", "freq_compressor_displacement_scale_scan", "freq_compressor_min_rpm_scan", HARDWARE_ABLATION_SWEEP_TYPE, PARAMSET_CROSS_SWEEP_TYPE, TERMINAL_COST_SWEEP_TYPE, TERMINAL_COST_REFINED_SWEEP_TYPE, TERMINAL_COST_SELECTED_FULL_SWEEP_TYPE, BASELINE_PARITY_SWEEP_TYPE, THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        print(f"START run_type={args.sweep_type} summary={summary_path}", flush=True)
    else:
        print(f"START sweep_type={args.sweep_type} summary={summary_path}", flush=True)
    simulate_case = None
    rows = []
    if args.sweep_type == "actuator_rate_ablation":
        tasks = [(case, 1.0, False, mode) for mode in ACTUATOR_RATE_MODES for case in cases]
    elif args.sweep_type == "comp_rate_tuning":
        tasks = [(case, 1.0, False, mode) for mode in COMP_RATE_TUNING_MODES for case in cases]
    elif args.sweep_type == "comp_rate_timeseries":
        tasks = [(case, 1.0, False, mode) for mode in COMP_RATE_TIMESERIES_MODES for case in cases]
    elif args.sweep_type == "adaptive_dmax_w_energy_comp":
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == "final_cv_weight_scan":
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == "stable_cv_weight_scan":
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == "final_cv_band_scan":
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == "final_pump_weight_scan":
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == "final_pump_dmax_scan":
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == "freq_compressor_displacement_scale_scan":
        tasks = [("freq", factor, False, "baseline") for factor in factors]
    elif args.sweep_type == "freq_compressor_min_rpm_scan":
        tasks = [("freq", factor, False, "baseline") for factor in factors]
    elif args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        tasks = [
            (case, hardware_ablation_factor(min_comp_rpm, disp_scale), False, "baseline")
            for case in cases
            for disp_scale in HARDWARE_ABLATION_DISP_SCALE_VALUES
            for min_comp_rpm in HARDWARE_ABLATION_MIN_COMP_RPM_VALUES
        ]
    elif args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        tasks = [(case, param_set, False, "baseline") for case in cases for param_set in PARAMSET_CROSS_PARAM_SETS]
    elif args.sweep_type == BASELINE_PARITY_SWEEP_TYPE:
        tasks = [(case, factor, False, "baseline") for case in cases for factor in factors]
    elif args.sweep_type == THREE_WAY_SWEEP_TYPE:
        tasks = [
            (case, factor, False, "baseline")
            for case in cases
            for factor in (factors if factors is not None else default_three_way_factors(case))
        ]
    elif args.sweep_type == THREE_WAY_FULL_SWEEP_TYPE:
        tasks = [
            (case, factor, False, "baseline")
            for case in cases
            for factor in (factors if factors is not None else default_three_way_full_factors(case))
        ]
    elif args.sweep_type == "final_adaptive_mpc":
        tasks = [(case, 1.0, False, "baseline") for case in cases]
    elif args.sweep_type in TERMINAL_COST_SWEEP_TYPES:
        tasks = [
            (case, factor, False, "baseline")
            for case in cases
            for factor in (factors if factors is not None else terminal_cost_factors_for_case(args.sweep_type, case))
        ]
    else:
        tasks = [(case, factor, False, args.rate_limit_mode) for case in cases for factor in factors]
        if args.sweep_type == "w_energy_comp" and not args.no_cv_only:
            tasks += [(case, 1.0, True, args.rate_limit_mode) for case in cases]
    if args.only_case:
        tasks = [
            task for task in tasks
            if task_label_for_filter(args.sweep_type, task[1], task[2], task[3]) == args.only_case
            or task[0] == args.only_case
        ]
    total = len(tasks)
    simulate_case = make_simulate_case_with_unified_initial_state()
    for index, (case, factor, is_cv_only, rate_limit_mode) in enumerate(tasks, start=1):
        label = task_label_for_filter(args.sweep_type, factor, is_cv_only, rate_limit_mode)
        if args.sweep_type == "final_pump_weight_scan":
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            pump_label = f"x{factor:g}"
            print(f"RUN [{index}/{total}] case={case} pump_weight={pump_label} controller=final_pump_weight_scan", flush=True)
            print(
                f"PARAM case={case} steps={args.max_steps} "
                f"CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g}",
                flush=True,
            )
        elif args.sweep_type == "final_pump_dmax_scan":
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            pump_label = format_pump_dmax_label(factor)
            print(f"RUN [{index}/{total}] case={case} pump_dmax={pump_label} controller=final_pump_dmax_scan", flush=True)
            print(
                f"PARAM case={case} steps={args.max_steps} "
                f"CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step pump_DMAX={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g}",
                flush=True,
            )
        elif args.sweep_type == "freq_compressor_displacement_scale_scan":
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case=freq disp_scale={factor:g} controller=freq_compressor_displacement_scale_scan", flush=True)
            print(
                f"PARAM case=freq CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} disp_scale={factor:g}",
                flush=True,
            )
        elif args.sweep_type == "freq_compressor_min_rpm_scan":
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case=freq min_comp_rpm={factor:g} controller=freq_compressor_min_rpm_scan", flush=True)
            print(
                f"PARAM case=freq CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} min_comp_rpm={params.n_comp_min:g}",
                flush=True,
            )
        elif args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
            min_comp_rpm, disp_scale = parse_hardware_ablation_factor(factor)
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case={case} min_comp_rpm={min_comp_rpm:g} disp_scale={disp_scale:g} controller={HARDWARE_ABLATION_SWEEP_TYPE}", flush=True)
            print(
                f"PARAM case={case} steps={args.max_steps} CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} "
                f"min_comp_rpm={params.n_comp_min:g} disp_scale={disp_scale:g}",
                flush=True,
            )
        elif args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
            param_set = normalize_param_set(factor)
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case={case} param_set={param_set} controller={PARAMSET_CROSS_SWEEP_TYPE}", flush=True)
            print(
                f"PARAM case={case} param_set={param_set} CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} "
                f"terminal={params.w_term_peak:g} tracking={params.w_temp_obj:g}",
                flush=True,
            )
        elif args.sweep_type in (THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
            terminal_label = format_three_way_label(factor)
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case={case} {terminal_label} controller={args.sweep_type}", flush=True)
            print(
                f"PARAM case={case} steps={args.max_steps} CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} "
                f"terminal_cost_enabled={params.terminal_cost_enabled} terminal_cost_type={params.terminal_cost_type} "
                f"w_terminal_temp={params.w_terminal_temp:g} terminal_temp_scale_c={params.terminal_temp_scale_c:g}",
                flush=True,
            )
        elif args.sweep_type == BASELINE_PARITY_SWEEP_TYPE:
            terminal_label = format_terminal_cost_label(factor)
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case={case} {terminal_label} controller={args.sweep_type}", flush=True)
            print(
                f"PARAM case={case} steps={args.max_steps} CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} "
                f"terminal_cost_enabled={params.terminal_cost_enabled} terminal_cost_type={params.terminal_cost_type} "
                f"w_terminal_temp={params.w_terminal_temp:g} terminal_temp_scale_c={params.terminal_temp_scale_c:g}",
                flush=True,
            )
        elif args.sweep_type in TERMINAL_COST_SWEEP_TYPES:
            terminal_enabled, w_terminal = normalize_terminal_cost_factor(factor)
            terminal_label = format_terminal_cost_label(factor)
            params = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print(f"RUN [{index}/{total}] case={case} {terminal_label} controller={args.sweep_type}", flush=True)
            print(
                f"PARAM case={case} steps={args.max_steps} CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g} "
                f"terminal_cost_enabled={terminal_enabled} terminal_cost_type={params.terminal_cost_type} "
                f"w_terminal_temp={w_terminal:g} terminal_temp_scale_c={params.terminal_temp_scale_c:g}",
                flush=True,
            )
        elif args.sweep_type == "final_adaptive_mpc":
            params = get_final_adaptive_mpc_params(case)
            print(f"RUN [{index}/{total}] case={case} controller=final_adaptive_mpc", flush=True)
            print(
                f"PARAM case={case} CV={25.0 - params.cv_band_half_width:g}-{25.0 + params.cv_band_half_width:g} "
                f"WSPHI={params.w_high_temp:g} WSPLO={params.w_cold_temp:g} "
                f"DMAX_comp={params.dmax_comp:g}rpm/step DMAX_pump={params.dmax_pump:g}rpm/step "
                f"w_energy_comp={params.w_energy_comp:g} w_energy_pump={params.w_energy_pump:g}",
                flush=True,
            )
        else:
            print(f"RUN [{index}/{total}] case={case} {args.sweep_type}={label} rate_limit_mode={rate_limit_mode}", flush=True)
        if args.sweep_type in TERMINAL_REPORT_SWEEP_TYPES:
            params_for_extra = build_params(base_params_for_case(case), args.sweep_type, factor, rate_limit_mode, is_cv_only, case=case)
            print_param_extra(case, params_for_extra, args.mpc_flow_mode, planned_simulation_steps(case, args.max_steps))
            print_terminal_param_diff(case, params_for_extra)
        row = run_one(args, case, factor, is_cv_only, simulate_case, log_path, rate_limit_mode)
        rows.append(row)
        done_prefix = f"DONE [{index}/{total}] case={case} {args.sweep_type}={label} "
        if args.sweep_type == "adaptive_dmax_w_energy_comp":
            done_prefix = f"DONE [{index}/{total}] case={case} w_energy_comp={label} "
        elif args.sweep_type in ("final_cv_weight_scan", "stable_cv_weight_scan"):
            label = format_cv_weight_label(factor)
            done_prefix = f"DONE [{index}/{total}] case={case} cv_weight={label} "
        elif args.sweep_type == "final_cv_band_scan":
            done_prefix = f"DONE [{index}/{total}] case={case} band={float(factor):g} "
        elif args.sweep_type == "final_pump_weight_scan":
            done_prefix = f"DONE [{index}/{total}] case={case} pump_weight=x{factor:g} "
        elif args.sweep_type == "final_pump_dmax_scan":
            done_prefix = f"DONE [{index}/{total}] case={case} pump_dmax={format_pump_dmax_label(factor)} "
        elif args.sweep_type == "freq_compressor_displacement_scale_scan":
            done_prefix = f"DONE [{index}/{total}] case=freq disp_scale={factor:g} "
        elif args.sweep_type == "freq_compressor_min_rpm_scan":
            done_prefix = f"DONE [{index}/{total}] case=freq min_comp_rpm={factor:g} "
        elif args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
            min_comp_rpm, disp_scale = parse_hardware_ablation_factor(factor)
            done_prefix = f"DONE [{index}/{total}] case={case} min_comp_rpm={min_comp_rpm:g} disp_scale={disp_scale:g} "
        elif args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
            done_prefix = f"DONE [{index}/{total}] case={case} param_set={normalize_param_set(factor)} "
        elif args.sweep_type in TERMINAL_REPORT_SWEEP_TYPES:
            done_prefix = f"DONE [{index}/{total}] case={case} {terminal_cost_case_label(case, factor).removeprefix(case + '_')} "
        elif args.sweep_type == "final_cv_band_scan":
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} MAE={row['temp_mae_to_ref']:.4f} "
                + f"RMSE={row['temp_rmse_to_ref']:.4f} "
                + f"hot_degree_seconds={row['hot_degree_seconds']:.2f} "
                + f"cold_degree_seconds={row['cold_degree_seconds']:.2f} "
                + f"E={row['Total_kWh']:.4f} mean_comp={row['mean_comp_rpm']:.0f} "
                + f"mean_pump={row['mean_pump_rpm']:.0f}",
                flush=True,
            )
        elif args.sweep_type == "final_pump_weight_scan":
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} MAE={row['temp_mae_to_ref']:.4f} "
                + f"RMSE={row['temp_rmse_to_ref']:.4f} "
                + f"hot_degree_seconds={row['hot_degree_seconds']:.2f} "
                + f"E={row['Total_kWh']:.4f} mean_comp={row['mean_comp_rpm']:.0f} "
                + f"mean_pump={row['mean_pump_rpm']:.0f}",
                flush=True,
            )
        elif args.sweep_type == "final_pump_dmax_scan":
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} MAE={row['temp_mae_to_ref']:.4f} "
                + f"RMSE={row['temp_rmse_to_ref']:.4f} "
                + f"hot_degree_seconds={row['hot_degree_seconds']:.2f} "
                + f"E={row['Total_kWh']:.4f} mean_comp={row['mean_comp_rpm']:.0f} "
                + f"mean_pump={row['mean_pump_rpm']:.0f}",
                flush=True,
            )
        elif args.sweep_type == "freq_compressor_displacement_scale_scan":
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} Tmean_min={row['T_cell_mean_min']:.4f} MAE={row['MAE']:.4f} "
                + f"RMSE={row['RMSE']:.4f} hot_band={row['hot_degree_seconds_band']:.2f} "
                + f"cold={row['cold_degree_seconds']:.2f} E={row['Total_kWh']:.4f} "
                + f"mean_comp={row['mean_comp_rpm']:.0f} comp_low={row['comp_lower_ratio']:.3f}",
                flush=True,
            )
        elif args.sweep_type in TERMINAL_REPORT_SWEEP_TYPES:
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} Tmean_min={row['T_cell_mean_min']:.4f} MAE={row['MAE']:.4f} "
                + f"RMSE={row['RMSE']:.4f} hot_band={row['hot_degree_seconds_band']:.2f} "
                + f"cold={row['cold_degree_seconds']:.2f} E={row['Total_kWh']:.4f} "
                + f"T_pred_end={row['T_pred_end_C']:.4f} J_terminal={row['J_terminal']:.4f} "
                + f"Weighted_J_terminal={row['Weighted_J_terminal']:.4f}",
                flush=True,
            )
        elif args.sweep_type in ("freq_compressor_min_rpm_scan", HARDWARE_ABLATION_SWEEP_TYPE, PARAMSET_CROSS_SWEEP_TYPE):
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} Tmean_min={row['T_cell_mean_min']:.4f} MAE={row['MAE']:.4f} "
                + f"RMSE={row['RMSE']:.4f} hot_band={row['hot_degree_seconds_band']:.2f} "
                + f"hot_25p5={row['hot_degree_seconds_25p5']:.2f} cold={row['cold_degree_seconds']:.2f} "
                + f"E={row['Total_kWh']:.4f} mean_comp={row['mean_comp_rpm']:.0f} "
                + f"comp_low={row['comp_lower_ratio']:.3f} comp_high={row['comp_upper_ratio']:.3f}",
                flush=True,
            )
        elif args.sweep_type in ("final_cv_weight_scan", "stable_cv_weight_scan"):
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} Tmean_min={row['T_cell_mean_min']:.4f} "
                + f"hot_degree_seconds={row['hot_degree_seconds']:.2f} "
                + f"cold_degree_seconds={row['cold_degree_seconds']:.2f} "
                + f"E={row['Total_kWh']:.4f} mean_comp={row['mean_comp_rpm']:.0f}",
                flush=True,
            )
        else:
            print(
                done_prefix
                + f"Tmean_max={row['T_cell_mean_max']:.4f} hot_degree_seconds={row['hot_degree_seconds']:.2f} "
                + f"E={row['Total_kWh']:.4f} mean_comp={row['mean_comp_rpm']:.0f}",
                flush=True,
            )
    summary_df = pd.DataFrame(rows)
    if args.sweep_type in TERMINAL_REPORT_SWEEP_TYPES:
        summary_df = add_terminal_baseline_metrics(summary_df)
    if args.sweep_type in (THREE_WAY_SWEEP_TYPE, THREE_WAY_FULL_SWEEP_TYPE):
        summary_df = add_three_way_baseline_metrics(summary_df)
    if args.sweep_type == THREE_WAY_FULL_SWEEP_TYPE and THREE_WAY_FULL_SKIP_NOTES:
        summary_df["freq_skipped"] = True
        summary_df["freq_skip_reason"] = "; ".join(THREE_WAY_FULL_SKIP_NOTES)
    if args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        required_columns = [
            "case", "min_comp_rpm", "disp_scale", "n_steps", "dt_s", "sim_duration_s",
            "T_cell_mean_max", "T_cell_mean_min", "T_cell_mean_final", "Tmean_max", "Tmean_min", "Tmean_final",
            "MAE", "RMSE", "hot_degree_seconds_band_mean", "hot_degree_seconds_25p5_cellmax",
            "T_cell_max_over_time", "T_cell_min_over_time", "DeltaT_cell_max", "DeltaT_cell_mean",
            "cold_degree_seconds", "Total_kWh", "Comp_kWh", "Pump_kWh", "Fan_kWh",
            "mean_comp_rpm", "max_comp_rpm", "mean_pump_rpm", "max_pump_rpm",
            "comp_lower_ratio", "comp_upper_ratio", "mean_dcomp", "mean_dpump", "time_series_csv",
        ]
        legacy_mean_columns = {"Tmax", "Tmin", "Tfinal"}
        ordered = [col for col in required_columns if col in summary_df.columns]
        remainder = [col for col in summary_df.columns if col not in ordered and col not in legacy_mean_columns]
        summary_df = summary_df[ordered + remainder]
    if args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        required_columns = [
            "case", "param_set", "n_steps", "dt_s", "sim_duration_s", "CV_band_low",
            "CV_band_high", "WSPHI_used", "WSPLO_used", "DMAX_comp", "DMAX_pump",
            "w_energy_comp", "w_energy_pump", "terminal_cost_on", "J_hot_on",
            "tracking_term_on", "disp_scale", "min_comp_rpm",
            "T_cell_mean_max", "T_cell_mean_min", "T_cell_mean_final", "Tmean_max", "Tmean_min", "Tmean_final",
            "MAE", "RMSE", "hot_degree_seconds_band_mean", "hot_degree_seconds_25p5_cellmax",
            "T_cell_max_over_time", "T_cell_min_over_time", "DeltaT_cell_max", "DeltaT_cell_mean",
            "cold_degree_seconds", "Total_kWh", "Comp_kWh", "Pump_kWh", "Fan_kWh",
            "mean_comp_rpm", "max_comp_rpm", "mean_pump_rpm", "max_pump_rpm",
            "mean_dcomp", "mean_dpump", "comp_lower_ratio", "comp_upper_ratio",
            "pump_lower_ratio", "pump_upper_ratio", "solve_status", "time_series_csv",
        ]
        legacy_mean_columns = {"Tmax", "Tmin", "Tfinal"}
        ordered = [col for col in required_columns if col in summary_df.columns]
        remainder = [col for col in summary_df.columns if col not in ordered and col not in legacy_mean_columns]
        summary_df = summary_df[ordered + remainder]
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if args.sweep_type == HARDWARE_ABLATION_SWEEP_TYPE:
        plot_hardware_ablation_outputs(args.output_root, summary_path)
    if args.sweep_type == PARAMSET_CROSS_SWEEP_TYPE:
        plot_paramset_cross_outputs(args.output_root, summary_path)
    if args.sweep_type in TERMINAL_REPORT_SWEEP_TYPES:
        plot_terminal_cost_outputs(args.output_root, summary_path)
    if args.sweep_type == THREE_WAY_SWEEP_TYPE:
        plot_three_way_compare_outputs(args.output_root, summary_path)
    if args.sweep_type == THREE_WAY_FULL_SWEEP_TYPE:
        plot_three_way_full_outputs(args.output_root, summary_path)
    print(f"END summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
































































