import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import uuid

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

import numpy as np
import pack as pack_module
import pandas as pd
import thermal_batch_config as thermal_batch_config_module
import thermal_loop as thermal_loop_module
import thermal_system as thermal_system_module

from pack import BatteryPack
from predictor_identification_data import (
    REQUIRED_COLUMNS,
    VALID_SPLITS,
    assign_scenario_splits,
    validate_identification_frame,
)
from thermal_batch_config import (
    EVAP_CAP_FACTOR,
    EVAP_FLOW_EXP,
    EVAP_UA_FACTOR,
    N_COMP_MAX_RPM,
    N_COMP_MIN_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
)
from thermal_loop import (
    build_pack_config,
    initialize_refrigeration_dynamic_state,
    simulate_thermal_loop_step,
    staged_fan_speed,
)
from thermal_system import pump_model, run_refrigeration_cycle


COMPRESSOR_LEVELS_RPM = (
    1000,
    1400,
    1800,
    1900,
    1950,
    1999,
    2000,
    2200,
    3000,
    4000,
    5000,
    6000,
)
PUMP_LEVELS_RPM = (1600, 2400, 3200, 4000, 4800)
COOLANT_LEVELS_C = (15, 17.5, 20, 25, 30, 35)
AMBIENT_LEVELS_C = (20, 25, 30, 35, 40)
DYNAMIC_COMPRESSOR_LEVELS_RPM = (1000.0, 2000.0, 3500.0, 4500.0, 6000.0)
DYNAMIC_PUMP_LEVELS_RPM = (1600.0, 2400.0, 3200.0, 4000.0, 4800.0)
DYNAMIC_CURRENT_LEVELS_A = (240.0, 400.0, 560.0)

_CONFIGURATION = {
    "n_comp_min_rpm": float(N_COMP_MIN_RPM),
    "n_comp_max_rpm": float(N_COMP_MAX_RPM),
    "n_pump_min_rpm": float(N_PUMP_MIN_RPM),
    "n_pump_max_rpm": float(N_PUMP_MAX_RPM),
    "evap_ua_factor": float(EVAP_UA_FACTOR),
    "evap_flow_exp": float(EVAP_FLOW_EXP),
    "evap_cap_factor": float(EVAP_CAP_FACTOR),
}


def _build_plant_source_hash(modules):
    digest = hashlib.sha256()
    for module in sorted(modules, key=lambda value: value.__name__):
        source_path = getattr(module, "__file__", None)
        if not source_path:
            raise RuntimeError(f"Cannot locate source file for {module.__name__}")
        try:
            source_bytes = Path(source_path).read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"Cannot read source file for {module.__name__}: {exc}"
            ) from exc
        digest.update(module.__name__.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_bytes)
        digest.update(b"\0")
    return digest.hexdigest()


def build_configuration_hash(configuration, plant_source_hash):
    snapshot = {
        "configuration": configuration,
        "plant_source_hash": str(plant_source_hash),
    }
    snapshot_json = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest()


def hash_file_content(path, label="file"):
    try:
        content = Path(path).read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Cannot read {label} content: {exc}") from exc
    return hashlib.sha256(content).hexdigest()


_HPPC_REQUIRED_KEYS = (
    "soc",
    "temp",
    "ocv",
    "r0_dis",
    "r0_chg",
    "r1_dis",
    "r1_chg",
    "c1_dis",
    "c1_chg",
    "r2_dis",
    "r2_chg",
    "c2_dis",
    "c2_chg",
)


def _validate_json_number_leaves(value, path, key, location=None):
    location = key if location is None else location
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_number_leaves(
                item,
                path,
                key,
                f"{location}[{index}]",
            )
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key={key} "
            f"shape=<invalid-json-number>; {location} must be a finite "
            f"JSON number, got {type(value).__name__}"
        )
    try:
        finite = math.isfinite(value)
    except (TypeError, OverflowError):
        finite = False
    if not finite:
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key={key} "
            f"shape=<invalid-json-number>; {location} must be a finite "
            "JSON number"
        )


def validate_hppc_parameter_file(path=pack_module.HPPC_PARAMS_PATH):
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key=<document> "
            f"shape=<unreadable>: {exc}"
        ) from exc

    def reject_constant(value):
        raise ValueError(f"non-standard JSON constant {value}")

    try:
        data = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key=<document> "
            f"shape=<invalid-json>: JSON decode failed: {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key=<document> "
            f"shape={type(data).__name__}; expected JSON object"
        )

    for key in _HPPC_REQUIRED_KEYS:
        if key not in data:
            raise ValueError(
                f"Invalid HPPC parameter file {path}: key={key} "
                "shape=<missing>"
            )
        _validate_json_number_leaves(data[key], path, key)

    axes = {}
    for key in ("soc", "temp"):
        try:
            values = np.asarray(data[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid HPPC parameter file {path}: key={key} "
                "shape=<non-numeric>"
            ) from exc
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError(
                f"Invalid HPPC parameter file {path}: key={key} "
                f"shape={values.shape}; expected non-empty finite 1D values"
            )
        axes[key] = values

    n_soc = axes["soc"].size
    n_temp = axes["temp"].size
    try:
        ocv = np.asarray(data["ocv"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key=ocv "
            "shape=<non-numeric>"
        ) from exc
    valid_ocv_shapes = (
        (n_soc,),
        (n_soc, n_temp),
        (n_soc * n_temp,),
    )
    if ocv.shape not in valid_ocv_shapes or not np.isfinite(ocv).all():
        raise ValueError(
            f"Invalid HPPC parameter file {path}: key=ocv shape={ocv.shape}; "
            f"expected one of {valid_ocv_shapes} with finite values"
        )

    expected_matrix_shape = (n_soc, n_temp)
    expected_flat_shape = (n_soc * n_temp,)
    for key in _HPPC_REQUIRED_KEYS[3:]:
        try:
            values = np.asarray(data[key], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid HPPC parameter file {path}: key={key} "
                "shape=<non-numeric>"
            ) from exc
        if values.shape not in (expected_matrix_shape, expected_flat_shape):
            raise ValueError(
                f"Invalid HPPC parameter file {path}: key={key} "
                f"shape={values.shape}; expected {expected_matrix_shape} "
                f"or {expected_flat_shape}"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"Invalid HPPC parameter file {path}: key={key} "
                f"shape={values.shape}; values must be finite"
            )
    return data


def build_dynamic_source_hash(
    plant_source_hash, battery_pack_source_hash, hppc_parameters_hash
):
    snapshot = {
        "plant_source_hash": str(plant_source_hash),
        "battery_pack_source_hash": str(battery_pack_source_hash),
        "hppc_parameters_hash": str(hppc_parameters_hash),
    }
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_dynamic_configuration_hash(
    configuration,
    plant_source_hash,
    battery_pack_source_hash,
    hppc_parameters_hash,
):
    snapshot = {
        "configuration": configuration,
        "plant_source_hash": str(plant_source_hash),
        "battery_pack_source_hash": str(battery_pack_source_hash),
        "hppc_parameters_hash": str(hppc_parameters_hash),
    }
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_PLANT_SOURCE_HASH = _build_plant_source_hash(
    (thermal_system_module, thermal_loop_module, thermal_batch_config_module)
)
_PLANT_MODEL_VERSION = f"sha256:{_PLANT_SOURCE_HASH[:12]}"
_CONFIGURATION_HASH = build_configuration_hash(
    _CONFIGURATION, _PLANT_SOURCE_HASH
)
_BATTERY_PACK_SOURCE_HASH = hash_file_content(
    pack_module.__file__, "BatteryPack source"
)
_HPPC_PARAMETERS_HASH = hash_file_content(
    pack_module.HPPC_PARAMS_PATH, "HPPC parameters"
)
_DYNAMIC_SOURCE_HASH = build_dynamic_source_hash(
    _PLANT_SOURCE_HASH,
    _BATTERY_PACK_SOURCE_HASH,
    _HPPC_PARAMETERS_HASH,
)
_DYNAMIC_CONFIGURATION_HASH = build_dynamic_configuration_hash(
    _CONFIGURATION,
    _PLANT_SOURCE_HASH,
    _BATTERY_PACK_SOURCE_HASH,
    _HPPC_PARAMETERS_HASH,
)


@dataclass(frozen=True)
class DynamicScenarioSpec:
    scenario_id: str
    steps: int
    dt_s: float
    seed: int
    excitation_kind: str
    flow_direction: int
    initial_soc: float
    initial_battery_c: float
    initial_coolant_c: float
    initial_plate_c: float
    ambient_c: float
    compressor_command_rpm: tuple[float, ...]
    pump_command_rpm: tuple[float, ...]
    current_a: tuple[float, ...]


def ramp_limited_multilevel_sequence(levels, steps, dmax, seed):
    levels_array = np.asarray(levels, dtype=float)
    if levels_array.ndim != 1 or levels_array.size == 0:
        raise ValueError("levels must be a non-empty one-dimensional sequence")
    if not np.isfinite(levels_array).all():
        raise ValueError("levels must contain only finite values")
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)):
        raise ValueError("steps must be a positive integer")
    if steps <= 0:
        raise ValueError("steps must be greater than zero")
    try:
        dmax_value = float(dmax)
    except (TypeError, ValueError) as exc:
        raise ValueError("dmax must be a finite non-negative number") from exc
    if not math.isfinite(dmax_value) or dmax_value < 0.0:
        raise ValueError("dmax must be a finite non-negative number")

    rng = np.random.default_rng(seed)
    targets = rng.choice(levels_array, size=steps)
    values = np.empty(steps, dtype=float)
    values[0] = targets[0]
    for index in range(1, steps):
        delta = np.clip(
            targets[index] - values[index - 1], -dmax_value, dmax_value
        )
        values[index] = values[index - 1] + delta
    return values


def _dynamic_profiles(excitation_kind, steps, seed):
    if excitation_kind == "compressor-only":
        compressor = ramp_limited_multilevel_sequence(
            DYNAMIC_COMPRESSOR_LEVELS_RPM, steps, 600.0, seed
        )
        pump = np.full(steps, 3200.0)
    elif excitation_kind == "pump-only":
        compressor = np.full(steps, 3500.0)
        pump = ramp_limited_multilevel_sequence(
            DYNAMIC_PUMP_LEVELS_RPM, steps, 300.0, seed + 1
        )
    elif excitation_kind == "combined":
        compressor = ramp_limited_multilevel_sequence(
            DYNAMIC_COMPRESSOR_LEVELS_RPM, steps, 600.0, seed
        )
        pump = ramp_limited_multilevel_sequence(
            DYNAMIC_PUMP_LEVELS_RPM, steps, 300.0, seed + 1
        )
    else:
        raise ValueError(f"Unsupported excitation kind: {excitation_kind!r}")
    current = ramp_limited_multilevel_sequence(
        DYNAMIC_CURRENT_LEVELS_A, steps, 80.0, seed + 2
    )
    return tuple(compressor), tuple(pump), tuple(current)


def _make_dynamic_spec(
    scenario_id,
    steps,
    seed,
    excitation_kind,
    flow_direction,
    initial_soc,
    initial_battery_c,
    initial_coolant_c,
    initial_plate_c,
    ambient_c,
):
    compressor, pump, current = _dynamic_profiles(
        excitation_kind, steps, seed
    )
    return DynamicScenarioSpec(
        scenario_id=scenario_id,
        steps=steps,
        dt_s=5.0,
        seed=seed,
        excitation_kind=excitation_kind,
        flow_direction=flow_direction,
        initial_soc=initial_soc,
        initial_battery_c=initial_battery_c,
        initial_coolant_c=initial_coolant_c,
        initial_plate_c=initial_plate_c,
        ambient_c=ambient_c,
        compressor_command_rpm=compressor,
        pump_command_rpm=pump,
        current_a=current,
    )


def build_dynamic_scenarios(mode="full", seed=20260714):
    if mode == "smoke":
        definitions = (
            ("dynamic_smoke_compressor_only_forward", "compressor-only", 1),
            ("dynamic_smoke_pump_only_forward", "pump-only", 1),
            ("dynamic_smoke_combined_forward", "combined", 1),
            ("dynamic_smoke_combined_reverse", "combined", -1),
        )
        return [
            _make_dynamic_spec(
                scenario_id=scenario_id,
                steps=20,
                seed=seed + index,
                excitation_kind=kind,
                flow_direction=direction,
                initial_soc=0.8,
                initial_battery_c=31.0,
                initial_coolant_c=26.0,
                initial_plate_c=27.0,
                ambient_c=35.0,
            )
            for index, (scenario_id, kind, direction) in enumerate(definitions)
        ]
    if mode != "full":
        raise ValueError(f"Unsupported dynamic-scenario mode: {mode!r}")

    definitions = (
        ("dynamic_full_comp_low_forward", 120, "compressor-only", 1, 0.95, 24.0, 20.0, 21.0, 20.0),
        ("dynamic_full_comp_mid_reverse", 150, "compressor-only", -1, 0.75, 32.0, 27.0, 28.0, 30.0),
        ("dynamic_full_comp_high_forward", 180, "compressor-only", 1, 0.55, 40.0, 34.0, 35.0, 40.0),
        ("dynamic_full_comp_cross_reverse", 150, "compressor-only", -1, 0.85, 24.0, 34.0, 28.0, 30.0),
        ("dynamic_full_comp_cross_forward", 120, "compressor-only", 1, 0.65, 40.0, 20.0, 35.0, 40.0),
        ("dynamic_full_pump_low_reverse", 120, "pump-only", -1, 0.90, 24.0, 27.0, 28.0, 40.0),
        ("dynamic_full_pump_mid_forward", 150, "pump-only", 1, 0.70, 32.0, 34.0, 35.0, 20.0),
        ("dynamic_full_pump_high_reverse", 180, "pump-only", -1, 0.50, 40.0, 20.0, 21.0, 30.0),
        ("dynamic_full_pump_cross_forward", 120, "pump-only", 1, 0.80, 24.0, 34.0, 35.0, 30.0),
        ("dynamic_full_pump_cross_reverse", 150, "pump-only", -1, 0.60, 40.0, 27.0, 21.0, 20.0),
        ("dynamic_full_combined_low_forward", 120, "combined", 1, 0.85, 24.0, 34.0, 21.0, 30.0),
        ("dynamic_full_combined_mid_reverse", 150, "combined", -1, 0.65, 32.0, 20.0, 28.0, 40.0),
        ("dynamic_full_combined_high_forward", 180, "combined", 1, 0.45, 40.0, 27.0, 35.0, 20.0),
        ("dynamic_full_combined_cross_reverse", 150, "combined", -1, 0.80, 32.0, 34.0, 21.0, 30.0),
        ("dynamic_full_combined_cross_forward", 180, "combined", 1, 0.60, 24.0, 20.0, 35.0, 40.0),
    )
    return [
        _make_dynamic_spec(
            scenario_id=scenario_id,
            steps=steps,
            seed=seed + index,
            excitation_kind=kind,
            flow_direction=direction,
            initial_soc=initial_soc,
            initial_battery_c=initial_battery_c,
            initial_coolant_c=initial_coolant_c,
            initial_plate_c=initial_plate_c,
            ambient_c=ambient_c,
        )
        for index, (
            scenario_id,
            steps,
            kind,
            direction,
            initial_soc,
            initial_battery_c,
            initial_coolant_c,
            initial_plate_c,
            ambient_c,
        ) in enumerate(definitions)
    ]


def _dynamic_split_rank(seed, scenario_id):
    return hashlib.sha256(f"{seed}:{scenario_id}".encode("utf-8")).hexdigest()


def assign_dynamic_scenario_splits(specs, seed=20260714):
    specs = list(specs)
    scenario_ids = [spec.scenario_id for spec in specs]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Duplicate dynamic scenario_id values are not allowed")

    base_splits = assign_scenario_splits(scenario_ids, seed=seed)
    kinds = ("compressor-only", "pump-only", "combined")
    grouped = {
        kind: [spec for spec in specs if spec.excitation_kind == kind]
        for kind in kinds
    }
    is_full_structure = (
        len(specs) == 15
        and all(len(grouped[kind]) == 5 for kind in kinds)
        and all(
            {spec.flow_direction for spec in grouped[kind]} == {-1, 1}
            for kind in kinds
        )
    )
    if not is_full_structure:
        return base_splits

    rank = {
        scenario_id: _dynamic_split_rank(seed, scenario_id)
        for scenario_id in scenario_ids
    }
    candidates = {
        kind: sorted(
            (spec.scenario_id for spec in grouped[kind]),
            key=lambda scenario_id: (rank[scenario_id], scenario_id),
        )
        for kind in kinds
    }
    by_id = {spec.scenario_id: spec for spec in specs}
    best = None
    for validation_ids in product(*(candidates[kind] for kind in kinds)):
        if {by_id[value].flow_direction for value in validation_ids} != {-1, 1}:
            continue
        remaining = {
            kind: [
                value
                for value in candidates[kind]
                if value != validation_ids[index]
            ]
            for index, kind in enumerate(kinds)
        }
        for test_ids in product(*(remaining[kind] for kind in kinds)):
            if {by_id[value].flow_direction for value in test_ids} != {-1, 1}:
                continue
            score = (
                sum(base_splits[value] != "validation" for value in validation_ids)
                + sum(base_splits[value] != "test" for value in test_ids),
                sum(base_splits[value] != "validation" for value in validation_ids),
                tuple(rank[value] for value in validation_ids),
                tuple(rank[value] for value in test_ids),
            )
            if best is None or score < best[0]:
                best = (score, validation_ids, test_ids)
    if best is None:
        return base_splits

    validation_ids = set(best[1])
    test_ids = set(best[2])
    return {
        scenario_id: (
            "validation"
            if scenario_id in validation_ids
            else "test"
            if scenario_id in test_ids
            else "train"
        )
        for scenario_id in sorted(scenario_ids)
    }


def build_steady_grid(mode="full"):
    if mode == "smoke":
        return [
            (1999.0, 1600.0, 25.0, 35.0),
            (2000.0, 1600.0, 25.0, 35.0),
        ]
    if mode != "full":
        raise ValueError(f"Unsupported steady-grid mode: {mode!r}")

    return [
        (float(n_comp), float(n_pump), float(t_cool), float(t_ambient))
        for n_comp in COMPRESSOR_LEVELS_RPM
        for n_pump in PUMP_LEVELS_RPM
        for t_cool in COOLANT_LEVELS_C
        for t_ambient in AMBIENT_LEVELS_C
    ]


def _format_axis_value(value):
    return f"{float(value):g}"


def _scenario_id(n_comp, n_pump, t_cool, t_ambient):
    return (
        f"steady_nc{_format_axis_value(n_comp)}"
        f"_np{_format_axis_value(n_pump)}"
        f"_tc{_format_axis_value(t_cool)}"
        f"_ta{_format_axis_value(t_ambient)}"
    )


def _finite_float(value, default=0.0):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _required_finite_cycle_value(cycle, key, scenario_id):
    if key not in cycle:
        raise ValueError(
            f"{scenario_id}: required cycle output {key}=missing"
        )
    value = cycle[key]
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{scenario_id}: required cycle output {key}={value!r} is not numeric"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"{scenario_id}: required cycle output {key}={value!r} is not finite"
        )
    return number


def _off_capacity_limit(cycle, key, scenario_id):
    if key not in cycle:
        return 0.0
    return _required_finite_cycle_value(cycle, key, scenario_id)


def generate_steady_rows(points, seed=20260714):
    points = [tuple(float(value) for value in point) for point in points]
    if not points:
        raise ValueError("steady identification points must not be empty")
    if len(set(points)) != len(points):
        raise ValueError("Duplicate steady identification points are not allowed")
    scenario_ids = [_scenario_id(*point) for point in points]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("scenario_id collision after stable input formatting")
    scenario_splits = assign_scenario_splits(scenario_ids, seed=seed)
    rows = []

    for scenario_id, (n_comp, n_pump, t_cool, t_ambient) in zip(
        scenario_ids, points
    ):
        m_dot_cool, w_pump = pump_model(n_pump)
        m_dot_cool = _finite_float(m_dot_cool)
        w_pump = _finite_float(w_pump)
        n_fan = _finite_float(staged_fan_speed(n_comp))
        t_cool_k = t_cool + 273.15
        t_ambient_k = t_ambient + 273.15
        cycle = run_refrigeration_cycle(
            n_comp,
            n_fan,
            t_cool_k,
            m_dot_cool,
            t_ambient_k,
        )

        q_evap = _required_finite_cycle_value(cycle, "Q_evap", scenario_id)
        q_cond = _required_finite_cycle_value(cycle, "Q_cond", scenario_id)
        w_comp = _required_finite_cycle_value(cycle, "W_comp", scenario_id)
        t_evap_sat = _required_finite_cycle_value(
            cycle, "T_evap_sat", scenario_id
        )
        t_cond_sat = _required_finite_cycle_value(
            cycle, "T_cond_sat", scenario_id
        )
        if n_comp <= N_COMP_OFF_RPM:
            q_hx_potential = _off_capacity_limit(
                cycle, "Q_hx_potential", scenario_id
            )
            q_ref_max = _off_capacity_limit(cycle, "Q_ref_max", scenario_id)
            limit_type = "compressor_off"
        else:
            q_hx_potential = _required_finite_cycle_value(
                cycle, "Q_hx_potential", scenario_id
            )
            q_ref_max = _required_finite_cycle_value(
                cycle, "Q_ref_max", scenario_id
            )
            limit_type = (
                "heat_exchanger_limit"
                if q_hx_potential <= q_ref_max
                else "refrigerant_limit"
            )

        row = {
            "scenario_id": scenario_id,
            "split": scenario_splits[scenario_id],
            "time_s": 0.0,
            "flow_direction": 1,
            "n_comp_cmd_rpm": n_comp,
            "n_pump_cmd_rpm": n_pump,
            "n_comp_eff_rpm": n_comp,
            "n_pump_eff_rpm": n_pump,
            "q_gen_w": 0.0,
            "t_ambient_c": t_ambient,
            "t_batt_c": t_cool,
            "t_cool_c": t_cool,
            "t_plate_c": t_cool,
            "t_supply_c": t_cool,
            "t_return_c": t_cool,
            "q_evap_ss_w": q_evap,
            "q_evap_eff_w": q_evap,
            "q_cond_ss_w": q_cond,
            "q_cond_eff_w": q_cond,
            "dataset_kind": "steady",
            "dataset_seed": int(seed),
            "source_model": "thermal_system.run_refrigeration_cycle",
            "n_fan_cmd_rpm": n_fan,
            "n_fan_eff_rpm": n_fan,
            "m_dot_cool_kg_s": m_dot_cool,
            "w_pump_w": w_pump,
            "w_comp_w": w_comp,
            "w_fan_w": _finite_float(cycle.get("W_fan")),
            "q_hx_potential_w": q_hx_potential,
            "q_ref_max_w": q_ref_max,
            "t_cool_out_c": _finite_float(
                cycle.get("T_cool_out"), t_cool_k
            )
            - 273.15,
            "t_evap_sat_c": t_evap_sat - 273.15,
            "t_cond_sat_c": t_cond_sat - 273.15,
            "limit_type": limit_type,
            "plant_model_version": _PLANT_MODEL_VERSION,
            "plant_source_hash": _PLANT_SOURCE_HASH,
            "configuration_hash": _CONFIGURATION_HASH,
            **_CONFIGURATION,
        }
        rows.append(row)

    frame = pd.DataFrame(rows)
    validate_identification_frame(frame)
    return rows


def _as_finite_profile(values, expected_steps, name, scenario_id):
    try:
        profile = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{scenario_id}: {name} must be numeric") from exc
    if profile.ndim != 1 or profile.size != expected_steps:
        raise ValueError(
            f"{scenario_id}: {name} length must equal steps={expected_steps}"
        )
    if not np.isfinite(profile).all():
        raise ValueError(f"{scenario_id}: {name} contains non-finite values")
    return profile


def _validate_dynamic_scenario(spec):
    if not isinstance(spec, DynamicScenarioSpec):
        raise ValueError("dynamic scenario must be a DynamicScenarioSpec")
    if not isinstance(spec.scenario_id, str) or not spec.scenario_id.strip():
        raise ValueError("dynamic scenario_id must be a non-empty string")
    if isinstance(spec.steps, bool) or not isinstance(
        spec.steps, (int, np.integer)
    ) or spec.steps <= 0:
        raise ValueError(f"{spec.scenario_id}: steps must be a positive integer")
    if isinstance(spec.seed, (bool, np.bool_)) or not isinstance(
        spec.seed, (int, np.integer)
    ):
        raise ValueError(f"{spec.scenario_id}: seed must be a non-boolean integer")
    if float(spec.dt_s) != 5.0:
        raise ValueError(f"{spec.scenario_id}: dt_s must be exactly 5 seconds")
    if spec.excitation_kind not in {
        "compressor-only",
        "pump-only",
        "combined",
    }:
        raise ValueError(
            f"{spec.scenario_id}: invalid excitation_kind={spec.excitation_kind!r}"
        )
    if (
        isinstance(spec.flow_direction, (bool, np.bool_))
        or not isinstance(spec.flow_direction, (int, np.integer))
        or spec.flow_direction not in (-1, 1)
    ):
        raise ValueError(
            f"{spec.scenario_id}: flow_direction must be integer +1 or -1"
        )
    scalar_names = (
        "initial_soc",
        "initial_battery_c",
        "initial_coolant_c",
        "initial_plate_c",
        "ambient_c",
    )
    for name in scalar_names:
        value = getattr(spec, name)
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{spec.scenario_id}: {name} must be finite") from exc
        if not math.isfinite(number):
            raise ValueError(f"{spec.scenario_id}: {name} must be finite")
    if not 0.0 <= float(spec.initial_soc) <= 1.0:
        raise ValueError(f"{spec.scenario_id}: initial_soc must be in [0, 1]")

    compressor = _as_finite_profile(
        spec.compressor_command_rpm,
        spec.steps,
        "compressor_command_rpm",
        spec.scenario_id,
    )
    pump = _as_finite_profile(
        spec.pump_command_rpm,
        spec.steps,
        "pump_command_rpm",
        spec.scenario_id,
    )
    current = _as_finite_profile(
        spec.current_a, spec.steps, "current_a", spec.scenario_id
    )
    if compressor.min() < 1000.0 or compressor.max() > 6000.0:
        raise ValueError(
            f"{spec.scenario_id}: compressor command must be in [1000, 6000] rpm"
        )
    if pump.min() < 1600.0 or pump.max() > 4800.0:
        raise ValueError(
            f"{spec.scenario_id}: pump command must be in [1600, 4800] rpm"
        )
    if compressor.size > 1 and np.max(np.abs(np.diff(compressor))) > 600.0:
        raise ValueError(
            f"{spec.scenario_id}: compressor command exceeds 600 rpm/step"
        )
    if pump.size > 1 and np.max(np.abs(np.diff(pump))) > 300.0:
        raise ValueError(
            f"{spec.scenario_id}: pump command exceeds 300 rpm/step"
        )
    if spec.excitation_kind == "compressor-only" and np.ptp(pump) != 0.0:
        raise ValueError(
            f"{spec.scenario_id}: compressor-only requires a constant pump command"
        )
    if spec.excitation_kind == "pump-only" and np.ptp(compressor) != 0.0:
        raise ValueError(
            f"{spec.scenario_id}: pump-only requires a constant compressor command"
        )
    return compressor, pump, current


def _required_finite_step_value(result, key, scenario_id, step_index):
    return _required_finite_cycle_value(
        result, key, f"{scenario_id} step {step_index}"
    )


def _required_finite_array(result, key, scenario_id, step_index, size):
    if key not in result:
        raise ValueError(
            f"{scenario_id} step {step_index}: required thermal output {key}=missing"
        )
    try:
        values = np.asarray(result[key], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{scenario_id} step {step_index}: required thermal output "
            f"{key}={result[key]!r} is not numeric"
        ) from exc
    if values.shape != (size,) or not np.isfinite(values).all():
        raise ValueError(
            f"{scenario_id} step {step_index}: required thermal output "
            f"{key} has invalid shape or non-finite values"
        )
    return values


def _clear_pack_step_history(pack):
    histories = getattr(pack, "history", None)
    if histories is not None:
        for history in histories:
            history.clear()
    branch_histories = getattr(pack, "branch_currents_history", None)
    if branch_histories is not None:
        for history in branch_histories:
            history.clear()


def run_dynamic_scenario(spec, split="train"):
    compressor, pump, current = _validate_dynamic_scenario(spec)
    if split not in VALID_SPLITS:
        raise ValueError(f"{spec.scenario_id}: invalid split={split!r}")
    validate_hppc_parameter_file(pack_module.HPPC_PARAMS_PATH)

    pack_config = build_pack_config(
        total_current=float(current[0]),
        initial_soc=float(spec.initial_soc),
        initial_temp_c=float(spec.initial_battery_c),
    )
    pack = BatteryPack(pack_config)
    t_ambient_k = float(spec.ambient_c) + 273.15
    t_tank_k = float(spec.initial_coolant_c) + 273.15
    t_plate_k = np.full(pack.cols, float(spec.initial_plate_c) + 273.15)
    dynamic_state = initialize_refrigeration_dynamic_state(
        float(compressor[0]),
        float(pump[0]),
        initial_temp_k=t_tank_k,
        dt=float(spec.dt_s),
    )
    rows = []

    for index in range(spec.steps):
        n_comp_cmd = float(compressor[index])
        n_pump_cmd = float(pump[index])
        t_tank_input_k = t_tank_k
        pack.current = float(current[index])
        thermal_step = simulate_thermal_loop_step(
            pack=pack,
            T_tank_K=t_tank_input_k,
            T_plate_K_array=t_plate_k,
            N_comp_cmd=n_comp_cmd,
            N_pump_cmd=n_pump_cmd,
            T_outdoor=t_ambient_k,
            dt=float(spec.dt_s),
            is_reversed=(spec.flow_direction < 0),
            dynamic_state=dynamic_state,
        )

        t_tank_k = _required_finite_step_value(
            thermal_step, "T_tank_K", spec.scenario_id, index
        )
        t_plate_k = _required_finite_array(
            thermal_step,
            "T_plate_K_array",
            spec.scenario_id,
            index,
            pack.cols,
        )
        n_comp_eff = _required_finite_step_value(
            thermal_step, "N_comp_eff", spec.scenario_id, index
        )
        n_pump_eff = _required_finite_step_value(
            thermal_step, "N_pump_eff", spec.scenario_id, index
        )
        n_fan_cmd = _required_finite_step_value(
            thermal_step, "N_fan_cmd", spec.scenario_id, index
        )
        n_fan_eff = _required_finite_step_value(
            thermal_step, "N_fan_eff", spec.scenario_id, index
        )
        q_evap_eff = _required_finite_step_value(
            thermal_step, "Q_dot_evap", spec.scenario_id, index
        )
        q_cond_eff = _required_finite_step_value(
            thermal_step, "Q_dot_cond", spec.scenario_id, index
        )
        t_supply_k = _required_finite_step_value(
            thermal_step, "T_pipe_supply_K", spec.scenario_id, index
        )
        t_return_k = _required_finite_step_value(
            thermal_step, "T_pipe_return_K", spec.scenario_id, index
        )
        dynamic_state = thermal_step.get("dynamic_state")
        if not isinstance(dynamic_state, Mapping):
            raise ValueError(
                f"{spec.scenario_id} step {index}: required thermal output "
                "dynamic_state is missing or not a mapping"
            )

        pack.step(
            float(spec.dt_s), T_plate=t_plate_k, T_cabinet=t_ambient_k
        )
        t_batt_k = float(pack.get_avg_temp())
        if not math.isfinite(t_batt_k):
            raise ValueError(
                f"{spec.scenario_id} step {index}: battery temperature is not finite"
            )
        q_gen_w = ((float(pack.current) / 4.0) ** 2) * 0.001 * 52.0
        _clear_pack_step_history(pack)

        diagnostic_pump = pump_model(n_pump_eff)
        if not isinstance(diagnostic_pump, (tuple, list)) or len(diagnostic_pump) != 2:
            raise ValueError(
                f"{spec.scenario_id} step {index}: pump_model returned invalid output"
            )
        m_dot_cool = _required_finite_step_value(
            {"m_dot_cool": diagnostic_pump[0]},
            "m_dot_cool",
            spec.scenario_id,
            index,
        )
        w_pump_ss = _required_finite_step_value(
            {"w_pump": diagnostic_pump[1]},
            "w_pump",
            spec.scenario_id,
            index,
        )
        try:
            n_fan_ss = float(staged_fan_speed(n_comp_eff))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{spec.scenario_id} step {index}: diagnostic n_fan_ss "
                "is not numeric"
            ) from exc
        if not math.isfinite(n_fan_ss):
            raise ValueError(
                f"{spec.scenario_id} step {index}: diagnostic n_fan_ss "
                "is not finite"
            )
        cycle = run_refrigeration_cycle(
            n_comp_eff,
            n_fan_ss,
            t_tank_input_k,
            m_dot_cool,
            t_ambient_k,
        )
        diagnostic_context = f"{spec.scenario_id} step {index}"
        q_evap_ss = _required_finite_cycle_value(
            cycle, "Q_evap", diagnostic_context
        )
        q_cond_ss = _required_finite_cycle_value(
            cycle, "Q_cond", diagnostic_context
        )
        w_comp_ss = _required_finite_cycle_value(
            cycle, "W_comp", diagnostic_context
        )
        w_fan_ss = _required_finite_cycle_value(
            cycle, "W_fan", diagnostic_context
        )
        t_cool_out_k = _required_finite_cycle_value(
            cycle, "T_cool_out", diagnostic_context
        )
        t_evap_sat_k = _required_finite_cycle_value(
            cycle, "T_evap_sat", diagnostic_context
        )
        t_cond_sat_k = _required_finite_cycle_value(
            cycle, "T_cond_sat", diagnostic_context
        )
        if n_comp_eff <= N_COMP_OFF_RPM:
            q_hx_potential = _off_capacity_limit(
                cycle, "Q_hx_potential", diagnostic_context
            )
            q_ref_max = _off_capacity_limit(
                cycle, "Q_ref_max", diagnostic_context
            )
            limit_type = "compressor_off"
        else:
            q_hx_potential = _required_finite_cycle_value(
                cycle, "Q_hx_potential", diagnostic_context
            )
            q_ref_max = _required_finite_cycle_value(
                cycle, "Q_ref_max", diagnostic_context
            )
            limit_type = (
                "heat_exchanger_limit"
                if q_hx_potential <= q_ref_max
                else "refrigerant_limit"
            )

        row = {
            "scenario_id": spec.scenario_id,
            "split": split,
            "time_s": (index + 1) * float(spec.dt_s),
            "flow_direction": spec.flow_direction,
            "n_comp_cmd_rpm": n_comp_cmd,
            "n_pump_cmd_rpm": n_pump_cmd,
            "n_comp_eff_rpm": n_comp_eff,
            "n_pump_eff_rpm": n_pump_eff,
            "q_gen_w": q_gen_w,
            "t_ambient_c": float(spec.ambient_c),
            "t_batt_c": t_batt_k - 273.15,
            "t_cool_c": t_tank_k - 273.15,
            "t_cool_cycle_input_c": t_tank_input_k - 273.15,
            "t_plate_c": float(np.mean(t_plate_k)) - 273.15,
            "t_supply_c": t_supply_k - 273.15,
            "t_return_c": t_return_k - 273.15,
            "q_evap_ss_w": q_evap_ss,
            "q_evap_eff_w": q_evap_eff,
            "q_cond_ss_w": q_cond_ss,
            "q_cond_eff_w": q_cond_eff,
            "dataset_kind": "dynamic",
            "source_model": "thermal_loop.simulate_thermal_loop_step",
            "dt_s": float(spec.dt_s),
            "scenario_seed": int(spec.seed),
            "excitation_kind": spec.excitation_kind,
            "current_a": float(pack.current),
            "initial_soc": float(spec.initial_soc),
            "n_fan_cmd_rpm": n_fan_cmd,
            "n_fan_eff_rpm": n_fan_eff,
            "n_fan_ss_rpm": n_fan_ss,
            "m_dot_cool_kg_s": m_dot_cool,
            "w_pump_w": _required_finite_step_value(
                thermal_step, "W_pump_val", spec.scenario_id, index
            ),
            "w_pump_ss_w": w_pump_ss,
            "w_comp_w": _required_finite_step_value(
                thermal_step, "W_comp_real", spec.scenario_id, index
            ),
            "w_comp_ss_w": w_comp_ss,
            "w_fan_w": _required_finite_step_value(
                thermal_step, "W_fan_real", spec.scenario_id, index
            ),
            "w_fan_ss_w": w_fan_ss,
            "q_hx_potential_w": q_hx_potential,
            "q_ref_max_w": q_ref_max,
            "t_cool_out_c": t_cool_out_k - 273.15,
            "t_evap_sat_c": t_evap_sat_k - 273.15,
            "t_cond_sat_c": t_cond_sat_k - 273.15,
            "limit_type": limit_type,
            "plant_model_version": _PLANT_MODEL_VERSION,
            "plant_source_hash": _PLANT_SOURCE_HASH,
            "battery_pack_source_hash": _BATTERY_PACK_SOURCE_HASH,
            "hppc_parameters_hash": _HPPC_PARAMETERS_HASH,
            "hppc_data_source": "file",
            "hppc_fallback_used": False,
            "dynamic_source_hash": _DYNAMIC_SOURCE_HASH,
            "configuration_hash": _DYNAMIC_CONFIGURATION_HASH,
            **_CONFIGURATION,
        }
        rows.append(row)

    return rows


def generate_dynamic_rows(specs, seed=20260714, progress=False):
    specs = list(specs)
    if not specs:
        raise ValueError("dynamic identification scenarios must not be empty")
    scenario_ids = [spec.scenario_id for spec in specs]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Duplicate dynamic scenario_id values are not allowed")
    for spec in specs:
        _validate_dynamic_scenario(spec)
    scenario_splits = assign_dynamic_scenario_splits(specs, seed=seed)
    if set(scenario_splits) != set(scenario_ids) or any(
        split not in VALID_SPLITS for split in scenario_splits.values()
    ):
        raise ValueError("assign_scenario_splits returned an invalid assignment")

    rows = []
    for index, spec in enumerate(specs, start=1):
        if progress:
            print(
                f"Generating dynamic scenario {index}/{len(specs)}: "
                f"{spec.scenario_id}",
                flush=True,
            )
        scenario_rows = run_dynamic_scenario(
            spec, scenario_splits[spec.scenario_id]
        )
        rows.extend(scenario_rows)
        if progress:
            print(
                f"Completed dynamic scenario {index}/{len(specs)}: "
                f"{spec.scenario_id} rows={len(scenario_rows)}",
                flush=True,
            )
    for row in rows:
        row["dataset_seed"] = int(seed)
    frame = pd.DataFrame(rows)
    validate_identification_frame(frame)
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Dynamic identification rows contain NaN or Inf values")
    return rows


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate steady and dynamic MPC predictor identification data."
    )
    parser.add_argument(
        "--dataset", choices=("steady", "dynamic", "all"), default="steady"
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing requested CSV targets atomically.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mpc_predictor_identification_v1"),
    )
    return parser.parse_args()


def _requested_datasets(dataset):
    return ("steady", "dynamic") if dataset == "all" else (dataset,)


def _atomic_publish_frames(frames, targets, overwrite):
    existing = [path for path in targets.values() if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output target already exists: {joined}. Use --overwrite to replace it."
        )

    token = uuid.uuid4().hex
    temporary = {
        kind: target.with_name(f".{target.name}.{token}.tmp")
        for kind, target in targets.items()
    }
    backups = {}
    published = []
    try:
        for kind, frame in frames.items():
            frame.to_csv(temporary[kind], index=False, encoding="utf-8")

        if overwrite:
            for kind, target in targets.items():
                if target.exists():
                    backup = target.with_name(f".{target.name}.{token}.bak")
                    os.replace(target, backup)
                    backups[kind] = backup

        for kind, target in targets.items():
            if overwrite:
                os.replace(temporary[kind], target)
                published.append(kind)
            else:
                try:
                    os.link(temporary[kind], target)
                except FileExistsError as exc:
                    raise FileExistsError(
                        f"Output target appeared during publish: {target}. "
                        "Use --overwrite to replace it."
                    ) from exc
                published.append(kind)
                temporary[kind].unlink()
    except Exception as publish_error:
        rollback_errors = []
        for kind in published:
            target = targets[kind]
            try:
                if target.exists():
                    target.unlink()
            except Exception as exc:
                rollback_errors.append(
                    f"remove published target {target}: {exc}"
                )
        for kind, backup in backups.items():
            try:
                if backup.exists():
                    os.replace(backup, targets[kind])
            except Exception as exc:
                rollback_errors.append(
                    f"restore backup {backup} -> {targets[kind]}: {exc}"
                )
        for path in temporary.values():
            try:
                if path.exists():
                    path.unlink()
            except Exception as exc:
                rollback_errors.append(f"remove temporary file {path}: {exc}")
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                f"Publish failed: {publish_error}; rollback errors: {details}"
            ) from publish_error
        raise
    else:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()
        for path in temporary.values():
            if path.exists():
                path.unlink()


def main():
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets = _requested_datasets(args.dataset)
    targets = {
        dataset_kind: args.output_root / f"{dataset_kind}_{args.mode}.csv"
        for dataset_kind in datasets
    }
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output target already exists: {joined}. Use --overwrite to replace it."
        )

    frames = {}
    for dataset_kind in datasets:
        if dataset_kind == "steady":
            rows = generate_steady_rows(
                build_steady_grid(args.mode), seed=args.seed
            )
        else:
            rows = generate_dynamic_rows(
                build_dynamic_scenarios(args.mode, seed=args.seed),
                seed=args.seed,
                progress=True,
            )
        frame = pd.DataFrame(rows)
        frame["dataset_seed"] = int(args.seed)
        validate_identification_frame(frame)
        frames[dataset_kind] = frame

    _atomic_publish_frames(frames, targets, overwrite=args.overwrite)
    for dataset_kind in datasets:
        configuration_hash = (
            _CONFIGURATION_HASH
            if dataset_kind == "steady"
            else _DYNAMIC_CONFIGURATION_HASH
        )
        print(
            f"Wrote {len(frames[dataset_kind])} rows to "
            f"{targets[dataset_kind]} seed={args.seed} "
            f"configuration_hash={configuration_hash}"
        )


if __name__ == "__main__":
    main()
