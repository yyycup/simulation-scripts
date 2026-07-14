import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import thermal_batch_config as thermal_batch_config_module
import thermal_loop as thermal_loop_module
import thermal_system as thermal_system_module

from predictor_identification_data import (
    REQUIRED_COLUMNS,
    assign_scenario_splits,
    validate_identification_frame,
)
from thermal_batch_config import (
    EVAP_CAP_FACTOR,
    EVAP_FLOW_EXP,
    EVAP_UA_FACTOR,
    N_COMP_MAX_RPM,
    N_COMP_MIN_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
)
from thermal_loop import staged_fan_speed
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
COOLANT_LEVELS_C = (20, 25, 30, 35)
AMBIENT_LEVELS_C = (20, 25, 30, 35, 40)

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


_PLANT_SOURCE_HASH = _build_plant_source_hash(
    (thermal_system_module, thermal_loop_module, thermal_batch_config_module)
)
_PLANT_MODEL_VERSION = f"sha256:{_PLANT_SOURCE_HASH[:12]}"
_CONFIGURATION_HASH = build_configuration_hash(
    _CONFIGURATION, _PLANT_SOURCE_HASH
)


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
        if n_comp < N_COMP_MIN_RPM:
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate steady-state MPC predictor identification data."
    )
    parser.add_argument("--dataset", choices=("steady",), default="steady")
    parser.add_argument("--mode", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/mpc_predictor_identification_v1"),
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    rows = generate_steady_rows(build_steady_grid(args.mode), seed=args.seed)
    frame = pd.DataFrame(rows)
    validate_identification_frame(frame)
    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"steady_{args.mode}.csv"
    frame.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Wrote {len(frame)} rows to {output_path}")


if __name__ == "__main__":
    main()
