"""Static chiller surrogate surfaces for the do-mpc cluster controller.

Fits two quadratic response surfaces against the frozen ``ClosedR134aCycle``:

- evaporator cooling capacity ``q_evap_cycle_w(w, mdot, t_tank)``
- compressor shaft power ``compressor_shaft_power_w(w, mdot, t_tank)``

Features per sample (8):
``[1, w, m, t, w^2, t^2, w*m, w*t]`` with ``w`` in rpm, ``m`` in kg/s and
``t`` in kelvin. The fan speed and ambient temperature are frozen to the
Stage 8D2 validation values. Coefficients are persisted in
``model_data/chiller_surrogates.json`` so the controller build is cheap.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from cluster_plant_v2.parameters import (
    MAXIMUM_COMPRESSOR_SPEED_RPM,
    MINIMUM_COMPRESSOR_SPEED_RPM,
)


SURROGATES_PATH = (
    Path(__file__).resolve().parents[1] / "model_data" / "chiller_surrogates.json"
)

FEATURE_NAMES = [
    "one",
    "w",
    "m",
    "t",
    "w2",
    "t2",
    "wm",
    "wt",
]

DEFAULT_COMPRESSOR_GRID_RPM = np.array(
    [1000.0, 1800.0, 2600.0, 3400.0, 4200.0, 5000.0, 6000.0]
)
DEFAULT_PUMP_GRID_RPM = np.array([1600.0, 2400.0, 3200.0, 4000.0, 4800.0])
DEFAULT_TANK_TEMP_GRID_K = np.array(
    [283.15, 288.15, 293.15, 298.15, 303.15, 308.15, 313.15]
)


def feature_matrix(
    compressor_rpm: np.ndarray,
    mass_flow_kg_s: np.ndarray,
    tank_temperature_k: np.ndarray,
) -> np.ndarray:
    """Build the 8-column quadratic feature matrix for the surrogate fit."""
    w = np.asarray(compressor_rpm, dtype=float)
    m = np.asarray(mass_flow_kg_s, dtype=float)
    t = np.asarray(tank_temperature_k, dtype=float)
    return np.column_stack(
        [
            np.ones_like(w),
            w,
            m,
            t,
            w * w,
            t * t,
            w * m,
            w * t,
        ]
    )


def fit_surrogates(
    *,
    refrigeration_cycle,
    pump,
    hydraulic_network,
    fan_speed_rpm: float = 1200.0,
    ambient_temperature_k: float = 308.15,
    compressor_grid_rpm: np.ndarray | None = None,
    pump_grid_rpm: np.ndarray | None = None,
    tank_temperature_grid_k: np.ndarray | None = None,
) -> dict[str, object]:
    """Sample the frozen cycle on a grid and fit both quadratic surfaces."""
    if compressor_grid_rpm is None:
        compressor_grid_rpm = DEFAULT_COMPRESSOR_GRID_RPM
    if pump_grid_rpm is None:
        pump_grid_rpm = DEFAULT_PUMP_GRID_RPM
    if tank_temperature_grid_k is None:
        tank_temperature_grid_k = DEFAULT_TANK_TEMP_GRID_K

    compressor_grid_rpm = np.asarray(compressor_grid_rpm, dtype=float)
    pump_grid_rpm = np.asarray(pump_grid_rpm, dtype=float)
    tank_temperature_grid_k = np.asarray(tank_temperature_grid_k, dtype=float)

    pump_flows = []
    for pump_rpm in pump_grid_rpm:
        operating_point = pump.solve_operating_point(pump_rpm, hydraulic_network)
        flow = float(operating_point["total_mass_flow_kg_s"])
        if not np.isfinite(flow) or flow <= 0.0:
            raise ValueError(f"pump operating point invalid at {pump_rpm} rpm")
        pump_flows.append(flow)
    pump_flows = np.asarray(pump_flows, dtype=float)

    samples_w: list[float] = []
    samples_m: list[float] = []
    samples_t: list[float] = []
    samples_q: list[float] = []
    samples_p: list[float] = []
    failures = 0
    for compressor_rpm in compressor_grid_rpm:
        for flow in pump_flows:
            for tank_temperature_k in tank_temperature_grid_k:
                try:
                    result = refrigeration_cycle.solve(
                        compressor_speed_rpm=float(compressor_rpm),
                        fan_speed_rpm=float(fan_speed_rpm),
                        coolant_inlet_temperature_k=float(tank_temperature_k),
                        coolant_mass_flow_kg_s=float(flow),
                        ambient_temperature_k=float(ambient_temperature_k),
                    )
                except ValueError:
                    # Physically infeasible corner (e.g. condensing below
                    # evaporating temperature); skip the grid point.
                    failures += 1
                    continue
                if not result["solver_success"]:
                    failures += 1
                    continue
                samples_w.append(float(compressor_rpm))
                samples_m.append(float(flow))
                samples_t.append(float(tank_temperature_k))
                samples_q.append(float(result["q_evaporator_w"]))
                samples_p.append(float(result["compressor_shaft_power_w"]))

    if failures:
        print(
            f"chiller surrogate fit: skipped {failures} failed cycle solves"
        )
    if len(samples_w) < 3 * len(FEATURE_NAMES):
        raise RuntimeError("too few successful cycle solves for a stable fit")

    features = feature_matrix(
        np.asarray(samples_w),
        np.asarray(samples_m),
        np.asarray(samples_t),
    )
    targets = {"q_evap_cycle_w": np.asarray(samples_q),
               "compressor_shaft_power_w": np.asarray(samples_p)}
    coefficients: dict[str, list[float]] = {}
    residuals: dict[str, float] = {}
    for name, target in targets.items():
        solution = least_squares(
            lambda beta: features @ beta - target,
            x0=np.zeros(len(FEATURE_NAMES)),
        )
        coefficients[name] = [float(value) for value in solution.x]
        residuals[name] = float(np.sqrt(np.mean(solution.fun**2)))

    return {
        "feature_names": FEATURE_NAMES,
        "coefficients": coefficients,
        "fit_rmse_w": residuals,
        "n_samples": len(samples_w),
        "n_failures": int(failures),
        "fan_speed_rpm": float(fan_speed_rpm),
        "ambient_temperature_k": float(ambient_temperature_k),
        "compressor_bounds_rpm": [
            float(MINIMUM_COMPRESSOR_SPEED_RPM),
            float(MAXIMUM_COMPRESSOR_SPEED_RPM),
        ],
    }


def save_surrogates(
    surrogates: dict[str, object], path: Path = SURROGATES_PATH
) -> Path:
    """Persist the fitted surrogate coefficients next to the other frozen data."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(surrogates, indent=2), encoding="utf-8"
    )
    return Path(path)


def load_surrogates(path: Path = SURROGATES_PATH) -> dict[str, object]:
    """Load the frozen surrogate coefficients, fitting them on first use."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            "chiller surrogate file missing; run "
            "cluster_plant_v2.control.chiller_surrogates fit first "
            f"(expected at {path})"
        )
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
        build_engineering_pump,
        build_medium_header_network,
    )
    from cluster_plant_v2.refrigeration import ClosedR134aCycle

    network = build_medium_header_network()
    fitted = fit_surrogates(
        refrigeration_cycle=ClosedR134aCycle(),
        pump=build_engineering_pump(network),
        hydraulic_network=network,
    )
    save_surrogates(fitted)
    print(f"saved {SURROGATES_PATH}")
    print(f"rmse q_evap = {fitted['fit_rmse_w']['q_evap_cycle_w']:.1f} W")
    print(
        "rmse shaft power = "
        f"{fitted['fit_rmse_w']['compressor_shaft_power_w']:.1f} W"
    )
