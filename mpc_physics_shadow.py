"""Read-only online diagnostics for the lightweight physics predictor P.

The shadow predictor starts every forecast from the currently observed plant
state.  It rolls P forward with the current actuator commands held constant.
Its output is diagnostic data only; this module has no control-system imports
and no mechanism for writing commands back to the simulator.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Mapping, Sequence
from numbers import Real
from pathlib import Path

from mpc_physics_predictor import (
    initialize_physics_state,
    load_physics_artifact,
    step_physics_predictor,
    validate_physics_artifact,
)


OBSERVED_STATE_FIELDS = (
    "n_comp_eff_rpm",
    "n_pump_eff_rpm",
    "q_cond_w",
    "q_evap_w",
    "t_supply_c",
    "t_plate_c",
    "t_return_c",
    "t_batt_c",
    "t_cool_c",
)


def battery_heat_generation_w(total_current_a: object) -> float:
    """Match the heat-load definition used to identify the P model."""
    if isinstance(total_current_a, bool) or not isinstance(total_current_a, Real):
        raise TypeError("total_current_a must be a finite real number")
    current = float(total_current_a)
    if not math.isfinite(current):
        raise ValueError("total_current_a must be finite")
    return ((current / 4.0) ** 2) * 0.001 * 52.0


def _horizon_label(horizon_s: float) -> str:
    rounded = round(horizon_s)
    if math.isclose(horizon_s, rounded, rel_tol=0.0, abs_tol=1e-9):
        return str(int(rounded))
    return f"{horizon_s:g}".replace(".", "p")


class PhysicsPShadowPredictor:
    """Generate independent P forecasts without changing plant or controls."""

    def __init__(
        self,
        artifact: str | Path | Mapping,
        *,
        dt_s: float,
        horizons_s: Sequence[float] = (50.0, 100.0, 300.0),
    ) -> None:
        self.dt_s = float(dt_s)
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be a positive finite number")
        if isinstance(artifact, Mapping):
            self.artifact = validate_physics_artifact(copy.deepcopy(dict(artifact)))
        else:
            self.artifact = load_physics_artifact(artifact, require_validated=True)

        if not horizons_s:
            raise ValueError("horizons_s must not be empty")
        horizon_steps: list[tuple[float, int]] = []
        for raw_horizon in horizons_s:
            horizon = float(raw_horizon)
            steps = int(round(horizon / self.dt_s))
            if (
                not math.isfinite(horizon)
                or horizon <= 0.0
                or steps <= 0
                or not math.isclose(
                    steps * self.dt_s,
                    horizon,
                    rel_tol=0.0,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError("each horizon must be a positive multiple of dt_s")
            horizon_steps.append((horizon, steps))
        if len({steps for _horizon, steps in horizon_steps}) != len(horizon_steps):
            raise ValueError("horizons_s must not contain duplicate step counts")
        self.horizon_steps = tuple(sorted(horizon_steps, key=lambda item: item[1]))
        self.max_forecast_steps = max(steps for _horizon, steps in self.horizon_steps)

    def forecast(
        self,
        *,
        observed_state: Mapping[str, object],
        n_comp_cmd_rpm: object,
        n_pump_cmd_rpm: object,
        q_gen_preview_w: Sequence[float],
        t_ambient_c: object,
        n_comp_cmd_preview_rpm: Sequence[float] | None = None,
        n_pump_cmd_preview_rpm: Sequence[float] | None = None,
    ) -> dict[str, object]:
        missing = [name for name in OBSERVED_STATE_FIELDS if name not in observed_state]
        if missing:
            raise KeyError(f"observed_state missing fields: {', '.join(missing)}")
        preview = tuple(q_gen_preview_w)
        if not preview:
            raise ValueError("q_gen_preview_w must not be empty")
        if (n_comp_cmd_preview_rpm is None) != (n_pump_cmd_preview_rpm is None):
            raise ValueError("compressor and pump command previews must be provided together")
        if n_comp_cmd_preview_rpm is None:
            comp_preview = (float(n_comp_cmd_rpm),)
            pump_preview = (float(n_pump_cmd_rpm),)
            command_assumption = "hold_current"
        else:
            comp_preview = tuple(n_comp_cmd_preview_rpm)
            pump_preview = tuple(n_pump_cmd_preview_rpm)
            if not comp_preview or not pump_preview:
                raise ValueError("command previews must not be empty")
            command_assumption = "provided_plan_hold_last"

        state = initialize_physics_state(
            **{name: observed_state[name] for name in OBSERVED_STATE_FIELDS}
        )
        record: dict[str, object] = {
            "P_Shadow_Enabled": 1,
            "P_Shadow_Command_Assumption": command_assumption,
            "P_Shadow_Load_Assumption": "provided_preview_hold_last",
            "P_Shadow_Hold_Comp_Command_RPM": float(n_comp_cmd_rpm),
            "P_Shadow_Hold_Pump_Command_RPM": float(n_pump_cmd_rpm),
            "P_Shadow_Comp_Plan_Length": len(comp_preview),
            "P_Shadow_Pump_Plan_Length": len(pump_preview),
        }
        started = time.perf_counter()
        horizons_by_step = {steps: horizon for horizon, steps in self.horizon_steps}
        for step_index in range(1, self.max_forecast_steps + 1):
            q_gen = preview[min(step_index - 1, len(preview) - 1)]
            n_comp_step = comp_preview[min(step_index - 1, len(comp_preview) - 1)]
            n_pump_step = pump_preview[min(step_index - 1, len(pump_preview) - 1)]
            state = step_physics_predictor(
                state,
                n_comp_cmd_rpm=n_comp_step,
                n_pump_cmd_rpm=n_pump_step,
                q_gen_w=q_gen,
                t_ambient_c=t_ambient_c,
                dt_s=self.dt_s,
                artifact=self.artifact,
            )
            if step_index not in horizons_by_step:
                continue
            label = _horizon_label(horizons_by_step[step_index])
            record.update(
                {
                    f"P_Shadow_T_Batt_Pred_{label}s_C": state.t_batt_c,
                    f"P_Shadow_T_Cool_Pred_{label}s_C": state.t_cool_c,
                    f"P_Shadow_T_Plate_Pred_{label}s_C": state.t_plate_c,
                    f"P_Shadow_T_Supply_Pred_{label}s_C": state.t_supply_c,
                    f"P_Shadow_T_Return_Pred_{label}s_C": state.t_return_c,
                    f"P_Shadow_Q_Evap_Pred_{label}s_W": state.q_evap_w,
                }
            )
        record["P_Shadow_Prediction_Time_Ms"] = (
            time.perf_counter() - started
        ) * 1000.0
        return record
