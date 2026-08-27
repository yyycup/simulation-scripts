"""Single-criterion predictive flow-reversal gate for the BTMS MPC tests."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PredictiveDeltaTDecision:
    trigger: bool
    crossing_time_s: float
    hold_time_satisfied: bool


class PredictiveDeltaTGate:
    """Trigger only when predicted maximum temperature spread crosses a limit soon."""

    def __init__(self, threshold_c, buffer_s, min_hold_s):
        self.threshold_c = float(threshold_c)
        self.buffer_s = float(buffer_s)
        self.min_hold_s = float(min_hold_s)

    def evaluate(self, delta_t_series_c, dt_s, current_time_s, last_switch_time_s):
        series = np.asarray(delta_t_series_c, dtype=float)
        crossings = np.flatnonzero(series >= self.threshold_c)
        crossing_time_s = float(crossings[0] * float(dt_s)) if crossings.size else float("nan")
        hold_time_satisfied = (float(current_time_s) - float(last_switch_time_s)) >= self.min_hold_s
        in_buffer = bool(np.isfinite(crossing_time_s) and crossing_time_s <= self.buffer_s)
        return PredictiveDeltaTDecision(
            trigger=bool(hold_time_satisfied and in_buffer),
            crossing_time_s=crossing_time_s,
            hold_time_satisfied=bool(hold_time_satisfied),
        )
