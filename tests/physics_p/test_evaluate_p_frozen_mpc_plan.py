import unittest
from unittest import mock

import numpy as np
import pandas as pd

from experiments.physics_p.evaluation import evaluate_p_frozen_mpc_plan as frozen
from experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan import (
    _extend_plan,
    evaluate_frozen_forecasts,
)


class _FakePack:
    cols = 1
    Ns = 1

    def __init__(self):
        self.total_current = 0.0
        self.history = [[]]
        self.step_count = 0

    def step(self, *_args, **_kwargs):
        self.step_count += 1

    def get_avg_temp(self):
        return 298.15


class FrozenMpcPlanEvaluationTests(unittest.TestCase):
    def test_short_plan_holds_its_last_command(self):
        self.assertEqual(_extend_plan([2000.0, 2300.0], 4).tolist(), [2000.0, 2300.0, 2300.0, 2300.0])

    def test_horizon_targets_post_step_row_steps_minus_one(self):
        frame = pd.DataFrame(
            {
                "Average temperature": range(60),
                "Coolant temperature": range(100, 160),
                "Plate solid temperature": range(200, 260),
                "Supply pipe coolant temperature": range(300, 360),
                "Return pipe coolant temperature": range(400, 460),
                "Evaporator cooling rate (kW)": [value / 1000.0 for value in range(500, 560)],
            }
        )
        forecast = {"P_Shadow_Command_Assumption": "provided_plan_hold_last"}
        actual_starts = {
            "T_Batt": 0,
            "T_Cool": 100,
            "T_Plate": 200,
            "T_Supply": 300,
            "T_Return": 400,
            "Q_Evap": 500,
        }
        for horizon_s, target_index in ((50, 9), (100, 19), (300, 59)):
            for state, start in actual_starts.items():
                unit = "W" if state == "Q_Evap" else "C"
                forecast[f"P_Shadow_{state}_Pred_{horizon_s}s_{unit}"] = start + target_index

        details = evaluate_frozen_forecasts("peak", frame, {"New": forecast})

        self.assertEqual(set(details["target_index"]), {9, 19, 59})
        self.assertTrue((details["abs_error"] == 0.0).all())

    def test_detailed_rollout_does_not_advance_saved_origin_pack(self):
        pack = _FakePack()
        captured = {
            "pack": pack,
            "t_tank_k": 298.15,
            "t_plate_k_array": np.array([298.15]),
            "dynamic_state": {},
            "is_reversed": False,
            "current_profile": np.ones(60),
            "n_comp_plan_rpm": np.full(60, 2000.0),
            "n_pump_plan_rpm": np.full(60, 1600.0),
            "origin_time_s": 0.0,
        }

        def fake_thermal_step(**kwargs):
            return {
                "dynamic_state": {},
                "T_tank_K": kwargs["T_tank_K"],
                "T_plate_K_array": kwargs["T_plate_K_array"],
                "T_pipe_supply_K": 298.15,
                "T_pipe_return_K": 298.15,
                "Q_dot_evap": 0.0,
                "N_comp_eff": kwargs["N_comp_cmd"],
                "N_pump_eff": kwargs["N_pump_cmd"],
            }

        with mock.patch.object(
            frozen.sim, "simulate_thermal_loop_step", side_effect=fake_thermal_step
        ):
            frame = frozen._rollout_detailed_plant(captured)

        self.assertEqual(len(frame), 60)
        self.assertEqual(pack.step_count, 0)


if __name__ == "__main__":
    unittest.main()
