import copy
import unittest

import pandas as pd

from evaluate_p_shadow_actual_replay import evaluate_actual_command_replay
from mpc_physics_predictor import (
    DEFAULT_PHYSICS_ARTIFACT,
    initialize_physics_state,
    step_physics_predictor,
)


def zero_delay_artifact():
    artifact = copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT)
    artifact["dynamic"].update(
        {
            "evap_input_delay_s": 0.0,
            "pump_flow_delay_s": 0.0,
            "supply_delay_s": 0.0,
            "return_delay_s": 0.0,
        }
    )
    return artifact


def exact_trajectory_frame():
    artifact = zero_delay_artifact()
    state = initialize_physics_state(
        n_comp_eff_rpm=2200.0,
        n_pump_eff_rpm=2000.0,
        q_cond_w=500.0,
        q_evap_w=400.0,
        t_supply_c=24.0,
        t_plate_c=25.0,
        t_return_c=25.5,
        t_batt_c=26.0,
        t_cool_c=25.0,
    )
    commands = [(2200.0, 2000.0), (3200.0, 2400.0), (4200.0, 2800.0)]
    rows = []
    for index, (n_comp, n_pump) in enumerate(commands):
        if index:
            state = step_physics_predictor(
                state,
                n_comp_cmd_rpm=n_comp,
                n_pump_cmd_rpm=n_pump,
                q_gen_w=800.0,
                t_ambient_c=25.0,
                dt_s=5.0,
                artifact=artifact,
            )
        rows.append(
            {
                "Time": index * 5.0,
                "Average temperature": state.t_batt_c,
                "Coolant temperature": state.t_cool_c,
                "Supply pipe coolant temperature": state.t_supply_c,
                "Return pipe coolant temperature": state.t_return_c,
                "P_Shadow_T_Plate_Actual_C": state.t_plate_c,
                "Total current": (800.0 / (0.001 * 52.0)) ** 0.5 * 4.0,
                "Compressor command": n_comp,
                "Pump command": n_pump,
                "Compressor Speed": state.n_comp_eff_rpm,
                "Pump Speed (RPM)": state.n_pump_eff_rpm,
                "Evaporator cooling rate (kW)": state.q_evap_w / 1000.0,
                "Condenser heat rejection rate (kW)": state.q_cond_w / 1000.0,
            }
        )
    return artifact, pd.DataFrame(rows)


class ActualCommandReplayTests(unittest.TestCase):
    def test_exact_future_commands_reproduce_zero_delay_predictor_trajectory(self):
        artifact, frame = exact_trajectory_frame()

        details, summary = evaluate_actual_command_replay(
            frame,
            artifact,
            dt_s=5.0,
            horizons_s=(5.0, 10.0),
            t_ambient_c=25.0,
        )

        self.assertEqual(len(details), 3 * 6)
        self.assertEqual(set(details["origin_index"]), {0, 1})
        self.assertEqual(set(details["target_index"]), {1, 2})
        self.assertTrue((details["abs_error"] < 1e-10).all())
        self.assertTrue((summary["mae"] < 1e-10).all())

    def test_target_state_is_not_used_to_generate_prediction(self):
        artifact, frame = exact_trajectory_frame()
        altered = frame.copy()
        altered.loc[2, "Average temperature"] += 1000.0

        original, _ = evaluate_actual_command_replay(
            frame,
            artifact,
            dt_s=5.0,
            horizons_s=(10.0,),
            t_ambient_c=25.0,
        )
        changed, _ = evaluate_actual_command_replay(
            altered,
            artifact,
            dt_s=5.0,
            horizons_s=(10.0,),
            t_ambient_c=25.0,
        )

        original_batt = original.loc[original["state"] == "T_Batt"].iloc[0]
        changed_batt = changed.loc[changed["state"] == "T_Batt"].iloc[0]
        self.assertEqual(original_batt["prediction"], changed_batt["prediction"])
        self.assertNotEqual(original_batt["actual"], changed_batt["actual"])

    def test_replay_rejects_horizon_longer_than_available_data(self):
        artifact, frame = exact_trajectory_frame()
        with self.assertRaisesRegex(ValueError, "horizon"):
            evaluate_actual_command_replay(
                frame,
                artifact,
                dt_s=5.0,
                horizons_s=(20.0,),
            )


if __name__ == "__main__":
    unittest.main()
