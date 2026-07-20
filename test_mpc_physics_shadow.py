import copy
import unittest

from mpc_physics_predictor import DEFAULT_PHYSICS_ARTIFACT
from mpc_physics_shadow import (
    PhysicsPShadowPredictor,
    battery_heat_generation_w,
)


class PhysicsPShadowPredictorTests(unittest.TestCase):
    def setUp(self):
        self.predictor = PhysicsPShadowPredictor(
            copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT),
            dt_s=5.0,
            horizons_s=(10.0, 20.0),
        )
        self.observed_state = {
            "n_comp_eff_rpm": 2400.0,
            "n_pump_eff_rpm": 2000.0,
            "q_cond_w": 800.0,
            "q_evap_w": 600.0,
            "t_supply_c": 24.0,
            "t_plate_c": 25.0,
            "t_return_c": 25.5,
            "t_batt_c": 26.0,
            "t_cool_c": 25.0,
        }

    def test_forecast_exports_each_requested_horizon(self):
        record = self.predictor.forecast(
            observed_state=self.observed_state,
            n_comp_cmd_rpm=3000.0,
            n_pump_cmd_rpm=2400.0,
            q_gen_preview_w=(1000.0, 1100.0),
            t_ambient_c=25.0,
        )

        self.assertEqual(record["P_Shadow_Enabled"], 1)
        self.assertEqual(record["P_Shadow_Command_Assumption"], "hold_current")
        self.assertEqual(record["P_Shadow_Load_Assumption"], "provided_preview_hold_last")
        self.assertEqual(record["P_Shadow_Hold_Comp_Command_RPM"], 3000.0)
        self.assertEqual(record["P_Shadow_Hold_Pump_Command_RPM"], 2400.0)
        self.assertGreaterEqual(record["P_Shadow_Prediction_Time_Ms"], 0.0)
        for horizon in (10, 20):
            for state in ("T_Batt", "T_Cool", "T_Plate", "T_Supply", "T_Return"):
                self.assertIn(f"P_Shadow_{state}_Pred_{horizon}s_C", record)
            self.assertIn(f"P_Shadow_Q_Evap_Pred_{horizon}s_W", record)

    def test_short_load_preview_is_extended_with_its_last_value(self):
        short = self.predictor.forecast(
            observed_state=self.observed_state,
            n_comp_cmd_rpm=3000.0,
            n_pump_cmd_rpm=2400.0,
            q_gen_preview_w=(1000.0,),
            t_ambient_c=25.0,
        )
        extended = self.predictor.forecast(
            observed_state=self.observed_state,
            n_comp_cmd_rpm=3000.0,
            n_pump_cmd_rpm=2400.0,
            q_gen_preview_w=(1000.0, 1000.0, 1000.0, 1000.0),
            t_ambient_c=25.0,
        )

        for key in short:
            if "_Pred_" in key and key != "P_Shadow_Prediction_Time_Ms":
                self.assertAlmostEqual(short[key], extended[key])

    def test_horizons_must_be_positive_multiples_of_step(self):
        for horizons in ((0.0,), (12.0,), ()):
            with self.subTest(horizons=horizons):
                with self.assertRaises(ValueError):
                    PhysicsPShadowPredictor(
                        copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT),
                        dt_s=5.0,
                        horizons_s=horizons,
                    )

    def test_battery_heat_formula_matches_identification_data_definition(self):
        self.assertAlmostEqual(battery_heat_generation_w(560.0), 1019.2)
        self.assertAlmostEqual(
            battery_heat_generation_w(-560.0),
            battery_heat_generation_w(560.0),
        )


if __name__ == "__main__":
    unittest.main()
