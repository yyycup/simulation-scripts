from io import StringIO
import unittest
from unittest.mock import patch

from predictor_identification_data import REQUIRED_COLUMNS
from generate_predictor_identification_data import (
    _parse_args,
    build_steady_grid,
    generate_steady_rows,
)


class GeneratePredictorIdentificationDataTest(unittest.TestCase):
    @staticmethod
    def _valid_cycle_result():
        return {
            "Q_evap": 900.0,
            "Q_cond": 1200.0,
            "Q_hx_potential": 900.0,
            "Q_ref_max": 1000.0,
            "W_comp": 100.0,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }

    def test_full_grid_has_1200_unique_points(self):
        grid = build_steady_grid(mode="full")

        self.assertEqual(len(grid), 12 * 5 * 4 * 5)
        self.assertEqual(len(set(grid)), len(grid))

    def test_smoke_grid_preserves_the_compressor_boundary(self):
        self.assertEqual(
            build_steady_grid(mode="smoke"),
            [
                (1999.0, 1600.0, 25.0, 35.0),
                (2000.0, 1600.0, 25.0, 35.0),
            ],
        )

    def test_invalid_grid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_steady_grid(mode="tiny")

    def test_empty_points_are_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            generate_steady_rows([])

    def test_completely_duplicate_points_are_rejected(self):
        point = (2000.0, 1600.0, 25.0, 35.0)

        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            generate_steady_rows([point, point])

    def test_scenario_id_formatting_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scenario_id.*collision"):
            generate_steady_rows(
                [
                    (2000.0, 1600.0, 25.0, 35.0),
                    (2000.0000001, 1600.0, 25.0, 35.0),
                ]
            )

    def test_cli_rejects_all_dataset_until_dynamic_generation_exists(self):
        with patch(
            "sys.argv",
            ["generate_predictor_identification_data.py", "--dataset", "all"],
        ), patch("sys.stderr", new_callable=StringIO):
            with self.assertRaises(SystemExit):
                _parse_args()

    def test_configuration_hash_tracks_configuration_and_plant_source(self):
        from generate_predictor_identification_data import build_configuration_hash

        configuration = {"evap_ua_factor": 2.0, "n_comp_min_rpm": 2000.0}
        first = build_configuration_hash(configuration, "source-a")

        self.assertEqual(
            first,
            build_configuration_hash(dict(configuration), "source-a"),
        )
        self.assertNotEqual(
            first,
            build_configuration_hash(
                {**configuration, "evap_ua_factor": 2.1}, "source-a"
            ),
        )
        self.assertNotEqual(
            first,
            build_configuration_hash(configuration, "source-b"),
        )

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.pump_model")
    def test_active_point_rejects_missing_or_nonfinite_critical_cycle_outputs(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.return_value = 800.0
        critical_keys = (
            "Q_evap",
            "Q_cond",
            "Q_hx_potential",
            "Q_ref_max",
            "W_comp",
            "T_evap_sat",
            "T_cond_sat",
        )
        invalid_values = (float("nan"), float("inf"), "not-a-number")

        for key in critical_keys:
            for value in (None, *invalid_values):
                with self.subTest(key=key, value=value):
                    result = self._valid_cycle_result()
                    if value is None:
                        result.pop(key)
                        expected_value = "missing"
                    else:
                        result[key] = value
                        expected_value = str(value)
                    run_cycle.return_value = result

                    with self.assertRaises(ValueError) as raised:
                        generate_steady_rows(
                            [(2000.0, 1600.0, 25.0, 35.0)]
                        )

                    message = str(raised.exception)
                    self.assertIn("steady_nc2000_np1600_tc25_ta35", message)
                    self.assertIn(key, message)
                    self.assertIn(expected_value, message)

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.pump_model")
    def test_off_point_allows_missing_capacity_limits_only(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.return_value = 0.0
        result = self._valid_cycle_result()
        result.update({"Q_evap": 0.0, "Q_cond": 0.0, "W_comp": 0.0})
        result.pop("Q_hx_potential")
        result.pop("Q_ref_max")
        run_cycle.return_value = result

        row = generate_steady_rows(
            [(1999.0, 1600.0, 25.0, 35.0)]
        )[0]

        self.assertEqual(row["q_hx_potential_w"], 0.0)
        self.assertEqual(row["q_ref_max_w"], 0.0)

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.pump_model")
    def test_steady_row_preserves_the_1999_2000_boundary(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.side_effect = lambda n_comp: (
            0.0 if n_comp < 2000.0 else 800.0
        )
        run_cycle.side_effect = lambda n_comp, *_args, **_kwargs: {
            "Q_evap": 0.0 if n_comp < 2000.0 else 900.0,
            "Q_cond": 0.0 if n_comp < 2000.0 else 1200.0,
            "Q_hx_potential": 900.0,
            "Q_ref_max": 1000.0,
            "W_comp": 100.0,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }

        rows = generate_steady_rows(
            [
                (1999.0, 1600.0, 25.0, 35.0),
                (2000.0, 1600.0, 25.0, 35.0),
            ]
        )

        self.assertEqual([row["q_evap_ss_w"] for row in rows], [0.0, 900.0])
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            [
                "steady_nc1999_np1600_tc25_ta35",
                "steady_nc2000_np1600_tc25_ta35",
            ],
        )
        self.assertEqual(run_cycle.call_count, 2)
        self.assertEqual(
            run_cycle.call_args_list[1].args,
            (2000.0, 800.0, 298.15, 0.25, 308.15),
        )
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(rows[0]))
        self.assertEqual(rows[0]["limit_type"], "compressor_off")
        self.assertEqual(rows[1]["limit_type"], "heat_exchanger_limit")
        self.assertEqual(rows[0]["w_fan_w"], 0.0)
        self.assertEqual(rows[0]["t_cool_out_c"], 25.0)
        self.assertEqual(rows[0]["configuration_hash"], rows[1]["configuration_hash"])
        self.assertRegex(rows[0]["plant_source_hash"], r"^[0-9a-f]{64}$")
        self.assertTrue(rows[0]["plant_model_version"])


if __name__ == "__main__":
    unittest.main()
