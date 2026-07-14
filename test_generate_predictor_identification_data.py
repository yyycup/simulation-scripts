import unittest
from unittest.mock import patch

from predictor_identification_data import REQUIRED_COLUMNS
from generate_predictor_identification_data import (
    build_steady_grid,
    generate_steady_rows,
)


class GeneratePredictorIdentificationDataTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
