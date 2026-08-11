import copy
import unittest

import pandas as pd

from mpc_physics_predictor import (
    DEFAULT_PHYSICS_ARTIFACT,
    DEFAULT_THERMAL_PARAMETERS,
)
from refine_p_thermal_capacities import (
    build_capacity_variants,
    select_capacity_variant,
)


class RefinePThermalCapacitiesTests(unittest.TestCase):
    def test_capacity_variants_change_only_requested_capacity(self):
        base = copy.deepcopy(DEFAULT_PHYSICS_ARTIFACT)
        base["fit"] = {"fit_status": "validated"}
        variants = build_capacity_variants(base)

        self.assertEqual(
            variants["battery_physical"]["thermal"][
                "battery_heat_capacity_j_k"
            ],
            DEFAULT_THERMAL_PARAMETERS["battery_heat_capacity_j_k"],
        )
        self.assertEqual(
            variants["battery_physical"]["thermal"]["plate_tau_s"],
            base["thermal"]["plate_tau_s"],
        )
        self.assertEqual(
            variants["plate_physical"]["thermal"][
                "battery_heat_capacity_j_k"
            ],
            base["thermal"]["battery_heat_capacity_j_k"],
        )
        self.assertNotEqual(
            variants["plate_physical"]["thermal"]["plate_tau_s"],
            base["thermal"]["plate_tau_s"],
        )
        self.assertEqual(
            variants["battery_physical"]["capacity"],
            base["capacity"],
        )
        self.assertEqual(
            variants["battery_physical"]["dynamic"],
            base["dynamic"],
        )

    def test_selection_rejects_variant_that_worsens_any_replay_horizon(self):
        summary = pd.DataFrame(
            [
                {
                    "variant": "base",
                    "fixed_endpoint_mae_c": 0.34,
                    "replay_battery_50s_mae_c": 0.07,
                    "replay_battery_100s_mae_c": 0.12,
                    "replay_battery_300s_mae_c": 0.25,
                },
                {
                    "variant": "battery_physical",
                    "fixed_endpoint_mae_c": 0.17,
                    "replay_battery_50s_mae_c": 0.06,
                    "replay_battery_100s_mae_c": 0.11,
                    "replay_battery_300s_mae_c": 0.24,
                },
                {
                    "variant": "plate_physical",
                    "fixed_endpoint_mae_c": 0.10,
                    "replay_battery_50s_mae_c": 0.05,
                    "replay_battery_100s_mae_c": 0.10,
                    "replay_battery_300s_mae_c": 0.30,
                },
            ]
        )

        self.assertEqual(
            select_capacity_variant(summary),
            "battery_physical",
        )


if __name__ == "__main__":
    unittest.main()
