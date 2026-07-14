import unittest

from plot_mpc_weight_combined_2x3 import PANEL_LAYOUT, load_relative_data


class CombinedWeightTwoByThreePlotTests(unittest.TestCase):
    def test_layout_has_temperature_row_and_energy_row_for_three_weights(self):
        self.assertEqual(
            PANEL_LAYOUT,
            (
                (("temp_change_pct", "cv"), ("temp_change_pct", "compressor"), ("temp_change_pct", "pump")),
                (("energy_change_pct", "cv"), ("energy_change_pct", "compressor"), ("energy_change_pct", "pump")),
            ),
        )
        self.assertEqual(set(load_relative_data()["scan_key"]), {"cv", "compressor", "pump"})


if __name__ == "__main__":
    unittest.main()
