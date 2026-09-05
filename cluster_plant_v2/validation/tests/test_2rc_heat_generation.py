"""Tests for independent reconstruction of PyBaMM 2RC heat generation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cluster_plant_v2 import ReferenceBatteryPack
from cluster_plant_v2.validation.verify_2rc_heat_generation import (
    _extract_step,
    manual_heat_components,
    pack_heat_interface_error,
    summarize_case,
)


class TwoRcHeatGenerationTests(unittest.TestCase):
    def test_actual_solution_records_zero_entropic_change(self) -> None:
        pack = ReferenceBatteryPack()
        pack.step(5.0, 560.0, np.full(13, 298.15), 298.15)

        records = _extract_step(pack, "560A", 5.0)

        self.assertEqual(len(records), 52)
        self.assertTrue(
            all(record["entropic_change_V_per_K"] == 0.0 for record in records)
        )

    def test_manual_formula_matches_pybamm_sign_convention(self) -> None:
        components = manual_heat_components(
            current_A=140.0,
            r0_ohm=0.001,
            eta1_V=-0.02,
            eta2_V=-0.01,
            q_reversible_W=0.0,
        )

        self.assertAlmostEqual(components["Q_R0_W"], 19.6)
        self.assertAlmostEqual(components["Q_RC1_W"], 2.8)
        self.assertAlmostEqual(components["Q_RC2_W"], 1.4)
        self.assertAlmostEqual(components["Q_manual_W"], 23.8)

    def test_pack_heat_interface_compares_all_52_cells(self) -> None:
        public_heat = np.arange(52, dtype=float)
        solution_heat = public_heat.copy()

        self.assertEqual(pack_heat_interface_error(public_heat, solution_heat), 0.0)
        solution_heat[-1] += 0.25
        self.assertEqual(pack_heat_interface_error(public_heat, solution_heat), 0.25)

    def test_case_summary_reports_rc2_energy_fraction_and_errors(self) -> None:
        frame = pd.DataFrame(
            {
                "time_s": [5.0, 10.0],
                "Q_R0_W": [8.0, 8.0],
                "Q_RC1_W": [1.0, 1.0],
                "Q_RC2_W": [1.0, 1.0],
                "Q_reversible_W": [0.0, 0.0],
                "Q_manual_W": [10.0, 10.0],
                "Q_pybamm_W": [10.0, 10.0],
                "abs_error_W": [0.0, 0.0],
                "relative_error": [0.0, 0.0],
                "pack_heat_sum_abs_error_W": [0.0, 0.0],
            }
        )

        summary = summarize_case("560A", frame)

        self.assertEqual(summary["mean_Q_total_W_per_cell"], 10.0)
        self.assertEqual(summary["RC2_contribution_ratio"], 0.1)
        self.assertEqual(summary["max_abs_error_W"], 0.0)


if __name__ == "__main__":
    unittest.main()
