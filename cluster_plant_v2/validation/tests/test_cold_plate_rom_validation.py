"""Contract tests for the isolated cold-plate ROM validation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cluster_plant_v2.validation.validate_cold_plate_rom import (
    aggregate_node_heat,
    aggregate_node_temperatures,
    build_case_specs,
    compare_one_step,
    liters_per_minute_to_mass_flow,
    run_case,
)


class ColdPlateRomValidationTests(unittest.TestCase):
    def test_liters_per_minute_conversion_is_explicit(self) -> None:
        self.assertAlmostEqual(liters_per_minute_to_mass_flow(4.0), 0.0714)
        self.assertAlmostEqual(liters_per_minute_to_mass_flow(37.0), 0.66045)

    def test_reference_nodes_use_mean_temperature_and_summed_heat(self) -> None:
        nodes = np.arange(13.0)

        np.testing.assert_allclose(aggregate_node_temperatures(nodes), [1.5, 6.0, 10.5])
        np.testing.assert_allclose(aggregate_node_heat(nodes), [6.0, 30.0, 42.0])

    def test_case_matrix_covers_future_and_existing_flow_ranges(self) -> None:
        cases = build_case_specs()

        self.assertEqual(len(cases), 13)
        self.assertEqual(len({case.case_id for case in cases}), 13)
        self.assertEqual({case.flow_L_min for case in cases}, {4.0, 5.0, 6.0, 10.0, 21.0, 37.0})
        nominal = [case for case in cases if case.flow_L_min == 21.0]
        self.assertEqual({case.flow_direction for case in nominal}, {-1, 1})
        self.assertEqual({case.heat_kind for case in nominal}, {"uniform", "forward", "reverse", "dynamic"})

    def test_uniform_one_step_records_expected_rescale_and_closes_energy(self) -> None:
        result = compare_one_step(flow_L_min=21.0, flow_direction=1)

        self.assertGreater(result["Tout_abs_error_C"], 0.1)
        self.assertGreater(result["weighted_plate_avg_abs_error_C"], 0.1)
        self.assertGreater(result["Q_total_relative_error"], 0.1)
        self.assertLess(result["rom_energy_closure_abs_error_W"], 1e-9)

    def test_short_case_writes_required_timeseries_contract(self) -> None:
        case = next(case for case in build_case_specs() if case.case_id == "21lpm_dynamic_reverse")
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(
                case,
                duration_s=10.0,
                dt=5.0,
                output_dir=Path(directory),
                make_plot=False,
            )
            columns = set(result["timeseries"].columns)

            self.assertEqual(len(result["timeseries"]), 2)
            self.assertEqual(len(result["zone_metrics"]), 3)
            self.assertTrue(
                {
                    "Ref_Tout_C",
                    "ROM_Tout_C",
                    "Ref_Tplate_avg_C",
                    "ROM_Tplate_avg_C",
                    "Ref_Tplate_zone2_C",
                    "ROM_Tplate_zone2_C",
                    "Ref_Q_plate_to_fluid_total_W",
                    "ROM_Q_plate_to_fluid_total_W",
                    "battery_side_Q_zone2_W",
                    "Ref_Tfluid_node12_C",
                }.issubset(columns)
            )


if __name__ == "__main__":
    unittest.main()
