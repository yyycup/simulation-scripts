"""Tests for the coupled Reference V2 versus ReducedPack validation tools."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cluster_plant_v2.validation.validate_reduced_pack import (
    CaseSpec,
    ReferenceCoupledPackAdapter,
    aggregate_reference_battery_zones,
    aggregate_reference_plate_zones,
    build_case_specs,
    liters_per_minute_to_mass_flow,
    run_case,
)


class ReducedPackValidationTests(unittest.TestCase):
    def test_formal_case_matrix_is_exactly_the_seven_requested_cases(self) -> None:
        cases = build_case_specs()

        self.assertEqual(len(cases), 7)
        self.assertEqual(
            [(case.current_kind, case.current_A, case.flow_L_min, case.flow_direction)
             for case in cases],
            [
                ("constant", 280.0, 5.0, 1),
                ("constant", 560.0, 5.0, 1),
                ("constant", 1120.0, 5.0, 1),
                ("regd", None, 5.0, 1),
                ("constant", 560.0, 4.0, 1),
                ("constant", 560.0, 6.0, 1),
                ("constant", 560.0, 5.0, -1),
            ],
        )

    def test_spatial_aggregation_and_flow_conversion_match_frozen_contract(self) -> None:
        cells = np.arange(52, dtype=float).reshape(4, 13)
        nodes = np.arange(13, dtype=float)

        battery_zones = aggregate_reference_battery_zones(cells)
        plate_zones = aggregate_reference_plate_zones(nodes)

        self.assertEqual(battery_zones.shape, (4, 3))
        np.testing.assert_allclose(battery_zones[0], [1.5, 6.0, 10.5])
        np.testing.assert_allclose(plate_zones, [1.5, 6.0, 10.5])
        self.assertAlmostEqual(liters_per_minute_to_mass_flow(5.0), 0.08925)

    def test_reference_adapter_uses_current_layer_cell_to_plate_heat(self) -> None:
        adapter = ReferenceCoupledPackAdapter(initial_current_A=560.0)

        result = adapter.step(5.0, 560.0, 293.15, 0.08925, 308.15, 1)

        np.testing.assert_array_equal(result["q_battery_to_plate_nodes"], 0.0)
        self.assertGreater(result["q_gen_total"], 0.0)
        self.assertGreater(result["q_plate_to_fluid_total"], 0.0)
        self.assertGreater(result["coolant_outlet_temperature"], 293.15)
        self.assertLess(abs(result["whole_pack_energy_residual_J"]), 1e-5)

    def test_short_case_produces_timeseries_metrics_energy_and_runtime(self) -> None:
        case = CaseSpec("test_560a", "constant", 560.0, 5.0, 1)
        with tempfile.TemporaryDirectory() as directory:
            result = run_case(
                case,
                duration_s=5.0,
                dt=5.0,
                regd_file=Path("unused.csv"),
                output_dir=Path(directory),
                make_plot=False,
            )

        self.assertEqual(len(result["timeseries"]), 1)
        self.assertTrue(result["summary"]["all_states_finite"])
        self.assertLess(result["energy"]["max_abs_coupling_residual_W"], 1e-12)
        self.assertGreater(result["runtime"]["speedup"], 0.0)


if __name__ == "__main__":
    unittest.main()
