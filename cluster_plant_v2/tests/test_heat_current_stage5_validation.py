"""Stage 5 — Smoke tests for the system-level final validation script.

The Stage 5 validation script writes per-case CSVs with a stable schema.
These tests verify that schema and that the helper functions produce
well-formed outputs without running the full sweep (which takes ~30 min).

The tests are deliberately short — Stage 5 is about validating the
*output shape*, not reproducing its numerics.
"""

from __future__ import annotations

import csv
import unittest
from pathlib import Path

import numpy as np

from cluster_plant_v2.validation.validate_heat_current_stage5 import (
    _build_case_specs,
    _compressor_cmd_at,
    _direction_at,
    _hc_row,
    _legacy_row,
    _pump_cmd_at,
    _resolve_currents,
)


EXPECTED_FIELDS = {
    "case_id",
    "step",
    "time_s",
    "tank_temperature_k",
    "t_b_avg_k",
    "t_b_max_k",
    "t_p_avg_k",
    "t_p_max_k",
    "t_supply_k",
    "t_return_k",
    "t_evap_out_k",
    "q_gen_w",
    "q_bp_w",
    "q_pf_w",
    "q_evap_cycle_w",
    "q_evap_applied_w",
    "compressor_speed_used_rpm",
    "refrigeration_solver_success",
    "all_states_finite",
}


class Stage5CaseSpecTests(unittest.TestCase):
    def test_nine_cases_present(self):
        specs = _build_case_specs()
        ids = [s.case_id for s in specs]
        self.assertEqual(len(specs), 9)
        expected = [
            "V1_constant_nominal_forward",
            "V2_current_step",
            "V3_compressor_step",
            "V4_pump_step",
            "V5_low_flow",
            "V6_high_flow",
            "V7_reverse_flow",
            "V8_flow_switch",
            "V9_regd",
        ]
        self.assertEqual(ids, expected)

    def test_resolve_currents_constant(self):
        specs = _build_case_specs()
        v1 = next(s for s in specs if s.case_id == "V1_constant_nominal_forward")
        currents = _resolve_currents(v1)
        self.assertEqual(len(currents), 120)
        self.assertTrue((currents == 560.0).all())

    def test_resolve_currents_step(self):
        specs = _build_case_specs()
        v2 = next(s for s in specs if s.case_id == "V2_current_step")
        currents = _resolve_currents(v2)
        # 200 s / 5 s = step 40 (0-indexed) is the cutoff
        self.assertTrue((currents[:40] == 560.0).all())
        self.assertTrue((currents[40:] == 800.0).all())

    def test_resolve_currents_regd_length(self):
        specs = _build_case_specs()
        v9 = next(s for s in specs if s.case_id == "V9_regd")
        currents = _resolve_currents(v9)
        self.assertEqual(len(currents), 120)
        # RegD profile should have nonzero variation
        self.assertGreater(float(currents.std()), 0.0)

    def test_direction_at_v8_switch(self):
        specs = _build_case_specs()
        v8 = next(s for s in specs if s.case_id == "V8_flow_switch")
        self.assertEqual(_direction_at(v8, 0.0), "forward")
        self.assertEqual(_direction_at(v8, 299.0), "forward")
        self.assertEqual(_direction_at(v8, 300.0), "reverse")
        self.assertEqual(_direction_at(v8, 599.0), "reverse")

    def test_direction_at_non_switch(self):
        specs = _build_case_specs()
        v7 = next(s for s in specs if s.case_id == "V7_reverse_flow")
        self.assertEqual(_direction_at(v7, 100.0), "reverse")

    def test_compressor_cmd_at_v3_step(self):
        specs = _build_case_specs()
        v3 = next(s for s in specs if s.case_id == "V3_compressor_step")
        # V3 starts at 2000 rpm, then jumps to 4000 at 200 s
        self.assertEqual(_compressor_cmd_at(v3, 0.0), 2000.0)
        self.assertEqual(_compressor_cmd_at(v3, 199.0), 2000.0)
        self.assertEqual(_compressor_cmd_at(v3, 200.0), 4000.0)
        self.assertEqual(_compressor_cmd_at(v3, 599.0), 4000.0)

    def test_pump_cmd_at_v4_step(self):
        specs = _build_case_specs()
        v4 = next(s for s in specs if s.case_id == "V4_pump_step")
        self.assertEqual(_pump_cmd_at(v4, 0.0), 3000.0)
        self.assertEqual(_pump_cmd_at(v4, 199.0), 3000.0)
        self.assertEqual(_pump_cmd_at(v4, 200.0), 4500.0)
        self.assertEqual(_pump_cmd_at(v4, 599.0), 4500.0)


class Stage5RowSchemaTests(unittest.TestCase):
    """Verify the per-step row schema is stable across legacy and HC."""

    def _fake_out(self) -> dict[str, object]:
        return {
            "tank_temperature_after_k": 295.0,
            "cluster_result": {
                "pack_battery_average_temperatures_k": np.asarray(
                    [298.1, 298.2, 298.3, 298.4, 298.5]
                ),
                "cluster_max_temperature_k": 298.6,
                "pack_plate_average_temperatures_k": np.asarray(
                    [295.1, 295.2, 295.3, 295.4, 295.5]
                ),
                "pack_plate_temperatures_k": np.asarray(
                    [[295.1] * 13 for _ in range(5)]
                ),
                "pack_q_gen_total_w": np.asarray(
                    [800.0] * 5
                ),
                "pack_q_battery_to_plate_total_w": np.asarray(
                    [700.0] * 5
                ),
                "pack_q_plate_to_fluid_total_w": np.asarray(
                    [600.0] * 5
                ),
            },
            "cluster_supply_temperature_k": 293.0,
            "cluster_return_temperature_k": 296.0,
            "evaporator_outlet_temperature_k": 293.5,
            "q_evap_cycle_w": 18000.0,
            "q_evap_applied_w": 1900.0,
            "compressor_speed_rpm": 4000.0,
            "compressor_speed_used_rpm": 4000.0,
            "refrigeration_solver_success": True,
            "all_states_finite": True,
        }

    def test_legacy_row_schema(self):
        spec = _build_case_specs()[0]
        row = _legacy_row(spec, 0, self._fake_out())
        self.assertEqual(set(row.keys()), EXPECTED_FIELDS)
        self.assertEqual(row["step"], 0)
        self.assertEqual(row["time_s"], 5.0)
        self.assertEqual(row["q_gen_w"], 4000.0)
        self.assertEqual(row["q_bp_w"], 3500.0)
        self.assertEqual(row["q_pf_w"], 3000.0)
        self.assertEqual(row["q_evap_cycle_w"], 18000.0)
        self.assertEqual(row["q_evap_applied_w"], 1900.0)

    def test_hc_row_schema(self):
        spec = _build_case_specs()[0]
        row = _hc_row(spec, 0, self._fake_out())
        self.assertEqual(set(row.keys()), EXPECTED_FIELDS)
        self.assertEqual(row["q_gen_w"], 4000.0)
        self.assertEqual(row["q_bp_w"], 3500.0)
        self.assertEqual(row["q_pf_w"], 3000.0)


class Stage5CsvRoundTripTests(unittest.TestCase):
    """Verify the CSV write helper produces header + rows the test consumer
    can re-read. We do not run the full sweep here; we synthesize rows."""

    def test_csv_round_trip(self):
        import tempfile

        from cluster_plant_v2.validation.validate_heat_current_stage5 import (
            _write_csv,
        )
        spec = _build_case_specs()[0]
        out = {
            "tank_temperature_after_k": 295.0,
            "cluster_result": {
                "pack_battery_average_temperatures_k": np.asarray(
                    [298.1] * 5
                ),
                "cluster_max_temperature_k": 298.6,
                "pack_plate_average_temperatures_k": np.asarray(
                    [295.1] * 5
                ),
                "pack_plate_temperatures_k": np.asarray(
                    [[295.1] * 13 for _ in range(5)]
                ),
                "pack_q_gen_total_w": np.asarray([800.0] * 5),
                "pack_q_battery_to_plate_total_w": np.asarray([700.0] * 5),
                "pack_q_plate_to_fluid_total_w": np.asarray([600.0] * 5),
            },
            "cluster_supply_temperature_k": 293.0,
            "cluster_return_temperature_k": 296.0,
            "evaporator_outlet_temperature_k": 293.5,
            "q_evap_cycle_w": 18000.0,
            "q_evap_applied_w": 1900.0,
            "compressor_speed_rpm": 4000.0,
            "refrigeration_solver_success": True,
            "all_states_finite": True,
        }
        rows = [_legacy_row(spec, 0, out)]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            _write_csv(csv_path, rows)
            with csv_path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames
                re_read = list(reader)
            self.assertIsNotNone(fieldnames)
            self.assertEqual(set(fieldnames or []), EXPECTED_FIELDS)
            self.assertEqual(len(re_read), 1)
            self.assertEqual(float(re_read[0]["q_gen_w"]), 4000.0)


if __name__ == "__main__":
    unittest.main()