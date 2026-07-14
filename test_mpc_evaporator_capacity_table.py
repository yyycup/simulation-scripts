"""Regression checks for the MPC evaporator-capacity lookup data."""

import csv
import inspect
import unittest

import numpy as np

from mpc_flow_direction_strategies import (
    DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH,
    load_evaporator_capacity_table,
    trilinear_capacity_value,
    MPCControllerDual,
)


class EvaporatorCapacityTableTests(unittest.TestCase):
    def setUp(self):
        self.table = load_evaporator_capacity_table(DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH)

    def test_grid_node_is_reproduced_exactly(self):
        q_hx = trilinear_capacity_value(self.table, 2000.0, 1600.0, 20.0, "q_hx_w")
        q_ref = trilinear_capacity_value(self.table, 2000.0, 1600.0, 20.0, "q_ref_max_w")
        self.assertAlmostEqual(q_hx, 900.1224299376285)
        self.assertAlmostEqual(q_ref, 953.2858604446943)

    def test_cell_centre_is_the_eight_node_trilinear_mean(self):
        with DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH.open(newline="", encoding="utf-8-sig") as fp:
            rows = list(csv.DictReader(fp))
        values = [
            float(row["Q_hx_potential_W"])
            for row in rows
            if float(row["N_comp_rpm"]) in (2000.0, 3000.0)
            and float(row["N_pump_rpm"]) in (1600.0, 2400.0)
            and float(row["T_cool_in_C"]) in (20.0, 25.0)
        ]
        expected = float(np.mean(values))
        actual = trilinear_capacity_value(self.table, 2500.0, 2000.0, 22.5, "q_hx_w")
        self.assertEqual(len(values), 8)
        self.assertAlmostEqual(actual, expected)
    def test_controller_uses_bounded_candidate_b_command(self):
        source = inspect.getsource(MPCControllerDual.__init__)
        self.assertIn("load_capacity_calibration", source)
        self.assertIn("evaporator_capacity_calibration", source)
        self.assertIn("q_evap_raw_w", source)
        self.assertIn("self.Q_evap_cmd_w", source)
        self.assertIn("lb=0.0", source)
        self.assertIn("ub=self._capacity_upper_w", source)
        self.assertNotIn("gekko_trilinear_capacity_expr", source)
        self.assertNotIn('Q_evap_cmd = self.m.Intermediate(p["kq"] * N_comp_delay)', source)

    def test_solver_exports_time_aligned_evaporator_predictions(self):
        source = inspect.getsource(MPCControllerDual.solve_step)
        self.assertIn("qevap_cmd_pred_1_w", source)
        self.assertIn("qcond_pred_1_w", source)
        self.assertIn("qevap_pred_1_w", source)

if __name__ == "__main__":
    unittest.main()

