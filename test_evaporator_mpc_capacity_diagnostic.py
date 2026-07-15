import unittest
from unittest.mock import patch

import run_evaporator_mpc_capacity_diagnostic as diagnostic


class EvaporatorMpcCapacityDiagnosticTest(unittest.TestCase):
    @patch.object(diagnostic, "run_refrigeration_cycle", return_value={"Q_evap": 800.0})
    @patch.object(diagnostic, "staged_fan_speed", return_value=1500.0)
    @patch.object(diagnostic, "pump_model", return_value=(0.25, 10.0))
    @patch.object(diagnostic, "evaluate_capacity", return_value=750.0)
    @patch.object(diagnostic, "load_capacity_calibration")
    def test_default_diagnostic_uses_candidate_b_capacity_model(
        self,
        load_calibration,
        evaluate_capacity,
        _pump_model,
        _staged_fan_speed,
        _run_refrigeration_cycle,
    ):
        calibration = {"model_type": "candidate_b"}
        load_calibration.return_value = calibration

        frame = diagnostic.build_comparison([3000.0], 2000.0, 25.0)

        load_calibration.assert_called_once()
        evaluate_capacity.assert_called_once_with(
            calibration, 3000.0, 2000.0, 25.0
        )
        self.assertEqual(frame.loc[0, "Q_evap_mpc_W"], 750.0)
        self.assertFalse(hasattr(diagnostic, "mpc_q_evap_steady_w"))


if __name__ == "__main__":
    unittest.main()
