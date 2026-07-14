import unittest

from run_evaporator_mpc_capacity_diagnostic import mpc_q_evap_steady_w


class EvaporatorMpcCapacityDiagnosticTest(unittest.TestCase):
    def test_linear_mpc_steady_capacity_uses_kq_in_watts_per_rpm(self):
        self.assertEqual(mpc_q_evap_steady_w(2000.0, 1.2), 2400.0)
        self.assertEqual(mpc_q_evap_steady_w(6000.0, 1.2), 7200.0)


if __name__ == "__main__":
    unittest.main()
