import unittest

import numpy as np

from single_pack_plant.controllers.dompc.adapter import forecast_pack_heat_generation_w


class _SyntheticPack:
    """Minimal read-only Pack surface for heat-forecast unit tests."""

    rows = 4
    cols = 13

    def __init__(self, soc: float) -> None:
        self.socs = np.full(self.rows * self.cols, soc)
        self.temps = np.full(self.rows * self.cols, 298.15)
        self.config = {"capacity": 280.0}
        self._hppc_data = {"soc": np.array([0.0, 1.0]), "temp": np.array([273.15, 323.15])}
        self.interp_R_dis = lambda points: 1.0e-3 + (1.0 - points[:, 0]) * 1.0e-3
        self.interp_R_chg = self.interp_R_dis


class DoMPCHeatForecastTests(unittest.TestCase):
    def test_low_soc_increases_same_current_heat_forecast(self):
        preview = np.full(60, 560.0)
        q_high_soc = forecast_pack_heat_generation_w(_SyntheticPack(0.95), preview, 5.0)
        q_low_soc = forecast_pack_heat_generation_w(_SyntheticPack(0.10), preview, 5.0)
        self.assertGreater(q_low_soc[0], q_high_soc[0])

    def test_future_discharge_advances_soc_and_heat(self):
        preview = np.full(60, 560.0)
        q_gen = forecast_pack_heat_generation_w(_SyntheticPack(0.95), preview, 5.0)
        self.assertGreater(q_gen[-1], q_gen[0])


if __name__ == "__main__":
    unittest.main()
