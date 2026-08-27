from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from single_pack_plant.plant import pack
from single_pack_plant.simulation import config as thermal_batch_config
from single_pack_plant.simulation.case import initialize_thermal_temperatures


class IsolatedSinglePackRuntimeTests(unittest.TestCase):
    def test_runtime_and_hppc_asset_are_owned_by_single_pack_folder(self):
        project_dir = Path(pack.__file__).resolve().parents[1]
        self.assertEqual(thermal_batch_config.WORKSPACE, project_dir)
        self.assertEqual(pack.HPPC_PARAMS_PATH, project_dir / "model_data" / "hppc_params.json")
        self.assertTrue(pack.HPPC_PARAMS_PATH.is_file())

    def test_thermal_initialization_is_finite_and_local(self):
        tank_temperature, plate_temperatures = initialize_thermal_temperatures(13, 25.0)
        self.assertEqual(plate_temperatures.shape, (13,))
        self.assertTrue(np.all(np.isfinite(plate_temperatures)))
        np.testing.assert_allclose(plate_temperatures, tank_temperature, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    unittest.main()
