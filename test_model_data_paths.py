import unittest
from pathlib import Path

import mpc_evaporator_capacity_model as evap_model
import mpc_flow_direction_strategies as flow_mpc
import pack


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DATA_ROOT = PROJECT_ROOT / "model_data"


class ModelDataPathTest(unittest.TestCase):
    def test_default_paths_use_tracked_model_data_directory(self):
        self.assertEqual(pack.HPPC_PARAMS_PATH, MODEL_DATA_ROOT / "hppc_params.json")
        self.assertEqual(
            evap_model.DEFAULT_CALIBRATION_PATH,
            MODEL_DATA_ROOT / "mpc_evaporator_capacity_candidate_b.json",
        )
        self.assertEqual(
            flow_mpc.DEFAULT_EVAPORATOR_CAPACITY_TABLE_PATH,
            MODEL_DATA_ROOT / "candidate_b_capacity_limits_grid.csv",
        )
        self.assertEqual(
            flow_mpc.DEFAULT_REDUCED_MODEL_CALIBRATION_PATH,
            MODEL_DATA_ROOT / "mpc_reduced_model_best_theta.json",
        )

    def test_required_model_data_files_exist(self):
        required = (
            MODEL_DATA_ROOT / "hppc_params.json",
            MODEL_DATA_ROOT / "mpc_evaporator_capacity_candidate_b.json",
            MODEL_DATA_ROOT / "candidate_b_capacity_limits_grid.csv",
            MODEL_DATA_ROOT / "mpc_reduced_model_best_theta.json",
        )
        self.assertEqual([path for path in required if not path.is_file()], [])


if __name__ == "__main__":
    unittest.main()
