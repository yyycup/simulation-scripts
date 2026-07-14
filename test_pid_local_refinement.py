import json
import tempfile
import unittest
from pathlib import Path

from run_pid_local_refinement import (
    SCENE_SETTINGS,
    coordinate_candidates,
    save_pid_selection,
    select_best_feasible,
)


class PidLocalRefinementTest(unittest.TestCase):
    def test_frequency_kp_grid_covers_constraint_boundary(self):
        kp_values = SCENE_SETTINGS["freq"]["axis_values"][0]

        self.assertTrue({0.20, 0.21, 0.22}.issubset(set(kp_values)))

    def test_coordinate_candidates_change_only_selected_axis_and_keep_base(self):
        base = (1.0, 0.01, 0.2)
        candidates = coordinate_candidates(base, axis=1, values=[0.0, 0.01, 0.02, 0.02])

        self.assertEqual(candidates, [(1.0, 0.0, 0.2), (1.0, 0.01, 0.2), (1.0, 0.02, 0.2)])

    def test_select_best_feasible_uses_lowest_temperature_score(self):
        records = [
            {"params": (1.0, 0.01, 0.0), "score": 0.30, "feasible": True},
            {"params": (1.2, 0.01, 0.0), "score": 0.20, "feasible": True},
            {"params": (1.4, 0.01, 0.0), "score": 0.10, "feasible": False},
        ]

        self.assertEqual(select_best_feasible(records)["params"], (1.2, 0.01, 0.0))

    def test_save_pid_selection_writes_peak_and_frequency_parameters(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            path = Path(temp_dir) / "PID最终参数.json"
            save_pid_selection(path, peak=(1.2, 0.01, 0.1), freq=(0.2, 0.0, 0.0))
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["peak"], [1.2, 0.01, 0.1])
        self.assertEqual(data["freq"], [0.2, 0.0, 0.0])
        self.assertEqual(data["objective"], "MAE + 0.5*RMSE + 2*OSC")


if __name__ == "__main__":
    unittest.main()
