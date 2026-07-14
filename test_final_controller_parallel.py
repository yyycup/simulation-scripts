import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_final_controller_parallel import build_worker_plan, merge_worker_summaries


class FinalControllerParallelTest(unittest.TestCase):
    def test_worker_plan_assigns_each_case_a_unique_gekko_temp_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            plan = build_worker_plan(Path(temp_dir) / "results", Path(temp_dir) / "gekko")

        self.assertEqual(len(plan), 12)
        self.assertEqual([item["case_index"] for item in plan], list(range(12)))
        self.assertEqual(len({item["gekko_temp"] for item in plan}), 12)
        self.assertTrue(all(item["gekko_temp"].name.startswith("case_") for item in plan))

    def test_merge_worker_summaries_orders_rows_by_case_index(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp_dir:
            root = Path(temp_dir)
            pd.DataFrame([{"case_index": 3, "scene_key": "peak", "control": "mpc"}]).to_csv(
                root / "case_03.csv", index=False, encoding="utf-8-sig"
            )
            pd.DataFrame([{"case_index": 0, "scene_key": "peak", "control": "on-off"}]).to_csv(
                root / "case_00.csv", index=False, encoding="utf-8-sig"
            )

            merged = merge_worker_summaries([root / "case_03.csv", root / "case_00.csv"])

        self.assertEqual(merged["case_index"].tolist(), [0, 3])


if __name__ == "__main__":
    unittest.main()
