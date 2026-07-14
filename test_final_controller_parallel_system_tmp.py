import tempfile
import unittest
from pathlib import Path

import pandas as pd

from run_final_controller_parallel import build_worker_plan, merge_worker_summaries


class FinalControllerParallelSystemTempTest(unittest.TestCase):
    def test_worker_plan_uses_distinct_case_homes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            plan = build_worker_plan(Path(temp_dir) / "results", Path(temp_dir) / "gekko")

        self.assertEqual([item["case_index"] for item in plan], list(range(12)))
        self.assertEqual(len({item["gekko_temp"] for item in plan}), 12)

    def test_merge_worker_summaries_sorts_by_case_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            later = root / "case_03.csv"
            first = root / "case_00.csv"
            pd.DataFrame([{"case_index": 3, "control": "mpc"}]).to_csv(later, index=False, encoding="utf-8-sig")
            pd.DataFrame([{"case_index": 0, "control": "on-off"}]).to_csv(first, index=False, encoding="utf-8-sig")
            merged = merge_worker_summaries([later, first])

        self.assertEqual(merged["case_index"].tolist(), [0, 3])


if __name__ == "__main__":
    unittest.main()
