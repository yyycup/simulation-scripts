import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from run_final_controller_parallel import build_worker_plan, merge_worker_summaries, parse_case_indices


class FinalControllerParallelPureTest(unittest.TestCase):
    def test_worker_plan_assigns_twelve_unique_case_homes(self):
        plan = build_worker_plan(Path("/results"), Path("/isolated-gekko"))

        self.assertEqual([item["case_index"] for item in plan], list(range(12)))
        self.assertEqual(len({item["gekko_temp"] for item in plan}), 12)
        self.assertEqual(plan[0]["gekko_temp"], Path("/isolated-gekko/case_00"))

    def test_merge_worker_summaries_sorts_rows_without_filesystem_fixtures(self):
        with patch(
            "run_final_controller_parallel.pd.read_csv",
            side_effect=[
                pd.DataFrame([{"case_index": 3, "control": "mpc"}]),
                pd.DataFrame([{"case_index": 0, "control": "on-off"}]),
            ],
        ):
            merged = merge_worker_summaries([Path("later.csv"), Path("first.csv")])

        self.assertEqual(merged["case_index"].tolist(), [0, 3])

    def test_case_index_parser_accepts_a_subset_for_resuming(self):
        self.assertEqual(parse_case_indices("1, 2,7"), [1, 2, 7])


if __name__ == "__main__":
    unittest.main()
