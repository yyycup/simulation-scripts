import inspect
import unittest

import run_mpc_sensitivity_60 as sensitivity


class MPCSensitivityDonePrefixTest(unittest.TestCase):
    def test_done_prefix_has_generic_fallback_for_new_sweep_types(self):
        source = inspect.getsource(sensitivity.main)
        fallback = 'done_prefix = f"DONE [{index}/{total}] case={case} {args.sweep_type}={label} "'
        completion_block = source.split("rows.append(row)", 1)[1]

        self.assertIn(fallback, completion_block)
        self.assertLess(
            completion_block.index(fallback),
            completion_block.index('if args.sweep_type == "adaptive_dmax_w_energy_comp"'),
        )


if __name__ == "__main__":
    unittest.main()
