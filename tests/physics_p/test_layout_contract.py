import ast
import unittest
from pathlib import Path

import p_mpc_run_support as support
import run_p_mpc_operational as operational


class PhysicsPLayoutContractTests(unittest.TestCase):
    def test_operational_runner_uses_root_shared_run_support(self):
        self.assertIs(operational.SCENES, support.SCENES)
        self.assertIs(operational.source_csv_for_scene, support.source_csv_for_scene)
        self.assertIs(operational.summarize_run, support.summarize_run)
        self.assertEqual(
            support.PROJECT_ROOT,
            Path(__file__).resolve().parents[2],
        )

    def test_operational_runner_does_not_import_experiment_modules(self):
        source = Path(operational.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.append(node.module)
        self.assertFalse(
            any(name.startswith("experiments") for name in imported_names),
            imported_names,
        )


if __name__ == "__main__":
    unittest.main()
