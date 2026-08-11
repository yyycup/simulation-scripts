import ast
import importlib
import unittest
from pathlib import Path

import p_mpc_run_support as support
import run_p_mpc_operational as operational


LEGACY_EXPERIMENT_FILES = (
    "predictor_identification_data.py",
    "generate_predictor_identification_data.py",
    "fit_mpc_physics_predictor.py",
    "fit_mpc_lpv_predictor.py",
    "mpc_lpv_predictor.py",
    "evaluate_mpc_predictors.py",
    "evaluate_p_identification_rollouts.py",
    "run_p_mpc_short_comparison.py",
    "run_p_mpc_controller_tuning.py",
    "run_p_mpc_local_formal.py",
    "evaluate_dual_p_shadow.py",
    "evaluate_p_frozen_mpc_plan.py",
    "evaluate_p_shadow_actual_replay.py",
    "run_p_model_boundary_validation.py",
    "compare_mpc_predictor_formal_results.py",
    "plot_p_peak_displacement_comparison.py",
    "build_p_compressor_displacement_variant.py",
    "build_p_heat_generation_corrected_artifact.py",
    "calibrate_p_actual_replay.py",
    "refine_p_thermal_capacities.py",
)

EXPERIMENT_MODULES = (
    "experiments.physics_p.identification.predictor_identification_data",
    "experiments.physics_p.identification.generate_predictor_identification_data",
    "experiments.physics_p.identification.fit_mpc_physics_predictor",
    "experiments.physics_p.identification.fit_mpc_lpv_predictor",
    "experiments.physics_p.identification.mpc_lpv_predictor",
    "experiments.physics_p.identification.evaluate_mpc_predictors",
    "experiments.physics_p.identification.evaluate_p_identification_rollouts",
    "experiments.physics_p.tuning.run_p_mpc_short_comparison",
    "experiments.physics_p.tuning.run_p_mpc_controller_tuning",
    "experiments.physics_p.tuning.run_p_mpc_local_formal",
    "experiments.physics_p.evaluation.evaluate_dual_p_shadow",
    "experiments.physics_p.evaluation.evaluate_p_frozen_mpc_plan",
    "experiments.physics_p.evaluation.evaluate_p_shadow_actual_replay",
    "experiments.physics_p.evaluation.run_p_model_boundary_validation",
    "experiments.physics_p.evaluation.compare_mpc_predictor_formal_results",
    "experiments.physics_p.evaluation.plot_p_peak_displacement_comparison",
    "experiments.physics_p.evaluation.plot_p_mpc_local_formal_results",
    "experiments.physics_p.calibration.build_p_compressor_displacement_variant",
    "experiments.physics_p.calibration.build_p_heat_generation_corrected_artifact",
    "experiments.physics_p.calibration.calibrate_p_actual_replay",
    "experiments.physics_p.calibration.refine_p_thermal_capacities",
)

P_TEST_FILES = (
    "test_build_p_compressor_displacement_variant.py",
    "test_build_p_heat_generation_corrected_artifact.py",
    "test_calibrate_p_actual_replay.py",
    "test_compare_mpc_predictor_formal_results.py",
    "test_evaluate_dual_p_shadow.py",
    "test_evaluate_mpc_predictors.py",
    "test_evaluate_p_frozen_mpc_plan.py",
    "test_evaluate_p_shadow_actual_replay.py",
    "test_generate_predictor_identification_data.py",
    "test_mpc_lpv_predictor.py",
    "test_mpc_physics_p_closed_loop.py",
    "test_mpc_physics_predictor.py",
    "test_mpc_physics_shadow.py",
    "test_mpc_predictor_selection.py",
    "test_p_model_boundary_validation.py",
    "test_p_mpc_controller_tuning.py",
    "test_p_mpc_local_formal.py",
    "test_p_mpc_operational.py",
    "test_plot_p_mpc_local_formal_results.py",
    "test_predictor_identification_data.py",
    "test_refine_p_thermal_capacities.py",
)


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

    def test_legacy_root_experiment_files_are_absent(self):
        present = [
            name
            for name in LEGACY_EXPERIMENT_FILES
            if (support.PROJECT_ROOT / name).exists()
        ]
        self.assertEqual(present, [])

    def test_experiment_modules_import_from_physics_p_package(self):
        for name in EXPERIMENT_MODULES:
            with self.subTest(module=name):
                module = importlib.import_module(name)
                self.assertEqual(module.__name__, name)

    def test_all_physics_p_tests_live_in_package(self):
        package_root = support.PROJECT_ROOT / "tests" / "physics_p"
        misplaced = [
            name
            for name in P_TEST_FILES
            if not (package_root / name).is_file()
            or (support.PROJECT_ROOT / name).exists()
        ]
        self.assertEqual(misplaced, [])

    def test_active_docs_route_to_supported_physics_p_commands(self):
        root_readme = (support.PROJECT_ROOT / "README.md").read_text(
            encoding="utf-8"
        )
        project_map = (support.PROJECT_ROOT / "PROJECT_MAP_MIN.md").read_text(
            encoding="utf-8"
        )
        codex_map = (support.PROJECT_ROOT / "CODEX_README_MIN.md").read_text(
            encoding="utf-8"
        )
        experiment_readme = (
            support.PROJECT_ROOT / "experiments" / "physics_p" / "README.md"
        ).read_text(encoding="utf-8")

        self.assertIn("run_p_mpc_operational.py", root_readme)
        self.assertIn("Candidate B", root_readme)
        self.assertIn("model_data/physics_p_operational_v1.json", project_map)
        self.assertIn("experiments/physics_p/", codex_map)
        self.assertIn("python -m experiments.physics_p", experiment_readme)
        self.assertIn("旧命令已退役，不提供根目录 wrapper", experiment_readme)


if __name__ == "__main__":
    unittest.main()
