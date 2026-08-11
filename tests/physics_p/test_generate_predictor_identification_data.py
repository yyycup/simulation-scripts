from dataclasses import replace
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

import experiments.physics_p.identification.generate_predictor_identification_data as generator_module
from experiments.physics_p.identification.predictor_identification_data import (
    REQUIRED_COLUMNS,
    assign_scenario_splits,
)
from experiments.physics_p.identification.generate_predictor_identification_data import (
    DynamicScenarioSpec,
    _parse_args,
    assign_dynamic_scenario_splits,
    build_dynamic_configuration_hash,
    build_dynamic_scenarios,
    build_dynamic_source_hash,
    build_steady_grid,
    generate_dynamic_rows,
    generate_steady_rows,
    hash_file_content,
    ramp_limited_multilevel_sequence,
    run_dynamic_scenario,
)


class GeneratePredictorIdentificationDataTest(unittest.TestCase):
    @staticmethod
    def _valid_hppc_fixture():
        data = {
            "soc": [0.2, 0.8],
            "temp": [20.0, 30.0],
            "ocv": [3.4, 3.8],
        }
        for key in (
            "r0_dis",
            "r0_chg",
            "r1_dis",
            "r1_chg",
            "c1_dis",
            "c1_chg",
            "r2_dis",
            "r2_chg",
            "c2_dis",
            "c2_chg",
        ):
            data[key] = [[1.0, 2.0], [3.0, 4.0]]
        return data

    @staticmethod
    def _valid_cycle_result():
        return {
            "Q_evap": 900.0,
            "Q_cond": 1200.0,
            "Q_hx_potential": 900.0,
            "Q_ref_max": 1000.0,
            "W_comp": 100.0,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }

    def test_full_grid_has_expected_axes_and_1800_unique_points(self):
        grid = build_steady_grid(mode="full")

        self.assertEqual(
            tuple(sorted({point[0] for point in grid})),
            (1000.0, 1400.0, 1800.0, 1900.0, 1950.0, 1999.0, 2000.0,
             2200.0, 3000.0, 4000.0, 5000.0, 6000.0),
        )
        self.assertEqual(
            tuple(sorted({point[1] for point in grid})),
            (1600.0, 2400.0, 3200.0, 4000.0, 4800.0),
        )
        self.assertEqual(
            tuple(sorted({point[2] for point in grid})),
            (15.0, 17.5, 20.0, 25.0, 30.0, 35.0),
        )
        self.assertEqual(
            tuple(sorted({point[3] for point in grid})),
            (20.0, 25.0, 30.0, 35.0, 40.0),
        )
        self.assertEqual(len(grid), 12 * 5 * 6 * 5)
        self.assertEqual(len(set(grid)), len(grid))

    def test_smoke_grid_preserves_the_compressor_boundary(self):
        self.assertEqual(
            build_steady_grid(mode="smoke"),
            [
                (1999.0, 1600.0, 25.0, 35.0),
                (2000.0, 1600.0, 25.0, 35.0),
            ],
        )

    def test_invalid_grid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            build_steady_grid(mode="tiny")

    def test_empty_points_are_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            generate_steady_rows([])

    def test_completely_duplicate_points_are_rejected(self):
        point = (2000.0, 1600.0, 25.0, 35.0)

        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            generate_steady_rows([point, point])

    def test_scenario_id_formatting_collisions_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "scenario_id.*collision"):
            generate_steady_rows(
                [
                    (2000.0, 1600.0, 25.0, 35.0),
                    (2000.0000001, 1600.0, 25.0, 35.0),
                ]
            )

    def test_cli_accepts_all_dataset(self):
        with patch(
            "sys.argv",
            ["experiments.physics_p.identification.generate_predictor_identification_data", "--dataset", "all"],
        ), patch("sys.stderr", new_callable=StringIO):
            self.assertEqual(_parse_args().dataset, "all")

    @staticmethod
    def _cli_rows(count, dataset_kind):
        rows = []
        for index in range(count):
            row = {column: 1.0 for column in REQUIRED_COLUMNS}
            row.update(
                {
                    "scenario_id": f"{dataset_kind}_{index // 20}",
                    "split": "train",
                    "flow_direction": 1,
                    "dataset_kind": dataset_kind,
                }
            )
            rows.append(row)
        return rows

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_dynamic_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_dynamic_scenarios")
    def test_cli_dynamic_writes_only_index_free_utf8_dynamic_csv(
        self, build_scenarios, generate_rows
    ):
        build_scenarios.return_value = [object()]
        generate_rows.return_value = self._cli_rows(80, "dynamic")
        with TemporaryDirectory() as tmp, patch(
            "sys.argv",
            [
                "experiments.physics_p.identification.generate_predictor_identification_data",
                "--dataset",
                "dynamic",
                "--mode",
                "smoke",
                "--output-root",
                tmp,
            ],
        ):
            from experiments.physics_p.identification.generate_predictor_identification_data import main

            main()
            output_root = Path(tmp)
            paths = sorted(path.name for path in output_root.glob("*.csv"))
            frame = pd.read_csv(output_root / "dynamic_smoke.csv", encoding="utf-8")

        self.assertEqual(paths, ["dynamic_smoke.csv"])
        self.assertEqual(len(frame), 80)
        self.assertFalse(any(column.startswith("Unnamed") for column in frame))
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(frame.columns))

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_dynamic_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_dynamic_scenarios")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_steady_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_steady_grid")
    def test_cli_all_writes_index_free_utf8_steady_and_dynamic_csvs(
        self,
        build_grid,
        generate_steady,
        build_scenarios,
        generate_dynamic,
    ):
        build_grid.return_value = [object()]
        build_scenarios.return_value = [object()]
        generate_steady.return_value = self._cli_rows(2, "steady")
        generate_dynamic.return_value = self._cli_rows(80, "dynamic")
        with TemporaryDirectory() as tmp, patch(
            "sys.argv",
            [
                "experiments.physics_p.identification.generate_predictor_identification_data",
                "--dataset",
                "all",
                "--mode",
                "smoke",
                "--output-root",
                tmp,
            ],
        ):
            from experiments.physics_p.identification.generate_predictor_identification_data import main

            main()
            output_root = Path(tmp)
            paths = sorted(path.name for path in output_root.glob("*.csv"))
            steady = pd.read_csv(output_root / "steady_smoke.csv", encoding="utf-8")
            dynamic = pd.read_csv(output_root / "dynamic_smoke.csv", encoding="utf-8")

        self.assertEqual(paths, ["dynamic_smoke.csv", "steady_smoke.csv"])
        self.assertEqual((len(steady), len(dynamic)), (2, 80))
        for frame in (steady, dynamic):
            self.assertFalse(any(column.startswith("Unnamed") for column in frame))
            self.assertTrue(set(REQUIRED_COLUMNS).issubset(frame.columns))

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_dynamic_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_dynamic_scenarios")
    def test_cli_refuses_existing_target_before_generation_without_overwrite(
        self, build_scenarios, generate_rows
    ):
        with TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            target = output_root / "dynamic_smoke.csv"
            target.write_text("old-dynamic", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "experiments.physics_p.identification.generate_predictor_identification_data",
                    "--dataset",
                    "dynamic",
                    "--mode",
                    "smoke",
                    "--output-root",
                    tmp,
                ],
            ):
                from experiments.physics_p.identification.generate_predictor_identification_data import main

                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    main()

            self.assertEqual(target.read_text(encoding="utf-8"), "old-dynamic")
            build_scenarios.assert_not_called()
            generate_rows.assert_not_called()

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_dynamic_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_dynamic_scenarios")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_steady_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_steady_grid")
    def test_cli_all_generation_failure_keeps_both_existing_targets_unchanged(
        self,
        build_grid,
        generate_steady,
        build_scenarios,
        generate_dynamic,
    ):
        build_grid.return_value = [object()]
        build_scenarios.return_value = [object()]
        generate_steady.return_value = self._cli_rows(2, "steady")
        generate_dynamic.side_effect = RuntimeError("dynamic failed")
        with TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            steady_path = output_root / "steady_smoke.csv"
            dynamic_path = output_root / "dynamic_smoke.csv"
            steady_path.write_text("old-steady", encoding="utf-8")
            dynamic_path.write_text("old-dynamic", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "experiments.physics_p.identification.generate_predictor_identification_data",
                    "--dataset",
                    "all",
                    "--mode",
                    "smoke",
                    "--output-root",
                    tmp,
                    "--overwrite",
                ],
            ):
                from experiments.physics_p.identification.generate_predictor_identification_data import main

                with self.assertRaisesRegex(RuntimeError, "dynamic failed"):
                    main()

            self.assertEqual(steady_path.read_text(encoding="utf-8"), "old-steady")
            self.assertEqual(dynamic_path.read_text(encoding="utf-8"), "old-dynamic")
            self.assertEqual(
                sorted(path.name for path in output_root.iterdir()),
                ["dynamic_smoke.csv", "steady_smoke.csv"],
            )

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_dynamic_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_dynamic_scenarios")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_steady_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_steady_grid")
    def test_cli_overwrite_atomically_updates_both_datasets_with_seed(
        self,
        build_grid,
        generate_steady,
        build_scenarios,
        generate_dynamic,
    ):
        build_grid.return_value = [object()]
        build_scenarios.return_value = [object()]
        generate_steady.return_value = self._cli_rows(2, "steady")
        generate_dynamic.return_value = self._cli_rows(80, "dynamic")
        with TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            (output_root / "steady_smoke.csv").write_text("old", encoding="utf-8")
            (output_root / "dynamic_smoke.csv").write_text("old", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "experiments.physics_p.identification.generate_predictor_identification_data",
                    "--dataset",
                    "all",
                    "--mode",
                    "smoke",
                    "--seed",
                    "123",
                    "--output-root",
                    tmp,
                    "--overwrite",
                ],
            ):
                from experiments.physics_p.identification.generate_predictor_identification_data import main

                main()
            steady = pd.read_csv(output_root / "steady_smoke.csv", encoding="utf-8")
            dynamic = pd.read_csv(output_root / "dynamic_smoke.csv", encoding="utf-8")
            names = sorted(path.name for path in output_root.iterdir())

        self.assertEqual((len(steady), len(dynamic)), (2, 80))
        self.assertEqual(set(steady["dataset_seed"]), {123})
        self.assertEqual(set(dynamic["dataset_seed"]), {123})
        self.assertEqual(names, ["dynamic_smoke.csv", "steady_smoke.csv"])

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_dynamic_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_dynamic_scenarios")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.generate_steady_rows")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.build_steady_grid")
    def test_cli_publish_failure_restores_both_backups_without_temp_files(
        self,
        build_grid,
        generate_steady,
        build_scenarios,
        generate_dynamic,
    ):
        build_grid.return_value = [object()]
        build_scenarios.return_value = [object()]
        generate_steady.return_value = self._cli_rows(2, "steady")
        generate_dynamic.return_value = self._cli_rows(80, "dynamic")
        real_replace = os.replace

        def fail_second_publish(source, target):
            source_path = Path(source)
            target_path = Path(target)
            if ".tmp" in source_path.name and target_path.name == "dynamic_smoke.csv":
                raise OSError("second publish failed")
            return real_replace(source, target)

        with TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            steady_path = output_root / "steady_smoke.csv"
            dynamic_path = output_root / "dynamic_smoke.csv"
            steady_path.write_text("old-steady", encoding="utf-8")
            dynamic_path.write_text("old-dynamic", encoding="utf-8")
            with patch(
                "sys.argv",
                [
                    "experiments.physics_p.identification.generate_predictor_identification_data",
                    "--dataset",
                    "all",
                    "--mode",
                    "smoke",
                    "--output-root",
                    tmp,
                    "--overwrite",
                ],
            ), patch(
                "experiments.physics_p.identification.generate_predictor_identification_data.os.replace",
                side_effect=fail_second_publish,
            ):
                from experiments.physics_p.identification.generate_predictor_identification_data import main

                with self.assertRaisesRegex(OSError, "second publish failed"):
                    main()

            self.assertEqual(steady_path.read_text(encoding="utf-8"), "old-steady")
            self.assertEqual(dynamic_path.read_text(encoding="utf-8"), "old-dynamic")
            self.assertEqual(
                sorted(path.name for path in output_root.iterdir()),
                ["dynamic_smoke.csv", "steady_smoke.csv"],
            )

    def test_no_overwrite_publish_preserves_concurrent_target_and_rolls_back_group(self):
        from experiments.physics_p.identification.generate_predictor_identification_data import _atomic_publish_frames

        frames = {
            "steady": pd.DataFrame({"value": [1]}),
            "dynamic": pd.DataFrame({"value": [2]}),
        }
        real_link = os.link
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = {
                "steady": root / "steady_smoke.csv",
                "dynamic": root / "dynamic_smoke.csv",
            }

            def create_concurrent_dynamic_before_link(source, target):
                target = Path(target)
                if target == targets["dynamic"]:
                    target.write_text("concurrent-dynamic", encoding="utf-8")
                return real_link(source, target)

            with patch(
                "experiments.physics_p.identification.generate_predictor_identification_data.os.link",
                side_effect=create_concurrent_dynamic_before_link,
            ):
                with self.assertRaisesRegex(FileExistsError, "dynamic_smoke.csv"):
                    _atomic_publish_frames(frames, targets, overwrite=False)

            self.assertFalse(targets["steady"].exists())
            self.assertEqual(
                targets["dynamic"].read_text(encoding="utf-8"),
                "concurrent-dynamic",
            )
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["dynamic_smoke.csv"],
            )

    def test_rollback_keeps_failed_backup_and_restores_other_targets(self):
        from experiments.physics_p.identification.generate_predictor_identification_data import _atomic_publish_frames

        frames = {
            "steady": pd.DataFrame({"value": [1]}),
            "dynamic": pd.DataFrame({"value": [2]}),
        }
        real_replace = os.replace
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = {
                "steady": root / "steady_smoke.csv",
                "dynamic": root / "dynamic_smoke.csv",
            }
            targets["steady"].write_text("old-steady", encoding="utf-8")
            targets["dynamic"].write_text("old-dynamic", encoding="utf-8")

            def fail_publish_and_one_restore(source, target):
                source = Path(source)
                target = Path(target)
                if source.name.endswith(".tmp") and target == targets["dynamic"]:
                    raise OSError("second publish failed")
                if source.name.endswith(".bak") and target == targets["steady"]:
                    raise OSError("steady restore failed")
                return real_replace(source, target)

            with patch(
                "experiments.physics_p.identification.generate_predictor_identification_data.os.replace",
                side_effect=fail_publish_and_one_restore,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "second publish failed.*steady restore failed",
                ) as caught:
                    _atomic_publish_frames(frames, targets, overwrite=True)

            self.assertIsInstance(caught.exception.__cause__, OSError)
            self.assertIn("second publish failed", str(caught.exception.__cause__))
            self.assertFalse(targets["steady"].exists())
            self.assertEqual(
                targets["dynamic"].read_text(encoding="utf-8"), "old-dynamic"
            )
            steady_backups = list(root.glob(".steady_smoke.csv.*.bak"))
            self.assertEqual(len(steady_backups), 1)
            self.assertEqual(
                steady_backups[0].read_text(encoding="utf-8"), "old-steady"
            )
            self.assertFalse(any(root.glob("*.tmp")))
            self.assertFalse(any(root.glob(".dynamic_smoke.csv.*.bak")))

    def test_rollback_continues_after_published_target_unlink_failure(self):
        from experiments.physics_p.identification.generate_predictor_identification_data import _atomic_publish_frames

        frames = {
            "steady": pd.DataFrame({"value": [1]}),
            "dynamic": pd.DataFrame({"value": [2]}),
        }
        real_replace = os.replace
        real_unlink = Path.unlink
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            targets = {
                "steady": root / "steady_smoke.csv",
                "dynamic": root / "dynamic_smoke.csv",
            }
            targets["steady"].write_text("old-steady", encoding="utf-8")
            targets["dynamic"].write_text("old-dynamic", encoding="utf-8")

            def fail_second_publish(source, target):
                source = Path(source)
                target = Path(target)
                if source.name.endswith(".tmp") and target == targets["dynamic"]:
                    raise OSError("second publish failed")
                return real_replace(source, target)

            unlink_failed = False

            def fail_first_published_unlink(path, *args, **kwargs):
                nonlocal unlink_failed
                if Path(path) == targets["steady"] and not unlink_failed:
                    unlink_failed = True
                    raise OSError("steady unlink failed")
                return real_unlink(path, *args, **kwargs)

            with patch(
                "experiments.physics_p.identification.generate_predictor_identification_data.os.replace",
                side_effect=fail_second_publish,
            ), patch.object(
                Path,
                "unlink",
                autospec=True,
                side_effect=fail_first_published_unlink,
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "second publish failed.*steady unlink failed",
                ):
                    _atomic_publish_frames(frames, targets, overwrite=True)

            self.assertTrue(unlink_failed)
            self.assertEqual(
                targets["steady"].read_text(encoding="utf-8"), "old-steady"
            )
            self.assertEqual(
                targets["dynamic"].read_text(encoding="utf-8"), "old-dynamic"
            )
            self.assertFalse(any(root.glob("*.tmp")))
            self.assertFalse(any(root.glob("*.bak")))

    def test_windows_runtime_path_is_prepared_before_pack_import(self):
        source = Path(generator_module.__file__).read_text(encoding="utf-8")

        self.assertLess(
            source.index("ensure_env_library_bin_on_path()"),
            source.index("from pack import BatteryPack"),
        )

    def test_configuration_hash_tracks_configuration_and_plant_source(self):
        from experiments.physics_p.identification.generate_predictor_identification_data import build_configuration_hash

        configuration = {"evap_ua_factor": 2.0, "n_comp_min_rpm": 2000.0}
        first = build_configuration_hash(configuration, "source-a")

        self.assertEqual(
            first,
            build_configuration_hash(dict(configuration), "source-a"),
        )
        self.assertNotEqual(
            first,
            build_configuration_hash(
                {**configuration, "evap_ua_factor": 2.1}, "source-a"
            ),
        )
        self.assertNotEqual(
            first,
            build_configuration_hash(configuration, "source-b"),
        )

    def test_dynamic_hashes_track_plant_pack_and_hppc_content(self):
        configuration = {"n_comp_min_rpm": 2000.0}
        first_source = build_dynamic_source_hash("thermal", "pack-a", "hppc-a")
        first_config = build_dynamic_configuration_hash(
            configuration, "thermal", "pack-a", "hppc-a"
        )

        self.assertEqual(
            first_source,
            build_dynamic_source_hash("thermal", "pack-a", "hppc-a"),
        )
        self.assertEqual(
            first_config,
            build_dynamic_configuration_hash(
                dict(configuration), "thermal", "pack-a", "hppc-a"
            ),
        )
        self.assertNotEqual(
            first_source,
            build_dynamic_source_hash("thermal", "pack-b", "hppc-a"),
        )
        self.assertNotEqual(
            first_config,
            build_dynamic_configuration_hash(
                configuration, "thermal", "pack-b", "hppc-a"
            ),
        )
        self.assertNotEqual(
            first_config,
            build_dynamic_configuration_hash(
                configuration, "thermal", "pack-a", "hppc-b"
            ),
        )

    def test_file_content_hash_reports_read_failure_clearly(self):
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-hppc.json"
            with self.assertRaisesRegex(RuntimeError, "read.*HPPC"):
                hash_file_content(missing, "HPPC parameters")

    def test_hppc_parameter_file_accepts_valid_minimal_fixture(self):
        from experiments.physics_p.identification.generate_predictor_identification_data import (
            validate_hppc_parameter_file,
        )

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hppc.json"
            path.write_text(
                json.dumps(self._valid_hppc_fixture()), encoding="utf-8"
            )

            validate_hppc_parameter_file(path)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_invalid_hppc_files_fail_before_battery_pack_construction(self, battery_pack):
        invalid_cases = []
        missing = self._valid_hppc_fixture()
        del missing["c2_chg"]
        invalid_cases.append(("malformed", "{", "JSON"))
        invalid_cases.append(("missing", json.dumps(missing), "c2_chg"))
        wrong_shape = self._valid_hppc_fixture()
        wrong_shape["r0_dis"] = [1.0, 2.0, 3.0]
        invalid_cases.append(("shape", json.dumps(wrong_shape), "r0_dis.*shape"))
        nonfinite = self._valid_hppc_fixture()
        nonfinite["r1_chg"][0][0] = float("nan")
        invalid_cases.append(("nan", json.dumps(nonfinite), "JSON.*NaN"))

        with TemporaryDirectory() as tmp:
            for name, content, message in invalid_cases:
                with self.subTest(name=name):
                    path = Path(tmp) / f"{name}.json"
                    path.write_text(content, encoding="utf-8")
                    with patch(
                        "experiments.physics_p.identification.generate_predictor_identification_data.pack_module.HPPC_PARAMS_PATH",
                        path,
                    ):
                        with self.assertRaisesRegex(ValueError, message) as caught:
                            run_dynamic_scenario(
                                DynamicRolloutTest._spec(), split="train"
                            )
                    self.assertIn(str(path), str(caught.exception))
                    battery_pack.assert_not_called()

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model", return_value=(0.23, 23.0))
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed", return_value=777.0)
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_hppc_numeric_strings_and_booleans_fail_before_battery_pack(
        self,
        battery_pack,
        initialize_state,
        simulate_step,
        _staged_fan,
        _pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = DynamicRolloutTest.FakePack
        initialize_state.return_value = {"token": "initial"}
        simulate_step.return_value = DynamicRolloutTest._thermal_result()
        run_cycle.return_value = DynamicRolloutTest._cycle_result()
        invalid_values = (
            ("soc", "0.1"),
            ("temp", "20.0"),
            ("ocv", "3.4"),
            ("r0_dis", "0.01"),
            ("soc", True),
            ("temp", False),
            ("ocv", True),
            ("r0_dis", False),
        )

        with TemporaryDirectory() as tmp:
            for index, (key, value) in enumerate(invalid_values):
                with self.subTest(key=key, value=value):
                    data = json.loads(json.dumps(self._valid_hppc_fixture()))
                    if key in ("soc", "temp", "ocv"):
                        data[key][0] = value
                    else:
                        data[key][0][0] = value
                    path = Path(tmp) / f"invalid_numeric_{index}.json"
                    path.write_text(json.dumps(data), encoding="utf-8")
                    with patch(
                        "experiments.physics_p.identification.generate_predictor_identification_data.pack_module.HPPC_PARAMS_PATH",
                        path,
                    ):
                        with self.assertRaisesRegex(
                            ValueError,
                            rf"key={key}.*JSON number",
                        ) as caught:
                            run_dynamic_scenario(
                                DynamicRolloutTest._spec(), split="train"
                            )
                    self.assertIn(str(path), str(caught.exception))

        battery_pack.assert_not_called()

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model")
    def test_active_point_rejects_missing_or_nonfinite_critical_cycle_outputs(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.return_value = 800.0
        critical_keys = (
            "Q_evap",
            "Q_cond",
            "Q_hx_potential",
            "Q_ref_max",
            "W_comp",
            "T_evap_sat",
            "T_cond_sat",
        )
        invalid_values = (float("nan"), float("inf"), "not-a-number")

        for key in critical_keys:
            for value in (None, *invalid_values):
                with self.subTest(key=key, value=value):
                    result = self._valid_cycle_result()
                    if value is None:
                        result.pop(key)
                        expected_value = "missing"
                    else:
                        result[key] = value
                        expected_value = str(value)
                    run_cycle.return_value = result

                    with self.assertRaises(ValueError) as raised:
                        generate_steady_rows(
                            [(2000.0, 1600.0, 25.0, 35.0)]
                        )

                    message = str(raised.exception)
                    self.assertIn("steady_nc2000_np1600_tc25_ta35", message)
                    self.assertIn(key, message)
                    self.assertIn(expected_value, message)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model")
    def test_off_point_allows_missing_capacity_limits_only(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.return_value = 0.0
        result = self._valid_cycle_result()
        result.update({"Q_evap": 0.0, "Q_cond": 0.0, "W_comp": 0.0})
        result.pop("Q_hx_potential")
        result.pop("Q_ref_max")
        run_cycle.return_value = result

        row = generate_steady_rows(
            [(300.0, 1600.0, 25.0, 35.0)]
        )[0]

        self.assertEqual(row["q_hx_potential_w"], 0.0)
        self.assertEqual(row["q_ref_max_w"], 0.0)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model")
    def test_steady_row_preserves_the_off_1000_boundary(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.side_effect = lambda n_comp: (
            0.0 if n_comp <= 300.0 else 800.0
        )
        run_cycle.side_effect = lambda n_comp, *_args, **_kwargs: {
            "Q_evap": 0.0 if n_comp <= 300.0 else 900.0,
            "Q_cond": 0.0 if n_comp <= 300.0 else 1200.0,
            "Q_hx_potential": 900.0,
            "Q_ref_max": 1000.0,
            "W_comp": 100.0,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }

        rows = generate_steady_rows(
            [
                (300.0, 1600.0, 25.0, 35.0),
                (1000.0, 1600.0, 25.0, 35.0),
            ]
        )

        self.assertEqual([row["q_evap_ss_w"] for row in rows], [0.0, 900.0])
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            [
                "steady_nc300_np1600_tc25_ta35",
                "steady_nc1000_np1600_tc25_ta35",
            ],
        )
        self.assertEqual(run_cycle.call_count, 2)
        self.assertEqual(
            run_cycle.call_args_list[1].args,
            (1000.0, 800.0, 298.15, 0.25, 308.15),
        )
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(rows[0]))
        self.assertEqual(rows[0]["limit_type"], "compressor_off")
        self.assertEqual(rows[1]["limit_type"], "heat_exchanger_limit")
        self.assertEqual(rows[0]["w_fan_w"], 0.0)
        self.assertEqual(rows[0]["t_cool_out_c"], 25.0)
        self.assertEqual(rows[0]["configuration_hash"], rows[1]["configuration_hash"])
        self.assertNotIn("battery_pack_source_hash", rows[0])
        self.assertNotIn("hppc_parameters_hash", rows[0])
        self.assertRegex(rows[0]["plant_source_hash"], r"^[0-9a-f]{64}$")
        self.assertTrue(rows[0]["plant_model_version"])


class DynamicExcitationTest(unittest.TestCase):
    def test_ramp_limited_multilevel_sequence_is_deterministic_and_limited(self):
        first = ramp_limited_multilevel_sequence(
            (1000.0, 2000.0, 4000.0, 6000.0), 80, 600.0, seed=17
        )
        second = ramp_limited_multilevel_sequence(
            (1000.0, 2000.0, 4000.0, 6000.0), 80, 600.0, seed=17
        )

        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual(len(first), 80)
        self.assertGreaterEqual(float(first.min()), 1000.0)
        self.assertLessEqual(float(first.max()), 6000.0)
        self.assertLessEqual(float(abs(first[1:] - first[:-1]).max()), 600.0)

    def test_ramp_limited_multilevel_sequence_rejects_invalid_inputs(self):
        invalid_calls = (
            ((), 5, 1.0),
            ((1.0,), 0, 1.0),
            ((1.0,), -1, 1.0),
            ((1.0,), 5, -1.0),
        )
        for levels, steps, dmax in invalid_calls:
            with self.subTest(levels=levels, steps=steps, dmax=dmax):
                with self.assertRaises(ValueError):
                    ramp_limited_multilevel_sequence(
                        levels, steps, dmax, seed=17
                    )

    def test_smoke_scenarios_are_stable_repeatable_and_cover_required_kinds(self):
        first = build_dynamic_scenarios("smoke", seed=17)
        second = build_dynamic_scenarios("smoke", seed=17)

        self.assertEqual(first, second)
        self.assertEqual(
            [spec.scenario_id for spec in first],
            [
                "dynamic_smoke_compressor_only_forward",
                "dynamic_smoke_pump_only_forward",
                "dynamic_smoke_combined_forward",
                "dynamic_smoke_combined_reverse",
            ],
        )
        self.assertTrue(all(spec.steps == 20 for spec in first))
        self.assertEqual(
            [spec.excitation_kind for spec in first],
            ["compressor-only", "pump-only", "combined", "combined"],
        )
        self.assertEqual([spec.flow_direction for spec in first], [1, 1, 1, -1])
        self._assert_profiles_are_valid(first)

    def test_full_scenarios_cover_conditions_and_exact_split_cardinality(self):
        specs = build_dynamic_scenarios("full", seed=23)

        self.assertEqual(len(specs), 15)
        self.assertEqual(len({spec.scenario_id for spec in specs}), len(specs))
        self.assertTrue(all(spec.steps in (120, 150, 180) for spec in specs))
        self.assertEqual(
            {spec.excitation_kind for spec in specs},
            {"compressor-only", "pump-only", "combined"},
        )
        self.assertEqual({spec.flow_direction for spec in specs}, {-1, 1})
        self.assertGreaterEqual(len({spec.initial_battery_c for spec in specs}), 3)
        self.assertGreaterEqual(len({spec.initial_coolant_c for spec in specs}), 3)
        self.assertGreaterEqual(len({spec.initial_plate_c for spec in specs}), 3)
        self.assertGreaterEqual(len({spec.ambient_c for spec in specs}), 3)
        self.assertTrue(any(np.ptp(spec.current_a) > 0.0 for spec in specs))
        self.assertEqual(
            {
                kind: sum(spec.excitation_kind == kind for spec in specs)
                for kind in ("compressor-only", "pump-only", "combined")
            },
            {"compressor-only": 5, "pump-only": 5, "combined": 5},
        )
        for kind in ("compressor-only", "pump-only", "combined"):
            self.assertEqual(
                {
                    spec.flow_direction
                    for spec in specs
                    if spec.excitation_kind == kind
                },
                {-1, 1},
            )
        self._assert_profiles_are_valid(specs)

    def test_full_dynamic_splits_are_stratified_and_input_order_independent(self):
        specs = build_dynamic_scenarios("full", seed=20260714)

        first = assign_dynamic_scenario_splits(specs, seed=20260714)
        second = assign_dynamic_scenario_splits(
            list(reversed(specs)), seed=20260714
        )

        self.assertEqual(first, second)
        self.assertEqual(
            {split: list(first.values()).count(split) for split in set(first.values())},
            {"train": 9, "validation": 3, "test": 3},
        )
        by_id = {spec.scenario_id: spec for spec in specs}
        for split in ("validation", "test"):
            selected = [by_id[key] for key, value in first.items() if value == split]
            self.assertEqual(
                {spec.excitation_kind for spec in selected},
                {"compressor-only", "pump-only", "combined"},
            )
            self.assertEqual({spec.flow_direction for spec in selected}, {-1, 1})

    def test_smoke_dynamic_splits_use_the_common_base_mapping(self):
        specs = build_dynamic_scenarios("smoke", seed=17)
        scenario_ids = [spec.scenario_id for spec in specs]

        self.assertEqual(
            assign_dynamic_scenario_splits(specs, seed=99),
            assign_scenario_splits(scenario_ids, seed=99),
        )

    def _assert_profiles_are_valid(self, specs):
        for spec in specs:
            with self.subTest(scenario_id=spec.scenario_id):
                self.assertEqual(len(spec.compressor_command_rpm), spec.steps)
                self.assertEqual(len(spec.pump_command_rpm), spec.steps)
                self.assertEqual(len(spec.current_a), spec.steps)
                self.assertGreaterEqual(min(spec.compressor_command_rpm), 1000.0)
                self.assertLessEqual(max(spec.compressor_command_rpm), 6000.0)
                self.assertGreaterEqual(min(spec.pump_command_rpm), 1600.0)
                self.assertLessEqual(max(spec.pump_command_rpm), 4800.0)
                self.assertLessEqual(
                    float(np.max(np.abs(np.diff(spec.compressor_command_rpm)))),
                    600.0,
                )
                self.assertLessEqual(
                    float(np.max(np.abs(np.diff(spec.pump_command_rpm)))),
                    300.0,
                )
                if spec.excitation_kind == "compressor-only":
                    self.assertEqual(len(set(spec.pump_command_rpm)), 1)
                if spec.excitation_kind == "pump-only":
                    self.assertEqual(len(set(spec.compressor_command_rpm)), 1)


class DynamicRolloutTest(unittest.TestCase):
    class FakePack:
        instances = []

        def __init__(self, config):
            self.config = config
            self.cols = 3
            self.rows = 4
            self.current = config["total_current"]
            self.history = [[object()] for _ in range(12)]
            self.step_calls = []
            self.__class__.instances.append(self)

        def step(self, dt, T_plate, T_cabinet):
            self.step_calls.append((dt, np.asarray(T_plate).copy(), T_cabinet))

        def get_avg_temp(self):
            return 303.15

    @staticmethod
    def _spec(scenario_id="dynamic_mock"):
        return DynamicScenarioSpec(
            scenario_id=scenario_id,
            steps=1,
            dt_s=5.0,
            seed=41,
            excitation_kind="combined",
            flow_direction=-1,
            initial_soc=0.8,
            initial_battery_c=31.0,
            initial_coolant_c=26.0,
            initial_plate_c=27.0,
            ambient_c=35.0,
            compressor_command_rpm=(3200.0,),
            pump_command_rpm=(2400.0,),
            current_a=(400.0,),
        )

    @staticmethod
    def _thermal_result():
        return {
            "T_tank_K": 300.15,
            "T_plate_K_array": np.array([299.15, 300.15, 301.15]),
            "W_comp_real": 111.0,
            "W_pump_val": 22.0,
            "W_fan_real": 8.0,
            "N_comp_eff": 3100.0,
            "N_pump_eff": 2300.0,
            "N_fan_cmd": 800.0,
            "N_fan_eff": 750.0,
            "dynamic_state": {"token": "next"},
            "m_dot_cool": 0.21,
            "T_pipe_supply_K": 296.15,
            "T_pipe_return_K": 302.15,
            "Q_dot_evap": 700.0,
            "Q_dot_cond": 950.0,
        }

    @staticmethod
    def _cycle_result():
        return {
            "Q_evap": 900.0,
            "Q_cond": 1200.0,
            "Q_hx_potential": 950.0,
            "Q_ref_max": 1000.0,
            "W_comp": 105.0,
            "W_fan": 7.0,
            "T_cool_out": 297.15,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }

    def setUp(self):
        self.FakePack.instances.clear()

    def test_dynamic_spec_rejects_boolean_or_noninteger_direction_and_seed(self):
        invalid_values = (
            ("flow_direction", True),
            ("flow_direction", False),
            ("flow_direction", 1.0),
            ("flow_direction", -1.0),
            ("seed", True),
            ("seed", 17.5),
        )
        for field, value in invalid_values:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    run_dynamic_scenario(
                        replace(self._spec(), **{field: value}), split="train"
                    )

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_one_step_uses_dynamic_plant_and_separate_steady_diagnostic(
        self,
        battery_pack,
        initialize_state,
        simulate_step,
        staged_fan,
        pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        initialize_state.return_value = {"token": "initial"}
        simulate_step.return_value = self._thermal_result()
        staged_fan.return_value = 777.0
        pump_model.return_value = (0.23, 23.0)
        run_cycle.return_value = self._cycle_result()

        row = run_dynamic_scenario(self._spec(), split="train")[0]

        self.assertTrue(set(REQUIRED_COLUMNS).issubset(row))
        self.assertEqual(row["q_gen_w"], ((400.0 / 4.0) ** 2) * 0.001 * 52.0)
        self.assertEqual(row["flow_direction"], -1)
        self.assertEqual(row["n_comp_eff_rpm"], 3100.0)
        self.assertEqual(row["n_pump_eff_rpm"], 2300.0)
        self.assertEqual(row["q_evap_eff_w"], 700.0)
        self.assertEqual(row["q_cond_eff_w"], 950.0)
        self.assertEqual(row["q_evap_ss_w"], 900.0)
        self.assertEqual(row["q_cond_ss_w"], 1200.0)
        self.assertEqual(row["t_cool_c"], 27.0)
        self.assertEqual(row["t_cool_cycle_input_c"], 26.0)
        self.assertEqual(row["t_batt_c"], 30.0)
        self.assertEqual(row["scenario_seed"], 41)
        self.assertIs(type(row["scenario_seed"]), int)
        for key in (
            "battery_pack_source_hash",
            "hppc_parameters_hash",
            "dynamic_source_hash",
            "configuration_hash",
        ):
            self.assertRegex(row[key], r"^[0-9a-f]{64}$")
        self.assertEqual(row["dataset_kind"], "dynamic")
        self.assertEqual(row["hppc_data_source"], "file")
        self.assertIs(row["hppc_fallback_used"], False)
        self.assertEqual(
            row["source_model"], "thermal_loop.simulate_thermal_loop_step"
        )
        initialize_state.assert_called_once_with(
            3200.0,
            2400.0,
            initial_temp_k=299.15,
            dt=5.0,
        )
        simulate_step.assert_called_once()
        self.assertEqual(simulate_step.call_args.kwargs["dynamic_state"], {"token": "initial"})
        self.assertTrue(simulate_step.call_args.kwargs["is_reversed"])
        self.assertEqual(simulate_step.call_args.kwargs["T_outdoor"], 308.15)
        staged_fan.assert_called_once_with(3100.0)
        pump_model.assert_called_once_with(2300.0)
        run_cycle.assert_called_once_with(3100.0, 777.0, 299.15, 0.23, 308.15)
        fake_pack = self.FakePack.instances[0]
        self.assertEqual(fake_pack.current, 400.0)
        self.assertEqual(fake_pack.step_calls[0][0], 5.0)
        np.testing.assert_array_equal(
            fake_pack.step_calls[0][1], np.array([299.15, 300.15, 301.15])
        )
        self.assertEqual(fake_pack.step_calls[0][2], 308.15)
        self.assertTrue(all(not history for history in fake_pack.history))

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model", return_value=(0.23, 23.0))
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed", return_value=777.0)
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_two_step_rollout_chains_dynamic_state_and_increments_time(
        self,
        battery_pack,
        initialize_state,
        simulate_step,
        _staged_fan,
        _pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        initialize_state.return_value = {"token": "initial"}
        first_result = self._thermal_result()
        first_result["T_tank_K"] = 300.15
        first_result["dynamic_state"] = {"token": "step-1"}
        second_result = self._thermal_result()
        second_result["T_tank_K"] = 302.15
        second_result["dynamic_state"] = {"token": "step-2"}
        simulate_step.side_effect = [first_result, second_result]
        run_cycle.return_value = self._cycle_result()
        spec = replace(
            self._spec(),
            steps=2,
            compressor_command_rpm=(3200.0, 3400.0),
            pump_command_rpm=(2400.0, 2600.0),
            current_a=(400.0, 440.0),
        )

        rows = run_dynamic_scenario(spec, split="train")

        self.assertEqual([row["time_s"] for row in rows], [5.0, 10.0])
        self.assertEqual(
            simulate_step.call_args_list[0].kwargs["dynamic_state"],
            {"token": "initial"},
        )
        self.assertEqual(
            simulate_step.call_args_list[1].kwargs["dynamic_state"],
            {"token": "step-1"},
        )
        self.assertEqual(
            [entry.kwargs["T_tank_K"] for entry in simulate_step.call_args_list],
            [299.15, 300.15],
        )
        self.assertEqual(
            [entry.args[2] for entry in run_cycle.call_args_list],
            [299.15, 300.15],
        )
        self.assertEqual(
            [row["t_cool_cycle_input_c"] for row in rows],
            [26.0, 27.0],
        )
        self.assertEqual([row["t_cool_c"] for row in rows], [27.0, 29.0])

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model", return_value=(0.23, 23.0))
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed", return_value=777.0)
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_two_scenarios_create_fresh_pack_and_dynamic_state(
        self,
        battery_pack,
        initialize_state,
        simulate_step,
        _staged_fan,
        _pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        initialize_state.side_effect = [
            {"scenario": "first"},
            {"scenario": "second"},
        ]
        first_result = self._thermal_result()
        first_result["dynamic_state"] = {"scenario": "first-next"}
        second_result = self._thermal_result()
        second_result["dynamic_state"] = {"scenario": "second-next"}
        simulate_step.side_effect = [first_result, second_result]
        run_cycle.return_value = self._cycle_result()
        first = self._spec("dynamic_first")
        second = replace(
            self._spec("dynamic_second"),
            seed=42,
            compressor_command_rpm=(3500.0,),
            pump_command_rpm=(2600.0,),
        )

        rows = generate_dynamic_rows([first, second], seed=99)

        self.assertEqual(len(rows), 2)
        self.assertEqual(battery_pack.call_count, 2)
        self.assertEqual(len(self.FakePack.instances), 2)
        self.assertEqual(
            initialize_state.call_args_list,
            [
                call(3200.0, 2400.0, initial_temp_k=299.15, dt=5.0),
                call(3500.0, 2600.0, initial_temp_k=299.15, dt=5.0),
            ],
        )
        self.assertEqual(
            simulate_step.call_args_list[0].kwargs["dynamic_state"],
            {"scenario": "first"},
        )
        self.assertEqual(
            simulate_step.call_args_list[1].kwargs["dynamic_state"],
            {"scenario": "second"},
        )

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_dynamic_thermal_step_rejects_each_missing_or_invalid_required_output(
        self,
        battery_pack,
        initialize_state,
        simulate_step,
        staged_fan,
        pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        initialize_state.return_value = {"token": "initial"}
        staged_fan.return_value = 777.0
        pump_model.return_value = (0.23, 23.0)
        run_cycle.return_value = self._cycle_result()
        invalid_values = {
            "T_tank_K": float("nan"),
            "T_plate_K_array": np.array([299.15]),
            "dynamic_state": [],
            "N_comp_eff": float("nan"),
            "N_pump_eff": float("nan"),
            "N_fan_cmd": float("nan"),
            "N_fan_eff": float("nan"),
            "Q_dot_evap": float("nan"),
            "Q_dot_cond": float("nan"),
            "T_pipe_supply_K": float("nan"),
            "T_pipe_return_K": float("nan"),
            "W_pump_val": float("nan"),
            "W_comp_real": float("nan"),
            "W_fan_real": float("nan"),
        }

        for key, invalid_value in invalid_values.items():
            for case in ("missing", "invalid"):
                with self.subTest(key=key, case=case):
                    result = self._thermal_result()
                    if case == "missing":
                        result.pop(key)
                    else:
                        result[key] = invalid_value
                    simulate_step.return_value = result

                    with self.assertRaises(ValueError) as raised:
                        run_dynamic_scenario(self._spec(), split="train")

                    message = str(raised.exception)
                    self.assertIn("dynamic_mock", message)
                    self.assertIn("step 0", message)
                    self.assertIn(key, message)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_dynamic_diagnostics_reject_nonfinite_pump_outputs_and_fan_speed(
        self,
        battery_pack,
        initialize_state,
        simulate_step,
        staged_fan,
        pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        initialize_state.return_value = {"token": "initial"}
        simulate_step.return_value = self._thermal_result()
        run_cycle.return_value = self._cycle_result()

        diagnostic_cases = (
            ("pump_flow", "m_dot_cool", (float("nan"), 23.0), 777.0),
            ("pump_power", "w_pump", (0.23, float("inf")), 777.0),
            ("fan", "n_fan_ss", (0.23, 23.0), float("nan")),
        )
        for diagnostic, key, pump_result, fan_result in diagnostic_cases:
            with self.subTest(diagnostic=diagnostic):
                pump_model.return_value = pump_result
                staged_fan.return_value = fan_result

                with self.assertRaises(ValueError) as raised:
                    run_dynamic_scenario(self._spec(), split="train")

                message = str(raised.exception)
                self.assertIn("dynamic_mock", message)
                self.assertIn("step 0", message)
                self.assertIn(key, message)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model", return_value=(0.23, 23.0))
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed", return_value=777.0)
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state", return_value={})
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_dynamic_active_cycle_rejects_each_missing_or_nonfinite_required_value(
        self,
        battery_pack,
        _initialize_state,
        simulate_step,
        _staged_fan,
        _pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        simulate_step.return_value = self._thermal_result()
        critical_keys = (
            "Q_evap",
            "Q_cond",
            "W_comp",
            "W_fan",
            "T_cool_out",
            "T_evap_sat",
            "T_cond_sat",
            "Q_hx_potential",
            "Q_ref_max",
        )
        for key in critical_keys:
            for case, value in (("missing", None), ("nonfinite", float("nan"))):
                with self.subTest(key=key, case=case):
                    result = self._cycle_result()
                    if value is None:
                        result.pop(key)
                    else:
                        result[key] = value
                    run_cycle.return_value = result
                    with self.assertRaises(ValueError) as raised:
                        run_dynamic_scenario(self._spec(), split="train")
                    message = str(raised.exception)
                    self.assertIn("dynamic_mock", message)
                    self.assertIn("step 0", message)
                    self.assertIn(key, message)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.pump_model", return_value=(0.23, 23.0))
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.staged_fan_speed", return_value=0.0)
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.initialize_refrigeration_dynamic_state", return_value={})
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.BatteryPack")
    def test_dynamic_off_cycle_allows_only_missing_capacity_limits(
        self,
        battery_pack,
        _initialize_state,
        simulate_step,
        _staged_fan,
        _pump_model,
        run_cycle,
    ):
        battery_pack.side_effect = self.FakePack
        thermal_result = self._thermal_result()
        thermal_result["N_comp_eff"] = 300.0
        simulate_step.return_value = thermal_result
        cycle_result = self._cycle_result()
        cycle_result.update({"Q_evap": 0.0, "Q_cond": 0.0, "W_comp": 0.0})
        cycle_result.pop("Q_hx_potential")
        cycle_result.pop("Q_ref_max")
        run_cycle.return_value = cycle_result
        spec = replace(
            self._spec(), compressor_command_rpm=(1000.0,)
        )

        row = run_dynamic_scenario(spec, split="train")[0]

        self.assertEqual(row["limit_type"], "compressor_off")
        self.assertEqual(row["q_hx_potential_w"], 0.0)
        self.assertEqual(row["q_ref_max_w"], 0.0)

    @patch("experiments.physics_p.identification.generate_predictor_identification_data.run_dynamic_scenario")
    @patch("experiments.physics_p.identification.generate_predictor_identification_data.assign_dynamic_scenario_splits")
    def test_generate_dynamic_rows_assigns_splits_once_before_rollout(
        self, assign_splits, run_scenario
    ):
        specs = build_dynamic_scenarios("smoke", seed=17)
        assignment = {
            spec.scenario_id: ("train", "validation", "test", "train")[index]
            for index, spec in enumerate(specs)
        }
        assign_splits.return_value = assignment
        run_scenario.side_effect = lambda spec, split: [
            {
                **{column: 1.0 for column in REQUIRED_COLUMNS},
                "scenario_id": spec.scenario_id,
                "split": split,
                "flow_direction": spec.flow_direction,
            }
        ]

        rows = generate_dynamic_rows(specs, seed=99)

        assign_splits.assert_called_once_with(specs, seed=99)
        self.assertEqual(
            run_scenario.call_args_list,
            [call(spec, assignment[spec.scenario_id]) for spec in specs],
        )
        self.assertEqual(len(rows), 4)

    def test_generate_dynamic_rows_rejects_empty_and_duplicate_scenarios(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            generate_dynamic_rows([])

        spec = build_dynamic_scenarios("smoke", seed=17)[0]
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            generate_dynamic_rows([spec, spec])


if __name__ == "__main__":
    unittest.main()
