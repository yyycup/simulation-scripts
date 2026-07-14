from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import call, patch

import numpy as np
import pandas as pd

from predictor_identification_data import REQUIRED_COLUMNS
from generate_predictor_identification_data import (
    DynamicScenarioSpec,
    _parse_args,
    build_dynamic_scenarios,
    build_steady_grid,
    generate_dynamic_rows,
    generate_steady_rows,
    ramp_limited_multilevel_sequence,
    run_dynamic_scenario,
)


class GeneratePredictorIdentificationDataTest(unittest.TestCase):
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

    def test_full_grid_has_1200_unique_points(self):
        grid = build_steady_grid(mode="full")

        self.assertEqual(len(grid), 12 * 5 * 4 * 5)
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
            ["generate_predictor_identification_data.py", "--dataset", "all"],
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

    @patch("generate_predictor_identification_data.generate_dynamic_rows")
    @patch("generate_predictor_identification_data.build_dynamic_scenarios")
    def test_cli_dynamic_writes_only_index_free_utf8_dynamic_csv(
        self, build_scenarios, generate_rows
    ):
        build_scenarios.return_value = [object()]
        generate_rows.return_value = self._cli_rows(80, "dynamic")
        with TemporaryDirectory() as tmp, patch(
            "sys.argv",
            [
                "generate_predictor_identification_data.py",
                "--dataset",
                "dynamic",
                "--mode",
                "smoke",
                "--output-root",
                tmp,
            ],
        ):
            from generate_predictor_identification_data import main

            main()
            output_root = Path(tmp)
            paths = sorted(path.name for path in output_root.glob("*.csv"))
            frame = pd.read_csv(output_root / "dynamic_smoke.csv", encoding="utf-8")

        self.assertEqual(paths, ["dynamic_smoke.csv"])
        self.assertEqual(len(frame), 80)
        self.assertFalse(any(column.startswith("Unnamed") for column in frame))
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(frame.columns))

    @patch("generate_predictor_identification_data.generate_dynamic_rows")
    @patch("generate_predictor_identification_data.build_dynamic_scenarios")
    @patch("generate_predictor_identification_data.generate_steady_rows")
    @patch("generate_predictor_identification_data.build_steady_grid")
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
                "generate_predictor_identification_data.py",
                "--dataset",
                "all",
                "--mode",
                "smoke",
                "--output-root",
                tmp,
            ],
        ):
            from generate_predictor_identification_data import main

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

    def test_windows_runtime_path_is_prepared_before_pack_import(self):
        source = Path("generate_predictor_identification_data.py").read_text(
            encoding="utf-8"
        )

        self.assertLess(
            source.index("ensure_env_library_bin_on_path()"),
            source.index("from pack import BatteryPack"),
        )

    def test_configuration_hash_tracks_configuration_and_plant_source(self):
        from generate_predictor_identification_data import build_configuration_hash

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

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.pump_model")
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

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.pump_model")
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
            [(1999.0, 1600.0, 25.0, 35.0)]
        )[0]

        self.assertEqual(row["q_hx_potential_w"], 0.0)
        self.assertEqual(row["q_ref_max_w"], 0.0)

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.pump_model")
    def test_steady_row_preserves_the_1999_2000_boundary(
        self, pump_model, staged_fan_speed, run_cycle
    ):
        pump_model.return_value = (0.25, 20.0)
        staged_fan_speed.side_effect = lambda n_comp: (
            0.0 if n_comp < 2000.0 else 800.0
        )
        run_cycle.side_effect = lambda n_comp, *_args, **_kwargs: {
            "Q_evap": 0.0 if n_comp < 2000.0 else 900.0,
            "Q_cond": 0.0 if n_comp < 2000.0 else 1200.0,
            "Q_hx_potential": 900.0,
            "Q_ref_max": 1000.0,
            "W_comp": 100.0,
            "T_evap_sat": 280.15,
            "T_cond_sat": 320.15,
        }

        rows = generate_steady_rows(
            [
                (1999.0, 1600.0, 25.0, 35.0),
                (2000.0, 1600.0, 25.0, 35.0),
            ]
        )

        self.assertEqual([row["q_evap_ss_w"] for row in rows], [0.0, 900.0])
        self.assertEqual(
            [row["scenario_id"] for row in rows],
            [
                "steady_nc1999_np1600_tc25_ta35",
                "steady_nc2000_np1600_tc25_ta35",
            ],
        )
        self.assertEqual(run_cycle.call_count, 2)
        self.assertEqual(
            run_cycle.call_args_list[1].args,
            (2000.0, 800.0, 298.15, 0.25, 308.15),
        )
        self.assertTrue(set(REQUIRED_COLUMNS).issubset(rows[0]))
        self.assertEqual(rows[0]["limit_type"], "compressor_off")
        self.assertEqual(rows[1]["limit_type"], "heat_exchanger_limit")
        self.assertEqual(rows[0]["w_fan_w"], 0.0)
        self.assertEqual(rows[0]["t_cool_out_c"], 25.0)
        self.assertEqual(rows[0]["configuration_hash"], rows[1]["configuration_hash"])
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

        self.assertGreaterEqual(len(specs), 10)
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
        self._assert_profiles_are_valid(specs)

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

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.pump_model")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("generate_predictor_identification_data.BatteryPack")
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
        self.assertEqual(row["t_batt_c"], 30.0)
        self.assertEqual(row["scenario_seed"], 41)
        self.assertIs(type(row["scenario_seed"]), int)
        self.assertEqual(row["dataset_kind"], "dynamic")
        self.assertEqual(
            row["source_model"], "thermal_loop.simulate_thermal_loop_step"
        )
        initialize_state.assert_called_once_with(3200.0, 2400.0)
        simulate_step.assert_called_once()
        self.assertEqual(simulate_step.call_args.kwargs["dynamic_state"], {"token": "initial"})
        self.assertTrue(simulate_step.call_args.kwargs["is_reversed"])
        self.assertEqual(simulate_step.call_args.kwargs["T_outdoor"], 308.15)
        staged_fan.assert_called_once_with(3100.0)
        pump_model.assert_called_once_with(2300.0)
        run_cycle.assert_called_once_with(3100.0, 777.0, 300.15, 0.23, 308.15)
        fake_pack = self.FakePack.instances[0]
        self.assertEqual(fake_pack.current, 400.0)
        self.assertEqual(fake_pack.step_calls[0][0], 5.0)
        np.testing.assert_array_equal(
            fake_pack.step_calls[0][1], np.array([299.15, 300.15, 301.15])
        )
        self.assertEqual(fake_pack.step_calls[0][2], 308.15)
        self.assertTrue(all(not history for history in fake_pack.history))

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.pump_model")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("generate_predictor_identification_data.BatteryPack")
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
            "Q_dot_evap": float("nan"),
            "Q_dot_cond": float("nan"),
            "T_pipe_supply_K": float("nan"),
            "T_pipe_return_K": float("nan"),
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

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.pump_model")
    @patch("generate_predictor_identification_data.staged_fan_speed")
    @patch("generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("generate_predictor_identification_data.initialize_refrigeration_dynamic_state")
    @patch("generate_predictor_identification_data.BatteryPack")
    def test_dynamic_diagnostics_reject_nonfinite_pump_flow_and_fan_speed(
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

        for diagnostic, key in (("pump", "m_dot_cool"), ("fan", "n_fan_ss")):
            with self.subTest(diagnostic=diagnostic):
                pump_model.return_value = (
                    (float("nan"), 23.0)
                    if diagnostic == "pump"
                    else (0.23, 23.0)
                )
                staged_fan.return_value = (
                    float("nan") if diagnostic == "fan" else 777.0
                )

                with self.assertRaises(ValueError) as raised:
                    run_dynamic_scenario(self._spec(), split="train")

                message = str(raised.exception)
                self.assertIn("dynamic_mock", message)
                self.assertIn("step 0", message)
                self.assertIn(key, message)

    @patch("generate_predictor_identification_data.run_refrigeration_cycle")
    @patch("generate_predictor_identification_data.pump_model", return_value=(0.23, 23.0))
    @patch("generate_predictor_identification_data.staged_fan_speed", return_value=777.0)
    @patch("generate_predictor_identification_data.simulate_thermal_loop_step")
    @patch("generate_predictor_identification_data.initialize_refrigeration_dynamic_state", return_value={})
    @patch("generate_predictor_identification_data.BatteryPack")
    def test_dynamic_diagnostic_rejects_missing_or_nonfinite_critical_values(
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
        for key, value in (("Q_evap", None), ("Q_cond", float("nan"))):
            with self.subTest(key=key, value=value):
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

    @patch("generate_predictor_identification_data.run_dynamic_scenario")
    @patch("generate_predictor_identification_data.assign_scenario_splits")
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

        assign_splits.assert_called_once_with(
            [spec.scenario_id for spec in specs], seed=99
        )
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
