import json
import math
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

import mpc_flow_direction_strategies as flow_mpc
import thermal_case_simulator as thermal_simulator
from mpc_physics_predictor import (
    CUBIC_CAPACITY_FEATURE_NAMES,
    CUBIC_CAPACITY_MODEL,
    DEFAULT_DYNAMIC_PARAMETERS,
    DEFAULT_INPUT_DOMAIN,
    DEFAULT_THERMAL_PARAMETERS,
    evaluate_physics_capacity,
)
from mpc_predictor_selection import CANDIDATE_B, PHYSICS_P
from thermal_control_strategies import create_controller


def _validated_physics_p_artifact():
    artifact = {
        "model_type": PHYSICS_P,
        "schema_version": 1,
        "gate": {
            "mode": "hard",
            "n_on_rpm": 1000.0,
            "width_rpm": 10.0,
        },
        "capacity": {
            "model": CUBIC_CAPACITY_MODEL,
            "coefficients": [0.5] + [0.0] * (len(CUBIC_CAPACITY_FEATURE_NAMES) - 1),
            "feature_names": list(CUBIC_CAPACITY_FEATURE_NAMES),
            "minimum_active_rpm": 1000.0,
            "n_pump_ref_rpm": 2000.0,
            "q_upper_w": 4800.0,
        },
        "input_domain": deepcopy(DEFAULT_INPUT_DOMAIN),
        "dynamic": {
            **deepcopy(DEFAULT_DYNAMIC_PARAMETERS),
            "tau_comp_s": 3.0,
            "tau_pump_s": 4.0,
            "evap_input_delay_s": 0.0,
            "pump_flow_delay_s": 5.0,
            "tau_cond_s": 28.0,
            "tau_evap_s": 45.0,
            "supply_delay_s": 15.0,
            "return_delay_s": 20.0,
            "evap_response_model": "direct",
        },
        "thermal": {
            **deepcopy(DEFAULT_THERMAL_PARAMETERS),
            "coolant_mass_flow_ref_kg_s": 0.35,
            "n_pump_ref_rpm": 3100.0,
            "battery_heat_capacity_j_k": 220000.0,
            "coolant_heat_capacity_j_k": 11000.0,
            "battery_plate_conductance_w_k": 520.0,
            "plate_tau_s": 1.5,
            "ambient_conductance_w_k": 7.5,
            "plate_fluid_effectiveness": 0.3,
            "battery_heat_generation_scale": 0.825,
            "battery_plate_conductance_model": "constant_physical",
        },
        "fit": {"fit_status": "validated"},
    }
    return artifact


class MpcPhysicsPClosedLoopTest(unittest.TestCase):
    def test_simulator_declares_temperature_bias_diagnostic_columns(self):
        self.assertEqual(
            thermal_simulator.OUTPUT_COLUMN_RENAMES[
                "MPC_P_Temp_Bias_Innovation_C"
            ],
            "MPC P temperature prediction innovation",
        )
        self.assertEqual(
            thermal_simulator.OUTPUT_COLUMN_RENAMES[
                "MPC_P_Temp_Bias_End_C"
            ],
            "MPC P terminal temperature bias compensation",
        )

    def test_temperature_bias_path_accumulates_filtered_positive_innovation(self):
        update = flow_mpc.physics_p_temperature_bias_update(
            previous_step_bias_c=0.002,
            measured_temp_c=25.010,
            previous_prediction_1_c=25.000,
            horizon_steps=14,
            alpha=0.25,
            gain=1.0,
            max_step_bias_c=0.02,
            max_horizon_bias_c=0.20,
        )

        self.assertAlmostEqual(update["innovation_c"], 0.010)
        self.assertAlmostEqual(update["step_bias_c"], 0.004)
        self.assertEqual(len(update["path_c"]), 15)
        self.assertAlmostEqual(update["path_c"][0], 0.0)
        self.assertAlmostEqual(update["path_c"][-1], 0.056)

    def test_p_prediction_domain_info_accepts_full_valid_coolant_trajectory(self):
        info = flow_mpc.physics_p_prediction_domain_info(
            [25.0, 24.5, 15.0, 35.0],
            _validated_physics_p_artifact(),
        )

        self.assertTrue(info["prediction_domain_valid"])
        self.assertEqual(info["t_cool_pred_min_c"], 15.0)
        self.assertEqual(info["t_cool_pred_max_c"], 35.0)
        self.assertEqual(info["t_cool_domain_violation_c"], 0.0)

    def test_p_prediction_domain_info_rejects_any_coolant_extrapolation(self):
        info = flow_mpc.physics_p_prediction_domain_info(
            [25.0, 14.75, 24.0],
            _validated_physics_p_artifact(),
        )

        self.assertFalse(info["prediction_domain_valid"])
        self.assertAlmostEqual(info["t_cool_domain_violation_c"], 0.25)

    def test_capacity_expression_matches_offline_p_inside_valid_domain(self):
        artifact = _validated_physics_p_artifact()

        expected = evaluate_physics_capacity(
            4000.0,
            2400.0,
            27.0,
            30.0,
            artifact=artifact,
        )
        actual = flow_mpc.physics_p_capacity_expr(
            4000.0,
            2400.0,
            27.0,
            30.0,
            artifact,
        )

        self.assertAlmostEqual(actual, expected)

    def test_capacity_expression_clips_coolant_to_validated_domain(self):
        artifact = _validated_physics_p_artifact()
        artifact["capacity"]["coefficients"][3] = 0.2
        artifact["capacity"]["coefficients"][19] = 0.1

        expected = flow_mpc.physics_p_capacity_expr(
            4000.0,
            2400.0,
            artifact["input_domain"]["t_cool_c"][0],
            30.0,
            artifact,
        )
        actual = flow_mpc.physics_p_capacity_expr(
            4000.0,
            2400.0,
            7.0,
            30.0,
            artifact,
        )

        self.assertAlmostEqual(actual, expected)

    def test_p_operating_capacity_has_no_redundant_upper_speed_smoothing(self):
        artifact = _validated_physics_p_artifact()
        n_comp_rpm = artifact["input_domain"]["n_comp_rpm"][1]
        expected = evaluate_physics_capacity(
            n_comp_rpm,
            2400.0,
            25.0,
            35.0,
            artifact=artifact,
        )
        actual = flow_mpc.physics_p_operating_capacity_expr(
            n_comp_rpm,
            2400.0,
            25.0,
            35.0,
            artifact,
            math.sqrt,
        )

        self.assertAlmostEqual(actual, expected, places=5)

    def test_smooth_capacity_bound_matches_physical_limits(self):
        upper = flow_mpc.smooth_bounded_capacity_expr(5105.0, 4800.0, math.sqrt)
        lower = flow_mpc.smooth_bounded_capacity_expr(-100.0, 4800.0, math.sqrt)

        self.assertAlmostEqual(upper, 4800.0, places=5)
        self.assertAlmostEqual(lower, 0.0, places=5)


    def test_p_operating_power_is_zero_off_and_continuous_to_1000rpm(self):
        power_off = flow_mpc.physics_p_operating_compressor_power_value(
            300.0, 25.0, 35.0
        )
        power_startup = flow_mpc.physics_p_operating_compressor_power_value(
            650.0, 25.0, 35.0
        )
        power_active = flow_mpc.physics_p_operating_compressor_power_value(
            1000.0, 25.0, 35.0
        )

        self.assertEqual(power_off, 0.0)
        self.assertAlmostEqual(power_startup, 0.5 * power_active)
        self.assertGreater(power_active, 0.0)

    def test_p_operating_power_scales_with_compressor_displacement(self):
        nominal = flow_mpc.physics_p_operating_compressor_power_value(
            3000.0,
            25.0,
            35.0,
        )
        try:
            scaled = flow_mpc.physics_p_operating_compressor_power_value(
                3000.0,
                25.0,
                35.0,
                compressor_displacement_scale=0.8,
            )
        except TypeError as exc:
            self.fail(f"operating power does not accept displacement scale: {exc}")

        self.assertAlmostEqual(scaled, 0.8 * nominal)

    def test_p_artifact_maps_dynamic_and_thermal_parameters(self):
        artifact = _validated_physics_p_artifact()
        base = {
            "C1": 1.0,
            "C2": 2.0,
            "h1_ref": 3.0,
            "N_pump_ref": 2000.0,
            "h2": 4.0,
            "cp_cool": 5.0,
            "m_dot_ref": 6.0,
        }

        mapped = flow_mpc.physics_p_params_from_artifact(base, artifact)

        self.assertEqual(mapped["tau_comp_s"], 3.0)
        self.assertEqual(mapped["tau_pump_s"], 4.0)
        self.assertEqual(mapped["theta_evap_input_s"], 0.0)
        self.assertEqual(mapped["theta_pump_flow_s"], 5.0)
        self.assertEqual(mapped["tau_cond_s"], 28.0)
        self.assertEqual(mapped["tau_evap_s"], 45.0)
        self.assertEqual(mapped["tau_pipe_supply_s"], 15.0)
        self.assertEqual(mapped["tau_pipe_return_s"], 20.0)
        self.assertEqual(mapped["C1"], 220000.0)
        self.assertEqual(mapped["C2"], 11000.0)
        self.assertEqual(mapped["h1_ref"], 520.0)
        self.assertEqual(mapped["N_pump_ref"], 3100.0)
        self.assertEqual(mapped["h2"], 7.5)
        self.assertEqual(mapped["cp_cool"], DEFAULT_THERMAL_PARAMETERS["coolant_cp_j_kg_k"])
        self.assertEqual(mapped["m_dot_ref"], 0.35)
        self.assertEqual(mapped["plate_fluid_effectiveness"], 0.3)
        self.assertEqual(mapped["battery_heat_generation_scale"], 0.825)
        self.assertAlmostEqual(
            mapped["C_plate"],
            1.5
            * 0.35
            * DEFAULT_THERMAL_PARAMETERS["coolant_cp_j_kg_k"],
        )

    def test_p_artifact_maps_compressor_displacement_scale(self):
        artifact = _validated_physics_p_artifact()
        artifact["thermal"]["compressor_displacement_scale"] = 0.8

        mapped = flow_mpc.physics_p_params_from_artifact({}, artifact)

        self.assertEqual(mapped["compressor_displacement_scale"], 0.8)

    def test_controller_factory_propagates_explicit_p_and_short_horizon(self):
        artifact = _validated_physics_p_artifact()
        original_model = flow_mpc.MPCControllerDual
        captured = []

        class FakeMPCControllerDual:
            def __init__(self, *args, **kwargs):
                self.params = args[0]
                self.np_horizon = kwargs["np_horizon"]
                self.mpc_params = kwargs["mpc_params"]
                self.predictor_name = kwargs["predictor"]
                self.predictor_artifact = kwargs["predictor_artifact"]
                captured.append(self)

        with TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "physics_p.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            flow_mpc.MPCControllerDual = FakeMPCControllerDual
            try:
                controller = create_controller(
                    "mpc",
                    [560.0, 560.0],
                    mpc_flow_mode="standard",
                    case_name="peak",
                    mpc_predictor=PHYSICS_P,
                    mpc_predictor_artifact=artifact_path,
                    mpc_horizon_override=8,
                )
            finally:
                flow_mpc.MPCControllerDual = original_model

        self.assertEqual(len(captured), 2)
        self.assertEqual(controller.mpc_params.mpc_horizon, 8)
        for model in captured:
            self.assertEqual(model.predictor_name, PHYSICS_P)
            self.assertEqual(model.predictor_artifact["model_type"], PHYSICS_P)
            self.assertEqual(model.params["C1"], 220000.0)

    def test_p_controller_applies_online_temperature_bias_to_horizon(self):
        controller = create_controller(
            "mpc",
            [560.0] * 20,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=_validated_physics_p_artifact(),
            mpc_horizon_override=8,
            physics_p_mpc_overrides={"physics_p_temp_bias_gain": 1.0},
        )
        model = controller.forward_model
        model._physics_p_previous_prediction_1_c = 25.000

        info = model._update_physics_p_temperature_bias(25.010)

        self.assertGreater(info["step_bias_c"], 0.0)
        self.assertGreater(
            float(np.asarray(model.physics_p_temp_bias_path_c.VALUE)[-1]),
            0.0,
        )

    def test_controller_factory_propagates_strict_predictor_ablation(self):
        artifact = _validated_physics_p_artifact()
        original_model = flow_mpc.MPCControllerDual
        captured = []

        class FakeMPCControllerDual:
            def __init__(self, *args, **kwargs):
                self.params = args[0]
                self.np_horizon = kwargs["np_horizon"]
                self.mpc_params = kwargs["mpc_params"]
                self.predictor_name = kwargs["predictor"]
                self.predictor_artifact = kwargs["predictor_artifact"]
                self.strict_predictor_ablation = kwargs[
                    "strict_predictor_ablation"
                ]
                captured.append(self)

        flow_mpc.MPCControllerDual = FakeMPCControllerDual
        try:
            controller = create_controller(
                "mpc",
                [560.0, 560.0],
                mpc_flow_mode="standard",
                case_name="peak",
                mpc_predictor=PHYSICS_P,
                mpc_predictor_artifact=artifact,
                mpc_horizon_override=8,
                strict_predictor_ablation=True,
            )
        finally:
            flow_mpc.MPCControllerDual = original_model

        self.assertTrue(controller.strict_predictor_ablation)
        self.assertEqual(len(captured), 2)
        self.assertTrue(all(model.strict_predictor_ablation for model in captured))

    def test_strict_ablation_aligns_external_mpc_and_solver_settings(self):
        artifact = _validated_physics_p_artifact()

        for scene in ("peak", "freq"):
            controllers = (
                create_controller(
                    "mpc",
                    [560.0] * 10,
                    mpc_flow_mode="standard",
                    case_name=scene,
                    mpc_predictor=CANDIDATE_B,
                    mpc_horizon_override=8,
                    strict_predictor_ablation=True,
                ),
                create_controller(
                    "mpc",
                    [560.0] * 10,
                    mpc_flow_mode="standard",
                    case_name=scene,
                    mpc_predictor=PHYSICS_P,
                    mpc_predictor_artifact=artifact,
                    mpc_horizon_override=8,
                    strict_predictor_ablation=True,
                ),
            )

            for controller in controllers:
                self.assertTrue(controller.strict_predictor_ablation)
                for model in (controller.forward_model, controller.reverse_model):
                    self.assertTrue(model.strict_predictor_ablation)
                    self.assertEqual(model._n_comp_command_lower_rpm, 1000.0)
                    self.assertEqual(model.u_ncomp.LOWER, 1000.0)
                    expected_mv_step_hor = 3 if scene == "freq" else 1
                    self.assertEqual(
                        model.m.options.MV_STEP_HOR,
                        expected_mv_step_hor,
                    )
                    if scene == "freq":
                        self.assertEqual(model.u_ncomp.MV_STEP_HOR, 3)
                        self.assertEqual(model.u_npump.MV_STEP_HOR, 3)
                    else:
                        self.assertIsNone(model.u_ncomp.MV_STEP_HOR)
                        self.assertIsNone(model.u_npump.MV_STEP_HOR)
                    self.assertEqual(model.m.options.RTOL, 1e-6)
                    self.assertEqual(model.m.options.OTOL, 1e-6)

            candidate_b, physics_p = controllers
            # Internal state bounds remain native to each predictor. Candidate
            # B's delay operator needs zero-valued pre-horizon history, while
            # P uses bounded physical states and deviation-form delays.
            self.assertEqual(candidate_b.forward_model.N_comp.LOWER, 0.0)
            self.assertEqual(physics_p.forward_model.N_comp.LOWER, 300.0)
            self.assertEqual(
                candidate_b.forward_model.u_ncomp.DMAX,
                physics_p.forward_model.u_ncomp.DMAX,
            )
            self.assertEqual(
                candidate_b.forward_model.u_npump.DMAX,
                physics_p.forward_model.u_npump.DMAX,
            )
            self.assertEqual(candidate_b.mpc_params, physics_p.mpc_params)

    def test_frequency_p_defaults_to_three_step_control_blocks(self):
        artifact = _validated_physics_p_artifact()
        physics_p = create_controller(
            "mpc",
            [0.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        candidate_b = create_controller(
            "mpc",
            [0.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=CANDIDATE_B,
            mpc_horizon_override=8,
        )

        self.assertEqual(physics_p.forward_model.m.options.MV_STEP_HOR, 3)
        self.assertEqual(physics_p.reverse_model.m.options.MV_STEP_HOR, 3)
        self.assertEqual(physics_p.forward_model.u_ncomp.MV_STEP_HOR, 3)
        self.assertEqual(physics_p.reverse_model.u_ncomp.MV_STEP_HOR, 3)
        self.assertEqual(physics_p.forward_model.u_npump.MV_STEP_HOR, 3)
        self.assertEqual(physics_p.reverse_model.u_npump.MV_STEP_HOR, 3)
        self.assertEqual(physics_p.forward_model.m.options.RTOL, 1e-6)
        self.assertEqual(physics_p.forward_model.m.options.OTOL, 1e-6)
        self.assertEqual(physics_p.forward_model.m.options.MAX_ITER, 250)
        self.assertEqual(physics_p.forward_model.m.options.SOLVER, 3)
        self.assertEqual(physics_p.forward_model.m.solver_options, [])
        self.assertEqual(candidate_b.forward_model.m.options.MV_STEP_HOR, 1)
        self.assertEqual(candidate_b.reverse_model.m.options.MV_STEP_HOR, 1)
        self.assertEqual(candidate_b.forward_model.m.options.RTOL, 1e-6)
        self.assertEqual(candidate_b.forward_model.m.options.OTOL, 1e-6)
        self.assertEqual(candidate_b.forward_model.m.options.MAX_ITER, 250)
        self.assertEqual(candidate_b.forward_model.m.options.SOLVER, 3)
        self.assertIsNone(candidate_b.forward_model.u_ncomp.MV_STEP_HOR)
        self.assertIsNone(candidate_b.forward_model.u_npump.MV_STEP_HOR)

    def test_peak_p_keeps_full_future_control_resolution(self):
        physics_p = create_controller(
            "mpc",
            [560.0] * 10,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=_validated_physics_p_artifact(),
            mpc_horizon_override=8,
        )

        self.assertEqual(physics_p.forward_model.m.options.MV_STEP_HOR, 1)
        self.assertEqual(physics_p.reverse_model.m.options.MV_STEP_HOR, 1)
        self.assertIsNone(physics_p.forward_model.u_ncomp.MV_STEP_HOR)
        self.assertIsNone(physics_p.reverse_model.u_ncomp.MV_STEP_HOR)
        self.assertIsNone(physics_p.forward_model.u_npump.MV_STEP_HOR)
        self.assertIsNone(physics_p.reverse_model.u_npump.MV_STEP_HOR)

    def test_peak_p_uses_relaxed_solver_tolerance(self):
        physics_p = create_controller(
            "mpc",
            [560.0] * 10,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=_validated_physics_p_artifact(),
            mpc_horizon_override=8,
        )

        self.assertEqual(physics_p.forward_model.m.options.RTOL, 3e-6)
        self.assertEqual(physics_p.reverse_model.m.options.RTOL, 3e-6)
        self.assertEqual(physics_p.forward_model.m.options.OTOL, 3e-6)
        self.assertEqual(physics_p.reverse_model.m.options.OTOL, 3e-6)

    def test_default_controller_factory_still_selects_candidate_b(self):
        original_model = flow_mpc.MPCControllerDual
        captured = []

        class FakeMPCControllerDual:
            def __init__(self, *args, **kwargs):
                self.params = args[0]
                self.np_horizon = kwargs["np_horizon"]
                self.mpc_params = kwargs["mpc_params"]
                self.predictor_name = kwargs["predictor"]
                self.predictor_artifact = kwargs["predictor_artifact"]
                captured.append(self)

        flow_mpc.MPCControllerDual = FakeMPCControllerDual
        try:
            create_controller(
                "mpc",
                [560.0, 560.0],
                mpc_flow_mode="standard",
                case_name="peak",
            )
        finally:
            flow_mpc.MPCControllerDual = original_model

        self.assertEqual(len(captured), 2)
        for model in captured:
            self.assertEqual(model.predictor_name, CANDIDATE_B)
            self.assertIsNone(model.predictor_artifact)

    def test_first_p_solve_uses_observed_thermal_and_pipe_states(self):
        artifact = _validated_physics_p_artifact()
        with TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "physics_p.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            controller = create_controller(
                "mpc",
                [560.0] * 10,
                mpc_flow_mode="standard",
                case_name="peak",
                mpc_predictor=PHYSICS_P,
                mpc_predictor_artifact=artifact_path,
                mpc_horizon_override=8,
            )

        model = controller.forward_model
        solution = model.solve_step(
            0,
            298.15,
            308.15,
            [1019.2] * 9,
            308.15,
            u_ncomp_value=1500.0,
            u_npump_value=3000.0,
            observed_thermal_state={
                "N_comp_eff": 1500.0,
                "N_pump_eff": 3000.0,
                "Q_evap_eff": 0.0,
                "Q_cond_eff": 0.0,
            },
            t_plate_meas=np.full(6, 308.15),
        )

        self.assertTrue(solution["solved"], solution["solve_error"])
        self.assertAlmostEqual(solution["t_cool_pred_c"][0], 35.0, places=6)
        self.assertAlmostEqual(float(model.T_plate.VALUE[0]), 35.0, places=6)
        self.assertAlmostEqual(float(model.Q_evap.VALUE[0]), 0.0, places=6)
        self.assertAlmostEqual(float(model.T_supply.VALUE[1]), 35.0, places=6)
        self.assertAlmostEqual(float(model.T_return.VALUE[1]), 35.0, places=6)

    def test_p_off_state_solves_with_zero_initial_cooling_and_valid_domain(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [0.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        model = controller.forward_model
        solution = model.solve_step(
            0,
            298.15,
            298.15,
            [0.0] * 9,
            308.15,
            u_ncomp_value=300.0,
            u_npump_value=1600.0,
            observed_thermal_state={
                "N_comp_eff": 300.0,
                "N_pump_eff": 1600.0,
                "Q_evap_eff": 0.0,
                "Q_cond_eff": 0.0,
                "T_pipe_supply_K": 298.15,
                "T_pipe_return_K": 298.15,
            },
            t_plate_meas=np.full(6, 298.15),
        )

        self.assertTrue(solution["solved"], solution["solve_error"])
        self.assertTrue(solution["prediction_domain_valid"])
        self.assertEqual(solution["n_comp"], 300.0)
        self.assertLess(solution["qevap_cmd_pred_1_w"], 1.0)

    def test_p_cold_solve_failure_falls_back_to_off_command(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [0.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        model = controller.forward_model

        def fail_solve():
            raise RuntimeError("forced failure")

        model._solve_with_fixed_control_recovery = fail_solve
        solution = model.solve_step(
            0,
            24.5 + 273.15,
            25.0 + 273.15,
            [0.0] * 9,
            35.0 + 273.15,
            u_ncomp_value=1800.0,
            u_npump_value=1600.0,
            target_temp_c=25.0,
        )

        self.assertFalse(solution["solved"])
        self.assertEqual(solution["n_comp"], 300.0)
        self.assertEqual(solution["n_comp_plan_rpm"], [300.0])

    def test_p_domain_floor_failure_falls_back_to_off_when_battery_is_warm(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [0.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        model = controller.forward_model

        def fail_solve():
            raise RuntimeError("forced failure")

        model._solve_with_fixed_control_recovery = fail_solve
        solution = model.solve_step(
            0,
            25.1 + 273.15,
            15.0 + 273.15,
            [0.0] * 9,
            35.0 + 273.15,
            u_ncomp_value=6000.0,
            u_npump_value=1600.0,
            target_temp_c=25.0,
            observed_thermal_state={
                "T_pipe_supply_K": 15.0 + 273.15,
                "T_pipe_return_K": 15.0 + 273.15,
            },
        )

        self.assertFalse(solution["solved"])
        self.assertEqual(solution["n_comp"], 300.0)
        self.assertEqual(solution["n_comp_plan_rpm"], [300.0])

    def test_p_warm_numeric_failure_holds_previous_feasible_actuator_commands(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [1200.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        model = controller.forward_model

        def fail_solve():
            raise RuntimeError("forced numerical failure")

        model._solve_with_fixed_control_recovery = fail_solve
        solution = model.solve_step(
            0,
            26.0 + 273.15,
            25.0 + 273.15,
            [1200.0] * 9,
            35.0 + 273.15,
            u_ncomp_value=6000.0,
            u_npump_value=3019.0,
            target_temp_c=25.0,
            observed_thermal_state={
                "T_pipe_supply_K": 25.0 + 273.15,
                "T_pipe_return_K": 25.0 + 273.15,
            },
        )

        self.assertFalse(solution["solved"])
        self.assertEqual(solution["n_comp"], 6000.0)
        self.assertEqual(solution["n_pump"], 3019.0)
        self.assertEqual(solution["n_comp_plan_rpm"], [6000.0])
        self.assertEqual(solution["n_pump_plan_rpm"], [3019.0])

    def test_p_observed_reseed_reanchors_failed_control_trajectories(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [1200.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        model = controller.forward_model
        captured = {}

        def fail_after_reseed(
            *,
            prime_first_physics_p_cycle=False,
            reseed_physics_p_trajectory=None,
        ):
            model.u_ncomp.VALUE = [300.0] * 9
            model.u_npump.VALUE = [4800.0] * 9
            reseed_physics_p_trajectory()
            captured["n_comp"] = np.asarray(
                list(model.u_ncomp.VALUE),
                dtype=float,
            ).reshape(-1).tolist()
            captured["n_pump"] = np.asarray(
                list(model.u_npump.VALUE),
                dtype=float,
            ).reshape(-1).tolist()
            raise RuntimeError("stop after observing recovery seed")

        model._solve_with_fixed_control_recovery = fail_after_reseed
        solution = model.solve_step(
            0,
            25.2 + 273.15,
            19.4 + 273.15,
            [1200.0] * 9,
            35.0 + 273.15,
            u_ncomp_value=6000.0,
            u_npump_value=3019.0,
            target_temp_c=25.0,
            observed_thermal_state={
                "N_comp_eff": 5900.0,
                "N_pump_eff": 2800.0,
                "Q_evap_eff": 3000.0,
                "Q_cond_eff": 3000.0,
                "T_pipe_supply_K": 19.0 + 273.15,
                "T_pipe_return_K": 20.0 + 273.15,
            },
            t_plate_meas=np.full(6, 21.0 + 273.15),
        )

        self.assertIn("n_comp", captured, solution["solve_error"])
        self.assertEqual(captured["n_comp"], [6000.0])
        self.assertEqual(captured["n_pump"], [3019.0])

    def test_p_exhausted_recovery_rebuilds_solver_once_from_observed_state(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [1200.0] * 10,
            mpc_flow_mode="standard",
            case_name="freq",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )
        model = controller.forward_model
        original_model_path = str(model.m.path)
        original_solve = flow_mpc.MPCControllerDual._solve_with_fixed_control_recovery
        solve_calls = []

        def fail_first_solver_instance(self, *args, **kwargs):
            solve_calls.append(str(self.m.path))
            if len(solve_calls) == 1:
                raise RuntimeError("forced exhausted recovery")
            return original_solve(self, *args, **kwargs)

        with patch.object(
            flow_mpc.MPCControllerDual,
            "_solve_with_fixed_control_recovery",
            fail_first_solver_instance,
        ):
            solution = model.solve_step(
                0,
                25.2 + 273.15,
                20.0 + 273.15,
                [1200.0] * 9,
                35.0 + 273.15,
                u_ncomp_value=6000.0,
                u_npump_value=3019.0,
                target_temp_c=25.0,
                observed_thermal_state={
                    "N_comp_eff": 5900.0,
                    "N_pump_eff": 2800.0,
                    "Q_evap_eff": 2500.0,
                    "Q_cond_eff": 1800.0,
                    "T_pipe_supply_K": 17.0 + 273.15,
                    "T_pipe_return_K": 19.0 + 273.15,
                },
                t_plate_meas=np.full(6, 18.0 + 273.15),
            )

        self.assertTrue(solution["solved"], solution["solve_error"])
        self.assertEqual(len(solve_calls), 2)
        self.assertEqual(solve_calls[0], original_model_path)
        self.assertNotEqual(solve_calls[1], original_model_path)
        self.assertIn(
            "physics-p solver model rebuilt from observed state",
            solution["solve_recovery_reason"],
        )

    def test_p_speed_states_use_validated_artifact_bounds(self):
        artifact = _validated_physics_p_artifact()
        with TemporaryDirectory() as tmp:
            artifact_path = Path(tmp) / "physics_p.json"
            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            controller = create_controller(
                "mpc",
                [560.0] * 10,
                mpc_flow_mode="standard",
                case_name="peak",
                mpc_predictor=PHYSICS_P,
                mpc_predictor_artifact=artifact_path,
                mpc_horizon_override=8,
            )

        model = controller.forward_model
        self.assertEqual(model.u_ncomp.LOWER, 300.0)
        self.assertEqual(model.N_comp.LOWER, 300.0)
        self.assertEqual(model.N_comp_delay.LOWER, 300.0)
        self.assertEqual(model.u_ncomp.DMAX, controller.mpc_params.dmax_comp)
        self.assertEqual(model.N_pump.LOWER, 1600.0)
        self.assertEqual(model.N_pump_delay.LOWER, 1600.0)
        self.assertAlmostEqual(
            float(model.N_pump_delay_ref_rpm.VALUE.value),
            float(model.N_pump_delay.VALUE.value),
        )
        self.assertAlmostEqual(
            float(model.N_pump_delay_delta_rpm.VALUE.value), 0.0
        )


    def test_p_coolant_state_uses_post_solve_validated_domain_gate(self):
        artifact = _validated_physics_p_artifact()
        controller = create_controller(
            "mpc",
            [560.0] * 10,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=artifact,
            mpc_horizon_override=8,
        )

        model = controller.forward_model
        self.assertAlmostEqual(float(model.T_batt_K.VALUE.value), 25.0 + 273.15)
        self.assertAlmostEqual(float(model.T_cool_K.VALUE.value), 25.0 + 273.15)
        self.assertAlmostEqual(float(model.T_plate.VALUE.value), 25.0)
        self.assertAlmostEqual(float(model.T_supply_ref_c.VALUE.value), 25.0)
        self.assertAlmostEqual(float(model.T_return_ref_c.VALUE.value), 25.0)
        self.assertIsNone(model.T_cool_K.LOWER)
        self.assertIsNone(model.T_cool_K.UPPER)
        domain_info = flow_mpc.physics_p_prediction_domain_info(
            [25.0, 15.0, 35.0], artifact
        )
        self.assertTrue(domain_info["prediction_domain_valid"])
        self.assertEqual(domain_info["t_cool_domain_lower_c"], 15.0)
        self.assertEqual(domain_info["t_cool_domain_upper_c"], 35.0)

    def test_candidate_b_command_minimum_remains_1000_rpm(self):
        controller = create_controller(
            "mpc",
            [560.0] * 10,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_horizon_override=8,
        )

        model = controller.forward_model

        self.assertEqual(model.u_ncomp.LOWER, 1000.0)
        self.assertEqual(controller.mpc_params.n_comp_min, 1000.0)

    def test_explicit_p_mpc_overrides_do_not_change_candidate_b_defaults(self):
        baseline = flow_mpc.runtime_mpc_params_for_scene("peak")
        original_model = flow_mpc.MPCControllerDual
        captured = []

        class FakeMPCControllerDual:
            def __init__(self, *args, **kwargs):
                self.mpc_params = kwargs["mpc_params"]
                captured.append(self)

        flow_mpc.MPCControllerDual = FakeMPCControllerDual
        try:
            controller = flow_mpc.create_mpc_flow_controller(
                [560.0, 560.0],
                mpc_flow_mode="standard",
                case_name="peak",
                predictor=PHYSICS_P,
                predictor_artifact=_validated_physics_p_artifact(),
                mpc_horizon_override=8,
                physics_p_mpc_overrides={
                    "w_energy_comp": baseline.w_energy_comp * 10.0,
                    "w_terminal_temp": baseline.w_terminal_temp * 0.5,
                    "w_dcomp": 0.1,
                    "w_dcomp_quadratic": 300.0,
                    "dmax_comp": 1000.0,
                    "cv_band_half_width": 0.30,
                    "w_temp_obj": 1000.0,
                },
            )
        finally:
            flow_mpc.MPCControllerDual = original_model

        self.assertEqual(controller.mpc_params.w_energy_comp, baseline.w_energy_comp * 10.0)
        self.assertEqual(controller.mpc_params.w_terminal_temp, baseline.w_terminal_temp * 0.5)
        self.assertEqual(controller.mpc_params.w_dcomp, 0.1)
        self.assertEqual(controller.mpc_params.w_dcomp_quadratic, 300.0)
        self.assertEqual(controller.mpc_params.dmax_comp, 1000.0)
        self.assertEqual(controller.mpc_params.cv_band_half_width, 0.30)
        self.assertEqual(controller.mpc_params.w_temp_obj, 1000.0)
        self.assertEqual(flow_mpc.runtime_mpc_params_for_scene("peak"), baseline)
        self.assertEqual(len(captured), 2)

        with self.assertRaisesRegex(ValueError, "only valid for physics_p"):
            flow_mpc.create_mpc_flow_controller(
                [560.0, 560.0],
                mpc_flow_mode="standard",
                case_name="peak",
                predictor=CANDIDATE_B,
                physics_p_mpc_overrides={"dmax_comp": 1000.0},
            )

    def test_p_quadratic_move_cost_tracks_current_applied_command(self):
        controller = create_controller(
            "mpc",
            [560.0] * 10,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=_validated_physics_p_artifact(),
            mpc_horizon_override=8,
            physics_p_mpc_overrides={
                "w_dcomp": 0.0,
                "w_dcomp_quadratic": 300.0,
            },
        )

        model = controller.forward_model
        self.assertEqual(model.u_ncomp.DCOST, 0.0)
        self.assertEqual(model.mpc_params.w_dcomp_quadratic, 300.0)
        self.assertTrue(model.quadratic_comp_move_cost_enabled)
        self.assertIsNotNone(model.previous_n_comp_cmd_rpm)
        self.assertEqual(
            list(model.quadratic_comp_move_mask.VALUE),
            [0.0, 1.0] + [0.0] * (len(model.m.time) - 2),
        )

        model._set_control_initial_guesses(2345.0, 1600.0)

        self.assertAlmostEqual(
            float(model.previous_n_comp_cmd_rpm.VALUE.value),
            2345.0,
        )

    def test_p_command_filter_blends_raw_command_with_applied_command(self):
        controller = create_controller(
            "mpc",
            [560.0] * 10,
            mpc_flow_mode="standard",
            case_name="peak",
            mpc_predictor=PHYSICS_P,
            mpc_predictor_artifact=_validated_physics_p_artifact(),
            mpc_horizon_override=8,
            physics_p_mpc_overrides={
                "comp_command_filter_alpha": 0.5,
            },
        )
        controller.last_n_comp = 2000.0
        solution = {
            "n_comp": 3000.0,
            "n_pump": 1600.0,
            "solved": True,
        }

        controller._condition_compressor_solution(solution)

        self.assertEqual(solution["n_comp_raw_rpm"], 3000.0)
        self.assertEqual(solution["n_comp"], 2500.0)
        self.assertEqual(solution["n_comp_applied_rpm"], 2500.0)
        self.assertEqual(solution["comp_command_filter_alpha"], 0.5)


if __name__ == "__main__":
    unittest.main()
