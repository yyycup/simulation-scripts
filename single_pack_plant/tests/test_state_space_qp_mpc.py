from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from single_pack_plant.predictor.physics_p import initialize_physics_state
from single_pack_plant.predictor.state_space import PhysicsPStateSpace
from single_pack_plant.controllers.fixed_qp.mpc import (
    QPMPCConfig,
    StateSpaceQPMPC,
)


ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_data"
    / "physics_p_operational_v1.json"
)


class StateSpaceQPMPCSingleStepTests(unittest.TestCase):
    def setUp(self):
        self.adapter = PhysicsPStateSpace(ARTIFACT_PATH)
        state = initialize_physics_state(
            n_comp_eff_rpm=3000.0,
            n_pump_eff_rpm=2800.0,
            q_cond_w=2200.0,
            q_evap_w=2200.0,
            t_supply_c=24.0,
            t_plate_c=25.0,
            t_return_c=26.0,
            t_batt_c=26.0,
            t_cool_c=25.0,
        )
        self.x = self.adapter.pack_state(state)
        self.preview = np.repeat([[560.0, 35.0]], 6, axis=0)
        self.previous = np.array([3000.0, 2800.0])

    def _controller(self, move_block_size=1):
        return StateSpaceQPMPC(
            self.adapter,
            QPMPCConfig(
                horizon=6,
                dmax_comp_rpm=6000.0,
                dmax_pump_rpm=300.0,
                move_block_size=move_block_size,
            ),
        )

    def test_hessian_is_symmetric_positive_semidefinite(self):
        problem = self._controller().build_problem(
            self.x, self.preview, 25.0, self.previous
        )
        np.testing.assert_allclose(problem.H, problem.H.T, rtol=0.0, atol=0.0)
        self.assertGreaterEqual(np.linalg.eigvalsh(problem.H).min(), -1.0e-10)

    def test_ltv_prediction_maps_reproduce_nominal_physics_p_trajectory(self):
        controller = self._controller()
        problem = controller.build_problem(
            self.x, self.preview, 25.0, self.previous
        )
        scaling = self.adapter.scaling
        scaled_controls = np.concatenate(
            [scaling.scale_input(row) for row in problem.nominal_controls]
        )
        predicted = np.asarray(
            [
                scaling.unscale_state(state_map @ scaled_controls + state_offset)
                for state_map, state_offset in zip(
                    problem.state_maps, problem.state_offsets
                )
            ]
        )
        actual = []
        state = self.x.copy()
        for control, disturbance in zip(problem.nominal_controls, self.preview):
            state = self.adapter.step(state, control, disturbance)
            actual.append(state)
        np.testing.assert_allclose(predicted, actual, rtol=0.0, atol=1.0e-9)

    def test_solution_satisfies_input_and_rate_constraints(self):
        solution = self._controller().solve(
            self.x, self.preview, 25.0, self.previous
        )
        self.assertTrue(solution.diagnostics.solved, solution.diagnostics)
        controls = solution.control_sequence_rpm
        self.assertTrue(np.all(controls[:, 0] >= 300.0 - 1.0e-6))
        self.assertTrue(np.all(controls[:, 0] <= 6000.0 + 1.0e-6))
        self.assertTrue(np.all(controls[:, 1] >= 1600.0 - 1.0e-6))
        self.assertTrue(np.all(controls[:, 1] <= 4800.0 + 1.0e-6))
        deltas = np.diff(np.vstack([self.previous, controls]), axis=0)
        self.assertLessEqual(np.max(np.abs(deltas[:, 0])), 6000.0 + 1.0e-6)
        self.assertLessEqual(np.max(np.abs(deltas[:, 1])), 300.0 + 1.0e-6)
        self.assertLessEqual(solution.diagnostics.max_constraint_violation, 1.0e-6)

    def test_move_blocking_holds_each_three_step_block(self):
        solution = self._controller(move_block_size=3).solve(
            self.x, self.preview, 25.0, self.previous
        )
        self.assertTrue(solution.diagnostics.solved, solution.diagnostics)
        np.testing.assert_allclose(
            solution.control_sequence_rpm[0:3],
            np.repeat(solution.control_sequence_rpm[0:1], 3, axis=0),
            rtol=0.0,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            solution.control_sequence_rpm[3:6],
            np.repeat(solution.control_sequence_rpm[3:4], 3, axis=0),
            rtol=0.0,
            atol=1.0e-6,
        )

    def test_repeated_identical_solve_is_deterministic(self):
        controller = self._controller()
        first = controller.solve(self.x, self.preview, 25.0, self.previous)
        second = controller.solve(self.x, self.preview, 25.0, self.previous)
        self.assertTrue(first.diagnostics.solved and second.diagnostics.solved)
        np.testing.assert_allclose(
            first.control_sequence_rpm,
            second.control_sequence_rpm,
            rtol=0.0,
            atol=1.0e-2,
        )

    def test_runtime_weights_update_hessian_without_changing_constraints(self):
        controller = self._controller()
        baseline = controller.build_problem(
            self.x, self.preview, 25.0, self.previous
        )
        controller.set_runtime_weight_multipliers(
            alpha_q_y=1.3, alpha_r_comp=0.7, alpha_r_pump=1.3
        )
        updated = controller.build_problem(
            self.x, self.preview, 25.0, self.previous
        )
        self.assertFalse(np.array_equal(baseline.H, updated.H))
        np.testing.assert_array_equal(
            baseline.constraint_matrix.toarray(), updated.constraint_matrix.toarray()
        )
        np.testing.assert_array_equal(baseline.lower_bounds, updated.lower_bounds)
        np.testing.assert_array_equal(baseline.upper_bounds, updated.upper_bounds)

    def test_single_step_timing_is_recorded(self):
        solution = self._controller().solve(
            self.x, self.preview, 25.0, self.previous
        )
        diagnostics = solution.diagnostics
        self.assertTrue(np.isfinite(diagnostics.solve_time_s))
        self.assertGreaterEqual(diagnostics.solve_time_s, 0.0)
        self.assertGreater(diagnostics.wall_time_s, 0.0)

    def test_infeasible_prediction_uses_executable_fallback(self):
        controller = StateSpaceQPMPC(
            self.adapter,
            QPMPCConfig(
                horizon=1,
                dmax_comp_rpm=6000.0,
                dmax_pump_rpm=300.0,
            ),
        )
        invalid_state = self.x.copy()
        invalid_state[1] = 50.0
        solution = controller.solve(
            invalid_state, self.preview[0], 25.0, self.previous
        )
        self.assertFalse(solution.diagnostics.solved)
        np.testing.assert_array_equal(solution.command_rpm, self.previous)
        self.assertIsNotNone(solution.diagnostics.fallback_reason)

    def test_formal_peak_horizon_solves_from_initial_plant_state(self):
        initial = initialize_physics_state(
            n_comp_eff_rpm=1500.0,
            n_pump_eff_rpm=3000.0,
            q_cond_w=0.0,
            q_evap_w=0.0,
            t_supply_c=35.0,
            t_plate_c=35.0,
            t_return_c=35.0,
            t_batt_c=25.0,
            t_cool_c=35.0,
        )
        state = self.adapter.pack_state(initial)
        controller = StateSpaceQPMPC(
            self.adapter,
            QPMPCConfig.for_scene("peak"),
        )
        preview = np.repeat([[560.0, 35.0]], 60, axis=0)
        solution = controller.solve(state, preview, 25.0, [1500.0, 3000.0])
        self.assertTrue(solution.diagnostics.solved, solution.diagnostics)


if __name__ == "__main__":
    unittest.main()
