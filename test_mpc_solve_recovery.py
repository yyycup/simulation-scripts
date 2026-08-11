import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mpc_flow_direction_strategies import (
    MPCControllerDual,
    PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S,
)
from mpc_predictor_selection import CANDIDATE_B, PHYSICS_P


class _FakeGekko:
    def __init__(self, outcomes, controls):
        self.outcomes = iter(outcomes)
        self.controls = controls
        self.control_statuses = []
        self.solver_values = []
        self.max_time_values = []
        self.time_shift_values = []
        self.state_variables = []
        self.state_statuses = []
        self.options = SimpleNamespace(SOLVER=3, MAX_TIME=1.0e20, TIME_SHIFT=1)

    def solve(self, disp=False):
        self.control_statuses.append(tuple(control.STATUS for control in self.controls))
        self.max_time_values.append(self.options.MAX_TIME)
        self.solver_values.append(self.options.SOLVER)
        self.time_shift_values.append(self.options.TIME_SHIFT)
        self.state_statuses.append(
            tuple(state.FSTATUS for state in self.state_variables)
        )
        if self.control_statuses[-1] == (0, 0):
            for state in self.state_variables:
                state.MEAS = None
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome


class MpcSolveRecoveryTest(unittest.TestCase):
    def _controller(
        self,
        outcomes,
        predictor=PHYSICS_P,
        strict_predictor_ablation=False,
    ):
        controller = MPCControllerDual.__new__(MPCControllerDual)
        controller.predictor_name = predictor
        controller.strict_predictor_ablation = strict_predictor_ablation
        controller.u_ncomp = SimpleNamespace(STATUS=1)
        controller.u_npump = SimpleNamespace(STATUS=1)
        controller.m = _FakeGekko(
            outcomes,
            (controller.u_ncomp, controller.u_npump),
        )
        return controller

    def test_direct_success_does_not_use_recovery(self):
        controller = self._controller([None])

        recovered, reason = controller._solve_with_fixed_control_recovery()

        self.assertFalse(recovered)
        self.assertEqual(reason, "")
        self.assertLessEqual(
            controller.m.max_time_values[0],
            PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S,
        )
        self.assertEqual(controller.m.options.MAX_TIME, 1.0e20)
        self.assertEqual(controller.m.control_statuses, [(1, 1)])
        self.assertEqual(controller.m.time_shift_values, [0])

    def test_physics_p_keeps_scalar_control_initialization(self):
        controller = MPCControllerDual.__new__(MPCControllerDual)
        controller.predictor_name = PHYSICS_P
        controller.np_horizon = 3
        controller._physics_p_last_cycle_solved = True
        controller.previous_n_comp_cmd_rpm = None
        controller.u_ncomp = SimpleNamespace(
            VALUE=[300.0, 1000.0, 1500.0, 2000.0]
        )
        controller.u_npump = SimpleNamespace(
            VALUE=[1600.0, 1800.0, 2000.0, 2200.0]
        )

        controller._set_control_initial_guesses(1000.0, 1800.0)

        self.assertEqual(controller.u_ncomp.VALUE, 1000.0)
        self.assertEqual(controller.u_npump.VALUE, 1800.0)

    def test_physics_p_does_not_reuse_plan_after_failed_cycle(self):
        controller = MPCControllerDual.__new__(MPCControllerDual)
        controller.predictor_name = PHYSICS_P
        controller.np_horizon = 3
        controller._physics_p_last_cycle_solved = False
        controller.previous_n_comp_cmd_rpm = None
        controller.u_ncomp = SimpleNamespace(
            VALUE=[300.0, 1000.0, 1500.0, 2000.0]
        )
        controller.u_npump = SimpleNamespace(
            VALUE=[1600.0, 1800.0, 2000.0, 2200.0]
        )

        controller._set_control_initial_guesses(1000.0, 1800.0)

        self.assertEqual(controller.u_ncomp.VALUE, 1000.0)
        self.assertEqual(controller.u_npump.VALUE, 1800.0)

    def test_candidate_b_keeps_scalar_control_initialization(self):
        controller = MPCControllerDual.__new__(MPCControllerDual)
        controller.predictor_name = CANDIDATE_B
        controller.np_horizon = 3
        controller._physics_p_last_cycle_solved = True
        controller.previous_n_comp_cmd_rpm = None
        controller.u_ncomp = SimpleNamespace(
            VALUE=[1000.0, 1500.0, 2000.0, 2500.0]
        )
        controller.u_npump = SimpleNamespace(
            VALUE=[1600.0, 1800.0, 2000.0, 2200.0]
        )

        controller._set_control_initial_guesses(1500.0, 1800.0)

        self.assertEqual(controller.u_ncomp.VALUE, 1500.0)
        self.assertEqual(controller.u_npump.VALUE, 1800.0)

    def test_first_physics_p_cycle_primes_fixed_trajectory_without_recovery(self):
        controller = self._controller([None, None])
        measured_state = SimpleNamespace(FSTATUS=1, MEAS=298.15)
        controller.T_batt_K = measured_state
        controller.m.state_variables.append(measured_state)

        recovered, reason = controller._solve_with_fixed_control_recovery(
            prime_first_physics_p_cycle=True
        )

        self.assertFalse(recovered)
        self.assertEqual(reason, "")
        self.assertEqual(controller.m.control_statuses, [(0, 0), (1, 1)])
        self.assertEqual(controller.m.time_shift_values, [0, 0])
        self.assertEqual(controller.m.options.TIME_SHIFT, 1)
        self.assertEqual(controller.u_ncomp.STATUS, 1)
        self.assertEqual(controller.m.state_statuses, [(0,), (1,)])
        self.assertEqual(measured_state.FSTATUS, 1)
        self.assertEqual(measured_state.MEAS, 298.15)
        self.assertEqual(controller.u_npump.STATUS, 1)

    def test_transient_failure_retries_normal_optimization_first(self):
        controller = self._controller([RuntimeError("first failure"), None])

        recovered, reason = controller._solve_with_fixed_control_recovery()

        self.assertTrue(recovered)
        self.assertIn("first failure", reason)
        self.assertEqual(controller.m.control_statuses, [(1, 1), (1, 1)])
        self.assertEqual(controller.m.time_shift_values, [0, 0])

    def test_two_failed_solves_use_fixed_controls_then_normal_optimization(self):
        controller = self._controller(
            [
                RuntimeError("first failure"),
                RuntimeError("normal retry failure"),
                RuntimeError("alternate solver failure"),
                None,
                None,
            ]
        )

        recovered, reason = controller._solve_with_fixed_control_recovery()

        self.assertTrue(recovered)
        self.assertIn("first failure", reason)
        self.assertIn("normal retry failure", reason)
        self.assertIn("alternate solver failure", reason)
        self.assertEqual(
            controller.m.control_statuses,
            [(1, 1), (1, 1), (1, 1), (0, 0), (1, 1)],
        )
        self.assertEqual(controller.m.solver_values, [3, 3, 1, 3, 3])
        self.assertEqual(controller.m.time_shift_values, [0, 0, 0, 0, 0])
        self.assertEqual(controller.m.options.SOLVER, 3)
        self.assertEqual(controller.m.options.MAX_TIME, 1.0e20)
        self.assertEqual(controller.u_ncomp.STATUS, 1)
        self.assertEqual(controller.u_npump.STATUS, 1)

    def test_physics_p_reseeds_observed_trajectory_before_fixed_control_recovery(self):
        controller = self._controller(
            [
                RuntimeError("first failure"),
                RuntimeError("normal retry failure"),
                RuntimeError("alternate solver failure"),
                None,
                None,
            ]
        )
        measured_state = SimpleNamespace(FSTATUS=1, MEAS=298.15)
        controller.T_batt_K = measured_state
        controller.m.state_variables.append(measured_state)
        reseed_calls = []

        def reseed_from_observed_state():
            reseed_calls.append(
                (
                    controller.u_ncomp.STATUS,
                    controller.u_npump.STATUS,
                    measured_state.FSTATUS,
                )
            )

        recovered, reason = controller._solve_with_fixed_control_recovery(
            reseed_physics_p_trajectory=reseed_from_observed_state,
        )

        self.assertTrue(recovered)
        self.assertEqual(reseed_calls, [(1, 1, 1)])
        self.assertIn("physics-p observed-state trajectory reseed", reason)
        self.assertEqual(
            controller.m.control_statuses,
            [(1, 1), (1, 1), (1, 1), (0, 0), (1, 1)],
        )
        self.assertEqual(
            controller.m.state_statuses,
            [(1,), (1,), (1,), (0,), (1,)],
        )
        self.assertEqual(measured_state.FSTATUS, 1)
        self.assertEqual(measured_state.MEAS, 298.15)

    def test_failed_recovery_restores_status_and_reports_all_errors(self):
        controller = self._controller(
            [
                RuntimeError("first failure"),
                RuntimeError("normal retry failure"),
                RuntimeError("alternate solver failure"),
                RuntimeError("fixed-control failure"),
            ]
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "initial solve failed.*normal retry failed.*alternate solver failed.*fixed-control recovery failed",
        ):
            controller._solve_with_fixed_control_recovery()

        self.assertEqual(controller.u_ncomp.STATUS, 1)
        self.assertEqual(controller.u_npump.STATUS, 1)
        self.assertEqual(controller.m.options.SOLVER, 3)
        self.assertTrue(controller._last_solve_recovery_used)
        self.assertIn("fixed-control recovery failed", controller._last_solve_recovery_reason)

        self.assertEqual(controller.m.options.MAX_TIME, 1.0e20)

    def test_physics_p_total_solve_budget_stops_additional_solver_processes(self):
        controller = self._controller([RuntimeError("first failure")])

        with patch(
            "mpc_flow_direction_strategies.time.perf_counter",
            side_effect=[0.0, 0.0, 61.0, 61.0, 61.0],
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "exhausted its 60.0 s solve budget",
            ):
                controller._solve_with_fixed_control_recovery()

        self.assertEqual(controller.m.control_statuses, [(1, 1)])
        self.assertEqual(controller.m.solver_values, [3])
        self.assertEqual(controller.m.options.SOLVER, 3)
        self.assertEqual(controller.m.options.MAX_TIME, 1.0e20)
        self.assertEqual(controller.u_ncomp.STATUS, 1)
        self.assertEqual(controller.u_npump.STATUS, 1)

    def test_candidate_b_keeps_original_fixed_control_recovery_order(self):
        controller = self._controller(
            [RuntimeError("first failure"), None, None],
            predictor=CANDIDATE_B,
        )

        recovered, reason = controller._solve_with_fixed_control_recovery()

        self.assertTrue(recovered)
        self.assertIn("first failure", reason)
        self.assertNotIn("normal retry", reason)
        self.assertEqual(controller.m.control_statuses, [(1, 1), (0, 0), (1, 1)])
        self.assertEqual(controller.m.solver_values, [3, 3, 3])
        self.assertEqual(controller.m.max_time_values, [1.0e20, 1.0e20, 1.0e20])
        self.assertEqual(controller.m.options.MAX_TIME, 1.0e20)
        self.assertEqual(controller.m.time_shift_values, [1, 1, 1])

    def test_strict_candidate_b_uses_common_retry_with_native_time_shift(self):
        controller = self._controller(
            [RuntimeError("first failure"), None],
            predictor=CANDIDATE_B,
            strict_predictor_ablation=True,
        )

        recovered, reason = controller._solve_with_fixed_control_recovery()

        self.assertTrue(recovered)
        self.assertIn("first failure", reason)
        self.assertEqual(controller.m.control_statuses, [(1, 1), (1, 1)])
        self.assertEqual(controller.m.solver_values, [3, 3])
        self.assertTrue(
            all(
                max_time <= PHYSICS_P_SOLVE_CYCLE_MAX_TIME_S
                for max_time in controller.m.max_time_values
            )
        )
        self.assertEqual(controller.m.time_shift_values, [1, 1])
        self.assertEqual(controller.m.options.TIME_SHIFT, 1)


if __name__ == "__main__":
    unittest.main()
