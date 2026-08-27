from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from single_pack_plant.predictor.physics_p import (
    PhysicsArtifactError,
    initialize_physics_state,
    load_physics_artifact,
    step_physics_predictor,
)
from single_pack_plant.predictor.state_space import (
    N_COMP,
    N_PUMP,
    Q_EVAP,
    T_BAT,
    T_PLATE,
    T_TANK,
    Z_PUMP_1,
    Z_RETURN,
    Z_SUPPLY,
    PhysicsPStateSpace,
    battery_heat_generation_w,
)


ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_data"
    / "physics_p_operational_v1.json"
)


class PhysicsPStateSpaceTests(unittest.TestCase):
    def setUp(self):
        self.adapter = PhysicsPStateSpace(ARTIFACT_PATH)
        self.state = initialize_physics_state(
            n_comp_eff_rpm=3200.0,
            n_pump_eff_rpm=2800.0,
            q_cond_w=2400.0,
            q_evap_w=2200.0,
            t_supply_c=24.0,
            t_plate_c=25.0,
            t_return_c=26.0,
            t_batt_c=26.0,
            t_cool_c=25.0,
        )
        self.x = self.adapter.pack_state(self.state)
        self.u = np.array([3500.0, 3000.0])
        self.d = np.array([560.0, 35.0])

    def test_pack_unpack_round_trip_is_exact_for_retained_state(self):
        restored = self.adapter.unpack_state(self.x)
        np.testing.assert_array_equal(self.adapter.pack_state(restored), self.x)

    def test_state_and_input_scaling_round_trips(self):
        scaling = self.adapter.scaling
        np.testing.assert_allclose(
            scaling.unscale_state(scaling.scale_state(self.x)),
            self.x,
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            scaling.unscale_input(scaling.scale_input(self.u)),
            self.u,
            rtol=0.0,
            atol=1.0e-12,
        )

    def test_compact_step_matches_authoritative_physics_p(self):
        canonical_state = self.adapter.unpack_state(self.x, q_cond_w=self.state.q_cond_w)
        expected = step_physics_predictor(
            canonical_state,
            n_comp_cmd_rpm=self.u[0],
            n_pump_cmd_rpm=self.u[1],
            q_gen_w=battery_heat_generation_w(self.d[0]),
            t_ambient_c=self.d[1],
            dt_s=5.0,
            artifact=self.adapter.artifact,
        )
        np.testing.assert_allclose(
            self.adapter.step(self.x, self.u, self.d),
            self.adapter.pack_state(expected),
            rtol=0.0,
            atol=0.0,
        )

    def test_q_cond_is_decoupled_in_direct_artifact(self):
        low = step_physics_predictor(
            self.adapter.unpack_state(self.x, q_cond_w=0.0),
            self.u[0],
            self.u[1],
            battery_heat_generation_w(self.d[0]),
            self.d[1],
            5.0,
            artifact=self.adapter.artifact,
        )
        high = step_physics_predictor(
            self.adapter.unpack_state(self.x, q_cond_w=5000.0),
            self.u[0],
            self.u[1],
            battery_heat_generation_w(self.d[0]),
            self.d[1],
            5.0,
            artifact=self.adapter.artifact,
        )
        np.testing.assert_allclose(
            self.adapter.pack_state(low),
            self.adapter.pack_state(high),
            rtol=0.0,
            atol=0.0,
        )

    def test_delay_states_shift_exactly_one_three_and_four_steps(self):
        x = self.x.copy()
        x[Z_PUMP_1] = 2100.0
        x[Z_SUPPLY] = [21.0, 22.0, 23.0]
        x[Z_RETURN] = [31.0, 32.0, 33.0, 34.0]
        result = self.adapter.step(x, self.u, self.d)
        self.assertNotEqual(result[Z_PUMP_1], x[Z_PUMP_1])
        np.testing.assert_array_equal(result[Z_SUPPLY][:2], [22.0, 23.0])
        np.testing.assert_array_equal(result[Z_RETURN][:3], [32.0, 33.0, 34.0])

    def test_local_linearization_error_decreases_quadratically(self):
        model = self.adapter.linearize(self.x, self.u, self.d)
        state_direction = np.zeros_like(self.x)
        state_direction[[T_BAT, T_TANK, T_PLATE, N_COMP, N_PUMP, Q_EVAP]] = [
            0.3,
            -0.2,
            0.1,
            20.0,
            -15.0,
            10.0,
        ]
        input_direction = np.array([30.0, -20.0])
        disturbance_direction = np.array([2.0, 0.05])

        def error(multiplier):
            x = self.x + multiplier * state_direction
            u = self.u + multiplier * input_direction
            d = self.d + multiplier * disturbance_direction
            nonlinear = self.adapter.step(x, u, d)
            linear = model.A @ x + model.B @ u + model.E @ d + model.c
            return np.linalg.norm(nonlinear - linear)

        coarse = error(1.0)
        fine = error(0.5)
        self.assertGreater(coarse, 0.0)
        self.assertLess(fine, 0.4 * coarse)

    def test_incompatible_cascaded_artifact_is_rejected(self):
        artifact = deepcopy(load_physics_artifact(ARTIFACT_PATH, require_validated=True))
        artifact["dynamic"]["evap_response_model"] = "cascaded"
        with self.assertRaises(PhysicsArtifactError):
            PhysicsPStateSpace(artifact)


if __name__ == "__main__":
    unittest.main()
