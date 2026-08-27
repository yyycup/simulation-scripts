from pathlib import Path
import unittest

import numpy as np

from single_pack_plant.predictor.physics_p import initialize_physics_state, step_physics_predictor
from single_pack_plant.predictor.physics_p_casadi import CasadiPhysicsP
from single_pack_plant.predictor.state_space import PhysicsPStateSpace


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "model_data" / "physics_p_operational_v1.json"


class CasadiPhysicsPEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.python_p = PhysicsPStateSpace(ARTIFACT, dt_s=5.0)
        self.casadi_p = CasadiPhysicsP(ARTIFACT, dt_s=5.0)
        state = initialize_physics_state(300.0, 1600.0, 0.0, 0.0, 25.0, 25.0, 25.0, 25.0, 25.0)
        self.x0 = self.python_p.pack_state(state)

    def test_one_step_matches_authoritative_python_p(self):
        u = np.array([3200.0, 2700.0])
        d = np.array([560.0, 35.0])
        np.testing.assert_allclose(self.casadi_p.step(self.x0, u, d), self.python_p.step(self.x0, u, d), rtol=0.0, atol=1e-10)

    def test_twelve_step_matches_authoritative_python_p(self):
        python_state = self.x0.copy()
        casadi_state = self.x0.copy()
        for index in range(12):
            u = np.array([1800.0 + 300.0 * index, 1600.0 + 120.0 * index])
            d = np.array([280.0 + 20.0 * index, 35.0])
            python_state = self.python_p.step(python_state, u, d)
            casadi_state = self.casadi_p.step(casadi_state, u, d)
            np.testing.assert_allclose(casadi_state, python_state, rtol=0.0, atol=1e-9)

    def test_explicit_heat_disturbance_matches_python_physics_p(self):
        """CasADi must use the supplied heat load, not infer fixed resistance."""
        u = np.array([3200.0, 2700.0])
        q_gen_w = 1500.0
        expected_state = step_physics_predictor(
            self.python_p.unpack_state(self.x0),
            n_comp_cmd_rpm=u[0],
            n_pump_cmd_rpm=u[1],
            q_gen_w=q_gen_w,
            t_ambient_c=35.0,
            dt_s=5.0,
            artifact=self.python_p.artifact,
        )
        expected = self.python_p.pack_state(expected_state)
        actual = self.casadi_p.step(self.x0, u, np.array([560.0, 35.0, q_gen_w]))
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
