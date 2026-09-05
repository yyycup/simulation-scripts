"""Regression tests for the runtime-tunable Physics-P NMPC (Stage 8D4j).

Covers the TD3-preparation layer:
- weight TVP channels exist on the tunable model only;
- ``build_tunable_mpc`` reproduces ``build_mpc`` under default weights;
- ``OnlineTunableNmpc`` retunes weights without rebuilding the compiled
  NLP and shares one weight store across the horizon bank;
- horizon switching carries x0/u0/_t0 over and rejects off-bank values.

Horizon steps are shrunk to 12/18 to keep the IPOPT transcriptions fast;
the algebra under test is horizon-independent.
"""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.control.physics_p_nmpc_controller import (
    OnlineTunableNmpc,
    PhysicsPNmpcParameters,
    build_mpc,
    build_tunable_mpc,
)
from cluster_plant_v2.control.physics_p_nmpc_model import (
    STATE_DIM,
    TUNABLE_WEIGHT_TVP_NAMES,
    battery_heat_generation_preview_w,
    build_physics_p_model,
)

CURRENT_A = 560.0
AMBIENT_C = 35.0
Q_GEN_W = battery_heat_generation_preview_w(CURRENT_A)
HORIZON = 12
HORIZON_BANK = (12, 18)


def current_preview(_t: float) -> float:
    return CURRENT_A


def q_gen_preview(_t: float, _current: float) -> float:
    return Q_GEN_W


def warm_state() -> np.ndarray:
    # Warm battery against a cool loop: tracking dominates under the
    # default weights, so the solution sits on the move-rate/absolute
    # bounds and is a stable regression anchor.
    x0 = np.empty(STATE_DIM, dtype=float)
    x0[0] = 25.35
    x0[1] = 25.0
    x0[2] = 25.2
    x0[3] = 2500.0
    x0[4] = 3600.0
    x0[5] = 1500.0
    x0[6] = 3600.0
    x0[7:10] = 24.6
    x0[10:14] = 26.0
    x0[14] = 2500.0
    x0[15] = 3600.0
    return x0


def seed(mpc, x0: np.ndarray) -> None:
    mpc.x0["x"] = x0.reshape(-1, 1)
    mpc.u0["n_comp_cmd"] = 2500.0
    mpc.u0["n_pump_cmd"] = 3600.0
    mpc._t0 = 0.0
    mpc.set_initial_guess()


class TunableNmpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.parameters = PhysicsPNmpcParameters(horizon_steps=HORIZON)
        cls.x_warm = warm_state()
        build_kwargs = dict(
            current_preview=current_preview,
            ambient_temperature_c=AMBIENT_C,
            q_gen_preview=q_gen_preview,
        )
        cls.build_kwargs = build_kwargs
        cls.model_fixed = build_physics_p_model()
        cls.model_tunable = build_physics_p_model(include_weight_tvp=True)
        cls.mpc_fixed = build_mpc(
            cls.model_fixed, cls.parameters, **build_kwargs
        )
        cls.mpc_tunable, cls.weights = build_tunable_mpc(
            cls.model_tunable, cls.parameters, **build_kwargs
        )

    def test_weight_tvp_channels_only_exist_on_the_tunable_model(self) -> None:
        fixed_names = {entry.name for entry in self.model_fixed.tvp.entries}
        tunable_names = {entry.name for entry in self.model_tunable.tvp.entries}
        self.assertNotIn("w_track", fixed_names)
        self.assertTrue(set(TUNABLE_WEIGHT_TVP_NAMES) <= tunable_names)
        self.assertEqual(
            len(tunable_names) - len(fixed_names), len(TUNABLE_WEIGHT_TVP_NAMES)
        )

    def test_default_weights_reproduce_the_fixed_build(self) -> None:
        seed(self.mpc_fixed, self.x_warm)
        u_fixed = self.mpc_fixed.make_step(self.x_warm.reshape(-1, 1))
        seed(self.mpc_tunable, self.x_warm)
        u_tunable = self.mpc_tunable.make_step(self.x_warm.reshape(-1, 1))
        np.testing.assert_allclose(u_fixed, u_tunable, rtol=0.0, atol=1e-6)

    def test_weight_retune_keeps_the_compiled_solver_and_moves_the_command(
        self,
    ) -> None:
        seed(self.mpc_tunable, self.x_warm)
        u_default = self.mpc_tunable.make_step(self.x_warm.reshape(-1, 1))
        solver_before = self.mpc_tunable.S
        # Collapse ALL temperature weights together: w_upper/w_lower alone
        # would keep pinning the compressor against the warm overshoot.
        self.weights.update(
            {"w_track": 0.01, "w_upper": 0.01, "w_lower": 0.01, "w_comp": 2.0}
        )
        self.mpc_tunable._t0 = 0.0
        u_retuned = self.mpc_tunable.make_step(self.x_warm.reshape(-1, 1))
        try:
            self.assertIs(
                self.mpc_tunable.S, solver_before, "no CasADi rebuild allowed"
            )
            self.assertLess(
                float(u_retuned[0, 0]),
                float(u_default[0, 0]) - 100.0,
                "energy-axis weights must collapse the compressor command",
            )
        finally:
            defaults = PhysicsPNmpcParameters()
            self.weights.update(
                {name: float(getattr(defaults, name)) for name in self.weights}
            )

    def test_tvp_template_carries_the_injected_weights(self) -> None:
        sentinel = 7.77
        self.weights["w_comp"] = sentinel
        try:
            template = self.mpc_tunable.tvp_fun(0.0)
            for name in TUNABLE_WEIGHT_TVP_NAMES:
                values = np.asarray(
                    template["_tvp", :, name], dtype=float
                ).reshape(-1)
                self.assertEqual(values.size, HORIZON + 1)
                self.assertTrue(np.all(np.isfinite(values)))
            filled = np.asarray(
                template["_tvp", :, "w_comp"], dtype=float
            ).reshape(-1)
            np.testing.assert_allclose(filled, sentinel)
        finally:
            self.weights["w_comp"] = PhysicsPNmpcParameters().w_comp

    def test_facade_switch_horizon_and_shared_weight_store(self) -> None:
        facade = OnlineTunableNmpc(
            PhysicsPNmpcParameters(horizon_steps=HORIZON),
            horizon_bank=HORIZON_BANK,
            **self.build_kwargs,
        )
        facade.prepare(self.x_warm, 2500.0, 3600.0)
        facade.align_time(0.0)
        u_short = facade.make_step(self.x_warm.reshape(-1, 1))
        self.assertEqual(facade.horizon_steps, HORIZON)
        # Re-seed the reference instance cold so both solves start from
        # identical guesses (a warm-started IPOPT may sit in another local
        # optimum and mask the comparison).
        seed(self.mpc_tunable, self.x_warm)
        np.testing.assert_allclose(
            u_short,
            self.mpc_tunable.make_step(self.x_warm.reshape(-1, 1)),
            atol=1e-3,
        )

        self.assertTrue(facade.switch_horizon(18))
        self.assertFalse(facade.switch_horizon(18))
        np.testing.assert_allclose(
            np.asarray(facade.active_mpc.x0["x"], dtype=float).reshape(-1),
            self.x_warm,
        )
        facade.align_time(15.0)
        u_long = facade.make_step(self.x_warm.reshape(-1, 1))
        self.assertTrue(np.all(np.isfinite(u_long)))

        facade.set_weights(w_comp=1.0)
        self.assertEqual(facade.weights["w_comp"], 1.0)
        with self.assertRaises(ValueError):
            facade.set_weights(w_bogus=1.0)
        with self.assertRaises(ValueError):
            facade.set_weights(w_comp=-1.0)
        with self.assertRaises(ValueError):
            facade.switch_horizon(HORIZON + 3)

    def test_facade_requires_the_default_horizon_in_the_bank(self) -> None:
        with self.assertRaises(ValueError):
            OnlineTunableNmpc(
                PhysicsPNmpcParameters(horizon_steps=HORIZON),
                horizon_bank=(HORIZON + 6,),
                **self.build_kwargs,
            )


if __name__ == "__main__":
    unittest.main()
