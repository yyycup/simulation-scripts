from __future__ import annotations

from pathlib import Path
import inspect
import tempfile
import unittest

import numpy as np
import pandas as pd

from single_pack_plant.controllers.fixed_qp.adapter import StateSpaceQPPlantController
from single_pack_plant.controllers.fixed_qp.mpc import QPMPCWeights
from single_pack_plant.simulation.case import simulate_case
from single_pack_plant.controllers.factory import create_controller


ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1]
    / "model_data"
    / "physics_p_operational_v1.json"
)


class _PackStub:
    def get_avg_temp(self):
        return 298.15


class StateSpaceQPPlantControllerTests(unittest.TestCase):
    def test_factory_creates_explicit_qp_without_changing_mpc_default(self):
        controller = create_controller(
            "state_space_qp",
            np.full(8, 560.0),
            case_name="peak short validation",
            mpc_predictor_artifact=ARTIFACT_PATH,
            mpc_horizon_override=3,
        )
        self.assertIsInstance(controller, StateSpaceQPPlantController)
        self.assertEqual(controller.qp.config.horizon, 3)

    def test_command_uses_measured_plant_state_and_returns_valid_diagnostics(self):
        controller = StateSpaceQPPlantController(
            np.full(8, 560.0),
            case_name="peak short validation",
            predictor_artifact=ARTIFACT_PATH,
            horizon_override=3,
        )
        command = controller.command(
            0,
            _PackStub(),
            298.15,
            308.15,
            thermal_state={
                "N_comp_eff": 1500.0,
                "N_pump_eff": 3000.0,
                "Q_cond_eff": 0.0,
                "Q_evap_eff": 0.0,
                "T_pipe_supply_K": 298.15,
                "T_pipe_return_K": 298.15,
            },
            plate_temps=np.full(13, 298.15),
        )
        self.assertTrue(controller.last_flow_info["solved"])
        self.assertTrue(300.0 <= command[0] <= 6000.0)
        self.assertTrue(1600.0 <= command[1] <= 4800.0)
        self.assertLessEqual(
            controller.last_flow_info["qp_max_constraint_violation"], 1.0e-6
        )

    def test_explicit_qp_weights_pass_through_simulator_factory(self):
        parameter = "state_space_qp_weights"
        self.assertIn(parameter, inspect.signature(simulate_case).parameters)
        self.assertIn(parameter, inspect.signature(create_controller).parameters)
        weights = QPMPCWeights(
            q_y=2.0,
            r_comp=0.2,
            r_pump=0.3,
            r_delta_comp=0.4,
            r_delta_pump=0.5,
            p_f=6.0,
        )
        controller = create_controller(
            "state_space_qp",
            np.full(8, 560.0),
            case_name="peak short validation",
            mpc_predictor_artifact=ARTIFACT_PATH,
            mpc_horizon_override=3,
            state_space_qp_weights=weights,
        )
        self.assertEqual(controller.qp.base_weights, weights)

    def test_default_qp_controller_uses_selected_compressor_weight(self):
        controller = StateSpaceQPPlantController(
            np.full(8, 560.0),
            case_name="peak short validation",
            predictor_artifact=ARTIFACT_PATH,
            horizon_override=3,
        )
        self.assertEqual(controller.qp.base_weights.q_y, 1.0)
        self.assertEqual(controller.qp.base_weights.r_comp, 1.0)

    def test_simulator_applies_and_logs_qp_runtime_weight_multipliers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            try:
                result = simulate_case(
                    control="state_space_qp",
                    scene="调峰",
                    flow="单向",
                    source_csv=Path(temporary_directory) / "missing.csv",
                    main_name="runtime_weights.csv",
                    snap_name="runtime_weights_snapshots.csv",
                    output_root=Path(temporary_directory),
                    duration_s=5.0,
                    current_profile_override=np.full(3, 560.0),
                    mpc_flow_mode="standard",
                    mpc_predictor="physics_p",
                    mpc_predictor_artifact=ARTIFACT_PATH,
                    mpc_horizon_override=3,
                    mpc_forecast_profile_steps=3,
                    mpc_runtime_weight_multipliers=(0.7, 1.3, 1.0),
                    force=True,
                    progress_interval_steps=1,
                    log_func=lambda _message: None,
                )
            except Exception as exc:
                self.fail(f"QP runtime multipliers were rejected: {exc}")
            frame = pd.read_csv(result["out_csv"], encoding="utf-8-sig")
        self.assertEqual(frame["MPC_Alpha_Temp"].iloc[0], 0.7)
        self.assertEqual(frame["MPC_Alpha_Comp"].iloc[0], 1.3)
        self.assertEqual(frame["MPC_Alpha_Pump"].iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
