import inspect
import unittest

import mpc_flow_direction_strategies as flow_mpc
from mpc_flow_direction_strategies import (
    freq_mpc_params,
    peak_mpc_params,
    runtime_mpc_params_for_scene,
    select_mpc_params_for_scene,
)
from thermal_control_strategies import create_controller


class MPCParameterSelectionTest(unittest.TestCase):
    def test_runtime_params_enable_final_terminal_cost_by_default(self):
        peak = runtime_mpc_params_for_scene("peak")
        freq = runtime_mpc_params_for_scene("freq")

        self.assertTrue(peak.terminal_cost_enabled)
        self.assertEqual(peak.w_terminal_temp, 1e6)
        self.assertTrue(freq.terminal_cost_enabled)
        self.assertEqual(freq.w_terminal_temp, 5e5)
        self.assertFalse(peak_mpc_params.terminal_cost_enabled)
        self.assertFalse(freq_mpc_params.terminal_cost_enabled)

    def test_mpc_core_uses_readable_physical_state_names(self):
        source = inspect.getsource(flow_mpc.MPCControllerDual.__init__)

        for name in (
            "T_batt_K",
            "T_cool_K",
            "N_comp",
            "N_pump",
            "Q_evap",
            "Q_cond",
            "T_plate",
            "T_supply",
            "T_return",
            "N_comp_delay",
            "N_pump_delay",
        ):
            self.assertIn(f"self.{name}", source, name)

        for old_name in (
            "x1_tbatt",
            "x2_tcool",
            "x3_ncomp_eff",
            "x4_npump_eff",
            "x5_qevap_eff",
            "x6_qcond_eff",
            "x7_tplate_c",
            "x8_tpipe_supply_c",
            "x9_tpipe_return_c",
            "x10_ncomp_cooling_eff",
            "x11_npump_flow_eff",
        ):
            self.assertNotIn(f"self.{old_name}", source, old_name)
    def test_selects_peak_params_from_peak_scene_name(self):
        params = select_mpc_params_for_scene("\u8c03\u5cf0")

        self.assertIs(params, peak_mpc_params)
        self.assertIs(select_mpc_params_for_scene("\u8c03\u5cf0"), peak_mpc_params)
        self.assertEqual(params.name, "peak_mpc_params")
        self.assertEqual(params.dynamic_target_min, 25.0)
        self.assertEqual(params.dynamic_target_max, 25.0)
        self.assertEqual(params.precool_max, 0.0)
        self.assertEqual(params.warm_relief_max, 0.0)
        self.assertEqual(params.w_high_temp, 5e6)
        self.assertEqual(params.w_cold_temp, 5e6)
        self.assertEqual(params.j_rev_on, 1.85)
        self.assertEqual(params.n_comp_min, 1000.0)
        self.assertEqual(params.dmax_comp, 6000.0)
        self.assertEqual(params.dmax_pump, 300.0)
        self.assertEqual(params.w_energy_comp, 600.0)
        self.assertEqual(params.w_energy_pump, 10000.0)
        self.assertEqual(params.w_temp_obj, 0.0)
        self.assertEqual(params.w_bat_safety, 0.0)
        self.assertEqual(params.w_term_peak, 0.0)
        self.assertEqual(params.T_term_peak, 25.0)

    def test_selects_frequency_params_from_frequency_scene_name(self):
        params = select_mpc_params_for_scene("\u8c03\u9891")

        self.assertIs(params, freq_mpc_params)
        self.assertIs(select_mpc_params_for_scene("\u8c03\u9891"), freq_mpc_params)
        self.assertEqual(params.name, "freq_mpc_params")
        self.assertEqual(params.dynamic_target_min, 25.0)
        self.assertEqual(params.dynamic_target_max, 25.0)
        self.assertEqual(params.precool_max, 0.0)
        self.assertEqual(params.warm_relief_max, 0.0)
        self.assertEqual(params.w_high_temp, 5e7)
        self.assertEqual(params.w_cold_temp, 5e7)
        self.assertEqual(params.j_rev_on, 1.95)
        self.assertEqual(params.n_comp_min, 1000.0)
        self.assertEqual(params.mpc_horizon, 45)
        self.assertEqual(params.dmax_comp, 6000.0)
        self.assertEqual(params.dmax_pump, 600.0)
        self.assertEqual(params.w_temp_obj, 0.0)
        self.assertEqual(params.w_bat_safety, 0.0)
        self.assertEqual(params.w_term_peak, 0.0)
        self.assertEqual(params.T_term_peak, 25.0)
        self.assertEqual(params.w_energy_comp, 300.0)
        self.assertEqual(params.w_energy_pump, 15000.0)
        self.assertEqual(params.w_dcomp, peak_mpc_params.w_dcomp)
        self.assertEqual(params.w_dpump, peak_mpc_params.w_dpump)

    def test_basic_mpc_objective_uses_only_energy_terms(self):
        source = inspect.getsource(flow_mpc.MPCControllerDual.__init__)

        objective = source.split("j_total_expr = (", 1)[1].split(
            "self.j_total = self.m.Intermediate(j_total_expr)",
            1,
        )[0]

        self.assertEqual(flow_mpc.MPC_CV_BAND_C, 0.5)
        self.assertNotIn("w_temp_obj", objective)
        self.assertNotIn("j_track", objective)
        self.assertIn("self.mpc_params.w_energy_comp * self.j_comp", objective)
        self.assertIn("self.mpc_params.w_energy_pump * self.j_pump", objective)
        self.assertNotIn("j_plate_reserve", objective)
        self.assertNotIn("j_tank_reserve", objective)
        self.assertNotIn("j_qevap_reserve", objective)
        self.assertNotIn("j_bat_safety", objective)
        self.assertIn("if self.terminal_cost_enabled", objective)
        self.assertIn("self.weighted_j_terminal_path", objective)

    def test_create_controller_passes_case_params_to_supervised_mpc(self):
        original_model = flow_mpc.MPCControllerDual

        class FakeMPCControllerDual:
            def __init__(self, *args, **kwargs):
                self.params = args[0]
                self.np_horizon = kwargs["np_horizon"]
                self.mpc_params = kwargs["mpc_params"]

        flow_mpc.MPCControllerDual = FakeMPCControllerDual
        try:
            controller = create_controller(
                "mpc",
                [560.0, 560.0],
                mpc_flow_mode="supervised",
                case_name="freq-reg",
            )
        finally:
            flow_mpc.MPCControllerDual = original_model

        self.assertTrue(controller.mpc_params.name.startswith(freq_mpc_params.name))
        self.assertTrue(controller.mpc_params.terminal_cost_enabled)
        self.assertEqual(controller.mpc_params.w_terminal_temp, 5e5)
        self.assertTrue(controller.forward_model.mpc_params.name.startswith("freq_mpc_params"))
        self.assertTrue(controller.reverse_model.mpc_params.name.startswith("freq_mpc_params"))
        self.assertEqual(controller.j_rev_on, freq_mpc_params.j_rev_on)
        self.assertEqual(controller.min_hold_time, freq_mpc_params.reverse_hold_s)
        self.assertEqual(controller.forward_model.params["N_comp_min"], freq_mpc_params.n_comp_min)
        self.assertEqual(controller.forward_model.np_horizon, freq_mpc_params.mpc_horizon)
        self.assertEqual(controller.reverse_model.np_horizon, freq_mpc_params.mpc_horizon)
        self.assertEqual(controller.forward_model.mpc_params.dmax_comp, freq_mpc_params.dmax_comp)
        self.assertEqual(controller.reverse_model.mpc_params.dmax_pump, freq_mpc_params.dmax_pump)


if __name__ == "__main__":
    unittest.main()




























