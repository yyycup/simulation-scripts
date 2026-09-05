import unittest

import numpy as np


from cluster_plant_v2.refrigeration import ClosedR134aCycle


class ClosedR134aCycleTests(unittest.TestCase):
    def test_cluster_sized_capacity_targets_close_at_nominal_conditions(self) -> None:
        cycle = ClosedR134aCycle()
        # Stage 8D4/8D4b (displacement 72 cc/rev, evap 5.5 m2, cond 8 m2,
        # air 6.5 kg/s, pump-rescaled coolant flow 0.899640 kg/s = 50.4 L/min).
        # 4000 rpm is gated at the 25 C nominal point (18009 W); 6000 rpm can
        # only close at spike-time coolant temperatures (the 25 C point hits
        # the 55 C condensing ceiling), so it is gated at 20 C: 21296 W.
        # Both cover the RegD 1C-charge peak (~17.4 kW cluster; scan:
        # validation/results/stage8d4_full_rescale/).
        nominal = cycle.solve(
            compressor_speed_rpm=4000.0,
            fan_speed_rpm=1200.0,
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=0.899640,
            ambient_temperature_k=308.15,
        )
        spike = cycle.solve(
            compressor_speed_rpm=6000.0,
            fan_speed_rpm=1200.0,
            coolant_inlet_temperature_k=293.15,
            coolant_mass_flow_kg_s=0.899640,
            ambient_temperature_k=308.15,
        )

        self.assertTrue(nominal["solver_success"])
        self.assertTrue(spike["solver_success"])
        self.assertTrue(17500.0 <= nominal["q_evaporator_w"] <= 18500.0)
        self.assertTrue(20800.0 <= spike["q_evaporator_w"] <= 21800.0)

    def test_component_connections_share_state_one_and_preserve_expansion_enthalpy(
        self,
    ) -> None:
        cycle = ClosedR134aCycle()
        result = cycle.evaluate_operating_point(
            compressor_speed_rpm=4000.0,
            fan_speed_rpm=1200.0,
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=0.45,
            ambient_temperature_k=308.15,
            evaporating_saturation_temperature_k=285.15,
            condensing_saturation_temperature_k=323.15,
        )

        self.assertIs(
            result["evaporator_outlet_state"],
            result["compressor_inlet_state"],
        )
        self.assertIs(result["compressor_inlet_state"], result["state_1"])
        self.assertAlmostEqual(
            result["state_4"].enthalpy_j_kg,
            result["state_3"].enthalpy_j_kg,
            places=9,
        )
        self.assertAlmostEqual(
            result["cycle_energy_residual_w"],
            0.0,
            places=9,
        )

    def test_solver_closes_compressor_flow_and_both_heat_exchangers(self) -> None:
        result = ClosedR134aCycle().solve(
            compressor_speed_rpm=4000.0,
            fan_speed_rpm=1200.0,
            coolant_inlet_temperature_k=298.15,
            coolant_mass_flow_kg_s=0.45,
            ambient_temperature_k=308.15,
        )

        self.assertTrue(result["solver_success"], result["solver_message"])
        self.assertLess(result["mass_flow_relative_residual"], 1e-3)
        self.assertLess(result["evaporator_relative_residual"], 5e-3)
        self.assertLess(result["condenser_relative_residual"], 5e-3)
        self.assertLess(result["cycle_energy_relative_residual"], 5e-3)
        self.assertGreater(result["q_evaporator_w"], 0.0)
        self.assertGreater(result["q_condenser_w"], result["q_evaporator_w"])
        self.assertGreater(result["compressor_shaft_power_w"], 0.0)
        self.assertLess(result["coolant_outlet_temperature_k"], 298.15)
        self.assertGreater(result["air_outlet_temperature_k"], 308.15)
        self.assertIsNotNone(result["state_4"].quality)
        self.assertTrue(0.0 <= result["state_4"].quality <= 1.0)
        finite_values = np.array(
            [
                result["evaporating_saturation_temperature_k"],
                result["condensing_saturation_temperature_k"],
                result["refrigerant_mass_flow_kg_s"],
                result["q_evaporator_w"],
                result["q_condenser_w"],
                result["compressor_shaft_power_w"],
            ]
        )
        self.assertTrue(np.all(np.isfinite(finite_values)))


if __name__ == "__main__":
    unittest.main()
