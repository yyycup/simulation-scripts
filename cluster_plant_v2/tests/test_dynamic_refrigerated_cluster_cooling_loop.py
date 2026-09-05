import math
import unittest

from cluster_plant_v2.tests.test_refrigerated_cluster_cooling_loop import (
    build_loop,
)

try:
    from cluster_plant_v2.refrigeration import CompressorSpeedActuator
    from cluster_plant_v2.validation.regression.dynamic_refrigerated_cluster_cooling_loop import (
        DynamicRefrigeratedClusterCoolingLoop,
    )
except ImportError:
    CompressorSpeedActuator = None
    DynamicRefrigeratedClusterCoolingLoop = None


class DynamicRefrigeratedClusterCoolingLoopTests(unittest.TestCase):
    def test_actual_compressor_speed_drives_frozen_stage8b_loop(self) -> None:
        self.assertIsNotNone(CompressorSpeedActuator)
        self.assertIsNotNone(DynamicRefrigeratedClusterCoolingLoop)
        base_loop = build_loop()
        loop = DynamicRefrigeratedClusterCoolingLoop(
            cooling_loop=base_loop,
            compressor_actuator=CompressorSpeedActuator(
                initial_speed_rpm=2000.0,
                time_constant_s=5.0,
            ),
        )

        result = loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_command_rpm=4000.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )

        expected_speed = 4000.0 - 2000.0 * math.exp(-1.0)
        self.assertAlmostEqual(
            result["compressor_speed_command_rpm"], 4000.0, places=12
        )
        self.assertAlmostEqual(
            result["compressor_speed_before_rpm"], 2000.0, places=12
        )
        self.assertAlmostEqual(
            result["compressor_speed_rpm"], expected_speed, places=12
        )
        self.assertEqual(
            loop.dynamic_state_count,
            base_loop.dynamic_state_count + 1,
        )
        self.assertTrue(result["refrigeration_solver_success"])
        self.assertTrue(result["all_states_finite"])
        self.assertLess(abs(result["cycle_energy_residual_w"]), 1e-8)
        self.assertLess(abs(result["total_energy_residual_j"]), 1e-5)

    def test_matched_initial_speed_recovers_stage8b_nominal_step(self) -> None:
        self.assertIsNotNone(CompressorSpeedActuator)
        self.assertIsNotNone(DynamicRefrigeratedClusterCoolingLoop)
        reference_loop = build_loop()
        expected = reference_loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_rpm=4000.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )
        loop = DynamicRefrigeratedClusterCoolingLoop(
            cooling_loop=build_loop(),
            compressor_actuator=CompressorSpeedActuator(
                initial_speed_rpm=4000.0,
                time_constant_s=5.0,
            ),
        )

        actual = loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_command_rpm=4000.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )

        for key in (
            "q_evaporator_w",
            "q_condenser_w",
            "compressor_shaft_power_w",
            "supply_temperature_k",
            "return_temperature_k",
            "tank_temperature_after_k",
        ):
            self.assertEqual(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
