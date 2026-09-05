import math
import unittest

try:
    from cluster_plant_v2.refrigeration import CompressorSpeedActuator
except ImportError:
    CompressorSpeedActuator = None


class CompressorSpeedActuatorTests(unittest.TestCase):
    def test_five_second_step_uses_exact_first_order_response(self) -> None:
        self.assertIsNotNone(CompressorSpeedActuator)
        actuator = CompressorSpeedActuator(
            initial_speed_rpm=2000.0,
            time_constant_s=5.0,
        )

        result = actuator.step(dt_s=5.0, speed_command_rpm=4000.0)

        expected_speed = 4000.0 - 2000.0 * math.exp(-1.0)
        self.assertAlmostEqual(result["speed_before_rpm"], 2000.0, places=12)
        self.assertAlmostEqual(result["speed_command_rpm"], 4000.0, places=12)
        self.assertAlmostEqual(result["speed_after_rpm"], expected_speed, places=12)
        self.assertAlmostEqual(actuator.speed_rpm, expected_speed, places=12)
        self.assertEqual(actuator.dynamic_state_count, 1)

    def test_equal_command_preserves_speed_without_jump(self) -> None:
        self.assertIsNotNone(CompressorSpeedActuator)
        actuator = CompressorSpeedActuator(
            initial_speed_rpm=4000.0,
            time_constant_s=5.0,
        )

        result = actuator.step(dt_s=5.0, speed_command_rpm=4000.0)

        self.assertEqual(result["speed_before_rpm"], 4000.0)
        self.assertEqual(result["speed_after_rpm"], 4000.0)

    def test_zero_command_is_explicit_off_and_spools_down(self) -> None:
        self.assertIsNotNone(CompressorSpeedActuator)
        actuator = CompressorSpeedActuator(
            initial_speed_rpm=1000.0,
            time_constant_s=5.0,
        )

        # Stage 8D4d: 0 rpm is the explicit compressor-off command; the
        # first-order lag spools the shaft down toward standstill.
        result = actuator.step(dt_s=5.0, speed_command_rpm=0.0)
        self.assertAlmostEqual(
            result["speed_after_rpm"], 1000.0 * math.exp(-1.0), places=12
        )
        result = actuator.step(dt_s=5.0, speed_command_rpm=0.0)
        self.assertAlmostEqual(
            result["speed_after_rpm"], 1000.0 * math.exp(-2.0), places=12
        )

        # A restart command ramps the shaft back up through the same lag.
        result = actuator.step(dt_s=5.0, speed_command_rpm=1000.0)
        self.assertGreater(result["speed_after_rpm"], 0.0)
        self.assertLess(result["speed_after_rpm"], 1000.0)

    def test_invalid_time_step_and_speed_limits_are_rejected(self) -> None:
        self.assertIsNotNone(CompressorSpeedActuator)
        actuator = CompressorSpeedActuator(
            initial_speed_rpm=4000.0,
            time_constant_s=5.0,
        )

        with self.assertRaisesRegex(ValueError, "dt_s must be positive"):
            actuator.step(dt_s=0.0, speed_command_rpm=4000.0)
        with self.assertRaisesRegex(ValueError, "between 1000 and 6000"):
            actuator.step(dt_s=5.0, speed_command_rpm=999.0)


if __name__ == "__main__":
    unittest.main()
