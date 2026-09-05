import unittest

try:
    from cluster_plant_v2.refrigeration import CoolantTransportDelay
except ImportError:
    CoolantTransportDelay = None


class CoolantTransportDelayTests(unittest.TestCase):
    def test_supply_delay_is_exactly_three_indices_at_five_seconds(self) -> None:
        self.assertIsNotNone(CoolantTransportDelay)
        delay = CoolantTransportDelay(
            delay_s=15.0,
            dt_s=5.0,
            initial_value=25.0,
        )
        inputs = [25.0, 25.0, 25.0, 20.0, 20.0, 20.0, 20.0]

        outputs = [delay.step(value) for value in inputs]

        self.assertEqual(outputs, [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 20.0])
        self.assertEqual(delay.delay_steps, 3)
        self.assertEqual(delay.dynamic_state_count, 3)

    def test_return_delay_is_exactly_four_indices_at_five_seconds(self) -> None:
        self.assertIsNotNone(CoolantTransportDelay)
        delay = CoolantTransportDelay(
            delay_s=20.0,
            dt_s=5.0,
            initial_value=25.0,
        )
        inputs = [20.0] * 5

        outputs = [delay.step(value) for value in inputs]

        self.assertEqual(outputs, [25.0, 25.0, 25.0, 25.0, 20.0])
        self.assertEqual(delay.delay_steps, 4)
        self.assertEqual(delay.dynamic_state_count, 4)

    def test_explicit_queue_state_preserves_fifo_order(self) -> None:
        self.assertIsNotNone(CoolantTransportDelay)
        delay = CoolantTransportDelay(
            delay_s=15.0,
            dt_s=5.0,
            initial_queue_values=[21.0, 22.0, 23.0],
        )

        self.assertEqual(delay.step(24.0), 21.0)
        self.assertEqual(delay.step(25.0), 22.0)
        self.assertEqual(delay.queue_values, (23.0, 24.0, 25.0))

    def test_invalid_delay_ratio_and_queue_length_are_rejected(self) -> None:
        self.assertIsNotNone(CoolantTransportDelay)
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            CoolantTransportDelay(
                delay_s=16.0,
                dt_s=5.0,
                initial_value=25.0,
            )
        with self.assertRaisesRegex(ValueError, "exactly 3"):
            CoolantTransportDelay(
                delay_s=15.0,
                dt_s=5.0,
                initial_queue_values=[25.0, 25.0],
            )


if __name__ == "__main__":
    unittest.main()
