"""Electrical-input accounting on :class:`ClusterPlantOutputs`.

The frozen R134a cycle stops at the compressor shaft: it includes neither the
drive train (motor + VFD), nor the condenser fan, nor the coolant pump. A real
unit draws all four from the grid.

These tests pin the grid-side ``electrical_power_w`` channel and - just as
importantly - pin that the physics itself is untouched by it.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from cluster_plant_v2.parameters import (
    CONDENSER_FAN_POWER_FULL_SPEED_W,
    CONDENSER_FAN_REFERENCE_SPEED_RPM,
    DRIVE_TRAIN_EFFICIENCY,
)
from cluster_plant_v2.plant import ClusterPlantInputs, condenser_fan_power_w
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    build_final_plant,
)


def _step(**overrides: float):
    plant = build_final_plant(
        4000.0, initial_cluster_current_a=560.0, direction="forward"
    )
    inputs = ClusterPlantInputs(
        cluster_current_a=560.0,
        compressor_command_rpm=4000.0,
        pump_rpm=3600.0,
        fan_rpm=1200.0,
        ambient_temperature_k=308.15,
        flow_direction="forward",
    )
    if overrides:
        inputs = replace(inputs, **overrides)
    return plant.step(inputs, dt_s=5.0)


class CondenserFanPowerTests(unittest.TestCase):
    def test_full_speed_matches_the_nominal_constant(self) -> None:
        self.assertAlmostEqual(
            condenser_fan_power_w(CONDENSER_FAN_REFERENCE_SPEED_RPM),
            CONDENSER_FAN_POWER_FULL_SPEED_W,
        )

    def test_cubic_speed_law(self) -> None:
        half_speed = CONDENSER_FAN_REFERENCE_SPEED_RPM / 2.0
        self.assertAlmostEqual(
            condenser_fan_power_w(half_speed),
            CONDENSER_FAN_POWER_FULL_SPEED_W / 8.0,
        )

    def test_zero_speed_draws_nothing(self) -> None:
        self.assertEqual(condenser_fan_power_w(0.0), 0.0)

    def test_rejects_invalid_speed(self) -> None:
        with self.assertRaises(ValueError):
            condenser_fan_power_w(-100.0)


class ElectricalAccountingTests(unittest.TestCase):
    def test_electrical_power_is_the_sum_of_all_four_terms(self) -> None:
        result = _step()
        expected = (
            result["compressor_shaft_power_w"] / DRIVE_TRAIN_EFFICIENCY
            + result["condenser_fan_power_w"]
            + result["pump_power_w"]
        )
        self.assertAlmostEqual(result["electrical_power_w"], expected, places=9)

    def test_electrical_power_exceeds_shaft_power_under_load(self) -> None:
        result = _step()
        self.assertGreater(result["compressor_shaft_power_w"], 0.0)
        self.assertGreater(
            result["electrical_power_w"], result["compressor_shaft_power_w"]
        )

    def test_fan_power_field_tracks_fan_command(self) -> None:
        result = _step(fan_rpm=2000.0)
        self.assertAlmostEqual(
            result["condenser_fan_power_w"], condenser_fan_power_w(2000.0)
        )

    def test_electrical_cop_sits_below_shaft_cop(self) -> None:
        result = _step()
        self.assertGreater(result["q_evap_applied_w"], 0.0)
        shaft_cop = (
            result["q_evap_applied_w"] / result["compressor_shaft_power_w"]
        )
        self.assertLess(result["electrical_cop"], shaft_cop)

    def test_stopped_compressor_still_draws_fan_and_pump(self) -> None:
        result = _step(compressor_command_rpm=0.0)
        self.assertEqual(result["compressor_shaft_power_w"], 0.0)
        self.assertAlmostEqual(
            result["electrical_power_w"],
            result["condenser_fan_power_w"] + result["pump_power_w"],
            places=9,
        )
        self.assertGreater(result["electrical_power_w"], 0.0)

    def test_shaft_power_field_still_matches_the_cycle_solve(self) -> None:
        """Regression guard: the electrical channel must not rewrite physics."""
        result = _step()
        self.assertAlmostEqual(
            result["compressor_shaft_power_w"],
            float(result["refrigeration_result"]["compressor_shaft_power_w"]),
            places=9,
        )

    def test_thermal_outputs_remain_finite(self) -> None:
        result = _step()
        for key in (
            "cluster_supply_temperature_k",
            "cluster_return_temperature_k",
            "tank_temperature_after_k",
            "evaporator_outlet_temperature_k",
            "q_evap_applied_w",
            "total_mass_flow_kg_s",
        ):
            self.assertTrue(np.isfinite(result[key]), key)


if __name__ == "__main__":
    unittest.main()
