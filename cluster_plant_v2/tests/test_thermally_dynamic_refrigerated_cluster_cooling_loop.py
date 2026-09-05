import math
import unittest

from cluster_plant_v2.refrigeration import CompressorSpeedActuator
from cluster_plant_v2.validation.regression.dynamic_refrigerated_cluster_cooling_loop import (
    DynamicRefrigeratedClusterCoolingLoop,
)
from cluster_plant_v2.refrigeration import EvaporatorThermalDynamics
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    build_loop,
)

try:
    from cluster_plant_v2.validation.regression.thermally_dynamic_refrigerated_cluster_cooling_loop import (
        ThermallyDynamicRefrigeratedClusterCoolingLoop,
    )
except ImportError:
    ThermallyDynamicRefrigeratedClusterCoolingLoop = None


def build_stage8c1_loop(
    initial_compressor_speed_rpm: float = 4000.0,
) -> DynamicRefrigeratedClusterCoolingLoop:
    return DynamicRefrigeratedClusterCoolingLoop(
        cooling_loop=build_loop(),
        compressor_actuator=CompressorSpeedActuator(
            initial_speed_rpm=initial_compressor_speed_rpm,
            time_constant_s=5.0,
        ),
    )


class ThermallyDynamicRefrigeratedClusterCoolingLoopTests(unittest.TestCase):
    def test_applied_heat_sets_supply_and_both_energy_boundaries_close(self) -> None:
        self.assertIsNotNone(ThermallyDynamicRefrigeratedClusterCoolingLoop)
        stage8c1_loop = build_stage8c1_loop()
        loop = ThermallyDynamicRefrigeratedClusterCoolingLoop(
            dynamic_cooling_loop=stage8c1_loop,
            evaporator_dynamics=EvaporatorThermalDynamics(
                initial_q_evap_applied_w=1000.0,
                time_constant_s=45.0,
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

        expected_applied = result["q_evap_cycle_w"] + (
            1000.0 - result["q_evap_cycle_w"]
        ) * math.exp(-5.0 / 45.0)
        expected_supply = result["tank_temperature_before_k"] - (
            expected_applied
            / (
                result["total_mass_flow_kg_s"]
                * stage8c1_loop.cooling_loop.tank.coolant_specific_heat_j_kg_k
            )
        )
        self.assertAlmostEqual(
            result["q_evap_applied_w"], expected_applied, places=12
        )
        self.assertAlmostEqual(
            result["supply_temperature_k"], expected_supply, places=12
        )
        self.assertNotEqual(
            result["supply_temperature_k"],
            result["refrigeration_result"]["coolant_outlet_temperature_k"],
        )
        self.assertAlmostEqual(
            result["q_condenser_cycle_w"],
            result["q_evap_cycle_w"]
            + result["refrigerant_compression_power_w"],
            places=8,
        )
        self.assertLess(abs(result["cycle_energy_residual_w"]), 1e-8)
        self.assertLess(abs(result["evaporator_coolant_residual_w"]), 1e-8)
        self.assertLess(
            abs(result["evaporator_dynamic_energy_residual_j"]), 1e-9
        )
        self.assertLess(abs(result["thermal_chain_residual_w"]), 1e-8)
        self.assertLess(abs(result["bpt_energy_residual_j"]), 1e-5)
        self.assertLess(
            abs(result["bpt_evaporator_energy_residual_j"]), 1e-5
        )
        self.assertTrue(result["all_states_finite"])

    def test_equilibrium_factory_matches_initial_cycle_target(self) -> None:
        self.assertIsNotNone(ThermallyDynamicRefrigeratedClusterCoolingLoop)
        stage8c1_loop = build_stage8c1_loop()
        loop = ThermallyDynamicRefrigeratedClusterCoolingLoop.from_equilibrium(
            dynamic_cooling_loop=stage8c1_loop,
            pump_speed_rpm=3600.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
            evaporator_time_constant_s=45.0,
        )
        base = stage8c1_loop.cooling_loop
        operating_point = base.pump.solve_operating_point(
            3600.0, base.cluster.hydraulic_network
        )
        cycle = base.refrigeration_cycle.solve(
            compressor_speed_rpm=4000.0,
            fan_speed_rpm=1200.0,
            coolant_inlet_temperature_k=base.tank.temperature_k,
            coolant_mass_flow_kg_s=operating_point["total_mass_flow_kg_s"],
            ambient_temperature_k=308.15,
        )

        self.assertAlmostEqual(
            loop.evaporator_dynamics.q_evap_applied_w,
            cycle["q_evaporator_w"],
            places=12,
        )
        self.assertEqual(
            loop.dynamic_state_count,
            stage8c1_loop.dynamic_state_count + 1,
        )
        self.assertEqual(loop.diagnostic_state_count, 1)

    def test_explicit_zero_initialization_represents_startup(self) -> None:
        self.assertIsNotNone(ThermallyDynamicRefrigeratedClusterCoolingLoop)
        stage8c1_loop = build_stage8c1_loop()
        loop = ThermallyDynamicRefrigeratedClusterCoolingLoop(
            dynamic_cooling_loop=stage8c1_loop,
            evaporator_dynamics=EvaporatorThermalDynamics(
                initial_q_evap_applied_w=0.0,
                time_constant_s=45.0,
            ),
        )

        result = loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_command_rpm=4000.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
        )

        self.assertGreater(result["q_evap_applied_w"], 0.0)
        self.assertLess(result["q_evap_applied_w"], result["q_evap_cycle_w"])
        self.assertEqual(result["tank_temperature_before_k"] > 0.0, True)


if __name__ == "__main__":
    unittest.main()
