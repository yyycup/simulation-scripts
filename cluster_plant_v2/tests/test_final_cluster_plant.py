import unittest

from cluster_plant_v2.validation.validate_final_cluster_plant import (
    build_final_plant,
)

try:
    from cluster_plant_v2.plant import (
        ClusterPlant,
        ClusterPlantInputs,
        ClusterPlantOutputs,
    )
except ImportError:
    ClusterPlant = None
    ClusterPlantInputs = None
    ClusterPlantOutputs = None


class ClusterPlantV2Tests(unittest.TestCase):
    def test_explicit_queues_feed_cluster_and_tank_in_declared_order(self) -> None:
        self.assertIsNotNone(ClusterPlant)
        plant = build_final_plant(
            4000.0,
            initial_cluster_current_a=560.0,
            direction="forward",
        )
        supply_delay = type(plant.supply_delay)(
            delay_s=15.0,
            dt_s=5.0,
            initial_queue_values=[290.0, 291.0, 292.0],
        )
        return_delay = type(plant.return_delay)(
            delay_s=20.0,
            dt_s=5.0,
            initial_queue_values=[300.0, 301.0, 302.0, 303.0],
        )
        plant = ClusterPlant(
            cluster=plant.cluster,
            hydraulic_network=plant.hydraulic_network,
            pump=plant.pump,
            tank=plant.tank,
            refrigeration_cycle=plant.refrigeration_cycle,
            compressor_actuator=plant.compressor_actuator,
            evaporator_dynamics=plant.evaporator_dynamics,
            supply_delay=supply_delay,
            return_delay=return_delay,
        )

        result = plant.step(
            ClusterPlantInputs(
                cluster_current_a=560.0,
                compressor_command_rpm=4000.0,
                pump_rpm=3600.0,
                fan_rpm=1200.0,
                ambient_temperature_k=308.15,
                flow_direction="forward",
            ),
            dt_s=5.0,
        )

        self.assertIsInstance(result, ClusterPlantOutputs)
        self.assertEqual(result["cluster_supply_temperature_k"], 290.0)
        self.assertEqual(result["tank_return_temperature_k"], 300.0)
        self.assertEqual(
            result["cluster_result"]["supply_temperature_k"], 290.0
        )
        self.assertEqual(
            result["tank_result"]["return_temperature_k"], 300.0
        )
        self.assertEqual(
            supply_delay.queue_values[-1],
            result["evaporator_outlet_temperature_k"],
        )
        self.assertEqual(
            return_delay.queue_values[-1],
            result["cluster_return_temperature_k"],
        )
        self.assertEqual(
            result.supply_temp_c,
            result["cluster_supply_temperature_k"] - 273.15,
        )
        self.assertEqual(
            result.compressor_actual_rpm,
            result["compressor_speed_rpm"],
        )
        self.assertEqual(result.pack_mass_flows_kg_s.shape, (5,))
        self.assertEqual(result.soc.shape, (5, 4))
        self.assertEqual(
            plant.transport_state_count,
            supply_delay.delay_steps + return_delay.delay_steps,
        )
        self.assertEqual(
            plant.dynamic_state_count,
            plant.cluster.dynamic_state_count + 1 + 1 + 1 + 7,
        )

    def test_local_component_energy_balances_remain_closed(self) -> None:
        self.assertIsNotNone(ClusterPlant)
        plant = build_final_plant(
            4000.0,
            initial_cluster_current_a=560.0,
            direction="forward",
        )

        result = plant.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_command_rpm=4000.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )

        self.assertTrue(result["refrigeration_solver_success"])
        self.assertTrue(result["all_states_finite"])
        self.assertLess(abs(result["cycle_energy_residual_w"]), 1e-8)
        self.assertLess(abs(result["evaporator_coolant_residual_w"]), 1e-8)
        self.assertLess(
            abs(result["evaporator_dynamic_energy_residual_j"]), 1e-9
        )
        self.assertLess(abs(result["cluster_fluid_residual_w"]), 1e-8)
        self.assertLess(abs(result["tank_energy_residual_j"]), 1e-8)
        self.assertFalse(result["cross_delay_energy_balance_is_modeled"])

    def test_equilibrium_initialization_fills_three_and_four_step_queues(self) -> None:
        self.assertIsNotNone(ClusterPlant)
        plant = build_final_plant(
            4000.0,
            initial_cluster_current_a=560.0,
            direction="forward",
        )
        operating_point = plant.pump.solve_operating_point(
            3600.0, plant.hydraulic_network
        )
        supply_initial = (
            plant.tank.temperature_k
            - plant.evaporator_dynamics.q_evap_applied_w
            / (
                operating_point["total_mass_flow_kg_s"]
                * plant.tank.coolant_specific_heat_j_kg_k
            )
        )
        expected_return = plant.return_delay.queue_values[0]

        self.assertEqual(plant.supply_delay.delay_steps, 3)
        self.assertEqual(plant.return_delay.delay_steps, 4)
        self.assertEqual(len(set(plant.supply_delay.queue_values)), 1)
        self.assertEqual(
            plant.return_delay.queue_values,
            (expected_return, expected_return, expected_return, expected_return),
        )
        self.assertNotEqual(expected_return, plant.tank.temperature_k)
        self.assertEqual(plant.supply_delay.queue_values[0], supply_initial)

    def test_step_rejects_dt_that_does_not_match_delay_discretization(self) -> None:
        self.assertIsNotNone(ClusterPlant)
        plant = build_final_plant(
            4000.0,
            initial_cluster_current_a=560.0,
            direction="forward",
        )

        with self.assertRaisesRegex(ValueError, "match transport-delay dt"):
            plant.step(
                dt_s=2.5,
                cluster_current_a=560.0,
                pump_speed_rpm=3600.0,
                compressor_speed_command_rpm=4000.0,
                fan_speed_rpm=1200.0,
                ambient_temperature_k=308.15,
            )


if __name__ == "__main__":
    unittest.main()
