import unittest

import numpy as np

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.validation.regression.refrigerated_cluster_cooling_loop import (
    RefrigeratedClusterCoolingLoop,
)
from cluster_plant_v2.refrigeration import ClosedR134aCycle


def build_loop() -> RefrigeratedClusterCoolingLoop:
    header_resistance = (
        BASE_BRANCH_RESISTANCE_PA_PER_KG_S2
        * MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO
    )
    network = ParallelHeaderHydraulicNetwork(
        n_packs=5,
        supply_segment_resistances=header_resistance,
        return_segment_resistances=header_resistance,
    )
    reference_flow = 28.0 / 1000.0 / 60.0 * 1071.0
    reference_pressure = float(
        network.solve(reference_flow)["network_delta_p"]
    )
    return RefrigeratedClusterCoolingLoop(
        cluster=ReducedCluster(
            n_packs=5,
            hydraulic_mode="header_network",
            hydraulic_network=network,
            pack_config={
                "initial_soc": 0.95,
                "initial_battery_temperature_k": 298.15,
                "initial_plate_temperature_k": 298.15,
            },
        ),
        pump=CoolantPump(reference_operating_delta_p_pa=reference_pressure),
        tank=CoolantTank(initial_temperature_k=298.15),
        refrigeration_cycle=ClosedR134aCycle(),
    )


class RefrigeratedClusterCoolingLoopTests(unittest.TestCase):
    def test_nominal_step_follows_serial_evaporator_topology_and_closes_energy(
        self,
    ) -> None:
        loop = build_loop()
        tank_before = loop.tank.temperature_k
        result = loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_rpm=4000.0,
            fan_speed_rpm=1200.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )

        self.assertEqual(result["tank_temperature_before_k"], tank_before)
        self.assertEqual(
            result["supply_temperature_k"],
            result["refrigeration_result"]["coolant_outlet_temperature_k"],
        )
        self.assertEqual(
            result["cluster_result"]["supply_temperature_k"],
            result["supply_temperature_k"],
        )
        self.assertLess(result["supply_temperature_k"], tank_before)
        self.assertAlmostEqual(
            result["refrigeration_coolant_mass_flow_kg_s"],
            result["total_mass_flow_kg_s"],
            places=12,
        )
        self.assertTrue(
            np.all(result["cluster_result"]["pack_mass_flows_kg_s"] > 0.0)
        )
        self.assertAlmostEqual(
            result["cluster_result"]["pack_mass_flows_kg_s"].sum(),
            result["total_mass_flow_kg_s"],
            places=12,
        )
        self.assertTrue(result["refrigeration_solver_success"])
        self.assertGreater(result["q_evaporator_w"], 0.0)
        self.assertGreater(result["q_condenser_w"], result["q_evaporator_w"])
        self.assertGreater(result["refrigerant_compression_power_w"], 0.0)
        self.assertGreaterEqual(
            result["compressor_shaft_power_w"],
            result["refrigerant_compression_power_w"],
        )
        self.assertAlmostEqual(
            result["compressor_mechanical_loss_w"],
            result["compressor_shaft_power_w"]
            - result["refrigerant_compression_power_w"],
            places=12,
        )
        self.assertLess(abs(result["evaporator_coolant_residual_w"]), 1e-8)
        self.assertLess(abs(result["cluster_fluid_residual_w"]), 1e-8)
        self.assertLess(abs(result["thermal_chain_residual_w"]), 1e-8)
        self.assertLess(abs(result["cycle_energy_residual_w"]), 1e-8)
        self.assertLess(abs(result["total_energy_residual_j"]), 1e-6)

        expected_tank_after = tank_before + (
            result["q_tank_w"]
            * 5.0
            / loop.tank.thermal_capacity_j_k
        )
        self.assertAlmostEqual(
            result["tank_temperature_after_k"], expected_tank_after, places=12
        )

    def test_cycle_heat_exchanger_gates_use_relative_residuals_at_r1_boundary(
        self,
    ) -> None:
        loop = build_loop()
        loop.tank.temperature_k = 25.369312994612585 + 273.15
        result = loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            compressor_speed_rpm=2000.0,
            fan_speed_rpm=800.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )

        self.assertTrue(np.isfinite(result["cycle_evaporator_residual_w"]))
        self.assertLess(result["cycle_evaporator_relative_residual"], 1e-9)
        self.assertLess(result["cycle_condenser_relative_residual"], 1e-9)


if __name__ == "__main__":
    unittest.main()
