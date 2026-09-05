import unittest

import numpy as np

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.validation.regression.cluster_cooling_loop import (
    ClusterCoolingLoop,
)
from cluster_plant_v2.hydraulics import (
    CoolantPump,
    CoolantTank,
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO,
    ParallelHeaderHydraulicNetwork,
)


def build_loop() -> ClusterCoolingLoop:
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
    reference_pressure = network.solve(reference_flow)["network_delta_p"]
    pump = CoolantPump(reference_operating_delta_p_pa=reference_pressure)
    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={
            "initial_soc": 0.95,
            "initial_battery_temperature_k": 298.15,
            "initial_plate_temperature_k": 298.15,
        },
    )
    tank = CoolantTank(initial_temperature_k=293.15)
    return ClusterCoolingLoop(cluster=cluster, pump=pump, tank=tank)


class ClusterCoolingLoopTests(unittest.TestCase):
    def test_supply_uses_old_tank_state_and_tank_updates_last(self) -> None:
        loop = build_loop()
        old_tank_temperature = loop.tank.temperature_k

        result = loop.step(
            dt_s=5.0,
            cluster_current_a=560.0,
            pump_speed_rpm=3600.0,
            ambient_temperature_k=308.15,
            direction="forward",
        )

        self.assertEqual(result["supply_temperature_k"], old_tank_temperature)
        self.assertEqual(
            result["cluster_result"]["supply_temperature_k"], old_tank_temperature
        )
        self.assertEqual(result["tank_temperature_after_k"], loop.tank.temperature_k)
        self.assertNotEqual(result["tank_temperature_after_k"], old_tank_temperature)

    def test_total_and_pack_flows_come_from_pump_network_working_point(self) -> None:
        loop = build_loop()
        expected = loop.pump.solve_operating_point(
            3600.0, loop.cluster.hydraulic_network
        )

        result = loop.step(5.0, 560.0, 3600.0, 308.15)

        self.assertAlmostEqual(
            result["total_mass_flow_kg_s"],
            expected["total_mass_flow_kg_s"],
            places=12,
        )
        self.assertAlmostEqual(
            result["cluster_result"]["pack_mass_flows_kg_s"].sum(),
            result["total_mass_flow_kg_s"],
            places=12,
        )
        self.assertLess(abs(result["pressure_balance_residual_pa"]), 1e-6)

    def test_return_mixing_matches_independent_recalculation(self) -> None:
        loop = build_loop()
        result = loop.step(5.0, 560.0, 3600.0, 308.15)
        cluster = result["cluster_result"]
        expected_return = np.sum(
            cluster["pack_mass_flows_kg_s"]
            * cluster["pack_coolant_outlet_temperatures_k"]
        ) / cluster["pack_mass_flows_kg_s"].sum()

        self.assertAlmostEqual(result["return_temperature_k"], expected_return, places=12)

    def test_energy_diagnostics_and_states_are_finite(self) -> None:
        loop = build_loop()
        result = loop.step(5.0, 560.0, 3600.0, 308.15)

        scalar_keys = [
            "pump_delta_p_pa",
            "network_delta_p_pa",
            "pump_power_w",
            "fluid_heat_gain_w",
            "plate_to_fluid_heat_w",
            "plate_to_fluid_residual_w",
            "return_to_tank_heat_w",
            "tank_energy_change_j",
            "tank_energy_residual_j",
        ]
        self.assertTrue(np.all(np.isfinite([result[key] for key in scalar_keys])))
        self.assertLess(abs(result["plate_to_fluid_residual_w"]), 1e-8)
        self.assertLess(abs(result["tank_energy_residual_j"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
