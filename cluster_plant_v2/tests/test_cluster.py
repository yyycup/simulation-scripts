"""Contract tests for the Stage 7A reduced cluster assembly."""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.hydraulics import (
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.thermal.reduced_pack import ReducedPack


PACK_FLOW_5_LPM_KG_S = 5.0 / 1000.0 / 60.0 * 1071.0
TOTAL_FLOW_25_LPM_KG_S = 5.0 * PACK_FLOW_5_LPM_KG_S


class ReducedClusterTests(unittest.TestCase):
    def test_default_cluster_contains_five_independent_reduced_packs(self) -> None:
        cluster = ReducedCluster()

        self.assertEqual(cluster.n_packs, 5)
        self.assertEqual(len(cluster.packs), 5)
        self.assertEqual(cluster.dynamic_state_count, 5 * 43)
        self.assertEqual(len({id(pack) for pack in cluster.packs}), 5)

    def test_invalid_pack_count_and_resistance_factors_are_rejected(self) -> None:
        for n_packs in (0, -1, 1.5, True):
            with self.subTest(n_packs=n_packs):
                with self.assertRaises(ValueError):
                    ReducedCluster(n_packs=n_packs)
        for factors in ([1.0, 0.0], [1.0, -1.0], [1.0, np.nan]):
            with self.subTest(factors=factors):
                with self.assertRaises(ValueError):
                    ReducedCluster(n_packs=2, branch_resistance_factors=factors)

    def test_equal_resistance_gives_uniform_machine_precision_flow_conservation(self) -> None:
        cluster = ReducedCluster()

        flows = cluster.allocate_mass_flows(TOTAL_FLOW_25_LPM_KG_S)

        np.testing.assert_allclose(flows, PACK_FLOW_5_LPM_KG_S, rtol=0.0, atol=1e-15)
        self.assertLessEqual(abs(float(flows.sum()) - TOTAL_FLOW_25_LPM_KG_S), 1e-15)

    def test_nonuniform_resistance_uses_inverse_square_root_flow_order(self) -> None:
        factors = np.array([1.00, 1.05, 0.95, 1.10, 0.90])
        cluster = ReducedCluster(branch_resistance_factors=factors)

        flows = cluster.allocate_mass_flows(TOTAL_FLOW_25_LPM_KG_S)

        expected = TOTAL_FLOW_25_LPM_KG_S * (1.0 / np.sqrt(factors)) / np.sum(
            1.0 / np.sqrt(factors)
        )
        np.testing.assert_allclose(flows, expected, rtol=0.0, atol=1e-15)
        self.assertLess(flows[np.argmax(factors)], flows[np.argmin(factors)])

    def test_zero_flow_allocation_and_mixing_have_explicit_boundary_behavior(self) -> None:
        cluster = ReducedCluster()

        flows = cluster.allocate_mass_flows(0.0)
        mixed = cluster.mix_return_temperature(
            np.full(5, 300.0), flows, supply_temperature_k=293.15
        )

        np.testing.assert_array_equal(flows, 0.0)
        self.assertEqual(mixed, 293.15)
        with self.assertRaisesRegex(ValueError, "positive total coolant mass flow"):
            cluster.step(5.0, 560.0, 293.15, 0.0, 308.15)

    def test_symmetric_step_broadcasts_current_and_matches_standalone_pack(self) -> None:
        cluster = ReducedCluster()
        standalone = ReducedPack()

        cluster_result = cluster.step(
            5.0,
            560.0,
            293.15,
            TOTAL_FLOW_25_LPM_KG_S,
            308.15,
            direction="forward",
        )
        standalone_result = standalone.step(
            5.0, 560.0, 293.15, PACK_FLOW_5_LPM_KG_S, 308.15, 1
        )

        np.testing.assert_array_equal(cluster_result["pack_currents_a"], 560.0)
        for values in (
            cluster_result["pack_battery_average_temperatures_k"],
            cluster_result["pack_coolant_outlet_temperatures_k"],
            cluster_result["pack_q_gen_total_w"],
            cluster_result["pack_q_battery_to_plate_total_w"],
        ):
            np.testing.assert_array_equal(values, values[0])
        np.testing.assert_array_equal(
            cluster_result["pack_battery_zone_temperatures_k"][0],
            standalone_result["battery_zone_temperatures"],
        )
        np.testing.assert_array_equal(
            cluster_result["pack_plate_temperatures_k"][0],
            standalone_result["plate_temperatures"],
        )
        self.assertAlmostEqual(
            cluster_result["return_temperature_k"],
            standalone_result["coolant_outlet_temperature"],
            places=12,
        )
        self.assertEqual(cluster_result["inter_pack_delta_temperature_k"], 0.0)
        self.assertLessEqual(cluster_result["flow_nonuniformity"], 1e-15)

    def test_return_temperature_is_mass_weighted_for_nonuniform_flow(self) -> None:
        cluster = ReducedCluster(
            branch_resistance_factors=[1.00, 1.05, 0.95, 1.10, 0.90]
        )

        result = cluster.step(
            5.0,
            560.0,
            293.15,
            TOTAL_FLOW_25_LPM_KG_S,
            308.15,
        )

        expected = np.sum(
            result["pack_mass_flows_kg_s"]
            * result["pack_coolant_outlet_temperatures_k"]
        ) / TOTAL_FLOW_25_LPM_KG_S
        self.assertAlmostEqual(result["return_temperature_k"], expected, places=12)
        self.assertLessEqual(abs(result["mass_flow_conservation_residual_kg_s"]), 1e-15)

    def test_forward_and_reverse_steps_are_finite(self) -> None:
        for direction in ("forward", "reverse"):
            with self.subTest(direction=direction):
                result = ReducedCluster().step(
                    5.0,
                    560.0,
                    293.15,
                    TOTAL_FLOW_25_LPM_KG_S,
                    308.15,
                    direction=direction,
                )
                for value in result.values():
                    if isinstance(value, np.ndarray):
                        self.assertTrue(np.all(np.isfinite(value)))
                    elif isinstance(value, (int, float, np.number)):
                        self.assertTrue(np.isfinite(value))

    def test_invalid_hydraulic_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReducedCluster(hydraulic_mode="unknown")

    def test_zero_header_network_recovers_lumped_uniform_flow(self) -> None:
        network = ParallelHeaderHydraulicNetwork(
            supply_segment_resistances=np.zeros(5),
            return_segment_resistances=np.zeros(5),
        )
        cluster = ReducedCluster(
            hydraulic_mode="header_network", hydraulic_network=network
        )

        result = cluster.step(
            5.0, 560.0, 293.15, TOTAL_FLOW_25_LPM_KG_S, 308.15
        )

        np.testing.assert_allclose(
            result["pack_mass_flows_kg_s"],
            PACK_FLOW_5_LPM_KG_S,
            rtol=0.0,
            atol=1e-13,
        )
        self.assertIsNotNone(cluster.last_hydraulic_result)
        self.assertLess(
            np.max(
                np.abs(cluster.last_hydraulic_result["path_pressure_residuals"])
            ),
            1e-6,
        )

    def test_zero_header_network_recovers_lumped_nonuniform_flow(self) -> None:
        factors = np.array([1.00, 1.05, 0.95, 1.10, 0.90])
        network = ParallelHeaderHydraulicNetwork(
            branch_resistances=BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 * factors,
            supply_segment_resistances=np.zeros(5),
            return_segment_resistances=np.zeros(5),
        )
        header_cluster = ReducedCluster(
            hydraulic_mode="header_network", hydraulic_network=network
        )
        lumped_cluster = ReducedCluster(branch_resistance_factors=factors)

        header = header_cluster.step(
            5.0, 560.0, 293.15, TOTAL_FLOW_25_LPM_KG_S, 308.15
        )
        lumped = lumped_cluster.step(
            5.0, 560.0, 293.15, TOTAL_FLOW_25_LPM_KG_S, 308.15
        )

        np.testing.assert_allclose(
            header["pack_mass_flows_kg_s"],
            lumped["pack_mass_flows_kg_s"],
            rtol=0.0,
            atol=1e-10,
        )

    def test_cold_plate_direction_does_not_reverse_header_hydraulics(self) -> None:
        header_k = BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 * 0.005

        def make_cluster() -> ReducedCluster:
            return ReducedCluster(
                hydraulic_mode="header_network",
                hydraulic_network=ParallelHeaderHydraulicNetwork(
                    supply_segment_resistances=np.full(5, header_k),
                    return_segment_resistances=np.full(5, header_k),
                ),
            )

        forward = make_cluster().step(
            5.0, 560.0, 293.15, TOTAL_FLOW_25_LPM_KG_S, 308.15, "forward"
        )
        reverse = make_cluster().step(
            5.0, 560.0, 293.15, TOTAL_FLOW_25_LPM_KG_S, 308.15, "reverse"
        )

        np.testing.assert_array_equal(
            forward["pack_mass_flows_kg_s"], reverse["pack_mass_flows_kg_s"]
        )


if __name__ == "__main__":
    unittest.main()
