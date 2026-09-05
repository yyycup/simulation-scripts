"""Contract tests for the Stage 7B parallel-header hydraulic network."""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.hydraulics import (
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    DESIGN_PACK_MASS_FLOW_KG_S,
    ParallelHeaderHydraulicNetwork,
)


TOTAL_MASS_FLOW_KG_S = 5.0 * DESIGN_PACK_MASS_FLOW_KG_S


class ParallelHeaderHydraulicNetworkTests(unittest.TestCase):
    def test_structure_is_algebraic_and_parameterized_by_pack_count(self) -> None:
        network = ParallelHeaderHydraulicNetwork(n_packs=3)

        self.assertEqual(network.n_packs, 3)
        self.assertEqual(network.dynamic_state_count, 0)
        self.assertEqual(network.branch_resistances.shape, (3,))
        self.assertEqual(network.supply_segment_resistances.shape, (3,))
        self.assertEqual(network.return_segment_resistances.shape, (3,))

    def test_invalid_counts_and_resistances_are_rejected(self) -> None:
        for n_packs in (0, -1, 1.5, True):
            with self.subTest(n_packs=n_packs):
                with self.assertRaises(ValueError):
                    ParallelHeaderHydraulicNetwork(n_packs=n_packs)
        with self.assertRaises(ValueError):
            ParallelHeaderHydraulicNetwork(
                n_packs=2, branch_resistances=[1.0, 0.0]
            )
        with self.assertRaises(ValueError):
            ParallelHeaderHydraulicNetwork(
                n_packs=2, supply_segment_resistances=[0.0, -1.0]
            )
        with self.assertRaises(ValueError):
            ParallelHeaderHydraulicNetwork(
                n_packs=2, return_segment_resistances=[0.0, np.nan]
            )

    def test_zero_total_flow_has_explicit_zero_algebraic_solution(self) -> None:
        result = ParallelHeaderHydraulicNetwork().solve(0.0)

        for key in (
            "pack_mass_flows",
            "supply_segment_flows",
            "return_segment_flows",
            "supply_segment_delta_p",
            "return_segment_delta_p",
            "branch_delta_p",
            "pack_path_delta_p",
            "node_mass_balance_residuals",
            "path_pressure_residuals",
        ):
            np.testing.assert_array_equal(result[key], 0.0)
        self.assertEqual(result["network_delta_p"], 0.0)
        self.assertTrue(result["solver_success"])
        self.assertEqual(result["solver_iterations"], 0)

    def test_zero_headers_and_equal_branches_recover_uniform_stage_7a_flow(self) -> None:
        network = ParallelHeaderHydraulicNetwork(
            supply_segment_resistances=np.zeros(5),
            return_segment_resistances=np.zeros(5),
        )

        result = network.solve(TOTAL_MASS_FLOW_KG_S)

        np.testing.assert_allclose(
            result["pack_mass_flows"],
            DESIGN_PACK_MASS_FLOW_KG_S,
            rtol=0.0,
            atol=1e-13,
        )
        self.assertLess(abs(result["total_mass_balance_residual"]), 1e-13)
        self.assertLess(np.max(np.abs(result["path_pressure_residuals"])), 1e-6)

    def test_zero_headers_and_unequal_branches_recover_stage_7a_analytic_solution(self) -> None:
        factors = np.array([1.00, 1.05, 0.95, 1.10, 0.90])
        network = ParallelHeaderHydraulicNetwork(
            branch_resistances=BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 * factors,
            supply_segment_resistances=np.zeros(5),
            return_segment_resistances=np.zeros(5),
        )
        expected = TOTAL_MASS_FLOW_KG_S * (1.0 / np.sqrt(factors)) / np.sum(
            1.0 / np.sqrt(factors)
        )

        result = network.solve(TOTAL_MASS_FLOW_KG_S)

        np.testing.assert_allclose(
            result["pack_mass_flows"], expected, rtol=0.0, atol=1e-10
        )
        self.assertLess(np.max(np.abs(result["path_pressure_residuals"])), 1e-5)

    def test_reversed_return_segment_flows_and_node_balances_follow_declared_topology(self) -> None:
        ratio = 0.001
        header_k = BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 * ratio
        network = ParallelHeaderHydraulicNetwork(
            supply_segment_resistances=np.full(5, header_k),
            return_segment_resistances=np.full(5, header_k),
        )

        result = network.solve(TOTAL_MASS_FLOW_KG_S)
        flows = result["pack_mass_flows"]

        np.testing.assert_allclose(
            result["supply_segment_flows"],
            [flows[index:].sum() for index in range(5)],
            rtol=0.0,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            result["return_segment_flows"],
            [flows[: index + 1].sum() for index in range(5)],
            rtol=0.0,
            atol=1e-13,
        )
        self.assertLess(
            np.max(np.abs(result["node_mass_balance_residuals"])), 1e-13
        )
        self.assertGreater(np.ptp(flows), 0.0)
        self.assertLess(np.max(np.abs(result["path_pressure_residuals"])), 1e-5)
        self.assertTrue(result["solver_success"])


if __name__ == "__main__":
    unittest.main()
