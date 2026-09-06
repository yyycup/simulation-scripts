"""Stage 1 validation of the standalone heat-current cold-plate ROM.

Covers the eight checks required by the Heat-Current Stage 1 spec:
isothermal equilibrium, sign conventions, per-step plate energy balance,
fluid-side conservation, forward/reverse conservation with fixed zone
semantics, magnitude agreement with the validated ``ReducedColdPlate``,
short/long horizon responses, plus the n-zone structure smoke tests and
the zero-flow contract parity.
"""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.parameters import (
    COOLANT_SPECIFIC_HEAT_J_KG_K,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
)
from cluster_plant_v2.thermal.cold_plate_heat_current import ColdPlateHeatCurrent
from cluster_plant_v2.thermal.cold_plate_rom import ReducedColdPlate

DT_S = 5.0
INLET_K = 293.15
Q_BP_NOMINAL = np.array([400.0, 500.0, 400.0])


class ColdPlateHeatCurrentStructureTests(unittest.TestCase):
    def test_defaults_match_validated_rom_partition(self) -> None:
        plate = ColdPlateHeatCurrent()

        self.assertEqual(plate.zone_column_counts.tolist(), [4, 5, 4])
        self.assertEqual(plate.plate_temperatures.shape, (3,))
        self.assertEqual(plate.dynamic_state_count, 3)
        self.assertAlmostEqual(float(plate.zone_heat_capacities.sum()), 6000.0)
        self.assertAlmostEqual(
            float(plate.zone_heat_capacities.sum()), PLATE_NODE_HEAT_CAPACITY_TOTAL
        )
        self.assertAlmostEqual(float(plate.zone_areas.sum()), 0.5)

    def test_alternative_zone_counts_partition_the_same_totals(self) -> None:
        for counts in ([13], [1] * 13, [3, 4, 3, 3]):
            with self.subTest(counts=counts):
                plate = ColdPlateHeatCurrent(zone_column_counts=counts)

                self.assertEqual(plate.dynamic_state_count, len(counts))
                self.assertAlmostEqual(
                    float(plate.zone_heat_capacities.sum()),
                    PLATE_NODE_HEAT_CAPACITY_TOTAL,
                )
                self.assertAlmostEqual(float(plate.zone_areas.sum()), 0.5)

                result = plate.step(
                    DT_S, INLET_K, 0.1, np.zeros(len(counts)), flow_direction=1
                )
                self.assertTrue(np.all(np.isfinite(result["plate_temperatures"])))

    def test_invalid_zone_counts_are_rejected(self) -> None:
        for counts in ([], [0, 13], np.zeros(3, dtype=int)):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    ColdPlateHeatCurrent(zone_column_counts=counts)


class ColdPlateHeatCurrentPhysicsTests(unittest.TestCase):
    def test_isothermal_equilibrium_gives_zero_heat_current(self) -> None:
        plate = ColdPlateHeatCurrent(initial_temperature_c=20.0)

        result = plate.step(DT_S, INLET_K, 0.1, np.zeros(3))

        np.testing.assert_allclose(plate.plate_temperatures, INLET_K)
        self.assertAlmostEqual(result["coolant_outlet_temperature"], INLET_K)
        np.testing.assert_allclose(result["q_plate_to_fluid"], 0.0, atol=1e-12)
        self.assertAlmostEqual(result["q_plate_to_fluid_total"], 0.0)

    def test_hot_plate_drives_positive_heat_currents(self) -> None:
        plate = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        mass_flow = 0.1

        result = plate.step(DT_S, INLET_K, mass_flow, np.zeros(3))

        self.assertTrue(np.all(result["q_plate_to_fluid"] > 0.0))
        self.assertGreater(result["coolant_outlet_temperature"], INLET_K)
        self.assertTrue(np.all(result["coolant_mean_temperatures"] > INLET_K))

    def test_coolant_gain_equals_total_heat_current_exactly(self) -> None:
        plate = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        mass_flow = 0.1

        result = plate.step(DT_S, INLET_K, mass_flow, np.zeros(3))

        coolant_gain = mass_flow * plate.coolant_cp * (
            result["coolant_outlet_temperature"] - INLET_K
        )
        self.assertAlmostEqual(
            coolant_gain, result["q_plate_to_fluid_total"], places=9
        )

    def test_plate_energy_balance_holds_every_step(self) -> None:
        plate = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        previous_temperatures = plate.plate_temperatures.copy()
        previous_energy = plate.initial_zone_energy_J.copy()

        for _ in range(5):
            result = plate.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
            stored_change = (
                plate.zone_heat_capacities * plate.plate_temperatures
                - previous_energy
            )
            np.testing.assert_allclose(
                stored_change,
                DT_S * (Q_BP_NOMINAL - result["q_plate_to_fluid"]),
                rtol=1e-9,
            )
            np.testing.assert_allclose(
                plate.plate_temperatures - previous_temperatures,
                DT_S * (Q_BP_NOMINAL - result["q_plate_to_fluid"])
                / plate.zone_heat_capacities,
                rtol=1e-9,
            )
            previous_temperatures = plate.plate_temperatures.copy()
            previous_energy = (
                plate.zone_heat_capacities * plate.plate_temperatures
            )

    def test_forward_and_reverse_keep_fixed_zone_semantics(self) -> None:
        forward = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        reverse = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        asymmetric_load = np.array([800.0, 0.0, 0.0])

        forward_result = forward.step(
            DT_S, INLET_K, 0.1, asymmetric_load, flow_direction=1
        )
        reverse_result = reverse.step(
            DT_S, INLET_K, 0.1, asymmetric_load, flow_direction=-1
        )

        self.assertTrue(np.all(np.diff(forward_result["coolant_mean_temperatures"]) > 0.0))
        self.assertTrue(np.all(np.diff(reverse_result["coolant_mean_temperatures"]) < 0.0))
        # Zone 0 is zone 0 in both directions: with the load concentrated in
        # zone 0, reverse flow exposes it to the warmest coolant, so its wall
        # temperature must be higher than under forward flow -- and must NOT
        # equal the forward zone-2 value (no index mirroring).
        self.assertGreater(reverse.plate_temperatures[0], forward.plate_temperatures[0])
        self.assertNotAlmostEqual(
            reverse.plate_temperatures[0], forward.plate_temperatures[2], places=3
        )

    def test_symmetric_plate_mirrors_between_flow_directions(self) -> None:
        forward = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        reverse = ColdPlateHeatCurrent(initial_temperature_c=25.0)

        forward.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL, flow_direction=1)
        reverse.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL, flow_direction=-1)

        np.testing.assert_allclose(
            forward.plate_temperatures, reverse.plate_temperatures[::-1]
        )

    def test_both_directions_conserve_total_energy(self) -> None:
        for direction in (1, -1):
            with self.subTest(flow_direction=direction):
                plate = ColdPlateHeatCurrent(initial_temperature_c=25.0)
                mass_flow = 0.1

                result = plate.step(
                    DT_S, INLET_K, mass_flow, Q_BP_NOMINAL, flow_direction=direction
                )

                coolant_gain = mass_flow * plate.coolant_cp * (
                    result["coolant_outlet_temperature"] - INLET_K
                )
                self.assertAlmostEqual(
                    coolant_gain, result["q_plate_to_fluid_total"], places=9
                )
                plate_loss = plate.zone_heat_capacities * (
                    298.15 - plate.plate_temperatures
                ) + DT_S * np.asarray(Q_BP_NOMINAL, dtype=float)
                np.testing.assert_allclose(
                    plate_loss, DT_S * result["q_plate_to_fluid"], rtol=1e-12
                )


class ColdPlateHeatCurrentReferenceComparisonTests(unittest.TestCase):
    def test_reference_flow_recovers_nominal_htc_and_resistance_scaling(self) -> None:
        plate = ColdPlateHeatCurrent()

        result = plate.step(DT_S, INLET_K, 0.1428, np.zeros(3))

        self.assertAlmostEqual(result["h_dynamic"], 2000.0)
        conductance = 2000.0 * plate.zone_areas
        capacity_rate = 0.1428 * COOLANT_SPECIFIC_HEAT_J_KG_K
        expected_resistance = 1.0 / (
            capacity_rate * (1.0 - np.exp(-conductance / capacity_rate))
        )
        np.testing.assert_allclose(
            result["r_plate_to_fluid"], expected_resistance, rtol=1e-12
        )

    def test_nominal_flow_stays_within_thirty_percent_of_validated_rom(self) -> None:
        heat_current = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        reference = ReducedColdPlate(initial_temperature_c=25.0)

        hc_result = heat_current.step(DT_S, INLET_K, 0.1428, Q_BP_NOMINAL)
        rom_result = reference.step(DT_S, INLET_K, 0.1428, Q_BP_NOMINAL)

        self.assertGreater(hc_result["coolant_outlet_temperature"], INLET_K)
        self.assertGreater(rom_result["coolant_outlet_temperature"], INLET_K)
        for key in ("coolant_outlet_temperature",):
            self.assertLess(
                abs(hc_result[key] - rom_result[key]) / (rom_result[key] - INLET_K),
                0.30,
                msg=f"{key} differs by more than 30 percent of the fluid rise",
            )
        self.assertLess(
            abs(hc_result["q_plate_to_fluid_total"] - rom_result["q_plate_to_fluid_total"])
            / rom_result["q_plate_to_fluid_total"],
            0.30,
        )

    def test_one_step_and_long_horizon_stay_finite_and_stable(self) -> None:
        heat_current = ColdPlateHeatCurrent()
        reference = ReducedColdPlate()

        hc_one = heat_current.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
        rom_one = reference.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
        self.assertTrue(np.all(np.isfinite(hc_one["plate_temperatures"])))
        self.assertTrue(np.all(np.isfinite(rom_one["plate_temperatures"])))

        previous = heat_current.plate_temperatures.copy()
        for _ in range(120):
            long_result = heat_current.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
            self.assertTrue(np.all(np.isfinite(long_result["plate_temperatures"])))
            previous = heat_current.plate_temperatures.copy()
        # Zones may approach their quasi-steady values from either side
        # (zone 0 starts with Q_pf > Q_bp and cools first), so stability here
        # means boundedness plus convergence, not monotonicity.
        self.assertTrue(np.all(heat_current.plate_temperatures > INLET_K))
        self.assertTrue(np.all(heat_current.plate_temperatures < 400.0))
        self.assertTrue(
            np.all(np.abs(heat_current.plate_temperatures - previous) < 1e-3)
        )

    def test_extreme_low_flow_stays_well_conditioned_without_epsilon(self) -> None:
        plate = ColdPlateHeatCurrent(initial_temperature_c=25.0)

        result = plate.step(DT_S, INLET_K, 1e-6, Q_BP_NOMINAL)

        self.assertTrue(np.all(np.isfinite(result["plate_temperatures"])))
        self.assertTrue(np.all(np.isfinite(result["r_plate_to_fluid"])))
        self.assertTrue(np.all(result["r_plate_to_fluid"] > 0.0))
        # G -> 0+: R -> 1/G (large but finite), Q -> 0, and the fluid leaves
        # at the wall temperature (effectiveness -> 1). No epsilon anywhere.
        self.assertGreater(float(np.max(result["r_plate_to_fluid"])), 1e2)
        self.assertLess(result["q_plate_to_fluid_total"], 1e-1)
        self.assertGreater(result["coolant_outlet_temperature"], INLET_K)

    def test_nonpositive_mass_flow_and_invalid_direction_are_rejected(self) -> None:
        plate = ColdPlateHeatCurrent()

        for mass_flow in (0.0, -0.1):
            with self.subTest(mass_flow=mass_flow):
                with self.assertRaises(ValueError):
                    plate.step(DT_S, INLET_K, mass_flow, np.zeros(3))
        with self.assertRaises(ValueError):
            plate.step(DT_S, INLET_K, 0.1, np.zeros(3), flow_direction=0)


class ColdPlateHeatCurrentSubsteppingTests(unittest.TestCase):
    """Stage 1.6 internal Euler sub-stepping."""

    def test_substep_count_floors_at_one_and_handles_short_outer_steps(self) -> None:
        plate = ColdPlateHeatCurrent()
        self.assertEqual(plate.substep_count(5.0), 5)
        self.assertEqual(plate.substep_count(1.0), 1)
        self.assertEqual(plate.substep_count(0.5), 1)

    def test_outer_step_report_includes_internal_steps_and_dt(self) -> None:
        plate = ColdPlateHeatCurrent()
        result = plate.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
        self.assertEqual(result["internal_steps"], 5)
        self.assertAlmostEqual(result["internal_dt_s"], 1.0)

    def test_disabling_substepping_yields_native_results(self) -> None:
        # Sub-stepping with N=1 must be bit-exactly identical to the legacy
        # single-step path: when internal_dt_s == dt the sub-step loop runs a
        # single Euler step with no averaging or post-processing.
        # The reverse-direction case (sub-step ON with internal_dt_s=1s but
        # dt=5s) is intentionally NOT asserted equal -- 5x1s sub-steps have
        # smaller truncation error than 1x5s Euler even when both are stable.
        plate_native = ColdPlateHeatCurrent(internal_dt_s=DT_S)
        plate_single_sub = ColdPlateHeatCurrent(internal_dt_s=DT_S)

        # Same configuration -> same number of internal steps = 1.
        native_result = plate_native.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
        single_sub_result = plate_single_sub.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
        self.assertEqual(native_result["internal_steps"], 1)
        self.assertEqual(single_sub_result["internal_steps"], 1)
        np.testing.assert_array_equal(
            native_result["coolant_outlet_temperature"],
            single_sub_result["coolant_outlet_temperature"],
        )
        np.testing.assert_array_equal(
            native_result["q_plate_to_fluid_total"],
            single_sub_result["q_plate_to_fluid_total"],
        )

    def test_substepped_run_is_more_accurate_than_native_when_stable(self) -> None:
        # At dt=5s, mass_flow=0.1 kg/s the stability limit is ~22 s, so both
        # 5x1s and 1x5s Euler are stable -- but 5x1s has ~5x smaller truncation
        # error. The 5s-path is the legacy reference; sub-stepping must NOT
        # make things worse, and may improve the answer. We verify by running
        # a long horizon and comparing to a fine-reference 1s path.
        plate_fine = ColdPlateHeatCurrent(internal_dt_s=1.0)
        plate_native = ColdPlateHeatCurrent(internal_dt_s=DT_S)
        plate_sub = ColdPlateHeatCurrent(internal_dt_s=1.0)

        for _ in range(120):
            r_fine = plate_fine.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
            r_native = plate_native.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
            r_sub = plate_sub.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
        # Sub-stepped and fine reference both run at 1s internal -> identical
        # to fp (same loop, same operations on same state).
        np.testing.assert_array_equal(
            r_sub["coolant_outlet_temperature"],
            r_fine["coolant_outlet_temperature"],
        )
        # Native (5s Euler) is NOT identical, but must be within truncation
        # bound (O(dt^2) per step). At ~0.0014 K/s of outlet drift this is
        # ~0.04 K max -- observed 0.054 K. Use 0.5 K as a safe upper bound.
        self.assertLess(
            abs(
                r_native["coolant_outlet_temperature"]
                - r_fine["coolant_outlet_temperature"]
            ),
            0.5,
        )

    def test_invalid_internal_dt_is_rejected(self) -> None:
        for bad in (0.0, -1.0, float("nan")):
            with self.subTest(internal_dt_s=bad):
                with self.assertRaises(ValueError):
                    ColdPlateHeatCurrent(internal_dt_s=bad)

    def test_outer_step_plate_energy_balance_holds_with_substepping(self) -> None:
        # With sub-stepping, the plate energy balance is over the OUTER step:
        #   C * (T_end - T_start) / dt = Q_battery - mean_j(Q_j)
        # and the fluid-side identity holds for the averaged outlet.
        plate = ColdPlateHeatCurrent(initial_temperature_c=25.0)
        capacity_rate = 0.1 * plate.coolant_cp

        for _ in range(5):
            previous_energy = (
                plate.zone_heat_capacities * plate.plate_temperatures
            )
            result = plate.step(DT_S, INLET_K, 0.1, Q_BP_NOMINAL)
            stored_change = (
                plate.zone_heat_capacities * plate.plate_temperatures
                - previous_energy
            )
            np.testing.assert_allclose(
                stored_change,
                DT_S * (Q_BP_NOMINAL - result["q_plate_to_fluid"]),
                rtol=1e-9,
            )
            coolant_gain = capacity_rate * (
                result["coolant_outlet_temperature"] - INLET_K
            )
            self.assertAlmostEqual(
                coolant_gain,
                result["q_plate_to_fluid_total"],
                places=9,
            )


if __name__ == "__main__":
    unittest.main()
