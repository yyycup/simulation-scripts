"""Regression contracts for physical storage, state seeding and HC routing."""
import copy
import unittest
from dataclasses import replace

import numpy as np

from cluster_plant_v2.tests.test_heat_current_plant_independence import (
    _build_legacy_plant, _constant_inputs,
)
from cluster_plant_v2.thermal.evaporator_heat_current import EvaporatorHeatCurrent
from cluster_plant_v2.thermal.heat_current_plant import (
    HeatCurrentPlant, build_independent_hc_plant,
)
from cluster_plant_v2.thermal.heat_current_energy_balance import (
    initial_energy_snapshot, ledger_from_legacy,
)
from cluster_plant_v2.thermal.heat_current_stage5_ledger import ledger_from_heat_current_plant
from cluster_plant_v2.refrigeration import CoolantTransportDelay


def snapshot(plant, hc=True, flow=None):
    if flow is None:
        flow = plant.pump.solve_operating_point(3600.0, plant.hydraulic_network)[
            "total_mass_flow_kg_s"
        ]
    return initial_energy_snapshot(
        plant, is_heat_current=hc, transport_reference_mass_flow_kg_s=flow,
    )


def hc_ledger(plant, result, prev):
    return ledger_from_heat_current_plant(
        plant, dt_s=5.0, step_result=result, prev_energy=prev,
        compressor_speed_used_rpm=result["compressor_speed_used_rpm"],
        tank_temperature_before_k=result["tank_temperature_before_k"],
        refrigeration_solver_success=result["refrigeration_solver_success"],
    )


class HeatCurrentClosureRepairsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = _build_legacy_plant()

    def test_running_state_and_configuration_are_copied_without_aliases(self):
        legacy = copy.deepcopy(self.seed)
        for index, pack in enumerate(legacy.cluster.packs):
            pack.battery.config["capacity"] = 250.0 + index
            pack.battery.temps += index + 1.0
            pack.battery.soc_branch[:] = 0.6 + index * 0.01
            pack.battery.eta1[:] = -0.1 - index * 0.01
            pack.battery.eta2[:] = -0.2 - index * 0.01
            pack.cold_plate.plate_temperatures += index + 2.0
        hc = build_independent_hc_plant(legacy_plant=legacy)
        for source, target in zip(legacy.cluster.packs, hc.cluster.packs):
            self.assertEqual(source.battery.config, target.battery.config)
            for name in ("temps", "soc_branch", "eta1", "eta2"):
                np.testing.assert_array_equal(getattr(source.battery, name), getattr(target.battery, name))
                self.assertFalse(np.shares_memory(getattr(source.battery, name), getattr(target.battery, name)))
            np.testing.assert_array_equal(source.cold_plate.plate_temperatures, target.cold_plate.plate_temperatures)
            self.assertFalse(np.shares_memory(source.cold_plate.plate_temperatures, target.cold_plate.plate_temperatures))

    def test_full_delay_history_and_evaporator_buffer_are_copied(self):
        legacy = copy.deepcopy(self.seed)
        legacy.supply_delay.step(291.0)
        legacy.return_delay.step(302.0)
        legacy.evaporator_dynamics.evaporator_buffer_energy_j = 12345.0
        hc = build_independent_hc_plant(legacy_plant=legacy)
        self.assertEqual(hc.supply_delay.queue_values, legacy.supply_delay.queue_values)
        self.assertEqual(hc.return_delay.queue_values, legacy.return_delay.queue_values)
        self.assertEqual(hc.evaporator_dynamics.evaporator_buffer_energy_j, 12345.0)

    def test_equilibrium_construction_does_not_advance_battery_or_plate(self):
        hc = build_independent_hc_plant(legacy_plant=self.seed)
        before = copy.deepcopy(hc.cluster)
        HeatCurrentPlant.from_equilibrium(
            cluster=hc.cluster, hydraulic_network=hc.hydraulic_network,
            pump=hc.pump, tank=hc.tank, refrigeration_cycle=hc.refrigeration_cycle,
            compressor_actuator=hc.compressor_actuator, dt_s=5.0,
            pump_speed_rpm=3600.0, fan_speed_rpm=1200.0,
            initial_cluster_current_a=560.0, ambient_temperature_k=308.15,
        )
        for a, b in zip(before.packs, hc.cluster.packs):
            np.testing.assert_array_equal(a.battery.temps, b.battery.temps)
            np.testing.assert_array_equal(a.battery.soc_branch, b.battery.soc_branch)
            np.testing.assert_array_equal(a.cold_plate.plate_temperatures, b.cold_plate.plate_temperatures)

    def test_injected_evaporator_is_used_and_receives_applied_heat(self):
        class RecordingEvaporator(EvaporatorHeatCurrent):
            def outlet_temperature_from_applied_heat(self, **kwargs):
                self.received = kwargs
                return super().outlet_temperature_from_applied_heat(**kwargs)
        hc = build_independent_hc_plant(legacy_plant=self.seed)
        exchanger = RecordingEvaporator()
        hc.evaporator_heat_current = exchanger
        result = hc.step(replace(_constant_inputs(), compressor_command_rpm=0.0), dt_s=5.0)
        self.assertTrue(hasattr(exchanger, "received"), "injected exchanger was bypassed")
        self.assertEqual(exchanger.received["q_applied_w"], result["q_evap_applied_w"])
        self.assertAlmostEqual(result["q_evap_from_coolant_w"], result["q_evap_applied_w"], places=8)

    def test_delay_energy_has_mass_and_is_invariant_to_time_discretization(self):
        hc = build_independent_hc_plant(legacy_plant=self.seed)
        for dt in (1.0, 5.0):
            hc.supply_delay = CoolantTransportDelay(delay_s=15.0, dt_s=dt, initial_value=300.0)
            hc.return_delay = CoolantTransportDelay(delay_s=20.0, dt_s=dt, initial_value=310.0)
            energy = snapshot(hc, flow=0.4)
            cp = hc.tank.coolant_specific_heat_j_kg_k
            self.assertAlmostEqual(energy["supply_delay_J"], 0.4 * 15.0 * cp * 300.0)
            self.assertAlmostEqual(energy["return_delay_J"], 0.4 * 20.0 * cp * 310.0)

    def test_constant_flow_full_loop_closes_for_both_plants(self):
        for is_hc in (False, True):
            plant = (build_independent_hc_plant(legacy_plant=self.seed) if is_hc else copy.deepcopy(self.seed))
            prev = snapshot(plant, hc=is_hc)
            for _ in range(4):
                if is_hc:
                    result = plant.step(_constant_inputs(), dt_s=5.0)
                    ledger, prev = hc_ledger(plant, result, prev)
                else:
                    result = plant.step(dt_s=5.0, cluster_current_a=560.0, pump_speed_rpm=3600.0,
                        compressor_speed_command_rpm=4000.0, fan_speed_rpm=1200.0,
                        ambient_temperature_k=308.15, direction="forward")
                    ledger, prev = ledger_from_legacy(plant, dt_s=5.0, result=result, prev_energy=prev)
                self.assertLess(abs(ledger.residual_system_w), 1e-6)
                self.assertLess(abs(ledger.residual_loop_implicit_transport_w), 1e-6)

    def test_flow_step_keeps_fixed_mass_and_reports_delay_model_mismatch(self):
        hc = build_independent_hc_plant(legacy_plant=self.seed)
        prev = snapshot(hc)
        reference_flow = hc.pump.solve_operating_point(3600.0, hc.hydraulic_network)["total_mass_flow_kg_s"]
        result = hc.step(_constant_inputs(pump_rpm=4500.0), dt_s=5.0)
        ledger, after = hc_ledger(hc, result, prev)
        cp = hc.tank.coolant_specific_heat_j_kg_k
        expected = (result["total_mass_flow_kg_s"] - reference_flow) * cp * (
            result["evaporator_outlet_temperature_k"] - result["cluster_supply_temperature_k"]
            + result["cluster_return_temperature_k"] - result["tank_return_temperature_k"]
        )
        self.assertGreater(abs(expected), 1.0)
        self.assertAlmostEqual(ledger.transport_flow_mismatch_w, expected, places=6)
        self.assertAlmostEqual(ledger.residual_system_w, expected, places=6)
        self.assertAlmostEqual(after["supply_delay_J"], reference_flow * 5.0 * cp * sum(hc.supply_delay.queue_values))


if __name__ == "__main__":
    unittest.main()
