"""Conservation and travel-time contracts for a fixed coolant inventory."""
import unittest
from dataclasses import replace

import numpy as np

from cluster_plant_v2.refrigeration import CoolantTransportDelay
try:
    from cluster_plant_v2.thermal.coolant_mass_transport import CoolantMassTransport
except ImportError:
    CoolantMassTransport = None


class CoolantMassTransportTests(unittest.TestCase):
    def make_pipe(self, temperatures=(300.0, 310.0), mass=10.0, dt=5.0):
        self.assertIsNotNone(CoolantMassTransport, "mass-conserving pipe is missing")
        return CoolantMassTransport(
            mass_kg=mass, dt_s=dt, initial_temperatures_k=temperatures,
            coolant_specific_heat_j_kg_k=3391.0,
        )

    def assert_step_conserves(self, pipe, inlet, flow):
        mass_before = pipe.stored_mass_kg
        energy_before = pipe.stored_energy_j
        outlet = pipe.step(inlet, mass_flow_kg_s=flow)
        self.assertAlmostEqual(pipe.stored_mass_kg, mass_before, delta=1e-11)
        self.assertAlmostEqual(
            pipe.stored_energy_j - energy_before,
            flow * pipe.dt_s * pipe.coolant_specific_heat_j_kg_k * (inlet - outlet),
            delta=1e-7,
        )
        return outlet

    def test_partial_parcel_withdrawal_returns_mass_weighted_outlet(self):
        pipe = self.make_pipe()
        outlet = self.assert_step_conserves(pipe, 290.0, 1.5)
        self.assertAlmostEqual(outlet, (5.0 * 300.0 + 2.5 * 310.0) / 7.5)
        np.testing.assert_allclose(pipe.parcel_masses_kg, (2.5, 7.5))
        self.assertEqual(pipe.queue_values, (310.0, 290.0))

    def test_more_than_one_pipe_volume_can_pass_in_one_step(self):
        pipe = self.make_pipe()
        outlet = self.assert_step_conserves(pipe, 290.0, 3.0)
        self.assertAlmostEqual(outlet, (5*300 + 5*310 + 5*290) / 15)
        self.assertEqual(pipe.queue_values, (290.0,))
        self.assertAlmostEqual(pipe.stored_mass_kg, 10.0)

    def test_zero_flow_preserves_all_state(self):
        pipe = self.make_pipe()
        before = (pipe.parcel_masses_kg, pipe.queue_values, pipe.stored_energy_j)
        self.assertEqual(self.assert_step_conserves(pipe, 280.0, 0.0), 300.0)
        self.assertEqual(before, (pipe.parcel_masses_kg, pipe.queue_values, pipe.stored_energy_j))

    def test_reference_flow_reproduces_fixed_fifo_for_nonuniform_history(self):
        self.make_pipe()
        delay = CoolantTransportDelay(delay_s=15.0, dt_s=5.0,
            initial_queue_values=(300.0, 310.0, 320.0))
        pipe = CoolantMassTransport.from_fixed_delay(delay, reference_mass_flow_kg_s=0.43,
            coolant_specific_heat_j_kg_k=3391.0)
        for inlet in (290.0, 295.0, 303.0, 312.0, 298.0, 287.0):
            self.assertAlmostEqual(self.assert_step_conserves(pipe, inlet, 0.43), delay.step(inlet), places=11)

    def test_temperature_front_tracks_cumulative_transported_mass(self):
        pipe = self.make_pipe((300.0,), mass=10.0, dt=1.0)
        outputs = [self.assert_step_conserves(pipe, 280.0, flow) for flow in (1.0, 2.0, 3.0, 5.0)]
        np.testing.assert_allclose(outputs, (300.0, 300.0, 300.0, 296.0))

    def test_subdividing_a_held_input_preserves_enthalpy_and_storage(self):
        full = self.make_pipe(dt=5.0)
        split = self.make_pipe(dt=1.0)
        outlet = self.assert_step_conserves(full, 290.0, 1.3)
        outlets = [self.assert_step_conserves(split, 290.0, 1.3) for _ in range(5)]
        self.assertAlmostEqual(outlet, np.mean(outlets), places=11)
        self.assertAlmostEqual(full.stored_energy_j, split.stored_energy_j, delta=1e-7)

    def test_repeated_variable_flow_preserves_mass_energy_and_temperature_bounds(self):
        pipe = self.make_pipe()
        rng = np.random.default_rng(20260908)
        for inlet, flow in zip(rng.uniform(280.0, 320.0, 200), rng.uniform(0.01, 4.0, 200)):
            outlet = self.assert_step_conserves(pipe, inlet, flow)
            self.assertTrue(280.0 <= outlet <= 320.0)
            self.assertAlmostEqual(pipe.stored_mass_kg, 10.0, delta=1e-11)

    def test_invalid_inputs_leave_state_unchanged(self):
        pipe = self.make_pipe()
        before = (pipe.parcel_masses_kg, pipe.queue_values)
        for inlet, flow in ((300.0, -1.0), (float('nan'), 1.0), (300.0, float('inf'))):
            with self.assertRaises(ValueError):
                pipe.step(inlet, mass_flow_kg_s=flow)
            self.assertEqual(before, (pipe.parcel_masses_kg, pipe.queue_values))


class HeatCurrentMassTransportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from cluster_plant_v2.tests.test_heat_current_plant_independence import _build_legacy_plant
        cls.seed = _build_legacy_plant()

    def test_variable_flow_closes_raw_plant_ledger_without_correction(self):
        from cluster_plant_v2.tests.test_heat_current_plant_independence import _constant_inputs
        from cluster_plant_v2.tests.test_heat_current_closure_repairs import snapshot, hc_ledger
        from cluster_plant_v2.thermal.heat_current_plant import build_independent_hc_plant
        self.assertIsNotNone(CoolantMassTransport)
        flow = self.seed.pump.solve_operating_point(3600.0, self.seed.hydraulic_network)["total_mass_flow_kg_s"]
        plant = build_independent_hc_plant(legacy_plant=self.seed, transport_reference_mass_flow_kg_s=flow)
        prev = snapshot(plant, flow=flow)
        initial_mass = (plant.supply_delay.stored_mass_kg, plant.return_delay.stored_mass_kg)
        seed_queue = self.seed.supply_delay.queue_values
        for k, rpm in enumerate((3600.0, 4500.0, 2400.0, 3000.0, 4500.0, 3600.0)):
            result = plant.step(replace(_constant_inputs(), pump_rpm=rpm,
                flow_direction="reverse" if k > 2 else "forward"), dt_s=5.0)
            ledger, prev = hc_ledger(plant, result, prev)
            self.assertLess(abs(ledger.residual_system_w), 1e-6)
            self.assertLess(abs(ledger.residual_loop_implicit_transport_w), 1e-6)
            self.assertEqual(ledger.transport_flow_mismatch_w, 0.0)
            self.assertAlmostEqual(prev["supply_delay_J"], plant.supply_delay.stored_energy_j)
            self.assertEqual(plant.transport_state_count,
                plant.supply_delay.dynamic_state_count + plant.return_delay.dynamic_state_count)
        np.testing.assert_allclose(initial_mass, (plant.supply_delay.stored_mass_kg, plant.return_delay.stored_mass_kg), atol=1e-12)
        self.assertEqual(seed_queue, self.seed.supply_delay.queue_values)


if __name__ == "__main__":
    unittest.main()
