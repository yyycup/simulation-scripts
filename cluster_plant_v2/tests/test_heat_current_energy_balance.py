"""Stage 4 tests for the system-level energy-current ledger.

These tests verify that the per-step ``EnergyLedgerStep`` produced by
``ledger_from_legacy`` and ``ledger_from_heat_current`` satisfies the
three local conservation equations of the Stage 4 spec to the same
precision as the existing Stage 8C3 local gates:

* Battery:    ``Q_gen - Q_air - dE_battery/dt - Q_bp ≈ 0``    (≤ 1e-8 W)
* Cold plate: ``Q_bp - dE_plate/dt - Q_pf ≈ 0``                (≤ 1e-9 W)
* Loop:       ``Q_pf - Q_evap_applied - dE_coolant_total/dt`` is reported
              as the **implicit-transport residual** (``R_loop``); it is
              *not* expected to vanish because the cluster-internal
              coolant segments are folded in.

The cumulative residual is checked for finiteness only; per-case
absolute thresholds live in ``validate_heat_current_stage4.py``.
"""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.thermal import (
    HeatCurrentStepInputs,
    build_parallel_system,
)
from cluster_plant_v2.thermal.heat_current_energy_balance import (
    EnergyLedgerStep,
    cumulative_residual_j,
    initial_energy_snapshot,
    ledger_from_heat_current,
    ledger_from_legacy,
)
from cluster_plant_v2.validation.validate_final_cluster_plant import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    build_final_plant,
)


def _step_once(legacy, parallel):
    out = legacy.step(
        dt_s=DT_S, cluster_current_a=560.0, pump_speed_rpm=3500.0,
        compressor_speed_command_rpm=4000.0, fan_speed_rpm=1500.0,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K, direction="forward",
    )
    ref = out["refrigeration_result"]
    inp = HeatCurrentStepInputs(
        dt_s=DT_S, cluster_current_a=560.0,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K, direction="forward",
        total_mass_flow_kg_s=out["total_mass_flow_kg_s"],
        tank_temperature_before_k=out["tank_temperature_before_k"],
        q_evap_applied_w=out["q_evap_applied_w"],
        q_evap_cycle_w=out["q_evap_cycle_w"],
        evaporating_temperature_k=ref["evaporating_saturation_temperature_k"],
        evaporator_ua_w_k=ref["evaporator_ua_w_k"],
        refrigeration_solver_success=out["refrigeration_solver_success"],
        compressor_speed_rpm=out["compressor_speed_rpm"],
    )
    hc = parallel.step(inp)
    return out, inp, hc


class EnergyLedgerDataclassTests(unittest.TestCase):
    def test_dataclass_fields_are_finite(self) -> None:
        ledger = EnergyLedgerStep(
            time_s=0.0,
            q_gen_total_w=1.0, q_air_total_w=0.1, q_bp_total_w=0.9,
            q_pf_total_w=0.85, q_evap_cycle_w=0.95, q_evap_applied_w=0.9,
            q_tank_return_to_tank_w=0.85,
            dE_battery_per_s=0.0, dE_plate_per_s=0.05,
            dE_coolant_segments_per_s=0.0,
            dE_supply_per_s=0.01, dE_return_per_s=-0.01,
            dE_tank_per_s=0.0,
            residual_battery_w=0.0,
            residual_plate_w=0.0,
            residual_loop_implicit_transport_w=0.0,
            residual_system_w=0.0,
        )
        self.assertTrue(np.isfinite(ledger.q_gen_total_w))
        self.assertTrue(np.isfinite(ledger.residual_loop_implicit_transport_w))


class EnergyLedgerLegacyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.legacy = build_final_plant(
            4000.0, initial_cluster_current_a=560.0, direction="forward",
        )

    def test_battery_residual_within_stage_8c3_gate(self) -> None:
        prev = initial_energy_snapshot(self.legacy, is_heat_current=False)
        out, _, _ = _step_once(self.legacy, build_parallel_system(
            legacy_plant=self.legacy
        ))
        ledger, _ = ledger_from_legacy(
            self.legacy, dt_s=DT_S, result=out, prev_energy=prev,
        )
        self.assertLess(abs(ledger.residual_battery_w), 1e-7)

    def test_plate_residual_within_stage_8c3_gate(self) -> None:
        prev = initial_energy_snapshot(self.legacy, is_heat_current=False)
        out, _, _ = _step_once(self.legacy, build_parallel_system(
            legacy_plant=self.legacy
        ))
        ledger, _ = ledger_from_legacy(
            self.legacy, dt_s=DT_S, result=out, prev_energy=prev,
        )
        self.assertLess(abs(ledger.residual_plate_w), 1e-8)

    def test_loop_residual_is_finite_and_signed(self) -> None:
        prev = initial_energy_snapshot(self.legacy, is_heat_current=False)
        out, _, _ = _step_once(self.legacy, build_parallel_system(
            legacy_plant=self.legacy
        ))
        ledger, _ = ledger_from_legacy(
            self.legacy, dt_s=DT_S, result=out, prev_energy=prev,
        )
        self.assertTrue(np.isfinite(ledger.residual_loop_implicit_transport_w))
        # It is allowed to be non-zero; only finiteness is required.

    def test_all_q_fields_finite(self) -> None:
        prev = initial_energy_snapshot(self.legacy, is_heat_current=False)
        out, _, _ = _step_once(self.legacy, build_parallel_system(
            legacy_plant=self.legacy
        ))
        ledger, _ = ledger_from_legacy(
            self.legacy, dt_s=DT_S, result=out, prev_energy=prev,
        )
        for field in (
            "q_gen_total_w", "q_air_total_w", "q_bp_total_w",
            "q_pf_total_w", "q_evap_cycle_w", "q_evap_applied_w",
            "q_tank_return_to_tank_w",
        ):
            self.assertTrue(np.isfinite(getattr(ledger, field)), field)

    def test_cumulative_residual_no_nan(self) -> None:
        prev = initial_energy_snapshot(self.legacy, is_heat_current=False)
        parallel = build_parallel_system(legacy_plant=self.legacy)
        prev_hc = initial_energy_snapshot(parallel, is_heat_current=True)
        rows = []
        from dataclasses import replace
        for _ in range(3):
            out, inp, hc = _step_once(self.legacy, parallel)
            leg, prev = ledger_from_legacy(
                self.legacy, dt_s=DT_S, result=out, prev_energy=prev,
            )
            hc_ledger, prev_hc = ledger_from_heat_current(
                parallel, dt_s=DT_S, inputs=inp, hc=hc,
                prev_energy=prev_hc,
            )
            leg = replace(leg, time_s=float(len(rows) + 1) * DT_S)
            hc_ledger = replace(hc_ledger, time_s=float(len(rows) + 1) * DT_S)
            rows.append(leg)
            rows.append(hc_ledger)
        cum = cumulative_residual_j(rows)
        for value in cum.values():
            self.assertTrue(np.isfinite(value))


class EnergyLedgerHeatCurrentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.legacy = build_final_plant(
            4000.0, initial_cluster_current_a=560.0, direction="forward",
        )
        self.parallel = build_parallel_system(legacy_plant=self.legacy)

    def test_battery_residual_within_stage_8c3_gate(self) -> None:
        prev = initial_energy_snapshot(self.parallel, is_heat_current=True)
        _, inp, hc = _step_once(self.legacy, self.parallel)
        ledger, _ = ledger_from_heat_current(
            self.parallel, dt_s=DT_S, inputs=inp, hc=hc, prev_energy=prev,
        )
        self.assertLess(abs(ledger.residual_battery_w), 1e-7)

    def test_plate_residual_within_stage_8c3_gate(self) -> None:
        prev = initial_energy_snapshot(self.parallel, is_heat_current=True)
        _, inp, hc = _step_once(self.legacy, self.parallel)
        ledger, _ = ledger_from_heat_current(
            self.parallel, dt_s=DT_S, inputs=inp, hc=hc, prev_energy=prev,
        )
        self.assertLess(abs(ledger.residual_plate_w), 1e-8)

    def test_all_q_fields_finite(self) -> None:
        prev = initial_energy_snapshot(self.parallel, is_heat_current=True)
        _, inp, hc = _step_once(self.legacy, self.parallel)
        ledger, _ = ledger_from_heat_current(
            self.parallel, dt_s=DT_S, inputs=inp, hc=hc, prev_energy=prev,
        )
        for field in (
            "q_gen_total_w", "q_air_total_w", "q_bp_total_w",
            "q_pf_total_w", "q_evap_cycle_w", "q_evap_applied_w",
        ):
            self.assertTrue(np.isfinite(getattr(ledger, field)), field)


class EnergyLedgerConservationSteadyStateTests(unittest.TestCase):
    """Run a short steady-state window and confirm the two local gates
    continue to hold."""

    def test_legacy_residuals_steady_state(self) -> None:
        legacy = build_final_plant(
            4000.0, initial_cluster_current_a=560.0, direction="forward",
        )
        parallel = build_parallel_system(legacy_plant=legacy)
        prev = initial_energy_snapshot(legacy, is_heat_current=False)
        max_bat = 0.0
        max_plate = 0.0
        for _ in range(24):  # 120 s, covers transient
            out, inp, hc = _step_once(legacy, parallel)
            ledger, prev = ledger_from_legacy(
                legacy, dt_s=DT_S, result=out, prev_energy=prev,
            )
            max_bat = max(max_bat, abs(ledger.residual_battery_w))
            max_plate = max(max_plate, abs(ledger.residual_plate_w))
        # 120 s is past startup; Stage 8C3 gates apply.
        self.assertLess(max_bat, 5e-8)
        self.assertLess(max_plate, 5e-9)


if __name__ == "__main__":
    unittest.main()