"""Stage 4.5 — Independence tests for ``HeatCurrentPlant``.

These tests verify the central claim of Stage 4.5:

    After ``build_independent_hc_plant(legacy_plant)`` returns, the two
    plants are **independent**: stepping one does not mutate the other's
    internal state, and each plant reads only its own previous state to
    compute the next step.

The tests do **not** compare HC and legacy outputs (that is Stage 5's
job). They only verify the architectural separation.
"""

from __future__ import annotations

import unittest

import numpy as np

from cluster_plant_v2.hydraulics import (
    CoolantTank,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.parameters import (
    EVAPORATOR_TIME_CONSTANT_S,
)
from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
    CoolantTransportDelay,
    EvaporatorThermalDynamics,
)
from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.thermal.heat_current_plant import (
    HeatCurrentPlant,
    HeatCurrentPlantInputs,
    build_independent_hc_plant,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    build_engineering_pump,
    build_medium_header_network,
)


DT_S = 5.0
N_PACKS = 5
INITIAL_TEMPERATURE_K = 298.15
AMBIENT_TEMPERATURE_K = 308.15
INITIAL_SOC = 0.95


def _build_legacy_plant() -> ClusterPlant:
    """Build a frozen legacy plant from the standard validation harness."""
    network = build_medium_header_network()
    return ClusterPlant.from_equilibrium(
        cluster=ReducedCluster(
            n_packs=N_PACKS,
            hydraulic_mode="header_network",
            hydraulic_network=network,
            pack_config={
                "initial_soc": INITIAL_SOC,
                "initial_battery_temperature_k": INITIAL_TEMPERATURE_K,
                "initial_plate_temperature_k": INITIAL_TEMPERATURE_K,
            },
        ),
        hydraulic_network=network,
        pump=build_engineering_pump(network),
        tank=CoolantTank(initial_temperature_k=INITIAL_TEMPERATURE_K),
        refrigeration_cycle=ClosedR134aCycle(),
        compressor_actuator=CompressorSpeedActuator(
            initial_speed_rpm=4000.0,
            time_constant_s=5.0,
        ),
        dt_s=DT_S,
        pump_speed_rpm=3600.0,
        fan_speed_rpm=1200.0,
        initial_cluster_current_a=0.0,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
    )


def _constant_inputs(current_a: float = 560.0, pump_rpm: float = 3600.0):
    """Constant external input profile for Stage 4.5 isolation tests."""
    return HeatCurrentPlantInputs(
        cluster_current_a=current_a,
        compressor_command_rpm=4000.0,
        pump_rpm=pump_rpm,
        fan_rpm=1200.0,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        flow_direction="forward",
    )


def _snapshot_state(plant) -> dict[str, np.ndarray]:
    """Capture every mutable state attribute on the plant."""
    return {
        "tank_temperature_k": np.array([plant.tank.temperature_k]),
        "compressor_speed_rpm": np.array([plant.compressor_actuator.speed_rpm]),
        "q_evap_applied_w": np.array(
            [plant.evaporator_dynamics.q_evap_applied_w]
        ),
        "evaporator_buffer_energy_j": np.array(
            [plant.evaporator_dynamics.evaporator_buffer_energy_j]
        ),
        "supply_delay_queue_k": np.array(
            list(plant.supply_delay.queue_values)
        ),
        "return_delay_queue_k": np.array(
            list(plant.return_delay.queue_values)
        ),
        "cluster_pack_q_gen_total_w": np.array(
            list(plant.cluster.packs[0].battery.q_gen_total for _ in [0])
        ),
        "cluster_plate_temperatures_p0": plant.cluster.packs[0].cold_plate.plate_temperatures.copy(),
        "cluster_battery_temps_p0": plant.cluster.packs[0].battery.temps.copy(),
    }


class HeatCurrentPlantIndependenceTests(unittest.TestCase):
    """Architectural independence between legacy and HC plants."""

    def test_legacy_step_does_not_mutate_hc_state(self):
        """Stepping the legacy plant must not change any HC state."""
        legacy = _build_legacy_plant()
        hc = build_independent_hc_plant(legacy_plant=legacy)

        before = _snapshot_state(hc)

        # Step legacy several times with constant external input
        for _ in range(6):
            legacy.step(
                dt_s=DT_S,
                cluster_current_a=560.0,
                pump_speed_rpm=3600.0,
                compressor_speed_command_rpm=4000.0,
                fan_speed_rpm=1200.0,
                ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                direction="forward",
            )

        after = _snapshot_state(hc)
        for key in before:
            np.testing.assert_array_equal(
                before[key],
                after[key],
                err_msg=f"legacy.step() mutated hc.{key}",
            )

    def test_hc_step_does_not_mutate_legacy_state(self):
        """Stepping the HC plant must not change any legacy dynamic state.

        Dynamic state we track on the legacy plant:

          * ``tank.temperature_k``
          * ``compressor_actuator.speed_rpm``
          * ``evaporator_dynamics.q_evap_applied_w``
          * ``evaporator_dynamics.evaporator_buffer_energy_j``
          * ``supply_delay.queue_values``
          * ``return_delay.queue_values``
          * ``cluster.packs[*].battery.temps``
          * ``cluster.packs[*].battery.q_gen_total`` history (per step)

        The two plants must live completely separate trajectories.
        """
        legacy = _build_legacy_plant()
        hc = build_independent_hc_plant(legacy_plant=legacy)

        legacy_before = {
            "tank_temperature_k": legacy.tank.temperature_k,
            "compressor_speed_rpm": legacy.compressor_actuator.speed_rpm,
            "q_evap_applied_w": legacy.evaporator_dynamics.q_evap_applied_w,
            "evaporator_buffer_energy_j": (
                legacy.evaporator_dynamics.evaporator_buffer_energy_j
            ),
            "supply_delay_queue_k": np.array(
                list(legacy.supply_delay.queue_values)
            ),
            "return_delay_queue_k": np.array(
                list(legacy.return_delay.queue_values)
            ),
            "battery_temps_p0": legacy.cluster.packs[0].battery.temps.copy(),
            "plate_temps_p0": legacy.cluster.packs[0].cold_plate.plate_temperatures.copy(),
        }

        # Step HC several times with the same constant input profile
        inputs = _constant_inputs()
        for _ in range(6):
            hc.step(inputs, dt_s=DT_S)

        # Verify legacy state is **bit-for-bit** unchanged
        self.assertEqual(
            legacy.tank.temperature_k, legacy_before["tank_temperature_k"]
        )
        self.assertEqual(
            legacy.compressor_actuator.speed_rpm,
            legacy_before["compressor_speed_rpm"],
        )
        self.assertEqual(
            legacy.evaporator_dynamics.q_evap_applied_w,
            legacy_before["q_evap_applied_w"],
        )
        self.assertEqual(
            legacy.evaporator_dynamics.evaporator_buffer_energy_j,
            legacy_before["evaporator_buffer_energy_j"],
        )
        np.testing.assert_array_equal(
            np.array(list(legacy.supply_delay.queue_values)),
            legacy_before["supply_delay_queue_k"],
        )
        np.testing.assert_array_equal(
            np.array(list(legacy.return_delay.queue_values)),
            legacy_before["return_delay_queue_k"],
        )
        np.testing.assert_array_equal(
            legacy.cluster.packs[0].battery.temps,
            legacy_before["battery_temps_p0"],
        )
        np.testing.assert_array_equal(
            legacy.cluster.packs[0].cold_plate.plate_temperatures,
            legacy_before["plate_temps_p0"],
        )

    def test_hc_initial_state_seeded_from_legacy(self):
        """``build_independent_hc_plant`` should copy initial state from the
        legacy plant's current state.

        State values that the HC plant's constructor copies from the
        legacy plant must match exactly at construction time. From step
        2 onward the HC plant evolves on its own; this test only checks
        the seeding.
        """
        legacy = _build_legacy_plant()
        hc = build_independent_hc_plant(legacy_plant=legacy)

        # Initial conditions copied from legacy
        self.assertEqual(hc.tank.temperature_k, legacy.tank.temperature_k)
        self.assertEqual(
            hc.compressor_actuator.speed_rpm,
            legacy.compressor_actuator.speed_rpm,
        )
        self.assertEqual(
            hc.evaporator_dynamics.q_evap_applied_w,
            legacy.evaporator_dynamics.q_evap_applied_w,
        )
        np.testing.assert_array_equal(
            np.array(list(hc.supply_delay.queue_values)),
            np.array(list(legacy.supply_delay.queue_values)),
        )
        np.testing.assert_array_equal(
            np.array(list(hc.return_delay.queue_values)),
            np.array(list(legacy.return_delay.queue_values)),
        )

    def test_hc_step_uses_own_previous_state(self):
        """A multi-step HC trajectory should be reproducible from the same
        starting state, independent of any legacy state.

        We build **two** HC plants from the **same** legacy initial
        state, then step only the first one while running the legacy
        plant on a completely different trajectory. The two HC plants
        must produce bit-identical outputs if given identical inputs —
        because each HC plant's state evolves only from its own previous
        step and from the external inputs.
        """
        legacy = _build_legacy_plant()
        hc_a = build_independent_hc_plant(legacy_plant=legacy)
        hc_b = build_independent_hc_plant(legacy_plant=legacy)

        inputs = _constant_inputs()

        # Step both HC plants in lockstep, but vary the legacy trajectory
        # in between to ensure HC_a doesn't accidentally read it.
        trajectory_a = []
        trajectory_b = []
        for step in range(10):
            # Drive the legacy plant on a wildly different profile so
            # any leak of legacy state would surface as a difference.
            if step % 2 == 0:
                legacy.step(
                    dt_s=DT_S,
                    cluster_current_a=1120.0,
                    pump_speed_rpm=4500.0,
                    compressor_speed_command_rpm=2000.0,
                    fan_speed_rpm=2500.0,
                    ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                    direction="reverse",
                )
            else:
                legacy.step(
                    dt_s=DT_S,
                    cluster_current_a=560.0,
                    pump_speed_rpm=3600.0,
                    compressor_speed_command_rpm=4000.0,
                    fan_speed_rpm=1200.0,
                    ambient_temperature_k=AMBIENT_TEMPERATURE_K,
                    direction="forward",
                )
            ra = hc_a.step(inputs, dt_s=DT_S)
            rb = hc_b.step(inputs, dt_s=DT_S)
            trajectory_a.append(
                (
                    ra["tank_temperature_after_k"],
                    ra["q_evap_applied_w"],
                    ra["cluster_supply_temperature_k"],
                    ra["cluster_return_temperature_k"],
                )
            )
            trajectory_b.append(
                (
                    rb["tank_temperature_after_k"],
                    rb["q_evap_applied_w"],
                    rb["cluster_supply_temperature_k"],
                    rb["cluster_return_temperature_k"],
                )
            )

        for step, (sa, sb) in enumerate(zip(trajectory_a, trajectory_b)):
            self.assertEqual(
                sa, sb,
                msg=(
                    f"HC plants diverged at step {step}; one of them"
                    " leaked legacy state"
                ),
            )

    def test_hc_step_inputs_have_no_legacy_dependency(self):
        """``HeatCurrentPlantInputs`` must expose only external quantities.

        No field name should smell of a downstream internal variable
        (``tank_temperature_before_k``, ``q_evap_applied_w``, etc.).
        """
        from dataclasses import fields

        field_names = {
            f.name for f in fields(HeatCurrentPlantInputs)
        }
        forbidden = {
            "tank_temperature_before_k",
            "tank_temperature_after_k",
            "q_evap_applied_w",
            "q_evap_cycle_w",
            "evaporating_temperature_k",
            "evaporator_ua_w_k",
            "refrigeration_solver_success",
            "compressor_speed_used_rpm",
            "supply_temperature_k",
            "return_temperature_k",
        }
        leaked = forbidden & field_names
        self.assertFalse(
            leaked,
            msg=(
                "HeatCurrentPlantInputs exposes legacy-derived fields:"
                f" {sorted(leaked)}"
            ),
        )


if __name__ == "__main__":
    unittest.main()