import inspect
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from pack import BatteryPack

import thermal_case_simulator as thermal_simulator
from thermal_batch_config import AMBIENT_TEMP_C, INITIAL_TEMP_C, SIM_DT
from thermal_case_simulator import initialize_thermal_temperatures, simulate_case
from thermal_loop import (
    DEFAULT_REFRIGERATION_DYNAMICS,
    build_pack_config,
    initialize_refrigeration_dynamic_state,
    pipe_delay_steps,
)


class ThermalInitialStateTests(unittest.TestCase):
    def test_standard_simulation_defaults_thermal_loop_to_ambient_temperature(self):
        parameter = inspect.signature(simulate_case).parameters.get(
            "initial_thermal_temp_c"
        )

        self.assertIsNotNone(parameter)
        self.assertEqual(AMBIENT_TEMP_C, 35.0)
        self.assertEqual(parameter.default, 35.0)

    def test_simulation_forwards_default_and_explicit_thermal_initial_temperature(self):
        with TemporaryDirectory() as tmp:
            for supplied_kwargs, expected_c in (
                ({}, 35.0),
                ({"initial_thermal_temp_c": 25.0}, 25.0),
            ):
                with self.subTest(expected_c=expected_c):
                    with patch.object(
                        thermal_simulator,
                        "initialize_thermal_temperatures",
                        side_effect=RuntimeError("stop at thermal initialization"),
                    ) as initialize:
                        with self.assertRaisesRegex(
                            RuntimeError,
                            "thermal initialization",
                        ):
                            simulate_case(
                                "pid",
                                "peak",
                                "unidirectional",
                                "missing_source.csv",
                                f"main_{expected_c:g}.csv",
                                f"snap_{expected_c:g}.csv",
                                output_root=tmp,
                                max_steps=1,
                                force=True,
                                **supplied_kwargs,
                            )

                    initialize.assert_called_once()
                    self.assertEqual(
                        initialize.call_args.kwargs["initial_temp_c"],
                        expected_c,
                    )

    def test_coolant_tank_and_all_plate_nodes_default_to_ambient_temperature(self):
        tank_temp_k, plate_temps_k = initialize_thermal_temperatures(6)

        expected_temp_k = AMBIENT_TEMP_C + 273.15
        self.assertEqual(tank_temp_k, expected_temp_k)
        np.testing.assert_array_equal(
            plate_temps_k,
            np.full(6, expected_temp_k),
        )

    def test_physics_p_can_explicitly_initialize_thermal_loop_at_25c(self):
        self.assertEqual(INITIAL_TEMP_C, 25.0)
        tank_temp_k, plate_temps_k = initialize_thermal_temperatures(
            6,
            initial_temp_c=INITIAL_TEMP_C,
        )

        expected_temp_k = INITIAL_TEMP_C + 273.15
        self.assertEqual(tank_temp_k, expected_temp_k)
        np.testing.assert_array_equal(
            plate_temps_k,
            np.full(6, expected_temp_k),
        )

    def test_battery_cells_start_at_global_25c_temperature(self):
        pack = BatteryPack(
            build_pack_config(
                initial_temp_c=INITIAL_TEMP_C,
                dynamic_resistance_update=False,
            )
        )

        np.testing.assert_array_equal(
            pack.temps,
            np.full(pack.Ns, INITIAL_TEMP_C + 273.15),
        )




    def test_refrigeration_pipe_states_and_delay_histories_start_at_25c(self):
        expected_temp_k = INITIAL_TEMP_C + 273.15
        state = initialize_refrigeration_dynamic_state(
            1000.0,
            2000.0,
            initial_temp_k=expected_temp_k,
            dt=SIM_DT,
        )
        supply_steps = pipe_delay_steps(
            DEFAULT_REFRIGERATION_DYNAMICS["tau_pipe_supply_s"], SIM_DT
        )
        return_steps = pipe_delay_steps(
            DEFAULT_REFRIGERATION_DYNAMICS["tau_pipe_return_s"], SIM_DT
        )

        self.assertEqual(state["T_pipe_supply_K"], expected_temp_k)
        self.assertEqual(state["T_pipe_return_K"], expected_temp_k)
        self.assertEqual(
            state["T_pipe_supply_history_K"],
            [expected_temp_k] * supply_steps,
        )
        self.assertEqual(
            state["T_pipe_return_history_K"],
            [expected_temp_k] * return_steps,
        )
if __name__ == "__main__":


    unittest.main()
