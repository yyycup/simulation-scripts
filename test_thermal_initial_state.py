import unittest

import numpy as np
from pack import BatteryPack

from thermal_batch_config import INITIAL_TEMP_C, SIM_DT
from thermal_case_simulator import initialize_thermal_temperatures
from thermal_loop import (
    DEFAULT_REFRIGERATION_DYNAMICS,
    build_pack_config,
    initialize_refrigeration_dynamic_state,
    pipe_delay_steps,
)


class ThermalInitialStateTests(unittest.TestCase):
    def test_coolant_tank_and_all_plate_nodes_share_initial_temperature(self):
        tank_temp_k, plate_temps_k = initialize_thermal_temperatures(6)

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
