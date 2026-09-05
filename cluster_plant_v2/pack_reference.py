"""Full-order 4P13S reference battery-pack plant.

This module intentionally contains only the reference pack.  The cold plate and
the rest of the coolant/refrigeration loop remain external plant components.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cluster_plant_v2.hppc_parameters import (
    HPPC_PARAMETERS_PATH as _HPPC_PARAMETERS,
    load_hppc_parameters as _load_hppc_parameters,
)
from cluster_plant_v2.runtime import ensure_env_library_bin_on_path

# CasADi on Windows needs the active Conda environment DLL directory before
# PyBaMM loads the solver backend.
ensure_env_library_bin_on_path()

import pybamm


_DEFAULT_CONFIG = {
    "rows": 4,
    "cols": 13,
    "capacity": 280.0,
    "initial_soc": 0.95,
    "initial_temp_c": 25.0,
    "total_current": 560.0,
    "cell_thermal_mass": 4747.0,
    "k_intercell": 0.5,
    "h_plate_convection": 10.0,
    "h_air_convection_edge": 0.1,
    "uniform_air_convection": False,
    "dynamic_resistance_update": True,
}


class _ExternalCellTemperature(pybamm.BaseSubModel):
    """Provide temperature as an input while retaining PyBaMM heat variables."""

    def __init__(self, parameter, options) -> None:
        super().__init__(parameter)
        self.options = options

    def get_fundamental_variables(self):
        temperature = pybamm.InputParameter("Cell temperature [K]")
        return {
            "Cell temperature [K]": temperature,
            "Cell temperature [degC]": temperature - 273.15,
        }

    def get_coupled_variables(self, variables):
        number_of_rc_elements = int(self.options["number of rc elements"])
        irreversible_heat = sum(
            variables[f"Element-{element} irreversible heat generation [W]"]
            for element in range(number_of_rc_elements + 1)
        )
        reversible_heat = variables["Reversible heat generation [W]"]
        variables.update(
            {
                "Irreversible heat generation [W]": irreversible_heat,
                "Total heat generation [W]": irreversible_heat + reversible_heat,
            }
        )
        return variables


class ReferenceBatteryPack:
    """A 52-cell, 4-parallel by 13-series full-order reference pack."""

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        self.config = dict(_DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.rows = int(self.config["rows"])
        self.cols = int(self.config["cols"])
        if (self.rows, self.cols) != (4, 13):
            raise ValueError("ReferenceBatteryPack topology is fixed at 4P13S")
        self.n_parallel = self.rows
        self.n_series = self.cols
        self.Ns = self.rows * self.cols

        self.total_current = float(self.config["total_current"])
        initial_temperature = float(self.config["initial_temp_c"]) + 273.15
        self.temps = np.full(self.Ns, initial_temperature, dtype=float)
        self.socs = np.full(self.Ns, float(self.config["initial_soc"]), dtype=float)
        self.q_gen_cells = np.zeros(self.Ns, dtype=float)
        self.branch_currents = np.zeros(self.rows, dtype=float)

        self._hppc_data = _load_hppc_parameters(_HPPC_PARAMETERS)
        self._build_branch_resistance_interpolators()

        self.model_template = self._build_electrical_model()
        self._verify_electrical_model(self.model_template)
        self.models = [self.model_template.new_copy() for _ in range(self.Ns)]
        self.sims = [
            pybamm.Simulation(
                model,
                parameter_values=self._parameter_values_for(model, index),
                solver=pybamm.CasadiSolver(mode="safe"),
            )
            for index, model in enumerate(self.models)
        ]

    @property
    def current(self) -> float:
        return self.total_current

    @current.setter
    def current(self, value: float) -> None:
        self.total_current = float(value)

    def cell_index(self, branch: int, series_position: int) -> int:
        if not 0 <= branch < self.rows or not 0 <= series_position < self.cols:
            raise IndexError("cell coordinates are outside the 4P13S topology")
        return branch * self.cols + series_position

    def _build_electrical_model(self):
        model = pybamm.equivalent_circuit.Thevenin(
            options={"number of rc elements": 2}, build=False
        )
        model.submodels["Thermal"] = _ExternalCellTemperature(
            model.param, model.options
        )
        model.build_model()
        return model

    @staticmethod
    def _verify_electrical_model(model) -> None:
        if int(model.options["number of rc elements"]) != 2:
            raise RuntimeError("PyBaMM Thevenin model is not configured for two RC elements")
        rhs_names = {variable.name for variable in model.rhs}
        for element in (1, 2):
            name = f"Element-{element} overpotential [V]"
            if name not in model.variables or name not in rhs_names:
                raise RuntimeError(f"PyBaMM Thevenin model is missing {name}")
        if "Total heat generation [W]" not in model.variables:
            raise RuntimeError("PyBaMM model does not expose total cell heat generation")
        if "Cell temperature [K]" in rhs_names:
            raise RuntimeError("cell temperature must be external to the electrical model")

    def _parameter_values_for(self, model, index: int):
        values = pybamm.ParameterValues(
            {
                "Cell capacity [A.h]": float(self.config["capacity"]),
                "Nominal cell capacity [A.h]": float(self.config["capacity"]),
                "Initial SoC": float(self.config["initial_soc"]),
                "Current function [A]": "[input]",
                "Upper voltage cut-off [V]": 4.5,
                "Lower voltage cut-off [V]": 2.5,
                "Entropic change [V/K]": 0.0,
                "Element-1 initial overpotential [V]": 0.0,
                "Element-2 initial overpotential [V]": 0.0,
            }
        )
        data = self._hppc_data
        soc = model.variables["SoC"]
        temperature = model.variables["Cell temperature [K]"]
        current = model.variables["Current [A]"]
        discharge_fraction = 0.5 * (1 + pybamm.tanh(0.1 * current))

        def interpolate(key: str):
            return pybamm.Interpolant(
                (data["soc"], data["temp"]),
                data[key],
                (soc, temperature),
                name=f"{key}_{index}",
            )

        values.update(
            {
                "Open-circuit voltage [V]": interpolate("ocv"),
                "R0 [Ohm]": discharge_fraction * interpolate("r0_dis")
                + (1 - discharge_fraction) * interpolate("r0_chg"),
                "R1 [Ohm]": discharge_fraction * interpolate("r1_dis")
                + (1 - discharge_fraction) * interpolate("r1_chg"),
                "C1 [F]": discharge_fraction * interpolate("c1_dis")
                + (1 - discharge_fraction) * interpolate("c1_chg"),
                "R2 [Ohm]": discharge_fraction * interpolate("r2_dis")
                + (1 - discharge_fraction) * interpolate("r2_chg"),
                "C2 [F]": discharge_fraction * interpolate("c2_dis")
                + (1 - discharge_fraction) * interpolate("c2_chg"),
            },
            check_already_exists=False,
        )
        return values

    def _build_branch_resistance_interpolators(self) -> None:
        data = self._hppc_data
        points = (data["soc"], data["temp"])
        discharge = data["r0_dis"] + data["r1_dis"] + data["r2_dis"]
        charge = data["r0_chg"] + data["r1_chg"] + data["r2_chg"]
        self._discharge_resistance = RegularGridInterpolator(
            points, discharge, bounds_error=False, fill_value=None
        )
        self._charge_resistance = RegularGridInterpolator(
            points, charge, bounds_error=False, fill_value=None
        )

    def calculate_branch_currents(self, total_current: float | None = None) -> np.ndarray:
        """Split pack current among the four series strings by conductance."""
        if total_current is not None:
            self.total_current = float(total_current)
        if not np.isfinite(self.total_current):
            raise ValueError("total current must be finite")

        if bool(self.config["dynamic_resistance_update"]):
            points = np.column_stack((self.socs, self.temps))
        else:
            points = np.column_stack(
                (
                    np.full(self.Ns, float(self.config["initial_soc"])),
                    np.full(self.Ns, 298.15),
                )
            )
        interpolator = (
            self._discharge_resistance
            if self.total_current >= 0.0
            else self._charge_resistance
        )
        cell_resistances = np.asarray(interpolator(points), dtype=float)
        branch_resistances = cell_resistances.reshape(self.rows, self.cols).sum(axis=1)
        conductances = 1.0 / (branch_resistances + 1e-9)
        conductance_sum = float(conductances.sum())
        if not np.isfinite(conductance_sum) or conductance_sum == 0.0:
            currents = np.full(self.rows, self.total_current / self.rows)
        else:
            currents = self.total_current * conductances / conductance_sum
        self.branch_currents = np.asarray(currents, dtype=float)
        return self.branch_currents.copy()

    def calculate_neighbor_heat(
        self, temperatures: np.ndarray | None = None
    ) -> np.ndarray:
        """Return conservative pairwise cell-to-cell heat rates in watts."""
        source = self.temps if temperatures is None else temperatures
        temperature_grid = np.asarray(source, dtype=float).reshape(self.rows, self.cols)
        neighbor_heat = np.zeros((self.rows, self.cols), dtype=float)
        conductance = float(self.config["k_intercell"])

        for row in range(self.rows):
            for col in range(self.cols - 1):
                heat = conductance * (
                    temperature_grid[row, col + 1] - temperature_grid[row, col]
                )
                neighbor_heat[row, col] += heat
                neighbor_heat[row, col + 1] -= heat
        for row in range(self.rows - 1):
            for col in range(self.cols):
                heat = conductance * (
                    temperature_grid[row + 1, col] - temperature_grid[row, col]
                )
                neighbor_heat[row, col] += heat
                neighbor_heat[row + 1, col] -= heat
        return neighbor_heat.reshape(self.Ns)

    def step(
        self,
        dt: float,
        total_current: float | None = None,
        plate_temperatures: float | np.ndarray = 298.15,
        ambient_temperature: float = 298.15,
    ) -> None:
        """Advance electrical states first, then the explicit 52-node thermal state."""
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a positive finite number")
        plate = np.asarray(plate_temperatures, dtype=float)
        if plate.ndim == 0:
            plate = np.full(self.cols, float(plate))
        if plate.shape != (self.cols,) or not np.all(np.isfinite(plate)):
            raise ValueError("plate_temperatures must be finite and scalar or shape (13,)")
        ambient_temperature = float(ambient_temperature)
        if not np.isfinite(ambient_temperature):
            raise ValueError("ambient_temperature must be finite")

        old_temperatures = self.temps.copy()
        branch_currents = self.calculate_branch_currents(total_current)
        new_socs = np.empty_like(self.socs)
        heat_generation = np.empty_like(self.q_gen_cells)
        for index, simulation in enumerate(self.sims):
            branch = index // self.cols
            solution = simulation.step(
                dt,
                inputs={
                    "Current function [A]": branch_currents[branch],
                    "Cell temperature [K]": old_temperatures[index],
                },
            )
            new_socs[index] = float(solution["SoC"].data[-1])
            heat_generation[index] = float(
                solution["Total heat generation [W]"].data[-1]
            )

        neighbor_heat = self.calculate_neighbor_heat(old_temperatures)
        local_plate_temperatures = np.tile(plate, self.rows)
        plate_heat_loss = float(self.config["h_plate_convection"]) * (
            old_temperatures - local_plate_temperatures
        )
        air_heat_loss = self._air_convection_coefficients() * (
            old_temperatures - ambient_temperature
        )
        new_temperatures = old_temperatures + dt / float(
            self.config["cell_thermal_mass"]
        ) * (heat_generation + neighbor_heat - plate_heat_loss - air_heat_loss)

        if not (
            np.all(np.isfinite(new_socs))
            and np.all(np.isfinite(heat_generation))
            and np.all(np.isfinite(new_temperatures))
        ):
            raise FloatingPointError("ReferenceBatteryPack produced a non-finite state")
        self.socs = new_socs
        self.q_gen_cells = heat_generation
        self.temps = new_temperatures

    def _air_convection_coefficients(self) -> np.ndarray:
        base_coefficient = float(self.config["h_air_convection_edge"])
        if bool(self.config["uniform_air_convection"]):
            return np.full(self.Ns, base_coefficient)

        row_factors = (1.5, 1.0, 0.7, 1.2)
        factors = np.empty((self.rows, self.cols), dtype=float)
        for row in range(self.rows):
            for col in range(self.cols):
                column_factor = 1.0 - 0.4 * col / (self.cols - 1)
                on_edge = row in (0, self.rows - 1) or col in (0, self.cols - 1)
                edge_boost = 1.8 if on_edge else 1.0
                factors[row, col] = row_factors[row] * column_factor * edge_boost
        return base_coefficient * factors.reshape(self.Ns)

    def get_avg_temp(self) -> float:
        return float(np.mean(self.temps))

    def get_max_temperature(self) -> float:
        return float(np.max(self.temps))

    def get_max_temp(self) -> float:
        return self.get_max_temperature()

    def get_min_temperature(self) -> float:
        return float(np.min(self.temps))

    def get_min_temp(self) -> float:
        return self.get_min_temperature()

    def get_temperature_delta(self) -> float:
        return self.get_max_temperature() - self.get_min_temperature()

    def get_delta_temp(self) -> float:
        return self.get_temperature_delta()

    def get_temperatures(self) -> np.ndarray:
        return self.temps.copy()

    def get_socs(self) -> np.ndarray:
        return self.socs.copy()

    def get_soc_array(self) -> np.ndarray:
        return self.get_socs()

    def get_branch_currents(self) -> np.ndarray:
        return self.branch_currents.copy()

    def get_heat_generation(self) -> np.ndarray:
        return self.q_gen_cells.copy()

