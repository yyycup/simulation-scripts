"""Four-branch by three-zone reduced electro-thermal battery pack."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from cluster_plant_v2.hppc_parameters import (
    HPPC_PARAMETERS_PATH,
    load_hppc_parameters,
)


ZONE_GROUPS = ((0, 1, 2, 3), (4, 5, 6, 7, 8), (9, 10, 11, 12))
ZONE_CELL_COUNTS = np.array([4, 5, 4], dtype=int)

_DEFAULT_CONFIG = {
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


class ReducedBatteryPack:
    """A 4P13S pack represented by twelve 4-by-3 electro-thermal zones."""

    rows = 4
    cols = 3
    n_parallel = 4
    n_series = 13
    dynamic_state_count = 40

    def __init__(self, config: Mapping[str, object] | None = None) -> None:
        self.config = dict(_DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        self.total_current = float(self.config["total_current"])
        self.soc_branch = np.full(4, float(self.config["initial_soc"]), dtype=float)
        self.eta1 = np.zeros((4, 3), dtype=float)
        self.eta2 = np.zeros((4, 3), dtype=float)
        self.temps = np.full(
            (4, 3), float(self.config["initial_temp_c"]) + 273.15, dtype=float
        )
        self.q_gen_zones = np.zeros((4, 3), dtype=float)
        self.q_gen_total = 0.0
        self.branch_currents = np.zeros(4, dtype=float)
        self.q_battery_to_plate_branch_zone = np.zeros((4, 3), dtype=float)
        self.q_battery_to_plate_zones = np.zeros(3, dtype=float)
        self.q_air_loss_branch_zone = np.zeros((4, 3), dtype=float)

        counts = ZONE_CELL_COUNTS.astype(float)
        self.zone_heat_capacities = np.tile(
            counts * float(self.config["cell_thermal_mass"]), (4, 1)
        )
        self.zone_plate_conductances = np.tile(
            counts * float(self.config["h_plate_convection"]), (4, 1)
        )
        self.zone_air_conductances = self._aggregate_air_conductances()

        self._hppc_data = load_hppc_parameters(HPPC_PARAMETERS_PATH)
        # SOC support of the HPPC tables: lookups outside this range would be
        # linear extrapolation, which can drive (e.g.) r1_dis through zero and
        # blow up the polarization updates.
        self._soc_lookup_bounds = (
            float(self._hppc_data["soc"][0]),
            float(self._hppc_data["soc"][-1]),
        )
        # Temperature support of the HPPC tables: same rationale as the SOC
        # support -- under strong overcooling (e.g. LP-unloaded compressor
        # holding the loop near 12 C) the pack temperature can fall below the
        # calibrated 15 C edge, and extrapolating r1 there crosses zero.
        self._temp_lookup_bounds = (
            float(self._hppc_data["temp"][0]),
            float(self._hppc_data["temp"][-1]),
        )
        axes = (self._hppc_data["soc"], self._hppc_data["temp"])
        self._interpolators = {
            key: RegularGridInterpolator(
                axes, values, bounds_error=False, fill_value=None
            )
            for key, values in self._hppc_data.items()
            if key not in {"soc", "temp"}
        }
        initial_current = np.zeros((4, 3), dtype=float)
        self.last_zone_parameters = self._zone_parameters(
            np.repeat(self.soc_branch[:, None], 3, axis=1),
            self.temps,
            initial_current,
        )

    @property
    def current(self) -> float:
        return self.total_current

    @current.setter
    def current(self, value: float) -> None:
        self.total_current = float(value)

    def _interpolate(self, key: str, soc, temperature):
        soc = np.clip(
            np.asarray(soc, dtype=float),
            self._soc_lookup_bounds[0],
            self._soc_lookup_bounds[1],
        )
        temperature = np.clip(
            np.asarray(temperature, dtype=float),
            self._temp_lookup_bounds[0],
            self._temp_lookup_bounds[1],
        )
        points = np.column_stack((soc.ravel(), temperature.ravel()))
        return np.asarray(self._interpolators[key](points), dtype=float).reshape(soc.shape)

    def _cell_parameters(self, soc, temperature, current) -> dict[str, np.ndarray]:
        current = np.asarray(current, dtype=float)
        discharge_fraction = 0.5 * (1.0 + np.tanh(0.1 * current))

        def switched(discharge_key: str, charge_key: str):
            discharge = self._interpolate(discharge_key, soc, temperature)
            charge = self._interpolate(charge_key, soc, temperature)
            return discharge_fraction * discharge + (1.0 - discharge_fraction) * charge

        return {
            "ocv": self._interpolate("ocv", soc, temperature),
            "r0": switched("r0_dis", "r0_chg"),
            "r1": switched("r1_dis", "r1_chg"),
            "c1": switched("c1_dis", "c1_chg"),
            "r2": switched("r2_dis", "r2_chg"),
            "c2": switched("c2_dis", "c2_chg"),
        }

    def _zone_parameters(self, soc, temperature, current) -> dict[str, np.ndarray]:
        cell = self._cell_parameters(soc, temperature, current)
        counts = ZONE_CELL_COUNTS.astype(float)[None, :]
        return {
            "ocv": cell["ocv"] * counts,
            "r0": cell["r0"] * counts,
            "r1": cell["r1"] * counts,
            "c1": cell["c1"] / counts,
            "r2": cell["r2"] * counts,
            "c2": cell["c2"] / counts,
        }

    def calculate_branch_currents(self, total_current: float | None = None) -> np.ndarray:
        if total_current is not None:
            self.total_current = float(total_current)
        if not np.isfinite(self.total_current):
            raise ValueError("total current must be finite")

        if bool(self.config["dynamic_resistance_update"]):
            soc = np.repeat(self.soc_branch[:, None], 3, axis=1)
            temperature = self.temps
        else:
            soc = np.full((4, 3), float(self.config["initial_soc"]))
            temperature = np.full((4, 3), 298.15)
        suffix = "dis" if self.total_current >= 0.0 else "chg"
        cell_resistance = sum(
            self._interpolate(f"r{element}_{suffix}", soc, temperature)
            for element in range(3)
        )
        branch_resistance = np.sum(
            cell_resistance * ZONE_CELL_COUNTS[None, :], axis=1
        )
        conductance = 1.0 / (branch_resistance + 1e-9)
        self.branch_currents = self.total_current * conductance / conductance.sum()
        return self.branch_currents.copy()

    def calculate_neighbor_heat(self, temperatures=None) -> np.ndarray:
        temperature = np.asarray(
            self.temps if temperatures is None else temperatures, dtype=float
        ).reshape(4, 3)
        heat = np.zeros((4, 3), dtype=float)
        cell_conductance = float(self.config["k_intercell"])

        for branch in range(4):
            for zone in range(2):
                flow = cell_conductance * (
                    temperature[branch, zone + 1] - temperature[branch, zone]
                )
                heat[branch, zone] += flow
                heat[branch, zone + 1] -= flow
        for branch in range(3):
            for zone, count in enumerate(ZONE_CELL_COUNTS):
                flow = count * cell_conductance * (
                    temperature[branch + 1, zone] - temperature[branch, zone]
                )
                heat[branch, zone] += flow
                heat[branch + 1, zone] -= flow
        return heat

    def step(
        self,
        dt: float,
        total_current: float | None = None,
        plate_temperatures: float | np.ndarray = 298.15,
        ambient_temperature: float = 298.15,
    ) -> None:
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        plate = np.asarray(plate_temperatures, dtype=float)
        if plate.ndim == 0:
            plate = np.full(3, float(plate))
        if plate.shape != (3,) or not np.all(np.isfinite(plate)):
            raise ValueError("plate_temperatures must be finite and scalar or shape (3,)")
        ambient_temperature = float(ambient_temperature)
        if not np.isfinite(ambient_temperature):
            raise ValueError("ambient_temperature must be finite")

        old_temperature = self.temps.copy()
        currents = self.calculate_branch_currents(total_current)
        # Saturation clamp on the HPPC lookup support: Coulomb counting beyond
        # full/empty is unphysical, and beyond the table support the parameter
        # lookups would be linear extrapolation (r1_dis crosses zero near
        # SOC ~ 0.995 and overflows the polarization exp update under long
        # net-charging profiles such as frequency regulation).
        new_soc = np.clip(
            self.soc_branch
            - currents * dt / (float(self.config["capacity"]) * 3600.0),
            self._soc_lookup_bounds[0],
            self._soc_lookup_bounds[1],
        )
        soc_zones = np.repeat(new_soc[:, None], 3, axis=1)
        current_zones = np.repeat(currents[:, None], 3, axis=1)
        parameters = self._zone_parameters(soc_zones, old_temperature, current_zones)

        eta1_inf = -current_zones * parameters["r1"]
        eta2_inf = -current_zones * parameters["r2"]
        eta1 = eta1_inf + (self.eta1 - eta1_inf) * np.exp(
            -dt / (parameters["r1"] * parameters["c1"])
        )
        eta2 = eta2_inf + (self.eta2 - eta2_inf) * np.exp(
            -dt / (parameters["r2"] * parameters["c2"])
        )
        heat_generation = (
            current_zones**2 * parameters["r0"]
            - current_zones * eta1
            - current_zones * eta2
        )

        neighbor_heat = self.calculate_neighbor_heat(old_temperature)
        plate_loss = self.zone_plate_conductances * (old_temperature - plate[None, :])
        air_loss = self.zone_air_conductances * (
            old_temperature - ambient_temperature
        )
        new_temperature = old_temperature + dt / self.zone_heat_capacities * (
            heat_generation + neighbor_heat - plate_loss - air_loss
        )
        states = (new_soc, eta1, eta2, heat_generation, new_temperature)
        if not all(np.all(np.isfinite(state)) for state in states):
            raise FloatingPointError("ReducedBatteryPack produced a non-finite state")

        self.soc_branch = new_soc
        self.eta1 = eta1
        self.eta2 = eta2
        self.q_gen_zones = heat_generation
        self.q_gen_total = float(heat_generation.sum())
        self.q_battery_to_plate_branch_zone = plate_loss.copy()
        self.q_battery_to_plate_zones = plate_loss.sum(axis=0)
        self.q_air_loss_branch_zone = air_loss.copy()
        self.temps = new_temperature
        self.last_zone_parameters = parameters

    def _aggregate_air_conductances(self) -> np.ndarray:
        base = float(self.config["h_air_convection_edge"])
        if bool(self.config["uniform_air_convection"]):
            return np.tile(base * ZONE_CELL_COUNTS[None, :], (4, 1))
        row_factors = (1.5, 1.0, 0.7, 1.2)
        cell_conductance = np.empty((4, 13), dtype=float)
        for branch in range(4):
            for column in range(13):
                column_factor = 1.0 - 0.4 * column / 12.0
                on_edge = branch in (0, 3) or column in (0, 12)
                edge_boost = 1.8 if on_edge else 1.0
                cell_conductance[branch, column] = (
                    base * row_factors[branch] * column_factor * edge_boost
                )
        return np.array(
            [
                [cell_conductance[branch, list(columns)].sum() for columns in ZONE_GROUPS]
                for branch in range(4)
            ]
        )

    def initialize_from_reference(self, reference_pack) -> None:
        reference_soc = np.asarray(reference_pack.socs, dtype=float).reshape(4, 13)
        reference_temperature = np.asarray(reference_pack.temps, dtype=float).reshape(4, 13)
        self.soc_branch = reference_soc.mean(axis=1)
        for branch in range(4):
            for zone, columns in enumerate(ZONE_GROUPS):
                indices = [branch * 13 + column for column in columns]
                self.temps[branch, zone] = reference_temperature[
                    branch, list(columns)
                ].mean()
                if all(sim.solution is not None for sim in reference_pack.sims):
                    self.eta1[branch, zone] = sum(
                        float(
                            reference_pack.sims[index]
                            .solution["Element-1 overpotential [V]"]
                            .data[-1]
                        )
                        for index in indices
                    )
                    self.eta2[branch, zone] = sum(
                        float(
                            reference_pack.sims[index]
                            .solution["Element-2 overpotential [V]"]
                            .data[-1]
                        )
                        for index in indices
                    )
                self.q_gen_zones[branch, zone] = np.asarray(
                    reference_pack.q_gen_cells, dtype=float
                )[indices].sum()
        self.q_gen_total = float(self.q_gen_zones.sum())
        self.total_current = float(reference_pack.total_current)
        self.branch_currents = np.asarray(reference_pack.branch_currents, dtype=float).copy()

    def reconstruct_cell_temperatures(self) -> np.ndarray:
        reconstructed = np.empty((4, 13), dtype=float)
        for zone, columns in enumerate(ZONE_GROUPS):
            reconstructed[:, list(columns)] = self.temps[:, zone, None]
        return reconstructed

    def get_avg_temp(self) -> float:
        return float(np.sum(self.temps * ZONE_CELL_COUNTS[None, :]) / 52.0)

    def get_max_temp(self) -> float:
        return float(np.max(self.temps))

    def get_min_temp(self) -> float:
        return float(np.min(self.temps))

    def get_delta_temp(self) -> float:
        return self.get_max_temp() - self.get_min_temp()

    def get_branch_currents(self) -> np.ndarray:
        return self.branch_currents.copy()

    def get_heat_generation(self) -> np.ndarray:
        return self.q_gen_zones.copy()

    def get_battery_to_plate_heat_branch_zone(self) -> np.ndarray:
        return self.q_battery_to_plate_branch_zone.copy()

    def get_battery_to_plate_heat(self) -> np.ndarray:
        return self.q_battery_to_plate_zones.copy()

    def get_air_heat_loss(self) -> np.ndarray:
        return self.q_air_loss_branch_zone.copy()

    def get_soc_array(self) -> np.ndarray:
        return self.soc_branch.copy()
