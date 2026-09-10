"""Adiabatic plug flow through a fixed coolant inventory.

Parcels are ordered from outlet to inlet. Each step displaces m_dot * dt kg
and returns its mass-weighted mean temperature. Inlet temperature and flow
are held constant over the step. No mixing, wall storage or heat loss is
added. The stored parcel masses, rather than an elapsed-time FIFO, determine
when a temperature front reaches the outlet.
"""
from collections import deque
import math


class CoolantMassTransport:
    def __init__(self, *, mass_kg: float, dt_s: float,
                 initial_temperatures_k, coolant_specific_heat_j_kg_k: float):
        temperatures = tuple(float(t) for t in initial_temperatures_k)
        if not temperatures or any(not math.isfinite(t) or t <= 0 for t in temperatures):
            raise ValueError("initial temperatures must be positive and finite in K")
        for value in (mass_kg, dt_s, coolant_specific_heat_j_kg_k):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("mass, dt_s and specific heat must be positive and finite")
        self.mass_kg = float(mass_kg)
        self.dt_s = float(dt_s)
        self.coolant_specific_heat_j_kg_k = float(coolant_specific_heat_j_kg_k)
        parcel_mass = self.mass_kg / len(temperatures)
        self._parcels = deque((parcel_mass, t) for t in temperatures)

    @classmethod
    def from_fixed_delay(cls, delay, *, reference_mass_flow_kg_s: float,
                         coolant_specific_heat_j_kg_k: float):
        """Interpret an existing FIFO history as equal-mass spatial parcels.

        This is an explicit initial-state mapping at the reference flow,
        not a reconstruction of the source's past variable-flow history.
        """
        return cls(mass_kg=float(reference_mass_flow_kg_s) * delay.delay_s,
                   dt_s=delay.dt_s, initial_temperatures_k=delay.queue_values,
                   coolant_specific_heat_j_kg_k=coolant_specific_heat_j_kg_k)

    @property
    def queue_values(self) -> tuple[float, ...]:
        """Parcel temperatures for diagnostics; pair with parcel_masses_kg."""
        return tuple(t for _, t in self._parcels)

    @property
    def parcel_masses_kg(self) -> tuple[float, ...]:
        return tuple(m for m, _ in self._parcels)

    @property
    def dynamic_state_count(self) -> int:
        """Stored coordinates (mass and temperature); count can vary with flow."""
        return 2 * len(self._parcels)

    @property
    def stored_mass_kg(self) -> float:
        return math.fsum(m for m, _ in self._parcels)

    @property
    def stored_energy_j(self) -> float:
        return self.coolant_specific_heat_j_kg_k * math.fsum(m * t for m, t in self._parcels)

    def step(self, input_value: float, *, mass_flow_kg_s: float) -> float:
        inlet, flow = float(input_value), float(mass_flow_kg_s)
        transferred = flow * self.dt_s
        if (not math.isfinite(inlet) or inlet <= 0 or not math.isfinite(flow)
                or flow < 0 or not math.isfinite(transferred)):
            raise ValueError("inlet must be positive and finite; mass flow must be finite and nonnegative")
        if transferred == 0:
            return float(self._parcels[0][1])

        inventory = self.stored_mass_kg
        if transferred >= inventory:
            # All old fluid leaves; any excess is new inlet fluid passing through.
            outlet = inlet + math.fsum(m * (t - inlet) for m, t in self._parcels) / transferred
            self._parcels = deque([(inventory, inlet)])
            return float(outlet)

        remaining = transferred
        base_temperature = self._parcels[0][1]
        removed_temperature_moments = []
        while remaining > 0:
            mass, temperature = self._parcels[0]
            take = min(remaining, mass)
            removed_temperature_moments.append(take * (temperature - base_temperature))
            if take == mass:
                self._parcels.popleft()
            else:
                self._parcels[0] = (mass - take, temperature)
            remaining -= take
        # Merge equal-temperature neighbors without changing their energy.
        if self._parcels and self._parcels[-1][1] == inlet:
            self._parcels[-1] = (self._parcels[-1][0] + transferred, inlet)
        else:
            self._parcels.append((transferred, inlet))
        return float(base_temperature + math.fsum(removed_temperature_moments) / transferred)
