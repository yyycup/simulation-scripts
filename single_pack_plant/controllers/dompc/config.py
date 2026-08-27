"""All user-adjustable settings for the direct Physics-P do-mpc controller."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicsPDoMPCWeights:
    """Dimensionless objective weights for direct Physics-P NMPC."""

    temperature_error_scale_c: float = 1.0
    w_tavg: float = 500.0
    w_temp_upper: float = 5000.0
    w_comp_energy: float = 0.1
    w_pump_energy: float = 1.0e-3
    w_delta_comp: float = 1.0e-2
    w_delta_pump: float = 1.0e-2

    def validated(self) -> "PhysicsPDoMPCWeights":
        if self.temperature_error_scale_c <= 0.0:
            raise ValueError("temperature_error_scale_c must be positive")
        if any(value < 0.0 for value in (
            self.w_tavg, self.w_temp_upper, self.w_comp_energy,
            self.w_pump_energy, self.w_delta_comp, self.w_delta_pump,
        )):
            raise ValueError("do-mpc weights must be nonnegative")
        return self


@dataclass(frozen=True)
class PhysicsPDoMPCConfig:
    """Prediction horizon and actuator limits for direct Physics-P NMPC."""

    name: str
    horizon: int
    dmax_comp_rpm: float
    dmax_pump_rpm: float
    # Number of independent 5 s moves before holding the final move to the
    # end of the prediction horizon. ``None`` preserves ``Nc = Np``.
    control_horizon: int | None = None
    control_interval_steps: int = 1
    n_comp_min_rpm: float = 300.0
    n_comp_max_rpm: float = 6000.0
    n_pump_min_rpm: float = 1600.0
    n_pump_max_rpm: float = 4800.0
    weights: PhysicsPDoMPCWeights = PhysicsPDoMPCWeights()

    def validated(self) -> "PhysicsPDoMPCConfig":
        if self.horizon <= 0 or self.control_interval_steps <= 0:
            raise ValueError("horizon and control interval must be positive")
        if self.dmax_comp_rpm <= 0.0 or self.dmax_pump_rpm <= 0.0:
            raise ValueError("DMAX values must be positive")
        if self.control_horizon is not None:
            if self.control_horizon <= 0 or self.control_horizon > self.horizon:
                raise ValueError("control_horizon must lie in [1, horizon]")
        if self.n_comp_min_rpm >= self.n_comp_max_rpm or self.n_pump_min_rpm >= self.n_pump_max_rpm:
            raise ValueError("actuator bounds are invalid")
        self.weights.validated()
        return self

    @property
    def control_moves(self) -> int:
        """Number of distinct control decisions in one NMPC optimization."""
        return self.horizon if self.control_horizon is None else self.control_horizon

    @property
    def terminal_hold_steps(self) -> int:
        """Number of 5 s tail steps holding the final optimized action."""
        return self.horizon - self.control_moves

    @classmethod
    def for_scene(cls, scene: object) -> "PhysicsPDoMPCConfig":
        """Return direct-NMPC settings without importing GEKKO or Fixed-QP."""
        label = str(scene).lower()
        if any(token in label for token in ("调频", "freq", "frequency", "reg")):
            return cls(name="physics_p_dompc_frequency", horizon=45, dmax_comp_rpm=6000.0, dmax_pump_rpm=600.0).validated()
        return cls(name="physics_p_dompc_peak", horizon=60, dmax_comp_rpm=6000.0, dmax_pump_rpm=300.0).validated()
