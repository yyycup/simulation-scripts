from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

import gymnasium as gym
from gymnasium import spaces

from pack import BatteryPack
from thermal_batch_config import (
    AGC_DATA_FILE,
    AMBIENT_TEMP_C,
    INITIAL_SOC_PEAK,
    INITIAL_SOC_REG,
    INITIAL_TEMP_C,
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    PLATE_NODE_HEAT_CAPACITY_TOTAL,
    SIM_DT,
)
from thermal_loop import build_pack_config, initialize_refrigeration_dynamic_state


DEFAULT_DURATION_S = {"peak": 6400.0, "freq": 3600.0}


def canonical_scene(scene: str) -> str:
    scene_key = str(scene).strip().lower()
    if scene_key not in DEFAULT_DURATION_S:
        raise ValueError("scene must be 'peak' or 'freq'")
    return scene_key


def _validated_profile(values) -> np.ndarray:
    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("current profile must be a non-empty 1D array")
    if not np.all(np.isfinite(profile)):
        raise ValueError("current profile values must be finite")
    return profile


def build_current_profile(
    scene: str,
    *,
    dt: float = SIM_DT,
    current_profile=None,
    agc_data_file=None,
    max_steps=None,
) -> np.ndarray:
    scene_key = canonical_scene(scene)
    dt = float(dt)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")

    if current_profile is not None:
        profile = _validated_profile(current_profile)
    elif scene_key == "peak":
        steps = int(DEFAULT_DURATION_S[scene_key] / dt)
        profile = np.full(steps, 560.0, dtype=np.float64)
    else:
        path = Path(AGC_DATA_FILE if agc_data_file is None else agc_data_file)
        if not path.is_file():
            raise FileNotFoundError(f"AGC data file does not exist: {path}")
        frame = pd.read_csv(path)
        required = {"Seconds", "RegD"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"AGC data file is missing columns: {missing}")
        source_time = frame["Seconds"].to_numpy(dtype=float)
        source_signal = frame["RegD"].to_numpy(dtype=float)
        if source_time.size == 0 or not np.all(np.isfinite(source_time)):
            raise ValueError("AGC Seconds must be non-empty and finite")
        if source_time.size > 1 and not np.all(np.diff(source_time) > 0.0):
            raise ValueError("AGC Seconds must be strictly increasing")
        if not np.all(np.isfinite(source_signal)):
            raise ValueError("AGC RegD values must be finite")
        times = np.arange(0.0, DEFAULT_DURATION_S[scene_key], dt)
        profile = np.interp(
            times,
            source_time,
            source_signal * 1120.0,
            left=0.0,
            right=0.0,
        )

    if max_steps is not None:
        max_steps = int(max_steps)
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        profile = profile[:max_steps]
    return _validated_profile(profile).copy()


def profile_sha256(profile) -> str:
    values = _validated_profile(profile).astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def map_action_to_rpm(action) -> tuple[float, float]:
    values = np.asarray(action, dtype=float)
    if values.shape != (2,):
        raise ValueError("action must have shape (2,)")
    if not np.all(np.isfinite(values)):
        raise ValueError("action values must be finite")
    values = np.clip(values, -1.0, 1.0)
    n_comp = N_COMP_OFF_RPM + 0.5 * (values[0] + 1.0) * (
        N_COMP_MAX_RPM - N_COMP_OFF_RPM
    )
    n_pump = N_PUMP_MIN_RPM + 0.5 * (values[1] + 1.0) * (
        N_PUMP_MAX_RPM - N_PUMP_MIN_RPM
    )
    return float(n_comp), float(n_pump)


class BTMSTd3Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str,
        *,
        current_profile=None,
        agc_data_file=None,
        max_steps=None,
        dt: float = SIM_DT,
    ):
        super().__init__()
        self.scene = canonical_scene(scene)
        self.dt = float(dt)
        self.agc_data_file = (
            Path(AGC_DATA_FILE if agc_data_file is None else agc_data_file).resolve()
            if self.scene == "freq" and current_profile is None
            else None
        )
        self.current_profile = build_current_profile(
            self.scene,
            dt=self.dt,
            current_profile=current_profile,
            agc_data_file=self.agc_data_file,
            max_steps=max_steps,
        )
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(9,),
            dtype=np.float32,
        )
        self.pack = None
        self.last_info = {}
        self._done = False

    def _metrics(self) -> dict[str, float]:
        temps_c = self.pack.temps - 273.15
        dynamic = self.dynamic_state
        profile_index = min(self.step_index, len(self.current_profile) - 1)
        return {
            "current_a": float(self.current_profile[profile_index]),
            "mean_soc": float(np.mean(self.pack.socs)),
            "mean_temp_c": float(np.mean(temps_c)),
            "max_temp_c": float(np.max(temps_c)),
            "delta_temp_c": float(np.max(temps_c) - np.min(temps_c)),
            "tank_temp_c": float(self.t_tank_k - 273.15),
            "plate_mean_temp_c": float(np.mean(self.t_plate_k) - 273.15),
            "n_comp_eff_rpm": float(dynamic["N_comp_eff"]),
            "n_pump_eff_rpm": float(dynamic["N_pump_eff"]),
        }

    def _observation(self, metrics) -> np.ndarray:
        return np.asarray(
            [
                metrics["current_a"] / 1120.0,
                2.0 * metrics["mean_soc"] - 1.0,
                (metrics["mean_temp_c"] - 25.0) / 10.0,
                (metrics["max_temp_c"] - 25.0) / 10.0,
                metrics["delta_temp_c"] / 2.0,
                (metrics["tank_temp_c"] - 25.0) / 10.0,
                (metrics["plate_mean_temp_c"] - 25.0) / 10.0,
                2.0
                * (metrics["n_comp_eff_rpm"] - N_COMP_OFF_RPM)
                / (N_COMP_MAX_RPM - N_COMP_OFF_RPM)
                - 1.0,
                2.0
                * (metrics["n_pump_eff_rpm"] - N_PUMP_MIN_RPM)
                / (N_PUMP_MAX_RPM - N_PUMP_MIN_RPM)
                - 1.0,
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        initial_soc = INITIAL_SOC_REG if self.scene == "freq" else INITIAL_SOC_PEAK
        config = build_pack_config(
            total_current=float(self.current_profile[0]),
            initial_soc=initial_soc,
            initial_temp_c=INITIAL_TEMP_C,
        )
        self.pack = BatteryPack(config)
        initial_loop_temp_k = AMBIENT_TEMP_C + 273.15
        self.t_tank_k = initial_loop_temp_k
        self.t_plate_k = np.full(
            self.pack.cols,
            initial_loop_temp_k,
            dtype=float,
        )
        self.c_plate_node = PLATE_NODE_HEAT_CAPACITY_TOTAL / self.pack.cols
        self.dynamic_state = initialize_refrigeration_dynamic_state(
            N_COMP_OFF_RPM,
            N_PUMP_MIN_RPM,
            initial_temp_k=self.t_tank_k,
            dt=self.dt,
        )
        self.step_index = 0
        self._done = False
        metrics = self._metrics()
        self.last_info = {"scene": self.scene, "step_index": 0, **metrics}
        return self._observation(metrics), dict(self.last_info)
