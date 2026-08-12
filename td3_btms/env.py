from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_batch_config import AGC_DATA_FILE, SIM_DT


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


class BTMSTd3Env:
    pass
