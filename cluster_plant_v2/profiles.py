"""Current-profile inputs owned by Cluster Plant V2 validation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


AGC_DATA_FILE = Path(
    os.environ.get(
        "THERMAL_AGC_DATA_FILE",
        r"C:\Users\24776\Desktop\科研\论文\小论文\数据\PJM_RegD_MaxLoad_2h.csv",
    )
)


def load_regd_profile(
    path: str | Path,
    *,
    duration_s: float = 600.0,
    dt: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    required = {"Seconds", "RegD"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"RegD file is missing columns: {missing}")
    source_times = frame["Seconds"].to_numpy(dtype=float)
    source_signal = frame["RegD"].to_numpy(dtype=float)
    if source_times.size == 0 or not np.all(np.isfinite(source_times)):
        raise ValueError("RegD Seconds must be non-empty and finite")
    if source_times.size > 1 and not np.all(np.diff(source_times) > 0.0):
        raise ValueError("RegD Seconds must be strictly increasing")
    if not np.all(np.isfinite(source_signal)):
        raise ValueError("RegD values must be finite")
    times = np.arange(0.0, float(duration_s), float(dt))
    currents = np.interp(times, source_times, source_signal) * 1120.0
    return times, currents
