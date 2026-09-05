"""Validated HPPC parameter loading local to Cluster Plant V2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


HPPC_PARAMETERS_PATH = Path(__file__).resolve().parent / "model_data" / "hppc_params.json"


def load_hppc_parameters(path: Path = HPPC_PARAMETERS_PATH) -> dict[str, np.ndarray]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    soc = np.asarray(raw["soc"], dtype=float)
    temperature = np.asarray(raw["temp"], dtype=float)
    if soc.ndim != 1 or temperature.ndim != 1:
        raise ValueError("HPPC SOC and temperature axes must be one-dimensional")
    if not (np.all(np.diff(soc) > 0.0) and np.all(np.diff(temperature) > 0.0)):
        raise ValueError("HPPC axes must be strictly increasing")
    if np.any(temperature < 200.0) or np.any(temperature > 400.0):
        raise ValueError("HPPC temperature axis must be expressed in kelvin")

    expected_shape = (soc.size, temperature.size)
    resistance_keys = {
        "r0_dis",
        "r0_chg",
        "r1_dis",
        "r1_chg",
        "r2_dis",
        "r2_chg",
    }
    result = {"soc": soc, "temp": temperature}
    for key in (
        "ocv",
        "r0_dis",
        "r0_chg",
        "r1_dis",
        "r1_chg",
        "c1_dis",
        "c1_chg",
        "r2_dis",
        "r2_chg",
        "c2_dis",
        "c2_chg",
    ):
        values = np.asarray(raw[key], dtype=float)
        if values.shape != expected_shape:
            raise ValueError(
                f"HPPC table {key!r} has shape {values.shape}, expected {expected_shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"HPPC table {key!r} contains non-finite values")
        result[key] = values * (1e-3 if key in resistance_keys else 1.0)
    return result
