"""Fixed offline scales for dimensionless MPC objective diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Mapping

import numpy as np


OBJECTIVE_KEYS = (
    "cv",
    "temperature",
    "terminal",
    "compressor",
    "pump",
    "dcomp",
    "dpump",
)


@dataclass(frozen=True)
class ObjectiveNormalizationScales:
    """Positive, fixed normalization denominators for MPC objective terms."""

    cv: float = 1.0
    temperature: float = 1.0
    terminal: float = 1.0
    compressor: float = 1.0
    pump: float = 1.0
    dcomp: float = 1.0
    dpump: float = 1.0

    def __post_init__(self):
        for key in OBJECTIVE_KEYS:
            value = float(getattr(self, key))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{key} scale must be finite and positive")
            object.__setattr__(self, key, value)

    def to_dict(self):
        return asdict(self)


def normalize_raw_terms(raw_terms: Mapping[str, float], scales):
    """Divide one complete raw-objective mapping by fixed scales."""
    if not isinstance(scales, ObjectiveNormalizationScales):
        scales = ObjectiveNormalizationScales(**dict(scales))
    normalized = {}
    for key in OBJECTIVE_KEYS:
        value = float(raw_terms[key])
        if not np.isfinite(value):
            raise ValueError(f"{key} raw objective must be finite")
        normalized[key] = value / getattr(scales, key)
    return normalized


def compensate_mpc_params_for_normalization(mpc_params, scales):
    """Scale nominal weights so fixed normalization is algebraically neutral."""
    if not isinstance(scales, ObjectiveNormalizationScales):
        scales = ObjectiveNormalizationScales(**dict(scales))
    return replace(
        mpc_params,
        w_high_temp=float(mpc_params.w_high_temp) * scales.cv,
        w_cold_temp=float(mpc_params.w_cold_temp) * scales.cv,
        w_temp_obj=float(mpc_params.w_temp_obj) * scales.temperature,
        w_terminal_temp=float(mpc_params.w_terminal_temp) * scales.terminal,
        w_energy_comp=float(mpc_params.w_energy_comp) * scales.compressor,
        w_energy_pump=float(mpc_params.w_energy_pump) * scales.pump,
        w_dcomp=float(mpc_params.w_dcomp) * scales.dcomp,
        w_dcomp_quadratic=(
            float(mpc_params.w_dcomp_quadratic) * scales.dcomp
        ),
        w_dpump=float(mpc_params.w_dpump) * scales.dpump,
    )


def robust_scale_statistics(values):
    """Summarize finite positive samples for one offline objective scale."""
    array = np.asarray(list(values), dtype=float)
    valid = array[np.isfinite(array) & (array > 0.0)]
    if valid.size == 0:
        raise ValueError("objective scale requires at least one positive sample")
    return {
        "count": int(valid.size),
        "median": float(np.median(valid)),
        "p10": float(np.percentile(valid, 10.0)),
        "p90": float(np.percentile(valid, 90.0)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
    }


def write_scale_candidate(path, scales, *, provenance):
    """Write a traceable fixed-scale candidate JSON file."""
    if not isinstance(scales, ObjectiveNormalizationScales):
        scales = ObjectiveNormalizationScales(**dict(scales))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "scales": scales.to_dict(),
        "provenance": dict(provenance),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path


def load_scale_candidate(path):
    """Load and validate a fixed-scale candidate JSON file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported objective normalization schema")
    scales = ObjectiveNormalizationScales(**payload["scales"])
    provenance = dict(payload.get("provenance", {}))
    return scales, provenance
