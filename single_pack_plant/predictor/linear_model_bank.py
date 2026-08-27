"""Offline Physics-P linear-model bank and runtime operating-point selection.

The bank stores scaled affine models generated offline by
``PhysicsPStateSpace.linearize_scaled``.  Runtime selection only computes a
small nearest-point distance; it never evaluates Physics-P or a Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .state_space import ND, NU, NX, LinearizedDiscreteModel


_SELECTION_SCALES = np.array([5.0, 5.0, 2000.0, 1000.0, 560.0, 10.0])


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have finite shape {shape}")
    return result


@dataclass(frozen=True)
class LinearModelBankEntry:
    label: str
    x_nominal: np.ndarray
    u_nominal: np.ndarray
    d_nominal: np.ndarray
    model: LinearizedDiscreteModel

    @property
    def selection_point(self) -> np.ndarray:
        return np.array(
            [
                self.x_nominal[0],
                self.x_nominal[1],
                self.u_nominal[0],
                self.u_nominal[1],
                self.d_nominal[0],
                self.d_nominal[1],
            ],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "x_nominal": self.x_nominal.tolist(),
            "u_nominal": self.u_nominal.tolist(),
            "d_nominal": self.d_nominal.tolist(),
            "A": self.model.A.tolist(),
            "B": self.model.B.tolist(),
            "E": self.model.E.tolist(),
            "c": self.model.c.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "LinearModelBankEntry":
        x = _array(payload["x_nominal"], (NX,), "x_nominal")
        u = _array(payload["u_nominal"], (NU,), "u_nominal")
        d = _array(payload["d_nominal"], (ND,), "d_nominal")
        model = LinearizedDiscreteModel(
            A=_array(payload["A"], (NX, NX), "A"),
            B=_array(payload["B"], (NX, NU), "B"),
            E=_array(payload["E"], (NX, ND), "E"),
            c=_array(payload["c"], (NX,), "c"),
            x_nominal=x,
            u_nominal=u,
            d_nominal=d,
        )
        return cls(str(payload["label"]), x, u, d, model)


class OfflinePhysicsPLinearModelBank:
    """Nearest operating-point lookup for precomputed Physics-P Jacobians."""

    schema = "physics_p_linear_model_bank_v1"

    def __init__(self, entries: Sequence[LinearModelBankEntry], dt_s: float):
        if not entries:
            raise ValueError("linear model bank must contain at least one entry")
        self.entries = tuple(entries)
        self.dt_s = float(dt_s)
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

    @classmethod
    def load(cls, path: str | Path, *, dt_s: float | None = None):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema") != cls.schema:
            raise ValueError("unsupported Physics-P linear model bank schema")
        bank_dt = float(payload["dt_s"])
        if dt_s is not None and not np.isclose(bank_dt, float(dt_s)):
            raise ValueError(f"model bank dt={bank_dt} does not match controller dt={dt_s}")
        entries = [LinearModelBankEntry.from_dict(item) for item in payload["entries"]]
        return cls(entries, bank_dt)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.schema,
            "dt_s": self.dt_s,
            "selection_features": ["t_bat", "t_tank", "n_comp_cmd", "n_pump_cmd", "current", "t_ambient"],
            "selection_scales": _SELECTION_SCALES.tolist(),
            "entries": [entry.to_dict() for entry in self.entries],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def select(self, state, control, disturbance) -> LinearizedDiscreteModel:
        x = _array(state, (NX,), "state")
        u = _array(control, (NU,), "control")
        d = _array(disturbance, (ND,), "disturbance")
        query = np.array([x[0], x[1], u[0], u[1], d[0], d[1]], dtype=float)
        points = np.asarray([entry.selection_point for entry in self.entries])
        distances = np.sum(((points - query) / _SELECTION_SCALES) ** 2, axis=1)
        return self.entries[int(np.argmin(distances))].model

    def select_trajectory(self, initial_state, controls, disturbances):
        controls = np.asarray(controls, dtype=float)
        disturbances = np.asarray(disturbances, dtype=float)
        if controls.ndim != 2 or controls.shape[1] != NU:
            raise ValueError("controls must have shape (N, 2)")
        if disturbances.shape != (controls.shape[0], ND):
            raise ValueError("disturbances must have shape (N, 2)")
        state = _array(initial_state, (NX,), "initial_state")
        models = []
        for control, disturbance in zip(controls, disturbances):
            model = self.select(state, control, disturbance)
            models.append(model)
            # The stored matrices are scaled-coordinate models.  Keep the
            # measured physical state fixed for selection; prediction itself
            # is handled by QP's scaled prediction maps.
        return models
