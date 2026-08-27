"""Generate a small offline Physics-P linear-model bank.

This is an offline preparation step. Runtime QP-MPC only selects among the
saved models and does not numerically linearize Physics-P.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from single_pack_plant.predictor.physics_p import initialize_physics_state
    from single_pack_plant.predictor.state_space import PhysicsPStateSpace
    from single_pack_plant.predictor.linear_model_bank import (
        LinearModelBankEntry,
        OfflinePhysicsPLinearModelBank,
    )
else:
    from ..predictor.physics_p import initialize_physics_state
    from ..predictor.state_space import PhysicsPStateSpace
    from ..predictor.linear_model_bank import (
        LinearModelBankEntry,
        OfflinePhysicsPLinearModelBank,
    )


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "model_data" / "physics_p_operational_v1.json"
OUTPUT = ROOT / "model_data" / "physics_p_linear_model_bank.json"


def build_bank(output: Path = OUTPUT) -> Path:
    state_space = PhysicsPStateSpace(ARTIFACT, dt_s=5.0)
    # Representative low/nominal/high load and actuator operating points.
    points = (
        ("low", 25.0, 35.0, 280.0, 2000.0, 2400.0),
        ("low_nominal", 25.0, 35.0, 280.0, 3200.0, 3200.0),
        ("nominal", 25.0, 35.0, 560.0, 4000.0, 3200.0),
        ("nominal_high_pump", 25.0, 35.0, 560.0, 4500.0, 4000.0),
        ("high", 25.0, 35.0, 840.0, 5000.0, 4000.0),
        ("high_max", 25.0, 35.0, 840.0, 6000.0, 4800.0),
    )
    entries = []
    for label, t_bat, t_cool, current, n_comp, n_pump in points:
        state = initialize_physics_state(
            n_comp, n_pump, 0.0, 0.0, t_cool, t_cool, t_cool, t_bat, t_cool
        )
        x = state_space.pack_state(state)
        u = np.array([n_comp, n_pump], dtype=float)
        d = np.array([current, 35.0], dtype=float)
        model = state_space.linearize_scaled(x, u, d)
        entries.append(LinearModelBankEntry(label, x, u, d, model))
    bank = OfflinePhysicsPLinearModelBank(entries, dt_s=5.0)
    bank.save(output)
    print(f"Wrote {len(entries)} offline Physics-P linear models: {output}")
    return output


if __name__ == "__main__":
    build_bank()
