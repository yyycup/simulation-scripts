"""Diagnose the refrigeration-envelope failure during static-preview NMPC on RegD."""

import sys
import time
import warnings

sys.path.insert(
    0,
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制",
)
warnings.filterwarnings("ignore")

import numpy as np

from cluster_plant_v2 import refrigeration as refrig

_history: list[dict] = []
_original = refrig.ClosedR134aCycle.solve


def instrumented_solve(self, **kwargs):
    _history.append({k: float(v) for k, v in kwargs.items()})
    try:
        return _original(self, **kwargs)
    except ValueError as exc:
        print("\n=== CAPTURED FAILURE ===", flush=True)
        print("exception:", exc)
        print("failed call kwargs:", kwargs)
        print("last 8 prior calls:")
        for row in _history[-9:-1]:
            print(
                {k: round(v, 3) for k, v in row.items()},
            )
        sys.stdout.flush()
        raise


refrig.ClosedR134aCycle.solve = instrumented_solve

from cluster_plant_v2.validation.validate_domp_mpc import build_cases
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc

case = [c for c in build_cases() if c.case_id == "T2_regd"][0]
start = time.time()
try:
    frame = run_nmpc(case, duration_s=6400.0, dt_s=5.0)
    frame.to_csv(sys.path[0] + "/_diag_static_partial.csv", index=False)
    print(f"completed without failure in {time.time() - start:.0f} s")
except Exception as exc:
    print(f"outer handler captured: {type(exc).__name__}: {exc}")
