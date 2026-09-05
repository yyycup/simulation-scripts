"""Pack-level probe: log per-pack thermal states each step; dump on divergence."""

import sys
import time
import warnings

sys.path.insert(
    0,
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制",
)
warnings.filterwarnings("ignore")

import numpy as np

from cluster_plant_v2.thermal import cluster as cluster_mod
from cluster_plant_v2 import plant as plant_mod

_records: list[str] = []
_orig_pack_step = cluster_mod.ReducedPack.step
_orig_plant_step = plant_mod.ClusterPlant.step


def pack_step(self, dt_s, current_a, supply_k, mass_flow, ambient_k, direction):
    result = _orig_pack_step(
        self, dt_s, current_a, supply_k, mass_flow, ambient_k, direction
    )
    batt = np.asarray(result["battery_zone_temperatures"], dtype=float)
    plate = np.asarray(result["plate_temperatures"], dtype=float)
    q = float(result["q_plate_to_fluid_total"])
    outlet = float(result["coolant_outlet_temperature"])
    soc = float(np.mean(result["soc"]))
    line = (
        f"pack flow={mass_flow:.4f} soc={soc:.4f} "
        f"batt={batt.mean():.3f}/{batt.max():.3f} "
        f"plate={plate.mean():.3f} q_pl={q:.1f} out={outlet:.3f}"
    )
    _records.append(line)
    if abs(outlet) > 400.0 or batt.max() > 400.0:
        print(f"\n=== DIVERGENCE DETECTED: {line}", flush=True)
        for prev in _records[-240:]:
            print(prev, flush=True)
        raise SystemExit(0)
    if len(_records) > 300:
        del _records[: len(_records) - 300]
    return result


def plant_step(self, *args, **kwargs):
    step_index = getattr(self, "_diag_step", 0)
    self._diag_step = step_index + 1
    _records.append(
        f"STEP {step_index} kwargs={ {k: v for k, v in kwargs.items() if k != 'inputs'} }"
    )
    return _orig_plant_step(self, *args, **kwargs)


cluster_mod.ReducedPack.step = pack_step
plant_mod.ClusterPlant.step = plant_step

from cluster_plant_v2.validation.validate_domp_mpc import build_cases
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc

case = [c for c in build_cases() if c.case_id == "T2_regd"][0]
start = time.time()
try:
    frame = run_nmpc(case, duration_s=6400.0, dt_s=5.0)
    print(f"completed without failure in {time.time() - start:.0f} s")
except SystemExit:
    print("stopped after divergence dump")
except Exception as exc:
    print(f"outer handler captured: {type(exc).__name__}: {exc}")
