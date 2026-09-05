"""Stage-level diagnosis: wrap every Plant sub-component and dump the fatal step."""

import sys
import time
import warnings

sys.path.insert(
    0,
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制",
)
warnings.filterwarnings("ignore")

import numpy as np

from cluster_plant_v2 import plant as plant_mod
from cluster_plant_v2 import refrigeration as refrig

_trace: list[str] = []


def log(msg: str) -> None:
    _trace.append(msg)
    if len(_trace) > 60:
        del _trace[: len(_trace) - 60]


_orig_refri = refrig.ClosedR134aCycle.solve
_orig_evap = plant_mod.EvaporatorThermalDynamics.step
_orig_tank = plant_mod.CoolantTank.step
_orig_cluster_step = None
_orig_plant_step = plant_mod.ClusterPlant.step


def refri(self, **kw):
    out = _orig_refri(self, **kw)
    log(
        f"refri: Tin={kw['coolant_inlet_temperature_k']:.4f} "
        f"m={kw['coolant_mass_flow_kg_s']:.4f} "
        f"n={kw['compressor_speed_rpm']:.1f} "
        f"q_evap={float(out['q_evaporator_w']):.2f}"
    )
    return out


def evap(self, *, dt_s, q_evap_cycle_w):
    out = _orig_evap(self, dt_s=dt_s, q_evap_cycle_w=q_evap_cycle_w)
    log(f"evap: q_cycle={q_evap_cycle_w:.2f} q_applied={out['q_evap_applied_w']:.2f}")
    return out


def tank(self, dt_s, return_temperature_k, mass_flow_kg_s):
    out = _orig_tank(self, dt_s, return_temperature_k, mass_flow_kg_s)
    log(
        f"tank: Tret={return_temperature_k:.4f} m={mass_flow_kg_s:.4f} "
        f"Tafter={out['tank_temperature_after_k']:.4f}"
    )
    return out


def plant_step(self, *args, **kwargs):
    step_index = getattr(self, "_diag_step", 0)
    self._diag_step = step_index + 1
    try:
        return _orig_plant_step(self, *args, **kwargs)
    except Exception as exc:
        print(f"\n=== FATAL at plant step {step_index} ===", flush=True)
        print(f"exception: {type(exc).__name__}: {exc}")
        print(f"tank T: {self.tank.temperature_k}")
        print("supply queue:", self.supply_delay.queue_values)
        print("return queue:", self.return_delay.queue_values)
        print(
            "evap applied / buffer:",
            self.evaporator_dynamics.q_evap_applied_w,
            self.evaporator_dynamics.evaporator_buffer_energy_j,
        )
        print("recent trace:")
        for line in _trace[-30:]:
            print(" ", line)
        sys.stdout.flush()
        raise


refrig.ClosedR134aCycle.solve = refri
plant_mod.EvaporatorThermalDynamics.step = evap
plant_mod.CoolantTank.step = tank
plant_mod.ClusterPlant.step = plant_step

from cluster_plant_v2.validation.validate_domp_mpc import build_cases
from cluster_plant_v2.validation.validate_physics_p_nmpc import run_nmpc

case = [c for c in build_cases() if c.case_id == "T2_regd"][0]
start = time.time()
try:
    frame = run_nmpc(case, duration_s=6400.0, dt_s=5.0)
    print(f"completed without failure in {time.time() - start:.0f} s")
except Exception as exc:
    print(f"outer handler captured: {type(exc).__name__}: {exc}")
