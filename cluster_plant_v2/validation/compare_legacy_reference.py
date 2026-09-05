"""Run aligned Legacy BatteryPack and ReferenceBatteryPack comparisons."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from cluster_plant_v2.pack_reference import ReferenceBatteryPack
from cluster_plant_v2.profiles import AGC_DATA_FILE, load_regd_profile


DEFAULT_DT_S = 5.0
DEFAULT_DURATION_S = 600.0
RESULTS_ROOT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    current_kind: str
    current_A: float | None
    plate_temp_c: float
    ambient_temp_c: float

    @property
    def current_condition(self) -> str:
        return "RegD [0, 600) s" if self.current_kind == "regd" else f"{self.current_A:g} A"


def build_case_specs() -> list[CaseSpec]:
    cases = []
    conditions = (("case0", 0.0), ("case1", 280.0), ("case2", 560.0), ("case3", 1120.0))
    for plate_temp_c, ambient_temp_c in ((25.0, 25.0), (20.0, 35.0)):
        boundary = f"p{int(plate_temp_c)}_a{int(ambient_temp_c)}"
        for label, current in conditions:
            cases.append(
                CaseSpec(
                    case_id=f"{label}_{int(current)}a_{boundary}",
                    current_kind="constant",
                    current_A=current,
                    plate_temp_c=plate_temp_c,
                    ambient_temp_c=ambient_temp_c,
                )
            )
        cases.append(
            CaseSpec(
                case_id=f"case4_regd_{boundary}",
                current_kind="regd",
                current_A=None,
                plate_temp_c=plate_temp_c,
                ambient_temp_c=ambient_temp_c,
            )
        )
    return cases


def _pack_config(total_current: float, *, neighbor_conductance: float = 0.5) -> dict:
    return {
        "rows": 4,
        "cols": 13,
        "capacity": 280.0,
        "initial_soc": 0.95,
        "initial_temp_c": 25.0,
        "total_current": float(total_current),
        "cell_thermal_mass": 4747.0,
        "k_intercell": float(neighbor_conductance),
        "h_plate_convection": 10.0,
        "h_air_convection_edge": 0.1,
        "h_air_convection_corner": 0.2,
        "uniform_air_convection": False,
        "dynamic_resistance_update": True,
    }


def _electrical_diagnostics(pack) -> tuple[np.ndarray, np.ndarray]:
    cell_heat = np.array(
        [float(sim.solution["Total heat generation [W]"].data[-1]) for sim in pack.sims]
    )
    cell_voltage = np.array(
        [float(sim.solution["Battery voltage [V]"].data[-1]) for sim in pack.sims]
    )
    return cell_heat, cell_voltage.reshape(4, 13).sum(axis=1)


def _state_record(
    *,
    time_s: float,
    current_A: float,
    temps: np.ndarray,
    socs: np.ndarray,
    branch_currents: np.ndarray,
    cell_heat: np.ndarray,
    branch_voltage: np.ndarray,
    neighbor_sum_W: float,
) -> dict[str, float]:
    temps_c = np.asarray(temps, dtype=float) - 273.15
    record = {
        "time_s": float(time_s),
        "current_A": float(current_A),
        "T_avg_C": float(np.mean(temps_c)),
        "T_max_C": float(np.max(temps_c)),
        "T_min_C": float(np.min(temps_c)),
        "DeltaT_C": float(np.ptp(temps_c)),
        "SOC_avg": float(np.mean(socs)),
        "SOC_max": float(np.max(socs)),
        "SOC_min": float(np.min(socs)),
        "Q_gen_total_W": float(np.sum(cell_heat)),
        "Q_gen_avg_W": float(np.mean(cell_heat)),
        "Q_gen_max_W": float(np.max(cell_heat)),
        "Q_neighbor_sum_W": float(neighbor_sum_W),
    }
    for index, value in enumerate(branch_currents, start=1):
        record[f"branch_current_{index}_A"] = float(value)
    for index, value in enumerate(branch_voltage, start=1):
        record[f"branch_voltage_{index}_V"] = float(value)
    return record


def _run_legacy(
    currents: np.ndarray, *, dt: float, plate_temp_k: float, ambient_temp_k: float
) -> tuple[pd.DataFrame, float]:
    from single_pack_plant.pack import BatteryPack

    started = time.perf_counter()
    pack = BatteryPack(_pack_config(float(currents[0])))
    plate = np.full(13, plate_temp_k)
    records = []
    for step_index, current in enumerate(currents, start=1):
        pack.current = float(current)
        pack.step(dt, plate, ambient_temp_k)
        cell_heat, branch_voltage = _electrical_diagnostics(pack)
        records.append(
            _state_record(
                time_s=step_index * dt,
                current_A=current,
                temps=pack.temps,
                socs=pack.socs,
                branch_currents=np.array(
                    [history[-1] for history in pack.branch_currents_history]
                ),
                cell_heat=cell_heat,
                branch_voltage=branch_voltage,
                neighbor_sum_W=np.nan,
            )
        )
    return pd.DataFrame(records), time.perf_counter() - started


def _run_reference(
    currents: np.ndarray,
    *,
    dt: float,
    plate_temp_k: float,
    ambient_temp_k: float,
    neighbor_conductance: float,
) -> tuple[pd.DataFrame, float]:
    started = time.perf_counter()
    pack = ReferenceBatteryPack(
        _pack_config(float(currents[0]), neighbor_conductance=neighbor_conductance)
    )
    plate = np.full(13, plate_temp_k)
    records = []
    for step_index, current in enumerate(currents, start=1):
        pack.step(dt, float(current), plate, ambient_temp_k)
        _, branch_voltage = _electrical_diagnostics(pack)
        neighbor_sum = float(np.sum(pack.calculate_neighbor_heat()))
        records.append(
            _state_record(
                time_s=step_index * dt,
                current_A=current,
                temps=pack.temps,
                socs=pack.socs,
                branch_currents=pack.branch_currents,
                cell_heat=pack.q_gen_cells,
                branch_voltage=branch_voltage,
                neighbor_sum_W=neighbor_sum,
            )
        )
    return pd.DataFrame(records), time.perf_counter() - started


def _prefixed(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    shared = {"time_s", "current_A"}
    return frame.rename(
        columns={column: f"{prefix}_{column}" for column in frame.columns if column not in shared}
    )


def run_case(
    case: CaseSpec,
    *,
    duration_s: float,
    dt: float,
    regd_file: Path,
    output_dir: Path,
) -> None:
    if case.current_kind == "regd":
        source_times, currents = load_regd_profile(regd_file, duration_s=duration_s, dt=dt)
        source_description = f"{regd_file}; source interval [0, {duration_s:g}) s"
    else:
        source_times = np.arange(0.0, duration_s, dt)
        currents = np.full(source_times.size, float(case.current_A))
        source_description = f"constant {case.current_A:g} A"
    if currents.size == 0:
        raise ValueError("comparison case contains no time steps")

    plate_temp_k = case.plate_temp_c + 273.15
    ambient_temp_k = case.ambient_temp_c + 273.15
    legacy, legacy_runtime = _run_legacy(
        currents, dt=dt, plate_temp_k=plate_temp_k, ambient_temp_k=ambient_temp_k
    )
    reference, reference_runtime = _run_reference(
        currents,
        dt=dt,
        plate_temp_k=plate_temp_k,
        ambient_temp_k=ambient_temp_k,
        neighbor_conductance=0.5,
    )
    reference_no_neighbor, no_neighbor_runtime = _run_reference(
        currents,
        dt=dt,
        plate_temp_k=plate_temp_k,
        ambient_temp_k=ambient_temp_k,
        neighbor_conductance=0.0,
    )

    combined = _prefixed(legacy, "legacy").merge(
        _prefixed(reference, "reference"), on=["time_s", "current_A"], validate="one_to_one"
    )
    combined = combined.merge(
        _prefixed(reference_no_neighbor, "reference_no_neighbor"),
        on=["time_s", "current_A"],
        validate="one_to_one",
    )
    combined.to_csv(output_dir / f"{case.case_id}_timeseries.csv", index=False)

    metadata = {
        **asdict(case),
        "current_condition": case.current_condition,
        "dt_s": float(dt),
        "duration_s": float(duration_s),
        "step_count": int(currents.size),
        "initial_soc": 0.95,
        "initial_cell_temp_c": 25.0,
        "current_source": source_description,
        "source_first_time_s": float(source_times[0]),
        "source_last_time_s": float(source_times[-1]),
        "current_min_A": float(np.min(currents)),
        "current_max_A": float(np.max(currents)),
        "legacy_runtime_s": legacy_runtime,
        "reference_runtime_s": reference_runtime,
        "reference_no_neighbor_runtime_s": no_neighbor_runtime,
        "reference_over_legacy_runtime_ratio": reference_runtime / legacy_runtime,
        "runtime_scope": "model construction plus all pack steps",
        "legacy_q_gen_variable": "Total heat generation [W]",
        "reference_q_gen_variable": "Total heat generation [W]",
        "cell_voltage_variable": "Battery voltage [V]",
    }
    (output_dir / f"{case.case_id}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT_S)
    parser.add_argument("--regd-file", type=Path, default=AGC_DATA_FILE)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()

    output_dir = args.output_dir or RESULTS_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    selected = set(args.case_id or [])
    cases = [case for case in build_case_specs() if not selected or case.case_id in selected]
    if selected.difference(case.case_id for case in cases):
        raise ValueError(f"unknown case ids: {sorted(selected.difference(case.case_id for case in cases))}")
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] running {case.case_id}", flush=True)
        run_case(
            case,
            duration_s=args.duration_s,
            dt=args.dt,
            regd_file=args.regd_file,
            output_dir=output_dir,
        )
    print(f"raw comparison results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
