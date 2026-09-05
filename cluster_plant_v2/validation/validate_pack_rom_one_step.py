"""Run the uniform-state one-step Reference-to-ROM acceptance check."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.pack_reference import ReferenceBatteryPack
from cluster_plant_v2.thermal.pack_rom import ReducedBatteryPack


def compare_one_step(total_current_A: float) -> dict[str, float]:
    reference = ReferenceBatteryPack()
    rom = ReducedBatteryPack()

    reference_started = time.perf_counter()
    reference.step(5.0, total_current_A, np.full(13, 298.15), 298.15)
    reference_runtime = time.perf_counter() - reference_started
    rom_started = time.perf_counter()
    rom.step(5.0, total_current_A, np.full(3, 298.15), 298.15)
    rom_runtime = time.perf_counter() - rom_started

    reference_soc = reference.socs.reshape(4, 13).mean(axis=1)
    reference_heat = float(reference.q_gen_cells.sum())
    heat_error = abs(rom.q_gen_total - reference_heat)
    return {
        "total_current_A": float(total_current_A),
        "branch_current_max_abs_error_A": float(
            np.max(np.abs(rom.branch_currents - reference.branch_currents))
        ),
        "soc_max_abs_error": float(np.max(np.abs(rom.soc_branch - reference_soc))),
        "reference_q_total_W": reference_heat,
        "rom_q_total_W": rom.q_gen_total,
        "q_total_abs_error_W": heat_error,
        "q_total_relative_error": heat_error / max(abs(reference_heat), 1e-12),
        "reference_weighted_Tavg_K": reference.get_avg_temp(),
        "rom_weighted_Tavg_K": rom.get_avg_temp(),
        "weighted_Tavg_abs_error_K": abs(
            rom.get_avg_temp() - reference.get_avg_temp()
        ),
        "reference_step_runtime_s": reference_runtime,
        "rom_step_runtime_s": rom_runtime,
        "runtime_ratio_rom_over_reference": rom_runtime / reference_runtime,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results = pd.DataFrame(
        [compare_one_step(current) for current in (280.0, 560.0, 1120.0)]
    )
    results.to_csv(args.output_dir / "pack_rom_one_step_comparison.csv", index=False)
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
