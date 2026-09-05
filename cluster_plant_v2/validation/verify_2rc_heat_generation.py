"""Independently reconstruct ReferenceBatteryPack 2RC heat generation."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pybamm

from cluster_plant_v2.pack_reference import ReferenceBatteryPack


DT_S = 5.0
DURATION_S = 60.0
PLATE_TEMPERATURE_K = 298.15
AMBIENT_TEMPERATURE_K = 298.15
EPS = 1e-12
REPRESENTATIVE_TIMES_S = {5.0, 30.0, 60.0}
REPRESENTATIVE_CELLS = {0, 13, 26, 39}


def manual_heat_components(
    *,
    current_A: float,
    r0_ohm: float,
    eta1_V: float,
    eta2_V: float,
    q_reversible_W: float,
) -> dict[str, float]:
    """Reproduce the PyBaMM 26.5.0 equivalent-circuit heat definitions."""
    q_r0 = current_A**2 * r0_ohm
    q_rc1 = -current_A * eta1_V
    q_rc2 = -current_A * eta2_V
    total = q_r0 + q_rc1 + q_rc2 + q_reversible_W
    return {
        "Q_R0_W": float(q_r0),
        "Q_RC1_W": float(q_rc1),
        "Q_RC2_W": float(q_rc2),
        "Q_reversible_W": float(q_reversible_W),
        "Q_manual_W": float(total),
    }


def pack_heat_interface_error(public_heat, solution_heat) -> float:
    public_heat = np.asarray(public_heat, dtype=float)
    solution_heat = np.asarray(solution_heat, dtype=float)
    if public_heat.shape != (52,) or solution_heat.shape != (52,):
        raise ValueError("both heat arrays must have shape (52,)")
    return float(abs(public_heat.sum() - solution_heat.sum()))


def summarize_case(case: str, frame: pd.DataFrame) -> dict[str, float | str]:
    total = float(frame["Q_manual_W"].sum())
    pack_totals = frame.groupby("time_s", sort=False)["Q_pybamm_W"].sum()
    return {
        "case": case,
        "mean_Q_R0_W_per_cell": float(frame["Q_R0_W"].mean()),
        "mean_Q_RC1_W_per_cell": float(frame["Q_RC1_W"].mean()),
        "mean_Q_RC2_W_per_cell": float(frame["Q_RC2_W"].mean()),
        "mean_Q_reversible_W_per_cell": float(frame["Q_reversible_W"].mean()),
        "mean_Q_total_W_per_cell": float(frame["Q_pybamm_W"].mean()),
        "mean_Q_total_W_pack": float(pack_totals.mean()),
        "R0_contribution_ratio": float(frame["Q_R0_W"].sum() / total) if total else np.nan,
        "RC1_contribution_ratio": float(frame["Q_RC1_W"].sum() / total) if total else np.nan,
        "RC2_contribution_ratio": float(frame["Q_RC2_W"].sum() / total) if total else np.nan,
        "max_abs_error_W": float(frame["abs_error_W"].max()),
        "max_relative_error": float(frame["relative_error"].max()),
        "mean_abs_error_W": float(frame["abs_error_W"].mean()),
        "max_pack_heat_sum_abs_error_W": float(
            frame["pack_heat_sum_abs_error_W"].max()
        ),
    }


def _value(solution, name: str) -> float:
    return float(solution[name].data[-1])


def _extract_step(pack: ReferenceBatteryPack, case: str, time_s: float) -> list[dict]:
    solution_heat = np.empty(52, dtype=float)
    solutions = []
    for cell_index, simulation in enumerate(pack.sims):
        solution = simulation.solution
        q_pybamm = _value(solution, "Total heat generation [W]")
        solution_heat[cell_index] = q_pybamm
        solutions.append(solution)
    pack_sum_error = pack_heat_interface_error(pack.q_gen_cells, solution_heat)

    records = []
    for cell_index, solution in enumerate(solutions):
        current = _value(solution, "Current [A]")
        r0 = _value(solution, "R0 [Ohm]")
        eta1 = _value(solution, "Element-1 overpotential [V]")
        eta2 = _value(solution, "Element-2 overpotential [V]")
        q_reversible = _value(solution, "Reversible heat generation [W]")
        components = manual_heat_components(
            current_A=current,
            r0_ohm=r0,
            eta1_V=eta1,
            eta2_V=eta2,
            q_reversible_W=q_reversible,
        )
        q_pybamm = solution_heat[cell_index]
        abs_error = abs(components["Q_manual_W"] - q_pybamm)
        records.append(
            {
                "case": case,
                "time_s": time_s,
                "cell_index": cell_index,
                "branch_index": cell_index // 13,
                "current_A": current,
                "soc": _value(solution, "SoC"),
                "temperature_K": _value(solution, "Cell temperature [K]"),
                "pack_temperature_after_step_K": float(pack.temps[cell_index]),
                "R0_ohm": r0,
                "R1_ohm": _value(solution, "R1 [Ohm]"),
                "R2_ohm": _value(solution, "R2 [Ohm]"),
                "eta1_V": eta1,
                "eta2_V": eta2,
                "entropic_change_V_per_K": _value(
                    solution, "Entropic change [V/K]"
                ),
                **components,
                "Q_R0_pybamm_W": _value(
                    solution, "Element-0 irreversible heat generation [W]"
                ),
                "Q_RC1_pybamm_W": _value(
                    solution, "Element-1 irreversible heat generation [W]"
                ),
                "Q_RC2_pybamm_W": _value(
                    solution, "Element-2 irreversible heat generation [W]"
                ),
                "Q_irreversible_pybamm_W": _value(
                    solution, "Irreversible heat generation [W]"
                ),
                "Q_pybamm_W": q_pybamm,
                "abs_error_W": abs_error,
                "relative_error": abs_error / max(abs(q_pybamm), EPS),
                "pack_public_heat_sum_W": float(pack.q_gen_cells.sum()),
                "solution_heat_sum_W": float(solution_heat.sum()),
                "pack_heat_sum_abs_error_W": pack_sum_error,
            }
        )
    return records


def _run_profile(case: str, currents: np.ndarray) -> pd.DataFrame:
    pack = ReferenceBatteryPack()
    plate = np.full(13, PLATE_TEMPERATURE_K)
    records = []
    for step_index, current in enumerate(currents, start=1):
        pack.step(
            DT_S,
            float(current),
            plate,
            AMBIENT_TEMPERATURE_K,
        )
        records.extend(_extract_step(pack, case, step_index * DT_S))
    return pd.DataFrame(records)


def _source_evidence() -> dict:
    from pybamm.models.submodels.equivalent_circuit_elements.ocv_element import (
        OCVElement,
    )
    from pybamm.models.submodels.equivalent_circuit_elements.rc_element import RCElement
    from pybamm.models.submodels.equivalent_circuit_elements.resistor_element import (
        ResistorElement,
    )

    sources = {}
    for cls in (ResistorElement, RCElement, OCVElement):
        lines, start = inspect.getsourcelines(cls)
        sources[cls.__name__] = {
            "path": inspect.getsourcefile(cls),
            "line_start": start,
            "line_end": start + len(lines) - 1,
        }
    return {
        "pybamm_version": pybamm.__version__,
        "formula": {
            "Q_R0_W": "Current [A]^2 * R0 [Ohm]",
            "Q_RC1_W": "-Current [A] * Element-1 overpotential [V]",
            "Q_RC2_W": "-Current [A] * Element-2 overpotential [V]",
            "Q_reversible_W": "-Current [A] * Cell temperature [K] * Entropic change [V/K]",
            "Q_total_W": "Q_R0_W + Q_RC1_W + Q_RC2_W + Q_reversible_W",
        },
        "solution_time_convention": "all variables read from the same Simulation.step() end point",
        "sources": sources,
    }


def _plot_components(frame: pd.DataFrame, output_path: Path) -> None:
    cases = ("280A", "560A", "1120A")
    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True, constrained_layout=True)
    for axis, case in zip(axes, cases):
        cell = frame[(frame["case"] == case) & (frame["cell_index"] == 0)]
        for column, label in (
            ("Q_R0_W", "R0"),
            ("Q_RC1_W", "RC1"),
            ("Q_RC2_W", "RC2"),
            ("Q_pybamm_W", "PyBaMM total"),
        ):
            axis.plot(cell["time_s"], cell[column], marker="o", label=label)
        axis.set(title=case, ylabel="Heat / W")
        axis.grid(alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("Time / s")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_verification(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    steps = int(DURATION_S / DT_S)
    profiles = {
        "0A_initial": np.zeros(steps),
        "280A": np.full(steps, 280.0),
        "560A": np.full(steps, 560.0),
        "1120A": np.full(steps, 1120.0),
        "560A_then_0A": np.concatenate((np.full(6, 560.0), np.zeros(6))),
    }
    frames = []
    for case, currents in profiles.items():
        print(f"running {case}", flush=True)
        frames.append(_run_profile(case, currents))
    verification = pd.concat(frames, ignore_index=True)
    verification.to_csv(output_dir / "heat_generation_verification.csv", index=False)

    summaries = pd.DataFrame(
        [
            summarize_case(case, verification[verification["case"] == case])
            for case in profiles
        ]
    )
    summaries.to_csv(output_dir / "heat_generation_summary.csv", index=False)
    representative = verification[
        verification["time_s"].isin(REPRESENTATIVE_TIMES_S)
        & verification["cell_index"].isin(REPRESENTATIVE_CELLS)
    ]
    representative.to_csv(output_dir / "representative_samples.csv", index=False)
    relaxation = verification[
        (verification["case"] == "560A_then_0A")
        & (verification["time_s"] >= 30.0)
        & (verification["cell_index"] == 0)
    ]
    relaxation.to_csv(output_dir / "polarization_relaxation_check.csv", index=False)
    (output_dir / "pybamm_formula_evidence.json").write_text(
        json.dumps(_source_evidence(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot_components(verification, output_dir / "heat_components_cell0.png")
    print(summaries.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    run_verification(args.output_dir)


if __name__ == "__main__":
    main()
