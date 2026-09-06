"""Stage 2B validation: EvaporatorHeatCurrent vs the incumbent cycle.

Scope
-----
This is an **interface-equivalence** check, not a physics upgrade. The claim
under test is that the new stand-alone heat-current unit reproduces the
incumbent ``refrigeration.ClosedR134aCycle`` epsilon-NTU exchanger relation
bit-for-bit (up to floating-point associativity), and that its two outlet
calibers close energy exactly.

Nothing here modifies ``refrigeration.py`` and nothing is wired into the plant.

Outputs
-------
* ``evaporator_heat_current_stage2b_grid.csv``  -- per-case comparison
* ``evaporator_heat_current_stage2b_summary.txt`` -- aggregate metrics
"""

from __future__ import annotations

import csv
from pathlib import Path

from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    EvaporatorThermalDynamics,
)
from cluster_plant_v2.thermal.evaporator_heat_current import EvaporatorHeatCurrent

RESULTS_DIR = Path(__file__).resolve().parent / (
    "results/heat_current_stage2b_evaporator_20260906"
)

COMPRESSOR_SPEEDS_RPM = (1200.0, 1800.0, 2400.0, 3000.0, 3600.0, 4200.0, 4800.0)
MASS_FLOWS_KG_S = (0.3, 0.6, 0.9, 1.2)
INLET_TEMPERATURES_K = (288.15, 293.15, 298.15, 303.15)
AMBIENT_TEMPERATURES_K = (298.15, 308.15, 318.15)
FAN_SPEED_RPM = 3000.0

#: The incumbent solver bounds T_cond >= T_amb + 0.5 and T_e <= T_in - 0.5.
#: Those intervals overlap once the coolant inlet approaches ambient, so the
#: solver can step into ``T_cond <= T_e`` and raise. The chiller always
#: rejects heat to a warmer ambient, so require a margin instead of touching
#: ``refrigeration.py``.
AMBIENT_MARGIN_ABOVE_INLET_K = 5.0


def _relative_error(actual: float, reference: float) -> float:
    return abs(actual - reference) / max(abs(reference), 1e-9)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cycle = ClosedR134aCycle()
    model = EvaporatorHeatCurrent()

    rows: list[dict[str, float]] = []
    solved_cases = 0
    attempted_cases = 0
    failed_cases = 0
    failure_reasons: dict[str, int] = {}
    failures: list[dict[str, object]] = []
    for speed in COMPRESSOR_SPEEDS_RPM:
        for mass_flow in MASS_FLOWS_KG_S:
            for inlet in INLET_TEMPERATURES_K:
                for ambient in AMBIENT_TEMPERATURES_K:
                    if ambient < inlet + AMBIENT_MARGIN_ABOVE_INLET_K:
                        continue
                    attempted_cases += 1
                    try:
                        solution = cycle.solve(
                            compressor_speed_rpm=speed,
                            fan_speed_rpm=FAN_SPEED_RPM,
                            coolant_inlet_temperature_k=inlet,
                            coolant_mass_flow_kg_s=mass_flow,
                            ambient_temperature_k=ambient,
                        )
                    except ValueError as exc:
                        failed_cases += 1
                        failure_reasons[str(exc)] = (
                            failure_reasons.get(str(exc), 0) + 1
                        )
                        failures.append(
                            {
                                "speed_rpm": speed,
                                "mass_flow_kg_s": mass_flow,
                                "inlet_k": inlet,
                                "ambient_k": ambient,
                                "reason": str(exc),
                            }
                        )
                        continue
                    if not bool(solution["solver_success"]):
                        failed_cases += 1
                        failure_reasons[str(solution["solver_message"])] = (
                            failure_reasons.get(
                                str(solution["solver_message"]), 0
                            )
                            + 1
                        )
                        failures.append(
                            {
                                "speed_rpm": speed,
                                "mass_flow_kg_s": mass_flow,
                                "inlet_k": inlet,
                                "ambient_k": ambient,
                                "reason": str(solution["solver_message"]),
                            }
                        )
                        continue
                    solved_cases += 1

                    t_e = float(
                        solution["evaporating_saturation_temperature_k"]
                    )
                    ua_e = float(solution["evaporator_ua_w_k"])
                    q_ntu = float(solution["q_evaporator_ntu_w"])
                    q_cycle = float(solution["q_evaporator_w"])

                    result = model.evaluate(
                        coolant_inlet_temperature_k=inlet,
                        coolant_mass_flow_kg_s=mass_flow,
                        evaporating_temperature_k=t_e,
                        evaporator_ua_w_k=ua_e,
                    )
                    q_hc = float(result["q_hc_w"])
                    t_out_ss = float(result["coolant_outlet_temperature_ss_k"])
                    capacity = float(result["coolant_capacity_rate_w_k"])

                    # Dynamic caliber: 45 s lag then the applied outlet.
                    # Start from a cold evaporator (Q_applied = 0) so the lag
                    # is actually exercised; seeding it with q_cycle would
                    # make the lag a no-op and hide the ss/applied gap.
                    dynamics = EvaporatorThermalDynamics(
                        initial_q_evap_applied_w=0.0
                    )
                    q_applied = float(
                        dynamics.step(dt_s=5.0, q_evap_cycle_w=q_cycle)[
                            "q_evap_applied_w"
                        ]
                    )
                    t_out_applied = model.outlet_temperature_from_applied_heat(
                        coolant_inlet_temperature_k=inlet,
                        coolant_mass_flow_kg_s=mass_flow,
                        q_applied_w=q_applied,
                    )

                    rows.append(
                        {
                            "speed_rpm": speed,
                            "mass_flow_kg_s": mass_flow,
                            "inlet_k": inlet,
                            "ambient_k": ambient,
                            "t_e_k": t_e,
                            "ua_e_w_k": ua_e,
                            "effectiveness": float(
                                result["evaporator_effectiveness"]
                            ),
                            "g_c_w_k": capacity,
                            "r_e_c_k_w": float(
                                result["evaporator_resistance_k_w"]
                            ),
                            "q_ntu_w": q_ntu,
                            "q_hc_w": q_hc,
                            "q_cycle_w": q_cycle,
                            "q_applied_w": q_applied,
                            "t_out_ss_k": t_out_ss,
                            "t_out_applied_k": t_out_applied,
                            "err_q_hc_vs_ntu_rel": _relative_error(q_hc, q_ntu),
                            "err_q_hc_vs_cycle_rel": _relative_error(
                                q_hc, q_cycle
                            ),
                            "err_energy_ss_w": abs(
                                capacity * (inlet - t_out_ss) - q_hc
                            ),
                            "err_energy_applied_w": abs(
                                capacity * (inlet - t_out_applied) - q_applied
                            ),
                            "outlet_gap_ss_vs_applied_k": abs(
                                t_out_ss - t_out_applied
                            ),
                        }
                    )

    csv_path = RESULTS_DIR / "evaporator_heat_current_stage2b_grid.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    if failures:
        failures_path = (
            RESULTS_DIR / "evaporator_heat_current_stage2b_failures.csv"
        )
        with failures_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(failures[0].keys())
            )
            writer.writeheader()
            writer.writerows(failures)

    worst_q_ntu = max(r["err_q_hc_vs_ntu_rel"] for r in rows)
    worst_q_cycle = max(r["err_q_hc_vs_cycle_rel"] for r in rows)
    worst_energy_ss = max(r["err_energy_ss_w"] for r in rows)
    worst_energy_applied = max(r["err_energy_applied_w"] for r in rows)
    skipped_cases = (
        len(COMPRESSOR_SPEEDS_RPM)
        * len(MASS_FLOWS_KG_S)
        * sum(
            1
            for inlet in INLET_TEMPERATURES_K
            for ambient in AMBIENT_TEMPERATURES_K
            if ambient < inlet + AMBIENT_MARGIN_ABOVE_INLET_K
        )
    )
    eff_min = min(r["effectiveness"] for r in rows)
    eff_max = max(r["effectiveness"] for r in rows)
    t_out_gap_max = max(r["outlet_gap_ss_vs_applied_k"] for r in rows)

    summary = f"""Stage 2B -- EvaporatorHeatCurrent vs incumbent cycle
====================================================
grid points attempted             : {attempted_cases}
solved cases                       : {solved_cases}
skipped (envelope guard)           : {skipped_cases}
failed (solver)                    : {failed_cases}

Identity check (the hard requirement)
-------------------------------------
max |Q_HC - Q_NTU| / Q_NTU         : {worst_q_ntu:.3e}
max |Q_HC - Q_cycle| / Q_cycle     : {worst_q_cycle:.3e}   (solver closure, not identity)

Energy closure
--------------
max |G_c(T_in - T_out_ss) - Q_HC|  : {worst_energy_ss:.3e} W
max |G_c(T_in - T_out_ap) - Q_ap|  : {worst_energy_applied:.3e} W

Operating envelope observed
---------------------------
evaporator effectiveness           : {eff_min:.4f} .. {eff_max:.4f}
max |T_out_ss - T_out_applied|     : {t_out_gap_max:.4f} K

Artifacts
---------
grid CSV                           : {csv_path.name}
"""
    summary_path = (
        RESULTS_DIR / "evaporator_heat_current_stage2b_summary.txt"
    )
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
