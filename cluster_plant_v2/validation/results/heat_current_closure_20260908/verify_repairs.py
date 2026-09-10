"""Run from workspace parent with btms Python; preserves before_trajectory.json."""
import copy
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from cluster_plant_v2.tests.test_heat_current_plant_independence import (
    _build_legacy_plant, _constant_inputs,
)
from cluster_plant_v2.thermal.heat_current_plant import build_independent_hc_plant
from cluster_plant_v2.thermal.heat_current_energy_balance import (
    initial_energy_snapshot, ledger_from_legacy,
)
from cluster_plant_v2.thermal.heat_current_stage5_ledger import ledger_from_heat_current_plant
from cluster_plant_v2.validation import validate_heat_current_stage4 as stage4
from cluster_plant_v2.validation import validate_heat_current_stage5 as stage5


OUT = Path(__file__).resolve().parent
DT = 5.0


def baseline_comparison(seed):
    before = json.loads((OUT / "before_trajectory.json").read_text(encoding="utf-8"))
    plant = build_independent_hc_plant(legacy_plant=seed)
    after = []
    for k in range(12):
        inputs = replace(_constant_inputs(), cluster_current_a=560.0 if k < 4 else 800.0,
            pump_rpm=3600.0 if k < 6 else 4500.0,
            compressor_command_rpm=4000.0 if k < 8 else 3000.0,
            flow_direction="forward" if k < 9 else "reverse")
        result = plant.step(inputs, dt_s=DT)
        after.append({key: result[key] for key in before[k]})
    differences = {key: max(abs(a[key] - b[key]) for a, b in zip(before, after)) for key in before[0]}
    (OUT / "after_trajectory.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    assert max(differences.values()) == 0.0, differences
    return {"steps": 12, "duration_s": 60.0, "max_abs_difference": max(differences.values()),
            "per_field_max_abs_difference": differences}


def smoke(seed, case):
    reference_flow = seed.pump.solve_operating_point(3600.0, seed.hydraulic_network)["total_mass_flow_kg_s"]
    summaries = {}
    for hc in (False, True):
        plant = build_independent_hc_plant(legacy_plant=seed) if hc else copy.deepcopy(seed)
        prev = initial_energy_snapshot(plant, is_heat_current=hc,
            transport_reference_mass_flow_kg_s=reference_flow)
        rows = []
        for k in range(24):
            inputs = _constant_inputs()
            if case == "constant_flow_excitations":
                inputs = replace(inputs, cluster_current_a=800.0 if k >= 8 else 560.0,
                    compressor_command_rpm=3000.0 if k >= 12 else 4000.0,
                    flow_direction="reverse" if k >= 16 else "forward")
            elif case == "flow_step":
                inputs = replace(inputs, pump_rpm=4500.0 if k >= 8 else 3600.0)
            if hc:
                result = plant.step(inputs, dt_s=DT)
                ledger, prev = ledger_from_heat_current_plant(plant, dt_s=DT,
                    step_result=result, prev_energy=prev,
                    compressor_speed_used_rpm=result["compressor_speed_used_rpm"],
                    tank_temperature_before_k=result["tank_temperature_before_k"],
                    refrigeration_solver_success=result["refrigeration_solver_success"])
            else:
                result = plant.step(dt_s=DT, cluster_current_a=inputs.cluster_current_a,
                    pump_speed_rpm=inputs.pump_rpm, compressor_speed_command_rpm=inputs.compressor_command_rpm,
                    fan_speed_rpm=inputs.fan_rpm, ambient_temperature_k=inputs.ambient_temperature_k,
                    direction=inputs.flow_direction)
                ledger, prev = ledger_from_legacy(plant, dt_s=DT, result=result, prev_energy=prev)
            assert result["all_states_finite"] and result["refrigeration_solver_success"]
            row = asdict(replace(ledger, time_s=(k + 1) * DT))
            # Derive the expected fixed-FIFO discrepancy independently from port temperatures.
            expected = (result["total_mass_flow_kg_s"] - reference_flow) * plant.tank.coolant_specific_heat_j_kg_k * (
                result["evaporator_outlet_temperature_k"] - result["cluster_supply_temperature_k"]
                + result["cluster_return_temperature_k"] - result["tank_return_temperature_k"])
            row["port_derived_flow_mismatch_w"] = expected
            row["unexplained_residual_w"] = ledger.residual_system_w - expected
            # Reconstruct the old dimensionally incomplete ledger on identical states.
            old_delay_rate = (ledger.dE_supply_per_s + ledger.dE_return_per_s) / (reference_flow * DT)
            row["old_ledger_system_residual_w"] = (
                ledger.q_gen_total_w - ledger.q_air_total_w - ledger.q_evap_applied_w
                - ledger.dE_battery_per_s - ledger.dE_plate_per_s - ledger.dE_tank_per_s - old_delay_rate)
            assert abs(row["unexplained_residual_w"]) < 1e-6, row
            assert abs(ledger.transport_flow_mismatch_w - expected) < 1e-6
            if case != "flow_step":
                assert abs(ledger.residual_system_w) < 1e-6, row
            rows.append(row)
        backend = "hc" if hc else "legacy"
        path = OUT / f"{case}_{backend}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with path.open(encoding="utf-8", newline="") as stream:
            saved = list(csv.DictReader(stream))
        assert len(saved) == 24
        summaries[backend] = {"steps": 24, "duration_s": 120.0,
            "all_states_finite": True, "all_cycle_solves_successful": True,
            **{f"max_abs_{key}": max(abs(float(row[key])) for row in saved) for key in (
                "residual_battery_w", "residual_plate_w", "residual_system_w",
                "transport_flow_mismatch_w", "unexplained_residual_w", "old_ledger_system_residual_w")}}
    return summaries


def integration_smoke():
    # Process-local duration overrides only; this does not change either saved harness.
    result = {}
    for module, label in ((stage4, "stage4"), (stage5, "stage5")):
        duration = module.DURATION_S
        try:
            module.DURATION_S = 20.0
            spec = module._build_case_specs()[0]
            values = module._run_one(spec, np.full(4, 560.0))
            summary = module._summarize(spec, *values)
            rendered = module._render_markdown([summary])
            rendered = "SMOKE ONLY: 20 s / 4 steps per backend; not a full validation run.\n\n" + rendered.replace("over 120 steps", "over 4 steps")
            assert "fixed" in rendered.lower()
            (OUT / f"{label}_20s_smoke.md").write_text(rendered, encoding="utf-8")
            result[label] = {"steps_per_backend": 4, "duration_s": 20.0, "completed": True}
        finally:
            module.DURATION_S = duration
    return result


def main():
    seed = _build_legacy_plant()
    report = {"baseline_comparison": baseline_comparison(seed), "smoke": {}}
    print("Baseline comparison: PASS, max_abs_difference=0", flush=True)
    for case in ("constant", "constant_flow_excitations", "flow_step"):
        report["smoke"][case] = smoke(seed, case)
        print(f"{case}: PASS (24 steps / 120 s per backend)", flush=True)
    report["harness_integration"] = integration_smoke()
    (OUT / "verification_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Harness integration: PASS. CSVs reread; verification_summary.json saved.", flush=True)


if __name__ == "__main__":
    main()
