"""Independently reread all completed CSVs; do not infer completion from logs."""
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "full"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 120, path
    assert [float(row["time_s"]) for row in rows] == [5.0 * (k + 1) for k in range(120)]
    for row in rows:
        for key, value in row.items():
            if key == "case_id":
                continue
            if key in ("all_states_finite", "refrigeration_solver_success"):
                assert value == "True", (path, key)
            else:
                assert math.isfinite(float(value)), (path, key)
    return rows


def main():
    assert (ROOT / "parallel_exit_code.txt").read_text().strip() == "0"
    summaries = json.loads((OUT / "stage5_summary.json").read_text(encoding="utf-8"))
    assert len(summaries) == 9
    expected = {"V1_constant_nominal_forward", "V2_current_step", "V3_compressor_step",
                "V4_pump_step", "V5_low_flow", "V6_high_flow", "V7_reverse_flow",
                "V8_flow_switch", "V9_regd"}
    assert {s["case_id"] for s in summaries} == expected
    audit = []
    nominal_inventory = None
    for summary in summaries:
        case = summary["case_id"]
        for suffix in ("legacy", "heat_current", "legacy_ledger", "heat_current_ledger"):
            read_rows(OUT / f"{case}_{suffix}.csv")
        rows = read_rows(OUT / f"{case}_heat_current_ledger.csv")
        inventory = [float(rows[0][f"{side}_mass_kg"]) for side in ("supply", "return")]
        if nominal_inventory is None:
            nominal_inventory = inventory
        assert max(abs(a-b) for a,b in zip(inventory, nominal_inventory)) < 1e-10
        max_mass_error = 0.0
        for k, row in enumerate(rows):
            raw_from_terms = (float(row["q_gen_total_w"]) - float(row["q_air_total_w"])
                - float(row["q_evap_applied_w"]) - sum(float(row[key]) for key in
                ("dE_battery_per_s", "dE_plate_per_s", "dE_coolant_segments_per_s",
                 "dE_supply_per_s", "dE_return_per_s", "dE_tank_per_s")))
            assert abs(raw_from_terms - float(row["residual_system_w"])) < 1e-8
            assert abs(float(row["residual_system_w"])) < 1e-6
            assert float(row["transport_flow_mismatch_w"]) == 0.0
            for index, side in enumerate(("supply", "return")):
                max_mass_error = max(max_mass_error, abs(float(row[f"{side}_mass_kg"]) - inventory[index]))
                if k:
                    storage_rate = (float(row[f"{side}_energy_j"]) - float(rows[k-1][f"{side}_energy_j"])) / 5.0
                    assert abs(storage_rate - float(row[f"dE_{side}_per_s"])) < 1e-8
        assert max_mass_error < 1e-10
        # Preserve the measured timing of reused serial cases, explicitly rounded by its log.
        if "runtime_note" in summary:
            log = (ROOT / "full_stdout.log").read_text(encoding="utf-8")
            match = re.search(r"\[" + re.escape(case) + r"\] runtime=([0-9.]+)s", log)
            assert match
            summary["runtime_s"] = float(match[1])
            summary["runtime_note"] = "completed serial run; timing rounded to 0.1 s in full_stdout.log"
            (OUT / f"{case}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        audit.append({"case_id": case, "steps_per_backend": 120,
            "max_abs_raw_system_residual_w": max(abs(float(r["residual_system_w"])) for r in rows),
            "max_abs_mass_change_kg": max_mass_error,
            "battery_average_rmse_vs_legacy_k": summary["temp_errors"]["t_b_avg_k"]["rmse_k"],
            "passed": True})
    (OUT / "stage5_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    result = {"completed_cases": len(audit), "csv_files_reread": 36,
              "pipe_inventory_kg": nominal_inventory, "cases": audit,
              "max_abs_raw_system_residual_w": max(r["max_abs_raw_system_residual_w"] for r in audit),
              "max_abs_mass_change_kg": max(r["max_abs_mass_change_kg"] for r in audit)}
    (ROOT / "full_validation_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
