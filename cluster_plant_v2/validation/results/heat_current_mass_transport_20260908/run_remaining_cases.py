"""Durable, bounded parallel execution of the existing Stage 5 case harness.

Completed 120-step CSV quartets are reused; incomplete cases are rerun in full.
No simulation equation or case definition is changed by this runner.
"""
import csv
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from cluster_plant_v2.validation import validate_heat_current_stage5 as v

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "full"


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if key == "case_id":
                continue
            if key in {"all_states_finite", "refrigeration_solver_success"}:
                row[key] = value == "True"
            else:
                row[key] = float(value)
                assert math.isfinite(row[key]), (path, key)
    assert len(rows) == 120 and rows[0]["time_s"] == 5 and rows[-1]["time_s"] == 600, path
    return rows


def finish_summary(spec, values):
    summary = v._summarize(spec, *values)
    summary["hc_transport_model"] = "fixed_inventory_mass_transport"
    summary["hc_raw_energy_gate_passed"] = bool(
        summary["ledger"]["heat_current"]["max_abs_residual_system_w"] < 1e-6
        and summary["all_states_finite"]["heat_current"]
        and summary["solver_success_all_steps"]["heat_current"]
    )
    hc_ledger = values[4]
    summary["pipe_inventory_kg"] = {
        "supply": float(hc_ledger[0]["supply_mass_kg"]),
        "return": float(hc_ledger[0]["return_mass_kg"]),
    }
    for side in ("supply", "return"):
        initial = summary["pipe_inventory_kg"][side]
        assert max(abs(row[f"{side}_mass_kg"] - initial) for row in hc_ledger) < 1e-10
    assert summary["hc_raw_energy_gate_passed"], summary
    (OUT / f"{spec.case_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_case(spec):
    values = v._run_one(spec, v._resolve_currents(spec), conservative_transport=True)
    for suffix, rows in zip(("legacy", "heat_current", "legacy_ledger", "heat_current_ledger"),
                           (values[0], values[1], values[3], values[4])):
        path = OUT / f"{spec.case_id}_{suffix}.csv"
        v._write_csv(path, rows)
        read_rows(path)
    return finish_summary(spec, values)


def main():
    specs = v._build_case_specs()
    completed, pending = {}, []
    for spec in specs:
        paths = [OUT / f"{spec.case_id}_{suffix}.csv" for suffix in
                 ("legacy", "heat_current", "legacy_ledger", "heat_current_ledger")]
        if all(path.exists() for path in paths):
            rows = [read_rows(path) for path in paths]
            saved = OUT / f"{spec.case_id}_summary.json"
            runtime = json.loads(saved.read_text(encoding="utf-8"))["runtime_s"] if saved.exists() else 0.0
            if runtime <= 0:
                log = (ROOT / "full_stdout.log").read_text(encoding="utf-8")
                match = re.search(r"\[" + re.escape(spec.case_id) + r"\] runtime=([0-9.]+)s", log)
                assert match, "completed CSVs need a saved runtime or serial timing log"
                runtime = float(match[1])
            summary = finish_summary(spec, (rows[0], rows[1], runtime, rows[2], rows[3]))
            completed[spec.case_id] = summary
            print(f"REUSED {spec.case_id}: 120 steps, raw-energy gate PASS", flush=True)
        else:
            pending.append(spec)
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_case, spec): spec for spec in pending}
        for future in as_completed(futures):
            summary = future.result()
            completed[summary["case_id"]] = summary
            print(f"DONE {summary['case_id']}: 120 steps, raw-energy gate PASS", flush=True)
    summaries = [completed[spec.case_id] for spec in specs]
    for side in ("supply", "return"):
        assert max(s["pipe_inventory_kg"][side] for s in summaries) - min(s["pipe_inventory_kg"][side] for s in summaries) < 1e-10
    (OUT / "stage5_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    note = ("HC: fixed-inventory mass transport, one nominal inventory across all nine cases. "
            "Legacy: fixed-time FIFO. Raw HC energy gate: 1e-6 W. "
            "This run completed all nine cases, 600 s / 120 steps per backend.\n\n")
    (OUT / "STAGE5_FINAL_VALIDATION.md").write_text(note + v._render_markdown(summaries), encoding="utf-8")
    print("FULL VALIDATION PASS: 9/9 cases; saved CSV quartets verified.", flush=True)


if __name__ == "__main__":
    main()
