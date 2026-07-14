"""Refine peak/frequency PID parameters on the common single-flow plant.

The search is deliberately local and reproducible.  It performs three
coordinate stages (Kp, Ki, Kd), checkpoints every completed candidate, and
writes the selected parameters for the final controller-comparison runner.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from btms_runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()


PROJECT_ROOT = Path(__file__).resolve().parent
GEKKO_TEMP = Path.home() / "btms_gekko_tmp"
GEKKO_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(GEKKO_TEMP)
os.environ["TEMP"] = str(GEKKO_TEMP)
tempfile.tempdir = str(GEKKO_TEMP)

from pid_param_search import (  # noqa: E402
    candidate_passes_constraints,
    evaluate_pid_candidate,
    infer_constraint_limits,
    select_representative_segment,
    selected_pid_cases,
)
from thermal_batch_config import FREQ_PID_PARAMS, PEAK_PID_PARAMS, SIM_DT  # noqa: E402
from thermal_case_simulator import load_current_profile  # noqa: E402


OUTPUT_ROOT = PROJECT_ROOT / "输出结果" / "最终控制器完整对比" / "PID局部优化"
SUMMARY_NAME = "PID局部细化汇总.csv"
SELECTION_NAME = "PID最终参数.json"
SINGLE_FLOW = "单向"

SCENE_SETTINGS = {
    "peak": {
        "scene": "调峰",
        "base": PEAK_PID_PARAMS,
        "axis_values": (
            (1.40, 1.70, 1.90, 2.00),
            (0.000, 0.002, 0.004, 0.006, 0.008),
            (0.000, 0.040, 0.080, 0.160),
        ),
    },
    "freq": {
        "scene": "调频",
        "base": FREQ_PID_PARAMS,
        "axis_values": (
            (0.080, 0.120, 0.150, 0.180, 0.200, 0.210, 0.220, 0.240, 0.320),
            (0.000, 0.0005, 0.001, 0.002),
            (0.000, 0.020, 0.050),
        ),
    },
}


def coordinate_candidates(base, axis, values):
    """Return sorted unique candidates that differ on one PID coordinate."""
    base = tuple(float(value) for value in base)
    axis_values = sorted({float(value) for value in values} | {base[int(axis)]})
    candidates = []
    for value in axis_values:
        candidate = list(base)
        candidate[int(axis)] = value
        candidates.append(tuple(candidate))
    return candidates


def select_best_feasible(records):
    feasible = [record for record in records if bool(record.get("feasible", False))]
    if not feasible:
        raise ValueError("No feasible PID candidate in this refinement stage")
    return min(feasible, key=lambda record: float(record["score"]))


def save_pid_selection(path, peak, freq):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "peak": [float(value) for value in peak],
        "freq": [float(value) for value in freq],
        "objective": "MAE + 0.5*RMSE + 2*OSC",
        "flow": "single",
        "reference_temperature_C": 25.0,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _record_key(scene_key, stage, params):
    return (
        str(scene_key),
        int(stage),
        *(round(float(value), 12) for value in params),
    )


def _load_records(summary_path):
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return []
    return pd.read_csv(summary_path, encoding="utf-8-sig").to_dict(orient="records")


def _checkpoint(records, summary_path):
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(summary_path, index=False, encoding="utf-8-sig")


def _representative_window(scene_key, duration_s, max_steps):
    if max_steps is not None or scene_key == "peak":
        return 0.0, None if max_steps is not None else float(duration_s)
    times = np.arange(0.0, 3600.0, SIM_DT)
    current = load_current_profile("freq", times)
    return select_representative_segment("freq", times, current, duration_s=duration_s)


def _apply_stage_constraints(records, scene, explicit_energy=None, explicit_action=None):
    max_energy, max_action = infer_constraint_limits(
        records,
        max_energy=explicit_energy,
        max_action_metric=explicit_action,
    )
    for record in records:
        record["feasible"] = candidate_passes_constraints(
            record,
            scene=scene,
            max_energy=max_energy,
            max_action_metric=max_action,
        )
        record["max_energy_limit"] = max_energy
        record["max_action_metric_limit"] = max_action


def refine_scene(scene_key, output_root, records, duration_s=1500.0, max_steps=None):
    settings = SCENE_SETTINGS[scene_key]
    scene = settings["scene"]
    cases = selected_pid_cases(scene, flow=SINGLE_FLOW)
    if len(cases) != 1:
        raise ValueError(f"Expected one single-flow PID case for {scene}, got {len(cases)}")

    start_time_s, window_duration_s = _representative_window(scene_key, duration_s, max_steps)
    base = tuple(float(value) for value in settings["base"])
    summary_path = Path(output_root) / SUMMARY_NAME

    for stage, values in enumerate(settings["axis_values"]):
        stage_rows = []
        existing = {
            _record_key(row["scene_key"], row["stage"], (row["kp"], row["ki"], row["kd"])): row
            for row in records
            if all(key in row for key in ("scene_key", "stage", "kp", "ki", "kd"))
        }
        for candidate_idx, params in enumerate(coordinate_candidates(base, stage, values)):
            key = _record_key(scene_key, stage, params)
            if key in existing:
                row = existing[key]
                stage_rows.append(row)
                print(f"REUSE scene={scene_key} stage={stage} params={params}", flush=True)
                continue

            candidate_root = Path(output_root) / scene / f"阶段{stage + 1}" / f"候选{candidate_idx + 1:02d}"
            print(f"RUN scene={scene_key} stage={stage} params={params}", flush=True)
            metrics = evaluate_pid_candidate(
                params,
                cases=cases,
                scene=scene,
                output_root=candidate_root,
                max_steps=max_steps,
                start_time_s=start_time_s,
                duration_s=window_duration_s,
            )
            row = dict(metrics)
            row.update(
                {
                    "scene_key": scene_key,
                    "scene": scene,
                    "stage": stage,
                    "axis": ("Kp", "Ki", "Kd")[stage],
                    "candidate": candidate_idx,
                    "start_time_s": start_time_s,
                    "duration_s": window_duration_s,
                    "max_steps": max_steps,
                }
            )
            records.append(row)
            stage_rows.append(row)
            _checkpoint(records, summary_path)

        _apply_stage_constraints(stage_rows, scene)
        best = select_best_feasible(stage_rows)
        base = (float(best["kp"]), float(best["ki"]), float(best["kd"]))
        _checkpoint(records, summary_path)
        print(
            f"BEST scene={scene_key} stage={stage} params={base} "
            f"score={float(best['score']):.6f} energy={float(best['energy_kWh']):.6f}",
            flush=True,
        )
    return base


def parse_args():
    parser = argparse.ArgumentParser(description="调峰/调频单向PID局部细化")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--duration-s", type=float, default=1500.0)
    parser.add_argument("--max-steps", type=int, default=None, help="仅用于冒烟检查；正式优化请省略")
    parser.add_argument("--restart", action="store_true", help="忽略已有汇总并从头计算")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / SUMMARY_NAME
    records = [] if args.restart else _load_records(summary_path)

    selected = {}
    for scene_key in ("peak", "freq"):
        selected[scene_key] = refine_scene(
            scene_key,
            args.output_root,
            records,
            duration_s=args.duration_s,
            max_steps=args.max_steps,
        )

    selection_path = save_pid_selection(
        args.output_root / SELECTION_NAME,
        peak=selected["peak"],
        freq=selected["freq"],
    )
    print(f"PID汇总: {summary_path}", flush=True)
    print(f"PID最终参数: {selection_path}", flush=True)
    print(f"调峰PID: {selected['peak']}", flush=True)
    print(f"调频PID: {selected['freq']}", flush=True)


if __name__ == "__main__":
    main()
