"""Checkpointed local MPC weight refinement runner for Windows."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from run_mpc_weight_refinement import OUTPUT_ROOT, PROJECT_ROOT, REFINEMENT_SCANS


CHECKPOINT_ROOT = OUTPUT_ROOT / "断点续跑"
CHECKPOINT_SUMMARY = "MPC权重局部细化_断点汇总.csv"


def _factor_label(value):
    return f"{float(value):g}".replace("+", "p").replace("-", "m").replace(".", "p")


def build_tasks():
    tasks = []
    for scan in REFINEMENT_SCANS:
        for factor in scan["factors"]:
            for case in ("peak", "freq"):
                tasks.append(
                    {
                        "scan_name": scan["name"],
                        "sweep_type": scan["sweep_type"],
                        "factor": float(factor),
                        "case": case,
                    }
                )
    return tasks


def _task_dir(output_root, task):
    return Path(output_root) / task["scan_name"] / f"{task['case']}_{_factor_label(task['factor'])}"


def _task_summary(task_dir):
    files = sorted(Path(task_dir).glob("summary_*.csv"))
    if len(files) != 1:
        return None
    return files[0]


def _write_checkpoint(output_root, tasks):
    frames = []
    for task in tasks:
        summary_path = _task_summary(_task_dir(output_root, task))
        if summary_path is None:
            continue
        frame = pd.read_csv(summary_path, encoding="utf-8-sig")
        frame["checkpoint_scan_name"] = task["scan_name"]
        frame["checkpoint_factor"] = task["factor"]
        frame["checkpoint_case"] = task["case"]
        frame["checkpoint_task_dir"] = str(summary_path.parent)
        frames.append(frame)
    if not frames:
        return None
    checkpoint_path = Path(output_root) / CHECKPOINT_SUMMARY
    pd.concat(frames, ignore_index=True).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
    return checkpoint_path


def _child_environment():
    env = os.environ.copy()
    library_bin = Path(sys.executable).resolve().parent / "Library" / "bin"
    env["PATH"] = str(library_bin) + os.pathsep + env.get("PATH", "")
    return env


def run_tasks(output_root, max_steps, timeout_s, resume=True):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks()
    entry = PROJECT_ROOT / "run_mpc_sensitivity_compatible_entry.py"
    failures = []
    for index, task in enumerate(tasks, start=1):
        task_dir = _task_dir(output_root, task)
        existing_summary = _task_summary(task_dir)
        if resume and existing_summary is not None:
            print(f"SKIP  [{index}/{len(tasks)}] {task['scan_name']} {task['case']} factor={task['factor']:g}", flush=True)
            continue
        task_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(entry),
            "--sweep-type", task["sweep_type"],
            "--factors", f"{task['factor']:g}",
            "--cases", task["case"],
            "--max-steps", str(max_steps),
            "--output-root", str(task_dir),
            "--save-detail",
        ]
        print(f"START [{index}/{len(tasks)}] {task['scan_name']} {task['case']} factor={task['factor']:g}", flush=True)
        try:
            subprocess.run(command, cwd=PROJECT_ROOT, env=_child_environment(), check=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            failures.append({**task, "failure": f"timeout after {timeout_s}s"})
            print(f"TIMEOUT [{index}/{len(tasks)}] {task['scan_name']} {task['case']} factor={task['factor']:g}", flush=True)
        except subprocess.CalledProcessError as exc:
            failures.append({**task, "failure": f"child exit code {exc.returncode}"})
            print(f"FAILED [{index}/{len(tasks)}] {task['scan_name']} {task['case']} factor={task['factor']:g}", flush=True)
        else:
            print(f"DONE  [{index}/{len(tasks)}] {task['scan_name']} {task['case']} factor={task['factor']:g}", flush=True)
        checkpoint = _write_checkpoint(output_root, tasks)
        if checkpoint is not None:
            print(f"CHECKPOINT: {checkpoint}", flush=True)
    if failures:
        failure_path = output_root / "失败任务.csv"
        pd.DataFrame(failures).to_csv(failure_path, index=False, encoding="utf-8-sig")
        print(f"FAILED TASKS: {failure_path}", flush=True)
    checkpoint = _write_checkpoint(output_root, tasks)
    print(f"END checkpoint={checkpoint}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run local MPC weight refinements with per-task checkpoints.")
    parser.add_argument("--output-root", type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout-s", type=int, default=1200)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    run_tasks(args.output_root, args.max_steps, args.timeout_s, resume=not args.rerun_completed)


if __name__ == "__main__":
    main()
