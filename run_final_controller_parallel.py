"""Run independent final-comparison cases concurrently with isolated GEKKO homes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent


def build_worker_plan(output_root: Path, gekko_root: Path):
    from run_final_controller_comparison import build_run_matrix

    return [
        {
            "case_index": index,
            "case": case,
            "output_root": Path(output_root),
            "gekko_temp": Path(gekko_root) / f"case_{index:02d}",
        }
        for index, case in enumerate(build_run_matrix())
    ]


def merge_worker_summaries(paths):
    frames = [pd.read_csv(path, encoding="utf-8-sig") for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("case_index").reset_index(drop=True)


def _worker_summary_path(output_root: Path, case_index: int):
    return Path(output_root) / ".parallel_worker_summaries" / f"case_{case_index:02d}.csv"


def run_worker(case_index: int, output_root: Path, worker_home: Path, pid_source: str, skip_existing: bool):
    worker_home = Path(worker_home)
    worker_home.mkdir(parents=True, exist_ok=True)
    gekko_temp = worker_home / "btms_gekko_tmp"
    gekko_temp.mkdir(parents=True, exist_ok=True)
    os.environ["HOME"] = str(worker_home)
    os.environ["TMP"] = str(gekko_temp)
    os.environ["TEMP"] = str(gekko_temp)

    from run_final_controller_comparison import build_run_matrix, configured_pid_selection, load_pid_selection, run_case

    matrix = build_run_matrix()
    if case_index < 0 or case_index >= len(matrix):
        raise ValueError(f"case index out of range: {case_index}")
    pid_selection = configured_pid_selection() if pid_source == "configured" else load_pid_selection()
    summary = run_case(matrix[case_index], pid_selection, output_root, skip_existing=skip_existing)
    summary["case_index"] = case_index
    summary_path = _worker_summary_path(output_root, case_index)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"DONE case={case_index} summary={summary_path}", flush=True)


def parse_case_indices(value: str):
    indices = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("at least one case index is required")
    return indices


def parse_args():
    parser = argparse.ArgumentParser(description="Final-controller comparison parallel dispatcher")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "parallel_results")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--case-indices", type=parse_case_indices, default=None)
    parser.add_argument("--gekko-root", type=Path, default=PROJECT_ROOT / ".parallel_gekko_homes")
    parser.add_argument("--pid-source", choices=("configured", "json"), default="configured")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case-index", type=int)
    parser.add_argument("--worker-home", type=Path)
    return parser.parse_args()


def run_dispatcher(args):
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    plan_by_index = {item["case_index"]: item for item in build_worker_plan(args.output_root, args.gekko_root)}
    if args.case_indices is None:
        args.case_indices = list(plan_by_index)
    unknown = sorted(set(args.case_indices) - set(plan_by_index))
    if unknown:
        raise ValueError(f"unknown case indices: {unknown}")

    logs_dir = Path(args.output_root) / ".parallel_worker_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    pending = list(dict.fromkeys(args.case_indices))
    running = {}
    completed = []

    while pending or running:
        while pending and len(running) < args.workers:
            case_index = pending.pop(0)
            item = plan_by_index[case_index]
            log_path = logs_dir / f"case_{case_index:02d}.log"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--case-index",
                str(case_index),
                "--output-root",
                str(args.output_root),
                "--worker-home",
                str(item["gekko_temp"]),
                "--pid-source",
                args.pid_source,
            ]
            if args.skip_existing:
                command.append("--skip-existing")
            handle = log_path.open("w", encoding="utf-8")
            process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
            running[case_index] = (process, handle, log_path)
            print(f"START case={case_index} pid={process.pid} log={log_path}", flush=True)

        finished = []
        for case_index, (process, handle, log_path) in running.items():
            code = process.poll()
            if code is not None:
                handle.close()
                if code != 0:
                    raise RuntimeError(f"case {case_index} failed with exit code {code}; see {log_path}")
                finished.append(case_index)
        for case_index in finished:
            del running[case_index]
            completed.append(case_index)
        if running:
            time.sleep(1)

    paths = [_worker_summary_path(args.output_root, case_index) for case_index in completed]
    merged = merge_worker_summaries(paths)
    summary_path = Path(args.output_root) / "parallel_controller_summary.csv"
    merged.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"COMPLETE cases={completed} summary={summary_path}", flush=True)


def main():
    args = parse_args()
    if args.worker:
        if args.case_index is None or args.worker_home is None:
            raise ValueError("worker mode requires --case-index and --worker-home")
        run_worker(args.case_index, args.output_root, args.worker_home, args.pid_source, args.skip_existing)
    else:
        run_dispatcher(args)


if __name__ == "__main__":
    main()
