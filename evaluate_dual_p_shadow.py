"""Evaluate labeled online P-shadow forecasts against later plant states."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


ACTUAL_FIELDS = (
    ("T_Batt", "Average temperature", "C", 1.0),
    ("T_Cool", "Coolant temperature", "C", 1.0),
    ("T_Plate", "P_Shadow_T_Plate_Actual_C", "C", 1.0),
    ("T_Supply", "Supply pipe coolant temperature", "C", 1.0),
    ("T_Return", "Return pipe coolant temperature", "C", 1.0),
    ("Q_Evap", "Evaporator cooling rate (kW)", "W", 1000.0),
)


def _horizon_steps(dt_s: float, horizons_s: Sequence[float]):
    result = []
    for horizon in horizons_s:
        steps = int(round(float(horizon) / float(dt_s)))
        if steps <= 0 or not np.isclose(steps * float(dt_s), float(horizon)):
            raise ValueError("each horizon must be a positive multiple of dt_s")
        result.append((float(horizon), steps))
    return tuple(result)


def evaluate_dual_shadow(
    frame: pd.DataFrame,
    *,
    model_labels: Sequence[str] = ("Old", "New"),
    dt_s: float = 5.0,
    horizons_s: Sequence[float] = (50.0, 100.0, 300.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for label in model_labels:
        assumption_column = f"P_Shadow_{label}_Command_Assumption"
        if assumption_column not in frame:
            raise ValueError(f"missing dual-shadow column: {assumption_column}")
        assumptions = set(frame[assumption_column].dropna().astype(str))
        if assumptions != {"provided_plan_hold_last"}:
            raise ValueError(f"{label} did not use MPC planned commands: {assumptions}")
        for horizon_s, steps in _horizon_steps(dt_s, horizons_s):
            shift = steps - 1
            if shift >= len(frame):
                raise ValueError("maximum horizon is longer than available online data")
            count = len(frame) - shift
            for state, actual_column, unit, actual_scale in ACTUAL_FIELDS:
                prediction_column = (
                    f"P_Shadow_{label}_{state}_Pred_{int(horizon_s)}s_{unit}"
                )
                if prediction_column not in frame or actual_column not in frame:
                    raise ValueError(
                        f"missing prediction or actual column: {prediction_column}, {actual_column}"
                    )
                predictions = pd.to_numeric(
                    frame[prediction_column].iloc[:count], errors="raise"
                ).to_numpy(dtype=float)
                actual = (
                    pd.to_numeric(frame[actual_column].iloc[shift:], errors="raise")
                    .to_numpy(dtype=float)
                    * actual_scale
                )
                for origin_index, (prediction, target) in enumerate(
                    zip(predictions, actual)
                ):
                    error = float(prediction - target)
                    rows.append(
                        {
                            "model": label,
                            "origin_index": origin_index,
                            "target_index": origin_index + shift,
                            "horizon_s": horizon_s,
                            "state": state,
                            "unit": unit,
                            "prediction": float(prediction),
                            "actual": float(target),
                            "error": error,
                            "abs_error": abs(error),
                        }
                    )
    details = pd.DataFrame(rows)
    summary_rows = []
    for (model, horizon_s, state, unit), group in details.groupby(
        ["model", "horizon_s", "state", "unit"], sort=True
    ):
        error = group["error"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "model": model,
                "horizon_s": horizon_s,
                "state": state,
                "unit": unit,
                "n": len(error),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    return details, pd.DataFrame(summary_rows)


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(value.strip() for value in raw.split(",") if value.strip())


def parse_args():
    parser = argparse.ArgumentParser(description="评估旧P和新P的在线MPC规划指令影子预测")
    parser.add_argument("--input-csv", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", default="Old,New")
    parser.add_argument("--horizons-s", default="50,100,300")
    parser.add_argument("--dt-s", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = _parse_csv(args.labels)
    horizons = tuple(float(value) for value in _parse_csv(args.horizons_s))
    all_details = []
    all_summaries = []
    for input_csv in args.input_csv:
        frame = pd.read_csv(input_csv, encoding="utf-8-sig")
        details, summary = evaluate_dual_shadow(
            frame,
            model_labels=labels,
            dt_s=args.dt_s,
            horizons_s=horizons,
        )
        details.insert(0, "case", input_csv.stem)
        summary.insert(0, "case", input_csv.stem)
        all_details.append(details)
        all_summaries.append(summary)
    details = pd.concat(all_details, ignore_index=True)
    summary = pd.concat(all_summaries, ignore_index=True)
    combined_rows = []
    for (model, horizon_s, state, unit), group in details.groupby(
        ["model", "horizon_s", "state", "unit"], sort=True
    ):
        error = group["error"].to_numpy(dtype=float)
        combined_rows.append(
            {
                "case": "combined",
                "model": model,
                "horizon_s": horizon_s,
                "state": state,
                "unit": unit,
                "n": len(error),
                "mae": float(np.mean(np.abs(error))),
                "rmse": float(np.sqrt(np.mean(error**2))),
                "bias": float(np.mean(error)),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    summary = pd.concat([summary, pd.DataFrame(combined_rows)], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(
        args.output_dir / "dual_shadow_details.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        args.output_dir / "dual_shadow_summary.csv", index=False, encoding="utf-8-sig"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
