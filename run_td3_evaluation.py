from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import TD3

from run_td3_training import create_run_directory, smoke_profile
from td3_btms.env import BTMSTd3Env


def summarize_trajectory(
    frame: pd.DataFrame,
    *,
    scene: str,
    dt: float,
) -> dict:
    return {
        "scene": scene,
        "steps": int(len(frame)),
        "temperature_mae_c": float(
            np.mean(np.abs(frame["mean_temp_c"] - 25.0))
        ),
        "max_temp_c": float(frame["max_temp_c"].max()),
        "max_delta_temp_c": float(frame["delta_temp_c"].max()),
        "mean_total_power_kw": float(
            frame["total_power_w"].mean() / 1000.0
        ),
        "total_energy_kwh": float(
            frame["total_power_w"].sum() * dt / 3_600_000.0
        ),
        "cumulative_reward": float(frame["reward"].sum()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate an isolated BTMS TD3 policy"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scene", choices=("peak", "freq"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--agc-data-file", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"TD3 model does not exist: {model_path}")

    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("scene") != args.scene:
            raise ValueError(
                f"model scene {metadata.get('scene')!r} "
                f"does not match {args.scene!r}"
            )

    output_dir = create_run_directory(args.output_dir)
    profile = smoke_profile(args.scene) if args.smoke else None
    env = BTMSTd3Env(
        args.scene,
        current_profile=profile,
        agc_data_file=args.agc_data_file,
        max_steps=args.max_steps,
    )
    model = TD3.load(model_path, env=env)
    rows = []
    try:
        observation, _ = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            rows.append(
                {
                    **info,
                    "action_comp_normalized": float(action[0]),
                    "action_pump_normalized": float(action[1]),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
    finally:
        env.close()

    frame = pd.DataFrame(rows)
    frame.to_csv(
        output_dir / "trajectory.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = summarize_trajectory(frame, scene=args.scene, dt=env.dt)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
