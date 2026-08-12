from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from td3_btms.env import BTMSTd3Env, REWARD_WEIGHTS, profile_sha256
from thermal_batch_config import (
    N_COMP_MAX_RPM,
    N_COMP_OFF_RPM,
    N_PUMP_MAX_RPM,
    N_PUMP_MIN_RPM,
    SIM_DT,
)


def create_run_directory(path) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"output directory already exists: {resolved}"
        ) from exc
    return resolved


def write_metadata(
    path,
    *,
    scene,
    seed,
    total_timesteps,
    dt,
    profile,
    agc_data_file,
):
    data = {
        "scene": scene,
        "seed": int(seed),
        "total_timesteps": int(total_timesteps),
        "dt_s": float(dt),
        "action_bounds_rpm": {
            "compressor": [N_COMP_OFF_RPM, N_COMP_MAX_RPM],
            "pump": [N_PUMP_MIN_RPM, N_PUMP_MAX_RPM],
        },
        "reward_weights": dict(REWARD_WEIGHTS),
        "agc_data_file": (
            None
            if agc_data_file is None
            else str(Path(agc_data_file).resolve())
        ),
        "profile_sha256": profile_sha256(profile),
    }
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_model(env, *, seed: int, smoke: bool) -> TD3:
    noise = NormalActionNoise(
        mean=np.zeros(2, dtype=np.float32),
        sigma=np.full(2, 0.1, dtype=np.float32),
    )
    kwargs = {
        "action_noise": noise,
        "device": "cpu",
        "seed": int(seed),
        "verbose": 1,
    }
    if smoke:
        kwargs.update(
            buffer_size=256,
            learning_starts=1,
            batch_size=8,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs={"net_arch": [32, 32]},
        )
    return TD3("MlpPolicy", env, **kwargs)


def smoke_profile(scene: str) -> np.ndarray:
    if scene == "peak":
        return np.full(16, 560.0, dtype=float)
    return np.asarray([0.0, 280.0, 560.0, 280.0] * 4, dtype=float)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train an isolated BTMS TD3 policy"
    )
    parser.add_argument("--scene", choices=("peak", "freq"), required=True)
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--agc-data-file", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.smoke and (
        args.total_timesteps is None or args.total_timesteps < 1
    ):
        parser.error("formal training requires positive --total-timesteps")
    if args.total_timesteps is not None and args.total_timesteps < 1:
        parser.error("--total-timesteps must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    total_timesteps = (
        args.total_timesteps if args.total_timesteps is not None else 16
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    requested = (
        args.output_dir
        or Path("outputs") / "td3" / args.scene / timestamp
    )
    run_dir = create_run_directory(requested)
    profile = smoke_profile(args.scene) if args.smoke else None
    env = BTMSTd3Env(
        args.scene,
        current_profile=profile,
        agc_data_file=args.agc_data_file,
    )
    try:
        model = build_model(env, seed=args.seed, smoke=args.smoke)
        model.learn(total_timesteps=total_timesteps)
        model.save(run_dir / "model")
        write_metadata(
            run_dir / "metadata.json",
            scene=args.scene,
            seed=args.seed,
            total_timesteps=total_timesteps,
            dt=SIM_DT,
            profile=env.current_profile,
            agc_data_file=env.agc_data_file,
        )
    finally:
        env.close()
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
