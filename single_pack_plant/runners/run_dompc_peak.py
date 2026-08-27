"""PyCharm entry point for the direct Physics-P do-mpc peak-load simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))
    from single_pack_plant.simulation.case import simulate_case
else:
    from ..simulation.case import simulate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct Physics-P do-mpc peak-load control.")
    parser.add_argument("--steps", type=int, default=1280, help="5 s simulation steps (default: 1280)")
    parser.add_argument("--control-interval-steps", type=int, default=3,
                        help="re-optimize every N Plant steps (default: 3 = 15 s)")
    parser.add_argument("--control-horizon", type=int, default=None,
                        help="optional uniform number of control moves over the fixed 60-step prediction horizon")
    parser.add_argument("--force", action="store_true", help="overwrite an existing CSV with the same name")
    args = parser.parse_args()
    if args.steps <= 0 or args.control_interval_steps <= 0:
        raise ValueError("--steps and --control-interval-steps must be positive")
    if args.control_horizon is not None and not 0 < args.control_horizon <= 60:
        raise ValueError("--control-horizon must lie in [1, 60]")

    output_root = ROOT / "outputs" / "dompc_peak"
    result = simulate_case(
        control="physics_p_dompc",
        scene="peak",
        flow="单向",
        source_csv=output_root / "_time_axis.csv",
        main_name=(f"dompc_peak_{args.steps}steps_update{args.control_interval_steps}"
                   f"{'_nc' + str(args.control_horizon) if args.control_horizon else ''}.csv"),
        snap_name=(f"dompc_peak_{args.steps}steps_update{args.control_interval_steps}"
                   f"{'_nc' + str(args.control_horizon) if args.control_horizon else ''}_snapshots.csv"),
        output_root=output_root,
        dt=5.0,
        duration_s=float(args.steps * 5),
        target_temp_c=25.0,
        initial_thermal_temp_c=25.0,
        current_profile_override=np.full(args.steps, 560.0),
        mpc_flow_mode="standard",
        dompc_control_interval_steps=args.control_interval_steps,
        dompc_control_horizon=args.control_horizon,
        force=args.force,
        progress_interval_steps=args.steps,
        log_func=lambda _message: None,
    )
    print(json.dumps({"csv": str(result["out_csv"]), "steps": args.steps,
                      "control_interval_steps": args.control_interval_steps,
                      "control_horizon": args.control_horizon}, ensure_ascii=False))


if __name__ == "__main__":
    main()
