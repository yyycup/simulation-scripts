"""PyCharm entry point for direct Physics-P do-mpc frequency regulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT.parent))
    from single_pack_plant.simulation.case import simulate_case
else:
    from ..simulation.case import simulate_case


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct Physics-P do-mpc frequency regulation.")
    parser.add_argument("--steps", type=int, default=720, help="5 s simulation steps (default: 720)")
    parser.add_argument("--control-interval-steps", type=int, default=3,
                        help="re-optimize every N Plant steps (default: 3 = 15 s)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing CSV with the same name")
    args = parser.parse_args()
    if args.steps <= 0 or args.control_interval_steps <= 0:
        raise ValueError("--steps and --control-interval-steps must be positive")

    output_root = ROOT / "outputs" / "dompc_frequency"
    result = simulate_case(
        control="physics_p_dompc",
        scene="freq",
        flow="单向",
        source_csv=output_root / "_time_axis.csv",
        main_name=f"dompc_frequency_{args.steps}steps_update{args.control_interval_steps}.csv",
        snap_name=f"dompc_frequency_{args.steps}steps_update{args.control_interval_steps}_snapshots.csv",
        output_root=output_root,
        dt=5.0,
        duration_s=float(args.steps * 5),
        target_temp_c=25.0,
        initial_thermal_temp_c=25.0,
        mpc_flow_mode="standard",
        dompc_control_interval_steps=args.control_interval_steps,
        force=args.force,
        progress_interval_steps=args.steps,
        log_func=lambda _message: None,
    )
    print(json.dumps({"csv": str(result["out_csv"]), "steps": args.steps,
                      "control_interval_steps": args.control_interval_steps}, ensure_ascii=False))


if __name__ == "__main__":
    main()
