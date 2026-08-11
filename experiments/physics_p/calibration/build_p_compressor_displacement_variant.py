"""Build a Physics-P artifact for a smaller compressor displacement."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from mpc_physics_predictor import load_physics_artifact, validate_physics_artifact


DEFAULT_SCALE = 0.8


def build_displacement_variant(base_artifact: dict, *, scale: float) -> dict:
    scale = float(scale)
    if not 0.0 < scale < 1.0:
        raise ValueError("compressor displacement scale must be within (0, 1)")
    base = copy.deepcopy(base_artifact)
    validate_physics_artifact(base)
    base_scale = float(
        base["thermal"].get("compressor_displacement_scale", 1.0)
    )
    if base_scale != 1.0:
        raise ValueError(
            "base artifact compressor displacement scale must be 1.0"
        )

    variant = copy.deepcopy(base)
    variant["capacity"]["coefficients"] = [
        scale * float(value) for value in base["capacity"]["coefficients"]
    ]
    variant["capacity"]["q_upper_w"] = (
        scale * float(base["capacity"]["q_upper_w"])
    )
    variant["thermal"]["compressor_displacement_scale"] = scale
    variant.setdefault("fit", {})["compressor_displacement_variant"] = {
        "scale_relative_to_nominal": scale,
        "nominal_displacement_cm3_per_rev": 5.525,
        "variant_displacement_cm3_per_rev": scale * 5.525,
        "capacity_scaling": "linear_with_displacement",
        "compressor_power_scaling": "linear_with_displacement",
        "dynamic_parameters_frozen": True,
        "thermal_parameters_frozen": True,
    }
    validate_physics_artifact(variant)
    return variant


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"output already exists; pass --force to replace it: {args.output}"
        )
    base = load_physics_artifact(args.base_artifact, require_validated=True)
    variant = build_displacement_variant(base, scale=args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(variant, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"ARTIFACT {args.output.resolve()}")
    print(
        "DISPLACEMENT "
        f"{variant['fit']['compressor_displacement_variant']['variant_displacement_cm3_per_rev']:.3f} "
        "cm^3/rev"
    )


if __name__ == "__main__":
    main()
