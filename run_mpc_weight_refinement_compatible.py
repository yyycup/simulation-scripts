"""Run the local weight refinements through the compatibility-enabled runner."""

import os
import sys
from pathlib import Path

from mpc_sensitivity_compat import get_compatible_runner
from run_mpc_weight_refinement import OUTPUT_ROOT, PROJECT_ROOT, REFINEMENT_SCANS


def main():
    runner = get_compatible_runner()
    library_bin = Path(sys.prefix) / "Library" / "bin"
    os.environ["PATH"] = str(library_bin) + os.pathsep + os.environ.get("PATH", "")
    for index, scan in enumerate(REFINEMENT_SCANS, start=1):
        output_dir = OUTPUT_ROOT / scan["name"]
        factors = ",".join(f"{value:g}" for value in scan["factors"])
        print(f"START [{index}/{len(REFINEMENT_SCANS)}] {scan['name']} factors={factors}", flush=True)
        original_argv = sys.argv
        try:
            sys.argv = [
                str(PROJECT_ROOT / "run_mpc_sensitivity_60.py"),
                "--sweep-type", scan["sweep_type"],
                "--factors", factors,
                "--cases", "peak,freq",
                "--max-steps", "120",
                "--output-root", str(output_dir),
            ]
            runner.main()
        finally:
            sys.argv = original_argv
        print(f"DONE  [{index}/{len(REFINEMENT_SCANS)}] {scan['name']}", flush=True)


if __name__ == "__main__":
    main()
