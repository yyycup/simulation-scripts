"""串行运行三类 MPC 权重的局部细化扫描。"""

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / "输出结果" / "MPC权重局部细化"
REFINEMENT_SCANS = (
    {
        "name": "CV软约束权重",
        "sweep_type": "final_cv_weight_scan",
        "factors": (5e7, 1e8, 2e8, 5e8),
    },
    {
        "name": "压缩机能耗权重",
        "sweep_type": "adaptive_dmax_w_energy_comp",
        "factors": (0.75, 1.0, 1.25, 1.5),
    },
    {
        "name": "水泵能耗权重",
        "sweep_type": "final_pump_weight_scan",
        "factors": (0.75, 1.0, 1.25, 1.5),
    },
)


def btms_environment():
    environment = os.environ.copy()
    library_bin = Path(sys.prefix) / "Library" / "bin"
    environment["PATH"] = str(library_bin) + os.pathsep + environment.get("PATH", "")
    return environment


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for index, scan in enumerate(REFINEMENT_SCANS, start=1):
        output_dir = OUTPUT_ROOT / scan["name"]
        factors = ",".join(f"{value:g}" for value in scan["factors"])
        command = [
            sys.executable,
            str(PROJECT_ROOT / "run_mpc_sensitivity_60.py"),
            "--sweep-type",
            scan["sweep_type"],
            "--factors",
            factors,
            "--cases",
            "peak,freq",
            "--max-steps",
            "120",
            "--output-root",
            str(output_dir),
        ]
        print(f"START [{index}/{len(REFINEMENT_SCANS)}] {scan['name']} factors={factors}", flush=True)
        subprocess.run(command, cwd=PROJECT_ROOT, env=btms_environment(), check=True)
        print(f"DONE  [{index}/{len(REFINEMENT_SCANS)}] {scan['name']}", flush=True)


if __name__ == "__main__":
    main()
