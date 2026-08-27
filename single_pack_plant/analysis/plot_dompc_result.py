"""Create a compact result figure for a direct Physics-P do-mpc run.

Usage
-----
python analysis/plot_dompc_result.py <result.csv> [--output <figure.png>]

The plot intentionally shows only the primary variables needed to judge a
closed-loop run: pack temperature and the *actual* compressor/pump speeds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_figure(csv_path: Path, output_path: Path, title: str) -> tuple[Path, Path]:
    """Read one NMPC CSV and save matching PNG/PDF summary figures."""
    data = pd.read_csv(csv_path)
    time_s = data["Time"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, (ax_temp, ax_speed) = plt.subplots(2, 1, figsize=(9.5, 6.3), sharex=True)
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_temp.plot(time_s, data["Average temperature"], color="#0072B2", lw=2.1, label="Average battery temperature")
    ax_temp.plot(time_s, data["T_cell_max_C"], color="#D55E00", lw=1.8, label="Maximum cell temperature")
    ax_temp.axhline(25.0, color="#555555", lw=1.1, ls="--", label="Temperature reference (25 °C)")
    ax_temp.set_ylabel("Temperature (°C)")
    ax_temp.set_title("Battery temperature")
    ax_temp.legend(loc="best")

    comp_col = "Compressor Speed" if "Compressor Speed" in data else "Compressor command"
    pump_col = "Pump Speed (RPM)" if "Pump Speed (RPM)" in data else "Pump command"
    ax_speed.plot(time_s, data[comp_col], color="#009E73", lw=2.0, label="Actual compressor speed")
    ax_speed.plot(time_s, data[pump_col], color="#E69F00", lw=2.0, label="Actual pump speed")
    ax_speed.set_xlabel("Time (s)")
    ax_speed.set_ylabel("Speed (rpm)")
    ax_speed.set_title("Actuator speeds")
    ax_speed.legend(loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return output_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=None, help="PNG output path")
    parser.add_argument("--title", default="Direct Physics-P do-mpc NMPC")
    args = parser.parse_args()

    output = args.output or args.csv_path.with_name(f"{args.csv_path.stem}_summary.png")
    png_path, pdf_path = make_figure(args.csv_path, output, args.title)
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")


if __name__ == "__main__":
    main()
