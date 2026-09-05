"""Per-scenario average-temperature figures: peak shaving S0 and RegD T2.

One figure per scenario: fixed baseline vs static-preview NMPC cluster
average temperature against the 25 C +/- 0.65 K control band.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

RESULTS = Path(
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制/"
    "cluster_plant_v2/validation/results"
)
REGD_DIR = RESULTS / "physics_p_nmpc_regd_full_20260827"

REF_C = 25.0
BAND_HALF = 0.65

STYLE = {
    "baseline": ("固定基线 4000/3600 rpm", "#7f7f7f", "-", 1.6),
    "static": ("NMPC·静态产热预览", "#d62728", "-", 1.9),
}


def _split_nmpc_halves(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    nmpc = frame[frame["controller"] == "physics_p_nmpc"].reset_index(drop=True)
    n = len(nmpc) // 2
    first = nmpc.iloc[:n].copy()
    second = nmpc.iloc[n:].copy()
    first["controller"] = "static"
    second["controller"] = "soc"
    return first, second


def load_peak() -> pd.DataFrame:
    static = pd.read_csv(
        RESULTS / "physics_p_nmpc_s0_peak_1280steps_20260827"
        / "case_S0_peak_1280_steps_timeseries.csv"
    ).copy()
    baseline = static[static["controller"] == "baseline_fixed"].copy()
    baseline["controller"] = "baseline"
    static_nmpc = static[static["controller"] == "physics_p_nmpc"].copy()
    static_nmpc["controller"] = "static"
    return pd.concat([baseline, static_nmpc], ignore_index=True)


def load_regd() -> pd.DataFrame:
    frame = pd.read_csv(REGD_DIR / "case_T2_regd_full_timeseries.csv").copy()
    baseline = frame[frame["controller"] == "baseline_fixed"].copy()
    baseline["controller"] = "baseline"
    static, _ = _split_nmpc_halves(frame)
    return pd.concat([baseline, static], ignore_index=True)


def draw(frame: pd.DataFrame, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhspan(
        REF_C - BAND_HALF,
        REF_C + BAND_HALF,
        color="#2ca02c",
        alpha=0.12,
        label="控制带 25±0.65 °C",
    )
    for controller, group in frame.groupby("controller"):
        label, color, ls, lw = STYLE[controller]
        group = group.sort_values("time_s")
        ax.plot(
            group["time_s"] / 60.0,
            group["battery_avg_temp_c"],
            color=color,
            ls=ls,
            lw=lw,
            label=label,
        )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("时间 (min)", fontsize=10)
    ax.set_ylabel("电池簇均温 (°C)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print("saved:", out)


def main() -> None:
    draw(
        load_peak(),
        "调峰 S0：恒流 560 A 放电（6400 s）",
        REGD_DIR / "fig_peak_avg_temp_cn.png",
    )
    draw(
        load_regd(),
        "调频 T2：PJM RegD × 1120 A（6400 s）",
        REGD_DIR / "fig_regd_avg_temp_cn.png",
    )


if __name__ == "__main__":
    main()
