"""调峰调频平均温度曲线：一张图两条曲线。

调峰（S0 恒流 560 A 放电）与调频（T2 RegD 净充电）各自取
SOC 感知 NMPC 闭环时程的电池簇平均温度，叠加 25±0.65 °C 控制带。

RegD 时程 CSV 中两个 NMPC 变体共用 ``physics_p_nmpc`` 标签，
按写入顺序拆分：前半段静态预览、后半段 SOC 感知（取后半段）。
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
OUT_BASE = REGD_DIR / "fig_avg_temp_peak_vs_regd"

REF_C = 25.0
BAND_HALF = 0.65

STYLE = {
    "调峰（S0 恒流 560 A）": ("#1f77b4", "-"),
    "调频（T2 RegD）": ("#d62728", "-"),
}


def load_peak_socaware():
    frame = pd.read_csv(
        RESULTS
        / "physics_p_nmpc_s0_socaware_1280steps_20260827"
        / "case_S0_peak_1280_socaware_timeseries.csv"
    )
    return frame.sort_values("time_s")


def load_regd_socaware():
    frame = pd.read_csv(REGD_DIR / "case_T2_regd_full_timeseries.csv")
    nmpc = frame[frame["controller"] == "physics_p_nmpc"].reset_index(drop=True)
    half = len(nmpc) // 2
    return nmpc.iloc[half:].sort_values("time_s")  # 后半段 = SOC 感知


def main():
    series = {
        "调峰（S0 恒流 560 A）": load_peak_socaware(),
        "调频（T2 RegD）": load_regd_socaware(),
    }

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axhspan(
        REF_C - BAND_HALF,
        REF_C + BAND_HALF,
        color="#2ca02c",
        alpha=0.15,
        label="控制带 25±0.65 °C",
    )
    ax.axhline(REF_C, color="#2ca02c", lw=0.8, ls="--", alpha=0.5)

    for label, frame in series.items():
        color, ls = STYLE[label]
        t_min = frame["time_s"].to_numpy() / 60.0
        temp = frame["battery_avg_temp_c"].to_numpy()
        final = temp[-1]
        ax.plot(
            t_min,
            temp,
            color=color,
            ls=ls,
            lw=2.0,
            label=f"{label}（终温 {final:.2f} °C）",
        )

    ax.set_xlabel("时间 (min)", fontsize=11)
    ax.set_ylabel("电池簇平均温度 (°C)", fontsize=11)
    ax.set_title("调峰与调频工况：NMPC 电池簇平均温度曲线", fontsize=12)
    ax.set_xlim(0, 6400 / 60.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="upper left")
    fig.tight_layout()
    fig.savefig(str(OUT_BASE) + ".png", dpi=200)
    fig.savefig(str(OUT_BASE) + ".pdf")
    print("saved:", str(OUT_BASE) + ".png")


if __name__ == "__main__":
    main()
