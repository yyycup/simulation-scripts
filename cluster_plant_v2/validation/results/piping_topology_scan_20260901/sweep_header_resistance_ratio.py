"""设计扫参：母管阻力比 → 流量不均衡度 → 预测电池温差。

纯水力求解，不跑热仿真。温差由实验反解的灵敏度系数预测：

    温度极差 ≈ 灵敏度 × 支路流量极差(%)

灵敏度取 1800 s 实测值（560 A: 0.0445，1120 A: 0.0508 K/1%）。
对 600 s 与 1800 s 全部工况回代，预测误差约 3%。
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.hydraulics import (
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.validation.scan_piping_topology_and_flow_direction import (
    DirectReturnHeaderNetwork,
)

RESULTS = Path(__file__).resolve().parent
TOTAL_FLOW_KG_S = 0.89964
SENSITIVITY_560A = 0.0445
SENSITIVITY_1120A = 0.0508
RATIOS = (0.0002, 0.0005, 0.001, 0.002, 0.0035, 0.005, 0.008, 0.012, 0.020)
CURRENT_RATIO = 0.005


def build(topology: str, ratio: float):
    header_k = BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 * ratio
    kwargs = dict(
        n_packs=5,
        branch_resistances=np.full(5, BASE_BRANCH_RESISTANCE_PA_PER_KG_S2),
        supply_segment_resistances=np.full(5, header_k),
        return_segment_resistances=np.full(5, header_k),
    )
    cls = (
        ParallelHeaderHydraulicNetwork
        if topology == "reverse_return"
        else DirectReturnHeaderNetwork
    )
    return cls(**kwargs)


def spread_pct(network) -> float:
    flows = np.asarray(
        network.solve(TOTAL_FLOW_KG_S)["pack_mass_flows"], dtype=float
    )
    return float(np.ptp(flows) / flows.mean() * 100.0)


def main() -> None:
    rows = []
    for ratio in RATIOS:
        row = {"header_ratio": ratio}
        for topology, label in (
            ("reverse_return", "同程"),
            ("direct_return", "异程"),
        ):
            spread = spread_pct(build(topology, ratio))
            row[f"{label}_flow_spread_pct"] = spread
            row[f"{label}_dt_560a_k"] = spread * SENSITIVITY_560A
            row[f"{label}_dt_1120a_k"] = spread * SENSITIVITY_1120A
        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS / "header_ratio_sweep.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(frame.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    baseline = frame[frame["header_ratio"] == CURRENT_RATIO].iloc[0]
    print(
        f"\n现状（母管阻力比 {CURRENT_RATIO}）："
        f"同程 {baseline['同程_flow_spread_pct']:.3f}% → "
        f"预测温差 {baseline['同程_dt_1120a_k']:.3f} K；"
        f"异程 {baseline['异程_flow_spread_pct']:.3f}% → "
        f"{baseline['异程_dt_1120a_k']:.3f} K"
    )

    target = baseline["同程_flow_spread_pct"]
    print(f"\n异程式要追平同程式现役的 {target:.3f}% 不均衡度，"
          f"母管阻力比需要降到：")
    for ratio in RATIOS:
        value = frame[frame["header_ratio"] == ratio]["异程_flow_spread_pct"]
        value = float(value.iloc[0])
        if value <= target:
            print(f"  {ratio:.4f}（不均衡 {value:.3f}%，"
                  f"为现役的 {ratio / CURRENT_RATIO:.2f} 倍 → "
                  f"母管阻力需降为 {ratio / CURRENT_RATIO:.2f} 分之一）")
            break

    reverse_best = float(frame.iloc[0]["同程_flow_spread_pct"])
    print(
        f"\n同程式在最小母管阻力比 {RATIOS[0]} 下残余 "
        f"{reverse_best:.4f}%（非线性本底，与母管无关）"
    )


if __name__ == "__main__":
    main()
