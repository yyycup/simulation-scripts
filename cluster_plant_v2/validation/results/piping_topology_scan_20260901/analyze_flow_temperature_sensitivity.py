"""把"支路流量偏差 → 电池温差"的灵敏度从扫描结果里反解出来。"""

from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent


def main() -> None:
    frame = pd.read_csv(RESULTS / "piping_topology_scan.csv")

    for snapshot_s in sorted(frame["time_s"].unique()):
        snap = frame[frame["time_s"] == snapshot_s]
        for current_a in sorted(snap["current_a"].unique()):
            sub = snap[snap["current_a"] == current_a]
            base = sub[sub["topology"].str.startswith("reverse_return")].iloc[0]
            other = sub[sub["topology"].str.startswith("direct_return")].iloc[0]

            flows_rr = np.array(
                [float(v) for v in base["pack_flows_kg_s"].split("|")]
            )
            flows_dr = np.array(
                [float(v) for v in other["pack_flows_kg_s"].split("|")]
            )
            temp_rr = np.array(
                [float(v) for v in base["pack_battery_max_c"].split("|")]
            )
            temp_dr = np.array(
                [float(v) for v in other["pack_battery_max_c"].split("|")]
            )

            flow_pct = (flows_dr / flows_rr - 1.0) * 100.0
            delta_t = temp_dr - temp_rr

            print(f"\n=== {current_a:.0f} A · {snapshot_s:.0f} s ===")
            print(
                f"{'Pack':<7}{'同程流量':>11}{'异程流量':>11}"
                f"{'流量%':>9}{'同程T':>9}{'异程T':>9}{'ΔT':>8}"
            )
            for index in range(len(flows_rr)):
                print(
                    f"Pack {index + 1:<2}"
                    f"{flows_rr[index]:>11.5f}{flows_dr[index]:>11.5f}"
                    f"{flow_pct[index]:>+9.2f}{temp_rr[index]:>9.3f}"
                    f"{temp_dr[index]:>9.3f}{delta_t[index]:>+8.3f}"
                )

            print(
                f"{'极差':<7}"
                f"{np.ptp(flows_rr) / flows_rr.mean() * 100:>11.2f}"
                f"{np.ptp(flows_dr) / flows_dr.mean() * 100:>11.2f}"
                f"{'':>9}{np.ptp(temp_rr):>9.3f}{np.ptp(temp_dr):>9.3f}"
                f"{np.ptp(temp_dr) - np.ptp(temp_rr):>+8.3f}"
            )
            print(
                f"{'均值':<7}{flows_rr.mean():>11.5f}{flows_dr.mean():>11.5f}"
                f"{'':>9}{temp_rr.mean():>9.3f}{temp_dr.mean():>9.3f}"
                f"{temp_dr.mean() - temp_rr.mean():>+8.3f}"
            )

            mask = np.abs(flow_pct) >= 0.5
            fitted = np.polyfit(flow_pct[mask], delta_t[mask], 1)
            residual = delta_t[mask] - np.polyval(fitted, flow_pct[mask])
            print(
                f"最小二乘拟合：ΔT = {fitted[0]:+.4f} × 流量偏差% "
                f"{fitted[1]:+.4f}，最大残差 {np.abs(residual).max():.4f} K"
            )


if __name__ == "__main__":
    main()
