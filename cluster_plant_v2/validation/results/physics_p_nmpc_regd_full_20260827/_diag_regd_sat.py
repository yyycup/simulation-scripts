import pandas as pd

RESULTS = (
    "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制/"
    "cluster_plant_v2/validation/results/"
)
frame = pd.read_csv(
    RESULTS + "physics_p_nmpc_regd_full_20260827/case_T2_regd_full_timeseries.csv"
)
nmpc = frame[frame["controller"] == "physics_p_nmpc"].reset_index(drop=True)
soc = nmpc.iloc[len(nmpc) // 2:].sort_values("time_s")

print("columns:", [c for c in soc.columns if "comp" in c or "rpm" in c or "pump" in c])
for lo, hi, tag in [
    (0, 3200, "前段带内  "),
    (3200, 4400, "大电流段  "),
    (4400, 5600, "回落段    "),
    (5600, 6400, "末端净充电"),
]:
    seg = soc[(soc["time_s"] >= lo) & (soc["time_s"] < hi)]
    comp_cols = [c for c in soc.columns if "compressor" in c and "rpm" in c]
    pump_cols = [c for c in soc.columns if "pump" in c and "rpm" in c]
    if comp_cols:
        comp = seg[comp_cols[0]]
        print(
            f"{tag} t={lo}-{hi}s  comp rpm: mean={comp.mean():.0f} "
            f"max={comp.max():.0f}  at-max frac={(comp >= comp.max() - 1).mean():.2f}"
        )
    if pump_cols:
        pump = seg[pump_cols[0]]
        print(
            f"           pump rpm: mean={pump.mean():.0f} max={pump.max():.0f}"
        )
    temp = seg["battery_avg_temp_c"]
    print(
        f"           avg temp: mean={temp.mean():.2f} max={temp.max():.2f}"
    )

# 电流与净电量
cur_cols = [c for c in soc.columns if "current" in c]
print("current col:", cur_cols)
if cur_cols:
    cur = soc[cur_cols[0]]
    print(f"current range: {cur.min():.0f} .. {cur.max():.0f} A, mean {cur.mean():.0f} A")
    print(f"net charge over 2h: {-cur.mean() * 6400 / 3600:.0f} Ah (branch capacity 280 Ah)")
