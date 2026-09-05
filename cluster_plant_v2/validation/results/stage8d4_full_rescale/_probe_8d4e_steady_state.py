"""Probe: 8D4e steady-state diagnosis for the S0 peak rerun.

Compares pre-fix (8d4d) vs post-fix (8d4e) steady segments (>30 min):
temperature mean/bias vs setpoint and compressor command spread, to
check whether the re-derived capacity artifact removed the model
mismatch (expected: bias -> 0, command converges to a constant).
Kept for reference, not deleted.
"""

from pathlib import Path

import pandas as pd

RES = Path("cluster_plant_v2/validation/results/stage8d4_full_rescale")

for tag, path in {
    "pre-fix  (8d4d)": "case_S0_peak_8d4d_nmpc_soc_aware_8d4d_timeseries.csv",
    "post-fix (8d4e)": "case_S0_peak_8d4e_nmpc_soc_aware_8d4d_timeseries.csv",
}.items():
    df = pd.read_csv(RES / path)
    late = df[df["time_s"] > 1800].copy()
    very_late = df[df["time_s"] > 5400].copy()
    cmd = late["compressor_command_rpm"]
    temp = late["battery_avg_temp_c"]
    print(f"--- {tag} ---")
    print(
        f"steady >30min: temp mean {temp.mean():.3f} (bias {temp.mean() - 25.0:+.3f} K), "
        f"min {temp.min():.3f} max {temp.max():.3f}"
    )
    print(
        f"cmd >30min: mean {cmd.mean():.0f}, std {cmd.std():.0f}, "
        f"min {cmd.min():.0f}, max {cmd.max():.0f}"
    )
    tl = very_late["battery_avg_temp_c"]
    cl = very_late["compressor_command_rpm"]
    print(
        f"last 1000s: temp mean {tl.mean():.3f} (bias {tl.mean() - 25.0:+.3f} K); "
        f"cmd mean {cl.mean():.0f}, std {cl.std():.0f}, "
        f"range [{cl.min():.0f}, {cl.max():.0f}]"
    )
    blocks = df.groupby(df["time_s"] // 600)["battery_avg_temp_c"].mean()
    print("10-min block temp means:", " ".join(f"{v:.2f}" for v in blocks.values))
    print()
