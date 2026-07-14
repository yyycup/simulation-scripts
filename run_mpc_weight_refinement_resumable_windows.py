"""Recommended Windows launcher for checkpointed MPC weight refinements."""

import os
from pathlib import Path


GEKKO_TEMP = Path.home() / "btms_gekko_tmp"
GEKKO_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["BTMS_GEKKO_TMP"] = str(GEKKO_TEMP)

from run_mpc_weight_refinement_resumable import main


if __name__ == "__main__":
    main()
