"""Single-process entry point for the compatibility-patched MPC sensitivity runner."""

import os
import tempfile
from pathlib import Path


GEKKO_TEMP = Path.home() / "btms_gekko_tmp"
GEKKO_TEMP.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(GEKKO_TEMP)
os.environ["TEMP"] = str(GEKKO_TEMP)
tempfile.tempdir = str(GEKKO_TEMP)

from mpc_sensitivity_compat import get_compatible_runner


if __name__ == "__main__":
    get_compatible_runner().main()
