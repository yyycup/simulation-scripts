"""Small Windows runtime helpers shared by the experiment entry points."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import MutableMapping


def ensure_env_library_bin_on_path(
    python_executable=None,
    environ: MutableMapping[str, str] | None = None,
):
    """Prepend the active Conda environment's Library/bin to PATH once."""
    executable = Path(sys.executable if python_executable is None else python_executable)
    library_bin = executable.resolve().parent / "Library" / "bin"
    target = os.environ if environ is None else environ
    current = target.get("PATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    if not any(entry.casefold() == str(library_bin).casefold() for entry in entries):
        target["PATH"] = os.pathsep.join([str(library_bin), *entries])
    return library_bin
