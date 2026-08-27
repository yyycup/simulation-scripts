"""Independent legacy single-Pack BTMS project package."""

from .simulation.runtime import ensure_env_library_bin_on_path

ensure_env_library_bin_on_path()

from .plant.pack import BatteryPack

__all__ = ["BatteryPack"]
