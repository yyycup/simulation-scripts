"""Compatibility layer for the reusable MPC sensitivity runner."""

import run_mpc_sensitivity_60 as _runner


def get_compatible_runner():
    """Restore the missing scene-specific compressor-rate helper in memory."""
    if not hasattr(_runner, "adaptive_comp_dmax_rpm_per_s"):
        def adaptive_comp_dmax_rpm_per_s(case: str) -> float:
            params = _runner.get_final_adaptive_mpc_params(case)
            return float(params.dmax_comp) / float(_runner.SIM_DT)

        _runner.adaptive_comp_dmax_rpm_per_s = adaptive_comp_dmax_rpm_per_s
    return _runner
