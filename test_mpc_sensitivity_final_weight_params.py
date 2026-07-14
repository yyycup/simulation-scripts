from run_mpc_sensitivity_60 import base_params_for_case, build_params


def test_final_cv_weight_scan_preserves_final_compressor_rate_limit():
    peak = build_params(
        base_params_for_case("peak"),
        "final_cv_weight_scan",
        1e8,
        "baseline",
        case="peak",
    )
    freq = build_params(
        base_params_for_case("freq"),
        "final_cv_weight_scan",
        1e8,
        "baseline",
        case="freq",
    )

    assert peak.dmax_comp == 1200.0
    assert freq.dmax_comp == 2400.0
