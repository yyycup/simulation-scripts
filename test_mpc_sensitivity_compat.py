from mpc_sensitivity_compat import get_compatible_runner


def test_compat_runner_restores_final_cv_weight_parameter_construction():
    runner = get_compatible_runner()
    peak = runner.build_params(
        runner.base_params_for_case("peak"),
        "final_cv_weight_scan",
        1e8,
        "baseline",
        case="peak",
    )

    assert peak.dmax_comp == 1200.0
