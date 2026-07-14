# Final Selected Terminal Cost Parameters

## Final decision

The final active terminal cost strategy is empirical terminal temperature cost.

DARE-based terminal cost has been abandoned and archived because the current reduced MPC model does not include representative cell temperature states required to construct a Hong-style representative-cell output terminal cost.

## Baseline

Baseline MPC remains unchanged:

terminal_cost_enabled = False
terminal_cost_type = "empirical_temp"
w_terminal_temp = 0
w_terminal_dare = 0
dare_p_source = ""

## Peak final empirical terminal cost

terminal_cost_enabled = True
terminal_cost_type = "empirical_temp"
w_terminal_temp = 1e6
terminal_temp_scale_c = 1.0
w_terminal_dare = 0
dare_p_source = ""

## Freq final empirical terminal cost

terminal_cost_enabled = True
terminal_cost_type = "empirical_temp"
w_terminal_temp = 5e5
terminal_temp_scale_c = 1.0
w_terminal_dare = 0
dare_p_source = ""

## Notes

The empirical terminal cost is:

J_terminal = w_terminal_temp * ((T_N - T_ref) / terminal_temp_scale_c)^2

where:

T_ref = 25 deg C
terminal_temp_scale_c = 1 deg C

The terminal cost is applied only at the final prediction step.

## DARE status

DARE terminal cost is deprecated and disabled.

Reasons:
1. representative cell temperatures are not available in the current reduced MPC state;
2. plant-only cell temperatures should not be used in the DARE terminal cost;
3. mean-temperature DARE was effective but weaker than empirical terminal cost;
4. empirical terminal cost produced the best full-case results.
