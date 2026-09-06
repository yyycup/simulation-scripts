# Stage 5 Final Validation — per-case summary

Each case reports the legacy-vs-HC temperature RMSE/MAE/max-error for 8 fields, the cumulative-energy relative difference for 5 heat-current quantities, the solver/finite-state flags, and the runtime. Domain-invalid means at least one step lost finite-state or refrigeration solver success — not a frozen- model failure but a model-boundary condition.

| case | runtime (s) | steps | legacy finite | HC finite | legacy solver | HC solver | domain-invalid (legacy / HC) |
|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 101.1 | 120 | True | True | True | True | False / False |
| V2_current_step | 101.0 | 120 | True | True | True | True | False / False |
| V3_compressor_step | 102.3 | 120 | True | True | True | True | False / False |
| V4_pump_step | 101.4 | 120 | True | True | True | True | False / False |
| V5_low_flow | 98.0 | 120 | True | True | True | True | False / False |
| V6_high_flow | 101.1 | 120 | True | True | True | True | False / False |
| V7_reverse_flow | 100.9 | 120 | True | True | True | True | False / False |
| V8_flow_switch | 101.0 | 120 | True | True | True | True | False / False |
| V9_regd | 101.2 | 120 | True | True | True | True | False / False |

## Temperature errors (legacy vs HC)

| case | T_b_avg RMSE (K) | T_b_max RMSE (K) | T_p_avg RMSE (K) | T_p_max RMSE (K) | T_tank RMSE (K) | T_supply RMSE (K) | T_return RMSE (K) | T_evap_out RMSE (K) |
|---|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.0780 | 0.0291 | 0.1685 | 0.0700 | 0.4616 | 0.3972 | 0.4676 | 0.4023 |
| V2_current_step | 0.0811 | 0.0291 | 0.1747 | 0.0695 | 0.4751 | 0.4081 | 0.4818 | 0.4137 |
| V3_compressor_step | 0.0584 | 0.0231 | 0.1359 | 0.0588 | 0.3857 | 0.3303 | 0.3949 | 0.3372 |
| V4_pump_step | 0.0655 | 0.0391 | 0.1579 | 0.0906 | 0.4622 | 0.4083 | 0.4540 | 0.4108 |
| V5_low_flow | 0.1199 | 0.0853 | 0.2591 | 0.1833 | 0.8752 | 0.7143 | 0.8897 | 0.7233 |
| V6_high_flow | 0.0611 | 0.0130 | 0.1530 | 0.0685 | 0.3612 | 0.3270 | 0.3320 | 0.3302 |
| V7_reverse_flow | 0.0781 | 0.0285 | 0.1686 | 0.0689 | 0.4617 | 0.3973 | 0.4677 | 0.4024 |
| V8_flow_switch | 0.0772 | 0.1337 | 0.1678 | 0.0858 | 0.4583 | 0.3951 | 0.4625 | 0.4000 |
| V9_regd | 0.0771 | 0.0292 | 0.1660 | 0.0703 | 0.4570 | 0.3936 | 0.4626 | 0.3984 |

## Cumulative-energy relative difference (%)

| case | E_gen (%) | E_bp (%) | E_pf (%) | E_evap_cycle (%) | E_evap_applied (%) |
|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.1383 | 1.8577 | 1.8923 | 1.2063 | 1.0939 |
| V2_current_step | 0.1194 | 1.8786 | 1.9251 | 1.2368 | 1.1201 |
| V3_compressor_step | 0.0933 | 1.9428 | 1.9707 | 1.0530 | 0.9672 |
| V4_pump_step | 0.1177 | 1.3937 | 1.4073 | 1.1711 | 1.0847 |
| V5_low_flow | 0.2102 | 3.0553 | 3.0747 | 2.2448 | 2.0295 |
| V6_high_flow | 0.1086 | 1.4201 | 1.4787 | 0.8400 | 0.7613 |
| V7_reverse_flow | 0.1384 | 1.8582 | 1.8929 | 1.2067 | 1.0943 |
| V8_flow_switch | 0.1370 | 1.8345 | 1.8688 | 1.1923 | 1.0821 |
| V9_regd | 0.1420 | 1.8435 | 1.8731 | 1.1950 | 1.0852 |

## Energy-conservation residuals (max |·| over 120 steps)

Only heat_current residuals are shown for battery/plate (legacy is included for completeness). Q_transport_implicit = residual_loop_implicit_transport_w (= Q_pf − Q_evap_applied − dE_coolant_total/dt), which folds un-exposed cluster-internal coolant storage — a structural, model-boundary term, NOT a system energy error.

| case | backend | R_battery (W) | R_plate (W) | Q_transport_implicit (W) | R_system (W) |
|---|---|---|---|---|---|
| V1_constant_nominal_forward | legacy | 2.232e-08 | 5.730e-10 | 8.189e+03 | 8.189e+03 |
| V1_constant_nominal_forward | heat_current | 2.540e-08 | 6.839e-10 | 7.733e+03 | 7.733e+03 |
| V2_current_step | legacy | 1.856e-08 | 5.457e-10 | 8.189e+03 | 8.189e+03 |
| V2_current_step | heat_current | 2.450e-08 | 5.202e-10 | 7.733e+03 | 7.733e+03 |
| V3_compressor_step | legacy | 2.223e-08 | 4.638e-10 | 4.445e+03 | 4.445e+03 |
| V3_compressor_step | heat_current | 2.419e-08 | 4.839e-10 | 4.195e+03 | 4.195e+03 |
| V4_pump_step | legacy | 2.126e-08 | 4.657e-10 | 7.692e+03 | 7.692e+03 |
| V4_pump_step | heat_current | 2.455e-08 | 5.220e-10 | 7.177e+03 | 7.177e+03 |
| V5_low_flow | legacy | 1.937e-08 | 4.802e-10 | 7.431e+03 | 7.431e+03 |
| V5_low_flow | heat_current | 1.964e-08 | 6.694e-10 | 6.985e+03 | 6.985e+03 |
| V6_high_flow | legacy | 1.659e-08 | 6.567e-10 | 1.241e+04 | 1.241e+04 |
| V6_high_flow | heat_current | 2.293e-08 | 4.366e-10 | 1.181e+04 | 1.181e+04 |
| V7_reverse_flow | legacy | 2.885e-08 | 5.275e-10 | 8.189e+03 | 8.189e+03 |
| V7_reverse_flow | heat_current | 2.445e-08 | 6.421e-10 | 7.733e+03 | 7.733e+03 |
| V8_flow_switch | legacy | 2.196e-08 | 4.384e-10 | 8.189e+03 | 8.189e+03 |
| V8_flow_switch | heat_current | 1.680e-08 | 4.293e-10 | 7.733e+03 | 7.733e+03 |
| V9_regd | legacy | 2.170e-08 | 4.911e-10 | 8.218e+03 | 8.218e+03 |
| V9_regd | heat_current | 2.005e-08 | 6.075e-10 | 7.770e+03 | 7.770e+03 |

