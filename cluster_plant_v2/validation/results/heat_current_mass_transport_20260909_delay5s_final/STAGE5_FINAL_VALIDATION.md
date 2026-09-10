HC: fixed-inventory mass transport, one nominal inventory across all nine cases. Legacy: fixed-time FIFO. Raw HC energy gate: 1e-6 W. This run completed all nine cases, 600 s / 120 steps per backend.

# Stage 5 Final Validation — per-case summary

Each case reports the legacy-vs-HC temperature RMSE/MAE/max-error for 8 fields, the cumulative-energy relative difference for 5 heat-current quantities, the solver/finite-state flags, and the runtime. Domain-invalid means at least one step lost finite-state or refrigeration solver success — not a frozen- model failure but a model-boundary condition.

| case | runtime (s) | steps | legacy finite | HC finite | legacy solver | HC solver | domain-invalid (legacy / HC) |
|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 188.7 | 120 | True | True | True | True | False / False |
| V2_current_step | 201.0 | 120 | True | True | True | True | False / False |
| V3_compressor_step | 202.9 | 120 | True | True | True | True | False / False |
| V4_pump_step | 197.3 | 120 | True | True | True | True | False / False |
| V5_low_flow | 199.4 | 120 | True | True | True | True | False / False |
| V6_high_flow | 208.1 | 120 | True | True | True | True | False / False |
| V7_reverse_flow | 209.3 | 120 | True | True | True | True | False / False |
| V8_flow_switch | 195.0 | 120 | True | True | True | True | False / False |
| V9_regd | 196.5 | 120 | True | True | True | True | False / False |

## Temperature errors (legacy vs HC)

| case | T_b_avg RMSE (K) | T_b_max RMSE (K) | T_p_avg RMSE (K) | T_p_max RMSE (K) | T_tank RMSE (K) | T_supply RMSE (K) | T_return RMSE (K) | T_evap_out RMSE (K) |
|---|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.0639 | 0.0473 | 0.1440 | 0.0978 | 0.5221 | 0.4542 | 0.5226 | 0.4557 |
| V2_current_step | 0.0658 | 0.0478 | 0.1482 | 0.0978 | 0.5332 | 0.4632 | 0.5338 | 0.4650 |
| V3_compressor_step | 0.0453 | 0.0409 | 0.1131 | 0.0892 | 0.4507 | 0.3938 | 0.4521 | 0.3958 |
| V4_pump_step | 0.0453 | 0.0645 | 0.1246 | 0.1406 | 0.5535 | 0.4998 | 0.5401 | 0.4871 |
| V5_low_flow | 0.0653 | 0.1407 | 0.1742 | 0.2894 | 1.0686 | 0.9257 | 1.0401 | 0.8885 |
| V6_high_flow | 0.0755 | 0.0075 | 0.1890 | 0.0634 | 0.3478 | 0.3389 | 0.3312 | 0.3163 |
| V7_reverse_flow | 0.0640 | 0.0467 | 0.1441 | 0.0966 | 0.5223 | 0.4543 | 0.5227 | 0.4559 |
| V8_flow_switch | 0.0633 | 0.1336 | 0.1438 | 0.1107 | 0.5184 | 0.4515 | 0.5174 | 0.4531 |
| V9_regd | 0.0632 | 0.0472 | 0.1424 | 0.0977 | 0.5180 | 0.4507 | 0.5184 | 0.4523 |

## Cumulative-energy relative difference (%)

| case | E_gen (%) | E_bp (%) | E_pf (%) | E_evap_cycle (%) | E_evap_applied (%) |
|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.1106 | 1.5593 | 1.5951 | 1.4198 | 1.2978 |
| V2_current_step | 0.0929 | 1.5714 | 1.6172 | 1.4446 | 1.3198 |
| V3_compressor_step | 0.0719 | 1.5081 | 1.5397 | 1.2663 | 1.1787 |
| V4_pump_step | 0.0754 | 1.1866 | 1.2062 | 1.3975 | 1.3060 |
| V5_low_flow | 0.1016 | 1.9222 | 1.9452 | 2.8428 | 2.5926 |
| V6_high_flow | 0.1347 | 1.6455 | 1.7227 | 0.8205 | 0.7468 |
| V7_reverse_flow | 0.1106 | 1.5597 | 1.5956 | 1.4202 | 1.2982 |
| V8_flow_switch | 0.1100 | 1.5418 | 1.5774 | 1.4040 | 1.2840 |
| V9_regd | 0.1125 | 1.5539 | 1.5858 | 1.4096 | 1.2894 |

## Energy-conservation residuals (max |·| over 120 steps)

Time-FIFO storage uses equivalent mass defined by the reference flow and delay; mass-transport storage reads actual parcel energies. R_loop (= Q_pf − Q_evap_applied − dE_coolant_total/dt) and R_system are raw energy residuals. At reference flow they should close numerically. Off-reference flow exposes the fixed-time FIFO model discrepancy, reported separately without subtracting it from the raw residuals. Mass transport must close the raw residual at every flow.

| case | backend | R_battery (W) | R_plate (W) | R_loop (W) | R_system (W) | FIFO flow mismatch (W) |
|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | legacy | 2.037e-08 | 5.111e-10 | 4.853e-10 | 2.061e-08 | 0.000e+00 |
| V1_constant_nominal_forward | heat_current | 2.210e-08 | 5.584e-10 | 5.675e-10 | 2.276e-08 | 0.000e+00 |
| V2_current_step | legacy | 2.477e-08 | 5.111e-10 | 4.180e-10 | 2.516e-08 | 0.000e+00 |
| V2_current_step | heat_current | 2.257e-08 | 5.657e-10 | 6.352e-10 | 2.276e-08 | 0.000e+00 |
| V3_compressor_step | legacy | 1.902e-08 | 4.165e-10 | 7.654e-10 | 1.931e-08 | 0.000e+00 |
| V3_compressor_step | heat_current | 2.148e-08 | 5.675e-10 | 6.261e-10 | 2.189e-08 | 0.000e+00 |
| V4_pump_step | legacy | 2.021e-08 | 5.894e-10 | 2.259e+03 | 2.259e+03 | 2.259e+03 |
| V4_pump_step | heat_current | 2.241e-08 | 5.184e-10 | 6.024e-10 | 2.244e-08 | 0.000e+00 |
| V5_low_flow | legacy | 2.134e-08 | 4.620e-10 | 3.118e-10 | 2.145e-08 | 0.000e+00 |
| V5_low_flow | heat_current | 2.088e-08 | 4.620e-10 | 4.325e-10 | 2.067e-08 | 0.000e+00 |
| V6_high_flow | legacy | 2.225e-08 | 4.711e-10 | 6.352e-10 | 2.293e-08 | 0.000e+00 |
| V6_high_flow | heat_current | 1.867e-08 | 6.603e-10 | 6.436e-10 | 1.899e-08 | 0.000e+00 |
| V7_reverse_flow | legacy | 2.364e-08 | 5.184e-10 | 5.515e-10 | 2.405e-08 | 0.000e+00 |
| V7_reverse_flow | heat_current | 1.864e-08 | 6.021e-10 | 4.493e-10 | 1.862e-08 | 0.000e+00 |
| V8_flow_switch | legacy | 1.886e-08 | 5.184e-10 | 5.017e-10 | 1.933e-08 | 0.000e+00 |
| V8_flow_switch | heat_current | 2.210e-08 | 5.366e-10 | 5.675e-10 | 2.276e-08 | 0.000e+00 |
| V9_regd | legacy | 2.758e-08 | 4.875e-10 | 6.359e-10 | 2.794e-08 | 0.000e+00 |
| V9_regd | heat_current | 2.599e-08 | 5.839e-10 | 5.115e-10 | 2.585e-08 | 0.000e+00 |
