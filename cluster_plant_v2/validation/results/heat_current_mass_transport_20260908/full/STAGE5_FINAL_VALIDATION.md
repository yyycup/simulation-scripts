HC: fixed-inventory mass transport, one nominal inventory across all nine cases. Legacy: fixed-time FIFO. Raw HC energy gate: 1e-6 W. This run completed all nine cases, 600 s / 120 steps per backend. CSVs independently reread and verified.

# Stage 5 Final Validation — per-case summary

Each case reports the legacy-vs-HC temperature RMSE/MAE/max-error for 8 fields, the cumulative-energy relative difference for 5 heat-current quantities, the solver/finite-state flags, and the runtime. Domain-invalid means at least one step lost finite-state or refrigeration solver success — not a frozen- model failure but a model-boundary condition.

| case | runtime (s) | steps | legacy finite | HC finite | legacy solver | HC solver | domain-invalid (legacy / HC) |
|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 188.8 | 120 | True | True | True | True | False / False |
| V2_current_step | 192.9 | 120 | True | True | True | True | False / False |
| V3_compressor_step | 196.1 | 120 | True | True | True | True | False / False |
| V4_pump_step | 190.7 | 120 | True | True | True | True | False / False |
| V5_low_flow | 200.6 | 120 | True | True | True | True | False / False |
| V6_high_flow | 206.7 | 120 | True | True | True | True | False / False |
| V7_reverse_flow | 207.9 | 120 | True | True | True | True | False / False |
| V8_flow_switch | 185.2 | 120 | True | True | True | True | False / False |
| V9_regd | 184.7 | 120 | True | True | True | True | False / False |

## Temperature errors (legacy vs HC)

| case | T_b_avg RMSE (K) | T_b_max RMSE (K) | T_p_avg RMSE (K) | T_p_max RMSE (K) | T_tank RMSE (K) | T_supply RMSE (K) | T_return RMSE (K) | T_evap_out RMSE (K) |
|---|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.0780 | 0.0291 | 0.1685 | 0.0700 | 0.4616 | 0.3972 | 0.4676 | 0.4023 |
| V2_current_step | 0.0811 | 0.0291 | 0.1747 | 0.0695 | 0.4751 | 0.4081 | 0.4818 | 0.4137 |
| V3_compressor_step | 0.0584 | 0.0231 | 0.1359 | 0.0588 | 0.3857 | 0.3303 | 0.3949 | 0.3372 |
| V4_pump_step | 0.0229 | 0.0787 | 0.1643 | 0.1898 | 0.6062 | 0.5959 | 0.5578 | 0.5413 |
| V5_low_flow | 0.0231 | 0.1821 | 0.1633 | 0.3841 | 1.2390 | 1.1590 | 1.1320 | 1.0315 |
| V6_high_flow | 0.1413 | 0.0582 | 0.3358 | 0.1692 | 0.3616 | 0.3849 | 0.2329 | 0.3500 |
| V7_reverse_flow | 0.0781 | 0.0285 | 0.1686 | 0.0689 | 0.4617 | 0.3973 | 0.4677 | 0.4024 |
| V8_flow_switch | 0.0772 | 0.1337 | 0.1678 | 0.0858 | 0.4583 | 0.3951 | 0.4625 | 0.4000 |
| V9_regd | 0.0771 | 0.0292 | 0.1660 | 0.0703 | 0.4570 | 0.3936 | 0.4626 | 0.3984 |

## Cumulative-energy relative difference (%)

| case | E_gen (%) | E_bp (%) | E_pf (%) | E_evap_cycle (%) | E_evap_applied (%) |
|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.1383 | 1.8577 | 1.8923 | 1.2063 | 1.0939 |
| V2_current_step | 0.1194 | 1.8786 | 1.9251 | 1.2368 | 1.1201 |
| V3_compressor_step | 0.0933 | 1.9428 | 1.9707 | 1.0530 | 0.9672 |
| V4_pump_step | 0.0268 | 0.8574 | 0.8910 | 1.4250 | 1.3406 |
| V5_low_flow | 0.0140 | 0.5865 | 0.6067 | 3.1975 | 2.9177 |
| V6_high_flow | 0.2555 | 3.0510 | 3.1141 | 0.2411 | 0.2020 |
| V7_reverse_flow | 0.1384 | 1.8582 | 1.8929 | 1.2067 | 1.0943 |
| V8_flow_switch | 0.1370 | 1.8345 | 1.8688 | 1.1923 | 1.0821 |
| V9_regd | 0.1420 | 1.8435 | 1.8731 | 1.1950 | 1.0852 |

## Energy-conservation residuals (max |·| over 120 steps)

Time-FIFO storage uses equivalent mass defined by the reference flow and delay; mass-transport storage reads actual parcel energies. R_loop (= Q_pf − Q_evap_applied − dE_coolant_total/dt) and R_system are raw energy residuals. At reference flow they should close numerically. Off-reference flow exposes the fixed-time FIFO model discrepancy, reported separately without subtracting it from the raw residuals. Mass transport must close the raw residual at every flow.

| case | backend | R_battery (W) | R_plate (W) | R_loop (W) | R_system (W) | FIFO flow mismatch (W) |
|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | legacy | 2.232e-08 | 5.730e-10 | 1.234e-09 | 2.268e-08 | 0.000e+00 |
| V1_constant_nominal_forward | heat_current | 2.540e-08 | 6.839e-10 | 9.193e-10 | 2.432e-08 | 0.000e+00 |
| V2_current_step | legacy | 1.856e-08 | 5.457e-10 | 1.238e-09 | 1.838e-08 | 0.000e+00 |
| V2_current_step | heat_current | 2.450e-08 | 5.202e-10 | 1.055e-09 | 2.482e-08 | 0.000e+00 |
| V3_compressor_step | legacy | 2.223e-08 | 4.638e-10 | 1.531e-09 | 2.133e-08 | 0.000e+00 |
| V3_compressor_step | heat_current | 2.419e-08 | 4.839e-10 | 1.425e-09 | 2.399e-08 | 0.000e+00 |
| V4_pump_step | legacy | 2.126e-08 | 4.657e-10 | 2.115e+03 | 2.115e+03 | 2.115e+03 |
| V4_pump_step | heat_current | 1.845e-08 | 4.784e-10 | 1.133e-09 | 1.825e-08 | 0.000e+00 |
| V5_low_flow | legacy | 1.937e-08 | 4.802e-10 | 1.033e-09 | 1.873e-08 | 0.000e+00 |
| V5_low_flow | heat_current | 2.247e-08 | 6.276e-10 | 1.129e-09 | 2.279e-08 | 0.000e+00 |
| V6_high_flow | legacy | 1.659e-08 | 6.567e-10 | 2.217e-09 | 1.635e-08 | 0.000e+00 |
| V6_high_flow | heat_current | 2.536e-08 | 4.729e-10 | 1.172e-09 | 2.534e-08 | 0.000e+00 |
| V7_reverse_flow | legacy | 2.885e-08 | 5.275e-10 | 1.541e-09 | 2.855e-08 | 0.000e+00 |
| V7_reverse_flow | heat_current | 2.445e-08 | 6.421e-10 | 1.398e-09 | 2.407e-08 | 0.000e+00 |
| V8_flow_switch | legacy | 2.196e-08 | 4.384e-10 | 1.761e-09 | 2.207e-08 | 0.000e+00 |
| V8_flow_switch | heat_current | 1.680e-08 | 4.293e-10 | 1.122e-09 | 1.688e-08 | 0.000e+00 |
| V9_regd | legacy | 2.170e-08 | 4.911e-10 | 1.916e-09 | 2.162e-08 | 0.000e+00 |
| V9_regd | heat_current | 2.005e-08 | 6.075e-10 | 1.127e-09 | 2.076e-08 | 0.000e+00 |
