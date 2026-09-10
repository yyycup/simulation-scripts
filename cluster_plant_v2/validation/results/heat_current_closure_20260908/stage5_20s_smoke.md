SMOKE ONLY: 20 s / 4 steps per backend; not a full validation run.

# Stage 5 Final Validation — per-case summary

Each case reports the legacy-vs-HC temperature RMSE/MAE/max-error for 8 fields, the cumulative-energy relative difference for 5 heat-current quantities, the solver/finite-state flags, and the runtime. Domain-invalid means at least one step lost finite-state or refrigeration solver success — not a frozen- model failure but a model-boundary condition.

| case | runtime (s) | steps | legacy finite | HC finite | legacy solver | HC solver | domain-invalid (legacy / HC) |
|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 3.1 | 4 | True | True | True | True | False / False |

## Temperature errors (legacy vs HC)

| case | T_b_avg RMSE (K) | T_b_max RMSE (K) | T_p_avg RMSE (K) | T_p_max RMSE (K) | T_tank RMSE (K) | T_supply RMSE (K) | T_return RMSE (K) | T_evap_out RMSE (K) |
|---|---|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.0008 | 0.0019 | 0.1271 | 0.1003 | 0.0000 | 0.0000 | 0.2063 | 0.0000 |

## Cumulative-energy relative difference (%)

| case | E_gen (%) | E_bp (%) | E_pf (%) | E_evap_cycle (%) | E_evap_applied (%) |
|---|---|---|---|---|---|
| V1_constant_nominal_forward | 0.0002 | 0.1071 | 4.5561 | 0.0000 | 0.0000 |

## Energy-conservation residuals (max |·| over 4 steps)

Pipe storage uses fixed equivalent mass defined by the initial reference flow and delay. R_loop (= Q_pf − Q_evap_applied − dE_coolant_total/dt) and R_system are raw energy residuals. At reference flow they should close numerically. Off-reference flow exposes the fixed-time FIFO model discrepancy, reported separately without subtracting it from the raw residuals.

| case | backend | R_battery (W) | R_plate (W) | R_loop (W) | R_system (W) | FIFO flow mismatch (W) |
|---|---|---|---|---|---|---|
| V1_constant_nominal_forward | legacy | 1.445e-08 | 2.710e-10 | 1.855e-10 | 1.475e-08 | 0.000e+00 |
| V1_constant_nominal_forward | heat_current | 1.445e-08 | 1.583e-10 | 7.112e-10 | 1.438e-08 | 0.000e+00 |
