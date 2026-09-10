SMOKE ONLY: 20 s / 4 steps per backend; not a full validation run.

# Stage 4 Closure Report — per-case residuals

| case | runtime (s) | legacy max|R_bat| (W) | legacy max|R_plate| (W) | legacy max|R_loop| (W) | legacy max|R_sys| (W) | HC max|R_bat| (W) | HC max|R_plate| (W) | HC max|R_loop| (W) | HC max|R_sys| (W) |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_constant_load | 1.51 | 1.445e-08 | 2.710e-10 | 1.855e-10 | 1.475e-08 | 1.445e-08 | 1.583e-10 | 7.112e-10 | 1.438e-08 |

Per-case CSVs live next to this report; see the schema for
field semantics.

Reading the residuals: ``R_bat`` and ``R_plate`` are reported
against the Stage 8C3 local-gate threshold of 1e-8 W (battery)
and 1e-9 W (plate). ``R_loop`` is the loop-balance residual
``Q_pf - Q_evap_applied - dE_coolant_total/dt``. Pipe storage uses
fixed equivalent mass from the explicit reference flow and delay.
At reference flow it should close numerically. At other flows,
the raw residual retains the fixed-time FIFO model discrepancy;
``transport_flow_mismatch_w`` in the CSV reports that term separately.
