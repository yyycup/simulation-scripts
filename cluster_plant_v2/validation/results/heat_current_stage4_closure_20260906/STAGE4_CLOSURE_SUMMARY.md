# Stage 4 Closure Report — per-case residuals

| case | runtime (s) | legacy max|R_bat| (W) | legacy max|R_plate| (W) | legacy max|R_loop| (W) | legacy max|R_sys| (W) | HC max|R_bat| (W) | HC max|R_plate| (W) | HC max|R_loop| (W) | HC max|R_sys| (W) |
|---|---|---|---|---|---|---|---|---|---|---|
| C0_constant_load | 107.63 | 2.232e-08 | 5.730e-10 | 8.189e+03 | 8.189e+03 | 2.141e-08 | 5.020e-10 | 7.733e+03 | 7.733e+03 |
| C1_current_step | 107.52 | 1.856e-08 | 5.457e-10 | 8.189e+03 | 8.189e+03 | 2.141e-08 | 5.384e-10 | 7.733e+03 | 7.733e+03 |
| C2_dynamic_regd | 108.60 | 2.170e-08 | 4.911e-10 | 8.218e+03 | 8.218e+03 | 1.871e-08 | 6.148e-10 | 7.770e+03 | 7.770e+03 |
| C3_reverse_flow | 106.28 | 2.885e-08 | 5.275e-10 | 8.189e+03 | 8.189e+03 | 2.602e-08 | 4.402e-10 | 7.733e+03 | 7.733e+03 |
| C4_flow_switch | 106.74 | 2.196e-08 | 4.384e-10 | 8.189e+03 | 8.189e+03 | 2.141e-08 | 6.203e-10 | 7.733e+03 | 7.733e+03 |
| C5_low_nominal_high_flow | 105.57 | 1.659e-08 | 6.567e-10 | 1.241e+04 | 1.241e+04 | 2.356e-08 | 5.784e-10 | 1.181e+04 | 1.181e+04 |

Per-case CSVs live next to this report; see the schema for
field semantics.

Reading the residuals: ``R_bat`` and ``R_plate`` are reported
against the Stage 8C3 local-gate threshold of 1e-8 W (battery)
and 1e-9 W (plate). ``R_loop`` is the loop-balance residual
``Q_pf - Q_evap_applied - dE_coolant_total/dt``; it is *not*
expected to vanish because the cluster-internal coolant segments
are folded into the implicit-transport residual together with the
unmodeled transport terms of the Stage 4 spec.

