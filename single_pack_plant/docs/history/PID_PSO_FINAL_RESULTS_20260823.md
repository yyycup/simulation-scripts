# Single-Pack PID-PSO Final Results (2026-08-23)

## Run identity

- Server: `user@10.0.102.10`
- Run directory: `/home/user/btms_pso_runs/single_pack_pid_pso_20260823_124054`
- Runner: `run_pid_pso_full_search_pycharm.py`
- Search setup: 6 particles, 4 iterations, 4 workers
- Evaluations: 24 particles per scene; each particle ran the unidirectional and bidirectional cases used by the runner
- Completion: normal (`PSO DONE` for both scenes; final parameter file generated)

## Final parameters

These are the best feasible parameters found **within this search range and this objective**, not a global-optimum claim.

| Scene | Kp | Ki | Kd | Feasible particles |
|---|---:|---:|---:|---:|
| Peak shaving | 4.9765531 | 0.019607463 | 0.36395721 | 22 / 24 |
| Frequency regulation | 3.4201798 | 0 | 0.20315323 | 24 / 24 |

Configuration form:

```python
peak_pid_params = (4.9765531, 0.019607463, 0.36395721)
freq_pid_params = (3.4201798, 0, 0.20315323)
```

## Final summary metrics

| Scene | Score | MAE | RMSE | OSC | Energy (kWh) |
|---|---:|---:|---:|---:|---:|
| Peak shaving | 0.3468094657 | 0.1412385409 | 0.3981653773 | 0.0032441180 | 0.5154994711 |
| Frequency regulation | 1.0436169058 | 0.6323035494 | 0.7773718847 | 0.0113137070 | 0.2791262959 |

Additional saved-summary fields:

| Scene | Tavg min (degC) | Tavg max (degC) | Tavg final (degC) | Action metric |
|---|---:|---:|---:|---:|
| Peak shaving | 24.7115023781 | 26.6709661765 | 25.0150126960 | 4837.6526148 |
| Frequency regulation | 23.7313967268 | 26.6192123041 | 26.1915932020 | 9708.2492582 |

## Authoritative server files

- `outputs/pid_pso_peak/progress.json`
- `outputs/pid_pso_freq/progress.json`
- `outputs/pid_pso_peak/pso_summary.csv`
- `outputs/pid_pso_freq/pso_summary.csv`
- `outputs/pid_pso_final_pid_params.txt`
- `pid_pso_6x4.log`

## Deployment status

The results are recorded only. No production/configuration file was changed, and the selected PID parameters were not automatically copied into `thermal_batch_config.py`.
