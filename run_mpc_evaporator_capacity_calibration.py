"""Fit low-order MPC evaporator-capacity surrogates to the candidate-B plant grid."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mpc_evaporator_capacity_model import evaluate_capacity
from thermal_batch_config import AMBIENT_TEMP_K, EVAP_CAP_FACTOR, EVAP_FLOW_EXP, EVAP_UA_FACTOR
from thermal_loop import staged_fan_speed
from thermal_system import chiller_model_NTU, compressor_model, condenser_model_NTU, p_sat_from_T, pump_model, safe_PropsSI, solve_saturation_temperatures, REF, superheat_desired


OUT = Path("outputs") / "mpc_evaporator_capacity_candidate_b"
N_COMP = (2000., 3000., 4000., 5000., 6000.)
N_PUMP = (1600., 2400., 3200., 4000., 4800.)
T_COOL = (20., 25., 30., 35.)


def plant_point(n_comp, n_pump, t_cool_c):
    m_cool, _ = pump_model(n_pump)
    t_cool = t_cool_c + 273.15
    n_fan = staged_fan_speed(n_comp)
    t_evap, t_cond = solve_saturation_temperatures(n_comp, n_fan, t_cool, m_cool, AMBIENT_TEMP_K)
    p_evap, p_cond = p_sat_from_T(t_evap), p_sat_from_T(t_cond)
    t_suc = safe_PropsSI("T", "P", p_evap, "Q", 1.0, REF) + superheat_desired
    comp = compressor_model(p_evap, t_suc, p_cond, n_comp)
    cond = condenser_model_NTU(comp["mdot"], comp["h_out"], p_cond, n_fan, AMBIENT_TEMP_K)
    chiller = chiller_model_NTU(comp["mdot"], p_evap, cond["h_cond_out"], t_cool, m_cool)
    return chiller["Q_evap"], "Q_hx" if chiller["Q_hx_potential"] <= chiller["Q_ref_max"] else "Q_ref_max"


def features(frame, kind):
    n, p, t = frame.N_comp_rpm.to_numpy(), frame.N_pump_rpm.to_numpy(), frame.T_cool_in_C.to_numpy()
    if kind == "candidate_a": return np.c_[n]
    if kind == "candidate_b": return np.c_[n, n * 2000. / p, n * (t - 25.)]
    return np.c_[np.ones(len(frame)), n, p, t, n*p, n*t, n**2]


def coefficient_names(kind):
    return {"candidate_a": ["kq"], "candidate_b": ["b0", "b1", "b2"], "candidate_c": ["b0", "b1", "b2", "b3", "b4", "b5", "b6"]}[kind]


def fit_one(frame, kind):
    x, y = features(frame, kind), frame.Q_evap_plant_W.to_numpy()
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    calibration = {"model_type": kind, "coefficients": dict(zip(coefficient_names(kind), map(float, beta))), "n_pump_ref_rpm": 2000., "q_evap_upper_bound_w": float(y.max())}
    prediction = np.array([evaluate_capacity(calibration, r.N_comp_rpm, r.N_pump_rpm, r.T_cool_in_C) for r in frame.itertuples()])
    error = prediction-y
    relative = np.abs(error)/np.maximum(y, 1e-9)*100
    metrics = {"model_type": kind, "MAE_W": float(np.mean(np.abs(error))), "RMSE_W": float(np.sqrt(np.mean(error**2))), "MAPE_percent": float(np.mean(relative)), "max_relative_error_percent": float(np.max(relative))}
    return calibration, metrics, prediction


def main():
    rows=[]
    for n in N_COMP:
        for p in N_PUMP:
            for t in T_COOL:
                q, limit = plant_point(n,p,t)
                rows.append({"N_comp_rpm":n,"N_pump_rpm":p,"T_cool_in_C":t,"Q_evap_plant_W":q,"limit_type":limit})
    frame=pd.DataFrame(rows)
    OUT.mkdir(parents=True,exist_ok=True)
    frame.to_csv(OUT/"candidate_b_capacity_grid.csv",index=False,encoding="utf-8-sig")
    fits=[]; selected=None
    for kind in ("candidate_a","candidate_b","candidate_c"):
        calibration, metrics, prediction=fit_one(frame,kind)
        fits.append(metrics)
        if selected is None and metrics["MAPE_percent"] < 10 and metrics["max_relative_error_percent"] < 20: selected=(calibration,metrics)
    if selected is None:
        selected=(fit_one(frame,"candidate_c")[0], fit_one(frame,"candidate_c")[1])
    calibration, metrics=selected
    calibration.update({"evaporator_model_version":"candidate B","EVAP_UA_FACTOR":EVAP_UA_FACTOR,"EVAP_FLOW_EXP":EVAP_FLOW_EXP,"EVAP_CAP_FACTOR":EVAP_CAP_FACTOR,"calibration_date":str(date.today()),"data_range":{"N_comp_rpm":list(N_COMP),"N_pump_rpm":list(N_PUMP),"T_cool_in_C":list(T_COOL)},"error_metrics":metrics})
    (OUT/"mpc_evaporator_capacity_candidate_b.json").write_text(json.dumps(calibration,indent=2),encoding="utf-8")
    pd.DataFrame(fits).to_csv(OUT/"candidate_model_metrics.csv",index=False,encoding="utf-8-sig")
    print(pd.DataFrame(fits).to_string(index=False)); print(json.dumps(calibration,indent=2))


if __name__ == "__main__": main()
