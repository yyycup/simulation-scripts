import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path


THETA0 = {
    "kq_scale": 1.0,
    "h1_scale": 1.0,
    "C1_scale": 1.0,
    "C2_scale": 1.0,
    "Cplate_scale": 1.0,
    "tau_evap_scale": 1.0,
    "tau_plate_scale": 1.0,
}

BOUNDS = {
    "kq_scale": (0.6, 1.2),
    "h1_scale": (0.6, 1.2),
    "C1_scale": (0.8, 2.5),
    "C2_scale": (0.8, 2.5),
    "Cplate_scale": (0.8, 3.0),
    "tau_evap_scale": (0.8, 3.0),
    "tau_plate_scale": (0.8, 3.0),
}

DEFAULT_HORIZONS_S = (50, 100, 300)
DEFAULT_START_STRIDE = 10
DEFAULT_DT_S = 5.0
CP_COOL = 3391.0
M_DOT_NOMINAL = 1.2
C_TANK = 3.0 / 1000.0 * 1071.0 * CP_COOL
DEFAULT_REFRIGERATION_DYNAMICS = {
    "tau_evap_s": 45.0,
    "tau_cond_s": 75.0,
}
TARGET_WEIGHTS = {
    "rmse_50": 1.0,
    "rmse_100": 1.5,
    "rmse_300": 1.0,
    "bias_100": 0.5,
    "bias_300": 0.5,
}


@dataclass(frozen=True)
class InputPoint:
    q_gen_w: float
    n_comp_rpm: float
    n_pump_rpm: float
    flow_direction: int


@dataclass(frozen=True)
class CalibrationSample:
    case_name: str
    start_index: int
    start_time_s: float
    dt_s: float
    init_state: dict
    input_seq: list
    real_future_c: dict
    horizon_steps: dict


@dataclass(frozen=True)
class EvaluationResult:
    objective: float
    error_by_horizon: dict
    case_error_by_horizon: dict


def _float(row, name, default=0.0):
    try:
        value = row.get(name, "")
        if value == "" or value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _flow_direction(row):
    value = _float(row, "Flow direction d", 1.0)
    return -1 if value < 0.0 else 1


def _q_gen_from_current(total_current_a):
    branch_current_a = total_current_a / 4.0
    return (branch_current_a**2) * 0.001 * 52.0


def _default_csv_dir():
    home = Path(os.environ.get("USERPROFILE", str(Path.home())))
    return home / "Desktop" / "\u79d1\u7814" / "\u8bba\u6587" / "\u5c0f\u8bba\u6587" / "\u4eff\u771f\u6570\u636e\u8f93\u51fa" / "\u76d1\u7763\u5f0f\u7efc\u5408\u5224\u636e" / "mpc"


def default_csv_paths():
    base = _default_csv_dir()
    names = [
        "\u76d1\u7763\u5f0f_\u8c03\u5cf0\u8f93\u51fa\u5355\u5411mpc_v2.csv",
        "\u76d1\u7763\u5f0f_\u8c03\u5cf0\u8f93\u51fa\u53cc\u5411mpc_v2.csv",
        "\u76d1\u7763\u5f0f_\u8c03\u9891\u8f93\u51fa\u5355\u5411mpc_v2.csv",
        "\u76d1\u7763\u5f0f_\u8c03\u9891\u8f93\u51fa\u53cc\u5411mpc_v2.csv",
    ]
    return [base / name for name in names]


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def build_samples_from_rows(rows, case_name, start_stride=DEFAULT_START_STRIDE, horizons_s=DEFAULT_HORIZONS_S):
    if len(rows) < 2:
        return []
    dt_s = _float(rows[1], "Time") - _float(rows[0], "Time")
    if dt_s <= 0.0:
        dt_s = DEFAULT_DT_S
    horizon_steps = {int(h): int(round(float(h) / dt_s)) for h in horizons_s}
    max_steps = max(horizon_steps.values())
    samples = []
    for start in range(0, len(rows) - max_steps, max(1, int(start_stride))):
        row = rows[start]
        plate_in_c = _float(row, "Cold plate inlet coolant temperature", _float(row, "Coolant temperature", 25.0))
        plate_out_c = _float(row, "Cold plate outlet coolant temperature", plate_in_c)
        q_evap_w = 1000.0 * _float(row, "Evaporator cooling rate (kW)", 0.0)
        init_state = {
            "t_batt_c": _float(row, "Average temperature", 25.0),
            "t_cool_c": _float(row, "Coolant temperature", 25.0),
            "t_plate_c": 0.5 * (plate_in_c + plate_out_c),
            "q_evap_eff_w": q_evap_w,
            "q_cond_eff_w": q_evap_w,
        }
        input_seq = []
        for j in range(start, start + max_steps + 1):
            r = rows[j]
            input_seq.append(
                InputPoint(
                    q_gen_w=_q_gen_from_current(_float(r, "Total current", 0.0)),
                    n_comp_rpm=_float(r, "Compressor Speed", _float(r, "Compressor command", 0.0)),
                    n_pump_rpm=_float(r, "Pump Speed (RPM)", _float(r, "Pump command", 0.0)),
                    flow_direction=_flow_direction(r),
                )
            )
        real_future_c = {
            int(horizon_s): _float(rows[start + steps], "Average temperature", math.nan)
            for horizon_s, steps in horizon_steps.items()
        }
        samples.append(
            CalibrationSample(
                case_name=case_name,
                start_index=start,
                start_time_s=_float(row, "Time"),
                dt_s=dt_s,
                init_state=init_state,
                input_seq=input_seq,
                real_future_c=real_future_c,
                horizon_steps=horizon_steps,
            )
        )
    return samples


def load_samples(csv_paths, start_stride=DEFAULT_START_STRIDE, horizons_s=DEFAULT_HORIZONS_S):
    samples = []
    for path in csv_paths:
        rows = read_csv_rows(path)
        samples.extend(build_samples_from_rows(rows, Path(path).name, start_stride, horizons_s))
    return samples


def _bounded_theta(theta):
    checked = dict(THETA0)
    checked.update(theta or {})
    for name, (lo, hi) in BOUNDS.items():
        checked[name] = min(hi, max(lo, float(checked[name])))
    return checked


def rollout_reduced_model(theta, init_state, input_seq, horizon_steps, dt_s=DEFAULT_DT_S):
    theta = _bounded_theta(theta)
    c1 = 52.0 * 4747.0 * theta["C1_scale"]
    c2 = C_TANK * theta["C2_scale"]
    c_plate = 52.0 * 1000.0 * theta["Cplate_scale"]
    h1_ref = 52.0 * 10.0 * theta["h1_scale"]
    h2 = 30.0 * 0.1
    n_pump_ref = 2000.0
    kq = 1.0 * theta["kq_scale"]
    tau_evap = DEFAULT_REFRIGERATION_DYNAMICS["tau_evap_s"] * theta["tau_evap_scale"]
    tau_cond = DEFAULT_REFRIGERATION_DYNAMICS["tau_cond_s"]
    tau_plate = 20.0 * theta["tau_plate_scale"]
    t_amb_c = 35.0

    t_batt_c = float(init_state["t_batt_c"])
    t_cool_c = float(init_state["t_cool_c"])
    t_plate_c = float(init_state["t_plate_c"])
    q_evap_eff = float(init_state.get("q_evap_eff_w", 0.0))
    q_cond_eff = float(init_state.get("q_cond_eff_w", q_evap_eff))
    out = {0: t_batt_c}

    max_step = max(horizon_steps.values())
    for step in range(1, max_step + 1):
        u = input_seq[min(step - 1, len(input_seq) - 1)]
        n_pump = max(1.0, u.n_pump_rpm)
        h1 = h1_ref * (n_pump / n_pump_ref) ** 0.8
        c_flow = M_DOT_NOMINAL * CP_COOL * n_pump / n_pump_ref
        q_evap_cmd = kq * max(0.0, u.n_comp_rpm)
        q_cond_eff += dt_s / (tau_cond + dt_s) * (q_evap_cmd - q_cond_eff)
        q_evap_eff += dt_s / (tau_evap + dt_s) * (q_cond_eff - q_evap_eff)

        q_batt_to_plate = h1 * (t_batt_c - t_plate_c)
        t_evap_out_c = t_cool_c - q_evap_eff / (c_flow + 1e-6)
        plate_flow_gain = 1.0 + 0.02 * (u.flow_direction < 0)
        t_plate_c += dt_s / (tau_plate + dt_s) * (t_evap_out_c - t_plate_c)
        t_plate_out_c = t_plate_c + plate_flow_gain * q_batt_to_plate / (c_flow + 1e-6)
        t_batt_c = _first_order_heat_update(
            t_batt_c,
            conductance_w_k=h1 + h2,
            forcing_w=u.q_gen_w + h1 * t_plate_c + h2 * t_amb_c,
            capacitance_j_k=c1,
            dt_s=dt_s,
        )
        t_cool_c = _first_order_heat_update(
            t_cool_c,
            conductance_w_k=c_flow,
            forcing_w=c_flow * t_plate_out_c,
            capacitance_j_k=c2,
            dt_s=dt_s,
        )
        t_plate_c += dt_s * q_batt_to_plate / c_plate
        out[step] = t_batt_c
    return {horizon_s: out[steps] for horizon_s, steps in horizon_steps.items()}


def _first_order_heat_update(current_c, conductance_w_k, forcing_w, capacitance_j_k, dt_s):
    if conductance_w_k <= 1e-12 or capacitance_j_k <= 1e-12:
        return current_c
    equilibrium_c = forcing_w / conductance_w_k
    alpha = 1.0 - math.exp(-conductance_w_k * dt_s / capacitance_j_k)
    return current_c + alpha * (equilibrium_c - current_c)


def summarize_errors(errors):
    arr = [float(x) for x in errors]
    if not arr:
        return {
            "n": 0,
            "mae_c": math.nan,
            "rmse_c": math.nan,
            "mean_bias_c": math.nan,
            "p95_abs_c": math.nan,
        }
    return {
        "n": len(arr),
        "mae_c": sum(abs(x) for x in arr) / len(arr),
        "rmse_c": math.sqrt(sum(x * x for x in arr) / len(arr)),
        "mean_bias_c": sum(arr) / len(arr),
        "p95_abs_c": _percentile([abs(x) for x in arr], 95),
    }


def _percentile(values, pct):
    vals = sorted(values)
    if not vals:
        return math.nan
    k = (len(vals) - 1) * pct / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def evaluate_theta(theta, samples):
    error_by_horizon = {horizon_s: [] for horizon_s in DEFAULT_HORIZONS_S}
    case_error_by_horizon = {}
    for sample in samples:
        pred = rollout_reduced_model(theta, sample.init_state, sample.input_seq, sample.horizon_steps, sample.dt_s)
        case_errors = case_error_by_horizon.setdefault(
            sample.case_name, {horizon_s: [] for horizon_s in sample.horizon_steps}
        )
        for horizon_s, real_c in sample.real_future_c.items():
            error = float(real_c) - float(pred[horizon_s])
            error_by_horizon.setdefault(horizon_s, []).append(error)
            case_errors.setdefault(horizon_s, []).append(error)
    stats = {h: summarize_errors(errors) for h, errors in error_by_horizon.items()}
    objective = (
        TARGET_WEIGHTS["rmse_50"] * stats[50]["rmse_c"]
        + TARGET_WEIGHTS["rmse_100"] * stats[100]["rmse_c"]
        + TARGET_WEIGHTS["rmse_300"] * stats[300]["rmse_c"]
        + TARGET_WEIGHTS["bias_100"] * abs(stats[100]["mean_bias_c"])
        + TARGET_WEIGHTS["bias_300"] * abs(stats[300]["mean_bias_c"])
    )
    return EvaluationResult(float(objective), error_by_horizon, case_error_by_horizon)


def _theta_to_vector(theta):
    return [theta[name] for name in THETA0]


def _vector_to_theta(vector):
    return {name: float(vector[i]) for i, name in enumerate(THETA0)}


def calibrate_theta(samples, maxiter=60, seed=7):
    bounds = [BOUNDS[name] for name in THETA0]
    try:
        from scipy.optimize import differential_evolution, minimize

        result = differential_evolution(
            lambda x: evaluate_theta(_vector_to_theta(x), samples).objective,
            bounds=bounds,
            seed=seed,
            maxiter=maxiter,
            polish=False,
            updating="immediate",
            workers=1,
        )
        local = minimize(
            lambda x: evaluate_theta(_vector_to_theta(x), samples).objective,
            result.x,
            method="Nelder-Mead",
            options={"maxiter": maxiter * 20, "xatol": 1e-4, "fatol": 1e-4},
        )
        vector = list(local.x if local.fun <= result.fun else result.x)
    except Exception:
        rng = random.Random(seed)
        best_vector = _theta_to_vector(THETA0)
        best_score = evaluate_theta(THETA0, samples).objective
        for _ in range(max(50, maxiter * 20)):
            candidate = [rng.uniform(lo, hi) for lo, hi in bounds]
            score = evaluate_theta(_vector_to_theta(candidate), samples).objective
            if score < best_score:
                best_score = score
                best_vector = candidate
        vector = best_vector
    theta = _bounded_theta(_vector_to_theta(vector))
    return theta, evaluate_theta(theta, samples)


def _report_rows(label, theta, result):
    rows = []
    for horizon_s, errors in sorted(result.error_by_horizon.items()):
        stats = summarize_errors(errors)
        rows.append({"scope": "all", "case": "all", "label": label, "horizon_s": horizon_s, **stats})
    for case_name, by_horizon in sorted(result.case_error_by_horizon.items()):
        for horizon_s, errors in sorted(by_horizon.items()):
            stats = summarize_errors(errors)
            rows.append({"scope": "case", "case": case_name, "label": label, "horizon_s": horizon_s, **stats})
    return rows


def write_reports(output_dir, best_theta, baseline, calibrated):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "best_theta.json", "w", encoding="utf-8") as fp:
        json.dump(best_theta, fp, indent=2, sort_keys=True)

    rows = _report_rows("baseline", THETA0, baseline) + _report_rows("calibrated", best_theta, calibrated)
    fieldnames = ["scope", "case", "label", "horizon_s", "n", "mae_c", "rmse_c", "mean_bias_c", "p95_abs_c"]
    with open(output / "calibration_report.csv", "w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_bar_plot(output / "calibration_error_bars.png", rows)


def _write_bar_plot(path, rows):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    all_rows = [row for row in rows if row["scope"] == "all"]
    labels = sorted(set(row["horizon_s"] for row in all_rows))
    x = list(range(len(labels)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for offset, label in [(-width / 2, "baseline"), (width / 2, "calibrated")]:
        vals = [next(row["mae_c"] for row in all_rows if row["label"] == label and row["horizon_s"] == h) for h in labels]
        ax.bar([v + offset for v in x], vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}s" for h in labels])
    ax.set_ylabel("MAE / C")
    ax.set_title("Reduced-order MPC open-loop prediction error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_args():
    parser = argparse.ArgumentParser(description="Calibrate MPC reduced-order open-loop prediction parameters.")
    parser.add_argument("--csv", action="append", dest="csv_paths", help="MPC output CSV path. Repeat for multiple cases.")
    parser.add_argument("--output-dir", default=str(Path("outputs") / "mpc_reduced_model_calibration"))
    parser.add_argument("--start-stride", type=int, default=DEFAULT_START_STRIDE)
    parser.add_argument("--maxiter", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main():
    args = _parse_args()
    csv_paths = [Path(p) for p in args.csv_paths] if args.csv_paths else default_csv_paths()
    samples = load_samples(csv_paths, start_stride=args.start_stride)
    if not samples:
        raise SystemExit("No calibration samples were built from the selected CSV files.")
    baseline = evaluate_theta(THETA0, samples)
    best_theta, calibrated = calibrate_theta(samples, maxiter=args.maxiter, seed=args.seed)
    write_reports(args.output_dir, best_theta, baseline, calibrated)
    print(f"samples={len(samples)}")
    print(f"baseline_objective={baseline.objective:.6f}")
    print(f"calibrated_objective={calibrated.objective:.6f}")
    print(json.dumps(best_theta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
