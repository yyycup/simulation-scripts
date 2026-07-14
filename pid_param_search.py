import argparse
import random
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_batch_config import CASES, NEW_ROOT
from thermal_case_simulator import simulate_case


DEFAULT_PARAM_RANGES = {
    "kp": (0.20, 2.00),
    "ki": (0.000, 0.080),
    "kd": (0.00, 6.00),
}

PID_TEMP_REF_C = 25.0
PID_IDEAL_LOW_C = 24.8
PID_IDEAL_HIGH_C = 26.0
PID_ALLOWED_LOW_C = 23.0
PID_ALLOWED_HIGH_C = 27.0
PID_W_MAE = 1.0
PID_W_RMSE = 0.5
PID_W_OSC = 2.0


def normalize_scene_arg(scene):
    aliases = {
        "peak": "调峰",
        "\u748b\u51a8\u5632": "调峰",
        "reg": "调频",
        "freq": "调频",
        "frequency": "调频",
        "\u748b\u51ae\ue576": "调频",
    }
    return aliases.get(scene.lower(), scene)


def ascii_log_text(message):
    return str(message).encode("unicode_escape").decode("ascii")


def _pick_column(df, *names):
    for name in names:
        if name in df.columns:
            return df[name]
    raise KeyError(f"None of these columns exist: {names}")


def pid_tracking_metrics(df, t_ref_c=PID_TEMP_REF_C):
    t_avg = _pick_column(df, "Average temperature", "Battery Temp (C)", "T_avg_C").astype(float)
    energy = _pick_column(df, "Cumulative energy consumption", "Cumulative Energy (kWh)").astype(float)
    n_comp = _pick_column(df, "Compressor Speed", "Compressor Speed (RPM)", "Compressor command").astype(float)
    comp_diff = np.diff(n_comp.to_numpy())
    action_metric = float(np.mean(comp_diff**2)) if comp_diff.size else 0.0
    temps = t_avg.to_numpy(dtype=float)
    errors = temps - float(t_ref_c)
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    osc = float(np.mean(np.abs(np.diff(temps)))) if temps.size > 1 else 0.0
    temp_score = PID_W_MAE * mae + PID_W_RMSE * rmse + PID_W_OSC * osc
    return {
        "score": float(temp_score),
        "pid_temp_score": float(temp_score),
        "MAE": mae,
        "RMSE": rmse,
        "OSC": osc,
        "T_avg_mean_error": float(np.mean(errors**2)),
        "T_avg_mae": mae,
        "T_avg_min": float(np.min(t_avg)),
        "T_avg_max": float(np.max(t_avg)),
        "T_final": float(t_avg.iloc[-1]),
        "below_24p8_count": int(np.sum(temps < PID_IDEAL_LOW_C)),
        "above_26_count": int(np.sum(temps > PID_IDEAL_HIGH_C)),
        "below_23_count": int(np.sum(temps < PID_ALLOWED_LOW_C)),
        "above_27_count": int(np.sum(temps > PID_ALLOWED_HIGH_C)),
        "energy": float(energy.iloc[-1]),
        "energy_kWh": float(energy.iloc[-1]),
        "compressor_action_count": action_metric,
    }


def pid_objective_score(candidate):
    if "pid_temp_score" in candidate and not pd.isna(candidate["pid_temp_score"]):
        return float(candidate["pid_temp_score"])
    return float(candidate["T_avg_mean_error"])



def candidate_passes_constraints(candidate, scene, max_energy=None, max_action_metric=None):
    if candidate["T_avg_max"] > PID_ALLOWED_HIGH_C:
        return False
    if candidate["T_avg_min"] < PID_ALLOWED_LOW_C:
        return False
    if max_energy is not None and candidate["energy"] > max_energy:
        return False
    if max_action_metric is not None and candidate["compressor_action_count"] > max_action_metric:
        return False
    return True


def infer_constraint_limits(candidates, max_energy=None, max_action_metric=None):
    inferred_energy = max_energy
    inferred_action = max_action_metric
    if inferred_energy is None and candidates:
        inferred_energy = float(np.median([candidate["energy"] for candidate in candidates]) * 1.5)
    if inferred_action is None and candidates:
        inferred_action = float(
            np.median([candidate["compressor_action_count"] for candidate in candidates]) * 2.0
        )
    return inferred_energy, inferred_action


def choose_best_pid_candidate(candidates, scene, max_energy=None, max_action_metric=None):
    feasible = choose_top_pid_candidates(
        candidates,
        scene=scene,
        top_n=len(candidates),
        max_energy=max_energy,
        max_action_metric=max_action_metric,
    )
    if not feasible:
        raise ValueError("No PID candidate satisfied the constraints")
    return feasible[0]


def choose_top_pid_candidates(candidates, scene, top_n=10, max_energy=None, max_action_metric=None):
    feasible = [
        candidate
        for candidate in candidates
        if candidate_passes_constraints(
            candidate,
            scene=scene,
            max_energy=max_energy,
            max_action_metric=max_action_metric,
        )
    ]
    return sorted(feasible, key=pid_objective_score)[:top_n]


def select_representative_segment(scene, times, current_profile, duration_s=1500.0):
    scene = normalize_scene_arg(scene)
    times = np.asarray(times, dtype=float)
    current_profile = np.asarray(current_profile, dtype=float)
    if times.size == 0 or current_profile.size == 0:
        return 0.0, float(duration_s)
    if times.size != current_profile.size:
        raise ValueError("times and current_profile must have the same length")
    if scene != normalize_scene_arg("freq"):
        return float(times[0]), float(duration_s)

    best_start = float(times[0])
    best_score = -np.inf
    dt = float(np.median(np.diff(times))) if times.size > 1 else float(duration_s)
    for start in times:
        stop = start + float(duration_s)
        mask = (times >= start) & (times < stop)
        if np.count_nonzero(mask) < 2:
            continue
        if times[mask][-1] < stop - 1.5 * dt:
            continue
        window = current_profile[mask]
        score = float(np.std(window) + 0.25 * (np.max(window) - np.min(window)))
        if score > best_score:
            best_score = score
            best_start = float(start)
    return best_start, float(duration_s)


def random_pid_params(rng, param_ranges=None):
    ranges = DEFAULT_PARAM_RANGES if param_ranges is None else param_ranges
    return (
        rng.uniform(*ranges["kp"]),
        rng.uniform(*ranges["ki"]),
        rng.uniform(*ranges["kd"]),
    )


def selected_pid_cases(scene, flow=None):
    cases = [case for case in CASES if case.control == "pid" and case.scene == scene]
    if flow is not None:
        cases = [case for case in cases if case.flow == flow]
    return cases


def _merge_case_metrics(params, case_metrics):
    return {
        "params": tuple(float(value) for value in params),
        "kp": float(params[0]),
        "ki": float(params[1]),
        "kd": float(params[2]),
        "score": float(np.mean([m["score"] for m in case_metrics])),
        "pid_temp_score": float(np.mean([m["pid_temp_score"] for m in case_metrics])),
        "MAE": float(np.mean([m["MAE"] for m in case_metrics])),
        "RMSE": float(np.mean([m["RMSE"] for m in case_metrics])),
        "OSC": float(np.mean([m["OSC"] for m in case_metrics])),
        "T_avg_mean_error": float(np.mean([m["T_avg_mean_error"] for m in case_metrics])),
        "T_avg_mae": float(np.mean([m["T_avg_mae"] for m in case_metrics])),
        "T_avg_min": float(min(m["T_avg_min"] for m in case_metrics)),
        "T_avg_max": float(max(m["T_avg_max"] for m in case_metrics)),
        "T_final": float(np.mean([m["T_final"] for m in case_metrics])),
        "below_24p8_count": int(sum(m["below_24p8_count"] for m in case_metrics)),
        "above_26_count": int(sum(m["above_26_count"] for m in case_metrics)),
        "below_23_count": int(sum(m["below_23_count"] for m in case_metrics)),
        "above_27_count": int(sum(m["above_27_count"] for m in case_metrics)),
        "energy": float(sum(m["energy"] for m in case_metrics)),
        "energy_kWh": float(sum(m["energy_kWh"] for m in case_metrics)),
        "compressor_action_count": float(np.mean([m["compressor_action_count"] for m in case_metrics])),
    }


def evaluate_pid_candidate(
    params,
    cases,
    scene,
    output_root,
    max_steps=None,
    start_time_s=None,
    duration_s=None,
    log_func=None,
):
    case_metrics = []
    target_temp_c = PID_TEMP_REF_C
    for case in cases:
        result = simulate_case(
            *case,
            output_root=output_root,
            force=True,
            max_steps=max_steps,
            start_time_s=start_time_s,
            duration_s=duration_s,
            target_temp_c=target_temp_c,
            pid_params=params,
            log_func=log_func,
        )
        df = pd.read_csv(result["out_csv"])
        case_metrics.append(pid_tracking_metrics(df, t_ref_c=PID_TEMP_REF_C))
    return _merge_case_metrics(params, case_metrics)


def _log_factory(log_file):
    def log(message):
        log_file.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        text = f"[{stamp}] {ascii_log_text(message)}"
        with log_file.open("a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(text, flush=True)

    return log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="调峰 or 调频")
    parser.add_argument("--flow", default=None, help="optional exact flow name")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-energy", type=float, default=None)
    parser.add_argument("--max-action-metric", type=float, default=None)
    parser.add_argument("--output-root", type=Path, default=NEW_ROOT.parent / "pid_param_search")
    args = parser.parse_args()
    args.scene = normalize_scene_arg(args.scene)
    if args.flow is not None:
        args.flow = normalize_scene_arg(args.flow)

    rng = random.Random(args.seed)
    output_root = Path(args.output_root)
    log = _log_factory(output_root / "pid_param_search_progress.log")
    cases = selected_pid_cases(args.scene, flow=args.flow)
    if not cases:
        raise ValueError(f"No PID cases selected for scene={args.scene!r} flow={args.flow!r}")

    log(
        "PID SEARCH START "
        f"scene={args.scene} "
        f"flow={args.flow if args.flow else 'all'} "
        f"samples={args.samples} max_steps={args.max_steps}"
    )
    candidates = []
    for idx in range(args.samples):
        params = random_pid_params(rng)
        candidate_root = output_root / f"candidate_{idx:04d}"
        try:
            metrics = evaluate_pid_candidate(
                params,
                cases=cases,
                scene=args.scene,
                output_root=candidate_root,
                max_steps=args.max_steps,
                log_func=log,
            )
            metrics["candidate"] = idx
            metrics["feasible"] = candidate_passes_constraints(
                metrics,
                scene=args.scene,
                max_energy=args.max_energy,
                max_action_metric=args.max_action_metric,
            )
            candidates.append(metrics)
            log(
                "CANDIDATE "
                f"{idx} kp={metrics['kp']:.6g} ki={metrics['ki']:.6g} kd={metrics['kd']:.6g} "
                f"score={metrics['score']:.6g} MAE={metrics['MAE']:.6g} RMSE={metrics['RMSE']:.6g} OSC={metrics['OSC']:.6g} "
                f"mse={metrics['T_avg_mean_error']:.6g} Tmin={metrics['T_avg_min']:.3f} "
                f"Tmax={metrics['T_avg_max']:.3f} energy={metrics['energy']:.6g} "
                f"action={metrics['compressor_action_count']:.6g} feasible={int(metrics['feasible'])}"
            )
        except Exception:
            log(f"CANDIDATE {idx} ERROR")
            with (output_root / "pid_param_search_errors.log").open("a", encoding="utf-8") as f:
                f.write(traceback.format_exc() + "\n")

    if not candidates:
        raise ValueError("No candidate finished successfully")

    summary_csv = output_root / "pid_param_search_summary.csv"
    pd.DataFrame(candidates).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    best = choose_best_pid_candidate(
        candidates,
        scene=args.scene,
        max_energy=args.max_energy,
        max_action_metric=args.max_action_metric,
    )
    log(
        "PID SEARCH BEST "
        f"kp={best['kp']:.8g} ki={best['ki']:.8g} kd={best['kd']:.8g} "
        f"score={pid_objective_score(best):.6g} MAE={best['MAE']:.6g} RMSE={best['RMSE']:.6g} OSC={best['OSC']:.6g} summary={summary_csv}"
    )


if __name__ == "__main__":
    main()



