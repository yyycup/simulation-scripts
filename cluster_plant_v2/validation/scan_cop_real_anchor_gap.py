"""Quantify the COP gap between the frozen 8D4 cycle and real hardware anchors.

Read-only audit: solves :class:`ClosedR134aCycle` over a speed sweep at the
Jilipow-comparable and project-typical operating conditions and reports the
shaft COP the model produces against three escalating definitions of
"input power":

1. ``cop_shaft``    - what the model reports today (mechanical efficiency only)
2. ``cop_drive``    - adds motor x inverter losses
3. ``cop_system``   - further adds condenser fan and coolant pump power

Real anchors live in ``model_data/real_data_anchors/``. Nothing here modifies
any frozen plant parameter.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cluster_plant_v2.hydraulics import (
    CoolantPump,
    ParallelHeaderHydraulicNetwork,
)
from cluster_plant_v2.parameters import COOLANT_DENSITY_KG_M3
from cluster_plant_v2.refrigeration import ClosedR134aCycle

OUTPUT_DIR = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "results"
    / "cop_real_anchor_gap_20260901"
)

SPEED_AXIS_RPM = (2000.0, 3000.0, 4000.0, 5000.0, 6000.0)
FAN_SPEED_RPM = 4000.0

# Drive-train loss chain absent from the frozen cycle.
MOTOR_EFFICIENCY_RANGE = (0.85, 0.92)
VFD_EFFICIENCY_RANGE = (0.94, 0.97)
ETA_DRIVE_MID = 0.85 * 0.945

# Condenser fan: no fan-power term exists in refrigeration.py at all.
# Axial condenser fan on an 8 m2 coil at ~6.5 kg/s air: order 150-400 W at
# full speed, scaled by the cube of the speed ratio.
FAN_POWER_FULL_SPEED_W = 275.0
FAN_POWER_RANGE_W = (150.0, 400.0)

# Real hardware anchors (model_data/real_data_anchors/, retrieved 2026-09-01).
REAL_ANCHORS = (
    ("Jilipow JL-BC-5-768280-L", 5600.0, 2340.0, "5 Packs / 280 Ah / 215 kWh cabinet"),
    ("GS Energy GS-005YN3A-W-C", 5000.0, 2600.0, "5 kW ESS chiller, W18/A45"),
    ("KSTAR liquid cooling unit", 5000.0, 1900.0, "5 kW class, 40 L/min"),
    ("Xinrex 232 kWh cabinet", 5000.0, 3200.0, "5 kW cooling rating"),
    ("Bus rooftop AC", 17500.0, 6350.0, "nearest capacity class to the 72 cc model"),
)
# Compressor-level catalog COP (ARI-like low evaporating temperature duty).
COMPRESSOR_CATALOG_COP_RANGE = (2.83, 3.59)


def fan_power_w(fan_speed_rpm: float) -> float:
    ratio = float(fan_speed_rpm) / 4000.0
    return FAN_POWER_FULL_SPEED_W * ratio**3


def build_pump(n_packs: int = 5) -> CoolantPump:
    """Pump built the way the project's own validation scripts build it."""
    network = ParallelHeaderHydraulicNetwork(n_packs)
    reference_mass_flow = 28.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3
    reference_delta_p = float(network.solve(reference_mass_flow)["network_delta_p"])
    return CoolantPump(reference_operating_delta_p_pa=reference_delta_p)


def pump_power_w(pump_speed_rpm: float, n_packs: int = 5) -> float:
    return float(build_pump(n_packs).power_w(pump_speed_rpm))


def hydraulic_power_w(mass_flow_kg_s: float, delta_p_pa: float) -> float:
    volume_flow_m3_s = float(mass_flow_kg_s) / COOLANT_DENSITY_KG_M3
    return volume_flow_m3_s * float(delta_p_pa)


def scan_conditions(
    cycle: ClosedR134aCycle,
    *,
    label: str,
    coolant_inlet_temperature_k: float,
    ambient_temperature_k: float,
    coolant_mass_flow_kg_s: float,
    pump_speed_rpm: float,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    pump_w = pump_power_w(pump_speed_rpm)
    fan_w = fan_power_w(FAN_SPEED_RPM)
    for speed in SPEED_AXIS_RPM:
        result = cycle.solve(
            compressor_speed_rpm=speed,
            fan_speed_rpm=FAN_SPEED_RPM,
            coolant_inlet_temperature_k=coolant_inlet_temperature_k,
            coolant_mass_flow_kg_s=coolant_mass_flow_kg_s,
            ambient_temperature_k=ambient_temperature_k,
        )
        if not result["solver_success"]:
            rows.append({"condition": label, "speed_rpm": speed, "solver_success": False})
            continue

        q_evap = float(result["q_evaporator_w"])
        shaft_w = float(result["compressor_shaft_power_w"])
        refrigerant_w = float(result["compressor_refrigerant_power_w"])
        t_evap = float(result["evaporating_saturation_temperature_k"])
        t_cond = float(result["condensing_saturation_temperature_k"])
        p_evap = float(result["state_1"].pressure_pa)
        p_cond = float(result["state_2"].pressure_pa)

        cop_shaft = q_evap / shaft_w if shaft_w > 0.0 else float("nan")
        electrical_w = shaft_w / ETA_DRIVE_MID
        cop_drive = q_evap / electrical_w if electrical_w > 0.0 else float("nan")
        system_w = electrical_w + fan_w + pump_w
        cop_system = q_evap / system_w if system_w > 0.0 else float("nan")

        row: dict[str, float] = {
            "condition": label,
            "speed_rpm": speed,
            "solver_success": True,
            "q_evap_w": q_evap,
            "refrigerant_power_w": refrigerant_w,
            "shaft_power_w": shaft_w,
            "t_evap_c": t_evap - 273.15,
            "t_cond_c": t_cond - 273.15,
            "pressure_ratio": p_cond / p_evap,
            "eta_isentropic": float(result["compressor_isentropic_efficiency"]),
            "eta_volumetric": float(result["compressor_volumetric_efficiency"]),
            "cop_shaft": cop_shaft,
            "cop_drive": cop_drive,
            "cop_system": cop_system,
            "fan_power_w": fan_w,
            "pump_power_w": pump_w,
        }
        # Drive efficiency the model would need to land on each real anchor.
        for name, q_real, p_real, _ in REAL_ANCHORS:
            real_cop = q_real / p_real
            key = "eta_drive_to_match_" + name.split()[0].lower()
            row[key] = real_cop / cop_shaft if cop_shaft > 0.0 else float("nan")
        rows.append(row)
    return rows


def main() -> None:
    cycle = ClosedR134aCycle()
    all_rows: list[dict[str, float]] = []

    all_rows += scan_conditions(
        cycle,
        label="A_Jilipow_W18_A45",
        coolant_inlet_temperature_k=291.15,
        ambient_temperature_k=318.15,
        coolant_mass_flow_kg_s=0.60,
        pump_speed_rpm=2400.0,
    )
    all_rows += scan_conditions(
        cycle,
        label="B_project_typical_W20_A35",
        coolant_inlet_temperature_k=293.15,
        ambient_temperature_k=308.15,
        coolant_mass_flow_kg_s=0.60,
        pump_speed_rpm=2400.0,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "cop_gap_scan.csv"
    fieldnames: list[str] = []
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Fan power assumed at {FAN_SPEED_RPM:.0f} rpm: {fan_power_w(FAN_SPEED_RPM):.1f} W")
    print(f"Pump power assumed at 2400 rpm: {pump_power_w(2400.0):.1f} W")
    print(f"Drive efficiency (motor x VFD) mid: {ETA_DRIVE_MID:.4f}")
    print()
    header = (
        f"{'condition':<26}{'rpm':>6}{'Qevap_W':>10}{'Tevap':>8}{'Tcond':>8}"
        f"{'PR':>7}{'etaV':>7}{'etaIS':>7}{'COPshaft':>10}{'COPdrive':>10}{'COPsys':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in all_rows:
        if not row.get("solver_success"):
            print(f"{row['condition']:<26}{row['speed_rpm']:>6.0f}   solve failed")
            continue
        print(
            f"{row['condition']:<26}{row['speed_rpm']:>6.0f}{row['q_evap_w']:>10.0f}"
            f"{row['t_evap_c']:>8.2f}{row['t_cond_c']:>8.2f}{row['pressure_ratio']:>7.2f}"
            f"{row['eta_volumetric']:>7.3f}{row['eta_isentropic']:>7.3f}"
            f"{row['cop_shaft']:>10.2f}{row['cop_drive']:>10.2f}{row['cop_system']:>9.2f}"
        )

    print()
    print("Real hardware anchors (system level):")
    for name, q, p, note in REAL_ANCHORS:
        print(f"  {name:<28}{q/1000.0:>6.1f} kW / {p/1000.0:>5.2f} kW = COP {q/p:>5.2f}   {note}")
    real_cops = [q / p for _, q, p, _ in REAL_ANCHORS]
    print(f"  real system COP range: {min(real_cops):.2f} - {max(real_cops):.2f}")
    print(
        "  compressor catalog COP range: "
        f"{COMPRESSOR_CATALOG_COP_RANGE[0]:.2f} - {COMPRESSOR_CATALOG_COP_RANGE[1]:.2f}"
    )

    print()
    print("Coolant-loop pressure drop: model assumption vs real hardware")
    network = ParallelHeaderHydraulicNetwork(5)
    model_dp_pa = float(network.solve(0.60)["network_delta_p"])
    model_hyd_w = hydraulic_power_w(0.60, model_dp_pa)
    jilipow_flow_kg_s = 40.0 / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3
    jilipow_hyd_w = hydraulic_power_w(jilipow_flow_kg_s, 160_000.0)
    print(
        f"  model network dP @ 0.60 kg/s     : {model_dp_pa / 1000.0:7.2f} kPa"
        f"   -> hydraulic {model_hyd_w:6.1f} W"
    )
    print(
        f"  Jilipow rated 40 L/min @ 160 kPa : {160.0:7.2f} kPa"
        f"   -> hydraulic {jilipow_hyd_w:6.1f} W"
    )
    print(f"  pressure-drop ratio real / model : {160_000.0 / model_dp_pa:7.1f}x")
    print(
        f"  CoolantPump reports {pump_power_w(2400.0):.1f} W at 2400 rpm, below the "
        f"model's own {model_hyd_w:.1f} W hydraulic power"
    )

    summary = {
        "eta_drive_mid": ETA_DRIVE_MID,
        "fan_power_full_speed_w": FAN_POWER_FULL_SPEED_W,
        "fan_power_range_w": list(FAN_POWER_RANGE_W),
        "motor_efficiency_range": list(MOTOR_EFFICIENCY_RANGE),
        "vfd_efficiency_range": list(VFD_EFFICIENCY_RANGE),
        "real_system_cop_range": [min(real_cops), max(real_cops)],
        "compressor_catalog_cop_range": list(COMPRESSOR_CATALOG_COP_RANGE),
        "rows": all_rows,
    }
    json_path = OUTPUT_DIR / "cop_gap_summary.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print()
    print(f"CSV  -> {csv_path}")
    print(f"JSON -> {json_path}")


if __name__ == "__main__":
    main()
