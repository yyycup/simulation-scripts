"""Stage 8D4b pump physics audit: model power vs hydraulic power vs efficiency.

Checks whether the rescaled pump selection (reference flow 56 L/min, head
re-matched to the network) keeps the pump power calibration physically
consistent: the CoolantPump.power_w model is reference_power_w * (N/N_ref)^3,
and reference_power_w defaults to the legacy 27.4 W that was calibrated for
the OLD 28 L/min pump.
"""

from pathlib import Path
import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

from cluster_plant_v2.hydraulics import CoolantPump
from cluster_plant_v2.parameters import (
    COOLANT_DENSITY_KG_M3,
    LEGACY_REFERENCE_FLOW_L_MIN,
    LEGACY_REFERENCE_POWER_W,
    LEGACY_REFERENCE_SPEED_RPM,
)
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    PUMP_SPEED_RPM,
    build_medium_header_network,
)

net = build_medium_header_network()


def audit(pump: CoolantPump, tag: str) -> None:
    for rpm in (3600.0, 4000.0, 4800.0):
        op = pump.solve_operating_point(rpm, net)
        q = float(op["total_mass_flow_kg_s"])
        vflow = float(op["total_volume_flow_l_min"])
        dp = float(op["network_delta_p_pa"])
        p_model = pump.power_w(rpm)
        p_hyd = q * dp / COOLANT_DENSITY_KG_M3 * 1000.0
        eta = p_hyd / p_model * 100.0 if p_model > 0 else float("nan")
        print(
            f"{tag} {rpm:.0f} rpm: q={q:.4f} kg/s ({vflow:.1f} L/min), "
            f"dp={dp / 1000:.1f} kPa, P_model={p_model:.1f} W, "
            f"P_hyd={p_hyd:.1f} W, implied_eta={eta:.0f}%"
        )


old_ref_dp = float(
    net.solve(LEGACY_REFERENCE_FLOW_L_MIN / 1000 / 60 * COOLANT_DENSITY_KG_M3)[
        "network_delta_p"
    ]
)
old_pump = CoolantPump(
    reference_operating_delta_p_pa=old_ref_dp,
    reference_volume_flow_l_min=LEGACY_REFERENCE_FLOW_L_MIN,
)
print(f"old pump: ref {LEGACY_REFERENCE_FLOW_L_MIN:.0f} L/min @ "
      f"{LEGACY_REFERENCE_SPEED_RPM:.0f} rpm, head {old_ref_dp / 1000:.1f} kPa, "
      f"P_ref {LEGACY_REFERENCE_POWER_W:.1f} W")
audit(old_pump, "OLD")

new_ref_flow_kg_s = 56.0 / 1000 / 60 * COOLANT_DENSITY_KG_M3
new_ref_dp = float(net.solve(new_ref_flow_kg_s)["network_delta_p"])
new_pump = CoolantPump(
    reference_operating_delta_p_pa=new_ref_dp,
    reference_volume_flow_l_min=56.0,
)
print(f"new pump (8D4b): ref 56 L/min @ 4000 rpm, head {new_ref_dp / 1000:.1f} kPa, "
      f"P_ref (default!) {new_pump.reference_power_w:.1f} W")
audit(new_pump, "NEW")
