"""Stage 8D4c probe: chiller capability at the half-flow operating envelope.

The pump domain scaling caps the actual pump command at 2400 rpm (virtual
2x = 4800 stays inside the frozen predictor's n_pump domain), so the NMPC
loop runs at 0.3-0.6 kg/s. Check the spike coverage there before rerunning.
"""

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

from cluster_plant_v2.refrigeration import ClosedR134aCycle
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    build_engineering_pump,
    build_medium_header_network,
)

net = build_medium_header_network()
pump = build_engineering_pump(net)
for rpm in (1600.0, 2400.0):
    op = pump.solve_operating_point(rpm, net)
    flow = float(op["total_mass_flow_kg_s"])
    print(f"new pump {rpm:.0f} rpm: {flow:.4f} kg/s")

cycle = ClosedR134aCycle()
for tin_c, rpm in ((20.0, 6000.0), (20.0, 4000.0), (23.0, 6000.0), (25.0, 4000.0)):
    result = cycle.solve(
        compressor_speed_rpm=rpm,
        fan_speed_rpm=1200.0,
        coolant_inlet_temperature_k=tin_c + 273.15,
        coolant_mass_flow_kg_s=0.599760,
        ambient_temperature_k=308.15,
    )
    print(
        f"{rpm:.0f} rpm @ {tin_c:.0f} C, 0.6 kg/s: "
        f"q_evap={float(result['q_evaporator_w']):.0f} W, "
        f"solver_ok={result['solver_success']}, "
        f"Tevap={float(result['evaporating_saturation_temperature_k']) - 273.15:.2f} C"
    )
