"""Stage 8D4d probe: can condenser fan modulation extend the capacity floor?

The 8D4c run exhausted both actuators (compressor pinned at the 1000 rpm
floor, pump at the 2400 rpm ceiling) and the tank still sank: the chiller
floor capacity at fan 1200 rpm exceeds the ~4.2 kW RegD mean load. Check
whether reducing the condenser fan speed brings the floor capacity below
the mean load, which would give the loop continuous authority without
compressor on/off cycling.
"""

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

from cluster_plant_v2.refrigeration import ClosedR134aCycle

cycle = ClosedR134aCycle()
print("capacity vs fan speed at compressor floor (0.6 kg/s coolant):")
for tin_c in (20.0, 23.0):
    for comp_rpm in (1000.0, 1600.0):
        for fan_rpm in (1200.0, 800.0, 400.0, 200.0):
            result = cycle.solve(
                compressor_speed_rpm=comp_rpm,
                fan_speed_rpm=fan_rpm,
                coolant_inlet_temperature_k=tin_c + 273.15,
                coolant_mass_flow_kg_s=0.599760,
                ambient_temperature_k=308.15,
            )
            ok = result["solver_success"]
            q = float(result["q_evaporator_w"]) if ok else float("nan")
            print(
                f"  Tin {tin_c:.0f} C, comp {comp_rpm:.0f} rpm, "
                f"fan {fan_rpm:.0f} rpm: q_evap={q:7.0f} W, ok={ok}"
            )
