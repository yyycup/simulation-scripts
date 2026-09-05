"""Frozen defaults shared by the independent Cluster Plant V2 runtime.

PyCharm navigation:
- Simulation and Cluster defaults
- Hydraulic network, pump, and tank defaults
- Compressor, evaporator, and transport defaults
- Electrical accounting assumptions (reporting only)
- Coolant and retained Pack-interface constants

These names centralize existing values; they do not introduce new physics.
The electrical accounting block is the one exception: those values are
reporting-only engineering assumptions from the real-data anchor audit
(``COP_REAL_ANCHOR_AUDIT_20260901.md``). They add wall-power outputs but
never feed back into plant dynamics.
"""


# Simulation and Cluster

SIMULATION_TIME_STEP_S = 5.0
DEFAULT_CLUSTER_PACK_COUNT = 5


# Hydraulic Network

DESIGN_PACK_MASS_FLOW_KG_S = 0.08925
BRANCH_DESIGN_DELTA_P_PA = 20_000.0
BASE_BRANCH_RESISTANCE_PA_PER_KG_S2 = (
    BRANCH_DESIGN_DELTA_P_PA / DESIGN_PACK_MASS_FLOW_KG_S**2
)
SMALL_HEADER_TO_BRANCH_RESISTANCE_RATIO = 0.001
MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO = 0.005


# Coolant Pump

LEGACY_REFERENCE_SPEED_RPM = 4000.0
LEGACY_REFERENCE_FLOW_L_MIN = 28.0
LEGACY_REFERENCE_POWER_W = 27.3979
DEFAULT_SHUTOFF_TO_OPERATING_PRESSURE_RATIO = 2.0
MIN_PUMP_SPEED_RPM = 1600.0
MAX_PUMP_SPEED_RPM = 4800.0


# Coolant Tank and Coolant

LEGACY_TANK_VOLUME_L = 3.0
COOLANT_DENSITY_KG_M3 = 1071.0
COOLANT_SPECIFIC_HEAT_J_KG_K = 3391.0


# Compressor, Evaporator, and Transport

MINIMUM_COMPRESSOR_SPEED_RPM = 1000.0
MAXIMUM_COMPRESSOR_SPEED_RPM = 6000.0
MAXIMUM_FAN_SPEED_RPM = 4000.0
COMPRESSOR_TIME_CONSTANT_S = 5.0
EVAPORATOR_TIME_CONSTANT_S = 45.0
SUPPLY_TRANSPORT_DELAY_S = 15.0
RETURN_TRANSPORT_DELAY_S = 20.0


# Retained Pack Interface

PLATE_NODE_HEAT_CAPACITY_TOTAL = 6000.0
BATTERY_PLATE_AREA_M2 = 0.5
BATTERY_PLATE_NOMINAL_HTC_W_M2_K = 2000.0
COLD_PLATE_REFERENCE_MASS_FLOW_KG_S = 0.1428
NOMINAL_COOLANT_MASS_FLOW_KG_S = 1.2


# Electrical Input Accounting

# The frozen R134a cycle reports compressor SHAFT power only: it stops at the
# mechanical efficiency of the compressor itself. A real unit draws more from
# the grid. These factors close that gap for energy reporting; they never
# enter the refrigerant or thermal equations, so every frozen physics result
# is untouched.
#
# Anchored on model_data/real_data_anchors/ (retrieved 2026-09-01): the
# CoolProp back-out of the Boyard JVSB150Z24 duty-A point implies
# eta_isentropic x eta_motor = 0.641, i.e. the electric motor is NOT part of
# the cycle model's efficiency map.
COMPRESSOR_MOTOR_EFFICIENCY = 0.90
COMPRESSOR_VFD_EFFICIENCY = 0.945
DRIVE_TRAIN_EFFICIENCY = COMPRESSOR_MOTOR_EFFICIENCY * COMPRESSOR_VFD_EFFICIENCY

# refrigeration.py has no condenser-fan power term at all, yet the 8 m2 coil
# is swept by up to 6.5 kg/s of air. Axial condenser fan on this class of
# unit: order 150-400 W at full speed. Cubic law in speed.
CONDENSER_FAN_POWER_FULL_SPEED_W = 275.0
CONDENSER_FAN_REFERENCE_SPEED_RPM = 4000.0
