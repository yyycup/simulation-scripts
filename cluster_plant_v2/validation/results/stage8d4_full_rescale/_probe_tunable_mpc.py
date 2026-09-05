"""Stage 8D4j quick check: TVP weight injection + horizon bank facade.

Verifies, without the full plant:
1. the tunable build matches build_mpc bit-for-bit under default weights;
2. set_weights retunes the objective with zero CasADi recompilation, and a
   cold-start solve under energy-dominant weights really moves the command
   (the warm-start tracking corner must not mask the injection);
3. the TVP template carries the injected weights over the whole horizon;
4. horizon switching carries x0/u0/_t0 over and keeps solving.
"""

import sys

sys.path.insert(0, "C:/Users/24776/PycharmProjects/PythonProject/集成仿真多种控制")

from time import perf_counter

import numpy as np

from cluster_plant_v2.control.physics_p_nmpc_controller import (
    OnlineTunableNmpc,
    PhysicsPNmpcParameters,
    build_mpc,
    build_tunable_mpc,
)
from cluster_plant_v2.control.physics_p_nmpc_model import (
    STATE_DIM,
    TUNABLE_WEIGHT_TVP_NAMES,
    battery_heat_generation_preview_w,
    build_physics_p_model,
)

CURRENT_A = 560.0
AMBIENT_C = 35.0
Q_GEN_W = battery_heat_generation_preview_w(CURRENT_A)


def current_preview(_t):
    return CURRENT_A


def q_gen_preview(_t, _i):
    return Q_GEN_W


def base_state() -> np.ndarray:
    x0 = np.empty(STATE_DIM, dtype=float)
    x0[1] = 25.0   # t_tank
    x0[2] = 25.2   # t_plate
    x0[3] = 2500.0  # n_comp
    x0[4] = 3600.0  # n_pump
    x0[5] = 1500.0  # q_evap
    x0[6] = 3600.0  # z_pump_1
    x0[7:10] = 24.6  # z_supply
    x0[10:14] = 26.0  # z_return
    x0[14] = 2500.0  # n_comp_cmd_prev
    x0[15] = 3600.0  # n_pump_cmd_prev
    return x0


def warm_state() -> np.ndarray:
    # Warm battery: tracking dominates under the default weights and the
    # optimizer rides the move-rate upper bound (the "tracking corner").
    state = base_state()
    state[0] = 25.35
    return state


parameters = PhysicsPNmpcParameters()
x_warm = warm_state().reshape(-1, 1)
u0_comp, u0_pump = 2500.0, 3600.0

# 1. equivalence with the fixed-weight build under default weights
model_fixed = build_physics_p_model()
t_build = perf_counter()
mpc_fixed = build_mpc(
    model_fixed,
    parameters,
    current_preview=current_preview,
    ambient_temperature_c=AMBIENT_C,
    q_gen_preview=q_gen_preview,
)
mpc_fixed.u0["n_comp_cmd"] = u0_comp
mpc_fixed.u0["n_pump_cmd"] = u0_pump
mpc_fixed.x0["x"] = x_warm
mpc_fixed._t0 = 0.0
mpc_fixed.set_initial_guess()
u_fixed = mpc_fixed.make_step(x_warm)
print(f"build_mpc build+solve: {perf_counter() - t_build:.2f} s, u={np.ravel(u_fixed)}")

model_tunable = build_physics_p_model(include_weight_tvp=True)
t_build = perf_counter()
mpc_tunable, weights = build_tunable_mpc(
    model_tunable,
    parameters,
    current_preview=current_preview,
    ambient_temperature_c=AMBIENT_C,
    q_gen_preview=q_gen_preview,
)
mpc_tunable.u0["n_comp_cmd"] = u0_comp
mpc_tunable.u0["n_pump_cmd"] = u0_pump
mpc_tunable.x0["x"] = x_warm
mpc_tunable._t0 = 0.0
mpc_tunable.set_initial_guess()
u_tunable = mpc_tunable.make_step(x_warm)
print(f"build_tunable_mpc build+solve: {perf_counter() - t_build:.2f} s, u={np.ravel(u_tunable)}")
diff = float(np.max(np.abs(u_fixed - u_tunable)))
print(f"default-weight equivalence max |du|: {diff:.3e}")
assert diff < 1e-4, "tunable build must reproduce build_mpc under default weights"

# 2. weight injection: the compiled solver object must survive retuning
# (zero CasADi recompilation). Behavioral proof uses a FRESH tunable
# instance seeded directly with energy-dominant weights: do-mpc's
# make_step always warm-starts from the previous solution, so only a
# never-solved instance gives an uncontaminated cold start. ALL THREE
# temperature weights must collapse together - leaving w_upper/w_lower at
# 5000 keeps pinning the compressor against the warm battery's overshoot.
weights["w_track"] = 0.01
weights["w_upper"] = 0.01
weights["w_lower"] = 0.01
weights["w_comp"] = 2.0
nlp_solver_before = mpc_tunable.S
mpc_tunable._t0 = 0.0
u_retuned = mpc_tunable.make_step(x_warm)
print(
    f"retuned warm-start solve: u={np.ravel(u_retuned)} "
    "(warm-start may still hold the corner; the fresh-instance test below is decisive)"
)
assert mpc_tunable.S is nlp_solver_before, (
    "retune must not rebuild the compiled NLP (zero-recompile injection)"
)
weights["w_track"] = 500.0  # restore defaults before the later checks
weights["w_upper"] = 5000.0
weights["w_lower"] = 5000.0
weights["w_comp"] = 0.1

energy_parameters = PhysicsPNmpcParameters(
    w_track=0.01, w_upper=0.01, w_lower=0.01, w_comp=2.0
)
mpc_energy, _ = build_tunable_mpc(
    model_tunable,
    energy_parameters,
    current_preview=current_preview,
    ambient_temperature_c=AMBIENT_C,
    q_gen_preview=q_gen_preview,
)
mpc_energy.u0["n_comp_cmd"] = u0_comp
mpc_energy.u0["n_pump_cmd"] = u0_pump
mpc_energy.x0["x"] = x_warm
mpc_energy._t0 = 0.0
mpc_energy.set_initial_guess()
u_energy = mpc_energy.make_step(x_warm)
print(f"fresh energy-dominant instance solve: u={np.ravel(u_energy)}")
assert u_energy[0, 0] < u_tunable[0, 0] - 100.0, (
    "energy-axis weights must collapse the compressor command"
)

# 3. the tvp template must carry the injected weights at every stage
weights["w_comp"] = 7.77  # sentinel visible in the template
template = mpc_tunable.tvp_fun(0.0)
for name in TUNABLE_WEIGHT_TVP_NAMES:
    values = np.asarray(template["_tvp", :, name], dtype=float).reshape(-1)
    assert np.all(np.isfinite(values))
    assert values.size == parameters.horizon_steps + 1
sentinel = np.asarray(template["_tvp", :, "w_comp"], dtype=float).reshape(-1)
assert np.allclose(sentinel, 7.77), "injected weight must fill the whole horizon"
weights["w_comp"] = 0.1
print("tvp template carries all 7 weight channels over the full horizon")

# 4. horizon bank facade: build, switch, solve
t_build = perf_counter()
facade = OnlineTunableNmpc(
    parameters,
    horizon_bank=(30, 60, 90),
    current_preview=current_preview,
    ambient_temperature_c=AMBIENT_C,
    q_gen_preview=q_gen_preview,
)
print(f"facade bank (30/60/90) build: {perf_counter() - t_build:.1f} s")

facade.prepare(warm_state(), u0_comp, u0_pump)
facade.align_time(0.0)
u60 = facade.make_step(x_warm)
print(f"horizon 60 solve: u={np.ravel(u60)}")
np.testing.assert_allclose(u60, u_tunable, atol=1e-3)

assert facade.switch_horizon(90) is True
assert facade.horizon_steps == 90
assert facade.switch_horizon(90) is False
# state carry-over check
np.testing.assert_allclose(
    np.asarray(facade.active_mpc.x0["x"], dtype=float).reshape(-1),
    warm_state(),
)
facade.align_time(15.0)
u90 = facade.make_step(x_warm)
print(f"horizon 90 solve: u={np.ravel(u90)}")

# shared weight store: one call must retune the whole bank
facade.set_weights(w_comp=1.0)
assert facade.weights["w_comp"] == 1.0
try:
    facade.set_weights(w_bogus=1.0)
except ValueError as exc:
    print(f"unknown weight rejected: {exc}")
else:
    raise AssertionError("unknown weight key must raise")
try:
    facade.set_weights(w_comp=-1.0)
except ValueError as exc:
    print(f"negative weight rejected: {exc}")
else:
    raise AssertionError("negative weight must raise")

try:
    facade.switch_horizon(45)
except ValueError as exc:
    print(f"off-bank horizon rejected: {exc}")
else:
    raise AssertionError("off-bank horizon must raise")

print("quick check OK")
