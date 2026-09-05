"""Discrete 16-state do-mpc model built on the frozen Physics-P predictor.

The internal prediction model of the NMPC is the frozen 14-state Physics-P
5 s one-step map from ``single_pack_plant``; it is neither re-identified nor
modified here. Two augmented states carry the previously commanded inputs so
that move-rate (delta-u) terms and bounds can be expressed on the augmented
state ``x = [x_p, n_comp_cmd_prev, n_pump_cmd_prev]``:

    x_p(k+1) = f_PhysicsP(x_p(k), u(k), d(k))
    x(k+1)   = [x_p(k+1), n_comp_cmd(k), n_pump_cmd(k)]

Physics-P temperatures are in degC throughout; the Plant adapter is
responsible for the K -> degC conversion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import casadi as ca
import do_mpc.model
import numpy as np

# ``single_pack_plant`` is a sibling project that is not installed as a
# package; expose the shared parent directory on ``sys.path``.
_PARENT_DIR = Path(__file__).resolve().parents[2]
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(_PARENT_DIR))

from single_pack_plant.predictor.physics_p_casadi import CasadiPhysicsP
from single_pack_plant.predictor.state_space import NU, NX

PHYSICS_P_ARTIFACT = (
    _PARENT_DIR
    / "single_pack_plant"
    / "model_data"
    / "physics_p_operational_8d4f.json"
)
# Stage 8D4e: the capacity block of the frozen predictor was re-derived on
# the 8D4 hardware (72 cc compressor envelope, q_upper 4800 -> 25900 W;
# validation MAPE 0.45%). Stage 8D4f adds the cluster-deployment thermal
# refit consumed by ClusterCasadiPhysicsP (top-level ``cluster_thermal``
# block: plate-fluid reference-flow anchor fitted to 280 W/K on the
# 2400 rpm pump sweep; heat-generation scale 1.0). The legacy artifact
# physics_p_operational_v1.json underestimated capacity by ~85% and is
# kept for reference only.

# 14 physical states followed by the two augmented previous-command states.
PHYSICS_STATE_NAMES = (
    "t_bat",
    "t_tank",
    "t_plate",
    "n_comp",
    "n_pump",
    "q_evap",
    "z_pump_1",
    "z_supply_1",
    "z_supply_2",
    "z_supply_3",
    "z_return_1",
    "z_return_2",
    "z_return_3",
    "z_return_4",
)
AUGMENTED_STATE_NAMES = ("n_comp_cmd_prev", "n_pump_cmd_prev")
STATE_DIM = NX + len(AUGMENTED_STATE_NAMES)

N_COMP_CMD_INDEX = NX
N_PUMP_CMD_INDEX = NX + 1

# Stage 8D4j: cost-weight TVP channels. When ``include_weight_tvp=True``
# the model carries the seven objective weights as time-varying
# parameters so a built MPC can be re-tuned at runtime by rewriting the
# TVP template (no CasADi recompile). They never enter the rhs, so the
# prediction map and ``check_step_equivalence`` are unaffected.
TUNABLE_WEIGHT_TVP_NAMES = (
    "w_track",
    "w_upper",
    "w_lower",
    "w_comp",
    "w_pump",
    "w_dcomp",
    "w_dpump",
)

# Stage 8D4c: the cluster pump selection was rescaled x2 (Stage 8D4b,
# reference flow 56 L/min), while the frozen predictor was calibrated on
# the legacy 28 L/min pump. Both pumps are affine in speed, so the model
# must see the virtual legacy-equivalent speed 2x the actual command to
# predict the true coolant flow (model 0.596 kg/s @ virtual 4800 rpm =
# rescaled pump 0.5998 kg/s @ actual 2400 rpm). The NMPC decides the pump
# input in this virtual domain, bounded to the predictor training range
# [1600, 4800] rpm; the application layer divides by this scale before
# commanding the plant and multiplies measured commands before mapping
# states back into the model.
PUMP_VIRTUAL_SPEED_SCALE = 2.0

_COMP_NOMINAL_RPM = 6000.0
_PUMP_NOMINAL_RPM = 4800.0
NOMINAL_RPM = {"n_comp_cmd": _COMP_NOMINAL_RPM, "n_pump_cmd": _PUMP_NOMINAL_RPM}


def load_frozen_predictor(
    artifact: str | Path = PHYSICS_P_ARTIFACT, dt_s: float = 5.0
) -> CasadiPhysicsP:
    """Load the validated frozen Physics-P artifact as a CasADi function."""
    return ClusterCasadiPhysicsP(artifact, dt_s=dt_s)


class ClusterCasadiPhysicsP(CasadiPhysicsP):
    """Stage 8D4f: cluster-deployment correction of the frozen P map.

    The frozen predictor was calibrated on the single-pack plant, where one
    pack sees the full coolant flow. The final cluster plant shares that
    flow across ``CLUSTER_PACK_COUNT`` thermally parallel packs, so the
    pack-level temperature channel must see 1/N of the flow (the plate
    epsilon-NTU coupling then strengthens automatically with the lower
    per-pack flow), while the return temperature rise fed to the tank sums
    the plate heat of all N packs. At the S0 settled operating point this
    moves the pack-fluid conductance from 545 to ~390 W/K and the return
    rise from 0.9 to ~4 K, matching the plant steady state; without it the
    model equilibrium battery temperature sits ~1.2 K low and the closed
    loop drifts warm (the residual 8D4e offset).
    """

    CLUSTER_PACK_COUNT = 5.0

    def _build_function(self) -> ca.Function:
        x = ca.SX.sym("x", NX)
        u = ca.SX.sym("u", NU)
        # ``q_gen_w`` is explicit so the NMPC can forecast SOC-dependent
        # battery heat rather than assuming a fixed resistance from current.
        d = ca.SX.sym("d", 3)  # [current_a, t_ambient_c, q_gen_w]
        dt = self.dt_s
        dynamic = self.artifact["dynamic"]
        thermal = self.artifact["thermal"]
        n_packs = float(self.CLUSTER_PACK_COUNT)

        t_bat, t_tank, t_plate = x[0], x[1], x[2]
        n_comp, n_pump, q_evap = x[3], x[4], x[5]
        z_pump = x[6]
        z_supply = x[7:10]
        z_return = x[10:14]
        n_comp_next = self._lag(n_comp, u[0], dt, float(dynamic["tau_comp_s"]))
        n_pump_next = self._lag(n_pump, u[1], dt, float(dynamic["tau_pump_s"]))
        q_steady = self._capacity(n_comp_next, z_pump, t_tank, d[1])
        q_evap_next = self._lag(q_evap, q_steady, dt, float(dynamic["tau_evap_s"]))

        pump_ref = float(thermal["n_pump_ref_rpm"])
        pump_ratio = ca.fmax(1.0e-6, z_pump / pump_ref)
        flow_capacity = float(thermal["coolant_mass_flow_ref_kg_s"]) * pump_ratio * float(thermal["coolant_cp_j_kg_k"])
        evaporator_out = t_tank - q_evap_next / flow_capacity
        t_supply = z_supply[0]
        conductance = float(thermal["battery_plate_conductance_w_k"])
        ref_flow = float(thermal["coolant_mass_flow_ref_kg_s"]) * float(thermal["coolant_cp_j_kg_k"])
        # Plate-fluid coupling mirrors the plant cold plate: the HTC scales
        # with the per-pack flow share to the 0.8 power, anchored at the
        # fitted reference-flow value (W/K) stored in the top-level
        # ``cluster_thermal`` block (the artifact validator whitelists the
        # legacy ``thermal`` keys). The legacy lumped ``eff x min(flow, ...)``
        # form caps below the plant-implied coupling under the 1/5 per-pack
        # flow and is not used here.
        cluster_thermal = self.artifact.get("cluster_thermal", {})
        plate_flow_ref = float(
            cluster_thermal.get("plate_fluid_ref_flow_w_k", ref_flow / n_packs)
        )
        plate_fluid = plate_flow_ref * pump_ratio**0.8
        total_g = conductance + plate_fluid
        plate_equilibrium = (conductance * t_bat + plate_fluid * t_supply) / total_g
        plate_tau = float(thermal["plate_tau_s"]) * ref_flow / total_g
        t_plate_next = self._lag(t_plate, plate_equilibrium, dt, plate_tau)
        q_batt_plate = conductance * (t_bat - t_plate_next)
        q_plate_fluid = plate_fluid * (t_plate_next - t_supply)
        # All N packs heat the common return line.
        plate_out = t_supply + n_packs * q_plate_fluid / flow_capacity
        q_gen = d[2] * float(thermal.get("battery_heat_generation_scale", 1.0))
        ambient_loss = float(thermal["ambient_conductance_w_k"]) * (t_bat - d[1])
        t_bat_next = t_bat + dt * (q_gen - q_batt_plate - ambient_loss) / float(thermal["battery_heat_capacity_j_k"])
        t_tank_next = t_tank + dt * flow_capacity * (z_return[0] - t_tank) / float(thermal["coolant_heat_capacity_j_k"])
        x_next = ca.vertcat(
            t_bat_next, t_tank_next, t_plate_next, n_comp_next, n_pump_next,
            q_evap_next, n_pump_next, z_supply[1], z_supply[2], evaporator_out,
            z_return[1], z_return[2], z_return[3], plate_out,
        )
        # Overriding ``_build_function`` means the parent ``__init__`` stores
        # this cluster-corrected map directly; the frozen-artifact validator
        # (single-pack era) still runs at load time and only constrains the
        # legacy ``plate_fluid_effectiveness`` field, which this structure
        # no longer consumes.
        return ca.Function("physics_p_step", [x, u, d], [x_next], ["x", "u", "d"], ["x_next"])


def build_physics_p_model(
    predictor: CasadiPhysicsP | None = None, *, include_weight_tvp: bool = False
):
    """Build the do-mpc discrete model ``x(k+1) = [P(x[:14], u, d), u]``.

    With ``include_weight_tvp=True`` the objective cost weights are added
    as TVP channels for the runtime-tunable NMPC facade; the dynamics are
    identical either way.
    """
    if predictor is None:
        predictor = load_frozen_predictor()

    model = do_mpc.model.Model(model_type="discrete")
    x = model.set_variable("_x", "x", shape=(STATE_DIM, 1))
    n_comp_cmd = model.set_variable("_u", "n_comp_cmd")
    n_pump_cmd = model.set_variable("_u", "n_pump_cmd")
    current_a = model.set_variable("_tvp", "current_a")
    t_amb_c = model.set_variable("_tvp", "t_amb_c")
    q_gen_w = model.set_variable("_tvp", "q_gen_w")
    if include_weight_tvp:
        for name in TUNABLE_WEIGHT_TVP_NAMES:
            model.set_variable("_tvp", name)

    u = ca.vertcat(n_comp_cmd, n_pump_cmd)
    d = ca.vertcat(current_a, t_amb_c, q_gen_w)
    # Element-wise assembly: CasADi row slicing on a dense SX column returns
    # a wide sparse block instead of a column vector.
    x_physics = ca.vertcat(*[x[i] for i in range(NX)])
    physics_next = predictor.function(x_physics, u, d)
    model.set_rhs("x", ca.vertcat(physics_next, n_comp_cmd, n_pump_cmd))
    model.setup()
    return model


def battery_heat_generation_preview_w(current_a: float) -> float:
    """4P ohmic heat preview consistent with the Physics-P training data."""
    current = float(current_a)
    return ((current / 4.0) ** 2) * 0.001 * 52.0


class ClusterHeatGenerationPreview:
    """SOC-aware per-pack heat forecaster mirroring ``ReducedBatteryPack``.

    In steady state the Plant ROM zone heat ``I^2 r0 - I eta1 - I eta2``
    converges to ``I^2 (r0 + r1 + r2)``, so the preview rolls branch SOC
    forward by Coulomb counting over the known current schedule, switches
    the HPPC resistance tables on the current direction, and returns
    ``sum_b I_branch^2 x 13 x (r0+r1+r2)(SOC, T)`` with cell temperatures
    frozen at their measured values. This reproduces the conventions of
    :meth:`cluster_plant_v2.thermal.pack_rom.ReducedBatteryPack.step`
    (hard direction switch in ``calculate_branch_currents``, capacity-based
    Coulomb counting) without tracking polarization-state dynamics.
    """

    def __init__(
        self,
        current_schedule_a,
        dt_s: float,
        *,
        initial_soc_branch=None,
        temperature_k: float = 298.15,
        capacity_ah: float = 280.0,
    ) -> None:
        from scipy.interpolate import RegularGridInterpolator

        from cluster_plant_v2.hppc_parameters import load_hppc_parameters

        self.schedule = np.asarray(current_schedule_a, dtype=float).reshape(-1)
        if self.schedule.size == 0 or not np.all(np.isfinite(self.schedule)):
            raise ValueError("current schedule must be non-empty and finite")
        self.dt_s = float(dt_s)
        if initial_soc_branch is None:
            soc = np.full(4, 0.95, dtype=float)
        else:
            soc = np.asarray(initial_soc_branch, dtype=float).reshape(-1)

        data = load_hppc_parameters()
        axes = (data["soc"], data["temp"])
        soc_min, soc_max = float(np.min(axes[0])), float(np.max(axes[0]))
        temp_min, temp_max = float(np.min(axes[1])), float(np.max(axes[1]))
        query_temp = min(max(float(temperature_k), temp_min), temp_max)
        # With temperatures frozen uniformly across zones the per-branch
        # total resistance reduces to n_series x the common cell value.
        series_count = 13.0

        def branch_total_resistance(
            soc_branches: np.ndarray, suffix: str
        ) -> np.ndarray:
            """Per-branch ``n_series x (r0+r1+r2)`` at the current SOC."""
            query_soc = np.clip(soc_branches, soc_min, soc_max)
            points = np.column_stack(
                (query_soc, np.full(query_soc.size, query_temp))
            )
            cell_total = np.zeros(query_soc.size, dtype=float)
            for element in range(3):
                interpolator = RegularGridInterpolator(
                    axes,
                    data[f"r{element}_{suffix}"],
                    bounds_error=False,
                    fill_value=None,
                )
                cell_total += np.asarray(interpolator(points), dtype=float)
            return cell_total * series_count

        steps = self.schedule.size
        soc_rollout = np.empty((steps + 1, soc.size), dtype=float)
        q_gen_w = np.empty(steps, dtype=float)
        branch_currents = np.empty((steps, soc.size), dtype=float)
        soc_rollout[0] = soc
        for index in range(steps):
            current = float(self.schedule[index])
            table = branch_total_resistance(
                soc_rollout[index], "dis" if current >= 0.0 else "chg"
            )
            conductance = 1.0 / (table + 1e-9)
            branches = current * conductance / conductance.sum()
            branch_currents[index] = branches
            q_gen_w[index] = float(np.sum(branches**2 * table))
            soc_rollout[index + 1] = soc_rollout[index] - branches * self.dt_s / (
                capacity_ah * 3600.0
            )
        self.q_gen_schedule_w = q_gen_w
        self.soc_rollout = soc_rollout
        self.branch_current_schedule = branch_currents

    @classmethod
    def from_pack(cls, battery, current_schedule_a, dt_s: float):
        """Build the preview from a live ``ReducedBatteryPack`` measurement."""
        return cls(
            current_schedule_a,
            dt_s,
            initial_soc_branch=battery.get_soc_array(),
            temperature_k=float(np.mean(battery.temps)),
            capacity_ah=float(battery.config["capacity"]),
        )

    def __call__(self, preview_time_s: float, current_a: float) -> float:
        index = int(round(float(preview_time_s) / self.dt_s))
        index = min(max(index, 0), self.q_gen_schedule_w.size - 1)
        return float(self.q_gen_schedule_w[index])

    def soc_at(self, preview_time_s: float) -> float:
        index = int(round(float(preview_time_s) / self.dt_s))
        index = min(max(index, 0), self.soc_rollout.shape[0] - 1)
        return float(np.mean(self.soc_rollout[index]))


def check_step_equivalence() -> dict[str, float]:
    """Smoke check: the 16-dim model rhs reproduces the frozen P map."""
    predictor = load_frozen_predictor()
    model = build_physics_p_model(predictor)
    state = np.concatenate(
        [
            np.array([25.0, 26.0, 25.5, 4000.0, 3600.0, 1500.0, 3600.0]),
            np.full(3, 24.0),
            np.full(4, 26.5),
            [4000.0, 3600.0],
        ]
    )
    control = np.array([4200.0, 3800.0])
    disturbance = np.array(
        [560.0, 35.0, battery_heat_generation_preview_w(560.0)]
    )

    direct_next = predictor.step(state[:NX], control, disturbance)
    rhs_function = ca.Function(
        "model_rhs",
        [model.x.cat, model.u.cat, model.tvp.cat],
        [model._rhs.cat],
    )
    rhs_next = np.asarray(
        rhs_function(
            state.reshape(-1, 1),
            control.reshape(-1, 1),
            disturbance.reshape(-1, 1),
        ),
        dtype=float,
    ).reshape(STATE_DIM)
    physics_difference = float(np.max(np.abs(rhs_next[:NX] - direct_next)))
    augmented_difference = float(
        np.max(np.abs(rhs_next[NX:] - control))
    )
    return {
        "physics_max_abs_difference": physics_difference,
        "augmented_max_abs_difference": augmented_difference,
    }


if __name__ == "__main__":
    summary = check_step_equivalence()
    print(f"Physics-P CasADi step equivalence: {summary}")
