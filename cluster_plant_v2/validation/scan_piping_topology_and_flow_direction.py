"""Scan how piping topology and coolant flow direction reshape cluster balance.

Two independent experiments. Both are read-only with respect to every frozen
artifact: nothing here edits parameters, the cycle, or the ROM.

E1  Flow direction
    Same reverse-return network; coolant traverses the three cold-plate zones
    either forward (zone 1 -> 2 -> 3) or reverse (3 -> 2 -> 1). The comparison
    is only interesting because the battery heat is *not* uniform along the
    plate: ``pack_reference`` scales generation by ``1 - 0.4 * col / 12``, so
    column 0 is the hottest. Flipping the traversal decides whether the
    coldest coolant meets the hottest cells or the coolest ones.

E2  Piping topology
    Reverse return (同程式) against direct return (异程式). In the project's
    reverse-return network every branch traverses N+1 = 6 header segments, so
    the five loops are exactly equal. In a direct-return network branch i
    traverses 2i segments, so branch 1 is the shortest and branch 5 the
    longest, and the flow decays monotonically along the header.

    Each layout gets a pump re-selected against its own network
    (``build_engineering_pump``), so both deliver the same reference flow.
    Sharing one pump would measure the pump mismatch instead of the piping.

Run from the parent of ``cluster_plant_v2``::

    "C:/Users/24776/miniforge3/envs/btms/python.exe" -m \
        cluster_plant_v2.validation.scan_piping_topology_and_flow_direction
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cluster_plant_v2.hydraulics import CoolantTank, ParallelHeaderHydraulicNetwork
from cluster_plant_v2.parameters import (
    COMPRESSOR_TIME_CONSTANT_S,
    EVAPORATOR_TIME_CONSTANT_S,
)
from cluster_plant_v2.plant import ClusterPlant
from cluster_plant_v2.refrigeration import (
    ClosedR134aCycle,
    CompressorSpeedActuator,
)
from cluster_plant_v2.thermal.cluster import ReducedCluster
from cluster_plant_v2.validation.validate_refrigerated_cluster_cooling_loop import (
    AMBIENT_TEMPERATURE_K,
    DT_S,
    INITIAL_SOC,
    INITIAL_TEMPERATURE_K,
    PUMP_SPEED_RPM,
    build_engineering_pump,
    build_medium_header_network,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_DIR = RESULTS_DIR / "piping_topology_scan_20260901"

FAN_SPEED_RPM = 1200.0
COMPRESSOR_RPM = 4000.0
CURRENTS_A = (560.0, 1120.0)
# Inter-pack spread is still climbing at 600 s, so run out to 1800 s and read
# the trend from the checkpoints rather than trusting a single snapshot.
CHECKPOINT_TIMES_S = (300.0, 600.0, 1200.0, 1800.0)

TOPOLOGY_LABELS = {
    "reverse_return": "同程式",
    "direct_return": "异程式",
}
DIRECTION_LABELS = {"forward": "正向 1→2→3", "reverse": "反向 3→2→1"}


class DirectReturnHeaderNetwork(ParallelHeaderHydraulicNetwork):
    """Direct-return (异程式) parallel-header network.

    The parent class encodes a reverse return: the return header runs the same
    way as the supply header, so branch i traverses supply segments 1..i plus
    return segments i..N, i.e. N+1 segments for every branch.

    Here the return header runs back the other way, so branch i traverses
    supply segments 1..i plus return segments 1..i, i.e. 2i segments. Branch 1
    is the shortest loop and branch N the longest, which is exactly the
    imbalance a reverse return exists to remove.

    The resistance coefficients are deliberately identical to the reverse-return
    case; only the topology changes.
    """

    def _quantities(self, pack_mass_flows: np.ndarray) -> dict[str, np.ndarray]:
        flows = np.asarray(pack_mass_flows, dtype=float).reshape(self.n_packs)
        supply_flows = np.cumsum(flows[::-1])[::-1]
        return_flows = np.cumsum(flows[::-1])[::-1]
        supply_delta_p = self.supply_segment_resistances * supply_flows**2
        return_delta_p = self.return_segment_resistances * return_flows**2
        branch_delta_p = self.branch_resistances * flows**2
        supply_path_delta_p = np.cumsum(supply_delta_p)
        return_path_delta_p = np.cumsum(return_delta_p)
        path_delta_p = supply_path_delta_p + branch_delta_p + return_path_delta_p

        supply_nodes = np.empty(self.n_packs, dtype=float)
        return_nodes = np.empty(self.n_packs, dtype=float)
        for index in range(self.n_packs):
            supply_downstream = (
                supply_flows[index + 1] if index + 1 < self.n_packs else 0.0
            )
            return_downstream = (
                return_flows[index + 1] if index + 1 < self.n_packs else 0.0
            )
            supply_nodes[index] = (
                supply_flows[index] - flows[index] - supply_downstream
            )
            return_nodes[index] = (
                return_downstream + flows[index] - return_flows[index]
            )
        return {
            "supply_segment_flows": supply_flows,
            "return_segment_flows": return_flows,
            "supply_segment_delta_p": supply_delta_p,
            "return_segment_delta_p": return_delta_p,
            "branch_delta_p": branch_delta_p,
            "pack_path_delta_p": path_delta_p,
            "node_mass_balance_residuals": np.concatenate(
                (supply_nodes, return_nodes)
            ),
        }


def build_direct_return_network() -> DirectReturnHeaderNetwork:
    """Direct-return network carrying the reverse-return resistances verbatim."""
    base = build_medium_header_network()
    return DirectReturnHeaderNetwork(
        n_packs=base.n_packs,
        branch_resistances=base.branch_resistances,
        supply_segment_resistances=base.supply_segment_resistances,
        return_segment_resistances=base.return_segment_resistances,
    )


def build_plant(topology: str, current_a: float, direction: str) -> ClusterPlant:
    """Assemble a plant whose pump is selected against its own network."""
    if topology == "reverse_return":
        network = build_medium_header_network()
    elif topology == "direct_return":
        network = build_direct_return_network()
    else:
        raise ValueError(f"unknown topology {topology!r}")

    cluster = ReducedCluster(
        n_packs=5,
        hydraulic_mode="header_network",
        hydraulic_network=network,
        pack_config={
            "initial_soc": INITIAL_SOC,
            "initial_battery_temperature_k": INITIAL_TEMPERATURE_K,
            "initial_plate_temperature_k": INITIAL_TEMPERATURE_K,
        },
    )
    return ClusterPlant.from_equilibrium(
        cluster=cluster,
        hydraulic_network=network,
        pump=build_engineering_pump(network),
        tank=CoolantTank(initial_temperature_k=INITIAL_TEMPERATURE_K),
        refrigeration_cycle=ClosedR134aCycle(),
        compressor_actuator=CompressorSpeedActuator(
            initial_speed_rpm=COMPRESSOR_RPM,
            time_constant_s=COMPRESSOR_TIME_CONSTANT_S,
        ),
        dt_s=DT_S,
        pump_speed_rpm=PUMP_SPEED_RPM,
        fan_speed_rpm=FAN_SPEED_RPM,
        initial_cluster_current_a=current_a,
        ambient_temperature_k=AMBIENT_TEMPERATURE_K,
        direction=direction,
        evaporator_time_constant_s=EVAPORATOR_TIME_CONSTANT_S,
    )


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    experiment: str
    topology: str
    direction: str
    current_a: float


def build_case_specs() -> list[CaseSpec]:
    """Unique cases only.

    ``reverse_return`` + ``forward`` is the shared reference point of both
    experiments, so it is emitted once and reused rather than run twice.
    """
    specs: list[CaseSpec] = []
    for current_a in CURRENTS_A:
        tag = "560A" if current_a == 560.0 else "1120A"
        for direction in ("forward", "reverse"):
            specs.append(
                CaseSpec(
                    f"E1_{tag}_{direction}",
                    "E1_flow_direction",
                    "reverse_return",
                    direction,
                    current_a,
                )
            )
        specs.append(
            CaseSpec(
                f"E2_{tag}_direct_return",
                "E2_piping_topology",
                "direct_return",
                "forward",
                current_a,
            )
        )
    return specs


def _snapshot(
    case: CaseSpec,
    plant: ClusterPlant,
    result: dict[str, object],
    time_s: float,
) -> dict[str, object]:
    """Flatten one Plant output into a comparable metric row."""
    cluster = result["cluster_result"]
    flows = np.asarray(cluster["pack_mass_flows_kg_s"], dtype=float)
    battery_max = np.asarray(
        cluster["pack_battery_max_temperatures_k"], dtype=float
    )
    battery_avg = np.asarray(
        cluster["pack_battery_average_temperatures_k"], dtype=float
    )
    hydraulic = plant.hydraulic_network.solve(float(flows.sum()))
    path_dp = np.asarray(hydraulic["pack_path_delta_p"], dtype=float)

    return {
        "case_id": case.case_id,
        "experiment": case.experiment,
        "topology": f"{case.topology}（{TOPOLOGY_LABELS[case.topology]}）",
        "direction": DIRECTION_LABELS[case.direction],
        "current_a": case.current_a,
        "time_s": time_s,
        "total_flow_kg_s": float(flows.sum()),
        "flow_spread_pct": float(np.ptp(flows) / flows.mean() * 100.0),
        "network_delta_p_pa": float(hydraulic["network_delta_p"]),
        "path_dp_spread_pa": float(np.ptp(path_dp)),
        "pump_power_w": float(result["pump_power_w"]),
        "supply_temp_c": float(result["cluster_supply_temperature_k"] - 273.15),
        "return_temp_c": float(result["cluster_return_temperature_k"] - 273.15),
        "battery_max_temp_c": float(battery_max.max() - 273.15),
        "battery_mean_temp_c": float(battery_avg.mean() - 273.15),
        "inter_pack_delta_t_k": float(np.ptp(battery_avg)),
        "cluster_delta_t_k": float(cluster["cluster_delta_temperature_k"]),
        "electrical_power_w": float(result["electrical_power_w"]),
        "electrical_cop": float(result["electrical_cop"]),
        "pack_flows_kg_s": "|".join(f"{value:.5f}" for value in flows),
        "pack_battery_max_c": "|".join(
            f"{value - 273.15:.3f}" for value in battery_max
        ),
    }


def run_case(case: CaseSpec) -> list[dict[str, object]]:
    """Run one case and snapshot it at every checkpoint."""
    plant = build_plant(case.topology, case.current_a, case.direction)
    steps = int(round(max(CHECKPOINT_TIMES_S) / DT_S))
    checkpoint_steps = {
        int(round(time_s / DT_S)): time_s for time_s in CHECKPOINT_TIMES_S
    }
    rows: list[dict[str, object]] = []
    result = None
    for step_index in range(steps):
        result = plant.step(
            dt_s=DT_S,
            cluster_current_a=case.current_a,
            pump_speed_rpm=PUMP_SPEED_RPM,
            compressor_speed_command_rpm=COMPRESSOR_RPM,
            fan_speed_rpm=FAN_SPEED_RPM,
            ambient_temperature_k=AMBIENT_TEMPERATURE_K,
            direction=case.direction,
        )
        completed = step_index + 1
        if completed in checkpoint_steps:
            rows.append(
                _snapshot(case, plant, result, checkpoint_steps[completed])
            )
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = build_case_specs()

    # One Plant step costs ~0.7 s, almost all of it inside the pump operating
    # point brentq, which re-solves the hydraulic network dozens of times per
    # step. Running the cases serially is ~25 minutes of pure waiting, so fan
    # them out across processes. Keep the fan-out modest: each worker carries
    # its own CoolProp state, and eight at once has proven to die part-way
    # through a 1800 s sweep on this machine.
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=min(3, len(specs))) as pool:
        for case_rows in pool.map(run_case, specs):
            rows.extend(case_rows)
            print(f"  completed {case_rows[-1]['case_id']}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "piping_topology_scan.csv", index=False)

    final = frame[frame.time_s == max(CHECKPOINT_TIMES_S)].set_index("case_id")
    show = [
        "flow_spread_pct",
        "battery_max_temp_c",
        "inter_pack_delta_t_k",
        "cluster_delta_t_k",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(f"--- snapshot at {max(CHECKPOINT_TIMES_S):.0f} s ---")
        print(final[show].to_string())

    for current_a in CURRENTS_A:
        tag = "560A" if current_a == 560.0 else "1120A"
        fwd = final.loc[f"E1_{tag}_forward"]
        rev = final.loc[f"E1_{tag}_reverse"]
        direct = final.loc[f"E2_{tag}_direct_return"]
        print(f"\n--- {tag} ---")
        print(
            "  E1 流向  同程正向 vs 同程反向: "
            f"cluster_dT {fwd.cluster_delta_t_k:6.3f} -> {rev.cluster_delta_t_k:6.3f} K "
            f"({rev.cluster_delta_t_k - fwd.cluster_delta_t_k:+.3f} K); "
            f"batt_max {fwd.battery_max_temp_c:6.3f} -> {rev.battery_max_temp_c:6.3f} C "
            f"({rev.battery_max_temp_c - fwd.battery_max_temp_c:+.3f} C)"
        )
        print(
            "  E2 拓扑  同程正向 vs 异程正向: "
            f"flow_spread {fwd.flow_spread_pct:6.3f}% -> {direct.flow_spread_pct:6.3f}% "
            f"({direct.flow_spread_pct / fwd.flow_spread_pct:.2f}x); "
            f"inter_pack_dT {fwd.inter_pack_delta_t_k:6.4f} -> "
            f"{direct.inter_pack_delta_t_k:6.4f} K "
            f"({direct.inter_pack_delta_t_k / fwd.inter_pack_delta_t_k:.2f}x)"
        )

    print(f"\nwrote {OUTPUT_DIR / 'piping_topology_scan.csv'}")
    print(f"rows: {len(frame)}")


if __name__ == "__main__":
    main()
