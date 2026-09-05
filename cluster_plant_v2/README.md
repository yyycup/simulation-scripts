# Cluster Plant V2

This folder owns the independent multi-level five-Pack plant runtime:

`Battery-ROM -> Cold-Plate-ROM -> five-Pack hydraulic cluster -> pump/tank -> R134a cycle -> actuator and transport dynamics`.

Runtime modules do not import the legacy single-Pack project or workspace-root
plant modules. Frozen HPPC data and physical constants needed by the runtime
are stored inside this folder.

The legacy/reference comparison scripts under `validation` are integration
tools by definition. They may load both projects, but they are not imported by
the Cluster runtime.

## Layout and module responsibilities

One file owns one responsibility:

| Path | Responsibility |
|---|---|
| `plant.py` | Authoritative `ClusterPlant` top-level assembly; fixes the per-step advance order (tank -> pump -> hydraulics -> compressor actuator -> R134a cycle -> evaporator dynamics -> supply delay -> cluster -> return delay -> tank) |
| `thermal/pack_rom.py` | Frozen 4-by-3 zoned battery thermal ROM (`ReducedBatteryPack`) |
| `thermal/cold_plate_rom.py` | Three-zone reduction of the thirteen-node cold plate (`ReducedColdPlate`) |
| `thermal/reduced_pack.py` | Energy-conservative coupling of one battery ROM and one cold-plate ROM (`ReducedPack`) |
| `thermal/cluster.py` | Five-Pack cluster organization, branch flow assignment, and return mixing (`ReducedCluster`) |
| `hydraulics.py` | Parallel-header hydraulic network, coolant pump, and coolant tank |
| `refrigeration.py` | Conservative R134a cycle, compressor speed actuator, evaporator thermal dynamics, and supply/return transport delays |
| `pack_reference.py` | Full-order 4P13S reference battery pack used as ROM validation baseline |
| `parameters.py` | Frozen shared defaults (time step, delays, speed bounds, coolant properties) |
| `hppc_parameters.py` | Validated loading of the frozen HPPC resistance/capacity tables |
| `profiles.py` | AGC RegD current-profile loading for validation cases |
| `runtime.py` | Conda environment PATH setup helper |
| `model_data/` | Frozen parameter assets (`hppc_params.json`) |
| `tests/` | Component-level contract tests |
| `validation/` | Stage validation scripts, their contract tests (`validation/tests/`), reusable regression loops (`validation/regression/`), and stored results (`validation/results/`) |

Public entry points are re-exported from `cluster_plant_v2/__init__.py`
(`ClusterPlant`, `ClusterPlantInputs`, `ClusterPlantOutputs`,
`ReferenceBatteryPack`), and the thermal chain from `cluster_plant_v2.thermal`
(`ReducedBatteryPack`, `ReducedColdPlate`, `ReducedPack`, `ReducedCluster`).

## Running the tests

Use the `btms` Conda environment (numpy, scipy, CoolProp):

```powershell
cd <workspace root>   # the folder that contains cluster_plant_v2
& "C:\Users\24776\miniforge3\envs\btms\python.exe" -m unittest discover -s cluster_plant_v2 -p "test_*.py" -t .
```
