"""Stage 8C1 compressor-actuator coupling for the Stage 8B plant."""

from __future__ import annotations

from cluster_plant_v2.refrigeration import CompressorSpeedActuator
from cluster_plant_v2.validation.regression.refrigerated_cluster_cooling_loop import (
    RefrigeratedClusterCoolingLoop,
)


class DynamicRefrigeratedClusterCoolingLoop:
    """Advance compressor speed, then solve the frozen Stage 8B loop."""

    def __init__(
        self,
        *,
        cooling_loop: RefrigeratedClusterCoolingLoop,
        compressor_actuator: CompressorSpeedActuator,
    ) -> None:
        self.cooling_loop = cooling_loop
        self.compressor_actuator = compressor_actuator
        self.dynamic_state_count = (
            cooling_loop.dynamic_state_count
            + compressor_actuator.dynamic_state_count
        )

    def step(
        self,
        *,
        dt_s: float,
        cluster_current_a: float,
        pump_speed_rpm: float,
        compressor_speed_command_rpm: float,
        fan_speed_rpm: float,
        ambient_temperature_k: float,
        direction: str = "forward",
    ) -> dict[str, object]:
        actuator_result = self.compressor_actuator.step(
            dt_s=dt_s,
            speed_command_rpm=compressor_speed_command_rpm,
        )
        result = self.cooling_loop.step(
            dt_s=dt_s,
            cluster_current_a=cluster_current_a,
            pump_speed_rpm=pump_speed_rpm,
            compressor_speed_rpm=actuator_result["speed_after_rpm"],
            fan_speed_rpm=fan_speed_rpm,
            ambient_temperature_k=ambient_temperature_k,
            direction=direction,
        )
        return {
            **result,
            "compressor_speed_command_rpm": actuator_result[
                "speed_command_rpm"
            ],
            "compressor_speed_before_rpm": actuator_result[
                "speed_before_rpm"
            ],
        }
