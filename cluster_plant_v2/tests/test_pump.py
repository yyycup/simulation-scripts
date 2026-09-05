import unittest

import numpy as np

from cluster_plant_v2.hydraulics import (
    CoolantPump,
    BASE_BRANCH_RESISTANCE_PA_PER_KG_S2,
    MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO,
    ParallelHeaderHydraulicNetwork,
)


COOLANT_DENSITY_KG_M3 = 1071.0
REFERENCE_SPEED_RPM = 4000.0
REFERENCE_FLOW_L_MIN = 28.0


def medium_header_network() -> ParallelHeaderHydraulicNetwork:
    header_resistance = (
        BASE_BRANCH_RESISTANCE_PA_PER_KG_S2
        * MEDIUM_HEADER_TO_BRANCH_RESISTANCE_RATIO
    )
    return ParallelHeaderHydraulicNetwork(
        n_packs=5,
        supply_segment_resistances=header_resistance,
        return_segment_resistances=header_resistance,
    )


def engineering_pump() -> tuple[CoolantPump, ParallelHeaderHydraulicNetwork]:
    network = medium_header_network()
    reference_mass_flow = (
        REFERENCE_FLOW_L_MIN / 1000.0 / 60.0 * COOLANT_DENSITY_KG_M3
    )
    reference_delta_p = network.solve(reference_mass_flow)["network_delta_p"]
    pump = CoolantPump(
        reference_operating_delta_p_pa=reference_delta_p,
        reference_speed_rpm=REFERENCE_SPEED_RPM,
        reference_volume_flow_l_min=REFERENCE_FLOW_L_MIN,
        coolant_density_kg_m3=COOLANT_DENSITY_KG_M3,
    )
    return pump, network


class CoolantPumpTests(unittest.TestCase):
    def test_reference_curve_matches_audited_legacy_anchor(self) -> None:
        pump, network = engineering_pump()
        reference_mass_flow = pump.reference_mass_flow_kg_s

        self.assertAlmostEqual(
            pump.pressure_rise_pa(reference_mass_flow, REFERENCE_SPEED_RPM),
            network.solve(reference_mass_flow)["network_delta_p"],
            places=8,
        )
        self.assertAlmostEqual(pump.power_w(REFERENCE_SPEED_RPM), 27.3979, places=8)

    def test_curve_and_power_follow_affinity_laws(self) -> None:
        pump, _ = engineering_pump()
        flow = 0.2
        low_speed = 2000.0
        high_speed = 4000.0

        low_shutoff = pump.pressure_rise_pa(0.0, low_speed)
        high_shutoff = pump.pressure_rise_pa(0.0, high_speed)
        self.assertAlmostEqual(high_shutoff / low_shutoff, 4.0, places=12)
        self.assertAlmostEqual(
            pump.power_w(high_speed) / pump.power_w(low_speed), 8.0, places=12
        )
        self.assertTrue(np.isfinite(pump.pressure_rise_pa(flow, high_speed)))

    def test_operating_flow_and_power_increase_with_speed(self) -> None:
        pump, network = engineering_pump()
        speeds = [1600.0, 2400.0, 3200.0, 4000.0, 4800.0]
        points = [pump.solve_operating_point(speed, network) for speed in speeds]

        flows = np.array([point["total_mass_flow_kg_s"] for point in points])
        powers = np.array([point["pump_power_w"] for point in points])
        self.assertTrue(np.all(np.diff(flows) > 0.0))
        self.assertTrue(np.all(np.diff(powers) > 0.0))

    def test_operating_point_closes_pressure_balance(self) -> None:
        pump, network = engineering_pump()
        point = pump.solve_operating_point(3600.0, network)

        self.assertTrue(point["solver_success"])
        self.assertLess(abs(point["pressure_balance_residual_pa"]), 1e-6)
        self.assertAlmostEqual(
            point["pump_delta_p_pa"], point["network_delta_p_pa"], places=6
        )
        self.assertGreater(point["total_mass_flow_kg_s"], 0.0)
        self.assertGreater(point["total_volume_flow_l_min"], 0.0)

    def test_missing_operating_point_raises_explicit_error(self) -> None:
        pump, _ = engineering_pump()

        class ExcessiveStaticHeadNetwork:
            @staticmethod
            def solve(total_mass_flow_kg_s: float) -> dict[str, float]:
                del total_mass_flow_kg_s
                return {"network_delta_p": 1.0e9}

        with self.assertRaisesRegex(RuntimeError, "no pump-network operating point"):
            pump.solve_operating_point(1600.0, ExcessiveStaticHeadNetwork())


if __name__ == "__main__":
    unittest.main()
