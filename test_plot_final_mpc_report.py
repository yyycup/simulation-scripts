import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from plot_final_mpc_report import (
    CASES,
    generate_all_figures,
    load_case,
    load_summary,
    plot_controller_temperature_metrics,
    plot_controller_overview_2x2,
    plot_controller_temperature_timeseries,
    plot_mpc_flow_overview_2x2,
    plot_mpc_flow_temperature,
    plot_mpc_other_metrics,
    validate_mpc_terminal_cost,
)


class FinalMpcReportPlotTests(unittest.TestCase):
    def test_loads_complete_summary_and_required_cases(self):
        summary = load_summary()
        self.assertEqual(len(summary), 12)
        for scene in ("peak", "freq"):
            for control in ("on-off", "pid", "mpc"):
                frame = load_case(scene, control, "single")
                self.assertGreater(len(frame), 500)

    def test_validates_terminal_cost_for_all_mpc_cases(self):
        for scene, flow, expected_weight in (
            ("peak", "single", 1_000_000.0),
            ("peak", "double", 1_000_000.0),
            ("freq", "single", 500_000.0),
            ("freq", "double", 500_000.0),
        ):
            frame = load_case(scene, "mpc", flow)
            validate_mpc_terminal_cost(frame, scene)
            self.assertEqual(float(frame["W_Terminal_Temp"].iloc[0]), expected_weight)

    def test_plot_builders_have_expected_axes_and_content(self):
        summary = load_summary()
        figures = (
            plot_controller_temperature_timeseries(),
            plot_controller_temperature_metrics(summary),
            plot_mpc_flow_temperature(),
            plot_mpc_other_metrics(summary),
        )
        self.assertEqual([len(fig.axes) for fig in figures], [2, 2, 4, 3])
        self.assertIn("时间", figures[0].axes[0].get_xlabel())
        self.assertIn("MAE", figures[1].axes[0].get_ylabel())
        self.assertIn("累计能耗", figures[1].axes[1].get_ylabel())
        self.assertIn("最大温差", figures[2].axes[1].get_ylabel())
        self.assertIn("累计能耗", figures[3].axes[0].get_ylabel())
        self.assertEqual(len(figures[0].axes[0].lines), 3)
        self.assertNotIn("动态目标温度", figures[0].axes[0].get_legend_handles_labels()[1])
        self.assertEqual(len(figures[3].axes[0].patches), len(CASES))

    def test_generate_all_figures_exports_png_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_all_figures(Path(tmp))
            pngs = sorted(Path(tmp).glob("*.png"))
            pdfs = sorted(Path(tmp).glob("*.pdf"))
            self.assertEqual(len(paths), 6)
            self.assertEqual(len(pngs), 6)
            self.assertEqual(len(pdfs), 0)
            self.assertTrue(all(path.stat().st_size > 0 for path in pngs + pdfs))

    def test_freq_flow_panels_do_not_keep_peak_duration_blank_space(self):
        figure = plot_mpc_flow_temperature()
        self.assertLessEqual(figure.axes[2].get_xlim()[1], 65.0)
        self.assertLessEqual(figure.axes[3].get_xlim()[1], 65.0)

    def test_ppt_overview_is_compact_two_by_two(self):
        figure = plot_controller_overview_2x2(load_summary())

        self.assertEqual(len(figure.axes), 4)
        self.assertLessEqual(figure.get_size_inches()[1], 6.0)
        self.assertIn("平均温度", figure.axes[0].get_ylabel())
        self.assertIn("MAE", figure.axes[2].get_ylabel())
        self.assertIn("累计能耗", figure.axes[3].get_ylabel())
        self.assertNotIn("目标温度", figure.axes[0].get_legend_handles_labels()[1])

    def test_flow_overview_is_compact_dual_axis_two_by_two(self):
        figure = plot_mpc_flow_overview_2x2(load_summary())

        self.assertEqual(len(figure.axes), 6)
        self.assertLessEqual(figure.get_size_inches()[1], 6.0)
        self.assertIn("平均温度", figure.axes[0].get_ylabel())
        self.assertIn("最大温差", figure.axes[4].get_ylabel())
        self.assertEqual(len(figure.axes[2].patches), 4)
        self.assertEqual(len(figure.axes[3].patches), 4)


if __name__ == "__main__":
    unittest.main()
