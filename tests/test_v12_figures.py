import json
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402

from scripts.build_v12_figures import (CommaFormatter, fmt_num, fmt_pct, mid_sentence, outline_path, parse_svg_path,
                                       plural)

ROOT = Path(__file__).resolve().parents[1]
MAP = ROOT / "reports/contest-v3/municipal_map.json"


class SvgPathTests(unittest.TestCase):
    def test_rings_are_closed(self):
        rings = parse_svg_path("M0,0L10,0L10,10L0,10Z M4,4L6,4L6,6L4,6")
        self.assertEqual(len(rings), 2)
        for ring in rings:
            self.assertEqual(ring.shape[1], 2)
            np.testing.assert_array_equal(ring[0], ring[-1])
        self.assertEqual(len(rings[1]), 5)  # the open subpath was closed

    def test_unsupported_commands_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_svg_path("M0,0 C1,1 2,2 3,3Z")

    def test_hole_is_left_empty(self):
        outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
        hole = [(4, 4), (6, 4), (6, 6), (4, 6)]  # same winding as the outer ring in the SVG
        d = "".join("M" + "L".join(f"{x},{y}" for x, y in ring) + "Z" for ring in (outer, hole))
        path = outline_path(parse_svg_path(d), height=10)
        fig = plt.figure(figsize=(1, 1), dpi=50)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.add_patch(PathPatch(path, facecolor="black", edgecolor="none"))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        fig.canvas.draw()
        red = np.asarray(fig.canvas.buffer_rgba())[:, :, 0]
        plt.close(fig)
        self.assertEqual(red[5, 5], 0)      # inside the outer ring: filled
        self.assertEqual(red[25, 25], 255)  # inside the hole: empty

    @unittest.skipUnless(MAP.exists(), "published map not available")
    def test_published_map_paths_are_closed(self):
        geometry = json.loads(MAP.read_text(encoding="utf-8"))
        for item in geometry["paths"][:50]:
            rings = parse_svg_path(item["d"])
            self.assertTrue(rings, item["id"])
            for ring in rings:
                np.testing.assert_array_equal(ring[0], ring[-1])


class FormattingTests(unittest.TestCase):
    def test_decimal_comma_and_signs(self):
        self.assertEqual(fmt_num(0.2345, 2), "0,23")
        self.assertEqual(fmt_num(-1.5, 1), "−1,5")
        self.assertEqual(fmt_num(3.1, 1, signed=True), "+3,1")
        self.assertEqual(fmt_num(-0.0001, 2), "0,00")
        self.assertEqual(fmt_num(2016, 0), "2 016")
        self.assertEqual(fmt_pct(0.481), "48%")
        self.assertNotIn(".", fmt_num(1234.5678, 3))

    def test_axis_formatter_uses_comma(self):
        formatter = CommaFormatter()
        formatter.set_locs([0.0, 0.25, 0.5])
        self.assertEqual(formatter(0.25), "0,25")
        self.assertEqual(formatter(0.0), "0,00")

    def test_russian_words(self):
        forms = ("регион", "региона", "регионов")
        self.assertEqual([plural(n, forms) for n in (1, 3, 5, 11, 21, 73)],
                         ["регион", "региона", "регионов", "регионов", "регион", "региона"])
        self.assertEqual(mid_sentence("Крупные города"), "крупные города")
        self.assertEqual(mid_sentence("Дальний Восток и Север"), "Дальний Восток и Север")
        self.assertEqual(mid_sentence("Гауссова смесь"), "гауссова смесь")
        self.assertEqual(mid_sentence("Leiden (граф)"), "Leiden (граф)")


if __name__ == "__main__":
    unittest.main()
