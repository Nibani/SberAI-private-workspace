import argparse
import base64
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path

from scripts import build_research_atlas as atlas

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/panel.csv"
GRAPH = ROOT / "reports/graph_preparation.json"
PARTITIONS = ROOT / "reports/experiments/2026-09-23/partitions.csv"
METRICS = ROOT / "reports/experiments/2026-09-23/metrics.json"
TEMPLATE = ROOT / "web/research_template.html"
HAS_PRIVATE_DATA = all(path.exists() for path in (PANEL, GRAPH, PARTITIONS, METRICS))


class ResearchAtlasUnitTests(unittest.TestCase):
    def test_script_json_escaping_and_nonfinite_rejection(self):
        encoded = atlas.encode_payload({"text": "</script>&\u2028"})
        self.assertNotIn("</script>", encoded)
        self.assertIn("\\u003c/script\\u003e\\u0026\\u2028", encoded)
        with self.assertRaises(ValueError):
            atlas.encode_payload({"bad": math.nan})

    def test_compressed_payload_is_deterministic_lossless_and_safe(self):
        value = {"name": "</script>\u2028", "values": [1, 2.25, None], "nested": {"id": "tid_1"}}
        encoded = atlas.package_payload(value)
        self.assertEqual(encoded, atlas.package_payload(value))
        self.assertNotIn("</script>", encoded)
        envelope = json.loads(encoded)
        self.assertEqual(json.loads(gzip.decompress(base64.b64decode(envelope["data"]))), value)
        with self.assertRaises(ValueError):
            atlas.package_payload({"invalid": math.inf})

    def test_stability_contract_accepts_scientific_ari_range(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stability.json"
            path.write_text('[{"candidate":"kmeans_k4","kind":"seed","parameter":2718,"ARI":-0.2,"k":4}]', encoding="utf-8")
            self.assertEqual(atlas.read_stability(path)[0]["ARI"], -0.2)
            path.write_text('[{"candidate":"kmeans_k4","kind":"seed","parameter":2718,"ARI":1.2,"k":4}]', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                atlas.read_stability(path)

    def test_public_template_states_neighbor_recalculation_scope(self):
        text = TEMPLATE.read_text(encoding="utf-8")
        self.assertIn("Кластеры не переобучаются", text)
        self.assertIn("не означает структурный перелом", text)
        self.assertNotIn("по поручению", text.lower())
        self.assertNotIn("искусственн", text.lower())


@unittest.skipUnless(HAS_PRIVATE_DATA, "private processed data are not present in a clean clone")
class ResearchAtlasIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = atlas.read_panel(PANEL, GRAPH)

    def test_panel_contract_preserves_real_complete_cohort(self):
        panel = self.panel
        self.assertEqual(len(panel["entities"]), 2016)
        self.assertEqual(len(panel["periods"]), 24)
        self.assertEqual(panel["periods"], sorted(panel["periods"]))
        self.assertTrue(all(len(entity["totals"]) == 24 for entity in panel["entities"]))
        self.assertTrue(all(len(entity["annual_features"]) == 5 for entity in panel["entities"]))
        self.assertTrue(all(value > 0 for entity in panel["entities"] for value in entity["totals"]))

    def test_pilot_is_exact_december_kmeans_join(self):
        ids = {entity["id"] for entity in self.panel["entities"]}
        pilot = atlas.read_pilot(PARTITIONS, METRICS, ids)
        self.assertEqual(pilot["period"], "2023-12-01")
        self.assertEqual(len(pilot["labels"]), 2016)
        self.assertEqual(sum(pilot["cluster_sizes"].values()), 2016)
        self.assertEqual(pilot["metric"]["k"], 8)
        self.assertAlmostEqual(pilot["metric"]["SW"], 0.21260261826094157)

    def test_incomplete_validation_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "frozen_prototypes.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                atlas.read_validation(path, {e["id"] for e in self.panel["entities"]}, set(self.panel["periods"]))

    def test_build_is_self_contained_and_uses_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            args = argparse.Namespace(panel=PANEL, graph_report=GRAPH, partitions=PARTITIONS, metrics=METRICS,
                                      reference_run=None, validation_run=None, stability_run=None,
                                      template=TEMPLATE, output=output, uncompressed=True)
            atlas.build(args)
            rendered = output.read_text(encoding="utf-8")
        self.assertNotIn("/*__ATLAS_DATA__*/null", rendered)
        self.assertIn('"entities":2016', rendered)
        self.assertIn('"status":"pilot"', rendered)
        self.assertNotIn(":Infinity", rendered)
        self.assertIn("https://creativecommons.org/licenses/by-sa/4.0", rendered)
        self.assertIn("https://www.openstreetmap.org/copyright", rendered)


if __name__ == "__main__":
    unittest.main()