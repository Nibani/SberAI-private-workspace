"""The committed v1.2 results are reproducible from repository files.

These checks rerun the cheap, deterministic parts of ``scripts/run_v12.py`` (the
final typology, monthly tracking and the chosen edge rule) and verify that the
Markdown tables are generated from the current results. Bootstrap tables are not
recomputed here; ``python -m scripts.run_v12`` regenerates them.
"""
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from sbercluster import edges as E
from sbercluster import panel as P
from sbercluster import tracking as T
from sbercluster import typology as TY
from sbercluster.graph import knn_graph
from sbercluster.joint import fit_graph_regularized_kmeans

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/v1.2"


class PublishedV12Results(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        cls.config = json.loads((ROOT / "configs/v12.json").read_text(encoding="utf-8"))
        cls.labels = pd.read_csv(RESULTS / "labels.csv")
        cls.panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")
        cls.scaler = P.fit_scaler(cls.panel)
        cls.monthly = P.monthly_attributes(cls.panel, cls.scaler)
        cls.x23 = P.annual_profile(cls.monthly, slice(0, 12))

    def test_inputs_match_recorded_hashes(self):
        manifest = json.loads((ROOT / "data/v12/panel.manifest.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256((ROOT / "data/v12/panel.csv.gz").read_bytes()).hexdigest()
        self.assertEqual(digest, manifest["sha256"])
        for name, item in self.summary["inputs"].items():
            actual = hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, item["sha256"], name)

    def test_final_typology_is_reproduced(self):
        chosen = self.summary["methods"]["chosen"]
        self.assertTrue(chosen.startswith("joint_"))
        start = KMeans(4, n_init=50, random_state=self.config["seed"]).fit_predict(self.x23)
        weights = knn_graph(self.x23, k=self.config["network"]["k"])[0]
        raw, _ = fit_graph_regularized_kmeans(self.x23, weights, start, alpha=float(chosen.split("_")[1]),
                                              seed=self.config["seed"], max_sweeps=200)
        naming = TY.name_types(np.median(self.panel.totals[:, :12], axis=1),
                               np.median(self.panel.shares[:, :12], axis=1), raw)
        types = TY.canonical_labels(raw, naming)
        self.assertEqual(list(self.labels["entity_id"]), list(self.panel.ids))
        np.testing.assert_array_equal(types, self.labels["type_2023"].to_numpy())
        self.assertEqual(np.bincount(types).tolist(), self.summary["types"]["sizes"])

    def test_monthly_tracking_is_reproduced(self):
        model = np.load(RESULTS / "model.npz")
        adjusted = P.remove_national_wave(self.monthly, self.x23)
        monthly = T.assign_monthly(adjusted, model["centers"])
        self.assertEqual(["".join(map(str, row)) for row in monthly], list(self.labels["monthly_types"].astype(str).str.zfill(24)))
        year = T.assign_monthly(P.annual_profile(adjusted, slice(12, 24))[:, None, :], model["centers"])[:, 0]
        np.testing.assert_array_equal(year, self.labels["type_2024"].to_numpy())

    def test_chosen_edge_rule_row_is_reproduced(self):
        table = pd.read_csv(RESULTS / "networks.csv").set_index("rule")
        rule = self.summary["network"]["chosen_rule"]
        self.assertEqual(rule, "euclid_profile")
        k = self.config["network"]["k"]
        x24 = P.annual_profile(self.monthly, slice(12, 24))
        a23 = E.knn_union(-E.pairwise_euclidean(self.x23), k)
        a24 = E.knn_union(-E.pairwise_euclidean(x24), k)
        cohort = pd.read_csv(ROOT / self.config["inputs"]["rosstat_cohort"]).set_index("entity_id").reindex(self.panel.ids)
        external = pd.DataFrame({"log_wage": cohort["log_annual_monthly_wage_rubles"]}, index=self.panel.ids)
        row = E.evaluate_graph(a23, external, self.panel.regions, later=a24)
        self.assertEqual(row["edges"], table.loc[rule, "edges"])
        self.assertAlmostEqual(row["stability_jaccard"], table.loc[rule, "stability_jaccard"], places=5)
        self.assertAlmostEqual(row["assort_log_wage"], table.loc[rule, "assort_log_wage"], places=5)

    def test_documentation_tables_are_current(self):
        from scripts.build_v12_doc_tables import DOCUMENTS, Tables, render
        tables = Tables(RESULTS)
        for name in DOCUMENTS:
            path = ROOT / name
            if path.exists():
                text = path.read_text(encoding="utf-8")
                self.assertEqual(render(text, tables), text, f"stale v1.2 numbers in {name}")


if __name__ == "__main__":
    unittest.main()
