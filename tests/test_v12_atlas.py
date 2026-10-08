"""The published atlas carries the v1.2 typology: map layers, per-territory fields, sections and a reproducible bundle."""
import copy
import csv
import hashlib
import re
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from sbercluster import panel as P
from sbercluster.typology import TYPE_COLORS, TYPE_NAMES, TYPE_ORDER
from scripts import build_v12_web
from scripts.atlas_payload import read_atlas_payload
from scripts.build_research_atlas import encode_payload, render_atlas
from scripts.extend_atlas_v12 import ENTITY_FIELDS, attach_v12, build, network_neighbors

ROOT = Path(__file__).resolve().parents[1]
V12 = ROOT / "reports/v1.2"
ATLAS = ROOT / "docs/index.html"
TEMPLATE = ROOT / "web/research_template.html"
SECTIONS = ROOT / "web/v12_sections.html"
SECTION_IDS = ("v12-types-title", "v12-network-title", "v12-methods-title", "v12-dynamics-title", "v12-practice-title")


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def payload_sources(html: str) -> list[str]:
    return re.findall(r'<script src="(assets/payload-[0-9a-f]{64}\.js)"></script>', html)


def without_v12(payload: dict) -> dict:
    """The payload as build_research_atlas.py produces it before the --v12 hook."""
    stripped = copy.deepcopy(payload)
    for entity in stripped["entities"]:
        for field in ENTITY_FIELDS:
            entity.pop(field, None)
    contest = stripped["contest"]
    contest["map_models"] = {k: v for k, v in contest["map_models"].items() if not k.startswith("v12_")}
    contest.pop("default_map_model", None)
    contest.pop("v12", None)
    contest["source_hashes"] = {k: v for k, v in contest.get("source_hashes", {}).items()
                                if not k.startswith(("reports/v1.2/", "data/v12/"))}
    return stripped


def section(html: str, anchor: str) -> str:
    start = html.rindex("<section", 0, html.index(f'id="{anchor}"'))
    return html[start:html.index("</section>", start)]


@unittest.skipUnless((V12 / "provenance.json").is_file(), "complete v1.2 results are not present")
class V12AtlasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.html = ATLAS.read_text(encoding="utf-8")
        cls.payload = read_atlas_payload(ATLAS)
        cls.ids = [entity["id"] for entity in cls.payload["entities"]]

    def test_attach_is_idempotent_and_replaces_previous_layers(self):
        first = attach_v12(copy.deepcopy(self.payload), V12)
        stale = copy.deepcopy(first)
        stale["contest"]["map_models"]["v12_obsolete"] = stale["contest"]["map_models"]["v12_types"]
        stale["entities"][0]["v12_neighbors"] = []
        second = attach_v12(stale, V12)
        self.assertNotIn("v12_obsolete", second["contest"]["map_models"])
        self.assertEqual(encode_payload(first), encode_payload(second))
        self.assertEqual(encode_payload(first), encode_payload(self.payload))
        self.assertEqual(list(first["contest"]["map_models"])[-2:], ["v12_types", "v12_types_2024"])

    def test_layers_cover_every_territory_with_the_four_types(self):
        contest = self.payload["contest"]
        self.assertEqual(contest.get("default_map_model"), "v12_types")
        colors = [TYPE_COLORS[key] for key in TYPE_ORDER]
        names = [TYPE_NAMES[key] for key in TYPE_ORDER]
        for key in ("v12_types", "v12_types_2024"):
            with self.subTest(layer=key):
                layer = contest["map_models"][key]
                self.assertEqual(list(layer["labels"]), self.ids)
                self.assertEqual(len(layer["labels"]), 2016)
                self.assertEqual(set(layer["labels"].values()), {0, 1, 2, 3})
                self.assertEqual([p["name"] for p in layer["profiles"]], names)
                self.assertEqual([p["id"] for p in layer["profiles"]], [0, 1, 2, 3])
                self.assertEqual(sum(p["n"] for p in layer["profiles"]), len(self.ids))
                self.assertEqual(layer["colors"], colors)
        main = contest["map_models"]["v12_types"]
        self.assertEqual(main["name"], "Структура и уровень расходов · 6 признаков")
        self.assertEqual(main["labels_2024"]["relative"], contest["map_models"]["v12_types_2024"]["labels"])
        for mode in ("raw", "relative"):
            after = main["labels_2024"][mode]
            self.assertEqual(set(after), set(self.ids))
            self.assertEqual(main["dynamics"][mode]["changed"], sum(after[i] != main["labels"][i] for i in self.ids))
        self.assertEqual([t["color"] for t in contest["v12"]["types"]], colors)
        self.assertEqual([t["name"] for t in contest["v12"]["types"]], names)

    def test_entities_carry_per_territory_v12_fields(self):
        contest = self.payload["contest"]
        layers = contest["map_models"]
        k = contest["v12"]["neighbors"]
        ids = set(self.ids)
        for entity in self.payload["entities"]:
            self.assertEqual(entity["v12_type"], layers["v12_types"]["labels"][entity["id"]])
            self.assertEqual(entity["v12_type_2024"], layers["v12_types_2024"]["labels"][entity["id"]])
            self.assertEqual(len(entity["v12_monthly"]), len(self.payload["periods"]))
            self.assertTrue(set(entity["v12_monthly"]) <= {0, 1, 2, 3})
            self.assertEqual(len(entity["v12_features"]), len(contest["v12"]["attributes"]))
            neighbours = entity["v12_neighbors"]
            self.assertEqual(len(neighbours), k)
            self.assertEqual(len(set(neighbours)), k)
            self.assertNotIn(entity["id"], neighbours)
            self.assertTrue(set(neighbours) <= ids)
        monthly = {row["entity_id"]: row["monthly_types"] for row in read_rows(V12 / "labels.csv")}
        self.assertTrue(all("".join(map(str, e["v12_monthly"])) == monthly[e["id"]] for e in self.payload["entities"]))

    def test_network_neighbours_reproduce_the_v12_analogue_check(self):
        # Median 2024/2023 growth of the stored neighbours gives the published network-analogue error.
        panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")
        row = {entity_id: i for i, entity_id in enumerate(panel.ids)}
        growth = panel.totals[:, 12:].mean(axis=1) / panel.totals[:, :12].mean(axis=1) - 1
        by_id = {e["id"]: e for e in self.payload["entities"]}
        prediction = np.array([np.median(growth[[row[j] for j in by_id[i]["v12_neighbors"]]]) for i in panel.ids])
        common = np.ones(len(panel.ids), dtype=bool)  # primary controls cover the full cohort
        published = {r["predictor"]: r for r in read_rows(V12 / "analogues.csv")}
        self.assertEqual(int(common.sum()), int(published["network_analogues"]["n"]))
        self.assertAlmostEqual(float(np.mean(100 * np.abs(prediction - growth)[common])),
                               float(published["network_analogues"]["mae_pp"]), places=5)

    def test_default_territory_is_the_largest_of_the_remote_type(self):
        types = {int(r["type"]): r for r in read_rows(V12 / "types.csv")}
        remote = TYPE_ORDER.index("remote")
        largest = types[remote]["largest_by_population"].split(";")[0].strip()
        entity = next(e for e in self.payload["entities"] if e["id"] == self.payload["contest"]["v12"]["default_entity"])
        self.assertEqual(f"{entity['name']} ({entity['region']})", largest)
        self.assertEqual(entity["v12_type"], remote)

    def test_page_references_one_existing_hash_named_asset(self):
        sources = payload_sources(self.html)
        self.assertEqual(len(sources), 1)
        asset = ROOT / "docs" / sources[0]
        self.assertTrue(asset.is_file())
        self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), asset.stem.removeprefix("payload-"))
        published = sorted(p.name for p in (ROOT / "docs/assets").glob("payload-*.js"))
        self.assertIn(asset.name, published)

    def test_sections_are_restored_losslessly_before_initialisation(self):
        template = TEMPLATE.read_text(encoding="utf-8")
        self.assertEqual(template.count("<!-- V12_SECTIONS -->"), 1)
        self.assertNotIn("<!-- V12_SECTIONS -->", self.html)
        self.assertIn('<div id="v12-content"></div>', self.html)
        restored = self.payload["contest"]["v12"]["ui_fragment"]
        self.assertEqual(restored, SECTIONS.read_text(encoding="utf-8"))
        for anchor in SECTION_IDS:
            self.assertEqual(restored.count(f'id="{anchor}"'), 1)
            self.assertIn(f'href="#{anchor}"', template)
        self.assertLess(template.index('.innerHTML=contest.v12.ui_fragment'), template.index('setupSixFeatures();'))
        self.assertLess(len(self.html.encode("utf-8")), 500_000)

    def test_main_interface_and_existing_layers_are_preserved(self):
        self.assertIn('<h1>Экономические соседи</h1>', self.html)
        stripped = without_v12(self.payload)
        prior_models = copy.deepcopy(stripped["contest"]["map_models"])
        prior_geometry = copy.deepcopy(stripped["contest"]["map"])
        attached = attach_v12(stripped, V12)
        self.assertEqual({k: v for k, v in attached["contest"]["map_models"].items()
                          if not k.startswith("v12_")}, prior_models)
        self.assertEqual(attached["contest"]["map"], prior_geometry)
        for anchor in ("neighbors-title", "overview-title", "accepted-title", "ego-title", "time-title"):
            self.assertEqual(self.html.count(f'id="{anchor}"'), 1)

    def test_optional_builder_keeps_old_payload_usable(self):
        stripped = without_v12(self.payload)
        html, assets = render_atlas(TEMPLATE.read_text(encoding="utf-8"), stripped, split_assets=True)
        self.assertNotIn('<div id="v12-content"></div>', html)
        self.assertNotIn('<!-- V12_SECTIONS -->', html)
        self.assertEqual(len(assets), 1)
        self.assertIn('<h1>Экономические соседи</h1>', html)

    def test_sections_file_is_generated_from_current_results(self):
        self.assertEqual(SECTIONS.read_text(encoding="utf-8"), build_v12_web.render_sections(V12))

    def test_sections_are_static_and_use_current_type_names(self):
        sections = SECTIONS.read_text(encoding="utf-8")
        for forbidden in ("<script", "<link", "@import", "url(", " src=", "nan", "None", "Infinity"):
            self.assertNotIn(forbidden, sections)
        self.assertNotIn("ресурсный север", sections.lower())
        self.assertNotIn("логарифм × 100", sections)
        self.assertNotIn("2024 · вне выборки", sections)
        self.assertIn("2024 участвовали в выборе", sections)
        self.assertIn("Вложенный выбор параметров не выполнен", sections)
        for key in TYPE_ORDER:
            self.assertIn(TYPE_NAMES[key], sections)
            self.assertIn(TYPE_COLORS[key], sections)
        self.assertIn("2\N{NARROW NO-BREAK SPACE}016", sections)

    def test_extend_script_reproduces_published_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "index.html"
            result = build(ATLAS, V12, TEMPLATE, output)
            self.assertEqual(output.read_bytes(), ATLAS.read_bytes())
            for name in result["assets"]:
                self.assertEqual((output.parent / "assets" / name).read_bytes(), (ROOT / "docs/assets" / name).read_bytes())

    def test_builder_hook_reproduces_published_bundle(self):
        # build_research_atlas.py --v12 attaches the v1.2 data to a payload that has none yet.
        attached = attach_v12(without_v12(self.payload), V12)
        html, assets = render_atlas(TEMPLATE.read_text(encoding="utf-8"), attached, split_assets=True)
        self.assertEqual(html.encode("utf-8"), ATLAS.read_bytes())
        for name, content in assets.items():
            self.assertEqual((ROOT / "docs/assets" / name).read_bytes(), content)


class V12FormattingTests(unittest.TestCase):
    def test_final_float_bits_do_not_change_published_bytes(self):
        import json
        summary = json.loads((V12 / 'summary.json').read_text('utf-8'))
        config = json.loads((V12 / 'provenance.json').read_text('utf-8'))['config']
        ids, profile, neighbors = network_neighbors(summary, config)
        payload = read_atlas_payload(ATLAS)
        reference = encode_payload(attach_v12(copy.deepcopy(payload), V12))
        for direction in (-np.inf, np.inf):
            with patch('scripts.extend_atlas_v12.network_neighbors',
                       return_value=(ids, np.nextafter(profile, direction), neighbors)):
                self.assertEqual(encode_payload(attach_v12(copy.deepcopy(payload), V12)), reference)

    def test_section_links_reach_existing_methodology(self):
        sections = build_v12_web.render_sections(V12)
        paths = re.findall(r'href="https://github.com/Nibani/SberAI/blob/main/docs/([^"#]+)(?:#[^"]*)?"', sections)
        self.assertEqual(len(paths), 5)
        for path in paths:
            self.assertTrue((ROOT / 'docs' / path).is_file(), path)
        self.assertIn('NETWORK_TYPOLOGY.md#девять-правил-рёбер', sections)

    def test_russian_number_format(self):
        fnum = build_v12_web.fnum
        self.assertEqual(fnum(-1234.56, 1), "\N{MINUS SIGN}1\N{NARROW NO-BREAK SPACE}234,6")
        self.assertEqual(fnum(-0.0001, 2), "0,00")
        self.assertEqual(fnum(5, sign=True), "+5")
        self.assertEqual(fnum(None), "нет")
        self.assertEqual(build_v12_web.fpct(0.1676), "16,8%")
        self.assertEqual(build_v12_web.count(21, ("правило", "правила", "правил")), "21\N{NO-BREAK SPACE}правило")

    def test_short_names_match_the_template_rule(self):
        short = build_v12_web.short_name
        self.assertEqual(short("городской округ город Черногорск"), "Черногорск")
        self.assertEqual(short("Ельнинский муниципальный район"), "Ельнинский р-н")
        self.assertEqual(short("внутригородская территория города федерального значения муниципальный округ Чертаново Центральное"),
                         "Чертаново Центральное")


if __name__ == "__main__":
    unittest.main()
