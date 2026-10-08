import argparse
import base64
import copy
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import build_research_atlas as atlas


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = '''<!doctype html><html><head><style>
/*__ATLAS_FONTS__*/
/*__ATLAS_THEME__*/
</style></head><body><script>
(async()=>{const packed=/*__ATLAS_DATA__*/null;globalThis.result=packed;})()
.catch(error=>{globalThis.failure=error.message;});
</script></body></html>'''
VALUE = {"entities": [{"id": "tid_1", "name": "</script>&\u2028"}],
         "contest": {"map": {"paths": [{"id": "tid_1", "d": "M0,0L1.25,2.5Z"}]}}}


def decode(envelope):
    if envelope.get("encoding") == "gzip-base64":
        return json.loads(gzip.decompress(base64.b64decode(envelope["data"])))
    return envelope


def standalone_payload(html):
    return json.loads(re.search(r"const packed=(.*?);", html).group(1))


def asset_payload(script):
    prefix = '"use strict";\nwindow.__SBERAI_ATLAS__={payload:'
    if not script.startswith(prefix) or not script.endswith('};\n'):
        raise AssertionError("Unexpected classic data script")
    return json.loads(script[len(prefix):-3])


class AtlasBundleTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        web = self.root / "web"
        web.mkdir()
        for filename, content in (("atlas_fonts.css", "/* offline font */"),
                                  ("atlas_theme.css", "/* theme */"),
                                  ("atlas_map.css", "/* map */")):
            (web / filename).write_text(content, encoding="utf-8")
        self.patch_root = patch.object(atlas, "ROOT", self.root)
        self.patch_root.start()
        self.addCleanup(self.patch_root.stop)

    def test_cli_requires_explicit_opt_in_for_split_assets(self):
        with patch("sys.argv", ["build_research_atlas.py"]):
            self.assertFalse(atlas.parse_args().split_assets)
        with patch("sys.argv", ["build_research_atlas.py", "--split-assets"]):
            self.assertTrue(atlas.parse_args().split_assets)

    def test_default_remains_standalone_and_both_modes_preserve_all_values(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                standalone, embedded_assets = atlas.render_atlas(TEMPLATE, VALUE, compressed=compressed)
                split, assets = atlas.render_atlas(TEMPLATE, VALUE, compressed=compressed, split_assets=True)
                self.assertEqual(embedded_assets, {})
                self.assertNotIn('<script src=', standalone)
                self.assertEqual(decode(standalone_payload(standalone)), VALUE)
                self.assertEqual(len(assets), 1)
                filename, content = next(iter(assets.items()))
                self.assertEqual(decode(asset_payload(content.decode("utf-8"))), VALUE)
                self.assertEqual(filename, "payload-" + hashlib.sha256(content).hexdigest() + ".js")
                self.assertIn(f'<script src="assets/{filename}"></script>\n<script>', split)
                self.assertLess(split.index('<script src='), split.index('const packed='))
                self.assertNotIn('type="module"', split)
                self.assertNotIn('fetch(', split)
                self.assertNotIn('eval(', split)
                self.assertLess(split.index('/* theme */'), split.index('/* map */'))

    def test_bundle_is_byte_reproducible_and_version_names_change_with_data(self):
        first = atlas.render_atlas(TEMPLATE, VALUE, split_assets=True)
        self.assertEqual(first, atlas.render_atlas(TEMPLATE, VALUE, split_assets=True))
        changed = atlas.render_atlas(TEMPLATE, {**VALUE, "version": 2}, split_assets=True)
        self.assertNotEqual(set(first[1]), set(changed[1]))

    def test_invalid_template_and_nonfinite_payload_do_not_render(self):
        for template in (TEMPLATE.replace('/*__ATLAS_DATA__*/null', 'null'),
                         TEMPLATE + '/*__ATLAS_DATA__*/null', '/*__ATLAS_DATA__*/null'):
            with self.subTest(template=template), self.assertRaises(ValueError):
                atlas.render_atlas(template, VALUE, split_assets=True)
        with self.assertRaises(ValueError):
            atlas.render_atlas(TEMPLATE, {"invalid": float("nan")}, split_assets=True)

    def test_assets_commit_before_html_and_failures_preserve_previous_version(self):
        output = self.root / "bundle/index.html"
        old_html, old_assets = atlas.render_atlas(TEMPLATE, VALUE, split_assets=True)
        atlas.write_atlas(output, old_html, old_assets)
        new_html, new_assets = atlas.render_atlas(TEMPLATE, {**VALUE, "version": 2}, split_assets=True)
        original_replace = atlas.os.replace
        replaced = []

        def fail_html(source, destination):
            replaced.append(Path(destination))
            if Path(destination) == output:
                raise OSError("simulated HTML commit failure")
            original_replace(source, destination)

        with patch.object(atlas.os, "replace", side_effect=fail_html), self.assertRaises(OSError):
            atlas.write_atlas(output, new_html, new_assets)
        self.assertEqual(output.read_text(encoding="utf-8"), old_html)
        self.assertEqual(replaced[-1], output)
        for filename, content in {**old_assets, **new_assets}.items():
            self.assertEqual((output.parent / "assets" / filename).read_bytes(), content)
        self.assertEqual(list(output.parent.rglob('.atlas-*')), [])

    def test_asset_failure_does_not_replace_html_and_existing_asset_is_repaired(self):
        output = self.root / "bundle/index.html"
        html, assets = atlas.render_atlas(TEMPLATE, VALUE, split_assets=True)
        atlas.write_atlas(output, html, assets)
        filename, content = next(iter(assets.items()))
        destination = output.parent / "assets" / filename
        destination.write_bytes(b"corrupt")
        with patch.object(atlas.os, "replace", side_effect=OSError("simulated asset failure")):
            with self.assertRaises(OSError):
                atlas.write_atlas(output, "new HTML", assets)
        self.assertEqual(output.read_text(encoding="utf-8"), html)
        atlas.write_atlas(output, html, assets)
        self.assertEqual(destination.read_bytes(), content)
        self.assertEqual(list(output.parent.rglob('.atlas-*')), [])

    def test_classic_script_executes_and_missing_or_corrupt_asset_reports_error(self):
        node = shutil.which("node")
        self.assertIsNotNone(node, "Node.js is required to verify the classic data script")
        html, assets = atlas.render_atlas(TEMPLATE, VALUE, split_assets=True)
        inline = re.findall(r'<script>(.*?)</script>', html, re.S)[0]
        data = next(iter(assets.values())).decode("utf-8")
        for script, success in ((data, True), ("", False),
                                ('window.__SBERAI_ATLAS__={wrong:true};', False)):
            with self.subTest(success=success, script=script[:20]):
                code = ('globalThis.window=globalThis;\n' + script + '\n' + inline + '\n'
                        'setImmediate(()=>console.log(JSON.stringify({result:globalThis.result,'
                        'failure:globalThis.failure})));')
                result = subprocess.run([node, "-e", code], capture_output=True,
                                        encoding="utf-8", check=True)
                observed = json.loads(result.stdout)
                if success:
                    self.assertEqual(decode(observed["result"]), VALUE)
                else:
                    self.assertIn("Не найден файл данных атласа", observed["failure"])
                    self.assertIn("assets", observed["failure"])


    def test_published_geometry_is_lossless_in_both_modes_without_private_panel(self):
        geometry = json.loads((ROOT / "reports/contest-v3/municipal_map.json").read_text(encoding="utf-8"))
        payload = {"contest": {"map": geometry}}
        standalone_html, _ = atlas.render_atlas(TEMPLATE, payload)
        split_html, assets = atlas.render_atlas(TEMPLATE, payload, split_assets=True)
        standalone = decode(standalone_payload(standalone_html))
        split = decode(asset_payload(next(iter(assets.values())).decode("utf-8")))
        self.assertEqual(standalone, payload)
        self.assertEqual(split, payload)
        self.assertEqual(len(split["contest"]["map"]["paths"]), 2016)
        self.assertLess(len(split_html.encode("utf-8")), 500_000)


    def test_dual_output_rejects_nonfinite_data_before_replacing_either_html(self):
        panel = {'entities': [{'id': 'one'}], 'periods': ['2023-12-01'],
                 'scaler': {'ratio_iqr': [float('nan')]}}
        template = self.root / 'template.html'
        template.write_text(TEMPLATE, encoding='utf-8')
        output, standalone = self.root / 'bundle.html', self.root / 'standalone.html'
        output.write_bytes(b'previous bundle')
        standalone.write_bytes(b'previous standalone')
        args = argparse.Namespace(validation_run=None, reference_run=None, panel=Path('panel.csv'),
                                  graph_report=Path('graph.json'), partitions=Path('partitions.csv'),
                                  metrics=Path('metrics.json'), stability_run=None, template=template,
                                  output=output, split_assets=True, standalone_output=standalone)
        with patch.object(atlas, 'read_panel', return_value=panel), \
             patch.object(atlas, 'read_pilot', return_value={'labels': {'one': 0}, 'metric': {'k': 1}}), \
             patch.object(atlas, 'read_stability', return_value={}), self.assertRaises(ValueError):
            atlas.build(args)
        self.assertEqual(output.read_bytes(), b'previous bundle')
        self.assertEqual(standalone.read_bytes(), b'previous standalone')
        self.assertFalse((self.root / 'assets').exists())

    def test_dual_output_matches_two_separate_builds_and_packages_once(self):
        panel = {'entities': [{'id': 'one', 'annual_features': [1, 2, 3, 4, 5]}],
                 'periods': ['2023-12-01'], 'scaler': {'ratio_iqr': [1, 1, 1, 1, 1]}}
        pilot = {'labels': {'one': 0}, 'metric': {'k': 1}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / 'template.html'
            template.write_text('<script>const packed=/*__ATLAS_DATA__*/null;</script>', encoding='utf-8')
            args = argparse.Namespace(validation_run=None, reference_run=None, panel=root / 'panel.csv',
                                      graph_report=root / 'graph.json', partitions=root / 'partitions.csv',
                                      metrics=root / 'metrics.json', stability_run=None, template=template,
                                      output=root / 'single.html', split_assets=False, standalone_output=None)
            for uncompressed in (False, True):
                args.uncompressed = uncompressed
                with self.subTest(uncompressed=uncompressed), \
                     patch.object(atlas, 'read_panel', side_effect=lambda *_: copy.deepcopy(panel)), \
                     patch.object(atlas, 'read_pilot', return_value=pilot), \
                     patch.object(atlas, 'read_stability', return_value={}), \
                     patch.object(atlas, 'package_payload', wraps=atlas.package_payload) as package:
                    args.output = root / 'single.html'
                    args.split_assets = False
                    args.standalone_output = None
                    atlas.build(args)
                    args.output = root / 'single-bundle/index.html'
                    args.split_assets = True
                    atlas.build(args)
                    self.assertEqual(package.call_count, 2)
                    package.reset_mock()
                    args.output = root / 'dual-bundle/index.html'
                    args.standalone_output = root / 'dual.html'
                    atlas.build(args)
                    self.assertEqual(package.call_count, 1)
                    self.assertEqual((root / 'single.html').read_bytes(), (root / 'dual.html').read_bytes())
                    self.assertEqual((root / 'single-bundle/index.html').read_bytes(), (root / 'dual-bundle/index.html').read_bytes())
                    left = {p.name: p.read_bytes() for p in (root / 'single-bundle/assets').iterdir()}
                    right = {p.name: p.read_bytes() for p in (root / 'dual-bundle/assets').iterdir()}
                    self.assertEqual(left, right)

    def test_invalid_output_options_fail_before_scientific_inputs_are_read(self):
        with patch.object(atlas, 'read_panel') as read_panel:
            for split_assets, standalone_output in ((False, Path('other.html')), (True, Path('output.html'))):
                args = argparse.Namespace(output=Path('output.html'), split_assets=split_assets,
                                          standalone_output=standalone_output)
                with self.subTest(split_assets=split_assets), self.assertRaises(ValueError):
                    atlas.build(args)
                read_panel.assert_not_called()
        with patch('sys.argv', ['build_research_atlas.py']):
            self.assertIsNone(atlas.parse_args().standalone_output)
        with patch('sys.argv', ['build_research_atlas.py', '--split-assets', '--standalone-output', 'preview.html']):
            self.assertEqual(atlas.parse_args().standalone_output, Path('preview.html'))

class SavedAtlasRebuildTests(unittest.TestCase):
    def test_saved_rebuild_cli_reproduces_published_html_and_assets(self):
        import io
        from contextlib import redirect_stdout
        from scripts import render_saved_atlas as renderer

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "ci-rebuild/index.html"
            stdout = io.StringIO()
            with patch('sys.argv', ['scripts.render_saved_atlas', '--output', str(output)]), redirect_stdout(stdout):
                renderer.main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result['output'], str(output))
            self.assertFalse(result['standalone'])
            self.assertEqual(output.read_bytes(), (ROOT / 'docs/index.html').read_bytes())
            rebuilt = {p.name: p.read_bytes() for p in (output.parent / 'assets').iterdir()}
            published = {name: (ROOT / 'docs/assets' / name).read_bytes() for name in result['assets']}
            self.assertEqual(rebuilt, published)
            self.assertEqual(len(published), 1)


if __name__ == "__main__":
    unittest.main()
