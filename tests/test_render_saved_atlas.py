import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import render_saved_atlas as renderer


class SavedAtlasSourcesTests(unittest.TestCase):
    def test_changed_or_missing_pinned_report_preserves_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reports = root / "reports/v1.2"
            reports.mkdir(parents=True)
            sources = {}
            for name in renderer.PINNED_REPORTS:
                content = name.encode()
                (reports / name).write_bytes(content)
                sources["reports/v1.2/" + name] = hashlib.sha256(content).hexdigest()
            contest = {"v12": {"types": []}, "source_hashes": sources}
            with patch.object(renderer, "ROOT", root):
                renderer.check_report_sources(contest)
                output = root / "index.html"
                output.write_bytes(b"previous output")
                for failure in ("changed", "missing_hash", "missing_file"):
                    with self.subTest(failure=failure):
                        target = reports / "types.csv"
                        target.write_bytes(b"changed" if failure == "changed" else b"types.csv")
                        if failure == "missing_hash":
                            contest["source_hashes"] = {}
                        else:
                            contest["source_hashes"] = sources
                        if failure == "missing_file":
                            target.unlink()
                        with patch.object(renderer, "read_atlas_payload", return_value={"contest": contest}), \
                             patch.object(renderer, "render_sections") as render:
                            with self.assertRaisesRegex(ValueError, "report source"):
                                renderer.build(root / "input.html", output)
                            render.assert_not_called()
                        self.assertEqual(output.read_bytes(), b"previous output")
