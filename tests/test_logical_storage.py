import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.logical_storage import LogicalStorage, safe_path


class LogicalStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        (self.root / "archive").mkdir(parents=True)
        self.raw = b"unchanged adverse scientific result\n"
        self.name = "reports/old/result.csv"
        self.archive = self.root / "archive/legacy-evidence.zip"
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(self.name, self.raw)
        self.manifest = {"archive_path": "archive/legacy-evidence.zip",
            "archive_sha256": hashlib.sha256(self.archive.read_bytes()).hexdigest(),
            "members": {self.name: {"sha256": hashlib.sha256(self.raw).hexdigest(), "bytes": len(self.raw)}}}
        self.save_manifest()

    def save_manifest(self):
        (self.root / "archive/legacy-evidence.manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_missing_physical_bytes_restore_only_into_separate_runtime(self):
        store = LogicalStorage(self.root)
        self.assertEqual(store.verify()["status"], "PASS")
        runtime = self.root.parent / "runtime"
        store.materialize(runtime)
        self.assertEqual((runtime / self.name).read_bytes(), self.raw)
        self.assertFalse((self.root / self.name).exists())
        with self.assertRaisesRegex(ValueError, "separate"):
            store.materialize(self.root)
        (runtime / self.name).write_bytes(b"new science")
        with self.assertRaisesRegex(ValueError, "different bytes"):
            store.materialize(runtime)

    def test_changed_retained_file_cannot_hide_behind_good_archive(self):
        retained = self.root / self.name
        retained.parent.mkdir(parents=True)
        retained.write_bytes(b"replaced result")
        with self.assertRaisesRegex(ValueError, "Changed historical"):
            LogicalStorage(self.root).verify()

    def test_archive_tampering_fails_even_if_container_sha_is_updated(self):
        with zipfile.ZipFile(self.archive, "w") as archive:
            archive.writestr(self.name, b"other scientific result")
        self.manifest["archive_sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "member fingerprint"):
            LogicalStorage(self.root).verify()

    def test_duplicate_archive_member_is_rejected(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(self.archive, "a") as archive:
                archive.writestr(self.name, self.raw)
        self.manifest["archive_sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "duplicates"):
            LogicalStorage(self.root).verify()

    def test_unsafe_paths_are_rejected(self):
        for name in ("../escape", "/absolute", "C:/absolute", "reports\\escape"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unsafe"):
                safe_path(self.root, name)

    def supplementary(self, name="reports/temporal-v4/checkpoints/evaluation-1.json"):
        path = self.root / "archive/temporal-v4-details.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr(name, self.raw)
        manifest = {"archive_path": "archive/temporal-v4-details.zip",
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "members": {name: {"sha256": hashlib.sha256(self.raw).hexdigest(), "bytes": len(self.raw)}}}
        (self.root / "archive/temporal-v4-details.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return name

    def test_two_archives_restore_disjoint_original_paths(self):
        extra = self.supplementary()
        store = LogicalStorage(self.root)
        verified = store.verify()
        self.assertEqual(verified["members"], 2)
        self.assertEqual(len(verified["archives"]), 2)
        runtime = self.root.parent / "runtime"
        store.materialize(runtime)
        self.assertEqual((runtime / self.name).read_bytes(), self.raw)
        self.assertEqual((runtime / extra).read_bytes(), self.raw)

    def test_supplement_cannot_shadow_legacy_even_with_identical_bytes(self):
        self.supplementary(self.name)
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            LogicalStorage(self.root)

    def test_missing_supplement_manifest_or_zip_is_not_optional(self):
        self.supplementary()
        (self.root / "archive/temporal-v4-details.zip").unlink()
        with self.assertRaisesRegex(ValueError, "both exist"):
            LogicalStorage(self.root)

    def test_corrupt_member_crc_is_rejected_after_container_rehash(self):
        raw = bytearray(self.archive.read_bytes())
        offset = raw.find(self.raw)
        self.assertGreaterEqual(offset, 0)
        raw[offset] ^= 1
        self.archive.write_bytes(raw)
        self.manifest["archive_sha256"] = hashlib.sha256(raw).hexdigest()
        self.save_manifest()
        with self.assertRaises(zipfile.BadZipFile):
            LogicalStorage(self.root).verify()

    def test_symlink_archive_member_is_rejected(self):
        import stat
        with zipfile.ZipFile(self.archive, "w") as archive:
            info = zipfile.ZipInfo(self.name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, self.raw)
        self.manifest["archive_sha256"] = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, "symlinks"):
            LogicalStorage(self.root).verify()


if __name__ == "__main__":
    unittest.main()
