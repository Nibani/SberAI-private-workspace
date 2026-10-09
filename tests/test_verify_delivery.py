import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.logical_storage import LogicalStorage
from scripts.verify_delivery import archive_declarations, file_fingerprint, verify_physical_manifest


class PhysicalManifestTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / 'SCIENTIFIC-SHA256.json').write_text('{}', encoding='utf-8')
        (self.root / 'result.csv').write_bytes(b'adverse result\n')
        self.save_manifest()

    def save_manifest(self):
        files = {path.relative_to(self.root).as_posix(): {**file_fingerprint(path),
                 'scientific_pin': False, 'source_kind': 'CURRENT_LIVE'}
                 for path in self.root.rglob('*') if path.is_file() and path.name != 'MANIFEST.json'}
        self.manifest = {'schema_version': 2, 'state': 'CURRENT_COMPACT_V4',
                         'files': files, 'file_count': len(files),
                         'total_bytes': sum(row['bytes'] for row in files.values()),
                         'scientific_pin_count': 0, 'archived_scientific_pins': [],
                         'logical_archives': archive_declarations(LogicalStorage(self.root))}
        self.write_manifest()

    def write_manifest(self):
        (self.root / 'MANIFEST.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_exact_delivery_passes_without_git(self):
        self.assertEqual(verify_physical_manifest(self.root)['status'], 'PASS')

    def test_same_length_byte_drift_is_rejected(self):
        (self.root / 'result.csv').write_bytes(b'changed result\n')
        with self.assertRaisesRegex(ValueError, 'bytes/SHA'):
            verify_physical_manifest(self.root)

    def test_arbitrary_runtime_extra_is_rejected(self):
        (self.root / 'unexpected.csv').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'unexpected extra'):
            verify_physical_manifest(self.root)

    def test_output_directory_excluded_from_runtime_inventory(self):
        (self.root / 'artifacts').mkdir()
        (self.root / 'artifacts/log.txt').write_text('runtime output', encoding='utf-8')
        self.assertEqual(verify_physical_manifest(self.root)['status'], 'PASS')

    def test_only_declared_restored_members_are_allowed(self):
        folder = self.root / 'archive'
        folder.mkdir()
        raw = b'negative historical result\n'
        name = 'reports/old/result.csv'
        archive = folder / 'legacy-evidence.zip'
        with zipfile.ZipFile(archive, 'w') as stored:
            stored.writestr(name, raw)
        declaration = {'archive_path': 'archive/legacy-evidence.zip',
                       'archive_sha256': file_fingerprint(archive)['sha256'],
                       'members': {name: {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}}}
        (folder / 'legacy-evidence.manifest.json').write_text(json.dumps(declaration), encoding='utf-8')
        self.save_manifest()
        target = self.root / name
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
        self.assertEqual(verify_physical_manifest(self.root)['status'], 'PASS')
        (target.parent / 'unexpected.csv').write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'unexpected extra'):
            verify_physical_manifest(self.root)

    def test_self_reference_is_rejected(self):
        self.manifest['files']['MANIFEST.json'] = self.manifest['files']['result.csv']
        self.manifest['file_count'] += 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'self-referential'):
            verify_physical_manifest(self.root)

    def test_count_and_total_are_checked(self):
        self.manifest['file_count'] += 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'file count'):
            verify_physical_manifest(self.root)
        self.manifest['file_count'] -= 1
        self.manifest['total_bytes'] += 1
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, 'byte total'):
            verify_physical_manifest(self.root)

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root / 'MANIFEST.json'
        text = path.read_text(encoding='utf-8')
        path.write_text(text[:-1]+', "schema_version": 2}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Duplicate manifest key'):
            verify_physical_manifest(self.root)


if __name__ == '__main__':
    unittest.main()
