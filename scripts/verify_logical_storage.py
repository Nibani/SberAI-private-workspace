"""Verify physical gzip and original logical hashes without restoring raw files."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re

BLOCK = 32768


def contained(root, relative):
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or PurePosixPath(relative).is_absolute() or PureWindowsPath(relative).drive
            or any(part in ("", ".", "..") for part in relative.split("/"))):
        raise ValueError("Unsafe storage path: " + repr(relative))
    root = Path(root).resolve()
    path = root / relative
    if path.is_symlink() or root not in path.resolve().parents:
        raise ValueError("Storage path escapes its root: " + relative)
    return path


def checksum(path, opener=open):
    digest = hashlib.sha256()
    size = 0
    with opener(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def valid_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Invalid SHA256")
    return value


def verify_storage(manifest, storage, storage_root, *, control_audit=None):
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("Expected historical files list")
    expected = {}
    for entry in entries:
        name = entry["path"]
        if name in expected:
            raise ValueError("Duplicate historical path: " + name)
        valid_sha(entry["sha256"])
        if type(entry.get("bytes")) is not int or entry["bytes"] < 0:
            raise ValueError("Invalid historical byte count")
        expected[name] = entry
    mapping = storage.get("compressed_storage")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError("Empty logical storage map")
    rows = {}
    for logical, record in sorted(mapping.items()):
        if logical not in expected or record["logical_original_path"] != logical:
            raise ValueError("Logical path is absent from historical manifest")
        original = expected[logical]
        if (record["original_sha256"] != original["sha256"]
                or record["original_bytes"] != original["bytes"]):
            raise ValueError("Storage map changes historical logical fingerprint")
        if not record["physical_storage_path"].endswith(".gz"):
            raise ValueError("Expected gzip physical storage")
        physical = contained(storage_root, record["physical_storage_path"])
        compressed_sha, compressed_size = checksum(physical)
        if (compressed_sha != valid_sha(record["gzip_sha256"])
                or compressed_size != record["gzip_bytes"]):
            raise ValueError("Physical gzip fingerprint differs: " + logical)
        logical_sha, logical_size = checksum(physical, gzip.open)
        if logical_sha != original["sha256"] or logical_size != original["bytes"]:
            raise ValueError("Decoded original fingerprint differs: " + logical)
        rows[logical] = {
            "physical_storage_path": record["physical_storage_path"],
            "gzip_sha256": compressed_sha, "gzip_bytes": compressed_size,
            "decoded_sha256": logical_sha, "decoded_bytes": logical_size,
            "historical_manifest_sha_matches": True,
            "raw_restoration_required": False,
        }
    if control_audit is not None:
        controls = control_audit.get("evidence_sha256", {})
        for logical, row in rows.items():
            candidates = [value for path, value in controls.items()
                          if path.replace("\\", "/") == logical]
            if candidates and any(valid_sha(value) != row["decoded_sha256"] for value in candidates):
                raise ValueError("Independent control audit differs: " + logical)
            row["independent_control_sha_present"] = bool(candidates)
            row["independent_control_sha_matches"] = True if candidates else None
    return {
        "status": "PASS_LOGICAL_STORAGE_ONLY",
        "logical_files_checked": len(rows), "files": rows,
        "scope": "Mapped logical payload only; not a scientific rerun or acceptance of other historical manifest entries",
        "authoritative_manifest_rewritten": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--storage-map", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--control-audit", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    actual_sha, _ = checksum(args.manifest)
    if actual_sha != valid_sha(args.expected_manifest_sha256):
        raise ValueError("Historical manifest fingerprint differs")
    result = verify_storage(json.loads(args.manifest.read_text(encoding="utf-8")),
                            json.loads(args.storage_map.read_text(encoding="utf-8")),
                            args.storage_root,
                            control_audit=json.loads(args.control_audit.read_text(encoding="utf-8"))
                            if args.control_audit else None)
    result["historical_manifest_sha256"] = actual_sha
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

