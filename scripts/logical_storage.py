"""Exact evidence bytes, in-place or in separate verified ZIP archives.

Readers use original repository-relative paths. Extraction writes only to a
separate destination; it never restores evidence into the source tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path, PurePosixPath
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "archive/legacy-evidence.manifest.json"
ARCHIVES = ((MANIFEST, "archive/legacy-evidence.zip"),
            ("archive/temporal-v4-details.manifest.json", "archive/temporal-v4-details.zip"))


def safe_path(root: Path, name: str) -> Path:
    value = PurePosixPath(name)
    target = (root / name).resolve()
    if (not name or value.is_absolute() or ".." in value.parts or
            "\\" in name or ":" in name or not target.is_relative_to(root.resolve())):
        raise ValueError(f"Unsafe logical path: {name}")
    return target


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate manifest key: {key}")
        result[key] = value
    return result


class LogicalStorage:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.manifests, self.members, self.member_archives = {}, {}, {}
        for manifest_name, archive_name in ARCHIVES:
            path = safe_path(self.root, manifest_name)
            archive = safe_path(self.root, archive_name)
            if path.is_file() != archive.is_file():
                raise ValueError(f"Archive and manifest must both exist: {archive_name}")
            if not path.is_file():
                continue
            declaration = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique)
            if declaration["archive_path"] != archive_name:
                raise ValueError(f"Manifest archive path differs: {manifest_name}")
            if not re.fullmatch(r"[a-f0-9]{64}", declaration["archive_sha256"]):
                raise ValueError(f"Invalid archive fingerprint: {manifest_name}")
            self.manifests[manifest_name] = declaration
            for name, row in declaration["members"].items():
                safe_path(self.root, name)
                if name in self.members:
                    raise ValueError(f"Overlapping logical archive member: {name}")
                if (not isinstance(row["bytes"], int) or row["bytes"] < 0 or
                        not re.fullmatch(r"[a-f0-9]{64}", row["sha256"])):
                    raise ValueError(f"Invalid archive member fingerprint: {name}")
                self.members[name] = row
                self.member_archives[name] = archive_name
        # Compatibility for callers that specifically report the immutable legacy archive.
        self.manifest = self.manifests.get(MANIFEST)

    def physical_files(self) -> list[str]:
        return [name for key, value in self.manifests.items() for name in (key, value["archive_path"])]

    def exists(self, name: str) -> bool:
        return safe_path(self.root, name).is_file() or name in self.members

    def read_bytes(self, name: str) -> bytes:
        path = safe_path(self.root, name)
        if path.is_file():
            raw = path.read_bytes()
        elif name in self.members:
            with zipfile.ZipFile(safe_path(self.root, self.member_archives[name])) as archive:
                raw = archive.read(name)
        else:
            raise FileNotFoundError(f"Missing logical evidence: {name}")
        if name in self.members:
            row = self.members[name]
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"Changed historical evidence: {name}")
        return raw

    def read_json(self, name: str):
        return json.loads(self.read_bytes(name).decode("utf-8"), object_pairs_hook=_unique)

    def verify(self) -> dict:
        if not self.manifests:
            return {"status": "LEGACY_NOT_APPLICABLE", "members": 0}
        verified = []
        for manifest_name, declaration in self.manifests.items():
            path = safe_path(self.root, declaration["archive_path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != declaration["archive_sha256"]:
                raise ValueError(f"Archive SHA256 differs: {declaration['archive_path']}")
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)) or set(names) != set(declaration["members"]):
                    raise ValueError("Archive member set differs or contains duplicates")
                for info in archive.infolist():
                    safe_path(self.root, info.filename)
                    if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
                        raise ValueError("Archive directories or symlinks are forbidden")
                    row = declaration["members"][info.filename]
                    if info.file_size != row["bytes"]:
                        raise ValueError(f"Archive member fingerprint differs: {info.filename}")
                    raw = archive.read(info)
                    if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                        raise ValueError(f"Archive member fingerprint differs: {info.filename}")
                if archive.testzip() is not None:
                    raise ValueError("Archive CRC differs")
            verified.append({"manifest": manifest_name, "archive": declaration["archive_path"],
                             "archive_sha256": declaration["archive_sha256"], "members": len(declaration["members"])})
        # A retained file must also preserve the archived bytes.
        for name in self.members:
            if safe_path(self.root, name).is_file():
                self.read_bytes(name)
        return {"status": "PASS", "members": len(self.members), "archives": verified,
                "archive_sha256": self.manifest["archive_sha256"] if self.manifest else None,
                "scope": "Exact member set, CRC, all member SHA256/bytes and retained originals"}

    def materialize(self, destination: Path, names=None) -> dict:
        destination = Path(destination).resolve()
        if destination == self.root or self.root.is_relative_to(destination):
            raise ValueError("Evidence destination must be separate from source root")
        if destination.is_relative_to(self.root) and destination.relative_to(self.root).parts[0] not in ("artifacts", "runs", ".local"):
            raise ValueError("A destination inside the source tree must be under artifacts, runs or .local")
        self.verify()
        records = {}
        for name in sorted(self.members if names is None else names):
            raw = self.read_bytes(name)
            target = safe_path(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not target.is_file() or target.read_bytes() != raw:
                    raise ValueError(f"Materialization would replace different bytes: {name}")
            else:
                with target.open("xb") as writer:
                    writer.write(raw)
            records[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        return records


def read_bytes(name: str, root: Path = ROOT) -> bytes:
    return LogicalStorage(root).read_bytes(name)


def read_json(name: str, root: Path = ROOT):
    return LogicalStorage(root).read_json(name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--materialize", type=Path, help="restore original paths into a separate directory")
    args = parser.parse_args()
    storage = LogicalStorage(args.root)
    result = storage.verify()
    if args.materialize:
        result["materialized"] = len(storage.materialize(args.materialize))
        result["destination"] = str(args.materialize.resolve())
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
