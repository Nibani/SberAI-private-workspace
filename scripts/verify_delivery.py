"""Check exact historical storage, scientific pins and deterministic saved-data rendering."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import stat
import subprocess

from scripts.logical_storage import LogicalStorage, safe_path, _unique

ROOT = Path(__file__).resolve().parents[1]


def file_fingerprint(path):
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return {"bytes": size, "sha256": digest.hexdigest()}


def physical_file(root, name):
    path = safe_path(root, name)
    if Path(name).as_posix() != name or name == "MANIFEST.json":
        raise ValueError(f"Noncanonical or self-referential manifest path: {name}")
    direct = root / name
    for part in (direct, *direct.parents):
        if part == root:
            break
        attributes = part.lstat()
        if part.is_symlink() or getattr(attributes, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError(f"Manifest symlink/reparse point forbidden: {name}")
    if not path.is_file():
        raise ValueError(f"Manifest file missing or not regular: {name}")
    return path


def git_inventory(root):
    result = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    names = set(result.stdout.decode("utf-8").split("\0")) - {"", "MANIFEST.json"}
    return sorted(name for name in names if (root / name).exists() or (root / name).is_symlink())


def archive_declarations(store):
    return {name: {**file_fingerprint(store.root / name),
                   "archive_path": data["archive_path"],
                   "archive_sha256": data["archive_sha256"], "members": len(data["members"])}
            for name, data in store.manifests.items()}


def runtime_inventory(root):
    # These are the same development/output exclusions as prepare_runtime_copy.
    excluded = {".git", ".swarm", ".serena", ".local", "artifacts", "runs", ".venv", "venv"}
    names = set()
    for directory, folders, files in os.walk(root, followlinks=False):
        base = Path(directory)
        folders[:] = [name for name in folders if name not in {"__pycache__", "node_modules"}
                      and not (base == root and name in excluded)]
        for name in folders:
            path = base / name
            info = path.lstat()
            if path.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
                raise ValueError(f"Runtime directory symlink/reparse forbidden: {path}")
        for name in files:
            if name.endswith(".pyc"):
                continue
            relative = (base / name).relative_to(root).as_posix()
            if relative == "MANIFEST.json":
                continue
            physical_file(root, relative)
            names.add(relative)
    return names


def verify_physical_manifest(root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if manifest.get("schema_version") != 2 or manifest.get("state") != "CURRENT_COMPACT_V4":
        raise ValueError("Final v4 physical MANIFEST required")
    files = manifest["files"]
    if not isinstance(files, dict) or manifest["file_count"] != len(files):
        raise ValueError("Physical manifest file count differs")
    pins = json.loads((root / "SCIENTIFIC-SHA256.json").read_text(encoding="utf-8"), object_pairs_hook=_unique)
    if manifest["scientific_pin_count"] != len(pins):
        raise ValueError("Logical scientific pin count differs")
    total = 0
    for name, expected in files.items():
        if (type(expected["bytes"]) is not int or expected["bytes"] < 0 or
                not re.fullmatch(r"[a-f0-9]{64}", expected["sha256"])):
            raise ValueError(f"Invalid physical fingerprint: {name}")
        actual = file_fingerprint(physical_file(root, name))
        if actual != {key: expected[key] for key in ("bytes", "sha256")}:
            raise ValueError(f"Physical manifest bytes/SHA differ: {name}")
        if expected["scientific_pin"] is not (name in pins) or expected["source_kind"] != "CURRENT_LIVE":
            raise ValueError(f"Physical manifest origin differs: {name}")
        total += actual["bytes"]
    if manifest["total_bytes"] != total:
        raise ValueError("Physical manifest byte total differs")
    store = LogicalStorage(root)
    if manifest["logical_archives"] != archive_declarations(store):
        raise ValueError("Logical archive declarations differ")
    if manifest["archived_scientific_pins"] != sorted(set(pins) - set(files)):
        raise ValueError("Archived scientific pin paths differ")
    if (root / ".git").exists():
        if set(git_inventory(root)) != set(files):
            raise ValueError("Git delivery inventory differs from physical MANIFEST")
    else:
        actual = runtime_inventory(root)
        required = set(files)
        if not required <= actual or not actual <= required | set(store.members):
            raise ValueError("Runtime contains missing delivery files or unexpected extra files")
    return {"status": "PASS", "files": len(files), "bytes": total,
            "scope": "All delivered physical bytes; exact nonignored Git source inventory or runtime inventory allowing only declared restored archive members and excluded development/output directories"}


def verify(root=ROOT):
    physical = verify_physical_manifest(root)
    store = LogicalStorage(root)
    archive = store.verify()
    pins = store.read_json("SCIENTIFIC-SHA256.json")
    for name, digest in pins.items():
        if hashlib.sha256(store.read_bytes(name)).hexdigest() != digest:
            raise ValueError(f"Scientific pin differs: {name}")
    return {"status": "PASS", "physical_manifest": physical, "legacy": archive, "scientific_pins": len(pins)}


def rebuild(root=ROOT):
    from scripts.render_saved_atlas import build
    import tempfile
    with tempfile.TemporaryDirectory(prefix="atlas-rebuild-") as temp:
        target = Path(temp)
        for name in ("first", "second"):
            build(root / "docs/index.html", target / name / "index.html")
        snapshots = []
        for name in ("first", "second"):
            folder = target / name
            snapshots.append({p.relative_to(folder).as_posix(): p.read_bytes() for p in folder.rglob("*") if p.is_file()})
        if snapshots[0] != snapshots[1]:
            raise ValueError("Two saved-data renders differ")
        for name, raw in snapshots[0].items():
            if (root / "docs" / name).read_bytes() != raw:
                raise ValueError(f"Rebuilt output differs from delivered atlas: {name}")
        return {"status": "PASS", "scope": "Two saved-payload/template renders equal delivered bytes; not reconstruction from raw data",
                "files": {name: hashlib.sha256(raw).hexdigest() for name, raw in snapshots[0].items()}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    print(json.dumps(rebuild() if args.rebuild else verify(), ensure_ascii=False))
