"""Run both required v4 verifiers; absent implementation remains UNVERIFIED."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
VERIFIERS = {"temporal": ("scripts.verify_temporal_delivery", ()),
             "conditional": ("scripts.verify_conditional_profiles", ("--check",))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=("all", *VERIFIERS), default="all", help="verify one saved scientific package, or both by default")
    parser.add_argument("--output", type=Path, help="parent directory for a new isolated runtime if archived details need restoring")
    args = parser.parse_args(argv)
    from scripts.logical_storage import LogicalStorage
    storage = LogicalStorage(ROOT)
    storage.verify()
    scopes = tuple(VERIFIERS) if args.only == "all" else (args.only,)
    prefixes = tuple(f"reports/{scope}-v4/" for scope in scopes)
    missing_archived = [name for name in storage.members if name.startswith(prefixes) and not (ROOT / name).is_file()]
    if missing_archived:
        # Reuse the same exact-byte runtime copy as run_all; pinned verifiers still
        # see their original paths and derive ROOT from their unchanged source files.
        import run_all as runner
        output = runner.output_directory(ROOT, args.output or ROOT / "artifacts/current-science-check")
        runtime, records = runner.prepare_runtime_copy(ROOT, output, {}, emit=lambda message: print(message, flush=True))
        env = runner.thread_environment()
        env["PYTHONPATH"] = str(runtime)
        command = [sys.executable, "-B", "-X", "utf8", "-m", "scripts.verify_current_results", "--only", args.only]
        result = subprocess.run(command, cwd=runtime, env=env)
        # A successful producer verifier cannot hide source/runtime byte drift.
        runner.check_runtime_sources(ROOT, runtime, records)
        print(json.dumps({"scope": args.only, "runtime_root": str(runtime), "runtime_source_integrity": "PASS", "returncode": result.returncode}))
        return result.returncode
    records = []
    for scope in scopes:
        module, extra = VERIFIERS[scope]
        script = ROOT / (module.replace(".", "/") + ".py")
        if not script.is_file():
            records.append({"module": module, "status": "UNVERIFIED", "reason": "Required current science verifier has not been delivered"})
            continue
        result = subprocess.run([sys.executable, "-B", "-X", "utf8", "-m", module, *extra], cwd=ROOT)
        records.append({"module": module, "status": "PASS" if result.returncode == 0 else "UNVERIFIED" if result.returncode == 2 else "FAIL", "returncode": result.returncode})
    status = "FAIL" if any(r["status"] == "FAIL" for r in records) else "UNVERIFIED" if any(r["status"] == "UNVERIFIED" for r in records) else "PASS"
    print(json.dumps({"status": status, "scope": args.only, "required_current_checks": records}))
    return {"PASS": 0, "UNVERIFIED": 2, "FAIL": 1}[status]


if __name__ == "__main__":
    raise SystemExit(main())
