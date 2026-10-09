"""Run the frozen CLI on one CPU; disk backoff stays in its own monitor."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
if len(sys.argv) < 2 or sys.argv[1] not in ("diagnostics", "run"):
    raise SystemExit("Expected diagnostics or run and the unchanged runner arguments")
repo = Path("C:/Users/Andreyka/Documents/Codex/2026-10-08/goal-goal-c-users-andreyka-documents-2/work/repo")
source = repo / "scripts/competition_synthetic.py"
if hashlib.sha256(source.read_bytes()).hexdigest() != "f549c10b7ff5571795dda451fadcaa3e3e8eca43a0ae95220efdc76115b6d619":
    raise SystemExit("Frozen runner changed")
sys.path.insert(0, str(repo))
import psutil
process = psutil.Process()
process.cpu_affinity([process.cpu_affinity()[0]])
process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
assert len(process.cpu_affinity()) == 1
print(json.dumps({"launcher": "single_cpu_frozen_runner", "affinity": process.cpu_affinity(),
                  "priority": int(process.nice()), "bytecode_disabled": sys.dont_write_bytecode}), flush=True)
from scripts import competition_synthetic as runner
sys.argv[0] = "competition_synthetic"
runner.main()
