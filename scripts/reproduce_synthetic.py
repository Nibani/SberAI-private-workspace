"""Explicit full rerun of the unchanged registered recipe, or a small CI fixture.

No paths to the original workstation are required. The saved-result verifier
is the default acceptance route; this module is an explicit optimizer run.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import subprocess
import threading
import time
from types import SimpleNamespace

for pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[pool] = "1"
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SHA = "8a94e00e8bfe1fe8823c3411fcf48ce066de578da1142bceb579ced37d7f2415"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe_vram(command):
    """Return measured fractions, preserving unknown readings as None."""
    from scripts.science_portable_resources import gpu_number
    if not command:
        return {}, "nvidia_smi_unavailable"
    try:
        raw = subprocess.check_output([command, "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=3, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), stderr=subprocess.DEVNULL)
        values = {}
        for line in raw.splitlines():
            index, used, total = line.split(",")
            used, total = gpu_number(used), gpu_number(total)
            values[index.strip()] = used / total if used is not None and total is not None and total > 0 else None
        return values, None if values and all(value is not None for value in values.values()) else "vram_measurement_partial"
    except Exception as exc:
        return {}, "vram_probe_unverified:" + type(exc).__name__


def vram_over_limit(values):
    return any(value is not None and value >= .95 for value in values.values())


def windows_monitor_with_vram(original_monitor):
    """Current reproduction guard v1, added after the completed historical run.

    Inherit the unchanged registered Windows PDH guard and add a VRAM gate.
    No optimizer, scientific parameter, original source or historical pin changes.
    """
    class WindowsVRAMMonitor(original_monitor):
        def __init__(self, output, heavy=False):
            super().__init__(output, heavy=heavy)
            self.vram_ready = threading.Event()
            self.vram_blocked = threading.Event()
            self.vram_samples = 0
            self.vram_output = Path(output).with_name(Path(output).stem + ".vram.jsonl")
            self.vram_thread = threading.Thread(target=self.vram_loop, daemon=True)

        def vram_loop(self):
            try:
                with self.vram_output.open("a", encoding="utf-8") as handle:
                    while not self.stop.is_set():
                        values, error = probe_vram(self.gpu_command)
                        if error:
                            self.errors.add(error)
                        busy = self.heavy and vram_over_limit(values)
                        (self.vram_blocked.set if busy else self.vram_blocked.clear)()
                        handle.write(json.dumps({"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "adapter": "original_windows_pdh_plus_vram_v1", "vram_fraction": values,
                            "vram_measurement_verified": error is None, "measurement_error": error,
                            "compute_paused_for_vram": busy, "threshold": .95,
                            "scope": "Added after original completed run; unavailable VRAM is partial, not zero."}) + "\n")
                        handle.flush()
                        self.vram_samples += 1
                        self.vram_ready.set()
                        self.stop.wait(5)
            except Exception as exc:
                self.errors.add("vram_monitor_failure:" + type(exc).__name__)
                self.vram_blocked.set()
                self.vram_ready.set()

        def start(self):
            self.vram_thread.start()
            if not self.vram_ready.wait(10) or not self.vram_samples:
                raise RuntimeError("Windows VRAM wrapper did not produce a measurement-status sample")
            super().start()

        def gate(self):
            while True:
                super().gate()
                if not self.vram_thread.is_alive():
                    raise RuntimeError("Windows VRAM wrapper stopped unexpectedly")
                if not self.vram_blocked.is_set():
                    return
                print("resource pause: measured VRAM >=95%", flush=True)
                self.stop.wait(5)

        def close(self):
            super().close()
            self.vram_thread.join(5)

    return WindowsVRAMMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--fixture", action="store_true", help="two tiny mechanical seeds, no comparative quality metrics")
    args = parser.parse_args()
    repo = args.repo.resolve()
    regpath = (args.registration or repo / "reports/competition-enhancement/synthetic/executed-20261009/provenance/registration.json").resolve()
    if digest(regpath) != REGISTRATION_SHA:
        raise ValueError("Original registration bytes changed")
    reg = json.loads(regpath.read_text(encoding="utf-8"))
    for name, expected in reg["files"].items():
        if digest(repo / name) != expected:
            raise ValueError("Registered source changed: " + name)
    for package, expected in reg["environment"]["packages"].items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise ValueError(f"{package}: installed {actual}, require {expected}; install artifact/provenance/requirements-reproduction.txt")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("No resume or overwrite: " + str(output))
    output.mkdir(parents=True)
    sys.path.insert(0, str(repo))
    import psutil
    from scripts import competition_synthetic as runner
    process = psutil.Process()
    if os.name == "nt":
        process.cpu_affinity([process.cpu_affinity()[0]])
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        runner.ResourceMonitor = windows_monitor_with_vram(runner.ResourceMonitor)
        adapter = "original_windows_pdh_plus_vram_v1"
    else:
        from scripts.science_portable_resources import PortableResourceMonitor
        runner.ResourceMonitor = PortableResourceMonitor
        adapter = "portable_psutil_partial_v2"
    runner.write_json(output / "reproduction-metadata.json", {
        "scope": "MECHANICAL_FIXTURE_NOT_FULL_SCIENTIFIC_REPEAT" if args.fixture else "FULL_ORIGINAL_REGISTERED_RECIPE",
        "registration_sha256": digest(regpath), "source_files_sha256": reg["files"],
        "resource_adapter": adapter, "recipe_changed": False,
        "reproduction_driver_sha256": digest(Path(__file__).resolve()),
        "resource_wrapper_revision_added_after_original_completed_run": True,
        "limitations": ["Extra VRAM95 gate was added after the original completed run. Unknown VRAM is explicit partial measurement in *.vram.jsonl, never zero."] if os.name == "nt" else ["Disk/GPU/VRAM may be unavailable or partial; inspect resource log. CPU/RAM and measured GPU/VRAM95 pause the next fit.", "Affinity or nice may be unsupported on macOS; errors are explicitly reported. Not the Windows PDH resource guarantee."]})
    if args.fixture:
        monitor = runner.ResourceMonitor(output / "resource-monitor.jsonl", heavy=True)
        try:
            monitor.start()
            runner.numeric()
            with runner.threadpool_limits(limits=1):
                for seed in (55001, 55002):
                    x, truth = runner.gaussian_data(seed, "weak", n=24)
                    graph = runner.make_graph(x, truth, seed, 1, "weak_noise", n_edges=40)
                    fitted = runner.fit_models(x, graph, runner.fit_seed(seed, 1), "evaluation", {"joint_alpha": 1., "kefrine_beta": 2.}, monitor, {})
                    if not all(len(runner.np.unique(labels)) == 4 for labels, _ in fitted.values()):
                        raise AssertionError("Mechanical fixture lost nonempty groups")
            runner.write_json(output / "fixture-result.json", {"status": "PASS_MECHANICAL_PORTABLE_FIXTURE", "seeds": [55001, 55002], "N": 24, "K": 4,
                "quality_metrics_computed": False, "scope": "same frozen imports and fit functions; not a full synthetic rerun or quality study",
                "resource_samples": monitor.samples, "resource_errors": sorted(monitor.errors), "resource_adapter": adapter})
        finally:
            monitor.close()
        print("PASS_MECHANICAL_PORTABLE_FIXTURE: no quality metrics; not full scientific repeat")
        return
    slot = "explicit-portable-full-reproduction"
    runner.diagnostics(SimpleNamespace(registration=regpath, output=output / "diagnostics", slot_id=slot))
    runner.run(SimpleNamespace(registration=regpath, diagnostics=output / "diagnostics/diagnostics.json", run_dir=output / "full-run", slot_id=slot))


if __name__ == "__main__":
    main()
