"""Exercise the jury runner with harmless commands, never with scientific fits."""
import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import run_all as runner


class QuietGuard:
    def tick(self, process):
        pass

    def before_io(self):
        pass

    def close(self):
        pass


class JuryRunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="jury runner spaces ")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = self.root / "artifacts/logs"
        self.output.mkdir(parents=True)
        self.input = self.root / "saved input.txt"
        self.input.write_bytes(b"preserved scientific input")
        self.expected = {self.input.name: hashlib.sha256(self.input.read_bytes()).hexdigest()}
        self.messages = []

    def command(self, key, code):
        return runner.Phase(key, key, (sys.executable, "-X", "utf8", "-B", "-c", code))

    def execute(self, phases, quick=False, timeout=5):
        summary = {"mode": "quick" if quick else "full", "phases": []}
        code = runner.execute(phases, self.root, self.output, runner.thread_environment(),
                              QuietGuard(), summary, self.messages.append, timeout)
        return code, json.loads((self.output / "summary.json").read_text(encoding="utf-8"))

    def test_success_aggregates_and_preserves_inputs_with_spaces(self):
        phases = [self.command("first", "print('first full output')"),
                  self.command("second", "print('second full output')")]
        code, summary = self.execute(phases)
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual([p["returncode"] for p in summary["phases"]], [0, 0])
        self.assertTrue(all(p["seconds"] >= 0 for p in summary["phases"]))
        self.assertTrue(any("[1/2]" in m for m in self.messages))
        self.assertTrue(any("[2/2]" in m for m in self.messages))
        log = (self.output / "run.log").read_text(encoding="utf-8")
        self.assertIn("first full output", log)
        self.assertIn("second full output", log)
        self.assertEqual(runner.check_inputs(self.root, self.expected), self.expected)

    @unittest.skipUnless(os.name == "nt", "The observed read-only copytree failure is Windows-specific")
    def test_runtime_readonly_copytree_can_overwrite_without_changing_claim(self):
        report = self.root / "reports/v1.2/analogue_predictions.csv"
        report.parent.mkdir(parents=True)
        report.write_bytes(b"identical saved prediction\n")
        report.chmod(stat.S_IREAD)
        self.addCleanup(report.chmod, stat.S_IWRITE | stat.S_IREAD)
        original = report.stat()
        broken = self.output / "broken-copy"
        shutil.copytree(report.parent, broken)
        broken_file = broken / report.name
        self.addCleanup(broken_file.chmod, stat.S_IWRITE | stat.S_IREAD)
        with self.assertRaises(PermissionError):
            broken_file.write_bytes(b"cannot overwrite the copied read-only file")
        runtime = runner.prepare_runtime_copy(self.root, self.output, self.expected, QuietGuard())
        code = ("import os,shutil; from pathlib import Path; "
                "p=Path(os.environ['TMP'])/'fresh-scientific-output'; "
                "shutil.copytree(Path('reports/v1.2'),p); "
                "(p/'analogue_predictions.csv').write_bytes(b'recomputed harmless fixture')")
        summary = {"mode": "full", "phases": []}
        env = dict(runner.thread_environment(), TMP=str(self.output))
        status = runner.execute([self.command("findings", code)], self.root, self.output,
                                env, QuietGuard(), summary, self.messages.append, runtime=runtime)
        self.assertEqual(status, 0)
        self.assertEqual((self.output / "fresh-scientific-output" / report.name).read_bytes(),
                         b"recomputed harmless fixture")
        self.assertEqual(report.read_bytes(), b"identical saved prediction\n")
        self.assertEqual(report.stat().st_mode, original.st_mode)
        self.assertEqual(report.stat().st_file_attributes, original.st_file_attributes)
        self.assertEqual(summary["phases"][0]["runtime_source_integrity"], "PASS")
        journal = json.loads((self.output / "runtime-copy.json").read_text(encoding="utf-8"))
        self.assertEqual(set(journal["files"]), {self.input.name, "reports/v1.2/analogue_predictions.csv"})
        self.assertEqual(journal["files"]["reports/v1.2/analogue_predictions.csv"]["sha256"],
                         hashlib.sha256(report.read_bytes()).hexdigest())

    def test_runtime_module_command_uses_identical_copy_and_its_pythonpath(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "__init__.py").write_text("", encoding="utf-8")
        (scripts / "validate_v12_findings.py").write_text(
            "import hashlib,json,os,sys\nfrom pathlib import Path\n"
            "root=Path(__file__).resolve().parents[1]\n"
            "print(json.dumps({'source':str(root),'argv':sys.argv[1:],"
            "'sha':hashlib.sha256((root/'saved input.txt').read_bytes()).hexdigest(),"
            "'pythonpath':os.environ['PYTHONPATH'],'threads':os.environ['OMP_NUM_THREADS']}))\n",
            encoding="utf-8")
        runtime = runner.prepare_runtime_copy(self.root, self.output, self.expected, QuietGuard())
        phase = next(p for p in runner.phases(self.root, "unused-node", False) if p.key == "findings")
        summary = {"mode": "full", "phases": []}
        env = dict(runner.thread_environment(), PYTHONPATH=str(scripts))
        code = runner.execute([phase], self.root, self.output, env, QuietGuard(), summary,
                              self.messages.append, runtime=runtime)
        self.assertEqual(code, 0)
        text = (self.output / "run.log").read_text(encoding="utf-8")
        observed = next(json.loads(line) for line in text.splitlines() if line.startswith('{"source"'))
        self.assertEqual(observed["source"], str(runtime[0]))
        self.assertEqual(observed["sha"], self.expected[self.input.name])
        self.assertEqual(observed["argv"], ["--check"])
        self.assertEqual(observed["threads"], "1")
        self.assertEqual(observed["pythonpath"], os.pathsep.join((str(runtime[0] / "tests"), str(runtime[0]))))
        self.assertEqual(env["PYTHONPATH"], str(scripts))
        self.assertEqual(summary["phases"][0]["cwd"], str(runtime[0]))

    def test_runtime_mutation_fails_and_stops_later_phase(self):
        runtime = runner.prepare_runtime_copy(self.root, self.output, self.expected, QuietGuard())
        summary = {"mode": "full", "phases": []}
        code = runner.execute([
            self.command("findings", "from pathlib import Path; Path('saved input.txt').write_bytes(b'changed')"),
            self.command("after", "from pathlib import Path; Path('must-not-run').touch()")],
            self.root, self.output, runner.thread_environment(), QuietGuard(), summary,
            self.messages.append, runtime=runtime)
        self.assertEqual(code, 1)
        self.assertEqual(len(summary["phases"]), 1)
        self.assertEqual(summary["phases"][0]["returncode"], 0)
        self.assertEqual(summary["phases"][0]["runtime_source_integrity"], "FAIL")
        self.assertFalse((self.root / "must-not-run").exists())
        self.assertEqual(runner.check_inputs(self.root, self.expected), self.expected)

    def test_runtime_requires_every_checked_source_and_keeps_failure_journal(self):
        expected = dict(self.expected, **{"absent-scientific-input.csv": "0" * 64})
        with self.assertRaisesRegex(ValueError, "пропустила проверенные входы"):
            runner.prepare_runtime_copy(self.root, self.output, expected, QuietGuard())
        journal = json.loads((self.output / "runtime-copy.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["status"], "FAIL")
        self.assertIn(self.input.name, journal["files"])

    def test_runtime_detects_source_change_before_starting_scientific_command(self):
        runtime = runner.prepare_runtime_copy(self.root, self.output, self.expected, QuietGuard())
        self.input.write_bytes(b"changed after copy but before scientific command")
        summary = {"mode": "full", "phases": []}
        code = runner.execute([self.command("findings", "print('must-not-start')")],
                              self.root, self.output, runner.thread_environment(), QuietGuard(), summary,
                              self.messages.append, runtime=runtime)
        self.assertEqual(code, 1)
        self.assertFalse(summary["phases"][0]["process_started"])
        self.assertNotIn("must-not-start\n", (self.output / "run.log").read_text(encoding="utf-8"))

    def test_runtime_keeps_scientific_named_results_and_nested_artifacts(self):
        for name in ("reports/results/saved.csv", "data/output/scientific.csv", "data/artifacts/input.json"):
            source = self.root / name
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(name.encode("utf-8"))
        runtime = runner.prepare_runtime_copy(self.root, self.output, self.expected, QuietGuard())
        self.assertEqual(set(runtime[1]), {self.input.name, "reports/results/saved.csv",
                                         "data/output/scientific.csv", "data/artifacts/input.json"})
        self.assertFalse((runtime[0] / "artifacts").exists())

    def test_main_prepares_and_records_runtime_for_scientific_phase(self):
        phase = self.command("findings", "from pathlib import Path; print('fixture cwd: '+str(Path.cwd()))")
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "dependencies", return_value={"node_executable": "unused"}), \
             patch.object(runner, "ResourceGuard", return_value=QuietGuard()), \
             patch.object(runner, "provenance_inputs", return_value=(self.expected, [], {"historical_code_sha256": {}})), \
             patch.object(runner, "snapshot_inputs", return_value=self.expected.copy()), \
             patch.object(runner, "phases", return_value=[phase]), \
             contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["--output", str(self.output)])
        summary_file = next(self.output.glob("*/summary.json"))
        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(summary["runtime_copy"]["status"], "PASS_BYTE_IDENTICAL_WRITABLE_COPY")
        self.assertEqual(summary["phases"][0]["cwd"], summary["runtime_copy"]["root"])
        self.assertEqual(summary["phases"][0]["source_root"], str(self.root))
        self.assertTrue(summary["checked_files_unchanged"])
        journal = json.loads(Path(summary["runtime_copy"]["source_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(journal["files"][self.input.name]["sha256"], self.expected[self.input.name])

    def test_runtime_copy_timeout_keeps_claim_and_failure_journal(self):
        with self.assertRaisesRegex(TimeoutError, "Подготовка рабочей копии"):
            runner.prepare_runtime_copy(self.root, self.output, self.expected, QuietGuard(), timeout=0)
        self.assertEqual(runner.check_inputs(self.root, self.expected), self.expected)
        journal = json.loads((self.output / "runtime-copy.json").read_text(encoding="utf-8"))
        self.assertEqual(journal["status"], "FAIL")
        self.assertGreaterEqual(journal["seconds"], 0)

    def test_failed_command_is_nonzero_and_later_phase_still_records(self):
        code, summary = self.execute([self.command("failure", "import sys; print('failure reason'); sys.exit(7)"),
                                      self.command("after", "print('still checked')")])
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")
        self.assertEqual(summary["phases"][0]["returncode"], 7)
        self.assertEqual(summary["phases"][1]["status"], "PASS")
        self.assertIn("failure reason", (self.output / "run.log").read_text(encoding="utf-8"))

    def test_missing_executable_and_timeout_do_not_look_successful(self):
        missing = runner.Phase("missing", "missing", (str(self.root / "missing program.exe"),))
        code, summary = self.execute([missing, self.command("timeout", "import time; time.sleep(10)")], timeout=0.05)
        self.assertEqual(code, 1)
        self.assertEqual([p["status"] for p in summary["phases"]], ["FAIL", "FAIL"])
        self.assertIn("error", summary["phases"][0])
        self.assertIn("превысил", summary["phases"][1]["error"])

    def run_tree_fixture(self, parent_exits):
        import psutil

        def record(name):
            return ("import json,os,psutil,subprocess,sys,time; from pathlib import Path; "
                    f"Path('{name}.json').write_text(json.dumps({{'pid':os.getpid(),'created':psutil.Process().create_time()}})); ")

        grandchild = record("grandchild") + "time.sleep(30)"
        child = record("child") + f"subprocess.Popen([sys.executable,'-B','-c',{grandchild!r}]); time.sleep(30)"
        parent = (record("parent") + f"subprocess.Popen([sys.executable,'-B','-c',{child!r}]);\n"
                  "while not Path('grandchild.json').exists(): time.sleep(0.01)\n" +
                  ("sys.exit(7)" if parent_exits else "time.sleep(30)"))
        code, summary = self.execute([self.command("tree", parent)], timeout=5 if parent_exits else 2)
        records = []
        try:
            for name in ("parent", "child", "grandchild"):
                records.append(json.loads((self.root / (name + ".json")).read_text(encoding="utf-8")))
            deadline = time.monotonic() + 2
            live = []
            while True:
                live = []
                for item in records:
                    try:
                        process = psutil.Process(item["pid"])
                        if (process.create_time() == item["created"] and process.is_running()
                                and process.status() != psutil.STATUS_ZOMBIE):
                            live.append(process)
                    except psutil.NoSuchProcess:
                        pass
                if not live or time.monotonic() >= deadline:
                    break
                time.sleep(0.02)
            self.assertEqual(live, [], "An owned descendant remained running after the phase")
            self.assertEqual(code, 1)
            self.assertEqual(summary["status"], "FAIL")
            if parent_exits:
                self.assertEqual(summary["phases"][0]["returncode"], 7)
            else:
                self.assertIn("превысил", summary["phases"][0]["error"])
        finally:
            # A failed implementation must not leave this safe fixture behind.
            for item in records:
                try:
                    process = psutil.Process(item["pid"])
                    if process.create_time() == item["created"] and process.status() != psutil.STATUS_ZOMBIE:
                        process.kill()
                except psutil.NoSuchProcess:
                    pass

    def test_timeout_terminates_real_child_and_grandchild(self):
        self.run_tree_fixture(parent_exits=False)

    def test_parent_failure_terminates_real_orphaned_descendants(self):
        self.run_tree_fixture(parent_exits=True)

    def test_quick_success_remains_unverified_and_nonzero(self):
        code, summary = self.execute([self.command("quick", "print('small fixture')")], quick=True)
        self.assertEqual(code, 2)
        self.assertEqual(summary["status"], "UNVERIFIED")

    def test_checked_input_mutation_is_detected(self):
        self.execute([self.command("mutate", "from pathlib import Path; Path('saved input.txt').write_bytes(b'changed')")])
        with self.assertRaisesRegex(ValueError, "SHA256"):
            runner.check_inputs(self.root, self.expected)
        self.input.unlink()
        with self.assertRaisesRegex(ValueError, "Нет нужного файла"):
            runner.check_inputs(self.root, self.expected)

    def test_main_dependency_failure_saves_summary_and_log(self):
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "dependencies", side_effect=RuntimeError("Node.js: не найден")), \
             contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["--output", str(self.output)])
        self.assertEqual(code, 1)
        files = list(self.output.glob("*/summary.json"))
        self.assertEqual(len(files), 1)
        summary = json.loads(files[0].read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "FAIL")
        self.assertIn("Node.js", summary["error"])
        self.assertIn("Node.js", (files[0].parent / "run.log").read_text(encoding="utf-8"))
        self.assertEqual(self.input.read_bytes(), b"preserved scientific input")

    def test_node_absence_is_detected_by_actual_preflight(self):
        with patch.object(runner.shutil, "which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Node.js: не найден"):
                runner.dependencies(quick=True)

    def test_inconsistent_stored_input_claims_are_rejected(self):
        digest = self.expected[self.input.name]
        fixtures = {
            "reports/v1.2/findings_provenance.json": {"sha256": {self.input.name: digest}, "cached_table_sha256": {}, "result_sha256": {}},
            "reports/v1.2/summary.json": {"inputs": {}},
            "reports/v1.2/strict-region-validation/summary.json": {"input_sha256": {self.input.name: "0" * 64}},
            "reports/v1.2-added-value/summary.json": {"inputs_sha256": {}},
            "data/v12/panel.manifest.json": {"sha256": digest},
        }
        for name, value in fixtures.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Противоречие"):
            runner.provenance_inputs(self.root)

    def test_main_successful_command_cannot_hide_changed_input(self):
        fixture = self.command("mutate", "from pathlib import Path; Path('saved input.txt').write_bytes(b'changed')")
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "dependencies", return_value={"node_executable": "unused"}), \
             patch.object(runner, "ResourceGuard", return_value=QuietGuard()), \
             patch.object(runner, "provenance_inputs", return_value=(self.expected, [], {"historical_code_sha256": {}})), \
             patch.object(runner, "snapshot_inputs", return_value=self.expected.copy()), \
             patch.object(runner, "phases", return_value=[fixture]), \
             contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["--output", str(self.output)])
        summary = json.loads(next(self.output.glob("*/summary.json")).read_text(encoding="utf-8"))
        self.assertEqual(summary["phases"][0]["status"], "PASS")
        self.assertEqual(summary["status"], "FAIL")
        self.assertFalse(summary["checked_files_unchanged"])
        self.assertEqual(code, 1)

    def test_scientific_success_is_separate_from_partial_resource_coverage(self):
        class PartialGuard(QuietGuard):
            def coverage(self):
                return {"status": "PARTIAL", "gpu_available": False, "gpus": None,
                        "gpu_error": "GPU reading unavailable", "strict_gpu_required": False}

        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "dependencies", return_value={"node_executable": "unused"}), \
             patch.object(runner, "ResourceGuard", return_value=PartialGuard()), \
             patch.object(runner, "provenance_inputs", return_value=(self.expected, [], {"historical_code_sha256": {}})), \
             patch.object(runner, "snapshot_inputs", return_value=self.expected.copy()), \
             patch.object(runner, "phases", return_value=[self.command("checked", "print('harmless check')")]), \
             contextlib.redirect_stdout(io.StringIO()):
            code = runner.main(["--output", str(self.output)])
        summary = json.loads(next(self.output.glob("*/summary.json")).read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(summary["status"], "PASS")
        self.assertTrue(summary["checked_files_unchanged"])
        self.assertEqual(summary["resource_coverage"]["status"], "PARTIAL")
        self.assertIsNone(summary["resource_coverage"]["gpus"])

    def test_environment_limits_reach_child_without_changing_parent(self):
        with patch.dict(os.environ, {"OMP_NUM_THREADS": "16"}):
            phase = self.command("environment", "import os; print(os.environ['OMP_NUM_THREADS'], os.environ['UV_THREADPOOL_SIZE'])")
            code, _ = self.execute([phase])
            self.assertEqual(code, 0)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "16")
            self.assertIn("1 1", (self.output / "run.log").read_text(encoding="utf-8"))
        with self.assertRaises(ValueError):
            runner.thread_environment(["OMP_NUM_THREADS=16"])

    def test_output_is_new_and_cannot_overwrite_inputs(self):
        first = runner.output_directory(self.root, self.root / "artifacts")
        second = runner.output_directory(self.root, self.root / "artifacts")
        self.assertNotEqual(first, second)
        with self.assertRaises(ValueError):
            runner.output_directory(self.root, self.root / "reports")
        with self.assertRaises(ValueError):
            runner.contained(self.root, "../outside.csv")
        self.assertEqual(self.input.read_bytes(), b"preserved scientific input")

    def test_skipped_existing_unit_is_not_full_acceptance(self):
        (self.root / "sample_unit.py").write_text(
            "import unittest\nclass Demo(unittest.TestCase):\n"
            "    @unittest.skip('fixture absent')\n    def test_example(self): pass\n", encoding="utf-8")
        phase = runner.Phase("unit", "unit", (sys.executable, "-B", "-c", runner.UNIT_RUNNER, "sample_unit"))
        code, summary = self.execute([phase])
        self.assertEqual(code, 1)
        self.assertEqual(summary["status"], "FAIL")
        self.assertIn("UNVERIFIED", (self.output / "run.log").read_text(encoding="utf-8"))

    def test_high_load_guard_pauses_only_its_tree_for_a_bounded_interval(self):
        events = []

        class Process:
            def __init__(self, name):
                self.name = name

            def children(self, recursive):
                return [child]

            def suspend(self):
                events.append((self.name, "suspend"))

            def resume(self):
                events.append((self.name, "resume"))

        parent, child = Process("parent"), Process("child")
        guard = object.__new__(runner.ResourceGuard)
        guard.strict_gpu = False
        guard.psutil = type("Psutil", (), {"NoSuchProcess": ProcessLookupError, "Process": staticmethod(lambda pid: parent)})
        guard.configure = lambda process, idle: events.append((process.name, "idle" if idle else "normal"))
        guard.last = {"time_utc": "sample", "cpu_percent": 1, "ram_percent": 40,
                      "disk_active_percent": {"disk": 1}, "gpu_available": True,
                      "gpus": [{"gpu_percent": 95, "vram_percent": 10}]}
        guard.sample = lambda: guard.last
        guard.last_worker_sample = "sample"
        with patch.object(runner.time, "sleep") as sleep:
            guard.tick(type("ChildHandle", (), {"pid": 123})())
        sleep.assert_called_once()
        self.assertLessEqual(sleep.call_args.args[0], 1)
        for name in ("parent", "child"):
            self.assertIn((name, "idle"), events)
            self.assertIn((name, "suspend"), events)
            self.assertIn((name, "resume"), events)

    def test_worker_io_denial_keeps_fast_process_pass_and_records_partial(self):
        import psutil

        class Monitor:
            def sample(self):
                return {"time_utc": "fixture sample", "cpu_percent": 1, "ram_percent": 40,
                        "disk_active_percent": {"fixture disk": 1}, "gpu_available": True,
                        "gpus": [{"gpu_percent": 1, "vram_percent": 10}], "gpu_error": None}

            def close(self):
                pass

        stream = io.StringIO()
        guard = runner.ResourceGuard(stream, monitor=Monitor())
        self.addCleanup(guard.close)
        phase = self.command("node_version", "import time; print('v22.23.3',flush=True); time.sleep(0.15)")
        summary = {"mode": "full", "phases": []}
        with patch.object(psutil.Process, "io_counters", side_effect=psutil.AccessDenied(pid=2130)):
            code = runner.execute([phase], self.root, self.output, runner.thread_environment(),
                                  guard, summary, self.messages.append)
        self.assertEqual(code, 0)
        self.assertEqual(summary["phases"][0]["status"], "PASS")
        samples = [json.loads(line) for line in stream.getvalue().splitlines()]
        worker = next(item for item in samples if item.get("event") == "worker_limits")
        self.assertFalse(worker["io_available"])
        self.assertIsNone(worker["io_bytes"])
        self.assertIn("AccessDenied", worker["io_error"])
        self.assertEqual(len(worker["cpu_affinity"]), 1)
        coverage = guard.coverage()
        self.assertTrue(coverage["gpu_available"])
        self.assertEqual(coverage["status"], "PARTIAL")
        self.assertEqual(coverage["worker_io"]["status"], "PARTIAL")
        denied_samples = [item for item in samples if item.get("event") == "worker_limits"
                          and item.get("io_available") is False]
        self.assertGreaterEqual(len(denied_samples), 1)
        self.assertEqual(coverage["worker_io"]["unavailable_samples"], len(denied_samples))
        self.assertTrue(coverage["worker_io"]["errors"])

    def test_worker_io_recovery_cannot_hide_prior_denial(self):
        import psutil
        guard = object.__new__(runner.ResourceGuard)
        guard.psutil = psutil
        guard.strict_gpu = False
        guard.worker_io_samples, guard.worker_io_unavailable_samples = 0, 0
        guard.worker_io_errors = set()
        guard.last = {"gpu_available": True, "gpus": [{"gpu_percent": 1, "vram_percent": 10}]}
        denied = type("Denied", (), {"pid": 2130, "io_counters": staticmethod(
            lambda: (_ for _ in ()).throw(psutil.AccessDenied(pid=2130)))})()
        good = type("Good", (), {"pid": 2131, "io_counters": staticmethod(
            lambda: type("Counters", (), {"_asdict": staticmethod(lambda: {"read_bytes": 7, "write_bytes": 11})})())})()
        self.assertIsNone(guard.worker_io(denied)["io_bytes"])
        self.assertEqual(guard.worker_io(good)["io_bytes"], {"read_bytes": 7, "write_bytes": 11})
        self.assertEqual(guard.coverage()["worker_io"]["samples"], 2)
        self.assertEqual(guard.coverage()["status"], "PARTIAL")

    def test_worker_io_partial_is_reported_without_false_gpu_failure(self):
        class PartialIOGuard(QuietGuard):
            def coverage(self):
                return {"status": "PARTIAL", "gpu_available": True, "gpu_unavailable_samples": 0,
                        "worker_io": {"status": "PARTIAL", "errors": ["psutil.AccessDenied reading io_counters"]}}

        captured = io.StringIO()
        with patch.object(runner, "ROOT", self.root), \
             patch.object(runner, "dependencies", return_value={"node_executable": "unused"}), \
             patch.object(runner, "ResourceGuard", return_value=PartialIOGuard()), \
             patch.object(runner, "provenance_inputs", return_value=(self.expected, [], {"historical_code_sha256": {}})), \
             patch.object(runner, "snapshot_inputs", return_value=self.expected.copy()), \
             patch.object(runner, "phases", return_value=[self.command("checked", "print('harmless check')")]), \
             contextlib.redirect_stdout(captured):
            code = runner.main(["--output", str(self.output)])
        self.assertEqual(code, 0)
        summary = json.loads(next(self.output.glob("*/summary.json")).read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["resource_coverage"]["status"], "PARTIAL")
        self.assertIn("IO отдельных процессов", captured.getvalue())
        self.assertNotIn("Не все показания GPU", captured.getvalue())

    def test_worker_io_fix_does_not_hide_mandatory_configure_denial(self):
        import psutil
        guard = object.__new__(runner.ResourceGuard)
        guard.psutil = psutil
        guard.configure = lambda process, idle: (_ for _ in ()).throw(psutil.AccessDenied(pid=process.pid))
        guard.high = lambda: False
        phase = self.command("must_fail", "import time; time.sleep(0.15)")
        summary = {"mode": "full", "phases": []}
        code = runner.execute([phase], self.root, self.output, runner.thread_environment(),
                              guard, summary, self.messages.append)
        self.assertEqual(code, 1)
        self.assertEqual(summary["phases"][0]["status"], "FAIL")
        self.assertIn("error", summary["phases"][0])
        self.assertIn("psutil.AccessDenied", (self.output / "run.log").read_text(encoding="utf-8"))

    def test_gpu_and_vram_pressure_use_the_same_guard_threshold(self):
        guard = object.__new__(runner.ResourceGuard)
        guard.strict_gpu = False
        sample = {"cpu_percent": 1, "ram_percent": 40, "disk_active_percent": {"disk": 1},
                  "gpu_available": True, "gpus": [{"gpu_percent": 94.9, "vram_percent": 20}]}
        guard.sample = lambda: sample
        self.assertFalse(guard.high())
        sample["gpus"][0]["gpu_percent"] = 95
        self.assertTrue(guard.high())
        sample["gpus"][0].update(gpu_percent=1, vram_percent=95)
        self.assertTrue(guard.high())

    def test_strict_monitoring_refuses_unknown_gpu(self):
        guard = object.__new__(runner.ResourceGuard)
        guard.strict_gpu = True
        guard.sample = lambda: {"cpu_percent": 1, "ram_percent": 40, "disk_active_percent": {"disk": 1},
                                "gpu_available": False, "gpu_error": "device inaccessible", "gpus": None}
        with self.assertRaisesRegex(RuntimeError, "Нагрузка GPU неизвестна"):
            guard.high()

    def test_portable_monitoring_keeps_unknown_gpu_and_partial_coverage(self):
        guard = object.__new__(runner.ResourceGuard)
        guard.strict_gpu = False
        guard.last = {"cpu_percent": 1, "ram_percent": 40, "disk_active_percent": {"disk": 1},
                      "gpu_available": False, "gpu_error": "non-NVIDIA device", "gpus": None}
        guard.sample = lambda: guard.last
        self.assertFalse(guard.high())
        self.assertEqual(guard.coverage()["status"], "PARTIAL")
        self.assertIsNone(guard.coverage()["gpus"])
        self.assertIn("non-NVIDIA", guard.coverage()["gpu_error"])
        guard.last["cpu_percent"] = 95
        self.assertTrue(guard.high())

    def test_gpu_adapter_records_reading_failure_as_unavailable(self):
        monitor = object.__new__(runner.LoadMonitor)
        monitor.gpu = type("UnavailableGPU", (), {"sample": staticmethod(lambda: (_ for _ in ()).throw(RuntimeError("driver failed")))})()
        sample = monitor.sample_gpu()
        self.assertFalse(sample["gpu_available"])
        self.assertIsNone(sample["gpus"])
        self.assertIn("driver failed", sample["gpu_error"])

    def test_later_gpu_success_does_not_hide_an_earlier_gap(self):
        guard = object.__new__(runner.ResourceGuard)
        guard.strict_gpu = False
        guard.gpu_unavailable_samples, guard.gpu_errors = 0, set()
        guard.stream, guard.next_sample = io.StringIO(), 0
        samples = iter([
            {"gpu_available": False, "gpus": None, "gpu_error": "temporary failure"},
            {"gpu_available": True, "gpus": [{"gpu_percent": 1, "vram_percent": 10}], "gpu_error": None},
        ])
        guard.monitor = type("Monitor", (), {"sample": staticmethod(lambda: next(samples))})()
        guard.sample()
        guard.next_sample = 0
        guard.sample()
        coverage = guard.coverage()
        self.assertTrue(coverage["gpu_available"])
        self.assertEqual(coverage["status"], "PARTIAL")
        self.assertEqual(coverage["gpu_unavailable_samples"], 1)
        self.assertIn("temporary failure", coverage["gpu_errors"])

    def storage_fixture(self):
        entries = {}
        for name in runner.STORAGE_AUDITS:
            compressed = self.root / (name + ".gz")
            compressed.parent.mkdir(parents=True, exist_ok=True)
            compressed.write_bytes(b"small safe fixture")
            entries[name] = {"physical_storage_path": name + ".gz"}
        storage = self.root / runner.STORAGE_MAP
        storage.parent.mkdir(parents=True, exist_ok=True)
        storage.write_text(json.dumps({"compressed_storage": entries}), encoding="utf-8")
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "verify_logical_storage.py").write_text(
            "import argparse,json; from pathlib import Path\n"
            "p=argparse.ArgumentParser()\n"
            "for key in ['manifest','storage-map','storage-root','expected-manifest-sha256','control-audit']: p.add_argument('--'+key, required=True)\n"
            "a=p.parse_args(); Path('storage-hook-ran.json').write_text(json.dumps(vars(a)))\n",
            encoding="utf-8")
        return storage

    def test_storage_hook_runs_in_full_and_quick_without_scientific_fits(self):
        self.storage_fixture()
        for quick in (False, True):
            with self.subTest(quick=quick):
                plan = runner.phases(self.root, "unused-node", quick)
                selected = [phase for phase in plan if phase.key == "logical_storage"]
                self.assertEqual(len(selected), 1)
                code, summary = self.execute(selected, quick=quick)
                self.assertEqual(code, 2 if quick else 0)
                self.assertEqual(summary["phases"][0]["status"], "PASS")
                observed = json.loads((self.root / "storage-hook-ran.json").read_text(encoding="utf-8"))
                self.assertEqual(observed["expected_manifest_sha256"],
                                 "dc2c95cc0bcc41ec8f1c23c5cc18a0791d9b7371ecbb2a1a32cf4955f601f8f9")
                self.assertEqual(observed["storage_root"], ".")
                self.assertEqual(observed["manifest"], runner.STORAGE_MANIFEST)
                self.assertEqual(observed["storage_map"], runner.STORAGE_MAP)
                self.assertEqual(observed["control_audit"], runner.STORAGE_CONTROL)

    def test_storage_incomplete_map_or_missing_map_cannot_omit_the_check(self):
        storage = self.storage_fixture()
        value = json.loads(storage.read_text(encoding="utf-8"))
        del value["compressed_storage"][runner.STORAGE_AUDITS[1]]
        storage.write_text(json.dumps(value), encoding="utf-8")
        for quick in (False, True):
            with self.subTest(quick=quick), self.assertRaisesRegex(ValueError, "Неполная карта"):
                runner.phases(self.root, "unused-node", quick)
        storage.unlink()
        with self.assertRaisesRegex(ValueError, "prepared-storage.json отсутствует"):
            runner.logical_storage_scope(self.root)

    def test_storage_legacy_scope_is_explicit_when_no_compressed_audits_exist(self):
        scope = runner.logical_storage_scope(self.root)
        self.assertEqual(scope["status"], "LEGACY_NOT_APPLICABLE")
        self.assertIn("Старый комплект", scope["scope"])
        for quick in (False, True):
            self.assertFalse(any(phase.key == "logical_storage" for phase in runner.phases(self.root, "unused-node", quick)))

    def test_storage_failure_stops_later_expensive_phases(self):
        storage = self.command("logical_storage", "import sys; print('storage validation failed'); sys.exit(7)")
        after = self.command("later", "from pathlib import Path; Path('later-was-run').write_text('unexpected')")
        code, summary = self.execute([storage, after])
        self.assertEqual(code, 1)
        self.assertEqual(len(summary["phases"]), 1)
        self.assertEqual(summary["phases"][0]["returncode"], 7)
        self.assertFalse((self.root / "later-was-run").exists())


if __name__ == "__main__":
    unittest.main()
