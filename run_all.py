"""One offline entry point for the jury's scientific and delivery checks."""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "POLARS_MAX_THREADS",
              "BLIS_NUM_THREADS", "UV_THREADPOOL_SIZE")
STORAGE_MAP = "reports/competition-enhancement/delivery/cleanup-new/prepared-storage.json"
STORAGE_MANIFEST = "reports/external-national-2026-10-03/manifest.json"
STORAGE_CONTROL = "reports/economic-generalization-2026-10-03/acceptance/control-cohort-audit.json"
STORAGE_MANIFEST_SHA256 = "dc2c95cc0bcc41ec8f1c23c5cc18a0791d9b7371ecbb2a1a32cf4955f601f8f9"
STORAGE_AUDITS = tuple(f"reports/external-national-2026-10-03/join-audit-{year}.csv" for year in (2023, 2024))
RUNTIME_PHASES = frozenset(("findings", "added_value", "science_units", "atlas_units"))
RUNTIME_EXCLUDED = frozenset((".git", ".swarm", ".serena", ".local", "artifacts", "runs", ".venv", "venv"))
SCIENCE_TESTS = ("test_v12_core", "test_v12_edges", "test_network_conventions",
                 "test_artifact_integrity", "test_v12_findings", "test_v12_results",
                 "test_added_value")
ATLAS_TESTS = ("test_atlas_bundle", "test_v12_atlas", "test_render_saved_atlas",
               "test_practical_cases", "test_peer_atlas_coverage",
               "test_continuous_atlas", "test_research_atlas.ResearchAtlasUnitTests")
QUICK_TESTS = ("test_artifact_integrity", "test_atlas_bundle", "test_render_saved_atlas",
               "test_peer_atlas_coverage", "test_research_atlas.ResearchAtlasUnitTests")
UNIT_RUNNER = """import sys, unittest
suite = unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.skipped:
    print('UNVERIFIED: skipped tests are not full acceptance', file=sys.stderr)
sys.exit(0 if result.wasSuccessful() and not result.skipped else 1)
"""
LIMITATIONS = [
    "Node проверяет код и переключатели на поддельном DOM; реальный браузер, карта, обрезание текста и скорость сети не проверены.",
    "Повторяется фиксированный научный рецепт и его проверки; сохранённые сетки выбора модели и все исторические эксперименты не пересчитываются.",
    "Проверка относится к сохранённым данным 2023–2024; внешние исходы 2025 не загружаются.",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def contained(root: Path, relative: str) -> Path:
    value = Path(relative)
    target = (root / value).resolve()
    if value.is_absolute() or ".." in value.parts or not target.is_relative_to(root.resolve()):
        raise ValueError(f"Путь выходит за каталог проекта: {relative}")
    return target


def fingerprint(path: Path, guard=None) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        while True:
            if guard:
                guard.before_io()
            block = source.read(32768)
            if not block:
                break
            result.update(block)
    return result.hexdigest()


def thread_environment(extra=()):
    env = os.environ.copy()
    for item in extra:
        key, separator, value = item.partition("=")
        if not separator or key not in THREAD_ENV or value != "1":
            raise ValueError("--thread-env принимает только имя вычислительного потока из списка и значение 1")
        env[key] = value
    env.update({key: "1" for key in THREAD_ENV})
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               CUDA_VISIBLE_DEVICES="-1")
    return env


def dependencies(quick=False):
    if sys.version_info < (3, 10):
        raise RuntimeError("Нужен Python 3.10 или новее.")
    expected = {"numpy": "2.2.6", "pandas": "2.2.3", "scipy": "1.14.1",
                "scikit-learn": "1.5.2", "networkx": "3.4.2", "psutil": "6.1.0"}
    if not quick:
        expected.update(igraph="0.11.8", leidenalg="0.10.2")
    packages, issues = {}, []
    for name, wanted in expected.items():
        try:
            packages[name] = importlib.metadata.version(name)
            if packages[name] != wanted:
                issues.append(f"{name}: установлен {packages[name]}, нужен {wanted}")
        except importlib.metadata.PackageNotFoundError:
            issues.append(f"{name}: не установлен")
    try:
        packages["threadpoolctl"] = importlib.metadata.version("threadpoolctl")
    except importlib.metadata.PackageNotFoundError:
        issues.append("threadpoolctl: не установлен")
    node = shutil.which("node")
    if not node:
        issues.append("Node.js: не найден. Установите Node.js 22 LTS или новее с nodejs.org и откройте терминал заново.")
    if issues:
        requirements = "requirements.txt" if quick else "requirements-models.txt"
        raise RuntimeError("\n".join(issues) + f"\nУстановите Python-зависимости: python -m pip install -r {requirements}")
    return {"python": sys.version, "python_executable": sys.executable,
            "packages": packages, "node_executable": node}


def provenance_inputs(root: Path):
    """Use stored provenance, refusing conflicting or missing SHA declarations."""
    hashes = {}
    manifests = ("reports/v1.2/findings_provenance.json", "reports/v1.2/summary.json",
                 "reports/v1.2/strict-region-validation/summary.json",
                 "reports/v1.2-added-value/summary.json", "data/v12/panel.manifest.json")

    def add(relative, digest):
        contained(root, relative)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Некорректный SHA256: {relative}")
        if relative in hashes and hashes[relative] != digest:
            raise ValueError(f"Противоречие между сохранёнными SHA256: {relative}")
        hashes[relative] = digest

    values = [json.loads(contained(root, p).read_text(encoding="utf-8")) for p in manifests]
    findings, summary, strict, added, panel = values
    for name, digest in findings["sha256"].items():
        add(name, digest)
    for key in ("cached_table_sha256", "result_sha256"):
        for name, digest in findings[key].items():
            add("reports/v1.2/" + name, digest)
    for item in summary["inputs"].values():
        add(item["path"], item["sha256"])
    historical_code = {}
    for name, digest in strict["input_sha256"].items():
        # The regional packet names an older recipe revision. Its old joint.py
        # hash describes that revision; findings provenance pins today's source.
        if (name == "sbercluster/joint.py" and strict.get("historical_recipe_source_commit") == "c7905a3"
                and digest == "33732d5133814e3f5794338343420abb6104494b356998e071c249b9672d9fb1"
                and name in hashes and hashes[name] != digest):
            historical_code[name] = digest
        else:
            add(name, digest)
    for name, digest in added["inputs_sha256"].items():
        add(name, digest)
    add("data/v12/panel.csv.gz", panel["sha256"])
    return hashes, manifests, {"recipe_source_commit": strict.get("historical_recipe_source_commit"),
                              "historical_code_sha256": historical_code,
                              "scope": "Historical source claims are retained separately; current sources are checked against findings_provenance.json."}


def logical_storage_scope(root: Path):
    compressed = [name + ".gz" for name in STORAGE_AUDITS]
    map_path = contained(root, STORAGE_MAP)
    if not map_path.is_file():
        if any(contained(root, name).exists() for name in compressed):
            raise ValueError("Сжатые join-audit присутствуют, но prepared-storage.json отсутствует")
        return {"status": "LEGACY_NOT_APPLICABLE", "files": [],
                "scope": "Старый комплект: карты хранения и сжатых join-audit нет; logical-storage этап не требуется."}
    storage = json.loads(map_path.read_text(encoding="utf-8")).get("compressed_storage")
    if not isinstance(storage, dict):
        raise ValueError("Карта prepared-storage.json не содержит compressed_storage")
    for name in STORAGE_AUDITS:
        item = storage.get(name)
        if not isinstance(item, dict) or item.get("physical_storage_path") != name + ".gz":
            raise ValueError(f"Неполная карта хранения join-audit: {name}")
        contained(root, item["physical_storage_path"])
    return {"status": "REQUIRED", "files": [STORAGE_MAP, STORAGE_MANIFEST, STORAGE_CONTROL,
            "scripts/verify_logical_storage.py", *compressed],
            "expected_manifest_sha256": STORAGE_MANIFEST_SHA256,
            "scope": "Точная проверка SHA сжатых файлов и восстановленных байтов по прежнему manifest и независимому control audit."}


def snapshot_inputs(root: Path, expected: dict, manifests, guard=None, storage=None):
    observed = check_inputs(root, expected, guard)
    needed = [*manifests, "reports/v1.2/model.npz", "reports/v1.2/provenance.json",
              "reports/v1.2/strict-region-validation/strict_predictions.csv",
              "reports/v1.2/strict-region-validation/strict_assignments.csv",
              "reports/v1.2/strict-region-validation/strict_metrics.csv",
              *[f"reports/v1.2-added-value/{name}.csv" for name in ("models", "contrasts", "calibration", "predictions", "folds", "fold_state_diagnostics")],
              "reports/temporal-core-2026-10-03/paired-reference/assignments.csv",
              "reports/round2-cases/abazinsky.json", "docs/index.html",
              "web/research_template.html", "tests/browser/main_model_controls.cjs", "run_all.py"]
    # Record the exact implementation as well as inputs; do not call Git.
    needed += [p.relative_to(root).as_posix() for folder in ("sbercluster", "scripts", "web", "docs/assets")
               for p in (root / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    needed += ["tests/" + name.split(".")[0] + ".py" for name in (*SCIENCE_TESTS, *ATLAS_TESTS)]
    needed += [p.relative_to(root).as_posix() for p in (root / "reports/v1.2/strict-region-validation").rglob("fold_*.json")]
    needed += (storage or logical_storage_scope(root))["files"]
    for name in sorted(set(needed) - observed.keys()):
        observed[name] = fingerprint(contained(root, name), guard)
    return observed


def check_inputs(root: Path, expected: dict, guard=None):
    observed, problems = {}, []
    for relative, wanted in sorted(expected.items()):
        path = contained(root, relative)
        if not path.is_file():
            problems.append(f"Нет нужного файла: {relative}")
            continue
        observed[relative] = fingerprint(path, guard)
        if observed[relative] != wanted:
            problems.append(f"SHA256 не совпал: {relative}")
    if problems:
        raise ValueError("\n".join(problems))
    return observed


def prepare_runtime_copy(root: Path, output: Path, checked: dict, guard=None, timeout=3600, emit=None):
    """Copy bytes, not read-only metadata, so scientific temporary copytree can overwrite CSV."""
    started = time.monotonic()
    destination = output / "tmp" / "writable-project"
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    journal = output / "runtime-copy.json"
    state = {"status": "COPYING", "source_root": str(root), "runtime_root": str(destination),
             "excluded_root_directories": sorted(RUNTIME_EXCLUDED), "files": records}

    class CopyBudget:
        next_progress = started + 15

        def before_io(self):
            if time.monotonic() - started >= timeout:
                raise TimeoutError(f"Подготовка рабочей копии превысила {timeout:g} с")
            if guard:
                guard.before_io()
            if emit and time.monotonic() >= self.next_progress:
                emit(f"    Рабочая копия: {len(records)} файлов записано, {time.monotonic() - started:.0f} с; SHA и атрибуты проверяются.")
                self.next_progress = time.monotonic() + 15

    budget = CopyBudget()
    try:
        for directory, directories, files in os.walk(root, followlinks=False):
            budget.before_io()
            relative_dir = Path(directory).relative_to(root)
            directories[:] = sorted(name for name in directories if name != "__pycache__"
                                    and (relative_dir.parts or name not in RUNTIME_EXCLUDED)
                                    and (Path(directory) / name).resolve() != output.resolve())
            for name in directories + sorted(files):
                source = Path(directory) / name
                attributes = getattr(source.lstat(), "st_file_attributes", 0)
                if source.is_symlink() or attributes & 0x400:
                    raise ValueError(f"Рабочая копия не допускает ссылки и reparse points: {source}")
            for name in sorted(files):
                if name.endswith((".pyc", ".pyo")) or (not relative_dir.parts and name in RUNTIME_EXCLUDED):
                    continue
                source = Path(directory) / name
                relative = source.relative_to(root).as_posix()
                source = contained(root, relative)
                original_stat = source.stat()
                if not stat.S_ISREG(original_stat.st_mode):
                    raise ValueError(f"Рабочая копия требует обычный файл: {relative}")
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with source.open("rb") as reader, target.open("xb") as writer:
                    while True:
                        budget.before_io()
                        block = reader.read(32768)
                        if not block:
                            break
                        writer.write(block)
                        digest.update(block)
                        size += len(block)
                # Change only the new file; keep executable bits for portable copies.
                target.chmod(stat.S_IMODE(original_stat.st_mode) | stat.S_IRUSR | stat.S_IWUSR)
                sha = digest.hexdigest()
                records[relative] = {"sha256": sha, "bytes": size,
                                     "source_mode": original_stat.st_mode,
                                     "source_attributes": getattr(original_stat, "st_file_attributes", None)}
                if fingerprint(target, budget) != sha or (relative in checked and checked[relative] != sha):
                    raise ValueError(f"SHA256 рабочей копии не совпал: {relative}")
        missing = set(checked) - records.keys()
        if missing:
            raise ValueError("Рабочая копия пропустила проверенные входы: " + ", ".join(sorted(missing)))
        check_runtime_sources(root, destination, records, budget)
        state["status"] = "PASS_BYTE_IDENTICAL_WRITABLE_COPY"
        return destination, records
    except BaseException as error:
        state.update(status="FAIL", error=str(error))
        raise
    finally:
        state["seconds"] = round(time.monotonic() - started, 3)
        journal.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def check_runtime_sources(root: Path, runtime: Path, records: dict, guard=None):
    check_inputs(root, {name: row["sha256"] for name, row in records.items()}, guard)
    check_inputs(runtime, {name: row["sha256"] for name, row in records.items()}, guard)
    for name, row in records.items():
        current = contained(root, name).stat()
        if (current.st_mode != row["source_mode"] or
                getattr(current, "st_file_attributes", None) != row["source_attributes"]):
            raise ValueError(f"Атрибуты исходного файла изменились: {name}")


class DiskValue(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", ctypes.c_double)]


class DiskItem(ctypes.Structure):
    _fields_ = [("name", wintypes.LPWSTR), ("value", DiskValue)]


class LoadMonitor:
    """Read CPU, memory, disks and GPU; unavailable readings remain explicit."""
    def __init__(self, psutil):
        self.psutil, self.query = psutil, None
        self.gpu, self.gpu_command, self.gpu_error = None, None, None
        try:
            if os.name == "nt":
                from sbercluster.resources import NvidiaLoad
                self.gpu = NvidiaLoad()
            else:
                self.gpu_command = shutil.which("nvidia-smi")
                if not self.gpu_command:
                    raise RuntimeError("nvidia-smi не найден")
        except Exception as error:
            self.gpu_error = str(error)
        self.previous_time = time.monotonic()
        self.previous_disks = psutil.disk_io_counters(perdisk=True)
        psutil.cpu_percent()
        if os.name == "nt":
            self.pdh = ctypes.WinDLL("pdh")
            self.pdh.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
            self.pdh.PdhAddEnglishCounterW.argtypes = [wintypes.HANDLE, wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
            self.pdh.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
            self.pdh.PdhGetFormattedCounterArrayW.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            self.pdh.PdhCloseQuery.argtypes = [wintypes.HANDLE]
            self.query, self.counter = wintypes.HANDLE(), wintypes.HANDLE()
            try:
                self._check(self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
                self._check(self.pdh.PdhAddEnglishCounterW(self.query, r"\PhysicalDisk(*)\% Idle Time", 0, ctypes.byref(self.counter)))
                self._check(self.pdh.PdhCollectQueryData(self.query))
            except BaseException:
                self.close()
                raise

    def sample_gpu(self):
        try:
            if self.gpu is not None:
                values = self.gpu.sample()
                source = "sbercluster.resources.NvidiaLoad/NVML"
            elif self.gpu_command:
                result = subprocess.run([self.gpu_command,
                    "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=3, check=True, shell=False)
                values = []
                for line in result.stdout.splitlines():
                    use, used, total = map(float, line.split(","))
                    if total <= 0 or not 0 <= used <= total:
                        raise ValueError("Некорректное измерение VRAM")
                    values.append({"gpu_percent": use, "vram_percent": 100 * used / total})
                source = "nvidia-smi"
            else:
                raise RuntimeError(self.gpu_error or "Измерение GPU недоступно")
            if not values or any(not math.isfinite(v) or not 0 <= v <= 100
                                 for value in values for v in (value["gpu_percent"], value["vram_percent"])):
                raise ValueError("Некорректное измерение GPU или VRAM")
            return {"gpu_available": True, "gpu_source": source, "gpu_error": None, "gpus": values}
        except Exception as error:
            return {"gpu_available": False, "gpu_source": None, "gpu_error": str(error), "gpus": None}

    @staticmethod
    def _check(code):
        if code:
            raise RuntimeError(f"Не удалось измерить диск: Windows PDH {code & 0xffffffff:#x}")

    def sample(self):
        now, disks = time.monotonic(), {}
        if os.name == "nt":
            self._check(self.pdh.PdhCollectQueryData(self.query))
            size, count = wintypes.DWORD(), wintypes.DWORD()
            code = self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
            if code & 0xffffffff != 0x800007d2:
                raise RuntimeError("Не удалось прочитать счётчики физических дисков")
            buffer = ctypes.create_string_buffer(size.value)
            self._check(self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x200, ctypes.byref(size), ctypes.byref(count), buffer))
            items = ctypes.cast(buffer, ctypes.POINTER(DiskItem))
            for i in range(count.value):
                item = items[i]
                if item.name == "_Total":
                    continue
                if item.value.status not in (0, 1) or not math.isfinite(item.value.value):
                    raise RuntimeError("Показатель нагрузки диска недоступен")
                disks[item.name] = max(0.0, min(100.0, 100 - item.value.value))
        else:
            current = self.psutil.disk_io_counters(perdisk=True)
            for name, value in current.items():
                old = self.previous_disks.get(name)
                if old and hasattr(value, "busy_time"):
                    disks[name] = max(0.0, min(100.0, (value.busy_time - old.busy_time) / (now - self.previous_time) / 10))
            self.previous_disks = current
        self.previous_time = now
        memory = self.psutil.virtual_memory()
        if not disks:
            raise RuntimeError("Нагрузка физических дисков недоступна; безопасный запуск не подтверждён")
        return {"time_utc": utc_now(), "cpu_percent": self.psutil.cpu_percent(),
                "ram_percent": 100 * (memory.total - memory.available) / memory.total,
                "disk_active_percent": disks, **self.sample_gpu()}

    def close(self):
        if self.gpu is not None:
            self.gpu.close()
            self.gpu = None
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = None


class ResourceGuard:
    """Throttle only our tree, with a finite deadline instead of waiting for calm."""
    def __init__(self, stream, monitor=None, strict_gpu=False):
        import psutil
        self.psutil = psutil
        self.strict_gpu = strict_gpu
        self.gpu_unavailable_samples = 0
        self.gpu_errors = set()
        self.worker_io_samples = 0
        self.worker_io_unavailable_samples = 0
        self.worker_io_errors = set()
        self.monitor = monitor or LoadMonitor(psutil)
        self.stream, self.last, self.next_sample = stream, {}, 0.0
        self.cpu = psutil.Process().cpu_affinity()[-1]
        self.configure(psutil.Process(), idle=False)
        current = psutil.Process()
        self.stream.write(json.dumps({"event": "supervisor_limits", "pid": current.pid,
                                      "cpu_affinity": current.cpu_affinity(),
                                      "priority": current.nice(), "io_priority": str(current.ionice())}) + "\n")
        self.last_worker_sample = None
        time.sleep(0.15)

    def configure(self, process, idle=False):
        process.cpu_affinity([self.cpu])
        if os.name == "nt":
            process.nice(self.psutil.IDLE_PRIORITY_CLASS if idle else self.psutil.BELOW_NORMAL_PRIORITY_CLASS)
            process.ionice(self.psutil.IOPRIO_VERYLOW if idle else self.psutil.IOPRIO_LOW)
        else:
            process.nice(max(process.nice(), 19 if idle else 10))
            if hasattr(process, "ionice"):
                process.ionice(self.psutil.IOPRIO_CLASS_IDLE if idle else self.psutil.IOPRIO_CLASS_BE, value=0 if idle else 7)

    def sample(self):
        if time.monotonic() >= self.next_sample:
            self.last = self.monitor.sample()
            if not self.last.get("gpu_available") or not self.last.get("gpus"):
                self.gpu_unavailable_samples += 1
                self.gpu_errors.add(self.last.get("gpu_error") or "нет показаний GPU")
            self.stream.write(json.dumps(self.last) + "\n")
            self.stream.flush()
            self.next_sample = time.monotonic() + 1
        return self.last

    def high(self):
        sample = self.sample()
        gpu_known = bool(sample.get("gpu_available") and sample.get("gpus"))
        if not gpu_known and self.strict_gpu:
            raise RuntimeError(f"Нагрузка GPU неизвестна: {sample.get('gpu_error', 'нет показаний')}; безопасный запуск не подтверждён")
        values = [sample["cpu_percent"], sample["ram_percent"], *sample["disk_active_percent"].values()]
        for gpu in sample["gpus"] if gpu_known else []:
            values.extend((gpu["gpu_percent"], gpu["vram_percent"]))
        return any(value >= 95 for value in values)

    def coverage(self):
        known = bool(self.last.get("gpu_available") and self.last.get("gpus"))
        missing = getattr(self, "gpu_unavailable_samples", 0)
        io_samples = getattr(self, "worker_io_samples", 0)
        io_missing = getattr(self, "worker_io_unavailable_samples", 0)
        io_status = "PARTIAL" if io_missing else ("COMPLETE" if io_samples else "UNVERIFIED")
        return {"status": "COMPLETE" if known and missing == 0 and io_status == "COMPLETE" else "PARTIAL", "gpu_available": known,
                "gpu_source": self.last.get("gpu_source"), "gpu_error": self.last.get("gpu_error"),
                "gpus": self.last.get("gpus"), "strict_gpu_required": self.strict_gpu,
                "gpu_unavailable_samples": missing, "gpu_errors": sorted(getattr(self, "gpu_errors", set())),
                "worker_io": {"status": io_status, "samples": io_samples, "unavailable_samples": io_missing,
                              "errors": sorted(getattr(self, "worker_io_errors", set()))},
                "scope": "CPU, RAM and every physical disk; GPU/VRAM and sampled worker IO only when readings are available"}

    def worker_io(self, process):
        # Per-worker counters can be denied while mandatory system load and limits remain available.
        try:
            counters = process.io_counters()._asdict()
        except self.psutil.AccessDenied as error:
            message = f"psutil.AccessDenied reading io_counters (pid={process.pid}): {error}"
            if error.__context__ is not None:
                message += "; " + str(error.__context__)
            self.worker_io_unavailable_samples += 1
            self.worker_io_errors.add(message)
            self.worker_io_samples += 1
            return {"io_available": False, "io_bytes": None, "io_error": message}
        self.worker_io_samples += 1
        return {"io_available": True, "io_bytes": counters, "io_error": None}

    def before_io(self):
        if self.high():
            self.configure(self.psutil.Process(), idle=True)
            time.sleep(0.1)

    def tick(self, popen):
        try:
            parent = self.psutil.Process(popen.pid)
            children = parent.children(recursive=True)
            high = self.high()
            tree = [parent, *children]
            for process in tree:
                try:
                    self.configure(process, idle=high)
                except self.psutil.NoSuchProcess:
                    pass
            if self.last.get("time_utc") != self.last_worker_sample:
                self.last_worker_sample = self.last.get("time_utc")
                for process in tree:
                    try:
                        self.stream.write(json.dumps({"event": "worker_limits", "time_utc": utc_now(),
                            "pid": process.pid, "cpu_affinity": process.cpu_affinity(),
                            "priority": process.nice(), "io_priority": str(process.ionice()),
                            "os_threads": process.num_threads(), **self.worker_io(process),
                            "throttled": high}) + "\n")
                    except self.psutil.NoSuchProcess:
                        pass
                self.stream.flush()
            if high:
                paused = []
                try:
                    for process in reversed(tree):
                        try:
                            process.suspend()
                            paused.append(process)
                        except self.psutil.NoSuchProcess:
                            pass
                    # One tenth of a second of work per second under sustained load.
                    time.sleep(0.9)
                finally:
                    for process in paused:
                        try:
                            process.resume()
                        except self.psutil.NoSuchProcess:
                            pass
        except self.psutil.NoSuchProcess:
            pass

    def close(self):
        self.monitor.close()


@dataclass(frozen=True)
class Phase:
    key: str
    title: str
    argv: tuple[str, ...]


def phases(root: Path, node: str, quick: bool, storage=None):
    py = (sys.executable, "-X", "utf8", "-B")
    result = [Phase("node_version", "Проверка Node.js", (node, "--version"))]
    scope = storage or logical_storage_scope(root)
    if scope["status"] == "REQUIRED":
        result.append(Phase("logical_storage", "Сжатые аудиты: точные физические SHA и SHA восстановленных байтов",
            (*py, "-m", "scripts.verify_logical_storage", "--manifest", STORAGE_MANIFEST,
             "--storage-map", STORAGE_MAP, "--storage-root", ".",
             "--expected-manifest-sha256", STORAGE_MANIFEST_SHA256, "--control-audit", STORAGE_CONTROL)))
    if not quick:
        result.extend([
            Phase("findings", "Научные выводы и все 73 исключаемых региона", (*py, "-m", "scripts.validate_v12_findings", "--check")),
            Phase("added_value", "Добавочная ценность групп: все регионы и исходные допуски", (*py, "-m", "scripts.check_added_value", "--check")),
            Phase("science_units", "Математика, научные таблицы и происхождение данных", (*py, "-c", UNIT_RUNNER, *SCIENCE_TESTS)),
        ])
    result.extend([
        Phase("atlas_units", "Пакет атласа, сохранённые входы и сценарии ошибок", (*py, "-c", UNIT_RUNNER, *(QUICK_TESTS if quick else ATLAS_TESTS))),
        Phase("node_controls", "Переключатели основной и архивной модели в Node (поддельный DOM)", (node, str(root / "tests/browser/main_model_controls.cjs"), str(root))),
    ])
    return result


def stop_tree(process):
    import psutil
    try:
        descendants = psutil.Process(process.pid).children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for child in reversed(descendants):
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.kill()
    process.wait()


class PhaseProcessGroup:
    """Own descendants even after their original parent exits."""
    def __init__(self):
        self.handle, self.pid = None, None
        if os.name == "nt":
            from sbercluster.resources import ExtendedLimits
            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
            self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
            self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateJobObjectW(None, None)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = 0x2000  # KILL_ON_JOB_CLOSE, inherited by descendants.
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                self.close()
                raise ctypes.WinError(ctypes.get_last_error())

    def attach(self, process):
        self.pid = process.pid
        if os.name == "nt":
            if not self.kernel.AssignProcessToJobObject(self.handle, wintypes.HANDLE(int(process._handle))):
                raise ctypes.WinError(ctypes.get_last_error())
            from sbercluster.resources import resume_initial_thread
            resume_initial_thread(process)

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
        elif os.name != "nt" and self.pid is not None:
            try:
                os.killpg(self.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.pid = None


def run_phase(phase: Phase, root: Path, env: dict, log, guard, timeout: float, emit):
    started, process, group = time.monotonic(), None, None
    log.write(("\n=== " + phase.key + " " + json.dumps(phase.argv, ensure_ascii=False) + " ===\n").encode("utf-8"))
    log.flush()
    flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW | 0x4 if os.name == "nt" else 0
    try:
        group = PhaseProcessGroup()
        process = subprocess.Popen(list(phase.argv), cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, shell=False, creationflags=flags,
                                   start_new_session=os.name != "nt")
        group.attach(process)
        heartbeat = started + 15
        while process.poll() is None:
            if time.monotonic() - started > timeout:
                group.close()
                stop_tree(process)
                raise TimeoutError(f"Этап превысил {timeout:g} с. При высокой нагрузке повторите позднее.")
            guard.tick(process)
            time.sleep(0.1)
            if time.monotonic() >= heartbeat:
                emit(f"    Выполняется {time.monotonic() - started:.0f} с; полный вывод сохраняется в run.log")
                heartbeat = time.monotonic() + 15
        result = {"key": phase.key, "title": phase.title, "argv": list(phase.argv),
                  "status": "PASS" if process.returncode == 0 else "FAIL",
                  "returncode": process.returncode, "seconds": round(time.monotonic() - started, 3)}
        if phase.key == "node_version" and process.returncode == 0:
            log.flush()
            # The version command is the first process in a fresh log.
            text = Path(log.name).read_text(encoding="utf-8", errors="replace")
            version = re.search(r"^v(\d+)\.\d+\.\d+\s*$", text, re.M)
            if not version or int(version.group(1)) < 22:
                result.update(status="FAIL", error="Нужен Node.js 22 LTS или новее; установите его с nodejs.org.")
        return result
    except Exception as error:
        log.write((traceback.format_exc() + "\n").encode("utf-8"))
        log.flush()
        return {"key": phase.key, "title": phase.title, "argv": list(phase.argv), "status": "FAIL",
                "error": str(error), "seconds": round(time.monotonic() - started, 3)}
    finally:
        if group is not None:
            group.close()
        if process is not None and process.poll() is None:
            stop_tree(process)


def execute(phases_to_run, root: Path, output: Path, env: dict, guard, summary: dict, emit=print, timeout=3600, runtime=None):
    """Aggregate every selected phase; a failed phase never becomes a complete PASS."""
    def save():
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with (output / "run.log").open("ab") as log:
        for index, phase in enumerate(phases_to_run, 1):
            emit(f"[{index}/{len(phases_to_run)}] {phase.title}")
            phase_root, phase_env = root, env
            uses_copy = runtime is not None and phase.key in RUNTIME_PHASES
            if uses_copy:
                phase_root = runtime[0]
                phase_env = dict(env, PYTHONPATH=os.pathsep.join((str(phase_root / "tests"), str(phase_root))))
            execution = {"cwd": str(phase_root), "pythonpath": phase_env.get("PYTHONPATH"),
                         "source_root": str(root), "writable_runtime_copy": uses_copy}
            log.write(("Execution: " + json.dumps(execution, ensure_ascii=False) + "\n").encode("utf-8"))
            result = None
            if uses_copy:
                try:
                    check_runtime_sources(root, runtime[0], runtime[1], guard)
                except Exception as error:
                    result = {"key": phase.key, "title": phase.title, "argv": list(phase.argv),
                              "status": "FAIL", "seconds": 0, "runtime_source_integrity": "FAIL",
                              "error": str(error), "process_started": False}
            if result is None:
                result = run_phase(phase, phase_root, phase_env, log, guard, timeout, emit)
            result.update(execution)
            if uses_copy and result.get("runtime_source_integrity") != "FAIL":
                try:
                    check_runtime_sources(root, runtime[0], runtime[1], guard)
                    result["runtime_source_integrity"] = "PASS"
                except Exception as error:
                    result.update(status="FAIL", runtime_source_integrity="FAIL", error=str(error))
            summary["phases"].append(result)
            emit(f"    {result['status']}, {result['seconds']:.1f} с" + (f": {result['error']}" if "error" in result else ""))
            save()
            if phase.key in ("node_version", "logical_storage") and result["status"] != "PASS":
                break
            if result.get("runtime_source_integrity") == "FAIL":
                break
    failed = any(p["status"] != "PASS" for p in summary["phases"])
    summary["status"] = "FAIL" if failed else ("UNVERIFIED" if summary["mode"] == "quick" else "PASS")
    save()
    return 1 if failed else (2 if summary["mode"] == "quick" else 0)


def output_directory(root: Path, selected: Path | None):
    base = (selected or root / "artifacts/run-all").resolve()
    if base.is_relative_to(root.resolve()):
        relative = base.relative_to(root.resolve())
        if relative.parts and relative.parts[0] not in ("artifacts", "runs", ".local"):
            raise ValueError("Внутри проекта --output должен находиться в artifacts, runs или .local, чтобы не затронуть входы")
        if not relative.parts:
            raise ValueError("Корень проекта нельзя использовать как --output")
    output = base / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    (output / "tmp").mkdir()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Одна команда для проверки сохранённых научных результатов и поставки атласа.")
    parser.add_argument("--quick", action="store_true", help="только быстрые технические проверки; итог UNVERIFIED и код 2")
    parser.add_argument("--output", type=Path, help="родительский каталог для нового каталога с логами")
    parser.add_argument("--phase-timeout", type=float, default=3600, help="максимум секунд на каждый процесс, включая замедление при нагрузке (3600)")
    parser.add_argument("--thread-env", action="append", default=[], metavar="NAME=1", help="явно передать одно из ограничений вычислительных потоков")
    parser.add_argument("--strict-gpu-monitoring", action="store_true", help="требовать реальные показания GPU и VRAM; при недоступности ошибка")
    args = parser.parse_args(argv)
    if not math.isfinite(args.phase_timeout) or args.phase_timeout <= 0:
        parser.error("--phase-timeout должен быть конечным положительным числом")
    env = thread_environment(args.thread_env)
    output = output_directory(ROOT, args.output)
    (output / "run.log").touch(exist_ok=False)

    def report(text):
        print(text, flush=True)
        with (output / "run.log").open("ab") as log:
            log.write((text + "\n").encode("utf-8"))

    env.update(TMP=str(output / "tmp"), TEMP=str(output / "tmp"), TMPDIR=str(output / "tmp"),
               PYTHONPATH=os.pathsep.join((str(ROOT / "tests"), str(ROOT))))
    summary = {"status": "RUNNING", "mode": "quick" if args.quick else "full",
               "started_at_utc": utc_now(), "root": str(ROOT), "output": str(output),
               "phases": [], "limitations": LIMITATIONS,
               "resources": {"compute_threads": 1, "cpu_affinity_count": 1,
                             "normal_priority": "BelowNormal on Windows; nice >=10 on Linux",
                             "high_load_policy": ">=95% CPU, RAM, any disk, GPU or VRAM: Idle/low IO and 90% pause of own process tree",
                             "gpu_unavailable_policy": "Record null and partial resource coverage; fail only with --strict-gpu-monitoring",
                             "strict_gpu_required": args.strict_gpu_monitoring,
                             "thread_environment": {k: env[k] for k in THREAD_ENV},
                             "phase_timeout_seconds": args.phase_timeout}}
    guard, resource_log, observed, runtime = None, None, {}, None
    started, code = time.monotonic(), 1
    report("Быстрая техническая проверка. Научное воспроизведение останется UNVERIFIED." if args.quick else "Полная проверка главных научных результатов и поставки атласа.")
    report(f"Логи: {output}")
    try:
        summary["environment"] = dependencies(args.quick)
        resource_log = (output / "resources.jsonl").open("w", encoding="utf-8")
        guard = ResourceGuard(resource_log, strict_gpu=args.strict_gpu_monitoring)
        report("Проверка нужных файлов и SHA256 из сохранённого происхождения данных…")
        expected, manifests, historical = provenance_inputs(ROOT)
        summary["historical_provenance"] = historical
        if historical["historical_code_sha256"]:
            report("    Старый SHA joint.py относится к историческому рецепту c7905a3; текущий код проверяется по findings_provenance.json.")
        storage = logical_storage_scope(ROOT)
        summary["logical_storage"] = storage
        report("    " + storage["scope"])
        observed = snapshot_inputs(ROOT, expected, manifests, guard, storage=storage)
        summary["checked_files_sha256"] = observed
        summary["stored_sha256_count"] = len(expected)
        summary["preflight"] = {"status": "PASS", "seconds": round(time.monotonic() - started, 3)}
        report(f"    PASS: {len(expected)} сохранённых SHA256; версия {len(observed)} файлов записана, {summary['preflight']['seconds']:.1f} с.")
        plan = phases(ROOT, summary["environment"]["node_executable"], args.quick, storage=storage)
        if any(phase.key in RUNTIME_PHASES for phase in plan):
            report("Подготовка побайтно проверенной рабочей копии с правом записи для Python-проверок…")
            runtime = prepare_runtime_copy(ROOT, output, observed, guard, timeout=args.phase_timeout, emit=report)
            summary["runtime_copy"] = {"status": "PASS_BYTE_IDENTICAL_WRITABLE_COPY", "root": str(runtime[0]),
                                       "source_manifest": str(output / "runtime-copy.json"), "files": len(runtime[1]),
                                       "seconds": json.loads((output / "runtime-copy.json").read_text(encoding="utf-8"))["seconds"]}
            report(f"    PASS: {len(runtime[1])} файлов; все исходные байты и атрибуты сохранены.")
        code = execute(plan, ROOT, output, env, guard, summary, emit=report,
                       timeout=args.phase_timeout, runtime=runtime)
    except KeyboardInterrupt:
        summary.update(status="INTERRUPTED", error="Проверка остановлена пользователем")
        code = 130
    except Exception as error:
        summary.update(status="FAIL", error=str(error))
        if "preflight" not in summary:
            summary["preflight"] = {"status": "FAIL", "seconds": round(time.monotonic() - started, 3)}
        report(f"Ошибка: {error}\n{traceback.format_exc()}")
        code = 1
    finally:
        if observed:
            integrity_started = time.monotonic()
            try:
                report("Проверка неизменности проверенных файлов…")
                check_inputs(ROOT, observed, guard)
                if runtime is not None:
                    check_runtime_sources(ROOT, runtime[0], runtime[1], guard)
                summary["checked_files_unchanged"] = True
                summary["input_integrity"] = {"status": "PASS", "seconds": round(time.monotonic() - integrity_started, 3)}
                report(f"    PASS: проверенные файлы прежние, {summary['input_integrity']['seconds']:.1f} с.")
            except Exception as error:
                summary.update(status="FAIL", checked_files_unchanged=False, integrity_error=str(error))
                summary["input_integrity"] = {"status": "FAIL", "seconds": round(time.monotonic() - integrity_started, 3)}
                report(f"    FAIL: {error}")
                code = 1
        if guard:
            if hasattr(guard, "coverage"):
                summary["resource_coverage"] = guard.coverage()
            guard.close()
        if resource_log:
            resource_log.close()
        summary.update(exit_code=code, finished_at_utc=utc_now(), seconds=round(time.monotonic() - started, 3))
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report(f"\nРезультат: {summary['status']}. Продолжительность: {summary['seconds']:.1f} с.")
        if code == 0:
            report("Главные научные результаты воспроизведены в пределах исходных допусков; технические этапы прошли.")
        elif code == 2:
            report("Технические этапы прошли. Для полного научного результата запустите python run_all.py.")
        else:
            report("Проверка не завершилась успешно; причина и полный вывод сохранены в каталоге логов.")
        report(f"Полный лог: {output / 'run.log'}\nИтог JSON: {output / 'summary.json'}")
        if summary.get("resource_coverage", {}).get("status") == "PARTIAL":
            coverage = summary["resource_coverage"]
            reasons = []
            if not coverage.get("gpu_available") or coverage.get("gpu_unavailable_samples", 0):
                reason = "; ".join(coverage.get("gpu_errors", [])) or coverage.get("gpu_error") or "нет показаний"
                reasons.append("Не все показания GPU и VRAM доступны: " + reason)
            worker_io = coverage.get("worker_io", {})
            if worker_io.get("status") == "PARTIAL":
                reasons.append("Не все показания IO отдельных процессов доступны: " + "; ".join(worker_io["errors"]))
            elif worker_io.get("status") == "UNVERIFIED":
                reasons.append("Показания IO отдельных процессов не получены")
            report("Контроль ресурсов: PARTIAL. " + "; ".join(reasons))
        for note in LIMITATIONS:
            report(note)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
