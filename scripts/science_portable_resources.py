"""Explicit partial Linux/macOS resource adapter for the unchanged science recipe.

Windows reproduction retains the original PDH monitor. This adapter reports
unsupported measurements as unavailable, never as zero utilization.
"""
from __future__ import annotations
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import psutil


def gpu_number(value):
    """Unavailable driver readings remain unknown rather than zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def gpu_over_limit(gpu):
    return any((item.get("utilization") is not None and item["utilization"] >= 95)
               or (item.get("memory_fraction") is not None and item["memory_fraction"] >= .95)
               for item in gpu.values())


class PortableResourceMonitor:
    def __init__(self, output, heavy=False):
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.heavy = heavy
        self.process = psutil.Process()
        self.samples = 0
        self.errors = set()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.blocked = threading.Event()
        self.io_busy = threading.Event()
        self.gpu_command = shutil.which("nvidia-smi")
        self.affinity = None
        try:
            affinity = self.process.cpu_affinity()
            self.process.cpu_affinity([affinity[0]])
            self.affinity = self.process.cpu_affinity()
        except (AttributeError, OSError, psutil.Error) as exc:
            self.errors.add("affinity_unavailable:" + type(exc).__name__)
        try:
            self.process.nice(10)
        except (OSError, psutil.Error) as exc:
            self.errors.add("nice_unavailable:" + type(exc).__name__)
        self.previous_disks = psutil.disk_io_counters(perdisk=True) or {}
        self.previous_time = time.monotonic()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def disk_sample(self):
        now = time.monotonic()
        current = psutil.disk_io_counters(perdisk=True) or {}
        elapsed = now - self.previous_time
        values = {}
        for name, counter in current.items():
            previous = self.previous_disks.get(name)
            if previous is not None and hasattr(counter, "busy_time") and elapsed > 0:
                delta = counter.busy_time - previous.busy_time
                if delta >= 0:
                    values[name] = min(100., 100. * delta / (1000. * elapsed))
        self.previous_time, self.previous_disks = now, current
        return values

    def loop(self):
        psutil.cpu_percent(interval=None)
        self.stop.wait(1)
        try:
            with self.output.open("a", encoding="utf-8") as handle:
                while not self.stop.is_set():
                    disk = self.disk_sample()
                    gpu, gpu_ok = {}, False
                    if self.gpu_command:
                        try:
                            raw = subprocess.check_output([self.gpu_command, "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=3, stderr=subprocess.DEVNULL)
                            for line in raw.splitlines():
                                index, use, used, total = line.split(",")
                                utilization, occupied, capacity = gpu_number(use), gpu_number(used), gpu_number(total)
                                fraction = occupied / capacity if occupied is not None and capacity is not None and capacity > 0 else None
                                gpu[index.strip()] = {"utilization": utilization, "memory_fraction": fraction,
                                                      "utilization_available": utilization is not None,
                                                      "memory_available": fraction is not None}
                            gpu_ok = bool(gpu) and all(item["utilization_available"] and item["memory_available"] for item in gpu.values())
                            if gpu and not gpu_ok:
                                self.errors.add("gpu_measurements_partial")
                        except Exception as exc:
                            self.errors.add("gpu_probe_unverified:" + type(exc).__name__)
                    cpu, memory = psutil.cpu_percent(interval=None), psutil.virtual_memory().percent
                    busy = memory >= 95 or (self.heavy and (cpu >= 95 or gpu_over_limit(gpu)))
                    (self.blocked.set if busy else self.blocked.clear)()
                    disk_busy = any(value >= 95 for value in disk.values())
                    (self.io_busy.set if disk_busy else self.io_busy.clear)()
                    row = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "adapter": "portable_psutil_partial_v2", "cpu_percent": cpu, "ram_percent": memory,
                           "disk_percent": disk, "disk_available": bool(disk),
                           "disk_measurement": "device busy_time delta; devices may be logical and overlapping" if disk else "unavailable",
                           "gpu": gpu, "gpu_available": bool(self.gpu_command), "gpu_probe_verified": gpu_ok,
                           "gpu_memory_measurement_verified": bool(gpu) and all(item["memory_available"] for item in gpu.values()),
                           "compute_paused": busy, "disk_backoff": disk_busy,
                           "priority_actual": self.process.nice(), "affinity": self.affinity,
                           "heavy_compute": self.heavy, "rss": self.process.memory_info().rss,
                           "scope": "CPU/RAM verified; optional disk/GPU measurements; not the Windows PDH guard"}
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    self.samples += 1
                    self.ready.set()
                    self.stop.wait(5)
        except Exception as exc:
            self.errors.add("monitor_failure:" + type(exc).__name__)
            self.blocked.set()
            self.ready.set()

    def start(self):
        self.thread.start()
        if not self.ready.wait(10) or self.samples == 0:
            raise RuntimeError("Portable monitor did not produce verified CPU/RAM sample")
        self.gate()

    def gate(self):
        if not self.thread.is_alive():
            raise RuntimeError("Portable resource monitor stopped unexpectedly")
        while self.blocked.is_set():
            if not self.thread.is_alive():
                raise RuntimeError("Portable resource monitor stopped unexpectedly")
            print("portable resource pause: CPU/RAM or measured GPU utilization/VRAM >=95%", flush=True)
            self.stop.wait(5)
        if self.io_busy.is_set():
            self.stop.wait(1)

    def close(self):
        self.stop.set()
        self.thread.join(5)

