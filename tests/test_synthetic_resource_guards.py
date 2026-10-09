"""Engineering guard checks only: no GPU, scientific inputs or optimizer fits."""
import threading
import unittest
from unittest.mock import Mock, patch

from scripts import reproduce_synthetic as reproduce
from scripts.science_portable_resources import gpu_number, gpu_over_limit


class FakeOriginalMonitor:
    def __init__(self, output, heavy=False):
        self.stop = threading.Event()
        self.heavy = heavy
        self.gpu_command = None
        self.errors = set()
        self.original_gate_calls = 0

    def gate(self):
        self.original_gate_calls += 1


class SyntheticResourceGuardsTest(unittest.TestCase):
    def test_portable_vram_and_utilization_boundaries(self):
        self.assertTrue(gpu_over_limit({"0": {"utilization": 0., "memory_fraction": .95}}))
        self.assertTrue(gpu_over_limit({"0": {"utilization": None, "memory_fraction": .96}}))
        self.assertTrue(gpu_over_limit({"0": {"utilization": 95., "memory_fraction": None}}))
        self.assertFalse(gpu_over_limit({"0": {"utilization": 94.9, "memory_fraction": .94999}}))
        self.assertFalse(gpu_over_limit({"0": {"utilization": None, "memory_fraction": None}}))

    def test_unknown_driver_readings_remain_unknown(self):
        for value in ("N/A", "nan", "-1", None):
            self.assertIsNone(gpu_number(value))
        self.assertEqual(gpu_number("0"), 0.)
        with patch.object(reproduce.subprocess, "check_output", return_value="0, N/A, 100\n"):
            values, error = reproduce.probe_vram("fake-nvidia-smi")
        self.assertEqual(values, {"0": None})
        self.assertEqual(error, "vram_measurement_partial")
        self.assertEqual(reproduce.probe_vram(None), ({}, "nvidia_smi_unavailable"))

    def test_windows_vram_boundary(self):
        with patch.object(reproduce.subprocess, "check_output", return_value="0, 95, 100\n"):
            values, error = reproduce.probe_vram("fake-nvidia-smi")
        self.assertIsNone(error)
        self.assertTrue(reproduce.vram_over_limit(values))
        self.assertFalse(reproduce.vram_over_limit({"0": .94999}))
        self.assertFalse(reproduce.vram_over_limit({"0": None}))

    def test_windows_wrapper_retains_original_gate_and_waits_for_vram(self):
        monitor = reproduce.windows_monitor_with_vram(FakeOriginalMonitor)("unused.jsonl", heavy=True)
        monitor.vram_thread = Mock()
        monitor.vram_thread.is_alive.return_value = True
        monitor.stop = Mock()
        monitor.stop.wait.side_effect = lambda seconds: monitor.vram_blocked.clear()
        monitor.vram_blocked.set()
        with patch("builtins.print"):
            monitor.gate()
        monitor.stop.wait.assert_called_once_with(5)
        self.assertEqual(monitor.original_gate_calls, 2)
        monitor.gate()
        self.assertEqual(monitor.original_gate_calls, 3)
        self.assertEqual(monitor.stop.wait.call_count, 1)

    def test_windows_wrapper_fails_closed_after_thread_failure(self):
        monitor = reproduce.windows_monitor_with_vram(FakeOriginalMonitor)("unused.jsonl", heavy=True)
        monitor.vram_thread = Mock()
        monitor.vram_thread.is_alive.return_value = False
        with self.assertRaises(RuntimeError):
            monitor.gate()


if __name__ == "__main__":
    unittest.main()
