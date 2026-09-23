"""One Windows CPU experiment with hard CPU/memory limits and sampled load monitoring."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(config):
    # Nothing beyond the standard library is imported until the parent has installed the limits.
    if sys.stdin.readline().strip() != 'run':
        raise RuntimeError('No start signal from resource supervisor')
    from sbercluster.resources import verify_worker_job
    verify_worker_job(os.environ.get('SBERCLUSTER_JOB_NAME'))
    from sbercluster.cli import main
    main(['run', '--root', str(ROOT), '--config', config, '--execute-clustering'])


def supervise(config, timeout):
    import psutil
    from sbercluster.resources import SystemLoad, WindowsJob, limit_breaches, resume_initial_thread
    if os.name != 'nt':
        raise RuntimeError('This supervisor requires Windows')
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('Timeout must be a finite positive number of seconds')
    config_path = (ROOT / config).resolve()
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    if cfg['execution'].get('allow_clustering') is not True:
        raise PermissionError('The selected experiment configuration is disabled')
    expected_profile = {'cpu_hard_cap_percent': 20, 'memory_limit_bytes': 1024**3, 'gpu': False,
                        'system_monitor_ceiling_percent': 70, 'sample_interval_seconds': 1}
    if cfg['execution'].get('resource_profile') != expected_profile:
        raise ValueError('The configuration must explicitly match the supported bounded resource profile')
    log_dir = ROOT / 'artifacts/resource_runs' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    log_dir.mkdir(parents=True)
    report = {'status': 'preflight', 'config': cfg, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'monitor_ceiling_percent': 70, 'target_monitor_interval_seconds': 1,
              'gpu_used_by_training': False, 'disk_limit_type': 'sampled stop; no instantaneous hard cap',
              'timeout_seconds': timeout, 'samples': [],
              'worker_duty_cycle': {'active_seconds': .15, 'pause_seconds': .35}}
    child = job = monitor = None
    started = time.monotonic()
    try:
        monitor = SystemLoad()
        time.sleep(1)
        baseline = monitor.sample()
        report['samples'].append({'seconds': 0, **baseline})
        if limit_breaches(baseline):
            report.update(status='blocked_by_existing_load', breaches=limit_breaches(baseline))
            return 2
        job = WindowsJob(cpu_percent=20, memory_bytes=1024**3)
        report.update(job.settings)
        env = {**os.environ, 'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
               'MKL_NUM_THREADS': '2', 'NUMEXPR_NUM_THREADS': '2', 'POLARS_MAX_THREADS': '2',
               'CUDA_VISIBLE_DEVICES': '-1', 'PYTHONUNBUFFERED': '1', 'SBERCLUSTER_JOB_NAME': job.name}
        with (log_dir / 'worker.log').open('w', encoding='utf-8') as output:
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                      '--worker', '--config', str(config_path)], cwd=ROOT,
                                     env=env, stdin=subprocess.PIPE, stdout=output, stderr=subprocess.STDOUT,
                                     creationflags=subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x4)
            job.assign(child)
            process = psutil.Process(child.pid)
            affinity = process.cpu_affinity()[-2:]
            process.cpu_affinity(affinity)
            report['cpu_affinity'] = affinity
            report['worker_pid'] = child.pid
            process.cpu_percent()
            resume_initial_thread(child)
            child.stdin.write(b'run\n')
            child.stdin.close()
            report['status'] = 'running'
            print(json.dumps({'status': 'running', 'log_dir': str(log_dir), **job.settings}), flush=True)
            last_sample = time.monotonic()
            next_sample = last_sample + 1
            while True:
                # Short compute bursts also spread synchronous import/read activity on HDDs.
                # This is pacing, not a hard disk-I/O cap; measured breaches still stop the job.
                time.sleep(.15)
                paused = []
                if child.poll() is None:
                    try:
                        process.suspend()
                        paused.append(process)
                        for member in process.children(recursive=True):
                            try:
                                member.suspend()
                                paused.append(member)
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                time.sleep(.35)
                if time.monotonic() < next_sample and child.poll() is None:
                    for member in reversed(paused):
                        try:
                            member.resume()
                        except psutil.NoSuchProcess:
                            pass
                    continue
                sample = monitor.sample()
                now = time.monotonic()
                sample['seconds'] = now - started
                sample['sample_gap_seconds'] = now - last_sample
                last_sample = now
                next_sample += 1
                try:
                    members = [process] + process.children(recursive=True)
                    sample['worker_tree_rss_bytes'] = sum(member.memory_info().rss for member in members)
                except psutil.NoSuchProcess:
                    pass
                report['samples'].append(sample)
                if sample['sample_gap_seconds'] > 1.5:
                    report['status'] = 'stopped_monitor_delay'
                    job.close()
                    child.wait(timeout=10)
                    return 6
                breaches = limit_breaches(sample)
                if breaches:
                    report.update(status='stopped_resource_limit', breaches=breaches)
                    job.close()
                    child.wait(timeout=10)
                    return 3
                if child.poll() is not None:
                    report.update(status='completed' if child.returncode == 0 else 'worker_failed',
                                  worker_exit_code=child.returncode)
                    return 0 if child.returncode == 0 else 4
                if time.monotonic() - started > timeout:
                    report['status'] = 'stopped_timeout'
                    job.close()
                    child.wait(timeout=10)
                    return 5
                for member in reversed(paused):
                    try:
                        member.resume()
                    except psutil.NoSuchProcess:
                        pass
    except BaseException as exc:
        report.update(status='supervisor_failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if job:
            job.close()
        if child and child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        if monitor:
            monitor.close()
        report['elapsed_seconds'] = time.monotonic() - started
        (log_dir / 'resources.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'status': report['status'], 'report': str(log_dir / 'resources.json')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--timeout', type=float, default=300)
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker:
        worker(args.config)
    else:
        sys.exit(supervise(args.config, args.timeout))
