#!/usr/bin/env python3
"""Bounded 300-second CPU/GPU/RAM/disk burst qualification for the R11 Docker runner."""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DURATION = int(os.environ.get("CB16_BURST_SECONDS", "300"))
SAMPLE_SECONDS = 5
GPU_ABORT_C = 88
MIN_FREE_DISK = 2 * 1024**3
MIN_MEM_AVAILABLE = 1024**3
DISK_BYTES = 512 * 1024**2
CHUNK = 8 * 1024**2


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def meminfo() -> dict[str, int]:
    out: dict[str, int] = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, val = line.split(':', 1)
        parts = val.strip().split()
        if parts:
            out[key] = int(parts[0]) * 1024
    return out


def cgroup_memory_events() -> dict[str, int]:
    for path in (Path('/sys/fs/cgroup/memory.events'), Path('/sys/fs/cgroup/memory/memory.failcnt')):
        if not path.exists():
            continue
        try:
            if path.name == 'memory.failcnt':
                return {'failcnt': int(path.read_text().strip())}
            rows = {}
            for line in path.read_text().splitlines():
                k, v = line.split()
                rows[k] = int(v)
            return rows
        except Exception:
            pass
    return {}


def cpu_worker(stop: mp.Event, counter: mp.Value, seed: int) -> None:
    block = bytearray(CHUNK)
    for i in range(0, len(block), 4096):
        block[i] = (seed + i) & 0xFF
    local = 0
    while not stop.is_set():
        hashlib.sha256(block).digest()
        local += 1
        if local % 16 == 0:
            with counter.get_lock():
                counter.value += 16
            local = 0
    if local:
        with counter.get_lock():
            counter.value += local


def gpu_worker(stop: threading.Event, state: dict, lock: threading.Lock) -> None:
    try:
        import torch
        assert torch.cuda.is_available(), 'CUDA_UNAVAILABLE'
        device = torch.device('cuda:0')
        a = torch.randn((3072, 3072), device=device, dtype=torch.float32)
        b = torch.randn((3072, 3072), device=device, dtype=torch.float32)
        n = 0
        while not stop.is_set():
            c = torch.mm(a, b)
            _ = c[0, 0].item()
            n += 1
            if n % 2 == 0:
                torch.cuda.synchronize()
                with lock:
                    state['iterations'] = n
                    state['allocated_bytes'] = int(torch.cuda.memory_allocated())
                    state['reserved_bytes'] = int(torch.cuda.memory_reserved())
        torch.cuda.synchronize()
        with lock:
            state['iterations'] = n
            state['status'] = 'PASS'
    except Exception as exc:
        with lock:
            state['status'] = 'FAIL'
            state['error'] = f'{type(exc).__name__}:{exc}'
        stop.set()


def ram_worker(stop: threading.Event, state: dict, lock: threading.Lock) -> None:
    try:
        info = meminfo()
        total = info.get('MemTotal', 0)
        target = min(2 * 1024**3, max(512 * 1024**2, total // 8))
        buf = bytearray(target)
        loops = 0
        while not stop.is_set():
            for i in range(0, len(buf), 4096):
                buf[i] = (buf[i] + 1) & 0xFF
                if stop.is_set():
                    break
            loops += 1
            with lock:
                state['bytes_allocated'] = target
                state['touch_passes'] = loops
        with lock:
            state['status'] = 'PASS'
    except Exception as exc:
        with lock:
            state['status'] = 'FAIL'
            state['error'] = f'{type(exc).__name__}:{exc}'
        stop.set()


def disk_worker(stop: threading.Event, state: dict, lock: threading.Lock, root: Path) -> None:
    path = root / f'cb16_burst_disk_{os.getpid()}.bin'
    pattern = bytes((i % 251 for i in range(CHUNK)))
    expected = hashlib.sha256()
    try:
        with path.open('wb', buffering=0) as fh:
            remaining = DISK_BYTES
            while remaining > 0 and not stop.is_set():
                b = pattern[:min(CHUNK, remaining)]
                fh.write(b)
                expected.update(b)
                remaining -= len(b)
            fh.flush()
            os.fsync(fh.fileno())
        if remaining:
            raise RuntimeError('DISK_WRITE_INTERRUPTED')
        expected_hex = expected.hexdigest()
        passes = 0
        bytes_read = 0
        while not stop.is_set():
            h = hashlib.sha256()
            with path.open('rb', buffering=0) as fh:
                while True:
                    b = fh.read(CHUNK)
                    if not b:
                        break
                    h.update(b)
                    bytes_read += len(b)
                    if stop.is_set():
                        break
            if stop.is_set():
                break
            if h.hexdigest() != expected_hex:
                raise RuntimeError('DISK_HASH_MISMATCH')
            passes += 1
            with lock:
                state['verify_passes'] = passes
                state['bytes_read'] = bytes_read
                state['written_bytes'] = DISK_BYTES
                state['sha256'] = expected_hex
        with lock:
            state['status'] = 'PASS'
    except Exception as exc:
        with lock:
            state['status'] = 'FAIL'
            state['error'] = f'{type(exc).__name__}:{exc}'
        stop.set()
    finally:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass


def nvidia_sample() -> dict:
    cmd = [
        'nvidia-smi',
        '--query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw',
        '--format=csv,noheader,nounits',
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    if p.returncode != 0:
        return {'ok': False, 'error': p.stderr.strip()}
    vals = [x.strip() for x in p.stdout.strip().split(',')]
    try:
        return {
            'ok': True,
            'temperature_c': float(vals[0]),
            'utilization_pct': float(vals[1]),
            'memory_used_mib': float(vals[2]),
            'memory_total_mib': float(vals[3]),
            'power_w': float(vals[4]),
        }
    except Exception as exc:
        return {'ok': False, 'error': f'PARSE:{exc}', 'raw': p.stdout.strip()}


def main() -> int:
    if DURATION != 300:
        print('CB16 burst contract requires exactly 300 seconds', file=sys.stderr)
        return 78
    runner_temp = Path(os.environ.get('RUNNER_TEMP', tempfile.gettempdir()))
    runner_temp.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(os.environ.get('CB16_BURST_OUT', str(runner_temp / 'cb16_r11_host_burst_5m.json')))

    cpu_count = os.cpu_count() or 1
    cpu_workers = max(1, min(12, cpu_count - 2 if cpu_count > 2 else cpu_count))
    ctx = mp.get_context('fork')
    cpu_stop = ctx.Event()
    cpu_counter = ctx.Value('Q', 0)
    procs = [ctx.Process(target=cpu_worker, args=(cpu_stop, cpu_counter, i), daemon=True) for i in range(cpu_workers)]

    thread_stop = threading.Event()
    lock = threading.Lock()
    gpu_state = {'status': 'RUNNING', 'iterations': 0}
    ram_state = {'status': 'RUNNING', 'touch_passes': 0, 'bytes_allocated': 0}
    disk_state = {'status': 'RUNNING', 'verify_passes': 0, 'bytes_read': 0, 'written_bytes': 0}
    gpu_t = threading.Thread(target=gpu_worker, args=(thread_stop, gpu_state, lock), daemon=True)
    ram_t = threading.Thread(target=ram_worker, args=(thread_stop, ram_state, lock), daemon=True)
    disk_t = threading.Thread(target=disk_worker, args=(thread_stop, disk_state, lock, runner_temp), daemon=True)

    start_events = cgroup_memory_events()
    failures: list[str] = []
    samples: list[dict] = []
    start = time.monotonic()
    started_at = utcnow()
    for p in procs:
        p.start()
    gpu_t.start(); ram_t.start(); disk_t.start()

    try:
        while True:
            elapsed = time.monotonic() - start
            if elapsed >= DURATION:
                break
            if thread_stop.is_set():
                failures.append('WORKER_REQUESTED_ABORT')
                break
            nv = nvidia_sample()
            mi = meminfo()
            free_disk = shutil.disk_usage(runner_temp).free
            sample = {
                'elapsed_seconds': round(elapsed, 3),
                'loadavg': Path('/proc/loadavg').read_text().strip(),
                'mem_available_bytes': mi.get('MemAvailable'),
                'swap_free_bytes': mi.get('SwapFree'),
                'disk_free_bytes': free_disk,
                'gpu': nv,
                'cgroup_memory_events': cgroup_memory_events(),
                'cpu_hash_iterations': int(cpu_counter.value),
            }
            samples.append(sample)
            print(json.dumps(sample, sort_keys=True), flush=True)
            if nv.get('ok') and nv.get('temperature_c', 0) >= GPU_ABORT_C:
                failures.append(f'GPU_THERMAL_ABORT:{nv["temperature_c"]}')
                break
            if mi.get('MemAvailable', 0) < MIN_MEM_AVAILABLE:
                failures.append(f'LOW_MEMORY_ABORT:{mi.get("MemAvailable", 0)}')
                break
            if free_disk < MIN_FREE_DISK:
                failures.append(f'LOW_DISK_ABORT:{free_disk}')
                break
            time.sleep(min(SAMPLE_SECONDS, max(0, DURATION - elapsed)))
    finally:
        cpu_stop.set(); thread_stop.set()
        for p in procs:
            p.join(timeout=15)
            if p.is_alive():
                p.terminate(); p.join(timeout=5)
        for t in (gpu_t, ram_t, disk_t):
            t.join(timeout=30)

    elapsed = time.monotonic() - start
    end_events = cgroup_memory_events()
    with lock:
        gpu = dict(gpu_state); ram = dict(ram_state); disk = dict(disk_state)
    if elapsed < 295 and not failures:
        failures.append(f'DURATION_TOO_SHORT:{elapsed:.3f}')
    if int(cpu_counter.value) <= 0:
        failures.append('CPU_NO_WORK')
    if gpu.get('status') != 'PASS' or int(gpu.get('iterations', 0)) <= 0:
        failures.append(f'GPU_WORK_FAILED:{gpu}')
    if ram.get('status') != 'PASS' or int(ram.get('touch_passes', 0)) <= 0:
        failures.append(f'RAM_WORK_FAILED:{ram}')
    if disk.get('status') != 'PASS' or int(disk.get('written_bytes', 0)) != DISK_BYTES:
        failures.append(f'DISK_WORK_FAILED:{disk}')
    for key in ('oom', 'oom_kill'):
        if end_events.get(key, 0) > start_events.get(key, 0):
            failures.append(f'CGROUP_{key.upper()}_INCREASED:{start_events.get(key,0)}->{end_events.get(key,0)}')
    for p in procs:
        if p.exitcode not in (0, None):
            failures.append(f'CPU_WORKER_EXIT:{p.pid}:{p.exitcode}')

    temps = [s['gpu']['temperature_c'] for s in samples if s.get('gpu', {}).get('ok')]
    utils = [s['gpu']['utilization_pct'] for s in samples if s.get('gpu', {}).get('ok')]
    receipt = {
        'schema': 'CB16_R11_HOST_BURST_5M_V1',
        'status': 'PASS' if not failures else 'FAIL_CLOSED',
        'started_at_utc': started_at,
        'completed_at_utc': utcnow(),
        'requested_seconds': DURATION,
        'elapsed_seconds': round(elapsed, 3),
        'cpu': {'logical_cpus': cpu_count, 'workers': cpu_workers, 'hash_iterations': int(cpu_counter.value)},
        'gpu': gpu,
        'ram': ram,
        'disk': disk,
        'telemetry': {
            'sample_count': len(samples),
            'gpu_temp_max_c': max(temps) if temps else None,
            'gpu_temp_min_c': min(temps) if temps else None,
            'gpu_util_max_pct': max(utils) if utils else None,
            'gpu_util_avg_pct': round(sum(utils)/len(utils), 2) if utils else None,
            'cgroup_memory_events_start': start_events,
            'cgroup_memory_events_end': end_events,
            'samples': samples,
        },
        'safety': {
            'gpu_abort_c': GPU_ABORT_C,
            'min_mem_available_bytes': MIN_MEM_AVAILABLE,
            'min_disk_free_bytes': MIN_FREE_DISK,
            'legacy_or_raw_data_mutated': False,
            'training_started': False,
            'final_holdout_payload_opened': False,
            'fresh_market_data_downloaded': False,
        },
        'failures': failures,
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not failures else 78


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    raise SystemExit(main())
