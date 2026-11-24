"""Timing, sampled process memory, configurations, logs and tabular records."""

import csv
import ctypes
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import threading
import time


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    versions = {}
    for name in ["numpy", "scipy", "scikit-learn", "matplotlib", "pandas", "torch", "threadpoolctl"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "processor": platform.processor(), "logical_cpus": os.cpu_count(),
            "gpu": gpu, "packages": versions,
            "memory_protocol": "absolute process working set sampled every 20 ms; includes shared data and libraries; not an isolated algorithm allocation"}


def logger_at(folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(str(folder.resolve()))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handlers = [logging.StreamHandler(), logging.FileHandler(folder / "run.log", encoding="utf-8")]
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
            logger.addHandler(handler)
    return logger


def working_set_mb():
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ["PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage"]]
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    memory = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
    memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    record = Counters()
    record.cb = ctypes.sizeof(record)
    if not memory(kernel.GetCurrentProcess(), ctypes.byref(record), record.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return record.WorkingSetSize / 1024 ** 2


class Measurement:
    def __enter__(self):
        self.peak_rss_mb = working_set_mb()
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.02):
                value = working_set_mb()
                if value is not None:
                    self.peak_rss_mb = max(self.peak_rss_mb or 0.0, value)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        self.started = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.seconds = time.perf_counter() - self.started
        self.stop.set()
        self.thread.join(timeout=1)
        end = working_set_mb()
        if end is not None:
            self.peak_rss_mb = max(self.peak_rss_mb or 0, end)


def aggregate(rows, groups, metrics):
    import numpy as np
    buckets = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in groups), []).append(row)
    output = []
    for key, values in buckets.items():
        item = dict(zip(groups, key))
        item["runs"] = len(values)
        for metric in metrics:
            numbers = [float(row[metric]) for row in values if row.get(metric) is not None]
            item[metric + "_mean"] = float(np.mean(numbers)) if numbers else None
            item[metric + "_std"] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        output.append(item)
    return output
