"""CPU energy and latency instrumentation.

Uses `psutil` to read CPU power consumption where available (RAPL on Linux
Intel/AMD); falls back to a calibrated estimate (TDP * CPU utilisation * time)
when RAPL is not accessible.
"""
from __future__ import annotations
from typing import Callable, Dict, Optional
import os
import time
import numpy as np

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


_RAPL_PATH = "/sys/class/powercap/intel-rapl:0/energy_uj"


def _read_rapl_uj() -> Optional[float]:
    """Read cumulative CPU energy in microjoules from RAPL, if available."""
    try:
        with open(_RAPL_PATH) as f:
            return float(f.read().strip())
    except Exception:
        return None


# Estimated TDP (Watts) of common CPUs — calibrated by `--calibrate` if needed.
_DEFAULT_TDP_W = 65.0


def measure_inference_energy(predict_fn: Callable, X, n_repeats: int = 200
                             ) -> Dict[str, float]:
    """Run `predict_fn(X)` n_repeats times and report mean latency, mean energy
    per inference, and total wall-clock time.

    Returns a dict with keys:
        latency_ms_p50, latency_ms_p95, latency_ms_mean,
        energy_mJ_mean, energy_mJ_p95, total_time_s, n_repeats, method
    """
    # warmup
    for _ in range(min(5, n_repeats)):
        predict_fn(X)

    latencies_ms = np.zeros(n_repeats, dtype=np.float64)
    energies_mJ = np.zeros(n_repeats, dtype=np.float64)

    rapl_start = _read_rapl_uj()
    cpu_start = (psutil.cpu_times() if _HAS_PSUTIL else None)

    for i in range(n_repeats):
        t0 = time.perf_counter()
        rapl_pre = _read_rapl_uj()
        _ = predict_fn(X)
        rapl_post = _read_rapl_uj()
        t1 = time.perf_counter()
        latencies_ms[i] = (t1 - t0) * 1000.0
        if rapl_pre is not None and rapl_post is not None:
            energies_mJ[i] = max(0.0, (rapl_post - rapl_pre) / 1000.0)  # uJ -> mJ

    rapl_end = _read_rapl_uj()
    total_time_s = time.perf_counter() - time.perf_counter()  # placeholder, recomputed below

    # If RAPL did not produce usable deltas, fall back to TDP-based estimate
    if (energies_mJ > 0).sum() < n_repeats * 0.5:
        method = "tdp_estimate"
        # Estimate utilisation via psutil
        if _HAS_PSUTIL:
            cpu_util = psutil.cpu_percent(interval=None) or 50.0
        else:
            cpu_util = 50.0
        # energy_mJ = TDP_W * cpu_util/100 * latency_s * 1000
        energies_mJ = latencies_ms * _DEFAULT_TDP_W * (cpu_util / 100.0)
    else:
        method = "rapl_direct"

    # Final wall-clock
    return {
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "latency_ms_mean": float(latencies_ms.mean()),
        "energy_mJ_mean": float(energies_mJ.mean()),
        "energy_mJ_p95": float(np.percentile(energies_mJ, 95)),
        "total_time_s": float(latencies_ms.sum() / 1000.0),
        "n_repeats": int(n_repeats),
        "method": method,
        "tdp_assumed_w": float(_DEFAULT_TDP_W),
    }
