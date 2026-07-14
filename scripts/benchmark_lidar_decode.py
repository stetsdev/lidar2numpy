#!/usr/bin/env python3
"""Benchmark raw JT128 packet decoding through the public streaming API."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

from lidar2numpy import Decoder, default_calibration, load_calibration
from lidar2numpy.calibration import Calibration
from lidar2numpy.pcap import read_pcap_payloads

BackendRequest = Literal["auto", "python", "cython"]
_TAIL_RETURN_MODE_OFFSET = 1056


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap", nargs="?", type=Path, help="Raw JT128 PCAP capture.")
    parser.add_argument("--calibration", type=Path, help="Per-unit JT128 calibration CSV.")
    parser.add_argument(
        "--backend",
        choices=("auto", "python", "cython"),
        default="auto",
        help="Decoder backend to benchmark. Cython is added in a later sub-increment.",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Number of complete benchmark runs.")
    parser.add_argument(
        "--warmup-packets",
        type=int,
        default=0,
        help="Packets decoded before recording metrics on each run.",
    )
    parser.add_argument("--max-packets", type=int, help="Maximum packets read from the PCAP.")
    parser.add_argument("--json-out", type=Path, help="Write JSON result to this path.")
    args = parser.parse_args(argv)
    if args.pcap is None:
        parser.error("pcap is required")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than 0")
    if args.warmup_packets < 0:
        parser.error("--warmup-packets must be non-negative")
    if args.max_packets is not None and args.max_packets <= 0:
        parser.error("--max-packets must be greater than 0")
    return args


def _resolve_backend(requested: BackendRequest) -> Literal["python"]:
    """Resolve a C1.1 backend request before the compiled backend exists."""
    if requested == "cython":
        raise ValueError("cython backend is unavailable in this build")
    return "python"


def _percentile(values: list[float], percentile: float) -> float:
    """Return a linear-interpolated percentile from a non-empty value list."""
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (percentile / 100.0) * (len(sorted_values) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = rank - lower
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * fraction


def _return_mode_counts(payloads: list[bytes]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for payload in payloads:
        code = f"0x{payload[_TAIL_RETURN_MODE_OFFSET]:02X}"
        counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items()))


def run_once(
    payloads: list[bytes],
    *,
    calibration: Calibration,
    backend: BackendRequest,
    warmup_packets: int,
) -> dict[str, Any]:
    """Decode one packet sequence and return comparable aggregate metrics."""
    resolved_backend = _resolve_backend(backend)
    if warmup_packets >= len(payloads):
        raise ValueError("warmup-packets must leave at least one timed packet")

    decoder = Decoder(calibration, output_mode="spherical")
    latencies_us: list[float] = []
    frames = 0
    points = 0
    wall_start_ns = time.perf_counter_ns()
    process_start_ns = time.process_time_ns()
    thread_start_ns = time.thread_time_ns()

    for index, payload in enumerate(payloads):
        start_ns = time.perf_counter_ns()
        frame = decoder.feed(payload)
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        if index < warmup_packets:
            continue
        latencies_us.append(elapsed_us)
        if frame is not None:
            frames += 1
            points += len(frame)

    wall_time_s = (time.perf_counter_ns() - wall_start_ns) / 1_000_000_000.0
    process_cpu_time_s = (time.process_time_ns() - process_start_ns) / 1_000_000_000.0
    thread_cpu_time_s = (time.thread_time_ns() - thread_start_ns) / 1_000_000_000.0
    packets = len(latencies_us)
    return {
        "backend": resolved_backend,
        "packets": packets,
        "frames": frames,
        "points": points,
        "wall_time_s": wall_time_s,
        "process_cpu_time_s": process_cpu_time_s,
        "thread_cpu_time_s": thread_cpu_time_s,
        "packets_per_second": packets / wall_time_s if wall_time_s > 0.0 else 0.0,
        "feed_latency_us": {
            "p50": _percentile(latencies_us, 50.0),
            "p95": _percentile(latencies_us, 95.0),
            "p99": _percentile(latencies_us, 99.0),
        },
    }


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _environment() -> dict[str, str | None]:
    return {
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "numpy_version": _package_version("numpy"),
        "lidar2numpy_version": _package_version("lidar2numpy"),
    }


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark and write one machine-readable result document."""
    args = parse_args(argv)
    payloads = list(read_pcap_payloads(args.pcap))
    if args.max_packets is not None:
        payloads = payloads[: args.max_packets]
    if not payloads:
        raise SystemExit(f"No 1100-byte JT128 payloads found in {args.pcap}")

    calibration = load_calibration(args.calibration) if args.calibration else default_calibration()
    backend: BackendRequest = args.backend
    runs = [
        run_once(
            payloads,
            calibration=calibration,
            backend=backend,
            warmup_packets=args.warmup_packets,
        )
        for _ in range(args.repeat)
    ]
    result = {
        "schema_version": 1,
        "pcap": str(args.pcap),
        "calibration": str(args.calibration) if args.calibration else "bundled-default",
        "backend": runs[0]["backend"],
        "repeat": args.repeat,
        "warmup_packets": args.warmup_packets,
        "packets_per_run": len(payloads) - args.warmup_packets,
        "return_mode_counts": _return_mode_counts(payloads),
        "environment": _environment(),
        "runs": runs,
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is None:
        print(encoded, end="")
    else:
        args.json_out.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
