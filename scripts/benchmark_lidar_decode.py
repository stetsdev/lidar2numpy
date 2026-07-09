#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Benchmark JT128 decode throughput with optional channel/azimuth filtering."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lidar2numpy import ChannelAzimuthFilter, Decoder, default_calibration, load_calibration  # noqa: E402
from lidar2numpy.pcap import read_pcap_payloads  # noqa: E402

FilterMode = Literal["off", "shadow", "drop"]


def main() -> int:
    args = parse_args()
    payloads = list(read_pcap_payloads(args.pcap, udp_payload_size=1100))
    if args.max_packets is not None:
        payloads = payloads[: args.max_packets]
    if not payloads:
        raise SystemExit(f"No 1100-byte JT128 UDP payloads found in {args.pcap}")

    calibration = load_calibration(args.calibration) if args.calibration else default_calibration()
    if args.calibration is None:
        print("warning: using nominal default calibration", file=sys.stderr)

    filter_spec, filter_enabled, filter_mode = load_filter_options(
        filter_config=args.filter_config,
        config_path=args.config_path,
        filter_mode_override=args.filter_mode,
    )

    repeats: list[dict[str, object]] = []
    for _ in range(args.repeat):
        repeats.append(
            run_once(
                payloads,
                calibration=calibration,
                output_mode=args.output_mode,
                filter_spec=filter_spec,
                filter_enabled=filter_enabled,
                filter_mode=filter_mode,
                warmup_packets=args.warmup_packets,
                include_dropped_by_channel=args.include_dropped_by_channel,
            )
        )

    result = aggregate_results(
        repeats,
        pcap=args.pcap,
        calibration_path=args.calibration,
        output_mode=args.output_mode,
        filter_enabled=filter_enabled,
        filter_mode=filter_mode,
    )

    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pcap", nargs="?", type=Path, help="Input pcap file path.")
    parser.add_argument("--pcap-path", type=Path, help="Alias for positional PCAP.")
    parser.add_argument("--calibration", type=Path, help="Per-unit JT128 angle correction CSV.")
    parser.add_argument(
        "--output-mode",
        choices=("cartesian", "spherical"),
        default="spherical",
        help="Decoder output mode. Defaults to spherical.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--filter-config",
        type=Path,
        help="JSON file containing only the lidar_decode_filter object.",
    )
    source_group.add_argument(
        "--config-path",
        type=Path,
        help="Full intersection.json containing perception.lidar_decode_filter.",
    )
    parser.add_argument("--filter-mode", choices=("off", "shadow", "drop"))
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--warmup-packets", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--include-dropped-by-channel",
        action="store_true",
        help="Include per-channel drop counts in the JSON result.",
    )
    args = parser.parse_args()

    pcap = args.pcap or args.pcap_path
    if pcap is None:
        parser.error("PCAP is required, either positional or via --pcap-path")
    args.pcap = pcap
    if args.max_packets is not None and args.max_packets <= 0:
        parser.error("--max-packets must be greater than 0")
    if args.warmup_packets < 0:
        parser.error("--warmup-packets must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be greater than 0")
    return args


def load_filter_options(
    *,
    filter_config: Path | None,
    config_path: Path | None,
    filter_mode_override: str | None,
) -> tuple[ChannelAzimuthFilter | None, bool, FilterMode]:
    payload: dict[str, object] = {
        "enabled": False,
        "mode": "off",
        "drop_channels": [],
        "channel_azimuth_drops": [],
    }
    if filter_config is not None:
        payload = _load_json_object(filter_config)
    elif config_path is not None:
        config = _load_json_object(config_path)
        perception = config.get("perception", {})
        if not isinstance(perception, dict):
            raise ValueError("config perception section must be a JSON object")
        raw_filter = perception.get("lidar_decode_filter", payload)
        if not isinstance(raw_filter, dict):
            raise ValueError("perception.lidar_decode_filter must be a JSON object")
        payload = raw_filter

    mode = _filter_mode(payload.get("mode", "off"))
    enabled = bool(payload.get("enabled", mode != "off"))
    if filter_mode_override is not None:
        mode = _filter_mode(filter_mode_override)
        enabled = mode != "off"

    if not enabled and mode != "off":
        raise ValueError("disabled lidar_decode_filter requires mode='off'")
    if enabled and mode == "off":
        raise ValueError("enabled lidar_decode_filter requires mode 'shadow' or 'drop'")
    if mode == "off":
        return None, False, "off"

    return ChannelAzimuthFilter.from_mapping(payload), True, mode


def run_once(  # noqa: PLR0913
    payloads: list[bytes],
    *,
    calibration: object,
    output_mode: Literal["cartesian", "spherical"],
    filter_spec: ChannelAzimuthFilter | None,
    filter_enabled: bool,
    filter_mode: FilterMode,
    warmup_packets: int,
    include_dropped_by_channel: bool,
) -> dict[str, object]:
    decoder = _build_decoder(
        calibration=calibration,
        output_mode=output_mode,
        filter_spec=filter_spec,
        filter_enabled=filter_enabled,
        filter_mode=filter_mode,
    )
    non_emitting_us: list[float] = []
    emitting_us: list[float] = []
    frames = 0
    input_points = 0
    output_points = 0
    dropped_points = 0
    dropped_by_channel = np.zeros(128, dtype=np.int64)

    total_start_ns = time.perf_counter_ns()
    timed_packets = 0
    for index, payload in enumerate(payloads):
        start_ns = time.perf_counter_ns()
        frame = decoder.feed(payload)
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        if index < warmup_packets:
            continue

        timed_packets += 1
        if frame is None:
            non_emitting_us.append(elapsed_us)
        else:
            emitting_us.append(elapsed_us)
            frames += 1
            frame_input, frame_output, frame_dropped, frame_by_channel = _frame_counts(
                decoder, frame, filter_enabled
            )
            input_points += frame_input
            output_points += frame_output
            dropped_points += frame_dropped
            dropped_by_channel += frame_by_channel

    frame = decoder.flush()
    if frame is not None:
        frames += 1
        frame_input, frame_output, frame_dropped, frame_by_channel = _frame_counts(
            decoder, frame, filter_enabled
        )
        input_points += frame_input
        output_points += frame_output
        dropped_points += frame_dropped
        dropped_by_channel += frame_by_channel
    total_decode_ms = (time.perf_counter_ns() - total_start_ns) / 1_000_000.0

    prepared = decoder._prepared_filter
    result: dict[str, object] = {
        "packets": timed_packets,
        "frames": frames,
        "mean_us_per_packet": total_decode_ms * 1_000.0 / max(timed_packets, 1),
        "feed_non_emitting_p50_us": _percentile(non_emitting_us, 50.0),
        "feed_non_emitting_p95_us": _percentile(non_emitting_us, 95.0),
        "feed_emitting_p50_us": _percentile(emitting_us, 50.0),
        "feed_emitting_p95_us": _percentile(emitting_us, 95.0),
        "total_decode_ms": total_decode_ms,
        "input_points": input_points,
        "output_points": output_points,
        "dropped_points": dropped_points,
        "filter_spec_fingerprint": prepared.spec_fingerprint if prepared is not None else None,
        "filter_calibration_fingerprint": (
            prepared.calibration_fingerprint if prepared is not None else None
        ),
        "filter_prepared_fingerprint": prepared.fingerprint if prepared is not None else None,
        "prepared_lookup_bytes": (
            int(prepared.drop_by_raw_azimuth_and_channel.nbytes) if prepared is not None else 0
        ),
    }
    if include_dropped_by_channel:
        result["dropped_points_by_channel"] = {
            str(channel): int(count)
            for channel, count in enumerate(dropped_by_channel, start=1)
            if count
        }
    return result


def aggregate_results(  # noqa: PLR0913
    repeats: list[dict[str, object]],
    *,
    pcap: Path,
    calibration_path: Path | None,
    output_mode: str,
    filter_enabled: bool,
    filter_mode: FilterMode,
) -> dict[str, object]:
    first = repeats[0]
    result: dict[str, object] = {
        "pcap": str(pcap),
        "calibration": str(calibration_path) if calibration_path is not None else None,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "output_mode": output_mode,
        "filter_mode": filter_mode,
        "filter_enabled": filter_enabled,
        "repeat": len(repeats),
    }
    for key in (
        "packets",
        "frames",
        "input_points",
        "output_points",
        "dropped_points",
        "filter_spec_fingerprint",
        "filter_calibration_fingerprint",
        "filter_prepared_fingerprint",
        "prepared_lookup_bytes",
        "dropped_points_by_channel",
    ):
        if key in first:
            result[key] = first[key]
    for key in (
        "mean_us_per_packet",
        "feed_non_emitting_p50_us",
        "feed_non_emitting_p95_us",
        "feed_emitting_p50_us",
        "feed_emitting_p95_us",
        "total_decode_ms",
    ):
        result[key] = _mean_float([repeat[key] for repeat in repeats])
    return result


def _build_decoder(
    *,
    calibration: object,
    output_mode: Literal["cartesian", "spherical"],
    filter_spec: ChannelAzimuthFilter | None,
    filter_enabled: bool,
    filter_mode: FilterMode,
) -> Decoder:
    if not filter_enabled or filter_mode == "off":
        return Decoder(calibration, output_mode=output_mode)
    return Decoder(
        calibration,
        output_mode=output_mode,
        point_filter=filter_spec,
        point_filter_mode=filter_mode,
    )


def _frame_counts(
    decoder: Decoder,
    frame: np.ndarray,
    filter_enabled: bool,
) -> tuple[int, int, int, np.ndarray]:
    if not filter_enabled:
        return len(frame), len(frame), 0, np.zeros(128, dtype=np.int64)

    diagnostics = decoder.last_filter_diagnostics()
    dropped_by_channel = np.zeros(128, dtype=np.int64)
    for channel, count in diagnostics.dropped_points_by_channel.items():
        dropped_by_channel[channel - 1] = count
    return (
        diagnostics.input_points,
        diagnostics.output_points,
        diagnostics.dropped_points,
        dropped_by_channel,
    )


def _filter_mode(value: object) -> FilterMode:
    if value not in ("off", "shadow", "drop"):
        raise ValueError(f"filter mode must be 'off', 'shadow', or 'drop'; got {value!r}")
    return value


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.array(values, dtype=np.float64), percentile))


def _mean_float(values: list[object]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    if not numeric:
        return None
    return float(np.mean(np.array(numeric, dtype=np.float64)))


if __name__ == "__main__":
    raise SystemExit(main())
