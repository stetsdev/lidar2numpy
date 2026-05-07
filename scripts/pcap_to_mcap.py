#!/usr/bin/env python3
"""Convert a Hesai JT128 pcap capture to an MCAP file for visualization in Foxglove.

Reads raw UDP packets from a pcap file, decodes them via lidar2numpy, and writes
each frame as a foxglove.PointCloud message in an MCAP container. The resulting
.mcap file can be opened directly in Foxglove's 3D panel.

Usage:
    uv run python scripts/pcap_to_mcap.py \
        tests/fixtures/sensor_a/capture.pcap \
        -c tests/fixtures/sensor_a/angle_correction.csv \
        -o output.mcap

    # Rebase 1970 timestamps to wall clock (for sensors without PTP/GPS sync):
    uv run python scripts/pcap_to_mcap.py capture.pcap -o output.mcap --rebase

    # Rebase to a specific start time:
    uv run python scripts/pcap_to_mcap.py capture.pcap -o output.mcap \
        --rebase --start-time "2026-05-06T14:30:00"

Dependencies (not in lidar2numpy's runtime deps — install manually):
    uv add --dev mcap
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lidar2numpy import Decoder, load_calibration, default_calibration

# Foxglove numeric type codes (from foxglove.NumericType enum)
FLOAT32 = 7
UINT8 = 1
UINT16 = 3

# foxglove.PointCloud JSON schema — embedded so we don't need an external file.
# Foxglove matches on schema name "foxglove.PointCloud" to enable 3D panel rendering.
POINTCLOUD_SCHEMA = json.dumps({
    "title": "foxglove.PointCloud",
    "description": "A collection of N-dimensional points, which may contain additional fields with information like normals, intensity, etc.",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999}
            },
            "description": "Timestamp of point cloud"
        },
        "frame_id": {"type": "string", "description": "Frame of reference"},
        "pose": {
            "title": "foxglove.Pose",
            "description": "The origin of the point cloud relative to the frame of reference",
            "type": "object",
            "properties": {
                "position": {
                    "title": "foxglove.Vector3",
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"}, "y": {"type": "number"}, "z": {"type": "number"}
                    },
                    "required": ["x", "y", "z"]
                },
                "orientation": {
                    "title": "foxglove.Quaternion",
                    "type": "object",
                    "properties": {
                        "x": {"type": "number"}, "y": {"type": "number"},
                        "z": {"type": "number"}, "w": {"type": "number"}
                    },
                    "required": ["x", "y", "z", "w"]
                }
            },
            "required": ["position", "orientation"]
        },
        "point_stride": {"type": "integer", "minimum": 0, "description": "Number of bytes between points in the data"},
        "fields": {
            "type": "array",
            "items": {
                "title": "foxglove.PackedElementField",
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "type": {
                        "title": "foxglove.NumericType",
                        "oneOf": [
                            {"title": "UINT8", "const": 1}, {"title": "INT8", "const": 2},
                            {"title": "UINT16", "const": 3}, {"title": "INT16", "const": 4},
                            {"title": "UINT32", "const": 5}, {"title": "INT32", "const": 6},
                            {"title": "FLOAT32", "const": 7}, {"title": "FLOAT64", "const": 8}
                        ]
                    }
                },
                "required": ["name", "offset", "type"]
            }
        },
        "data": {"type": "string", "contentEncoding": "base64", "description": "Point data, interpreted using fields"}
    },
    "required": ["timestamp", "frame_id", "pose", "point_stride", "fields", "data"]
}).encode("utf-8")


# Point layout for foxglove: x, y, z (float32) + intensity (float32)
# We pack a subset of POINT_DTYPE fields into a tighter per-point buffer.
FOXGLOVE_POINT_STRIDE = 4 * 4  # 16 bytes: x, y, z, intensity
FOXGLOVE_FIELDS = [
    {"name": "x", "offset": 0, "type": FLOAT32},
    {"name": "y", "offset": 4, "type": FLOAT32},
    {"name": "z", "offset": 8, "type": FLOAT32},
    {"name": "intensity", "offset": 12, "type": FLOAT32},
]


# ---------------------------------------------------------------------------
# Minimal pcap reader (same approach as test_integration.py)
# ---------------------------------------------------------------------------

_PCAP_MAGIC_US = 0xA1B2C3D4
_PCAP_MAGIC_NS = 0xA1B23C4D


def read_pcap_payloads(pcap_path: Path, udp_payload_size: int = 1100):
    """Yield UDP payloads from a pcap file, stripping Ethernet+IP+UDP headers."""
    with open(pcap_path, "rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError("Truncated pcap global header")

        magic = struct.unpack("<I", global_header[:4])[0]
        if magic == _PCAP_MAGIC_US:
            endian = "<"
        elif magic == _PCAP_MAGIC_NS:
            endian = "<"
        elif struct.unpack(">I", global_header[:4])[0] in (_PCAP_MAGIC_US, _PCAP_MAGIC_NS):
            endian = ">"
        else:
            raise ValueError(f"Not a pcap file: bad magic {global_header[:4].hex()}")

        while True:
            rec_header = f.read(16)
            if len(rec_header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", rec_header)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break

            # Strip: 14 Ethernet + 20 IP + 8 UDP = 42 bytes
            if len(data) >= 42 + udp_payload_size:
                payload = data[42 : 42 + udp_payload_size]
                if len(payload) == udp_payload_size:
                    yield payload


# ---------------------------------------------------------------------------
# Timestamp rebasing
# ---------------------------------------------------------------------------

def detect_needs_rebase(first_frame: np.ndarray, threshold_year: int = 2000) -> bool:
    """Return True if the frame's timestamps predate threshold_year (unsynced clock)."""
    if len(first_frame) == 0:
        return False
    t = float(first_frame["timestamp"][0])
    try:
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        return dt.year < threshold_year
    except (OSError, OverflowError, ValueError):
        return True


def compute_rebase_offset(
    first_frame: np.ndarray,
    start_time: datetime | None = None,
) -> float:
    """Compute the offset to add to all timestamps for rebasing.

    If start_time is provided, rebases to that wall clock time.
    Otherwise, rebases to the current wall clock time.
    """
    frame_t0 = float(first_frame["timestamp"].min())
    if start_time is not None:
        target_t0 = start_time.timestamp()
    else:
        target_t0 = time.time()
    return target_t0 - frame_t0


# ---------------------------------------------------------------------------
# MCAP writing
# ---------------------------------------------------------------------------

def frame_to_pointcloud_msg(
    frame: np.ndarray,
    rebase_offset: float = 0.0,
) -> tuple[dict, int]:
    """Convert a lidar2numpy frame to a foxglove.PointCloud JSON dict.

    Returns (message_dict, log_time_ns).
    """
    # Pack x, y, z, intensity into a contiguous buffer
    n = len(frame)
    buf = bytearray(n * FOXGLOVE_POINT_STRIDE)
    for i in range(n):
        struct.pack_into(
            "<ffff",
            buf,
            i * FOXGLOVE_POINT_STRIDE,
            float(frame["x"][i]),
            float(frame["y"][i]),
            float(frame["z"][i]),
            float(frame["intensity"][i]),
        )

    # Frame timestamp: use the minimum timestamp in the frame
    frame_ts = float(frame["timestamp"].min()) + rebase_offset
    sec = int(frame_ts)
    nsec = int((frame_ts - sec) * 1e9)

    msg = {
        "timestamp": {"sec": sec, "nsec": nsec},
        "frame_id": "lidar",
        "pose": {
            "position": {"x": 0, "y": 0, "z": 0},
            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
        },
        "point_stride": FOXGLOVE_POINT_STRIDE,
        "fields": FOXGLOVE_FIELDS,
        "data": base64.b64encode(buf).decode("utf-8"),
    }

    log_time_ns = sec * 1_000_000_000 + nsec
    return msg, log_time_ns


def convert(
    pcap_path: Path,
    output_path: Path,
    calibration_path: Path | None,
    rebase: bool,
    start_time: datetime | None,
    max_frames: int | None,
) -> None:
    """Read pcap, decode frames, write MCAP."""
    # Late import so the script fails with a clear message if mcap isn't installed
    try:
        from mcap.writer import Writer
        from mcap.well_known import SchemaEncoding, MessageEncoding
    except ImportError:
        print("Error: mcap package not installed. Run: uv add --dev mcap", file=sys.stderr)
        sys.exit(1)

    # Load calibration
    if calibration_path is not None:
        cal = load_calibration(calibration_path)
    else:
        cal = default_calibration()
        print("No calibration CSV specified — using nominal defaults.", file=sys.stderr)

    decoder = Decoder(calibration=cal)

    # First pass: collect frames (we need the first frame to decide on rebasing)
    frames: list[np.ndarray] = []
    packet_count = 0
    for payload in read_pcap_payloads(pcap_path):
        frame = decoder.feed(payload)
        packet_count += 1
        if frame is not None and len(frame) > 0:
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break

    # Flush the last partial frame
    last = decoder.flush()
    if last is not None and len(last) > 0:
        if max_frames is None or len(frames) < max_frames:
            frames.append(last)

    if not frames:
        print(f"No complete frames decoded from {pcap_path} ({packet_count} packets read).", file=sys.stderr)
        sys.exit(1)

    # Determine rebase offset
    rebase_offset = 0.0
    if rebase and detect_needs_rebase(frames[0]):
        rebase_offset = compute_rebase_offset(frames[0], start_time)
        rebased_start = datetime.fromtimestamp(
            float(frames[0]["timestamp"].min()) + rebase_offset,
            tz=timezone.utc,
        )
        print(
            f"Timestamps rebased: sensor clock is unsynced (pre-2000). "
            f"First frame mapped to {rebased_start.isoformat()}.",
            file=sys.stderr,
        )
    elif rebase:
        print("Timestamps appear synced (post-2000) — skipping rebase.", file=sys.stderr)

    # Write MCAP
    with open(output_path, "wb") as f:
        writer = Writer(f)
        writer.start()

        schema_id = writer.register_schema(
            name="foxglove.PointCloud",
            encoding=SchemaEncoding.JSONSchema,
            data=POINTCLOUD_SCHEMA,
        )

        channel_id = writer.register_channel(
            topic="/lidar/points",
            message_encoding=MessageEncoding.JSON,
            schema_id=schema_id,
        )

        for frame in frames:
            msg, log_time_ns = frame_to_pointcloud_msg(frame, rebase_offset)
            writer.add_message(
                channel_id,
                log_time=log_time_ns,
                data=json.dumps(msg).encode("utf-8"),
                publish_time=log_time_ns,
            )

        writer.finish()

    total_points = sum(len(f) for f in frames)
    print(
        f"Wrote {len(frames)} frames ({total_points:,} points) to {output_path}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Hesai JT128 pcap to MCAP for Foxglove visualization.",
    )
    parser.add_argument("pcap", type=Path, help="Input pcap file path")
    parser.add_argument(
        "-c", "--calibration", type=Path, default=None,
        help="Per-unit angle correction CSV from Hesai (default: nominal values)",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Output .mcap file path (default: input stem + .mcap)",
    )
    parser.add_argument(
        "--rebase", action="store_true",
        help="Rebase timestamps to wall clock if sensor clock is unsynced (pre-2000)",
    )
    parser.add_argument(
        "--start-time", type=str, default=None,
        help="ISO 8601 start time for rebasing (default: current time). "
             'Example: "2026-05-06T14:30:00"',
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Maximum number of frames to convert (default: all)",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = args.pcap.with_suffix(".mcap")

    start_time = None
    if args.start_time is not None:
        start_time = datetime.fromisoformat(args.start_time)
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

    convert(
        pcap_path=args.pcap,
        output_path=args.output,
        calibration_path=args.calibration,
        rebase=args.rebase,
        start_time=start_time,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
