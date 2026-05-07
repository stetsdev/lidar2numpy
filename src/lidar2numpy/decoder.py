"""JT128 single-packet decoder.

Two public functions:

- ``decode_packet(payload, calibration) -> np.ndarray``
      Parse one 1100-byte UDP payload into a POINT_DTYPE structured array.
      Every packet is decoded unconditionally; the FrameAssembler decides
      when to emit a complete frame.

- ``block1_azimuth(payload) -> int``
      Cheap 2-byte read of Block 1's raw azimuth (uint16, unit 0.01°).
      Called by Decoder.feed after decode_packet to give the azimuth value
      to FrameAssembler without a second full decode.
"""

from __future__ import annotations

import struct
from datetime import datetime, timezone

import numpy as np

from .calibration import Calibration
from .firing_times import FIRING_OFFSETS_S
from .structs import (
    _TAIL_OFF_DATETIME,
    _TAIL_OFF_FRAC_SEC,
    _TAIL_OFF_RETURN_MODE,
    BLOCK1_START_US,
    BLOCK2_START_US,
    DIS_UNIT_M,
    POINT_DTYPE,
    SOP,
    ReturnMode,
)

# Per-channel record layout with confidence enabled (header bit 5 = 1).
# Exactly 4 bytes: distance(uint16) + reflectivity(uint8) + confidence(uint8).
_CHANNEL_DTYPE: np.dtype[np.void] = np.dtype(
    [("distance", "<u2"), ("reflectivity", "u1"), ("confidence", "u1")]
)

# Absolute byte offsets within the 1100-byte payload.
_PROTO_MAJOR_OFFSET: int = 2
_PROTO_MINOR_OFFSET: int = 3
_FLAGS_OFFSET: int = 11
_BODY_OFFSET: int = 12
_BLOCK1_AZ_OFFSET: int = _BODY_OFFSET  # 12
_BLOCK1_CH_OFFSET: int = _BODY_OFFSET + 2  # 14
_BLOCK2_AZ_OFFSET: int = _BLOCK1_CH_OFFSET + 128 * 4  # 526
_BLOCK2_CH_OFFSET: int = _BLOCK2_AZ_OFFSET + 2  # 528
_TAIL_OFFSET: int = _BLOCK2_CH_OFFSET + 128 * 4 + 4  # 1044

_CONFIDENCE_FLAG: int = 0x20  # header flags bit[5]

_BLOCK_AZ_OFFSETS: tuple[int, int] = (_BLOCK1_AZ_OFFSET, _BLOCK2_AZ_OFFSET)
_BLOCK_CH_OFFSETS: tuple[int, int] = (_BLOCK1_CH_OFFSET, _BLOCK2_CH_OFFSET)
_BLOCK_START_US: tuple[float, float] = (BLOCK1_START_US, BLOCK2_START_US)


def decode_packet(payload: bytes, calibration: Calibration) -> np.ndarray:
    """Decode one 1100-byte JT128 UDP payload into a POINT_DTYPE array.

    Parameters
    ----------
    payload:
        Raw UDP payload bytes. Must be exactly 1100 bytes.
    calibration:
        Per-channel elevation and azimuth-offset angles.

    Returns
    -------
    np.ndarray
        Structured array with dtype POINT_DTYPE. Length = number of channels
        with distance > 0 across both blocks. May be empty.

    Raises
    ------
    ValueError
        If ``len(payload) != 1100``, SOP bytes are wrong, protocol version is
        unsupported, or the confidence flag (header bit 5) is not set.
    NotImplementedError
        If the Return Mode byte indicates a dual-return mode (codes 0x39,
        0x3B, 0x3C) — not yet supported.
    """
    # ── Validation ───────────────────────────────────────────────────────────
    if len(payload) != 1100:
        raise ValueError(f"JT128 payload must be exactly 1100 bytes; got {len(payload)}")
    if payload[0:2] != SOP:
        raise ValueError(
            f"Invalid SOP bytes: 0x{payload[0]:02X} 0x{payload[1]:02X} (expected 0xEE 0xFF)"
        )
    if payload[_PROTO_MAJOR_OFFSET] != 1 or payload[_PROTO_MINOR_OFFSET] != 4:
        raise ValueError(
            f"Unsupported protocol version: "
            f"{payload[_PROTO_MAJOR_OFFSET]}.{payload[_PROTO_MINOR_OFFSET]} "
            "(expected 1.4)"
        )
    if not (payload[_FLAGS_OFFSET] & _CONFIDENCE_FLAG):
        raise ValueError(
            "JT128 packet has confidence disabled (3-byte channel records); "
            "lidar2numpy v0.1 requires 4-byte records with confidence"
        )

    # ── Tail: return mode, timestamp ──────────────────────────────────────────
    return_mode_byte: int = payload[_TAIL_OFFSET + _TAIL_OFF_RETURN_MODE]
    try:
        return_mode = ReturnMode(return_mode_byte)
    except ValueError:
        raise ValueError(f"Unknown return mode byte: 0x{return_mode_byte:02X}") from None

    if return_mode.is_dual:
        raise NotImplementedError(
            f"Dual return mode 0x{return_mode_byte:02X} ({return_mode.name}) "
            "is not yet emitted by production JT128 firmware; not implemented"
        )

    year_: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    year_, month, day, hour, minute, second = struct.unpack_from(
        "<6B", payload, _TAIL_OFFSET + _TAIL_OFF_DATETIME
    )
    frac_us: int = struct.unpack_from("<I", payload, _TAIL_OFFSET + _TAIL_OFF_FRAC_SEC)[0]
    t0: float = datetime(
        year_ + 1900,
        month,
        day,
        hour,
        minute,
        second,
        frac_us % 1_000_000,
        tzinfo=timezone.utc,
    ).timestamp()

    # ── Decode body blocks ────────────────────────────────────────────────────
    block_arrays: list[np.ndarray] = []
    for blk in range(2):
        az_raw: int = struct.unpack_from("<H", payload, _BLOCK_AZ_OFFSETS[blk])[0]
        channels: np.ndarray = np.frombuffer(
            payload, dtype=_CHANNEL_DTYPE, count=128, offset=_BLOCK_CH_OFFSETS[blk]
        )

        mask: np.ndarray = channels["distance"] > 0
        if not np.any(mask):
            continue

        ring_0: np.ndarray = np.where(mask)[0]  # 0-based ring indices of valid channels
        valid = channels[mask]

        dist_m: np.ndarray = valid["distance"].astype(np.float64) * DIS_UNIT_M
        horiz_deg: np.ndarray = az_raw * 0.01 + calibration.azimuth_offsets_deg[ring_0]
        horiz_rad: np.ndarray = np.deg2rad(horiz_deg)
        elev_rad: np.ndarray = calibration.elevations_rad[ring_0]

        cos_elev: np.ndarray = np.cos(elev_rad)
        x: np.ndarray = dist_m * cos_elev * np.sin(horiz_rad)
        y: np.ndarray = dist_m * cos_elev * np.cos(horiz_rad)
        z: np.ndarray = dist_m * np.sin(elev_rad)

        block_start_s: float = _BLOCK_START_US[blk] * 1e-6
        timestamps: np.ndarray = t0 + block_start_s + FIRING_OFFSETS_S[ring_0]

        n: int = int(np.count_nonzero(mask))
        arr = np.empty(n, dtype=POINT_DTYPE)
        arr["x"] = x.astype(np.float32)
        arr["y"] = y.astype(np.float32)
        arr["z"] = z.astype(np.float32)
        arr["intensity"] = valid["reflectivity"].astype(np.float32)
        arr["ring"] = (ring_0 + 1).astype(np.uint16)
        arr["timestamp"] = timestamps
        arr["contamination"] = (valid["confidence"] >> 6).astype(np.uint8)
        arr["noise_level"] = (valid["confidence"] & 0x3F).astype(np.uint8)

        block_arrays.append(arr)

    if not block_arrays:
        return np.empty(0, dtype=POINT_DTYPE)
    return np.concatenate(block_arrays)


def block1_azimuth(payload: bytes) -> int:
    """Return Block 1's raw azimuth value (uint16 little-endian, unit 0.01°).

    Reads only 2 bytes from the start of the body (offset 12). Called by
    ``Decoder.feed`` *after* ``decode_packet`` to pass the azimuth to
    ``FrameAssembler`` without re-parsing the full packet. The double read
    is intentional — two bytes is trivially cheap.
    """
    return int(struct.unpack_from("<H", payload, _BLOCK1_AZ_OFFSET)[0])
