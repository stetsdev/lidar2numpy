"""JT128 single-packet decoder.

Public functions:

- ``decode_packet(payload, calibration) -> np.ndarray``
      Parse one 1100-byte UDP payload into a POINT_DTYPE structured array.
      Every packet is decoded unconditionally; the FrameAssembler decides
      when to emit a complete frame.

- ``block1_azimuth(payload) -> int``
      Cheap 2-byte read of Block 1's raw azimuth (uint16, unit 0.01°).
      Called by Decoder.feed after decode_packet to give the azimuth value
      to FrameAssembler without a second full decode.

- ``to_cartesian(spherical, calibration) -> np.ndarray``
      Convert a SPHERICAL_DTYPE array (or any subset) to POINT_DTYPE by
      applying the same calibration-aware trig used by decode_packet.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
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
    SPHERICAL_DTYPE,
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


# ── Shared helpers ────────────────────────────────────────────────────────────


def _validate_payload(payload: bytes) -> None:
    """Raise ValueError / NotImplementedError for malformed or unsupported packets."""
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


def _parse_tail(payload: bytes) -> tuple[ReturnMode, float]:
    """Extract return mode and t0 (Unix epoch seconds) from the tail.

    Must be called after ``_validate_payload``.
    """
    return_mode_byte: int = payload[_TAIL_OFFSET + _TAIL_OFF_RETURN_MODE]
    try:
        return_mode = ReturnMode(return_mode_byte)
    except ValueError:
        raise ValueError(f"Unknown return mode byte: 0x{return_mode_byte:02X}") from None

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
    if frac_us >= 1_000_000:
        raise ValueError(
            f"Fractional microseconds field is {frac_us}; expected < 1 000 000"
        )
    t0: float = datetime(
        year_ + 1900,
        month,
        day,
        hour,
        minute,
        second,
        frac_us,
        tzinfo=timezone.utc,
    ).timestamp()
    return return_mode, t0


def _spherical_to_xyz(
    dist_m: np.ndarray,
    horiz_rad: np.ndarray,
    elev_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert polar coordinates to Cartesian XYZ (float64 arrays).

    Single implementation shared by decode_packet and to_cartesian so the
    two output paths cannot drift apart.

    Coordinate convention (Y = 0° azimuth, clockwise positive):
        x = dist * cos(elev) * sin(horiz)
        y = dist * cos(elev) * cos(horiz)
        z = dist * sin(elev)
    """
    cos_elev = np.cos(elev_rad)
    return (
        dist_m * cos_elev * np.sin(horiz_rad),
        dist_m * cos_elev * np.cos(horiz_rad),
        dist_m * np.sin(elev_rad),
    )


# ── Shared block extraction ──────────────────────────────────────────────────


@dataclass
class _BlockData:
    """Intermediate per-block arrays shared by both decode paths."""

    ring_0: np.ndarray  # 0-based channel indices of valid returns
    valid: np.ndarray  # _CHANNEL_DTYPE subset (distance > 0 only)
    dist_m: np.ndarray  # float64, metres
    horiz_deg: np.ndarray  # float64, calibration-corrected
    timestamps: np.ndarray  # float64, Unix epoch


def _extract_blocks(
    payload: bytes, calibration: Calibration, return_mode: ReturnMode, t0: float
) -> list[_BlockData]:
    """Parse both blocks from a validated payload into intermediate arrays."""
    blocks: list[_BlockData] = []
    block_start_us = (
        (BLOCK2_START_US, BLOCK2_START_US) if return_mode.is_dual else _BLOCK_START_US
    )
    for blk in range(2):
        az_raw: int = struct.unpack_from("<H", payload, _BLOCK_AZ_OFFSETS[blk])[0]
        channels: np.ndarray = np.frombuffer(
            payload, dtype=_CHANNEL_DTYPE, count=128, offset=_BLOCK_CH_OFFSETS[blk]
        )

        mask: np.ndarray = channels["distance"] > 0
        if not np.any(mask):
            continue

        ring_0: np.ndarray = np.where(mask)[0]
        valid = channels[mask]

        dist_m: np.ndarray = valid["distance"].astype(np.float64) * DIS_UNIT_M
        horiz_deg: np.ndarray = az_raw * 0.01 + calibration.azimuth_offsets_deg[ring_0]

        block_start_s: float = block_start_us[blk] * 1e-6
        timestamps: np.ndarray = t0 + block_start_s + FIRING_OFFSETS_S[ring_0]

        blocks.append(_BlockData(ring_0, valid, dist_m, horiz_deg, timestamps))
    return blocks


# ── Public decode functions ───────────────────────────────────────────────────


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
    ValueError
        If the Return Mode byte is unknown.
    """
    _validate_payload(payload)
    return_mode, t0 = _parse_tail(payload)

    block_arrays: list[np.ndarray] = []
    for bd in _extract_blocks(payload, calibration, return_mode, t0):
        horiz_rad = np.deg2rad(bd.horiz_deg)
        elev_rad = calibration.elevations_rad[bd.ring_0]
        x, y, z = _spherical_to_xyz(bd.dist_m, horiz_rad, elev_rad)

        n = len(bd.valid)
        arr = np.empty(n, dtype=POINT_DTYPE)
        arr["x"] = x.astype(np.float32)
        arr["y"] = y.astype(np.float32)
        arr["z"] = z.astype(np.float32)
        arr["intensity"] = bd.valid["reflectivity"].astype(np.float32)
        arr["ring"] = (bd.ring_0 + 1).astype(np.uint16)
        arr["timestamp"] = bd.timestamps
        arr["contamination"] = (bd.valid["confidence"] >> 6).astype(np.uint8)
        arr["noise_level"] = (bd.valid["confidence"] & 0x3F).astype(np.uint8)

        block_arrays.append(arr)

    if not block_arrays:
        return np.empty(0, dtype=POINT_DTYPE)
    return np.concatenate(block_arrays)


def _decode_packet_spherical(payload: bytes, calibration: Calibration) -> np.ndarray:
    """Decode one payload into a SPHERICAL_DTYPE array (no trig conversion).

    Not part of the public API; accessed via Decoder(output_mode="spherical").
    """
    _validate_payload(payload)
    return_mode, t0 = _parse_tail(payload)

    block_arrays: list[np.ndarray] = []
    for bd in _extract_blocks(payload, calibration, return_mode, t0):
        n = len(bd.valid)
        arr = np.empty(n, dtype=SPHERICAL_DTYPE)
        arr["channel"] = (bd.ring_0 + 1).astype(np.uint16)
        arr["azimuth_deg"] = bd.horiz_deg.astype(np.float32)
        arr["distance_m"] = bd.dist_m.astype(np.float32)
        arr["intensity"] = bd.valid["reflectivity"].astype(np.float32)
        arr["timestamp"] = bd.timestamps
        arr["contamination"] = (bd.valid["confidence"] >> 6).astype(np.uint8)
        arr["noise_level"] = (bd.valid["confidence"] & 0x3F).astype(np.uint8)

        block_arrays.append(arr)

    if not block_arrays:
        return np.empty(0, dtype=SPHERICAL_DTYPE)
    return np.concatenate(block_arrays)


def to_cartesian(spherical: np.ndarray, calibration: Calibration) -> np.ndarray:
    """Convert a SPHERICAL_DTYPE array to a POINT_DTYPE array.

    Applies the same calibration-aware trig as the cartesian decode path.
    Accepts any subset of a spherical frame — callers can filter to foreground
    points first (e.g. background subtraction on the range image) and then
    convert only the survivors.

    Parameters
    ----------
    spherical:
        Structured array with dtype SPHERICAL_DTYPE. May be a subset/slice of
        a frame; does not need to represent a complete 360° rotation.
    calibration:
        The same Calibration object used to produce the spherical array.

    Returns
    -------
    np.ndarray
        Structured array with dtype POINT_DTYPE containing one row per input
        point. Fields other than x/y/z are carried over directly from the
        spherical array (intensity, timestamp, contamination, noise_level).
        The ring field equals the channel field from the spherical array.
    """
    channel_idx = spherical["channel"].astype(np.intp) - 1  # 0-based index into calibration
    horiz_rad = np.deg2rad(spherical["azimuth_deg"].astype(np.float64))
    elev_rad = calibration.elevations_rad[channel_idx]
    dist_m = spherical["distance_m"].astype(np.float64)

    x, y, z = _spherical_to_xyz(dist_m, horiz_rad, elev_rad)

    n = len(spherical)
    arr = np.empty(n, dtype=POINT_DTYPE)
    arr["x"] = x.astype(np.float32)
    arr["y"] = y.astype(np.float32)
    arr["z"] = z.astype(np.float32)
    arr["intensity"] = spherical["intensity"]
    arr["ring"] = spherical["channel"].astype(np.uint16)
    arr["timestamp"] = spherical["timestamp"]
    arr["contamination"] = spherical["contamination"]
    arr["noise_level"] = spherical["noise_level"]
    return arr


def block1_azimuth(payload: bytes) -> int:
    """Return Block 1's raw azimuth value (uint16 little-endian, unit 0.01°).

    Reads only 2 bytes from the start of the body (offset 12). Called by
    ``Decoder.feed`` *after* ``decode_packet`` to pass the azimuth to
    ``FrameAssembler`` without re-parsing the full packet. The double read
    is intentional — two bytes is trivially cheap.
    """
    return int(struct.unpack_from("<H", payload, _BLOCK1_AZ_OFFSET)[0])
