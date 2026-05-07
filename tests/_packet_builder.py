"""Synthetic JT128 packet builder for unit testing.

Builds valid 1100-byte UDP payloads with programmer-controlled field
values. All fields not supplied default to the minimum valid value (zeros
or sensor-reported fixed values). The confidence flag (header bit 5) is
enabled by default, matching real captured packets.
"""

from __future__ import annotations

import struct

# Absolute offsets derived from docs/jt128-packet-format.md
_TAIL_OFFSET: int = 1044  # pre-header(6) + header(6) + body(1032)
_BLOCK1_AZ_OFFSET: int = 12
_BLOCK1_CH_OFFSET: int = 14  # 12 + 2
_BLOCK2_AZ_OFFSET: int = 526  # 14 + 128*4
_BLOCK2_CH_OFFSET: int = 528  # 526 + 2


def build_packet(  # noqa: PLR0913
    *,
    return_mode: int = 0x37,
    block1_az: int = 0,
    block2_az: int = 100,
    # Keyed by 0-based ring index → (distance_raw, reflectivity, confidence)
    block1_channels: dict[int, tuple[int, int, int]] | None = None,
    block2_channels: dict[int, tuple[int, int, int]] | None = None,
    year: int = 125,  # year + 1900 → 2025
    month: int = 5,
    day: int = 6,
    hour: int = 12,
    minute: int = 0,
    second: int = 0,
    frac_us: int = 0,
    confidence_flag: bool = True,
) -> bytes:
    """Return a 1100-byte packet bytes object.

    Parameters
    ----------
    return_mode:
        Tail Return Mode byte (default 0x37 = Strongest, single return).
    block1_az / block2_az:
        Raw azimuth values (0–35999, unit 0.01°).
    block1_channels / block2_channels:
        Mapping from 0-based ring index to (distance_raw, reflectivity,
        confidence).  Rings not specified default to all-zero (no return).
    year:
        Date & Time Year field (year + 1900 = calendar year).
    confidence_flag:
        If False, clears header bit 5 to simulate 3-byte channel records.
    """
    buf = bytearray(1100)

    # Pre-header
    buf[0] = 0xEE
    buf[1] = 0xFF
    buf[2] = 0x01
    buf[3] = 0x04

    # Header
    buf[6] = 0x80  # channel num = 128
    buf[7] = 0x02  # block num = 2
    buf[9] = 0x04  # dis unit = 4 mm
    buf[10] = 0x01  # return num = 1 (single)
    # flags: bit[5]=confidence, bit[1]=IMU (fixed), bit[0]=UDP seq (fixed)
    buf[11] = 0x23 if confidence_flag else 0x03

    # Block 1
    struct.pack_into("<H", buf, _BLOCK1_AZ_OFFSET, block1_az)
    if block1_channels:
        for ring_0, (dist, refl, conf) in block1_channels.items():
            off = _BLOCK1_CH_OFFSET + ring_0 * 4
            struct.pack_into("<HBB", buf, off, dist, refl, conf)

    # Block 2
    struct.pack_into("<H", buf, _BLOCK2_AZ_OFFSET, block2_az)
    if block2_channels:
        for ring_0, (dist, refl, conf) in block2_channels.items():
            off = _BLOCK2_CH_OFFSET + ring_0 * 4
            struct.pack_into("<HBB", buf, off, dist, refl, conf)

    # Tail
    buf[_TAIL_OFFSET + 11] = 0x00  # Working Mode = operating
    buf[_TAIL_OFFSET + 12] = return_mode
    struct.pack_into("<H", buf, _TAIL_OFFSET + 13, 600)  # motor speed (60 RPM)
    buf[_TAIL_OFFSET + 15] = year
    buf[_TAIL_OFFSET + 16] = month
    buf[_TAIL_OFFSET + 17] = day
    buf[_TAIL_OFFSET + 18] = hour
    buf[_TAIL_OFFSET + 19] = minute
    buf[_TAIL_OFFSET + 20] = second
    struct.pack_into("<I", buf, _TAIL_OFFSET + 21, frac_us)

    return bytes(buf)
