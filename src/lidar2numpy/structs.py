"""Packet-format constants, return-mode codes, and the output array dtype.

All values come from ``docs/jt128-packet-format.md``. Treat that document as the
authoritative reference; the literals here are derived from it.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

UDP_PAYLOAD_SIZE: int = 1100
PRE_HEADER_SIZE: int = 6
HEADER_SIZE: int = 6
BODY_SIZE: int = 1032
TAIL_SIZE: int = 56

NUM_CHANNELS: int = 128
NUM_BLOCKS: int = 2

# Distance reported in the per-channel record is in 4 mm units (header byte
# "Dis Unit" = 0x04). Multiply raw uint16 by this factor to get metres.
DIS_UNIT_M: float = 0.004

SOP: bytes = b"\xee\xff"
PROTO_VERSION: tuple[int, int] = (1, 4)

# Tail field offsets, relative to the start of the 56-byte tail. The 11 leading
# reserved bytes and the 1-byte Working Mode push Return Mode out to offset 12.
_TAIL_OFF_RETURN_MODE: int = 12
_TAIL_OFF_MOTOR_SPEED: int = 13
_TAIL_OFF_DATETIME: int = 15
_TAIL_OFF_FRAC_SEC: int = 21

# Single-return block start times (microseconds, relative to the packet's
# tail timestamp t0). Block 1 fires 111.111 us before Block 2 in single
# return; both blocks land at t0 - 1888 us in dual return (not yet
# supported by Hesai, so only the single-return values are defined here).
BLOCK1_START_US: float = -1999.111
BLOCK2_START_US: float = -1888.0


class ReturnMode(IntEnum):
    """Return-mode codes in the tail's Return Mode byte."""

    FIRST = 0x33
    STRONGEST = 0x37
    LAST = 0x38
    LAST_STRONGEST = 0x39
    LAST_FIRST = 0x3B
    FIRST_STRONGEST = 0x3C

    @property
    def is_dual(self) -> bool:
        """True for modes that emit two returns per channel (block 1 + block 2)."""
        return self in (
            ReturnMode.LAST_STRONGEST,
            ReturnMode.LAST_FIRST,
            ReturnMode.FIRST_STRONGEST,
        )


# Output point dtype, matching the contract documented in CLAUDE.md §3.
# Unaligned (no padding) — itemsize = 28 bytes.
POINT_DTYPE: np.dtype[np.void] = np.dtype(
    [
        ("x", np.float32),
        ("y", np.float32),
        ("z", np.float32),
        ("intensity", np.float32),
        ("ring", np.uint16),
        ("timestamp", np.float64),
        ("contamination", np.uint8),
        ("noise_level", np.uint8),
    ]
)

# Spherical (polar) output dtype — emitted when output_mode="spherical".
# Stops the decode pipeline before the trig XYZ step, giving callers the
# native sensor coordinates for range-image processing (background subtraction
# etc.) before committing to the more expensive Cartesian conversion.
# channel is 1-based (1–128) to match the ring field in POINT_DTYPE.
SPHERICAL_DTYPE: np.dtype[np.void] = np.dtype(
    [
        ("channel", np.uint8),
        ("azimuth_deg", np.float32),  # calibration-corrected, degrees
        ("distance_m", np.float32),  # metres
        ("intensity", np.float32),
        ("timestamp", np.float64),
        ("contamination", np.uint8),
        ("noise_level", np.uint8),
    ]
)
