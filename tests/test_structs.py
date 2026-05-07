from __future__ import annotations

import numpy as np
import pytest

from lidar2numpy.structs import (
    BLOCK1_START_US,
    BLOCK2_START_US,
    BODY_SIZE,
    DIS_UNIT_M,
    HEADER_SIZE,
    NUM_BLOCKS,
    NUM_CHANNELS,
    POINT_DTYPE,
    PRE_HEADER_SIZE,
    PROTO_VERSION,
    SOP,
    TAIL_SIZE,
    UDP_PAYLOAD_SIZE,
    ReturnMode,
    _TAIL_OFF_DATETIME,
    _TAIL_OFF_FRAC_SEC,
    _TAIL_OFF_MOTOR_SPEED,
    _TAIL_OFF_RETURN_MODE,
)


class TestPacketSizes:
    """Constants from docs/jt128-packet-format.md §Packet Layout."""

    def test_udp_payload_is_1100_bytes(self) -> None:
        assert UDP_PAYLOAD_SIZE == 1100

    def test_section_sizes_sum_to_payload(self) -> None:
        assert PRE_HEADER_SIZE + HEADER_SIZE + BODY_SIZE + TAIL_SIZE == UDP_PAYLOAD_SIZE

    def test_individual_section_sizes(self) -> None:
        assert PRE_HEADER_SIZE == 6
        assert HEADER_SIZE == 6
        assert BODY_SIZE == 1032
        assert TAIL_SIZE == 56

    def test_channel_and_block_counts(self) -> None:
        assert NUM_CHANNELS == 128
        assert NUM_BLOCKS == 2

    def test_distance_unit_is_4mm(self) -> None:
        assert DIS_UNIT_M == 0.004

    def test_sop_marker(self) -> None:
        assert SOP == b"\xee\xff"

    def test_protocol_version(self) -> None:
        assert PROTO_VERSION == (1, 4)


class TestTailOffsets:
    """Tail field offsets per docs/jt128-packet-format.md §Tail.

    Layout (relative to tail start, total 56 bytes):
        [0..10] Reserved (11 bytes)
        [11]    Working Mode
        [12]    Return Mode
        [13..14] Motor Speed (uint16)
        [15..20] Date & Time (6 × uint8)
        [21..24] UTC Fractional Seconds (uint32 microseconds)
    """

    def test_return_mode_offset(self) -> None:
        assert _TAIL_OFF_RETURN_MODE == 12

    def test_motor_speed_offset(self) -> None:
        assert _TAIL_OFF_MOTOR_SPEED == 13

    def test_datetime_offset(self) -> None:
        assert _TAIL_OFF_DATETIME == 15

    def test_frac_sec_offset(self) -> None:
        assert _TAIL_OFF_FRAC_SEC == 21

    def test_offsets_within_tail(self) -> None:
        assert 0 <= _TAIL_OFF_RETURN_MODE < TAIL_SIZE
        assert 0 <= _TAIL_OFF_MOTOR_SPEED < TAIL_SIZE
        assert 0 <= _TAIL_OFF_DATETIME < TAIL_SIZE
        assert 0 <= _TAIL_OFF_FRAC_SEC + 4 <= TAIL_SIZE


class TestBlockStartOffsets:
    """Per Step 3 of the spec / packet format reference §Block start times.

    Single return: Block 1 = t0 - 111.111 - 1888 us = t0 - 1999.111 us
                   Block 2 = t0 - 1888 us
    """

    def test_block1_start_us(self) -> None:
        assert BLOCK1_START_US == pytest.approx(-1999.111)

    def test_block2_start_us(self) -> None:
        assert BLOCK2_START_US == pytest.approx(-1888.0)

    def test_block1_earlier_than_block2(self) -> None:
        assert BLOCK1_START_US < BLOCK2_START_US


class TestReturnMode:
    """Return mode codes per docs/jt128-packet-format.md §Return Mode codes."""

    def test_first_code(self) -> None:
        assert ReturnMode.FIRST == 0x33

    def test_strongest_code(self) -> None:
        assert ReturnMode.STRONGEST == 0x37

    def test_last_code(self) -> None:
        assert ReturnMode.LAST == 0x38

    def test_last_strongest_code(self) -> None:
        assert ReturnMode.LAST_STRONGEST == 0x39

    def test_last_first_code(self) -> None:
        assert ReturnMode.LAST_FIRST == 0x3B

    def test_first_strongest_code(self) -> None:
        assert ReturnMode.FIRST_STRONGEST == 0x3C

    @pytest.mark.parametrize(
        "mode",
        [ReturnMode.FIRST, ReturnMode.STRONGEST, ReturnMode.LAST],
    )
    def test_single_returns_are_not_dual(self, mode: ReturnMode) -> None:
        assert mode.is_dual is False

    @pytest.mark.parametrize(
        "mode",
        [ReturnMode.LAST_STRONGEST, ReturnMode.LAST_FIRST, ReturnMode.FIRST_STRONGEST],
    )
    def test_dual_returns_report_dual(self, mode: ReturnMode) -> None:
        assert mode.is_dual is True

    def test_construct_from_byte(self) -> None:
        assert ReturnMode(0x37) is ReturnMode.STRONGEST

    def test_unknown_byte_raises(self) -> None:
        with pytest.raises(ValueError):
            ReturnMode(0x42)


class TestPointDtype:
    """POINT_DTYPE matches the output contract in CLAUDE.md §3."""

    def test_field_names_in_order(self) -> None:
        assert POINT_DTYPE.names == (
            "x",
            "y",
            "z",
            "intensity",
            "ring",
            "timestamp",
            "contamination",
            "noise_level",
        )

    @pytest.mark.parametrize(
        ("field", "expected"),
        [
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("intensity", np.float32),
            ("ring", np.uint16),
            ("timestamp", np.float64),
            ("contamination", np.uint8),
            ("noise_level", np.uint8),
        ],
    )
    def test_field_dtypes(self, field: str, expected: type) -> None:
        assert POINT_DTYPE.fields is not None
        assert POINT_DTYPE.fields[field][0] == np.dtype(expected)

    def test_itemsize_is_packed(self) -> None:
        # 4+4+4+4 (xyz, intensity) + 2 (ring) + 8 (timestamp) + 1+1 (confidence) = 28 bytes
        assert POINT_DTYPE.itemsize == 28

    def test_can_construct_empty_array(self) -> None:
        arr = np.empty(0, dtype=POINT_DTYPE)
        assert arr.shape == (0,)
        assert arr.dtype == POINT_DTYPE

    def test_can_construct_single_point(self) -> None:
        arr = np.array(
            [(1.0, 2.0, 3.0, 128.0, 5, 1.5e9, 1, 17)],
            dtype=POINT_DTYPE,
        )
        assert arr["x"][0] == np.float32(1.0)
        assert arr["ring"][0] == 5
        assert arr["timestamp"][0] == 1.5e9
        assert arr["contamination"][0] == 1
        assert arr["noise_level"][0] == 17
