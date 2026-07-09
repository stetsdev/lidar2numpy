"""Tests for lidar2numpy.frame_assembler.

Key contract (CLAUDE.md §6):
  - Frame boundary: Block 1 azimuth decreases AND (prev - current) > 18000.
  - First partial frame at startup is discarded.
  - Subsequent rollovers emit the buffered frame.

Azimuth conventions used in these tests:
  - _STARTUP  = [35000, 36]  triggers startup discard (Δ = 34964 > 18000)
  - _WRAP_AZ  = 35800        high value just before a full-revolution wrap
  - _NEXT_AZ  = 100          low value just after the wrap (Δ = 35700 > 18000)
"""

from __future__ import annotations

import numpy as np

from lidar2numpy.frame_assembler import FrameAssembler
from lidar2numpy.structs import POINT_DTYPE, SPHERICAL_DTYPE

# ---------------------------------------------------------------------------
# Azimuth constants for realistic rollover sequences
# ---------------------------------------------------------------------------
_STARTUP: list[int] = [35000, 36]  # startup discard (Δ = 34964 > 18000)
_WRAP_AZ: int = 35800  # near end of a revolution
_NEXT_AZ: int = 100  # after the wrap (Δ = 35700 > 18000)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pts(n: int = 1) -> np.ndarray:
    """Return a trivial POINT_DTYPE array of length n (all zeros)."""
    return np.zeros(n, dtype=POINT_DTYPE)


def _empty() -> np.ndarray:
    return np.empty(0, dtype=POINT_DTYPE)


def _sph_pts(start_channel: int, n: int) -> np.ndarray:
    pts = np.empty(n, dtype=SPHERICAL_DTYPE)
    pts["channel"] = np.arange(start_channel, start_channel + n, dtype=np.uint16)
    pts["azimuth_deg"] = np.arange(n, dtype=np.float32) + start_channel
    pts["distance_m"] = np.arange(n, dtype=np.float32) * np.float32(0.004)
    pts["intensity"] = np.arange(n, dtype=np.float32)
    pts["timestamp"] = np.arange(n, dtype=np.float64) + start_channel
    pts["contamination"] = np.arange(n, dtype=np.uint8) & np.uint8(0x03)
    pts["noise_level"] = np.arange(n, dtype=np.uint8) & np.uint8(0x3F)
    return pts


def _feed_sequence(
    assembler: FrameAssembler, azimuths: list[int], points_per_packet: int = 1
) -> list[np.ndarray]:
    """Feed a list of block1 azimuths, return all non-None results."""
    results = []
    for az in azimuths:
        frame = assembler.add_packet(_pts(points_per_packet), az)
        if frame is not None:
            results.append(frame)
    return results


def _do_startup(a: FrameAssembler) -> None:
    """Prime assembler past the startup discard using empty packets.

    After this call: _started=True, buffer is empty (0 points), _last_az=36.
    """
    a.add_packet(_empty(), _STARTUP[0])
    a.add_packet(_empty(), _STARTUP[1])  # triggers startup discard


# ---------------------------------------------------------------------------
# No rollover — nothing emitted
# ---------------------------------------------------------------------------


class TestNoRollover:
    def test_monotone_increasing_emits_nothing(self) -> None:
        a = FrameAssembler()
        results = _feed_sequence(a, [100, 200, 300, 400])
        assert results == []

    def test_single_packet_emits_nothing(self) -> None:
        a = FrameAssembler()
        assert a.add_packet(_pts(), 5000) is None

    def test_same_azimuth_twice_not_rollover(self) -> None:
        # Equal azimuth — not a decrease, so no rollover.
        a = FrameAssembler()
        results = _feed_sequence(a, [5000, 5000, 5000])
        assert results == []

    def test_small_decrease_not_rollover(self) -> None:
        # 300 → 200: decreases but delta = 100 ≤ 18000 — not a wrap.
        a = FrameAssembler()
        results = _feed_sequence(a, [300, 200])
        assert results == []


# ---------------------------------------------------------------------------
# Startup discard
# ---------------------------------------------------------------------------


class TestStartupDiscard:
    def test_first_rollover_discards_partial_frame(self) -> None:
        # Packets at [10000, 20000, 35000] then rollover at 100.
        # The partial [10000, 20000, 35000] frame must be silently discarded.
        a = FrameAssembler()
        results = _feed_sequence(a, [10000, 20000, 35000, _NEXT_AZ])
        assert results == []

    def test_second_rollover_emits_first_complete_frame(self) -> None:
        # Use empty startup packets so no spurious points enter the buffer.
        a = FrameAssembler()
        _do_startup(a)
        # Frame 1: three increasing packets, then a real rollover.
        results = _feed_sequence(a, [200, 500, _WRAP_AZ, _NEXT_AZ])
        assert len(results) == 1

    def test_flush_after_startup_discard_returns_partial(self) -> None:
        # After startup discard (with empty packets), flush returns in-progress frame.
        a = FrameAssembler()
        _do_startup(a)  # empty startup — buffer is empty after this
        a.add_packet(_pts(10), 100)
        a.add_packet(_pts(10), 200)
        frame = a.flush()
        assert frame is not None
        assert len(frame) == 20


# ---------------------------------------------------------------------------
# Half-revolution guard (the key correctness test)
# ---------------------------------------------------------------------------


class TestHalfRevolutionGuard:
    def test_small_decrease_is_not_rollover(self) -> None:
        # 35000 → 34990: decrease of only 10 — must NOT trigger rollover.
        a = FrameAssembler()
        result = a.add_packet(_pts(), 35000)
        assert result is None
        result = a.add_packet(_pts(), 34990)  # Δ = 10, not a wrap
        assert result is None

    def test_large_decrease_is_rollover(self) -> None:
        # _WRAP_AZ → _NEXT_AZ: Δ = 35700 > 18000 — IS a rollover.
        a = FrameAssembler()
        _do_startup(a)  # first rollover (startup discarded)
        # Build some frame content, then trigger a real rollover.
        _feed_sequence(a, [200, 500, _WRAP_AZ])
        results = _feed_sequence(a, [_NEXT_AZ])  # second rollover
        assert len(results) == 1

    def test_sequence_35000_34990_100_triggers_exactly_one_boundary(self) -> None:
        """[35000, 34990, 100]: rollover only at 100 (not at 34990)."""
        a = FrameAssembler()
        boundaries_seen = 0
        for az in [35000, 34990, 100]:
            frame = a.add_packet(_pts(), az)
            if frame is not None:
                boundaries_seen += 1
        # The first rollover (35000→34990 is NOT a rollover; 34990→100 IS) triggers
        # the startup discard.  No frame is emitted yet (startup is discarded).
        assert boundaries_seen == 0

        # Now add more packets and trigger a second rollover to confirm _started.
        a.add_packet(_pts(), 200)
        a.add_packet(_pts(), _WRAP_AZ)
        frame = a.add_packet(_pts(), _NEXT_AZ)  # real second rollover
        assert frame is not None

    def test_guard_threshold_boundary(self) -> None:
        # Δ = 18001 → rollover.  Δ = 18000 → not a rollover.
        a = FrameAssembler()
        # Δ = 18001 (just above threshold)
        result = a.add_packet(_pts(), 18001)
        assert result is None
        # 18001 → 0: Δ = 18001 > 18000 → rollover (startup discard)
        a.add_packet(_pts(), 0)
        # Now _started=True. Build a frame and verify next big drop emits it.
        a.add_packet(_pts(5), 100)
        a.add_packet(_pts(5), _WRAP_AZ)
        frame = a.add_packet(_pts(), _NEXT_AZ)
        assert frame is not None
        # Δ = 18000 (exactly at threshold, not > 18000)
        a2 = FrameAssembler()
        a2.add_packet(_pts(), 18000)
        result2 = a2.add_packet(_pts(), 0)  # 18000 - 0 = 18000, NOT > 18000
        assert result2 is None


# ---------------------------------------------------------------------------
# Complete frame emission
# ---------------------------------------------------------------------------


class TestFrameEmission:
    def test_frame_contains_all_buffered_points(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        # Frame 1: 3 packets of 5 points each
        a.add_packet(_pts(5), 100)
        a.add_packet(_pts(5), 200)
        a.add_packet(_pts(5), _WRAP_AZ)
        # Rollover → emit frame 1 (the 2 rollover-packet points go to next frame)
        frame = a.add_packet(_pts(2), _NEXT_AZ)
        assert frame is not None
        assert len(frame) == 15  # 3 packets × 5 points

    def test_emitted_frame_has_correct_dtype(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        a.add_packet(_pts(3), 100)
        a.add_packet(_pts(3), _WRAP_AZ)
        frame = a.add_packet(_pts(1), _NEXT_AZ)
        assert frame is not None
        assert frame.dtype == POINT_DTYPE

    def test_two_successive_frames(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        # Frame 1
        a.add_packet(_pts(10), 100)
        a.add_packet(_pts(10), _WRAP_AZ)
        frame1 = a.add_packet(_pts(1), _NEXT_AZ)  # rollover → emit frame 1
        assert frame1 is not None
        assert len(frame1) == 20
        # Frame 2 (1 point carried from frame1 rollover packet)
        a.add_packet(_pts(7), 200)
        a.add_packet(_pts(7), _WRAP_AZ)
        frame2 = a.add_packet(_pts(1), _NEXT_AZ)  # rollover → emit frame 2
        assert frame2 is not None
        # frame2 = 1 (rollover packet from frame1 boundary) + 7 + 7 (interior)
        assert len(frame2) == 1 + 7 + 7

    def test_current_packet_included_in_next_frame(self) -> None:
        # The packet that triggers the rollover goes to the NEXT frame.
        a = FrameAssembler()
        _do_startup(a)
        a.add_packet(_pts(5), 100)
        a.add_packet(_pts(5), _WRAP_AZ)
        # This packet triggers rollover; its 3 points go to next frame
        frame1 = a.add_packet(_pts(3), _NEXT_AZ)
        assert frame1 is not None
        assert len(frame1) == 10  # only the two pre-rollover packets
        # Flush to get the "next frame" started with the 3 rollover points
        frame2 = a.flush()
        assert frame2 is not None
        assert len(frame2) == 3


# ---------------------------------------------------------------------------
# Empty packets
# ---------------------------------------------------------------------------


class TestEmptyPackets:
    def test_empty_packet_tolerated_before_startup(self) -> None:
        a = FrameAssembler()
        a.add_packet(_empty(), 35000)
        a.add_packet(_empty(), _NEXT_AZ)  # first rollover — no error

    def test_empty_packet_tolerated_after_startup(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        a.add_packet(_empty(), 100)
        a.add_packet(_empty(), _WRAP_AZ)
        frame = a.add_packet(_empty(), _NEXT_AZ)
        # Frame emitted but all packets were empty — empty array
        assert frame is not None
        assert len(frame) == 0
        assert frame.dtype == POINT_DTYPE

    def test_empty_packets_do_not_prevent_frame_emission(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        a.add_packet(_pts(5), 100)
        a.add_packet(_empty(), 200)  # empty — no effect on point count
        a.add_packet(_pts(3), _WRAP_AZ)
        frame = a.add_packet(_pts(1), _NEXT_AZ)
        assert frame is not None
        assert len(frame) == 8  # 5 + 3 (empty excluded from concat)


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_before_any_packet_returns_none(self) -> None:
        a = FrameAssembler()
        assert a.flush() is None

    def test_flush_before_startup_rollover_returns_none(self) -> None:
        a = FrameAssembler()
        a.add_packet(_pts(), 35000)
        assert a.flush() is None  # not yet past startup discard

    def test_flush_returns_in_progress_frame(self) -> None:
        a = FrameAssembler()
        _do_startup(a)  # empty startup — buffer starts empty
        a.add_packet(_pts(4), 100)
        a.add_packet(_pts(6), 200)
        frame = a.flush()
        assert frame is not None
        assert len(frame) == 10

    def test_flush_clears_buffer(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        a.add_packet(_pts(4), 100)
        a.flush()
        # After flush, buffer should be cleared.
        frame2 = a.flush()
        assert frame2 is None or len(frame2) == 0

    def test_flush_empty_frame_has_correct_dtype(self) -> None:
        a = FrameAssembler()
        _do_startup(a)
        frame = a.flush()
        # Buffer is empty after empty-packet startup discard.
        assert frame is None or frame.dtype == POINT_DTYPE


# ---------------------------------------------------------------------------
# Concatenation parity guardrail
# ---------------------------------------------------------------------------


def _legacy_concat_or_empty(buffered: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
    non_empty = [a for a in buffered if len(a) > 0]
    if not non_empty:
        return np.empty(0, dtype=dtype)
    return np.concatenate(non_empty)


def _legacy_assembler_outputs(
    packets: list[tuple[np.ndarray, int]], dtype: np.dtype, *, flush: bool
) -> list[np.ndarray]:
    started = False
    last_az: int | None = None
    buffered: list[np.ndarray] = []
    outputs: list[np.ndarray] = []

    for points, azimuth in packets:
        if last_az is not None and azimuth < last_az and (last_az - azimuth) > 18_000:
            if started:
                outputs.append(_legacy_concat_or_empty(buffered, dtype))
                buffered = []
            else:
                buffered = []
                started = True
        buffered.append(points)
        last_az = azimuth

    if flush and started and buffered:
        outputs.append(_legacy_concat_or_empty(buffered, dtype))
    return outputs


def _actual_assembler_outputs(
    packets: list[tuple[np.ndarray, int]], dtype: np.dtype, *, flush: bool
) -> list[np.ndarray]:
    assembler = FrameAssembler(dtype=dtype)
    outputs: list[np.ndarray] = []
    for points, azimuth in packets:
        frame = assembler.add_packet(points, azimuth)
        if frame is not None:
            outputs.append(frame)
    if flush:
        frame = assembler.flush()
        if frame is not None:
            outputs.append(frame)
    return outputs


class TestConcatenationParity:
    def test_rollover_emits_same_spherical_frame_as_legacy_concatenate(self) -> None:
        packets = [
            (_sph_pts(1, 2), 35000),
            (_sph_pts(3, 3), 36),
            (_sph_pts(6, 4), 100),
            (np.empty(0, dtype=SPHERICAL_DTYPE), 200),
            (_sph_pts(10, 5), _WRAP_AZ),
            (_sph_pts(15, 1), _NEXT_AZ),
        ]

        expected = _legacy_assembler_outputs(packets, SPHERICAL_DTYPE, flush=False)
        actual = _actual_assembler_outputs(packets, SPHERICAL_DTYPE, flush=False)

        assert len(actual) == len(expected) == 1
        assert actual[0].dtype == SPHERICAL_DTYPE
        np.testing.assert_array_equal(actual[0], expected[0])

    def test_flush_returns_same_in_progress_frame_as_legacy_concatenate(self) -> None:
        packets = [
            (np.empty(0, dtype=SPHERICAL_DTYPE), 35000),
            (np.empty(0, dtype=SPHERICAL_DTYPE), 36),
            (_sph_pts(1, 3), 100),
            (_sph_pts(4, 2), 200),
        ]

        expected = _legacy_assembler_outputs(packets, SPHERICAL_DTYPE, flush=True)
        actual = _actual_assembler_outputs(packets, SPHERICAL_DTYPE, flush=True)

        assert len(actual) == len(expected) == 1
        np.testing.assert_array_equal(actual[0], expected[0])

    def test_empty_frame_matches_legacy_concatenate(self) -> None:
        packets = [
            (np.empty(0, dtype=SPHERICAL_DTYPE), 35000),
            (np.empty(0, dtype=SPHERICAL_DTYPE), 36),
            (np.empty(0, dtype=SPHERICAL_DTYPE), 100),
            (np.empty(0, dtype=SPHERICAL_DTYPE), _WRAP_AZ),
            (np.empty(0, dtype=SPHERICAL_DTYPE), _NEXT_AZ),
        ]

        expected = _legacy_assembler_outputs(packets, SPHERICAL_DTYPE, flush=False)
        actual = _actual_assembler_outputs(packets, SPHERICAL_DTYPE, flush=False)

        assert len(actual) == len(expected) == 1
        assert len(actual[0]) == 0
        assert actual[0].dtype == SPHERICAL_DTYPE
        np.testing.assert_array_equal(actual[0], expected[0])
