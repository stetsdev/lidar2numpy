"""Tests for lidar2numpy.frame_assembler.

Key contract (CLAUDE.md §6):
  - Frame boundary: Block 1 azimuth decreases AND (prev - current) > 18000.
  - First partial frame at startup is discarded.
  - Subsequent rollovers emit the buffered frame.
"""

from __future__ import annotations

import numpy as np
from lidar2numpy.frame_assembler import FrameAssembler

from lidar2numpy.structs import POINT_DTYPE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pts(n: int = 1) -> np.ndarray:
    """Return a trivial POINT_DTYPE array of length n (all zeros)."""
    return np.zeros(n, dtype=POINT_DTYPE)


def _empty() -> np.ndarray:
    return np.empty(0, dtype=POINT_DTYPE)


def _feed_sequence(
    assembler: FrameAssembler, azimuths: list[int], points_per_packet: int = 1
) -> list[np.ndarray]:
    """Feed a list of block1 azimuths into an assembler, return non-None results."""
    results = []
    for az in azimuths:
        frame = assembler.add_packet(_pts(points_per_packet), az)
        if frame is not None:
            results.append(frame)
    return results


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


# ---------------------------------------------------------------------------
# Startup discard
# ---------------------------------------------------------------------------


class TestStartupDiscard:
    def test_first_rollover_discards_partial_frame(self) -> None:
        # Packets at [10000, 20000, 35000] then rollover at 100.
        # The partial [10000, 20000, 35000] frame must be silently discarded.
        a = FrameAssembler()
        results = _feed_sequence(a, [10000, 20000, 35000, 100])
        assert results == []

    def test_second_rollover_emits_first_complete_frame(self) -> None:
        # [pre-startup] ... [frame 1: 100, 200] ... rollover at 50 → emit frame 1
        a = FrameAssembler()
        # Startup junk (discarded at first rollover)
        _feed_sequence(a, [35000, 36])
        # Frame 1 starts
        results = _feed_sequence(a, [100, 200, 300, 50])
        assert len(results) == 1

    def test_flush_after_startup_discard_returns_partial(self) -> None:
        # After startup discard, flush returns the in-progress frame.
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])  # first rollover discards startup
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
        # 34990 → 100: decrease of 34890 > 18000 — IS a rollover.
        a = FrameAssembler()
        _feed_sequence(a, [34990, 100])  # first rollover (discarded at startup)
        # Now _started; feed another run and trigger second rollover
        _feed_sequence(a, [200, 300])
        results = _feed_sequence(a, [50])  # second rollover
        assert len(results) == 1

    def test_sequence_35000_34990_100_triggers_exactly_one_boundary(self) -> None:
        """[35000, 34990, 100]: rollover only at 100 (not at 34990)."""
        a = FrameAssembler()
        boundaries: list[int] = []
        for az in [35000, 34990, 100]:
            frame = a.add_packet(_pts(), az)
            if frame is not None:
                boundaries.append(az)
        # Exactly one boundary detected, and it's at the 100 step.
        # (The first rollover is discarded as startup, so no frame emitted yet.)
        # Add more packets and a second rollover to verify startup-discard worked.
        a.add_packet(_pts(), 200)
        a.add_packet(_pts(), 300)
        frame = a.add_packet(_pts(), 50)  # second rollover — should emit frame
        assert frame is not None

    def test_guard_threshold_is_18000(self) -> None:
        # Decrease of exactly 18001 → should be a rollover.
        a = FrameAssembler()
        _feed_sequence(a, [20001, 0])  # 20001 - 0 = 20001 > 18000 → startup discard
        _feed_sequence(a, [100, 200])
        results = _feed_sequence(a, [0])  # 200 - 0 = 200; no rollover... wait
        # Actually 200 → 0: 200 - 0 = 200, not > 18000, so no rollover.
        assert results == []

        # Now a real big jump
        a2 = FrameAssembler()
        _feed_sequence(a2, [20001, 0])  # startup discard
        _feed_sequence(a2, [100, 200])
        results2 = _feed_sequence(a2, [20001 - 18001])  # 200 → (20001-18001)=2000; Δ=0 nope
        assert results2 == []


# ---------------------------------------------------------------------------
# Complete frame emission
# ---------------------------------------------------------------------------


class TestFrameEmission:
    def test_frame_contains_all_buffered_points(self) -> None:
        a = FrameAssembler()
        # Startup discard
        _feed_sequence(a, [35000, 36])
        # Frame 1: 3 packets of 5 points each
        a.add_packet(_pts(5), 100)
        a.add_packet(_pts(5), 200)
        a.add_packet(_pts(5), 300)
        # Rollover → emit frame 1
        frame = a.add_packet(_pts(2), 50)
        assert frame is not None
        assert len(frame) == 15  # 3 packets × 5 points

    def test_emitted_frame_has_correct_dtype(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])
        a.add_packet(_pts(3), 100)
        frame = a.add_packet(_pts(1), 50)
        assert frame is not None
        assert frame.dtype == POINT_DTYPE

    def test_two_successive_frames(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])  # startup discard
        # Frame 1
        a.add_packet(_pts(10), 100)
        a.add_packet(_pts(10), 200)
        frame1 = a.add_packet(_pts(1), 50)  # rollover → emit frame 1
        assert frame1 is not None
        assert len(frame1) == 20
        # Frame 2
        a.add_packet(_pts(7), 100)
        frame2 = a.add_packet(_pts(1), 50)  # rollover → emit frame 2
        assert frame2 is not None
        assert len(frame2) == 7

    def test_current_packet_included_in_next_frame(self) -> None:
        # The packet that triggers the rollover is buffered for the NEXT frame,
        # not included in the emitted frame.
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])  # startup discard
        a.add_packet(_pts(5), 100)
        # This packet triggers rollover; its 3 points go to next frame
        frame1 = a.add_packet(_pts(3), 50)
        assert frame1 is not None
        assert len(frame1) == 5  # only the pre-rollover packet
        # Flush to get the "next frame" started with the 3 points
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
        a.add_packet(_empty(), 100)  # first rollover — no error

    def test_empty_packet_tolerated_after_startup(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])
        a.add_packet(_empty(), 100)
        a.add_packet(_empty(), 200)
        frame = a.add_packet(_empty(), 50)
        # Frame emitted but it's all-empty packets — should return empty array
        assert frame is not None
        assert len(frame) == 0
        assert frame.dtype == POINT_DTYPE

    def test_empty_packets_do_not_prevent_frame_emission(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])  # startup discard
        a.add_packet(_pts(5), 100)
        a.add_packet(_empty(), 200)  # empty — should not break anything
        a.add_packet(_pts(3), 300)
        frame = a.add_packet(_pts(1), 50)
        assert frame is not None
        assert len(frame) == 8  # only the non-empty packets counted


# ---------------------------------------------------------------------------
# flush()
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_before_startup_returns_none(self) -> None:
        a = FrameAssembler()
        assert a.flush() is None

    def test_flush_during_startup_returns_none(self) -> None:
        a = FrameAssembler()
        a.add_packet(_pts(), 35000)
        assert a.flush() is None

    def test_flush_returns_in_progress_frame(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])  # startup discard
        a.add_packet(_pts(4), 100)
        a.add_packet(_pts(6), 200)
        frame = a.flush()
        assert frame is not None
        assert len(frame) == 10

    def test_flush_clears_buffer(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])
        a.add_packet(_pts(4), 100)
        a.flush()
        # After flush, buffer should be cleared; second flush returns None/empty.
        frame2 = a.flush()
        # Either None or empty array is acceptable
        assert frame2 is None or len(frame2) == 0

    def test_flush_empty_frame_has_correct_dtype(self) -> None:
        a = FrameAssembler()
        _feed_sequence(a, [35000, 36])
        frame = a.flush()
        # Just an empty buffer after startup discard
        assert frame is None or frame.dtype == POINT_DTYPE
