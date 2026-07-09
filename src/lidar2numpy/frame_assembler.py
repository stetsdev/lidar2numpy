"""Accumulates decoded per-packet point arrays into complete 360° frames.

Frame boundary detection uses azimuth rollover: when Block 1's azimuth
decreases by more than half a revolution (> 18 000 raw units, i.e. 180°)
relative to the previous packet, a new frame has started. This matches
hydra4's approach (``last_phase - current_phase > 18000``) and avoids
spurious splits from normal monotone-decreasing noise or jitter.

The first partial frame at startup is silently discarded; the assembler
begins emitting only after the first rollover is observed.
"""

from __future__ import annotations

import numpy as np

from .structs import POINT_DTYPE

_ROLLOVER_THRESHOLD: int = 18_000  # raw azimuth units (half revolution)


class FrameAssembler:
    """Stateful accumulator that yields one NumPy frame per 360° rotation.

    Usage::

        assembler = FrameAssembler()
        for payload in udp_payloads:
            points = decode_packet(payload, calibration)
            az = block1_azimuth(payload)
            frame = assembler.add_packet(points, az)
            if frame is not None:
                process(frame)
        # Drain the last in-progress frame at end of stream:
        final = assembler.flush()
        if final is not None:
            process(final)
    """

    def __init__(self, dtype: np.dtype = POINT_DTYPE) -> None:
        self._dtype = dtype
        self._last_az: int | None = None
        self._buffer: np.ndarray = np.empty(4096, dtype=dtype)
        self._size: int = 0
        self._has_buffered_packets: bool = False
        self._started: bool = False  # True after the first rollover is observed

    def add_packet(self, points: np.ndarray, block1_az: int) -> np.ndarray | None:
        """Add a decoded packet to the assembler.

        Parameters
        ----------
        points:
            Structured array with dtype POINT_DTYPE (may be empty).
        block1_az:
            Block 1 raw azimuth value (0–35 999, unit 0.01°).

        Returns
        -------
        np.ndarray | None
            A complete frame (POINT_DTYPE structured array) when a rollover
            is detected and the assembler has passed the startup discard
            phase. ``None`` otherwise.
        """
        frame, should_buffer = self._begin_packet(block1_az)
        if should_buffer:
            out = self._reserve(len(points))
            if len(points) > 0:
                out[:] = points
        return frame

    def _begin_packet(self, block1_az: int) -> tuple[np.ndarray | None, bool]:
        """Advance rollover state before buffering the current packet."""
        frame: np.ndarray | None = None
        if self._last_az is not None and self._is_rollover(block1_az, self._last_az):
            if self._started:
                frame = self._finish_frame()
            else:
                self._started = True

        self._last_az = block1_az
        return frame, self._started

    def flush(self) -> np.ndarray | None:
        """Return and clear the in-progress frame.

        Call at end-of-stream (e.g. end of pcap replay) to drain whatever
        partial frame is buffered. Returns ``None`` if the startup discard
        phase has not completed yet (no rollover seen) or the buffer is empty.
        """
        if not self._started or not self._has_buffered_packets:
            return None
        return self._finish_frame()

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_rollover(current_az: int, last_az: int) -> bool:
        """True when azimuth wraps from a high value back toward 0°.

        Both conditions must hold to avoid false triggers on normal
        monotone-decreasing steps (e.g. minor sensor jitter or dual-return
        blocks that share the same azimuth):
          1. current < last  (azimuth decreased)
          2. (last - current) > 18 000  (decreased by more than half a revolution)
        """
        return current_az < last_az and (last_az - current_az) > _ROLLOVER_THRESHOLD

    def _reserve(self, n_points: int) -> np.ndarray:
        self._has_buffered_packets = True
        if n_points == 0:
            return self._buffer[self._size : self._size]

        required = self._size + n_points
        if required > len(self._buffer):
            new_capacity = max(required, max(1, len(self._buffer) * 2))
            new_buffer = np.empty(new_capacity, dtype=self._dtype)
            if self._size > 0:
                new_buffer[: self._size] = self._buffer[: self._size]
            self._buffer = new_buffer

        start = self._size
        self._size = required
        return self._buffer[start:required]

    def _finish_frame(self) -> np.ndarray:
        if self._size == 0:
            self._has_buffered_packets = False
            return np.empty(0, dtype=self._dtype)

        frame = self._buffer[: self._size]
        self._buffer = np.empty(len(self._buffer), dtype=self._dtype)
        self._size = 0
        self._has_buffered_packets = False
        return frame
