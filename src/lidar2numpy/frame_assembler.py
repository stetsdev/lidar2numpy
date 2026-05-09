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
        self._buffered: list[np.ndarray] = []
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
        frame: np.ndarray | None = None

        if self._last_az is not None and self._is_rollover(block1_az, self._last_az):
            if self._started:
                # Emit the completed frame; start fresh with the current packet.
                frame = self._concat_buffer()
                self._buffered = []
            else:
                # First rollover: discard the startup partial frame and start tracking.
                self._buffered = []
                self._started = True

        self._buffered.append(points)
        self._last_az = block1_az
        return frame

    def flush(self) -> np.ndarray | None:
        """Return and clear the in-progress frame.

        Call at end-of-stream (e.g. end of pcap replay) to drain whatever
        partial frame is buffered. Returns ``None`` if the startup discard
        phase has not completed yet (no rollover seen) or the buffer is empty.
        """
        if not self._started or not self._buffered:
            return None
        frame = self._concat_buffer()
        self._buffered = []
        return frame

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

    def _concat_buffer(self) -> np.ndarray:
        """Concatenate buffered arrays; return empty array of self._dtype if all empty."""
        non_empty = [a for a in self._buffered if len(a) > 0]
        if not non_empty:
            return np.empty(0, dtype=self._dtype)
        return np.concatenate(non_empty)
