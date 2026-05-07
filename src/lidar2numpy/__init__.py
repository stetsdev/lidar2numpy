"""lidar2numpy — decode Hesai JT128 LiDAR UDP packets into NumPy arrays.

Public API
----------
``Decoder``
    Convenience class: stream raw 1100-byte payloads in, complete 360°
    frames out.  Wraps calibration loading, decode_packet, and
    FrameAssembler into a single object.

``decode_packet(payload, calibration) -> np.ndarray``
    Decode one payload unconditionally.  For callers that manage frame
    assembly themselves.

``block1_azimuth(payload) -> int``
    Cheap 2-byte peek at Block 1's raw azimuth.  Used by Decoder.feed
    to give FrameAssembler the azimuth without re-parsing the packet.

``FrameAssembler``
    Stateful azimuth-rollover detector for callers that decode packets
    themselves and want to assemble frames manually.

``load_calibration(source) -> Calibration``
    Load a Hesai angle-correction CSV from a path or file-like object.

``default_calibration() -> Calibration``
    Load the bundled nominal Appendix A calibration.

``Calibration``
    Frozen dataclass holding per-channel elevations (radians) and
    azimuth offsets (degrees).

``POINT_DTYPE``
    NumPy dtype of the output structured array.

``ReturnMode``
    IntEnum of return-mode codes in the JT128 tail byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

from .calibration import Calibration, default_calibration, load_calibration
from .decoder import block1_azimuth, decode_packet
from .frame_assembler import FrameAssembler
from .structs import POINT_DTYPE, ReturnMode

__all__ = [
    "POINT_DTYPE",
    "ReturnMode",
    "Calibration",
    "default_calibration",
    "load_calibration",
    "decode_packet",
    "block1_azimuth",
    "FrameAssembler",
    "Decoder",
]

_CalSource = Union[str, Path, BinaryIO, Calibration, None]


class Decoder:
    """Convenience wrapper: stream raw 1100-byte payloads in, complete frames out.

    Every payload is decoded unconditionally via :func:`decode_packet`.
    :class:`FrameAssembler` decides when to emit a complete 360° frame.

    Parameters
    ----------
    calibration:
        One of:
        - A :class:`Calibration` object (used directly).
        - A ``str`` or :class:`~pathlib.Path` to a Hesai CSV file.
        - A binary file-like object containing CSV bytes.
        - ``None`` — loads the bundled nominal Appendix A calibration.

    Examples
    --------
    ::

        decoder = Decoder("angle_corrections.csv")
        for payload in udp_source:
            frame = decoder.feed(payload)
            if frame is not None:
                process(frame)
        final = decoder.flush()
        if final is not None:
            process(final)
    """

    def __init__(self, calibration: _CalSource = None) -> None:
        if isinstance(calibration, Calibration):
            self._calibration = calibration
        elif calibration is None:
            self._calibration = default_calibration()
        else:
            self._calibration = load_calibration(calibration)
        self._assembler = FrameAssembler()

    def feed(self, payload: bytes) -> np.ndarray | None:
        """Decode one payload and return a complete frame if one is ready.

        Decodes *every* packet unconditionally.  The azimuth rollover check
        in :class:`FrameAssembler` determines when to emit a frame.

        Parameters
        ----------
        payload:
            Raw UDP payload (must be exactly 1100 bytes).

        Returns
        -------
        np.ndarray | None
            A POINT_DTYPE structured array containing all valid points from
            one complete 360° rotation, or ``None`` if the current frame is
            still being accumulated.
        """
        points = decode_packet(payload, self._calibration)
        az = block1_azimuth(payload)
        return self._assembler.add_packet(points, az)

    def flush(self) -> np.ndarray | None:
        """Return and clear any in-progress frame.

        Call after the last payload to drain whatever partial frame is
        buffered (e.g. at end of a pcap replay). Returns ``None`` if
        the startup discard phase has not completed yet.
        """
        return self._assembler.flush()
