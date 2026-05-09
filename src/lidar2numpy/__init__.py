"""lidar2numpy — decode Hesai JT128 LiDAR UDP packets into NumPy arrays.

Public API
----------
``Decoder``
    Convenience class: stream raw 1100-byte payloads in, complete 360°
    frames out.  Wraps calibration loading, decode_packet, and
    FrameAssembler into a single object.  Accepts an optional
    ``output_mode`` parameter: ``"cartesian"`` (default) or ``"spherical"``.

``decode_packet(payload, calibration) -> np.ndarray``
    Decode one payload unconditionally into a POINT_DTYPE array.  For
    callers that manage frame assembly themselves.

``to_cartesian(spherical, calibration) -> np.ndarray``
    Convert a SPHERICAL_DTYPE array (or any subset) to POINT_DTYPE using
    the same calibration-aware trig as decode_packet.

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
    NumPy dtype of the Cartesian output structured array.

``SPHERICAL_DTYPE``
    NumPy dtype of the spherical (polar) output structured array.

``ReturnMode``
    IntEnum of return-mode codes in the JT128 tail byte.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Literal, Union

import numpy as np

from .calibration import Calibration, default_calibration, load_calibration
from .decoder import _decode_packet_spherical, block1_azimuth, decode_packet, to_cartesian
from .frame_assembler import FrameAssembler
from .structs import POINT_DTYPE, SPHERICAL_DTYPE, ReturnMode

__all__ = [
    "POINT_DTYPE",
    "SPHERICAL_DTYPE",
    "ReturnMode",
    "Calibration",
    "default_calibration",
    "load_calibration",
    "decode_packet",
    "to_cartesian",
    "block1_azimuth",
    "FrameAssembler",
    "Decoder",
]

_CalSource = Union[str, Path, BinaryIO, Calibration, None]


class Decoder:
    """Convenience wrapper: stream raw 1100-byte payloads in, complete frames out.

    Every payload is decoded unconditionally.
    :class:`FrameAssembler` decides when to emit a complete 360° frame.

    Parameters
    ----------
    calibration:
        One of:
        - A :class:`Calibration` object (used directly).
        - A ``str`` or :class:`~pathlib.Path` to a Hesai CSV file.
        - A binary file-like object containing CSV bytes.
        - ``None`` — loads the bundled nominal Appendix A calibration.
    output_mode:
        ``"cartesian"`` (default) — frames are POINT_DTYPE arrays with x/y/z
        fields. Existing callers are unaffected.
        ``"spherical"`` — frames are SPHERICAL_DTYPE arrays with channel,
        azimuth_deg, and distance_m fields. The trig XYZ step is skipped;
        call :func:`to_cartesian` on the result (or a subset) when needed.

    Examples
    --------
    Cartesian (default)::

        decoder = Decoder("angle_corrections.csv")
        for payload in udp_source:
            frame = decoder.feed(payload)
            if frame is not None:
                process(frame)
        final = decoder.flush()
        if final is not None:
            process(final)

    Spherical with foreground filtering::

        cal = load_calibration("angle_corrections.csv")
        decoder = Decoder(cal, output_mode="spherical")
        for payload in udp_source:
            sph_frame = decoder.feed(payload)
            if sph_frame is not None:
                foreground = sph_frame[~background_mask(sph_frame)]
                xyz_points = to_cartesian(foreground, cal)
                process(xyz_points)
    """

    def __init__(
        self,
        calibration: _CalSource = None,
        output_mode: Literal["cartesian", "spherical"] = "cartesian",
    ) -> None:
        if isinstance(calibration, Calibration):
            self._calibration = calibration
        elif calibration is None:
            self._calibration = default_calibration()
        else:
            self._calibration = load_calibration(calibration)

        if output_mode not in ("cartesian", "spherical"):
            raise ValueError(f"output_mode must be 'cartesian' or 'spherical'; got {output_mode!r}")
        self._output_mode = output_mode
        frame_dtype = SPHERICAL_DTYPE if output_mode == "spherical" else POINT_DTYPE
        self._assembler = FrameAssembler(dtype=frame_dtype)

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
            A structured array (POINT_DTYPE or SPHERICAL_DTYPE, depending on
            ``output_mode``) containing all valid points from one complete 360°
            rotation, or ``None`` if the current frame is still accumulating.
        """
        if self._output_mode == "spherical":
            points = _decode_packet_spherical(payload, self._calibration)
        else:
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
