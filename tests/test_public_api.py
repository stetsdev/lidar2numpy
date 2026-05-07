"""Smoke tests for the lidar2numpy public API (__init__.py).

Verifies that all documented symbols are importable from the top-level
package and that Decoder.feed() + flush() work end-to-end with the
default calibration.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _packet_builder import build_packet

import lidar2numpy
from lidar2numpy import (
    POINT_DTYPE,
    Calibration,
    Decoder,
    FrameAssembler,
    ReturnMode,
    block1_azimuth,
    decode_packet,
    default_calibration,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_CAL_2368 = _FIXTURES / "lidar-Port 2368" / "angle_corrections - 2368.csv"

# ---------------------------------------------------------------------------
# Import surface
# ---------------------------------------------------------------------------


class TestPublicImports:
    def test_all_symbols_in___all__(self) -> None:
        missing = {
            "POINT_DTYPE",
            "ReturnMode",
            "Calibration",
            "default_calibration",
            "load_calibration",
            "decode_packet",
            "block1_azimuth",
            "FrameAssembler",
            "Decoder",
        } - set(lidar2numpy.__all__)
        assert not missing, f"Missing from __all__: {missing}"

    def test_point_dtype_importable(self) -> None:
        assert POINT_DTYPE is not None

    def test_return_mode_importable(self) -> None:
        assert ReturnMode.STRONGEST == 0x37

    def test_calibration_importable(self) -> None:
        cal = default_calibration()
        assert isinstance(cal, Calibration)

    def test_frame_assembler_importable(self) -> None:
        a = FrameAssembler()
        assert a is not None


# ---------------------------------------------------------------------------
# Decoder class
# ---------------------------------------------------------------------------


class TestDecoderClass:
    def test_default_calibration_used_when_none(self) -> None:
        dec = Decoder()
        assert dec._calibration is not None
        assert len(dec._calibration.elevations_rad) == 128

    def test_calibration_from_path(self) -> None:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = Decoder(_CAL_2368)
        assert len(dec._calibration.elevations_rad) == 128

    def test_calibration_from_calibration_object(self) -> None:
        cal = default_calibration()
        dec = Decoder(cal)
        assert dec._calibration is cal

    def test_feed_returns_none_before_startup(self) -> None:
        dec = Decoder()
        pkt = build_packet(block1_az=1000)
        assert dec.feed(pkt) is None

    def test_feed_emits_frame_after_rollover(self) -> None:
        dec = Decoder()
        # Startup discard
        dec.feed(build_packet(block1_az=35000))
        dec.feed(build_packet(block1_az=100))
        # Build frame content
        dec.feed(build_packet(block1_az=200, block1_channels={0: (250, 100, 0)}))
        dec.feed(build_packet(block1_az=35800, block1_channels={1: (250, 100, 0)}))
        # Rollover → emit frame
        frame = dec.feed(build_packet(block1_az=100))
        assert frame is not None
        assert frame.dtype == POINT_DTYPE
        assert len(frame) >= 2  # at least the 2 points from our packets

    def test_flush_returns_in_progress_frame(self) -> None:
        dec = Decoder()
        # Startup
        dec.feed(build_packet(block1_az=35000))
        dec.feed(build_packet(block1_az=100))
        # Add some points
        dec.feed(build_packet(block1_az=200, block1_channels={0: (250, 100, 0)}))
        frame = dec.flush()
        assert frame is not None
        assert len(frame) >= 1

    def test_feed_validates_payload(self) -> None:
        dec = Decoder()
        with pytest.raises(ValueError):
            dec.feed(b"\x00" * 1099)

    def test_decode_packet_and_block1_azimuth_importable(self) -> None:
        pkt = build_packet(block1_az=5000)
        assert block1_azimuth(pkt) == 5000
        result = decode_packet(pkt, default_calibration())
        assert isinstance(result, np.ndarray)
        assert result.dtype == POINT_DTYPE
