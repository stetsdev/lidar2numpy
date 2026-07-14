"""Independent Python-versus-Cython spherical decoder equivalence tests."""

from __future__ import annotations

import numpy as np
import pytest
from _packet_builder import build_packet

from lidar2numpy import Decoder, default_calibration
from lidar2numpy import decoder as python_decoder


def _dense_frame(decoder: Decoder) -> np.ndarray:
    channels = {index: (index + 1, index, index ^ 0xC0) for index in range(128)}
    for azimuth in (35_000, 100, 1_000, 35_000):
        decoder.feed(
            build_packet(block1_az=azimuth, block1_channels=channels, block2_channels=channels)
        )
    frame = decoder.feed(
        build_packet(block1_az=100, block1_channels=channels, block2_channels=channels)
    )
    assert frame is not None
    return frame


def test_dense_single_return_cython_matches_python() -> None:
    calibration = default_calibration()
    python_frame = _dense_frame(Decoder(calibration, output_mode="spherical", backend="python"))
    cython_frame = _dense_frame(Decoder(calibration, output_mode="spherical", backend="cython"))

    assert cython_frame.dtype == python_frame.dtype
    np.testing.assert_array_equal(cython_frame, python_frame)


def test_dense_single_return_cython_does_not_call_python_block_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_python_fill(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compiled dense path called Python _fill_spherical_block")

    monkeypatch.setattr(python_decoder, "_fill_spherical_block", fail_python_fill)
    frame = _dense_frame(Decoder(default_calibration(), output_mode="spherical", backend="cython"))

    assert len(frame) == 768
