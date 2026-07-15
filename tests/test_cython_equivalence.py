"""Independent Python-versus-Cython spherical decoder equivalence tests."""

from __future__ import annotations

import numpy as np
import pytest
from _packet_builder import build_packet

from lidar2numpy import Decoder, default_calibration
from lidar2numpy import decoder as python_decoder


def _dense_frame(decoder: Decoder, *, return_mode: int) -> np.ndarray:
    channels = {index: (index + 1, index, index ^ 0xC0) for index in range(128)}
    for azimuth in (35_000, 100, 1_000, 35_000):
        decoder.feed(
            build_packet(
                return_mode=return_mode,
                block1_az=azimuth,
                block1_channels=channels,
                block2_channels=channels,
            )
        )
    frame = decoder.feed(
        build_packet(
            return_mode=return_mode,
            block1_az=100,
            block1_channels=channels,
            block2_channels=channels,
        )
    )
    assert frame is not None
    return frame


@pytest.mark.parametrize("return_mode", [0x37, 0x39])
def test_dense_cython_matches_python(return_mode: int) -> None:
    calibration = default_calibration()
    python_frame = _dense_frame(
        Decoder(calibration, output_mode="spherical", backend="python"),
        return_mode=return_mode,
    )
    cython_frame = _dense_frame(
        Decoder(calibration, output_mode="spherical", backend="cython"),
        return_mode=return_mode,
    )

    assert cython_frame.dtype == python_frame.dtype
    np.testing.assert_array_equal(cython_frame, python_frame)


def test_dense_single_return_cython_does_not_call_python_block_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_python_fill(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compiled dense path called Python _fill_spherical_block")

    monkeypatch.setattr(python_decoder, "_fill_spherical_block", fail_python_fill)
    frame = _dense_frame(
        Decoder(default_calibration(), output_mode="spherical", backend="cython"),
        return_mode=0x37,
    )

    assert len(frame) == 768


@pytest.mark.parametrize("return_mode", [0x37, 0x39])
def test_sparse_and_dual_cython_do_not_call_python_block_fill(
    monkeypatch: pytest.MonkeyPatch, return_mode: int
) -> None:
    channels_1 = {0: (250, 99, 0xC1), 127: (500, 12, 0x3F)}
    channels_2 = {3: (125, 33, 0x80)}

    def fail_python_fill(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compiled path called Python _fill_spherical_block")

    monkeypatch.setattr(python_decoder, "_fill_spherical_block", fail_python_fill)
    decoder = Decoder(default_calibration(), output_mode="spherical", backend="cython")
    for azimuth in (35_000, 100, 1_000, 35_000):
        decoder.feed(
            build_packet(
                return_mode=return_mode,
                block1_az=azimuth,
                block1_channels=channels_1,
                block2_channels=channels_2,
            )
        )
    frame = decoder.feed(
        build_packet(
            return_mode=return_mode,
            block1_az=100,
            block1_channels=channels_1,
            block2_channels=channels_2,
        )
    )

    assert frame is not None
    assert len(frame) == 9
