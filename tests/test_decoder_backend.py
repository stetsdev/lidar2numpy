"""Backend selection and production-enforcement tests."""

from __future__ import annotations

import numpy as np
import pytest
from _packet_builder import build_packet

import lidar2numpy
from lidar2numpy import Decoder, default_calibration
from lidar2numpy import _decoder_backend as backend_module


def _complete_frame(decoder: Decoder) -> np.ndarray:
    channels = {0: (250, 100, 0), 127: (500, 120, 0xC5)}
    for azimuth in (35_000, 100, 1_000, 35_000):
        decoder.feed(build_packet(block1_az=azimuth, block1_channels=channels))
    frame = decoder.feed(build_packet(block1_az=100, block1_channels=channels))
    assert frame is not None
    return frame


def test_public_backend_identity_reports_python_without_extension() -> None:
    assert lidar2numpy.decoder_backend() == "python"


def test_explicit_python_backend_matches_auto_output() -> None:
    calibration = default_calibration()
    auto_frame = _complete_frame(Decoder(calibration, output_mode="spherical", backend="auto"))
    python_decoder = Decoder(calibration, output_mode="spherical", backend="python")
    python_frame = _complete_frame(python_decoder)

    assert python_decoder.backend == "python"
    np.testing.assert_array_equal(python_frame, auto_frame)


def test_explicit_cython_backend_fails_when_extension_is_absent() -> None:
    with pytest.raises(RuntimeError, match="cython backend is unavailable"):
        Decoder(default_calibration(), output_mode="spherical", backend="cython")


def test_production_guard_fails_when_only_python_backend_is_available() -> None:
    with pytest.raises(RuntimeError, match="compiled Cython decoder backend is required"):
        lidar2numpy.require_compiled_backend()


def test_broken_compiled_import_is_not_silently_converted_to_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_loader() -> None:
        raise ImportError("undefined symbol: PyArray_API")

    monkeypatch.setattr(backend_module, "_load_compiled_feed", broken_loader)

    with pytest.raises(ImportError, match="undefined symbol"):
        backend_module.resolve_spherical_backend("auto")


@pytest.mark.parametrize("backend", ["unexpected", "", "Python"])
def test_invalid_backend_name_fails_clearly(backend: str) -> None:
    with pytest.raises(ValueError, match="backend must be"):
        Decoder(default_calibration(), output_mode="spherical", backend=backend)
