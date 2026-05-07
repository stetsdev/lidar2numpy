"""Tests for lidar2numpy.calibration.

CSV format handled:
  - Optional first row starting with 'EEFF' (skipped)
  - Required column header row: Channel,Elevation,Azimuth
  - 128 data rows: channel number (1..128), elevation (degrees), azimuth offset (degrees)
  - Optional trailing SHA-256 checksum row (validated; warn on mismatch or absence, never fail)
"""

from __future__ import annotations

import io
import textwrap
import warnings
from pathlib import Path

import numpy as np
import pytest

from lidar2numpy.calibration import Calibration, default_calibration, load_calibration

_FIXTURES = Path(__file__).parent / "fixtures"
_CAL_2368 = _FIXTURES / "lidar-Port 2368" / "angle_corrections - 2368.csv"
_CAL_2369 = _FIXTURES / "lidar-Port 2369" / "angle_corrections - 2369.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_csv_bytes(
    *,
    eeff_header: bool = False,
    num_channels: int = 128,
    include_checksum: bool = False,
    bad_checksum: bool = False,
) -> bytes:
    """Build a synthetic calibration CSV as bytes."""
    lines: list[str] = []
    if eeff_header:
        lines.append("EEFF,1,1")
    lines.append("Channel,Elevation,Azimuth")
    for ch in range(1, num_channels + 1):
        elev = float(ch) * 0.5
        az = float(ch) * -0.1
        lines.append(f"{ch},{elev:.6f},{az:.6f}")
    if include_checksum:
        import hashlib

        body = "\n".join(lines) + "\n"
        digest = hashlib.sha256(body.encode()).hexdigest()
        # Must be exactly 64 hex chars to be detected as a checksum row.
        checksum = "a" * 63 + "b" if bad_checksum else digest
        lines.append(f"{checksum},,")
    return "\n".join(lines).encode()


# ---------------------------------------------------------------------------
# Calibration dataclass
# ---------------------------------------------------------------------------


class TestCalibrationDataclass:
    def test_elevations_shape(self) -> None:
        cal = default_calibration()
        assert cal.elevations_rad.shape == (128,)

    def test_azimuth_offsets_shape(self) -> None:
        cal = default_calibration()
        assert cal.azimuth_offsets_deg.shape == (128,)

    def test_elevations_dtype(self) -> None:
        cal = default_calibration()
        assert cal.elevations_rad.dtype == np.float64

    def test_azimuth_offsets_dtype(self) -> None:
        cal = default_calibration()
        assert cal.azimuth_offsets_deg.dtype == np.float64

    def test_is_frozen(self) -> None:
        cal = default_calibration()
        with pytest.raises((AttributeError, TypeError)):
            cal.elevations_rad = np.zeros(128)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# default_calibration()
# ---------------------------------------------------------------------------


class TestDefaultCalibration:
    def test_channel_count(self) -> None:
        cal = default_calibration()
        assert len(cal.elevations_rad) == 128
        assert len(cal.azimuth_offsets_deg) == 128

    def test_ch1_elevation_approx(self) -> None:
        # jt128_default.csv: channel 1 elevation = -4.43070058 degrees
        cal = default_calibration()
        assert np.rad2deg(cal.elevations_rad[0]) == pytest.approx(-4.43070058, abs=1e-4)

    def test_elevations_are_in_radians(self) -> None:
        cal = default_calibration()
        # All angles should be within ±π/2 (±90°) for a hyper-hemispherical sensor.
        assert np.all(np.abs(cal.elevations_rad) <= np.pi / 2 + 1e-6)


# ---------------------------------------------------------------------------
# load_calibration() — per-unit fixture files
# ---------------------------------------------------------------------------


class TestLoadCalibration2368:
    @pytest.fixture(scope="class")
    def cal(self) -> Calibration:
        return load_calibration(_CAL_2368)

    def test_channel_count(self, cal: Calibration) -> None:
        assert len(cal.elevations_rad) == 128

    def test_ch1_elevation(self, cal: Calibration) -> None:
        # angle_corrections - 2368.csv: ch1 elevation = -3.882 degrees
        assert np.rad2deg(cal.elevations_rad[0]) == pytest.approx(-3.882, abs=1e-3)

    def test_ch128_azimuth(self, cal: Calibration) -> None:
        # angle_corrections - 2368.csv: ch128 azimuth = 105.289001 degrees
        assert cal.azimuth_offsets_deg[127] == pytest.approx(105.289, abs=1e-2)

    def test_loads_with_checksum_field(self, cal: Calibration) -> None:
        # Per-unit file has a 64-hex-char trailing field.  Hesai's exact
        # checksum algorithm is undocumented; we verify the file loads (no
        # ValueError) rather than asserting zero warnings, since a mismatch
        # only ever warns, never fails (CLAUDE.md §5).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = load_calibration(_CAL_2368)
        assert len(result.elevations_rad) == 128


# ---------------------------------------------------------------------------
# load_calibration() — synthetic CSV edge cases
# ---------------------------------------------------------------------------


class TestLoadCalibrationEdgeCases:
    def test_eeff_header_row_skipped(self) -> None:
        data = _make_csv_bytes(eeff_header=True)
        cal = load_calibration(io.BytesIO(data))
        assert len(cal.elevations_rad) == 128

    def test_without_eeff_header(self) -> None:
        data = _make_csv_bytes(eeff_header=False)
        cal = load_calibration(io.BytesIO(data))
        assert len(cal.elevations_rad) == 128

    def test_with_valid_checksum_no_warning(self) -> None:
        data = _make_csv_bytes(include_checksum=True, bad_checksum=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            load_calibration(io.BytesIO(data))

    def test_bad_checksum_emits_warning(self) -> None:
        data = _make_csv_bytes(include_checksum=True, bad_checksum=True)
        with pytest.warns(UserWarning, match="checksum"):
            load_calibration(io.BytesIO(data))

    def test_missing_checksum_emits_warning(self) -> None:
        data = _make_csv_bytes(include_checksum=False)
        with pytest.warns(UserWarning, match="checksum"):
            load_calibration(io.BytesIO(data))

    def test_missing_checksum_still_loads(self) -> None:
        data = _make_csv_bytes(include_checksum=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = load_calibration(io.BytesIO(data))
        assert len(cal.elevations_rad) == 128

    def test_wrong_channel_count_raises(self) -> None:
        data = _make_csv_bytes(num_channels=64)
        with pytest.raises(ValueError, match="128"):
            load_calibration(io.BytesIO(data))

    def test_channels_stored_sorted_by_channel_number(self) -> None:
        # Synthetic CSV has channels 1..128 in order; verify index 0 = ch1.
        data = _make_csv_bytes()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = load_calibration(io.BytesIO(data))
        # ch1 elevation = 1 * 0.5 = 0.5 degrees
        assert np.rad2deg(cal.elevations_rad[0]) == pytest.approx(0.5, abs=1e-6)

    def test_load_from_path_string(self, tmp_path: Path) -> None:
        p = tmp_path / "cal.csv"
        data = _make_csv_bytes()
        p.write_bytes(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = load_calibration(str(p))
        assert len(cal.elevations_rad) == 128

    def test_load_from_path_object(self, tmp_path: Path) -> None:
        p = tmp_path / "cal.csv"
        p.write_bytes(_make_csv_bytes())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = load_calibration(p)
        assert len(cal.elevations_rad) == 128

    def test_duplicate_channels_raises(self) -> None:
        lines = ["Channel,Elevation,Azimuth"]
        for ch in range(1, 128):
            lines.append(f"{ch},1.0,0.5")
        lines.append("1,2.0,0.1")  # duplicate channel 1
        data = "\n".join(lines).encode()
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            load_calibration(io.BytesIO(data))

    def test_checksum_line_not_counted_as_channel(self) -> None:
        # A 64-hex-char row at the end should not be treated as a channel row.
        csv_text = textwrap.dedent(
            """\
            Channel,Elevation,Azimuth
            """
        )
        for ch in range(1, 129):
            csv_text += f"{ch},{ch * 0.5:.4f},{ch * -0.1:.4f}\n"
        checksum = "a" * 64 + ",,"
        csv_text += checksum + "\n"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = load_calibration(io.BytesIO(csv_text.encode()))
        assert len(cal.elevations_rad) == 128
