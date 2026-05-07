"""Tests for lidar2numpy.decoder.

decode_packet() runs on every packet unconditionally. The rollover
decision lives in FrameAssembler and is not tested here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
from _packet_builder import build_packet

from lidar2numpy.calibration import Calibration
from lidar2numpy.decoder import block1_azimuth, decode_packet
from lidar2numpy.firing_times import FIRING_OFFSETS_S
from lidar2numpy.structs import BLOCK1_START_US, BLOCK2_START_US, POINT_DTYPE

# ---------------------------------------------------------------------------
# Calibration fixtures
# ---------------------------------------------------------------------------


def _flat_cal() -> Calibration:
    """All channels: elevation=0°, azimuth_offset=0°."""
    return Calibration(
        elevations_rad=np.zeros(128, dtype=np.float64),
        azimuth_offsets_deg=np.zeros(128, dtype=np.float64),
    )


def _elev_cal(ring_0: int, elev_deg: float) -> Calibration:
    """Single channel has a specific elevation; rest are 0°."""
    elevs = np.zeros(128, dtype=np.float64)
    elevs[ring_0] = np.deg2rad(elev_deg)
    return Calibration(
        elevations_rad=elevs,
        azimuth_offsets_deg=np.zeros(128, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# block1_azimuth
# ---------------------------------------------------------------------------


class TestBlock1Azimuth:
    def test_reads_block1_azimuth(self) -> None:
        pkt = build_packet(block1_az=9000)
        assert block1_azimuth(pkt) == 9000

    def test_zero_azimuth(self) -> None:
        pkt = build_packet(block1_az=0)
        assert block1_azimuth(pkt) == 0

    def test_max_azimuth(self) -> None:
        pkt = build_packet(block1_az=35999)
        assert block1_azimuth(pkt) == 35999

    def test_does_not_read_block2(self) -> None:
        # block2 azimuth is different; block1_azimuth must return block1's value.
        pkt = build_packet(block1_az=1000, block2_az=9000)
        assert block1_azimuth(pkt) == 1000


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


class TestValidationErrors:
    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="1100"):
            decode_packet(b"\x00" * 1099, _flat_cal())

    def test_bad_sop_raises(self) -> None:
        pkt = bytearray(build_packet())
        pkt[0] = 0xAA
        with pytest.raises(ValueError, match="SOP"):
            decode_packet(bytes(pkt), _flat_cal())

    def test_bad_protocol_version_raises(self) -> None:
        pkt = bytearray(build_packet())
        pkt[2] = 0x02
        with pytest.raises(ValueError, match="[Pp]rotocol"):
            decode_packet(bytes(pkt), _flat_cal())

    def test_confidence_flag_cleared_raises(self) -> None:
        pkt = build_packet(confidence_flag=False)
        with pytest.raises(
            ValueError,
            match="confidence disabled",
        ):
            decode_packet(pkt, _flat_cal())

    @pytest.mark.parametrize("code", [0x39, 0x3B, 0x3C])
    def test_dual_return_mode_raises(self, code: int) -> None:
        pkt = build_packet(return_mode=code)
        with pytest.raises(NotImplementedError):
            decode_packet(pkt, _flat_cal())


# ---------------------------------------------------------------------------
# Zero-distance / empty output
# ---------------------------------------------------------------------------


class TestEmptyOutput:
    def test_all_zero_distance_returns_empty(self) -> None:
        pkt = build_packet()  # all channels default to distance=0
        result = decode_packet(pkt, _flat_cal())
        assert result.dtype == POINT_DTYPE
        assert len(result) == 0

    def test_empty_array_has_correct_dtype(self) -> None:
        pkt = build_packet()
        result = decode_packet(pkt, _flat_cal())
        assert result.dtype == POINT_DTYPE


# ---------------------------------------------------------------------------
# XYZ coordinate math
# ---------------------------------------------------------------------------


class TestXYZ:
    """Hand-computed XYZ values to verify the spherical-to-Cartesian formulas.

    Coordinate system (docs/jt128-packet-format.md §XYZ):
        x = dist * cos(elev) * sin(horiz)
        y = dist * cos(elev) * cos(horiz)   ← Y = 0° azimuth
        z = dist * sin(elev)
    """

    def test_ch1_azimuth0_points_along_y(self) -> None:
        # distance = 250 * 0.004 = 1.0 m, elev=0°, horiz=0° → y=1, x=0, z=0
        pkt = build_packet(block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        # Exactly one point from block 1 (ch1, ring 0).
        pts = result[result["ring"] == 1]
        assert len(pts) == 1
        assert pts["x"][0] == pytest.approx(0.0, abs=1e-5)
        assert pts["y"][0] == pytest.approx(1.0, abs=1e-4)
        assert pts["z"][0] == pytest.approx(0.0, abs=1e-5)

    def test_ch1_azimuth90_points_along_x(self) -> None:
        # block_az=9000 → horiz=90°, elev=0° → x=1, y≈0, z=0
        pkt = build_packet(block1_az=9000, block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        pts = result[result["ring"] == 1]
        assert len(pts) == 1
        assert pts["x"][0] == pytest.approx(1.0, abs=1e-4)
        assert pts["y"][0] == pytest.approx(0.0, abs=1e-5)
        assert pts["z"][0] == pytest.approx(0.0, abs=1e-5)

    def test_ch64_elevation45(self) -> None:
        # distance = 2500 * 0.004 = 10 m, elev=45°, horiz=0°
        # z = 10 * sin(45°) = 7.0711, y = 10 * cos(45°) = 7.0711, x = 0
        cal = _elev_cal(ring_0=63, elev_deg=45.0)
        pkt = build_packet(block1_channels={63: (2500, 200, 0)})
        result = decode_packet(pkt, cal)
        pts = result[result["ring"] == 64]
        assert len(pts) == 1
        assert pts["x"][0] == pytest.approx(0.0, abs=1e-4)
        assert pts["y"][0] == pytest.approx(10.0 * np.cos(np.deg2rad(45.0)), rel=1e-4)
        assert pts["z"][0] == pytest.approx(10.0 * np.sin(np.deg2rad(45.0)), rel=1e-4)

    def test_azimuth_offset_applied(self) -> None:
        # horiz = block_az * 0.01 + azimuth_offset = 0 + 90 = 90° → x=dist
        az_cal = Calibration(
            elevations_rad=np.zeros(128),
            azimuth_offsets_deg=np.full(128, 90.0),
        )
        pkt = build_packet(block1_az=0, block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, az_cal)
        pts = result[result["ring"] == 1]
        assert pts["x"][0] == pytest.approx(1.0, abs=1e-4)
        assert pts["y"][0] == pytest.approx(0.0, abs=1e-5)

    def test_both_blocks_decoded(self) -> None:
        pkt = build_packet(
            block1_channels={0: (250, 100, 0)},
            block2_channels={1: (250, 100, 0)},
        )
        result = decode_packet(pkt, _flat_cal())
        # Should have one point from block 1 (ring=1) and one from block 2 (ring=2)
        assert len(result) == 2
        rings = set(result["ring"].tolist())
        assert rings == {1, 2}

    def test_coordinates_are_float32(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["x"].dtype == np.float32
        assert result["y"].dtype == np.float32
        assert result["z"].dtype == np.float32


# ---------------------------------------------------------------------------
# Ring field
# ---------------------------------------------------------------------------


class TestRingField:
    def test_ring_is_1_indexed(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["ring"][0] == 1

    def test_ring_127_maps_to_128(self) -> None:
        pkt = build_packet(block1_channels={127: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["ring"][0] == 128

    def test_ring_dtype_is_uint16(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["ring"].dtype == np.uint16


# ---------------------------------------------------------------------------
# Intensity field
# ---------------------------------------------------------------------------


class TestIntensityField:
    def test_reflectivity_stored_in_intensity(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 200, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["intensity"][0] == pytest.approx(200.0)

    def test_intensity_dtype_is_float32(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 128, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["intensity"].dtype == np.float32


# ---------------------------------------------------------------------------
# Confidence byte decomposition
# ---------------------------------------------------------------------------


class TestConfidenceField:
    def test_contamination_bits_7_6(self) -> None:
        # 0xC5 = 0b11000101 → contamination = bits[7:6] = 0b11 = 3
        pkt = build_packet(block1_channels={0: (250, 100, 0xC5)})
        result = decode_packet(pkt, _flat_cal())
        assert result["contamination"][0] == 3

    def test_noise_level_bits_5_0(self) -> None:
        # 0xC5 = 0b11000101 → noise_level = bits[5:0] = 0b000101 = 5
        pkt = build_packet(block1_channels={0: (250, 100, 0xC5)})
        result = decode_packet(pkt, _flat_cal())
        assert result["noise_level"][0] == 5

    def test_max_contamination(self) -> None:
        # 0xFF = 0b11111111 → contamination = 3, noise_level = 63
        pkt = build_packet(block1_channels={0: (250, 100, 0xFF)})
        result = decode_packet(pkt, _flat_cal())
        assert result["contamination"][0] == 3
        assert result["noise_level"][0] == 63

    def test_zero_confidence(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 100, 0x00)})
        result = decode_packet(pkt, _flat_cal())
        assert result["contamination"][0] == 0
        assert result["noise_level"][0] == 0

    @pytest.mark.parametrize(
        ("conf_byte", "expected_cont", "expected_noise"),
        [
            (0x00, 0, 0),
            (0x40, 1, 0),
            (0x80, 2, 0),
            (0xC0, 3, 0),
            (0x3F, 0, 63),
            (0xFF, 3, 63),
        ],
    )
    def test_confidence_decomposition(
        self, conf_byte: int, expected_cont: int, expected_noise: int
    ) -> None:
        pkt = build_packet(block1_channels={0: (250, 100, conf_byte)})
        result = decode_packet(pkt, _flat_cal())
        assert result["contamination"][0] == expected_cont
        assert result["noise_level"][0] == expected_noise


# ---------------------------------------------------------------------------
# Timestamp reconstruction
# ---------------------------------------------------------------------------


class TestTimestamp:
    def test_block1_timestamp_structure(self) -> None:
        """Per-point timestamp = t0 + block1_start_us*1e-6 + firing_offset[ring]."""
        year, month, day, hour, minute, second, frac_us = 125, 5, 6, 12, 0, 0, 500_000
        pkt = build_packet(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            frac_us=frac_us,
            block1_channels={0: (250, 100, 0)},
        )
        result = decode_packet(pkt, _flat_cal())
        pts = result[result["ring"] == 1]
        assert len(pts) == 1

        t0 = datetime(
            year + 1900, month, day, hour, minute, second, frac_us, tzinfo=timezone.utc
        ).timestamp()
        expected_ts = t0 + BLOCK1_START_US * 1e-6 + FIRING_OFFSETS_S[0]
        assert pts["timestamp"][0] == pytest.approx(expected_ts, abs=1e-9)

    def test_block2_timestamp_uses_different_offset(self) -> None:
        year, month, day, hour, minute, second, frac_us = 125, 5, 6, 12, 0, 0, 0
        pkt = build_packet(
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            frac_us=frac_us,
            block1_channels={0: (250, 100, 0)},
            block2_channels={0: (250, 100, 0)},
        )
        result = decode_packet(pkt, _flat_cal())
        # ring 1 appears twice (once per block); timestamps must differ
        pts = result[result["ring"] == 1]
        assert len(pts) == 2
        ts_sorted = sorted(pts["timestamp"].tolist())
        # block1 fires earlier (BLOCK1_START_US < BLOCK2_START_US)
        delta = ts_sorted[1] - ts_sorted[0]
        expected_delta = (BLOCK2_START_US - BLOCK1_START_US) * 1e-6
        # float64 at ~1.746e9 epoch has ~400 ns ULP; allow 1 µs tolerance
        assert delta == pytest.approx(expected_delta, abs=1e-6)

    def test_timestamp_dtype_is_float64(self) -> None:
        pkt = build_packet(block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result["timestamp"].dtype == np.float64

    def test_year_offset_applied(self) -> None:
        # year=0 → 1900; year=125 → 2025; year=124 → 2024
        pkt_2025 = build_packet(year=125, frac_us=0, block1_channels={0: (250, 100, 0)})
        pkt_2024 = build_packet(year=124, frac_us=0, block1_channels={0: (250, 100, 0)})
        ts_2025 = decode_packet(pkt_2025, _flat_cal())["timestamp"][0]
        ts_2024 = decode_packet(pkt_2024, _flat_cal())["timestamp"][0]
        # 2025 timestamp should be ~366 days (leap year 2024) > 2024 timestamp
        diff_days = (ts_2025 - ts_2024) / 86400.0
        assert 364 < diff_days < 367
