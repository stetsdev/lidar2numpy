"""Tests for spherical output mode and to_cartesian().

Key contracts verified:
  - Round-trip: decode_packet XYZ == to_cartesian(_decode_packet_spherical) XYZ
  - Subset: sliced spherical array → to_cartesian → matches same slice of cartesian
  - Zero-distance exclusion in spherical mode
  - Backward compatibility: Decoder() with no output_mode produces POINT_DTYPE
"""

from __future__ import annotations

import numpy as np
import pytest
from _packet_builder import build_packet

from lidar2numpy.calibration import Calibration
from lidar2numpy.decoder import _decode_packet_spherical, decode_packet, to_cartesian
from lidar2numpy.structs import POINT_DTYPE, SPHERICAL_DTYPE

# ---------------------------------------------------------------------------
# Calibration fixtures
# ---------------------------------------------------------------------------


def _flat_cal() -> Calibration:
    """All channels: elevation=0°, azimuth_offset=0°."""
    return Calibration(
        elevations_rad=np.zeros(128, dtype=np.float64),
        azimuth_offsets_deg=np.zeros(128, dtype=np.float64),
    )


def _mixed_cal() -> Calibration:
    """Non-trivial calibration: varying elevations and azimuth offsets."""
    elevs = np.zeros(128, dtype=np.float64)
    az_offs = np.zeros(128, dtype=np.float64)
    # Spread elevations linearly from -15° to +15° across 128 channels.
    for i in range(128):
        elevs[i] = np.deg2rad(-15.0 + 30.0 * i / 127.0)
    # Give every channel a small per-channel azimuth offset.
    for i in range(128):
        az_offs[i] = (i % 10) * 0.1  # 0.0 – 0.9° repeating
    return Calibration(elevations_rad=elevs, azimuth_offsets_deg=az_offs)


# ---------------------------------------------------------------------------
# Round-trip equivalence
# ---------------------------------------------------------------------------


class TestRoundTrip:
    """The single most important test: spherical → to_cartesian must produce
    bit-for-bit identical float32 XYZ as the direct cartesian decode, up to
    the quantization introduced by storing distance_m and azimuth_deg as
    float32 in SPHERICAL_DTYPE (≤ 1e-5 absolute error at typical ranges)."""

    def _compare_xyz(self, pkt: bytes, cal: Calibration, *, atol: float = 1e-5) -> None:
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)
        cart_rt = to_cartesian(sph, cal)

        assert len(cart) == len(sph) == len(cart_rt), "point counts must match"

        # Both decode paths emit points in the same order (block1 then block2,
        # masked channel order), so indices align without sorting.
        np.testing.assert_allclose(cart_rt["x"], cart["x"], atol=atol)
        np.testing.assert_allclose(cart_rt["y"], cart["y"], atol=atol)
        np.testing.assert_allclose(cart_rt["z"], cart["z"], atol=atol)

    def test_single_block_azimuth0_flat_cal(self) -> None:
        """1 m point at 0° azimuth, flat calibration — y ≈ 1.0, x = z = 0."""
        pkt = build_packet(block1_az=0, block1_channels={0: (250, 100, 0)})
        self._compare_xyz(pkt, _flat_cal())

    def test_single_block_azimuth90_flat_cal(self) -> None:
        pkt = build_packet(block1_az=9000, block1_channels={0: (250, 100, 0)})
        self._compare_xyz(pkt, _flat_cal())

    def test_both_blocks_multiple_channels(self) -> None:
        """Multiple channels in both blocks; all must round-trip correctly."""
        pkt = build_packet(
            block1_az=4500,
            block1_channels={0: (500, 100, 0), 32: (1000, 150, 0), 63: (750, 80, 0)},
            block2_az=4600,
            block2_channels={64: (400, 200, 0), 127: (600, 220, 0)},
        )
        self._compare_xyz(pkt, _flat_cal())

    def test_mixed_calibration(self) -> None:
        """Non-trivial per-channel elevation and azimuth offsets."""
        pkt = build_packet(
            block1_az=1800,
            block1_channels={i: (500 + i * 2, 100, 0) for i in range(0, 64, 8)},
            block2_az=1900,
            block2_channels={i: (600 + i, 120, 0) for i in range(64, 128, 8)},
        )
        self._compare_xyz(pkt, _mixed_cal())

    def test_non_xyz_fields_are_identical(self) -> None:
        """intensity, ring/channel, timestamp, contamination, noise_level carry over."""
        pkt = build_packet(
            block1_az=9000,
            block1_channels={0: (250, 200, 0xC5)},
        )
        cal = _flat_cal()
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)
        cart_rt = to_cartesian(sph, cal)

        assert cart_rt["intensity"][0] == pytest.approx(cart["intensity"][0])
        assert cart_rt["ring"][0] == cart["ring"][0]
        assert cart_rt["timestamp"][0] == pytest.approx(cart["timestamp"][0], abs=1e-9)
        assert cart_rt["contamination"][0] == cart["contamination"][0]
        assert cart_rt["noise_level"][0] == cart["noise_level"][0]

    def test_channel_equals_ring(self) -> None:
        """Spherical 'channel' field and Cartesian 'ring' field are the same value."""
        pkt = build_packet(
            block1_az=0,
            block1_channels={0: (250, 100, 0), 63: (500, 100, 0), 127: (750, 100, 0)},
        )
        cal = _flat_cal()
        sph = _decode_packet_spherical(pkt, cal)
        cart_rt = to_cartesian(sph, cal)
        np.testing.assert_array_equal(sph["channel"].astype(np.uint16), cart_rt["ring"])

    def test_empty_packet_round_trips(self) -> None:
        """No valid channels → both decoders produce empty arrays."""
        pkt = build_packet()  # all distances = 0
        cal = _flat_cal()
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)
        cart_rt = to_cartesian(sph, cal)
        assert len(cart) == 0
        assert len(sph) == 0
        assert len(cart_rt) == 0
        assert cart_rt.dtype == POINT_DTYPE


# ---------------------------------------------------------------------------
# Subset conversion
# ---------------------------------------------------------------------------


class TestSubsetConversion:
    """to_cartesian() must work on arbitrary subsets of a spherical array,
    not just complete frames. The converted subset must match the
    corresponding elements of a full cartesian decode."""

    def test_every_third_point(self) -> None:
        """Slice [::3] of spherical frame converts identically to cartesian [::3]."""
        pkt = build_packet(
            block1_az=2700,
            block1_channels={i: (300 + i * 3, 100, 0) for i in range(0, 64, 1)},
            block2_az=2800,
            block2_channels={i: (400 + i, 110, 0) for i in range(64, 128, 1)},
        )
        cal = _mixed_cal()
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)

        sph_subset = sph[::3]
        cart_subset_rt = to_cartesian(sph_subset, cal)
        cart_expected = cart[::3]

        np.testing.assert_allclose(cart_subset_rt["x"], cart_expected["x"], atol=1e-5)
        np.testing.assert_allclose(cart_subset_rt["y"], cart_expected["y"], atol=1e-5)
        np.testing.assert_allclose(cart_subset_rt["z"], cart_expected["z"], atol=1e-5)

    def test_boolean_mask_subset(self) -> None:
        """Boolean-masked spherical subset converts correctly."""
        pkt = build_packet(
            block1_az=4500,
            block1_channels={i: (500 + i * 4, 128, 0) for i in range(0, 32)},
        )
        cal = _flat_cal()
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)

        # Keep only points with distance_m > 0.08 m (arbitrary foreground mask).
        mask = sph["distance_m"] > 0.08
        sph_fg = sph[mask]
        cart_fg_rt = to_cartesian(sph_fg, cal)
        cart_expected = cart[mask]

        assert len(cart_fg_rt) == int(mask.sum())
        np.testing.assert_allclose(cart_fg_rt["x"], cart_expected["x"], atol=1e-5)
        np.testing.assert_allclose(cart_fg_rt["y"], cart_expected["y"], atol=1e-5)

    def test_single_point_subset(self) -> None:
        """Single-element array handled correctly by to_cartesian."""
        pkt = build_packet(
            block1_az=9000,
            block1_channels={0: (250, 100, 0)},
        )
        cal = _flat_cal()
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)

        cart_rt = to_cartesian(sph[:1], cal)
        assert len(cart_rt) == 1
        assert cart_rt["x"][0] == pytest.approx(cart["x"][0], abs=1e-5)
        assert cart_rt["y"][0] == pytest.approx(cart["y"][0], abs=1e-5)

    def test_empty_subset(self) -> None:
        """Empty-array input returns empty POINT_DTYPE array."""
        sph_empty = np.empty(0, dtype=SPHERICAL_DTYPE)
        cal = _flat_cal()
        result = to_cartesian(sph_empty, cal)
        assert result.dtype == POINT_DTYPE
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Zero-distance exclusion
# ---------------------------------------------------------------------------


class TestZeroDistanceExclusion:
    def test_zero_distance_channels_excluded(self) -> None:
        """Channels with distance=0 must not appear in spherical output."""
        pkt = build_packet(
            # channel 0 has a valid return; channels 1–127 have distance=0
            block1_channels={0: (500, 100, 0)},
        )
        sph = _decode_packet_spherical(pkt, _flat_cal())
        assert len(sph) == 1
        assert sph["channel"][0] == 1  # 1-based

    def test_all_zero_returns_empty_spherical(self) -> None:
        """All-zero packet → empty SPHERICAL_DTYPE array."""
        pkt = build_packet()
        sph = _decode_packet_spherical(pkt, _flat_cal())
        assert sph.dtype == SPHERICAL_DTYPE
        assert len(sph) == 0

    def test_mixed_zero_nonzero_counts(self) -> None:
        """Mixed packet: exactly the non-zero channels appear in spherical output."""
        pkt = build_packet(
            block1_channels={5: (300, 100, 0), 10: (400, 100, 0)},
            block2_channels={20: (500, 100, 0)},
        )
        sph = _decode_packet_spherical(pkt, _flat_cal())
        assert len(sph) == 3
        channels = set(int(c) for c in sph["channel"])
        assert channels == {6, 11, 21}  # 1-based

    def test_zero_distance_same_as_cartesian(self) -> None:
        """Spherical mode excludes the same set of points as cartesian mode."""
        pkt = build_packet(
            block1_channels={0: (500, 100, 0), 5: (0, 0, 0), 10: (600, 100, 0)},
            block2_channels={64: (300, 100, 0)},
        )
        cal = _flat_cal()
        cart = decode_packet(pkt, cal)
        sph = _decode_packet_spherical(pkt, cal)
        # Both have the same count: channels 0, 10 (blk1), 64 (blk2) — 3 total
        assert len(cart) == len(sph) == 3


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Decoder() with no output_mode must behave identically to the old code."""

    def test_default_mode_returns_point_dtype(self) -> None:
        """Decoder() default produces POINT_DTYPE frames."""
        from lidar2numpy import Decoder

        cal = _flat_cal()
        decoder = Decoder(cal)
        pkt = build_packet(block1_channels={0: (250, 100, 0)})
        # Prime past startup discard then emit a frame via flush.
        decoder.feed(build_packet(block1_az=35000))
        decoder.feed(build_packet(block1_az=36))  # startup rollover
        decoder.feed(pkt)
        frame = decoder.flush()
        assert frame is not None
        assert frame.dtype == POINT_DTYPE

    def test_explicit_cartesian_mode_identical_to_default(self) -> None:
        """output_mode='cartesian' produces the same result as omitting it."""
        from lidar2numpy import Decoder

        cal = _flat_cal()
        pkt = build_packet(
            block1_az=4500,
            block1_channels={0: (500, 100, 0), 32: (1000, 150, 0)},
        )
        startup = [build_packet(block1_az=35000), build_packet(block1_az=36)]

        def _get_flush(mode_kwargs: dict) -> np.ndarray:  # type: ignore[type-arg]
            d = Decoder(cal, **mode_kwargs)
            for s in startup:
                d.feed(s)
            d.feed(pkt)
            result = d.flush()
            assert result is not None
            return result

        default_frame = _get_flush({})
        explicit_frame = _get_flush({"output_mode": "cartesian"})

        np.testing.assert_array_equal(default_frame["x"], explicit_frame["x"])
        np.testing.assert_array_equal(default_frame["y"], explicit_frame["y"])
        np.testing.assert_array_equal(default_frame["z"], explicit_frame["z"])

    def test_spherical_mode_returns_spherical_dtype(self) -> None:
        """Decoder(output_mode='spherical') produces SPHERICAL_DTYPE frames."""
        from lidar2numpy import Decoder

        cal = _flat_cal()
        decoder = Decoder(cal, output_mode="spherical")
        decoder.feed(build_packet(block1_az=35000))
        decoder.feed(build_packet(block1_az=36))  # startup rollover
        decoder.feed(build_packet(block1_az=100, block1_channels={0: (250, 100, 0)}))
        frame = decoder.flush()
        assert frame is not None
        assert frame.dtype == SPHERICAL_DTYPE

    def test_invalid_output_mode_raises(self) -> None:
        from lidar2numpy import Decoder

        with pytest.raises(ValueError, match="output_mode"):
            Decoder(output_mode="polar")  # type: ignore[arg-type]

    def test_existing_decoder_tests_unaffected(self) -> None:
        """Smoke test: decode_packet still works as before (POINT_DTYPE, x/y/z present)."""
        pkt = build_packet(block1_az=9000, block1_channels={0: (250, 100, 0)})
        result = decode_packet(pkt, _flat_cal())
        assert result.dtype == POINT_DTYPE
        assert result["x"][0] == pytest.approx(1.0, abs=1e-4)
        assert result["y"][0] == pytest.approx(0.0, abs=1e-5)
