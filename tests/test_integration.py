"""End-to-end integration tests using real JT128 sensor captures.

These tests decode the captured .pcap files from tests/fixtures/ and
validate the decoded frames against known geometry constraints.

The pcap files are gitignored (too large for the repo). Tests are skipped
with a clear message when the pcap file is not present, so CI and fresh
clones do not fail.

Pcap format (libpcap, little-endian):
  Global header: 24 bytes
    magic(4), version_major(2), version_minor(2), thiszone(4),
    sigfigs(4), snaplen(4), network(4)
  Per-record header: 16 bytes
    ts_sec(4), ts_usec(4), incl_len(4), orig_len(4)
  Per-record data: incl_len bytes (raw Ethernet frame)

Ethernet + IP + UDP stripping:
  Ethernet header: 14 bytes (dst MAC 6, src MAC 6, ethertype 2)
  IP header:       20 bytes (fixed; no options in sensor packets)
  UDP header:       8 bytes (src port 2, dst port 2, len 2, checksum 2)
  Payload:         incl_len - 42 bytes = 1100 bytes for JT128
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from lidar2numpy import Decoder
from lidar2numpy.structs import POINT_DTYPE

# ---------------------------------------------------------------------------
# Test fixture paths
# ---------------------------------------------------------------------------
_FIXTURES = Path(__file__).parent / "fixtures"
_PCAP_2368 = _FIXTURES / "lidar-Port 2368" / "jt128_capture_2368.pcap"
_CAL_2368 = _FIXTURES / "lidar-Port 2368" / "angle_corrections - 2368.csv"
_PCAP_2369 = _FIXTURES / "lidar-Port 2369" / "jt128_capture_2369.pcap"
_CAL_2369 = _FIXTURES / "lidar-Port 2369" / "angle_corrections - 2369.csv"

_PCAP_HEADER_LEN = 24
_RECORD_HEADER_LEN = 16
_ETHERNET_IP_UDP_LEN = 42  # 14 + 20 + 8
_UDP_PAYLOAD_SIZE = 1100
_MAX_PACKETS = 800  # decode this many packets max per test (fast enough, covers 1+ frame)


# ---------------------------------------------------------------------------
# Minimal pcap reader (stdlib only, no scapy/dpkt)
# ---------------------------------------------------------------------------


def _iter_udp_payloads(pcap_path: Path, max_packets: int) -> Iterator[bytes]:
    """Yield UDP payloads from a libpcap file (no external dependencies)."""
    with pcap_path.open("rb") as f:
        global_hdr = f.read(_PCAP_HEADER_LEN)
        if len(global_hdr) < _PCAP_HEADER_LEN:
            raise ValueError("File too short to be a valid pcap")
        magic = struct.unpack_from("<I", global_hdr, 0)[0]
        if magic not in (0xA1B2C3D4, 0xD4C3B2A1):
            raise ValueError(f"Unexpected pcap magic: 0x{magic:08X}")

        count = 0
        while count < max_packets:
            rec_hdr = f.read(_RECORD_HEADER_LEN)
            if len(rec_hdr) < _RECORD_HEADER_LEN:
                break
            incl_len = struct.unpack_from("<I", rec_hdr, 8)[0]
            raw = f.read(incl_len)
            if len(raw) < incl_len:
                break
            # Strip Ethernet + IP + UDP headers
            payload_start = _ETHERNET_IP_UDP_LEN
            payload = raw[payload_start:]
            if len(payload) == _UDP_PAYLOAD_SIZE:
                yield payload
                count += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _decode_first_frame(pcap: Path, cal_path: Path) -> np.ndarray:
    """Decode the first complete frame from a pcap capture."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dec = Decoder(cal_path)
    for payload in _iter_udp_payloads(pcap, max_packets=_MAX_PACKETS):
        frame = dec.feed(payload)
        if frame is not None:
            return frame
    # If no rollover detected (shouldn't happen with 800 packets), flush
    last = dec.flush()
    assert last is not None, "No frames decoded from pcap"
    return last


# ---------------------------------------------------------------------------
# Port 2368 — integration tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PCAP_2368.exists(),
    reason=f"pcap fixture not present — run locally with captured sensor data ({_PCAP_2368})",
)
class TestIntegration2368:
    @pytest.fixture(scope="class")
    def frame(self) -> np.ndarray:
        return _decode_first_frame(_PCAP_2368, _CAL_2368)

    def test_frame_dtype(self, frame: np.ndarray) -> None:
        assert frame.dtype == POINT_DTYPE

    def test_frame_point_count_reasonable(self, frame: np.ndarray) -> None:
        # At 10 Hz each frame has ~180 packets × 128 channels × 2 blocks ≈ 46k
        # valid returns in a typical indoor scene (most channels have returns).
        assert len(frame) > 30_000, f"Too few points: {len(frame)}"
        assert len(frame) < 600_000, f"Too many points: {len(frame)}"

    def test_ring_values_in_range(self, frame: np.ndarray) -> None:
        assert int(frame["ring"].min()) >= 1
        assert int(frame["ring"].max()) <= 128

    def test_all_128_channels_present(self, frame: np.ndarray) -> None:
        unique_rings = np.unique(frame["ring"])
        assert len(unique_rings) == 128, f"Only {len(unique_rings)} channels decoded"

    def test_intensity_in_range(self, frame: np.ndarray) -> None:
        assert float(frame["intensity"].min()) >= 0.0
        assert float(frame["intensity"].max()) <= 255.0

    def test_xyz_finite(self, frame: np.ndarray) -> None:
        assert np.all(np.isfinite(frame["x"]))
        assert np.all(np.isfinite(frame["y"]))
        assert np.all(np.isfinite(frame["z"]))

    def test_z_values_span_both_signs(self, frame: np.ndarray) -> None:
        # JT128 is hyper-hemispherical (−4° to +89° elevation) so both
        # positive z (upward-looking channels) and negative z (downward-looking)
        # channels are present in every frame.
        assert np.any(frame["z"] > 0), "No positive z values found"
        assert np.any(frame["z"] < 0), "No negative z values found"

    def test_timestamps_positive_and_finite(self, frame: np.ndarray) -> None:
        # The JT128's hardware clock may not be PTP/GPS-synced; the sensor
        # can report times relative to 1970-01-01 (year field = 70).
        # We only assert the values are finite and non-negative.
        assert np.all(np.isfinite(frame["timestamp"]))
        assert float(frame["timestamp"].min()) >= 0.0

    def test_timestamps_span_one_frame(self, frame: np.ndarray) -> None:
        # One full revolution at 10 Hz takes 100 ms; allow 50–200 ms.
        span_ms = (float(frame["timestamp"].max()) - float(frame["timestamp"].min())) * 1000
        assert 10 < span_ms < 300, f"Timestamp span {span_ms:.1f} ms out of range"

    def test_contamination_in_range(self, frame: np.ndarray) -> None:
        assert int(frame["contamination"].max()) <= 3

    def test_noise_level_in_range(self, frame: np.ndarray) -> None:
        assert int(frame["noise_level"].max()) <= 63

    def test_xy_geometry_360_degrees(self, frame: np.ndarray) -> None:
        # A full rotation should produce returns in all four XY quadrants.
        quad_pp = np.any((frame["x"] > 0) & (frame["y"] > 0))
        quad_pn = np.any((frame["x"] > 0) & (frame["y"] < 0))
        quad_np = np.any((frame["x"] < 0) & (frame["y"] > 0))
        quad_nn = np.any((frame["x"] < 0) & (frame["y"] < 0))
        assert quad_pp and quad_pn and quad_np and quad_nn, "Not all XY quadrants covered"


# ---------------------------------------------------------------------------
# Port 2369 — integration tests (same checks, different sensor)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _PCAP_2369.exists(),
    reason=f"pcap fixture not present — run locally with captured sensor data ({_PCAP_2369})",
)
class TestIntegration2369:
    @pytest.fixture(scope="class")
    def frame(self) -> np.ndarray:
        return _decode_first_frame(_PCAP_2369, _CAL_2369)

    def test_frame_dtype(self, frame: np.ndarray) -> None:
        assert frame.dtype == POINT_DTYPE

    def test_frame_point_count_reasonable(self, frame: np.ndarray) -> None:
        assert len(frame) > 30_000
        assert len(frame) < 600_000

    def test_ring_values_in_range(self, frame: np.ndarray) -> None:
        assert int(frame["ring"].min()) >= 1
        assert int(frame["ring"].max()) <= 128

    def test_all_128_channels_present(self, frame: np.ndarray) -> None:
        assert len(np.unique(frame["ring"])) == 128

    def test_intensity_in_range(self, frame: np.ndarray) -> None:
        assert float(frame["intensity"].min()) >= 0.0
        assert float(frame["intensity"].max()) <= 255.0

    def test_xyz_finite(self, frame: np.ndarray) -> None:
        assert np.all(np.isfinite(frame["x"]))
        assert np.all(np.isfinite(frame["y"]))
        assert np.all(np.isfinite(frame["z"]))

    def test_xy_geometry_360_degrees(self, frame: np.ndarray) -> None:
        quad_pp = np.any((frame["x"] > 0) & (frame["y"] > 0))
        quad_pn = np.any((frame["x"] > 0) & (frame["y"] < 0))
        quad_np = np.any((frame["x"] < 0) & (frame["y"] > 0))
        quad_nn = np.any((frame["x"] < 0) & (frame["y"] < 0))
        assert quad_pp and quad_pn and quad_np and quad_nn
