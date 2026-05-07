from __future__ import annotations

import numpy as np
import pytest

from lidar2numpy.firing_times import FIRING_OFFSETS_S, FIRING_TIME_OFFSETS_US


class TestFiringTimeOffsets:
    """FIRING_TIME_OFFSETS_US is the per-channel table from Appendix B.4."""

    def test_table_has_128_entries(self) -> None:
        assert len(FIRING_TIME_OFFSETS_US) == 128

    def test_keys_are_channels_1_through_128(self) -> None:
        assert set(FIRING_TIME_OFFSETS_US.keys()) == set(range(1, 129))

    @pytest.mark.parametrize(
        ("channel", "expected_us"),
        [
            (1, 95.18),
            (2, 23.24),
            (22, 5.00),
            (64, 94.42),
            (128, 29.74),
        ],
    )
    def test_spot_values(self, channel: int, expected_us: float) -> None:
        assert FIRING_TIME_OFFSETS_US[channel] == pytest.approx(expected_us)

    def test_all_values_are_positive(self) -> None:
        assert all(v > 0 for v in FIRING_TIME_OFFSETS_US.values())

    def test_all_values_within_one_frame_period(self) -> None:
        # At 10 Hz the frame period is 100_000 µs; firing offsets must be << that.
        # The table's largest value should be well under 200 µs.
        assert max(FIRING_TIME_OFFSETS_US.values()) < 200.0


class TestFiringOffsetsSeconds:
    """FIRING_OFFSETS_S is a pre-built 1-D numpy array for use in the decoder."""

    def test_is_numpy_array(self) -> None:
        assert isinstance(FIRING_OFFSETS_S, np.ndarray)

    def test_shape_is_128(self) -> None:
        assert FIRING_OFFSETS_S.shape == (128,)

    def test_dtype_is_float64(self) -> None:
        assert FIRING_OFFSETS_S.dtype == np.float64

    def test_indexed_by_zero_based_ring(self) -> None:
        # ring 0 (channel 1) → 95.18 µs in seconds
        assert FIRING_OFFSETS_S[0] == pytest.approx(95.18e-6)

    def test_last_entry_is_channel_128(self) -> None:
        # ring 127 (channel 128) → 29.74 µs in seconds
        assert FIRING_OFFSETS_S[127] == pytest.approx(29.74e-6)

    def test_values_match_us_table_converted(self) -> None:
        for ch in range(1, 129):
            assert FIRING_OFFSETS_S[ch - 1] == pytest.approx(FIRING_TIME_OFFSETS_US[ch] * 1e-6)
