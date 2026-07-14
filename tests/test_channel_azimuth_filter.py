"""Decode-time channel/azimuth filter contract tests."""

from __future__ import annotations

import numpy as np
import pytest
from _packet_builder import build_packet

from lidar2numpy import (
    ChannelAzimuthFilter,
    Decoder,
    PreparedChannelAzimuthFilter,
)
from lidar2numpy.calibration import Calibration
from lidar2numpy.decoder import _decode_packet_spherical
from lidar2numpy.structs import SPHERICAL_DTYPE


def _flat_cal() -> Calibration:
    return Calibration(
        elevations_rad=np.zeros(128, dtype=np.float64),
        azimuth_offsets_deg=np.zeros(128, dtype=np.float64),
    )


def _azimuth_cal(channel: int, offset_deg: float) -> Calibration:
    offsets = np.zeros(128, dtype=np.float64)
    offsets[channel - 1] = offset_deg
    return Calibration(
        elevations_rad=np.zeros(128, dtype=np.float64),
        azimuth_offsets_deg=offsets,
    )


def _emit_one_spherical_frame(decoder: Decoder, *packets: bytes) -> np.ndarray:
    decoder.feed(build_packet(block1_az=35000))
    decoder.feed(build_packet(block1_az=100))
    for packet in packets:
        assert decoder.feed(packet) is None
    assert decoder.feed(build_packet(block1_az=35900)) is None
    frame = decoder.feed(build_packet(block1_az=50))
    assert frame is not None
    assert frame.dtype == SPHERICAL_DTYPE
    return frame


def _post_filter_spherical(frame: np.ndarray, spec: ChannelAzimuthFilter) -> np.ndarray:
    keep = np.ones(len(frame), dtype=np.bool_)
    drop_channels = set(spec.drop_channels)
    for index, point in enumerate(frame):
        channel = int(point["channel"])
        if channel in drop_channels:
            keep[index] = False
            continue
        ranges = (
            spec.drop_azimuth_ranges_by_channel[channel]
            if channel in spec.drop_azimuth_ranges_by_channel
            else ()
        )
        azimuth = float(point["azimuth_deg"]) % 360.0
        for start, end in ranges:
            if (start < end and start <= azimuth < end) or (
                start > end and (azimuth >= start or azimuth < end)
            ):
                keep[index] = False
                break
    return frame[keep]


class TestChannelAzimuthFilterSpec:
    def test_normalizes_channels_ranges_and_fingerprint_order(self) -> None:
        left = ChannelAzimuthFilter(
            drop_channels=(14, 12, 14),
            drop_azimuth_ranges_by_channel={
                37: ((65.0, 80.0), (15.0, 30.0)),
                12: ((90.0, 100.0),),
            },
        )
        right = ChannelAzimuthFilter(
            drop_channels=(12, 14),
            drop_azimuth_ranges_by_channel={
                12: ((90.0, 100.0),),
                37: ((15.0, 30.0), (65.0, 80.0)),
            },
        )

        assert left.drop_channels == (12, 14)
        assert left.drop_azimuth_ranges_by_channel[37] == ((15.0, 30.0), (65.0, 80.0))
        assert 12 not in left.drop_azimuth_ranges_by_channel
        assert left.fingerprint() == right.fingerprint()

    def test_full_circle_range_becomes_channel_wide_drop_before_modulo(self) -> None:
        spec = ChannelAzimuthFilter(
            drop_azimuth_ranges_by_channel={
                5: ((0.0, 360.0),),
                6: ((-180.0, 180.0), (0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0)),
            }
        )

        assert spec.drop_channels == (5, 6)
        assert spec.drop_azimuth_ranges_by_channel == {}

    def test_from_mapping_rejects_conflicting_duplicate_channel_records(self) -> None:
        payload = {
            "drop_channels": [],
            "channel_azimuth_drops": [
                {"channel": 37, "ranges_deg": [[15.0, 30.0]]},
                {"channel": 37, "ranges_deg": [[16.0, 30.0]]},
            ],
        }

        with pytest.raises(ValueError, match="duplicate channel 37"):
            ChannelAzimuthFilter.from_mapping(payload)

    def test_from_mapping_allows_equivalent_duplicate_channel_records(self) -> None:
        payload = {
            "drop_channels": [],
            "channel_azimuth_drops": [
                {"channel": 37, "ranges_deg": [[15.0, 30.0], [65.0, 80.0]]},
                {"channel": 37, "ranges_deg": [[65.0, 80.0], [15.0, 30.0]]},
            ],
        }

        spec = ChannelAzimuthFilter.from_mapping(payload)

        assert spec.drop_azimuth_ranges_by_channel[37] == ((15.0, 30.0), (65.0, 80.0))

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"drop_channels": (0,)}, "drop_channels"),
            ({"drop_channels": (129,)}, "drop_channels"),
            ({"drop_channels": (True,)}, "drop_channels"),
            ({"drop_azimuth_ranges_by_channel": {1: ((float("nan"), 10.0),)}}, "range"),
            ({"drop_azimuth_ranges_by_channel": {1: ((10.0, 10.0),)}}, "zero-width"),
            (
                {
                    "drop_azimuth_ranges_by_channel": {
                        1: ((0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (6.0, 7.0), (8.0, 9.0))
                    }
                },
                "at most 4",
            ),
        ],
    )
    def test_validation_errors(self, kwargs: dict[str, object], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            ChannelAzimuthFilter(**kwargs)


class TestPreparedChannelAzimuthFilter:
    def test_lookup_shape_dtype_view_and_fingerprints(self) -> None:
        spec = ChannelAzimuthFilter(drop_channels=(1,))
        prepared = PreparedChannelAzimuthFilter.from_spec(spec, _flat_cal())

        assert prepared.drop_by_raw_azimuth_and_channel.shape == (36000, 128)
        assert prepared.drop_by_raw_azimuth_and_channel.dtype == np.bool_
        row = prepared.drop_mask_for_raw_azimuth(0)
        assert np.shares_memory(row, prepared.drop_by_raw_azimuth_and_channel)
        assert bool(row[0]) is True
        assert prepared.spec_fingerprint == spec.fingerprint()
        assert prepared.fingerprint != prepared.spec_fingerprint

    def test_calibration_fingerprint_changes_with_azimuth_offsets(self) -> None:
        spec = ChannelAzimuthFilter(drop_azimuth_ranges_by_channel={1: ((1.0, 2.0),)})

        left = PreparedChannelAzimuthFilter.from_spec(spec, _flat_cal())
        right = PreparedChannelAzimuthFilter.from_spec(spec, _azimuth_cal(1, 0.5))

        assert left.calibration_fingerprint != right.calibration_fingerprint
        assert left.fingerprint != right.fingerprint

    def test_prepared_fingerprint_changes_with_spec(self) -> None:
        left = PreparedChannelAzimuthFilter.from_spec(
            ChannelAzimuthFilter(drop_channels=(1,)), _flat_cal()
        )
        right = PreparedChannelAzimuthFilter.from_spec(
            ChannelAzimuthFilter(drop_channels=(2,)), _flat_cal()
        )

        assert left.spec_fingerprint != right.spec_fingerprint
        assert left.fingerprint != right.fingerprint

    def test_lookup_uses_calibrated_azimuth(self) -> None:
        spec = ChannelAzimuthFilter(drop_azimuth_ranges_by_channel={1: ((10.0, 11.0),)})
        prepared = PreparedChannelAzimuthFilter.from_spec(spec, _azimuth_cal(1, 1.0))

        assert bool(prepared.drop_mask_for_raw_azimuth(900)[0]) is True
        assert bool(prepared.drop_mask_for_raw_azimuth(800)[0]) is False

    def test_bad_raw_azimuth_rejected(self) -> None:
        prepared = PreparedChannelAzimuthFilter.from_spec(
            ChannelAzimuthFilter(drop_channels=(1,)), _flat_cal()
        )

        with pytest.raises(ValueError, match="raw_azimuth"):
            prepared.drop_mask_for_raw_azimuth(36000)


class TestDecoderFilter:
    def test_empty_filter_uses_no_prepared_lookup_and_matches_none(self) -> None:
        pkt = build_packet(block1_az=200, block1_channels={0: (250, 10, 0)})
        no_filter = Decoder(_flat_cal(), output_mode="spherical")
        empty_filter = Decoder(
            _flat_cal(),
            output_mode="spherical",
            point_filter=ChannelAzimuthFilter(),
        )

        frame_none = _emit_one_spherical_frame(no_filter, pkt)
        frame_empty = _emit_one_spherical_frame(empty_filter, pkt)

        assert empty_filter._prepared_filter is None
        np.testing.assert_array_equal(frame_empty, frame_none)
        assert empty_filter.last_filter_diagnostics().mode == "off"

    def test_drop_mode_removes_channel_and_preserves_order(self) -> None:
        spec = ChannelAzimuthFilter(drop_channels=(2,))
        decoder = Decoder(_flat_cal(), output_mode="spherical", point_filter=spec)
        pkt = build_packet(
            block1_az=200,
            block1_channels={0: (250, 10, 0), 1: (250, 20, 0), 2: (250, 30, 0)},
        )

        frame = _emit_one_spherical_frame(decoder, pkt)

        assert frame["channel"].tolist() == [1, 3]
        assert frame["intensity"].tolist() == pytest.approx([10.0, 30.0])
        diagnostics = decoder.last_filter_diagnostics()
        assert diagnostics.mode == "drop"
        assert diagnostics.input_points == 3
        assert diagnostics.output_points == 2
        assert diagnostics.dropped_points == 1
        assert diagnostics.dropped_points_by_channel == {2: 1}
        assert diagnostics.to_dict()["dropped_points_by_channel"] == {"2": 1}

    def test_shadow_mode_reports_would_drop_without_changing_frame(self) -> None:
        spec = ChannelAzimuthFilter(drop_channels=(2,))
        decoder = Decoder(
            _flat_cal(),
            output_mode="spherical",
            point_filter=spec,
            point_filter_mode="shadow",
        )
        pkt = build_packet(
            block1_az=200,
            block1_channels={0: (250, 10, 0), 1: (250, 20, 0), 2: (250, 30, 0)},
        )

        frame = _emit_one_spherical_frame(decoder, pkt)

        assert frame["channel"].tolist() == [1, 2, 3]
        diagnostics = decoder.last_filter_diagnostics()
        assert diagnostics.mode == "shadow"
        assert diagnostics.input_points == 3
        assert diagnostics.output_points == 3
        assert diagnostics.dropped_points == 1
        assert diagnostics.dropped_points_by_channel == {2: 1}

    def test_each_block_uses_its_own_raw_azimuth(self) -> None:
        spec = ChannelAzimuthFilter(drop_azimuth_ranges_by_channel={1: ((10.0, 11.0),)})
        decoder = Decoder(_flat_cal(), output_mode="spherical", point_filter=spec)
        pkt = build_packet(
            block1_az=900,
            block2_az=1000,
            block1_channels={0: (250, 10, 0)},
            block2_channels={0: (300, 20, 0)},
        )

        frame = _emit_one_spherical_frame(decoder, pkt)

        assert frame["intensity"].tolist() == pytest.approx([10.0])
        diagnostics = decoder.last_filter_diagnostics()
        assert diagnostics.input_points == 2
        assert diagnostics.output_points == 1
        assert diagnostics.dropped_points_by_channel == {1: 1}

    def test_wraparound_range_drops_packet_points(self) -> None:
        spec = ChannelAzimuthFilter(drop_azimuth_ranges_by_channel={1: ((350.0, 10.0),)})
        decoder = Decoder(_flat_cal(), output_mode="spherical", point_filter=spec)
        pkt = build_packet(
            block1_az=35500,
            block2_az=2000,
            block1_channels={0: (250, 10, 0)},
            block2_channels={0: (300, 20, 0)},
        )

        frame = _emit_one_spherical_frame(decoder, pkt)

        assert frame["intensity"].tolist() == pytest.approx([20.0])
        assert decoder.last_filter_diagnostics().dropped_points_by_channel == {1: 1}

    @pytest.mark.parametrize(
        "packet",
        [
            build_packet(
                block1_az=1000,
                block2_az=1100,
                block1_channels={i: (800 + i, i % 256, i % 64) for i in range(128)},
                block2_channels={i: (900 + i, i % 256, i % 64) for i in range(128)},
            ),
            build_packet(block1_az=1000, block1_channels={0: (250, 10, 0)}),
            build_packet(
                block1_az=1000,
                block1_channels={i: (800 + i, i % 256, i % 64) for i in range(120)},
            ),
            build_packet(
                return_mode=0x39,
                block1_az=1000,
                block2_az=1000,
                block1_channels={0: (250, 10, 0), 1: (300, 11, 0)},
                block2_channels={0: (350, 12, 0), 1: (400, 13, 0)},
            ),
        ],
    )
    def test_filtered_path_handles_representative_packet_shapes(self, packet: bytes) -> None:
        spec = ChannelAzimuthFilter(drop_channels=(2,))
        decoder = Decoder(_flat_cal(), output_mode="spherical", point_filter=spec)

        frame = _emit_one_spherical_frame(decoder, packet)

        assert 2 not in frame["channel"].tolist()
        diagnostics = decoder.last_filter_diagnostics()
        assert diagnostics.input_points >= diagnostics.output_points
        assert diagnostics.dropped_points == diagnostics.dropped_points_by_channel.get(2, 0)

    def test_prepared_lookup_output_matches_post_filtered_spherical_output(self) -> None:
        spec = ChannelAzimuthFilter(
            drop_channels=(3,),
            drop_azimuth_ranges_by_channel={1: ((350.0, 10.0),), 2: ((20.0, 30.0),)},
        )
        packets = [
            build_packet(
                block1_az=35500,
                block2_az=2500,
                block1_channels={0: (250, 10, 0), 1: (250, 11, 0), 2: (250, 12, 0)},
                block2_channels={0: (300, 20, 0), 1: (300, 21, 0), 2: (300, 22, 0)},
            ),
            build_packet(
                block1_az=35600,
                block1_channels={0: (350, 30, 0), 1: (350, 31, 0), 2: (350, 32, 0)},
            ),
        ]
        expected = np.concatenate(
            [
                _post_filter_spherical(_decode_packet_spherical(packet, _flat_cal()), spec)
                for packet in packets
            ]
        )
        decoder = Decoder(_flat_cal(), output_mode="spherical", point_filter=spec)

        actual = _emit_one_spherical_frame(decoder, *packets)

        np.testing.assert_array_equal(actual, expected)

    def test_cartesian_filter_rejected(self) -> None:
        with pytest.raises(ValueError, match="output_mode"):
            Decoder(_flat_cal(), output_mode="cartesian", point_filter=ChannelAzimuthFilter((1,)))

    def test_invalid_filter_mode_rejected(self) -> None:
        with pytest.raises(ValueError, match="point_filter_mode"):
            Decoder(
                _flat_cal(),
                output_mode="spherical",
                point_filter=ChannelAzimuthFilter((1,)),
                point_filter_mode="audit",  # type: ignore[arg-type]
            )
