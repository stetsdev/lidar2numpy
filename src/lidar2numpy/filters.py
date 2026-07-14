"""Decode-time channel/azimuth filtering contracts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np

from .calibration import Calibration

_CHANNEL_COUNT = 128
_RAW_AZIMUTH_COUNT = 36_000
_MAX_RANGES_PER_CHANNEL = 4
_ALGORITHM = "channel_azimuth_filter.v1"
_Range = tuple[float, float]
_RangeItems = tuple[tuple[int, tuple[_Range, ...]], ...]


class _FrozenRangeMapping(Mapping[int, tuple[_Range, ...]]):
    """Small immutable mapping backed by sorted tuple items."""

    def __init__(self, items: _RangeItems) -> None:
        self._items = items

    def __getitem__(self, key: int) -> tuple[_Range, ...]:
        for channel, ranges in self._items:
            if channel == key:
                return ranges
        raise KeyError(key)

    def __iter__(self) -> Iterator[int]:
        return (channel for channel, _ranges in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self._items) == dict(other.items())
        return NotImplemented

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def items_tuple(self) -> _RangeItems:
        return self._items


@dataclass(frozen=True)
class ChannelAzimuthFilter:
    """Immutable channel and calibrated-azimuth drop-rule specification."""

    drop_channels: tuple[int, ...] = ()
    drop_azimuth_ranges_by_channel: Mapping[int, tuple[_Range, ...]] | None = None

    def __post_init__(self) -> None:
        drop_channels = {_validate_channel(ch, "drop_channels") for ch in self.drop_channels}
        range_items: dict[int, tuple[_Range, ...]] = {}

        if self.drop_azimuth_ranges_by_channel is not None:
            for raw_channel, raw_ranges in self.drop_azimuth_ranges_by_channel.items():
                channel = _validate_channel(raw_channel, "drop_azimuth_ranges_by_channel")
                ranges, full_circle = _normalize_ranges(channel, raw_ranges)
                if full_circle:
                    drop_channels.add(channel)
                    continue
                if channel in drop_channels:
                    continue
                range_items[channel] = ranges

        normalized_items = tuple(
            (channel, ranges)
            for channel, ranges in sorted(range_items.items())
            if channel not in drop_channels
        )
        object.__setattr__(self, "drop_channels", tuple(sorted(drop_channels)))
        object.__setattr__(
            self,
            "drop_azimuth_ranges_by_channel",
            _FrozenRangeMapping(normalized_items),
        )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ChannelAzimuthFilter:
        """Build a filter spec from the JSON-compatible summary shape."""
        drop_channels = payload.get("drop_channels", ())
        channel_ranges = payload.get("channel_azimuth_drops", ())
        if not isinstance(channel_ranges, Iterable) or isinstance(channel_ranges, (str, bytes)):
            raise ValueError("channel_azimuth_drops must be a list of channel records")

        ranges_by_channel: dict[int, tuple[_Range, ...]] = {}
        full_circle_channels: set[int] = set()
        for record in channel_ranges:
            if not isinstance(record, Mapping):
                raise ValueError("channel_azimuth_drops entries must be objects")
            channel_value = record.get("channel")
            ranges_value = record.get("ranges_deg", ())
            channel = _validate_channel(channel_value, "channel_azimuth_drops.channel")
            ranges, full_circle = _normalize_ranges(channel, _coerce_ranges(ranges_value, channel))
            if full_circle:
                if channel in ranges_by_channel:
                    raise ValueError(f"duplicate channel {channel} has conflicting ranges")
                full_circle_channels.add(channel)
                continue
            if channel in full_circle_channels or (
                channel in ranges_by_channel and ranges_by_channel[channel] != ranges
            ):
                raise ValueError(f"duplicate channel {channel} has conflicting ranges")
            ranges_by_channel[channel] = ranges

        if not isinstance(drop_channels, Iterable) or isinstance(drop_channels, (str, bytes)):
            raise ValueError("drop_channels must be an iterable of channel integers")
        return cls(
            tuple(_validate_channel(ch, "drop_channels") for ch in drop_channels)
            + tuple(sorted(full_circle_channels)),
            ranges_by_channel,
        )

    def is_empty(self) -> bool:
        return not self.drop_channels and not self.drop_azimuth_ranges_by_channel

    def normalized(self) -> ChannelAzimuthFilter:
        return self

    def fingerprint(self) -> str:
        return _sha256_json(_spec_payload(self))

    def summary(self) -> dict[str, object]:
        return _spec_payload(self)


@dataclass(frozen=True)
class PreparedChannelAzimuthFilter:
    """Calibration-compiled lookup table for channel/azimuth drop rules."""

    spec: ChannelAzimuthFilter
    spec_fingerprint: str
    calibration_fingerprint: str
    fingerprint: str
    drop_by_raw_azimuth_and_channel: np.ndarray

    @classmethod
    def from_spec(
        cls,
        spec: ChannelAzimuthFilter,
        calibration: Calibration,
    ) -> PreparedChannelAzimuthFilter:
        normalized = spec.normalized()
        lookup = np.zeros((_RAW_AZIMUTH_COUNT, _CHANNEL_COUNT), dtype=np.bool_)

        for channel in normalized.drop_channels:
            lookup[:, channel - 1] = True

        raw_degrees = np.arange(_RAW_AZIMUTH_COUNT, dtype=np.float64) * 0.01
        normalized_ranges = normalized.drop_azimuth_ranges_by_channel
        assert normalized_ranges is not None
        for channel, ranges in normalized_ranges.items():
            channel_index = channel - 1
            corrected = (raw_degrees + calibration.azimuth_offsets_deg[channel_index]) % 360.0
            drop = np.zeros(_RAW_AZIMUTH_COUNT, dtype=np.bool_)
            for start, end in ranges:
                if start < end:
                    drop |= (corrected >= start) & (corrected < end)
                else:
                    drop |= (corrected >= start) | (corrected < end)
            lookup[:, channel_index] = drop

        spec_fingerprint = normalized.fingerprint()
        calibration_fingerprint = calibration_azimuth_fingerprint(calibration)
        fingerprint = _sha256_json(
            {
                "algorithm": _ALGORITHM,
                "spec_fingerprint": spec_fingerprint,
                "calibration_fingerprint": calibration_fingerprint,
                "shape": [_RAW_AZIMUTH_COUNT, _CHANNEL_COUNT],
            }
        )
        return cls(
            spec=normalized,
            spec_fingerprint=spec_fingerprint,
            calibration_fingerprint=calibration_fingerprint,
            fingerprint=fingerprint,
            drop_by_raw_azimuth_and_channel=lookup,
        )

    def drop_mask_for_raw_azimuth(self, raw_azimuth: int) -> np.ndarray:
        if not isinstance(raw_azimuth, int) or not 0 <= raw_azimuth < _RAW_AZIMUTH_COUNT:
            raise ValueError(
                f"raw_azimuth must be an integer in 0..{_RAW_AZIMUTH_COUNT - 1}; "
                f"got {raw_azimuth!r}"
            )
        return cast(np.ndarray, self.drop_by_raw_azimuth_and_channel[raw_azimuth])

    def summary(self) -> dict[str, object]:
        payload = self.spec.summary()
        return {
            "algorithm": _ALGORITHM,
            "spec_fingerprint": self.spec_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "fingerprint": self.fingerprint,
            **payload,
            "lookup_shape": [_RAW_AZIMUTH_COUNT, _CHANNEL_COUNT],
            "dropped_lookup_cells": int(np.count_nonzero(self.drop_by_raw_azimuth_and_channel)),
        }


@dataclass(frozen=True)
class ChannelAzimuthFilterDiagnostics:
    """Point-count diagnostics for the last frame emitted by a filtered decoder."""

    mode: Literal["off", "shadow", "drop"]
    enabled: bool
    active: bool
    spec_fingerprint: str | None
    calibration_fingerprint: str | None
    prepared_fingerprint: str | None
    input_points: int
    output_points: int
    dropped_points: int
    dropped_points_by_channel: dict[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "enabled": self.enabled,
            "active": self.active,
            "spec_fingerprint": self.spec_fingerprint,
            "calibration_fingerprint": self.calibration_fingerprint,
            "prepared_fingerprint": self.prepared_fingerprint,
            "input_points": self.input_points,
            "output_points": self.output_points,
            "dropped_points": self.dropped_points,
            "dropped_points_by_channel": {
                str(channel): count
                for channel, count in sorted(self.dropped_points_by_channel.items())
            },
        }


def calibration_azimuth_fingerprint(calibration: Calibration) -> str:
    offsets = np.asarray(calibration.azimuth_offsets_deg, dtype="<f8")
    return hashlib.sha256(offsets.tobytes()).hexdigest()


def _validate_channel(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} channel must be an integer in 1..128; got {value!r}")
    if not 1 <= value <= _CHANNEL_COUNT:
        raise ValueError(f"{field_name} channel must be in 1..128; got {value}")
    return value


def _normalize_ranges(
    channel: int,
    raw_ranges: Iterable[tuple[float, float]],
) -> tuple[tuple[_Range, ...], bool]:
    ranges = _coerce_ranges(raw_ranges, channel)

    normalized: list[_Range] = []
    for start, end in ranges:
        span = end - start
        if not math.isclose(span, 0.0) and math.isclose(span % 360.0, 0.0):
            return (), True

        norm_start = start % 360.0
        norm_end = end % 360.0
        if math.isclose(norm_start, norm_end):
            raise ValueError(
                f"drop_azimuth_ranges_by_channel channel {channel} has zero-width range"
            )
        normalized.append((norm_start, norm_end))

    deduplicated = tuple(sorted(set(normalized)))
    if len(deduplicated) > _MAX_RANGES_PER_CHANNEL:
        raise ValueError(
            f"drop_azimuth_ranges_by_channel channel {channel} supports at most "
            f"{_MAX_RANGES_PER_CHANNEL} ranges"
        )
    return deduplicated, False


def _coerce_ranges(raw_ranges: object, channel: int) -> tuple[_Range, ...]:
    if not isinstance(raw_ranges, Iterable) or isinstance(raw_ranges, (str, bytes)):
        raise ValueError(f"ranges for channel {channel} must be an iterable of 2-item ranges")

    ranges: list[_Range] = []
    for raw_range in raw_ranges:
        if not isinstance(raw_range, Iterable) or isinstance(raw_range, (str, bytes)):
            raise ValueError(f"range for channel {channel} must be a 2-item iterable")
        values = tuple(raw_range)
        if len(values) != 2:
            raise ValueError(f"range for channel {channel} must contain exactly 2 values")
        start = _validate_degree(values[0], channel)
        end = _validate_degree(values[1], channel)
        ranges.append((start, end))
    return tuple(ranges)


def _validate_degree(value: object, channel: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"range for channel {channel} must contain finite numeric degrees")
    degree = float(value)
    if not math.isfinite(degree):
        raise ValueError(f"range for channel {channel} must contain finite numeric degrees")
    return degree


def _spec_payload(spec: ChannelAzimuthFilter) -> dict[str, object]:
    ranges_by_channel = spec.drop_azimuth_ranges_by_channel
    assert ranges_by_channel is not None
    return {
        "drop_channels": list(spec.drop_channels),
        "channel_azimuth_drops": [
            {
                "channel": channel,
                "ranges_deg": [[start, end] for start, end in ranges],
            }
            for channel, ranges in ranges_by_channel.items()
        ],
    }


def _sha256_json(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
