"""Angle correction calibration loader for the Hesai JT128.

The per-unit CSV file shipped with each sensor is the authoritative source
of per-channel elevation and azimuth-offset angles. This module loads it into
an immutable ``Calibration`` dataclass for use by the decoder.

CSV format (from docs/jt128-packet-format.md §Angle Correction File):

    [EEFF,1,1]              ← optional first row, skipped if present
    Channel,Elevation,Azimuth
    1,<elev_deg>,<az_deg>
    ...
    128,<elev_deg>,<az_deg>
    [<sha256_hex>,,]        ← optional trailing checksum row

Elevation is stored in radians (converted on load). Azimuth offset is kept
in degrees so it can be added directly to the block's ``azimuth × 0.01``
value before the combined angle is converted to radians for XYZ math.
"""

from __future__ import annotations

import csv
import hashlib
import io
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Union

import numpy as np

_SOURCE = Union[str, Path, BinaryIO]

# A checksum row has a 64-hex-character first field (SHA-256 digest).
_SHA256_HEX_LEN = 64


@dataclass(frozen=True)
class Calibration:
    """Immutable per-channel angle corrections.

    Both arrays are indexed ``[channel - 1]`` (i.e. 0-based ring index).
    """

    elevations_rad: np.ndarray  # shape (128,), float64, radians
    azimuth_offsets_deg: np.ndarray  # shape (128,), float64, degrees


def load_calibration(source: _SOURCE) -> Calibration:
    """Load a Hesai angle correction CSV into a :class:`Calibration`.

    Parameters
    ----------
    source:
        A file path (``str`` or ``Path``) or a binary file-like object
        (e.g. ``io.BytesIO``). Binary mode is used so that callers can
        pass raw bytes without creating a temporary file.

    Returns
    -------
    Calibration
        128-channel elevation (radians) and azimuth-offset (degrees) arrays.

    Raises
    ------
    ValueError
        If the CSV does not contain exactly 128 data rows or if any
        channel number appears more than once.
    """
    raw_bytes = _read_bytes(source)

    # Reparse as text for SHA-256 validation (computed over the body before
    # the checksum row, including the trailing newline of the last data row).
    text = raw_bytes.decode(errors="replace")
    lines = text.splitlines()

    # Strip the optional EEFF header row.
    if lines and lines[0].startswith("EEFF"):
        lines = lines[1:]

    # Separate the optional trailing checksum row.
    checksum_expected: str | None = None
    if lines and len(lines[-1]) >= _SHA256_HEX_LEN and _is_hex(lines[-1][:_SHA256_HEX_LEN]):
        checksum_expected = lines[-1][:_SHA256_HEX_LEN]
        body_lines = lines[:-1]
    else:
        body_lines = lines

    # Validate checksum if present; warn (never raise) if absent or mismatched.
    body_text = "\r\n".join(body_lines) + "\r\n"
    if hashlib.sha256(body_text.encode()).hexdigest() != (checksum_expected or ""):
        body_text = "\n".join(body_lines) + "\n"
    if checksum_expected is not None:
        actual = hashlib.sha256(body_text.encode()).hexdigest()
        if actual != checksum_expected:
            warnings.warn(
                f"Calibration CSV checksum mismatch "
                f"(expected {checksum_expected}, got {actual}); "
                "loading anyway",
                UserWarning,
                stacklevel=2,
            )
    else:
        warnings.warn(
            "Calibration CSV has no SHA-256 checksum; integrity cannot be verified",
            UserWarning,
            stacklevel=2,
        )

    # Parse the column-header row and data rows via csv.DictReader.
    reader = csv.DictReader(io.StringIO(body_text))
    rows: list[tuple[int, float, float]] = []  # (channel, elevation_deg, azimuth_deg)
    for row in reader:
        ch = int(row["Channel"])
        elev = float(row["Elevation"])
        az = float(row["Azimuth"])
        rows.append((ch, elev, az))

    if len(rows) != 128:
        raise ValueError(f"Calibration CSV must contain exactly 128 channel rows; got {len(rows)}")

    channels = [r[0] for r in rows]
    if len(set(channels)) != 128:
        raise ValueError(f"Duplicate channel numbers found in calibration CSV: {channels}")

    # Sort by channel number and build arrays indexed 0..(127).
    rows.sort(key=lambda r: r[0])
    elevations_rad = np.array([np.deg2rad(r[1]) for r in rows], dtype=np.float64)
    elevations_rad.flags.writeable = False
    azimuth_offsets_deg = np.array([r[2] for r in rows], dtype=np.float64)
    azimuth_offsets_deg.flags.writeable = False

    return Calibration(elevations_rad=elevations_rad, azimuth_offsets_deg=azimuth_offsets_deg)


def default_calibration() -> Calibration:
    """Load the bundled nominal calibration from Appendix A of the JT128 manual.

    The per-unit file shipped with the sensor is always preferred. This
    function exists so the decoder can run without a per-unit file (e.g.
    during development before hardware is available).
    """
    csv_path = Path(__file__).parent / "calibrations" / "jt128_default.csv"
    with warnings.catch_warnings():
        # The bundled file has no checksum by design; suppress the warning.
        warnings.simplefilter("ignore", UserWarning)
        return load_calibration(csv_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _read_bytes(source: _SOURCE) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    # File-like object — read all remaining bytes.
    return source.read()


def _is_hex(s: str) -> bool:
    try:
        int(s, 16)
        return True
    except ValueError:
        return False
