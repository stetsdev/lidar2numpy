#!/usr/bin/env python3
"""Verify an installed lidar2numpy ARM64 wheel outside its source checkout."""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def require_native_arm64_linux() -> None:
    """Fail unless this is the native Linux ARM64 release-builder host."""
    if sys.platform != "linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise SystemExit("compiled-wheel verification requires native ARM64 Linux")


def validate_wheel_filename(wheel: Path) -> None:
    """Reject pure-Python, non-CPython-3.12, or non-ARM64 Linux wheels."""
    name = wheel.name
    if not name.endswith(".whl") or "none-any" in name:
        raise ValueError(f"wheel must be platform-specific, not pure Python: {name}")
    if "cp312-cp312" not in name:
        raise ValueError(f"wheel must target CPython 3.12: {name}")
    if "linux_aarch64" not in name and "linux_arm64" not in name:
        raise ValueError(f"wheel must target Linux aarch64: {name}")


def inspect_wheel(wheel: Path) -> None:
    """Check that the wheel carries the extension and bundled calibration."""
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    has_extension = any(
        name.startswith("lidar2numpy/_decoder_core") and name.endswith(".so") for name in names
    )
    if not has_extension:
        raise ValueError("wheel does not contain the compiled lidar2numpy._decoder_core extension")
    if "lidar2numpy/calibrations/jt128_default.csv" not in names:
        raise ValueError("wheel does not contain the bundled default calibration")


_SMOKE_CODE = r"""
import struct
import lidar2numpy
from lidar2numpy import Decoder, default_calibration

assert lidar2numpy.decoder_backend() == "cython"
lidar2numpy.require_compiled_backend()
assert default_calibration().azimuth_offsets_deg.shape == (128,)

def packet(return_mode, azimuth):
    payload = bytearray(1100)
    payload[0:2] = b"\xee\xff"
    payload[2:4] = b"\x01\x04"
    payload[6] = 128
    payload[7] = 2
    payload[9] = 4
    payload[10] = 1
    payload[11] = 0x23
    struct.pack_into("<H", payload, 12, azimuth)
    struct.pack_into("<H", payload, 526, (azimuth + 100) % 36000)
    for offset in (14, 528):
        struct.pack_into("<HBB", payload, offset, 250, 100, 0xC5)
    payload[1056] = return_mode
    payload[1059:1065] = bytes((125, 5, 6, 12, 0, 0))
    struct.pack_into("<I", payload, 1065, 0)
    return bytes(payload)

for mode in (0x37, 0x39):
    decoder = Decoder(default_calibration(), output_mode="spherical", backend="cython")
    for azimuth in (35000, 100, 1000, 35000):
        decoder.feed(packet(mode, azimuth))
    frame = decoder.feed(packet(mode, 100))
    assert frame is not None and len(frame) == 6
"""


def verify_clean_install(wheel: Path) -> None:
    """Install the exact wheel in a disposable environment and run smoke decodes."""
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required to verify the compiled wheel")
    with tempfile.TemporaryDirectory(prefix="lidar2numpy-wheel-") as directory:
        root = Path(directory)
        venv_python = root / "venv" / "bin" / "python"
        subprocess.run([uv, "venv", str(root / "venv"), "--python", "3.12"], check=True)
        subprocess.run([uv, "pip", "install", "--python", str(venv_python), str(wheel)], check=True)
        subprocess.run([str(venv_python), "-c", _SMOKE_CODE], check=True, cwd=root)


def main(argv: list[str] | None = None) -> int:
    """Verify one native ARM64 wheel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args(argv)
    require_native_arm64_linux()
    wheel = args.wheel.resolve()
    validate_wheel_filename(wheel)
    inspect_wheel(wheel)
    verify_clean_install(wheel)
    print(f"verified compiled ARM64 wheel: {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
