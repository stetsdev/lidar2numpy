#!/usr/bin/env python3
"""Build and verify the native ARM64 lidar2numpy wheel for internal Debian use."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

_ROOT = Path(__file__).parents[1]


def require_native_arm64_linux() -> None:
    """Fail unless this is the native Linux ARM64 release-builder host."""
    if sys.platform != "linux" or platform.machine().lower() not in {"aarch64", "arm64"}:
        raise SystemExit("ARM64 wheel builds require native ARM64 Linux")


def _command_output(command: list[str]) -> str:
    return subprocess.check_output(command, cwd=_ROOT, text=True).strip()


def _build_info(wheel: Path) -> dict[str, str]:
    return {
        "wheel": wheel.name,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "numpy": version("numpy"),
        "cython": version("cython"),
        "compiler": _command_output(["cc", "--version"]).splitlines()[0],
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": str(bool(_command_output(["git", "status", "--porcelain"]))),
    }


def main() -> int:
    """Build one PEP 517 wheel, verify it, and emit adjacent provenance JSON."""
    require_native_arm64_linux()
    subprocess.run(["uv", "build", "--sdist", "--wheel"], cwd=_ROOT, check=True)
    wheels = sorted(_ROOT.glob("dist/lidar2numpy-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one wheel in dist/, found: {wheels}")
    wheel = wheels[0]
    subprocess.run(
        [sys.executable, "scripts/verify_compiled_wheel.py", str(wheel)],
        cwd=_ROOT,
        check=True,
    )
    info_path = wheel.with_suffix(".build-info.json")
    info_path.write_text(
        json.dumps(_build_info(wheel), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"built and verified {wheel.name}")
    print(f"build provenance: {info_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
