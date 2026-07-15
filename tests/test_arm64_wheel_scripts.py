"""Contracts for the native ARM64 wheel build and verification scripts."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[1]


def _load_script(name: str) -> object:
    path = _ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_native_arm64_guard_rejects_wrong_host(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script("build_arm64_wheel.py")
    monkeypatch.setattr(module.sys, "platform", "darwin")
    monkeypatch.setattr(module.platform, "machine", lambda: "arm64")

    with pytest.raises(SystemExit, match="native ARM64 Linux"):
        module.require_native_arm64_linux()


def test_wheel_verifier_rejects_pure_python_and_wrong_platform_tags() -> None:
    module = _load_script("verify_compiled_wheel.py")

    with pytest.raises(ValueError, match="platform-specific"):
        module.validate_wheel_filename(Path("lidar2numpy-0.1.0-py3-none-any.whl"))
    with pytest.raises(ValueError, match="aarch64"):
        module.validate_wheel_filename(Path("lidar2numpy-0.1.0-cp312-cp312-linux_x86_64.whl"))


def test_sdist_manifest_includes_cython_source() -> None:
    manifest = (_ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "src/lidar2numpy/_decoder_core.pyx" in manifest
