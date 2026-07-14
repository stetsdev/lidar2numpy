"""Packaging contract for the optional compiled decoder backend."""

from __future__ import annotations

import tomllib
from pathlib import Path

_ROOT = Path(__file__).parents[1]


def test_build_configuration_declares_cython_and_numpy() -> None:
    """A source build has the isolated requirements needed for the extension."""
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    build_system = config["build-system"]

    assert build_system["build-backend"] == "setuptools.build_meta"
    requirements = build_system["requires"]
    assert any(requirement.lower().startswith("cython") for requirement in requirements)
    assert any(requirement.lower().startswith("numpy") for requirement in requirements)


def test_compiled_decoder_source_and_build_entrypoint_exist() -> None:
    """The distribution contains an extension source and setuptools entrypoint."""
    assert (_ROOT / "setup.py").is_file()
    assert (_ROOT / "src" / "lidar2numpy" / "_decoder_core.pyx").is_file()
