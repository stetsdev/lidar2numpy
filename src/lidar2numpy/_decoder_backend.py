"""Select the spherical decoder implementation without changing public frame semantics."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import Literal, cast

import numpy as np

from .calibration import Calibration
from .decoder import _feed_packet_spherical, _SphericalFrameAssembler

BackendRequest = Literal["auto", "python", "cython"]
BackendName = Literal["python", "cython"]
SphericalFeed = Callable[[bytes, Calibration, _SphericalFrameAssembler], np.ndarray | None]


def _load_compiled_feed() -> SphericalFeed | None:
    """Return the compiled feed function when its extension is genuinely absent or present."""
    try:
        module = import_module(f"{__package__}._decoder_core")
    except ModuleNotFoundError as exc:
        if exc.name == "lidar2numpy._decoder_core":
            return None
        raise
    return cast(SphericalFeed, module.feed_packet_spherical)


def resolve_spherical_backend(requested: str) -> tuple[BackendName, SphericalFeed]:
    """Resolve one requested backend and raise rather than silently changing an explicit request."""
    if requested not in ("auto", "python", "cython"):
        raise ValueError(f"backend must be one of 'auto', 'python', or 'cython'; got {requested!r}")
    if requested == "python":
        return "python", _feed_packet_spherical

    compiled_feed = _load_compiled_feed()
    if compiled_feed is not None:
        return "cython", compiled_feed
    if requested == "cython":
        raise RuntimeError("cython backend is unavailable in this build")
    return "python", _feed_packet_spherical


def decoder_backend() -> BackendName:
    """Return the backend selected by normal automatic decoder construction."""
    backend, _feed = resolve_spherical_backend("auto")
    return backend


def require_compiled_backend() -> None:
    """Raise if the compiled backend is unavailable for a production caller."""
    if decoder_backend() != "cython":
        raise RuntimeError("compiled Cython decoder backend is required but unavailable")
