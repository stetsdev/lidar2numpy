# cython: language_level=3
"""Compiled entrypoint for the spherical decoder.

C1.3 establishes the distribution artifact. C1.4 replaces this delegation
with the native packet-decode hot path while preserving its call contract.
"""

from .decoder import _feed_packet_spherical


def feed_packet_spherical(bytes payload, object calibration, object assembler):
    """Feed one payload through the stable spherical decoder call contract."""
    return _feed_packet_spherical(payload, calibration, assembler)
