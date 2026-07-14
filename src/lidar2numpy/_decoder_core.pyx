# cython: language_level=3
"""Compiled dense single-return implementation of the spherical decoder."""

import numpy as np
cimport numpy as cnp
from libc.string cimport memcpy

from .decoder import _feed_packet_spherical, _parse_tail, _validate_payload
from .firing_times import FIRING_OFFSETS_S


cdef int _BLOCK1_AZ_OFFSET = 12
cdef int _BLOCK1_CH_OFFSET = 14
cdef int _BLOCK2_AZ_OFFSET = 526
cdef int _BLOCK2_CH_OFFSET = 528
cdef int _POINT_SIZE = 24


cdef inline unsigned short _u16(const unsigned char[::1] payload, int offset):
    return <unsigned short>payload[offset] | (<unsigned short>payload[offset + 1] << 8)


cdef bint _is_dense(const unsigned char[::1] payload):
    cdef int block_offset
    cdef int channel
    for block_offset in (_BLOCK1_CH_OFFSET, _BLOCK2_CH_OFFSET):
        for channel in range(128):
            if _u16(payload, block_offset + channel * 4) == 0:
                return False
    return True


cdef void _assert_spherical_layout(object out):
    cdef object fields = out.dtype.fields
    if out.dtype.itemsize != _POINT_SIZE:
        raise RuntimeError("SPHERICAL_DTYPE itemsize changed; compiled decoder layout is invalid")
    if (
        fields["channel"][1] != 0
        or fields["azimuth_deg"][1] != 2
        or fields["distance_m"][1] != 6
        or fields["intensity"][1] != 10
        or fields["timestamp"][1] != 14
        or fields["contamination"][1] != 22
        or fields["noise_level"][1] != 23
    ):
        raise RuntimeError("SPHERICAL_DTYPE field offsets changed; compiled decoder layout is invalid")


cdef void _fill_dense_block(
    cnp.uint8_t[::1] raw_out,
    Py_ssize_t output_start,
    const unsigned char[::1] payload,
    int channel_offset,
    unsigned short az_raw,
    const cnp.float64_t[::1] azimuth_offsets,
    const cnp.float64_t[::1] firing_offsets,
    double block_start_s,
):
    cdef int channel
    cdef int source_offset
    cdef Py_ssize_t destination_offset
    cdef unsigned short channel_number
    cdef unsigned short distance_raw
    cdef unsigned char reflectivity
    cdef unsigned char confidence
    cdef float azimuth_deg
    cdef float distance_m
    cdef float intensity
    cdef double timestamp

    for channel in range(128):
        source_offset = channel_offset + channel * 4
        destination_offset = (output_start + channel) * _POINT_SIZE
        channel_number = <unsigned short>(channel + 1)
        distance_raw = _u16(payload, source_offset)
        reflectivity = payload[source_offset + 2]
        confidence = payload[source_offset + 3]
        azimuth_deg = <float>(az_raw * 0.01 + azimuth_offsets[channel])
        distance_m = <float>(distance_raw * 0.004)
        intensity = <float>reflectivity
        timestamp = block_start_s + firing_offsets[channel]

        memcpy(&raw_out[destination_offset], &channel_number, sizeof(unsigned short))
        memcpy(&raw_out[destination_offset + 2], &azimuth_deg, sizeof(float))
        memcpy(&raw_out[destination_offset + 6], &distance_m, sizeof(float))
        memcpy(&raw_out[destination_offset + 10], &intensity, sizeof(float))
        memcpy(&raw_out[destination_offset + 14], &timestamp, sizeof(double))
        raw_out[destination_offset + 22] = confidence >> 6
        raw_out[destination_offset + 23] = confidence & 0x3F


def feed_packet_spherical(bytes payload, object calibration, object assembler):
    """Decode dense single-return packets directly into the frame buffer.

    Sparse, empty, and dual-return packets retain the reference implementation
    until their separately tested C1.5 paths are added.
    """
    cdef const unsigned char[::1] packet = payload
    cdef object return_mode
    cdef double t0
    cdef object frame
    cdef bint should_buffer
    cdef object out
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] raw_array
    cdef cnp.uint8_t[::1] raw_out
    cdef const cnp.float64_t[::1] azimuth_offsets = calibration.azimuth_offsets_deg
    cdef const cnp.float64_t[::1] firing_offsets = FIRING_OFFSETS_S
    cdef unsigned short block1_az
    cdef unsigned short block2_az

    _validate_payload(payload)
    return_mode, t0 = _parse_tail(payload)
    if return_mode.is_dual or not _is_dense(packet):
        return _feed_packet_spherical(payload, calibration, assembler)

    block1_az = _u16(packet, _BLOCK1_AZ_OFFSET)
    frame, should_buffer = assembler._begin_packet(block1_az)
    if not should_buffer:
        return frame

    out = assembler._reserve(256)
    _assert_spherical_layout(out)
    raw_array = out.view(np.uint8).reshape(-1)
    raw_out = raw_array
    block2_az = _u16(packet, _BLOCK2_AZ_OFFSET)
    _fill_dense_block(
        raw_out,
        0,
        packet,
        _BLOCK1_CH_OFFSET,
        block1_az,
        azimuth_offsets,
        firing_offsets,
        t0 - 0.001999111,
    )
    _fill_dense_block(
        raw_out,
        128,
        packet,
        _BLOCK2_CH_OFFSET,
        block2_az,
        azimuth_offsets,
        firing_offsets,
        t0 - 0.001888,
    )
    return frame
