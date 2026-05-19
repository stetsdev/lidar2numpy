# lidar2numpy Usage Guide

`lidar2numpy` decodes raw Hesai JT128 LiDAR UDP payloads into structured NumPy arrays.
It is a library first: live UDP capture, pcap replay orchestration, visualization, and
downstream point cloud processing are caller responsibilities.

The package expects raw JT128 UDP payloads, not full Ethernet packets. A valid JT128
payload is 1100 bytes.

## Installation

From the repository root:

```bash
uv sync
```

The runtime library depends only on NumPy. The repository's development environment also
installs tools and optional script dependencies such as `mcap`.

## Library API

Import the public API from `lidar2numpy`:

```python
from lidar2numpy import (
    Decoder,
    FrameAssembler,
    block1_azimuth,
    decode_packet,
    default_calibration,
    load_calibration,
    read_pcap_payloads,
    to_cartesian,
)
```

### Recommended Streaming Decoder

Use `Decoder` when you want to feed packets one at a time and receive complete 360 degree
frames when azimuth rollover is detected.

```python
from lidar2numpy import Decoder, load_calibration

calibration = load_calibration("angle_corrections.csv")
decoder = Decoder(calibration)

for payload in udp_payloads:  # iterable of 1100-byte bytes objects
    frame = decoder.feed(payload)
    if frame is not None:
        process_frame(frame)

final_frame = decoder.flush()
if final_frame is not None:
    process_frame(final_frame)
```

`Decoder.feed(payload)` returns `None` until a complete frame is ready. The first partial
frame after startup is discarded. `Decoder.flush()` drains the in-progress frame at end
of stream after the first rollover has been seen.

Constructor:

```python
Decoder(calibration=None, output_mode="cartesian")
```

Parameters:

| Parameter | Description |
| --- | --- |
| `calibration` | `None`, a `Calibration`, a path string, a `Path`, or a binary file-like object. `None` loads the bundled nominal JT128 calibration. |
| `output_mode` | `"cartesian"` for XYZ `POINT_DTYPE` frames, or `"spherical"` for polar `SPHERICAL_DTYPE` frames. |

### Calibration

Prefer the per-unit angle correction CSV supplied by Hesai:

```python
from lidar2numpy import load_calibration

calibration = load_calibration("angle_corrections.csv")
```

`load_calibration(source)` accepts a path or binary file-like object. It expects 128
channel rows with `Channel`, `Elevation`, and `Azimuth` columns. If the CSV includes a
SHA-256 checksum row, the checksum is validated. If the checksum is missing or mismatched,
the loader emits a `UserWarning` and continues.

For development without a per-unit file:

```python
from lidar2numpy import default_calibration

calibration = default_calibration()
```

The bundled default is nominal calibration data and should not replace the sensor-specific
CSV in production.

### Decode One Packet

Use `decode_packet()` when you only need per-packet points or when you want to manage
frame assembly yourself.

```python
from lidar2numpy import decode_packet, load_calibration

calibration = load_calibration("angle_corrections.csv")
points = decode_packet(payload, calibration)
```

`decode_packet(payload, calibration)`:

| Behavior | Details |
| --- | --- |
| Input | One 1100-byte raw JT128 UDP payload. |
| Output | A NumPy structured array with `POINT_DTYPE`. |
| Invalid returns | Channels with raw distance `0` are excluded. |
| Validation errors | Raises `ValueError` for invalid length, SOP bytes, unsupported protocol version, or missing confidence flag. |
| Unsupported modes | Raises `ValueError` for unknown return-mode bytes. Known single and dual return modes are decoded. |

### Manual Frame Assembly

Use `FrameAssembler` if you decode packets directly and still want complete frames.

```python
from lidar2numpy import FrameAssembler, block1_azimuth, decode_packet, load_calibration

calibration = load_calibration("angle_corrections.csv")
assembler = FrameAssembler()

for payload in udp_payloads:
    points = decode_packet(payload, calibration)
    azimuth = block1_azimuth(payload)
    frame = assembler.add_packet(points, azimuth)
    if frame is not None:
        process_frame(frame)

final_frame = assembler.flush()
if final_frame is not None:
    process_frame(final_frame)
```

`block1_azimuth(payload)` returns Block 1's raw azimuth as an integer in 0.01 degree
units. `FrameAssembler` detects a new frame when Block 1 azimuth rolls over from a high
value back toward zero.

### Spherical Mode

Use spherical mode when you want to filter or index data in polar coordinates before
paying the cost of XYZ conversion.

```python
from lidar2numpy import Decoder, load_calibration, to_cartesian

calibration = load_calibration("angle_corrections.csv")
decoder = Decoder(calibration, output_mode="spherical")

for payload in udp_payloads:
    spherical_frame = decoder.feed(payload)
    if spherical_frame is None:
        continue

    foreground = spherical_frame[~background_mask(spherical_frame)]
    xyz = to_cartesian(foreground, calibration)
    process_frame(xyz)
```

`to_cartesian(spherical, calibration)` accepts a complete spherical frame or any slice,
mask, or subset of one.

### Reading pcap Files

`read_pcap_payloads()` is a minimal helper for standard pcap files containing Ethernet,
IPv4, UDP, and a JT128 payload.

```python
from pathlib import Path

from lidar2numpy import Decoder, load_calibration, read_pcap_payloads

calibration = load_calibration("angle_corrections.csv")
decoder = Decoder(calibration)

for payload in read_pcap_payloads(Path("capture.pcap")):
    frame = decoder.feed(payload)
    if frame is not None:
        process_frame(frame)
```

Signature:

```python
read_pcap_payloads(pcap_path, udp_payload_size=1100)
```

The helper strips fixed 14-byte Ethernet, 20-byte IP, and 8-byte UDP headers, then yields
payloads of `udp_payload_size`. It supports microsecond and nanosecond pcap magic values
and both byte orders. It does not parse pcapng.

## Output Arrays

All output arrays exclude points where distance is zero.

### `POINT_DTYPE`

Emitted by `decode_packet()` and `Decoder(..., output_mode="cartesian")`.

| Field | Type | Description |
| --- | --- | --- |
| `x` | `float32` | X coordinate in metres. |
| `y` | `float32` | Y coordinate in metres. Positive Y is 0 degree azimuth. |
| `z` | `float32` | Z coordinate in metres. |
| `intensity` | `float32` | Reflectivity, 0-255. |
| `ring` | `uint16` | Channel number, 1-128. |
| `timestamp` | `float64` | Per-point Unix epoch seconds from the LiDAR hardware clock. |
| `contamination` | `uint8` | Confidence byte bits 7:6, range 0-3. |
| `noise_level` | `uint8` | Confidence byte bits 5:0, range 0-63. |

### `SPHERICAL_DTYPE`

Emitted by `Decoder(..., output_mode="spherical")`.

| Field | Type | Description |
| --- | --- | --- |
| `channel` | `uint16` | Channel number, 1-128. |
| `azimuth_deg` | `float32` | Calibration-corrected azimuth in degrees. |
| `distance_m` | `float32` | Measured distance in metres. |
| `intensity` | `float32` | Reflectivity, 0-255. |
| `timestamp` | `float64` | Per-point Unix epoch seconds from the LiDAR hardware clock. |
| `contamination` | `uint8` | Confidence byte bits 7:6, range 0-3. |
| `noise_level` | `uint8` | Confidence byte bits 5:0, range 0-63. |

## CLI Scripts

Run repository scripts through `uv run`.

### `scripts/pcap_to_mcap.py`

Converts a JT128 pcap capture to an MCAP file containing Foxglove
`foxglove.PointCloud` JSON messages on topic `/lidar/points`.

Basic usage:

```bash
uv run python scripts/pcap_to_mcap.py capture.pcap \
    --calibration angle_corrections.csv \
    --output capture.mcap
```

Use nominal calibration when no per-unit CSV is available:

```bash
uv run python scripts/pcap_to_mcap.py capture.pcap --output capture.mcap
```

Convert only the first few frames:

```bash
uv run python scripts/pcap_to_mcap.py capture.pcap --max-frames 10
```

Rebase unsynced sensor timestamps to the current wall clock:

```bash
uv run python scripts/pcap_to_mcap.py capture.pcap --output capture.mcap --rebase
```

Rebase unsynced sensor timestamps to a specific UTC start time:

```bash
uv run python scripts/pcap_to_mcap.py capture.pcap --output capture.mcap \
    --rebase --start-time "2026-05-06T14:30:00"
```

Command line reference:

```text
uv run python scripts/pcap_to_mcap.py [OPTIONS] PCAP
```

| Argument | Required | Description |
| --- | --- | --- |
| `PCAP` | Yes | Input pcap file path. |
| `-c`, `--calibration CALIBRATION` | No | Per-unit angle correction CSV from Hesai. Defaults to nominal calibration values. |
| `-o`, `--output OUTPUT` | No | Output `.mcap` file path. Defaults to the input path with `.mcap` suffix. |
| `--rebase` | No | If the first decoded frame timestamp is before year 2000, add an offset so the output starts at wall clock time or `--start-time`. If timestamps appear synced, no offset is applied. |
| `--start-time START_TIME` | No | ISO 8601 start time used with `--rebase`. Naive datetimes are treated as UTC. Defaults to the current time. |
| `--max-frames MAX_FRAMES` | No | Maximum number of frames to convert. Defaults to all frames. |
| `-h`, `--help` | No | Show help and exit. |

Notes:

| Topic | Details |
| --- | --- |
| Dependency | The script imports `mcap` at runtime. In this repository, `uv sync` installs it from the dev dependency group. If missing, install it with `uv add --dev mcap`. |
| Calibration | The script prints a warning to stderr when no calibration CSV is provided and nominal defaults are used. |
| Timestamp source | Point timestamps come from the LiDAR hardware clock decoded from the packet tail. |
| Output fields | The Foxglove point buffer contains `x`, `y`, `z`, and `intensity`. Other `POINT_DTYPE` fields are not written to the MCAP point records. |
| Empty captures | The script exits with status 1 if no complete frames are decoded. |

## Common Patterns

### Live UDP Input

The library does not open sockets, but a caller can pass received UDP payloads directly:

```python
import socket

from lidar2numpy import Decoder, load_calibration

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", 2368))

decoder = Decoder(load_calibration("angle_corrections.csv"))

while True:
    payload, _addr = sock.recvfrom(2048)
    frame = decoder.feed(payload)
    if frame is not None:
        process_frame(frame)
```

### Saving Decoded Frames

Because frames are ordinary structured NumPy arrays, they can be saved directly:

```python
from pathlib import Path

import numpy as np

from lidar2numpy import Decoder, load_calibration, read_pcap_payloads

decoder = Decoder(load_calibration("angle_corrections.csv"))

for i, payload in enumerate(read_pcap_payloads(Path("capture.pcap"))):
    frame = decoder.feed(payload)
    if frame is not None:
        np.save(f"frame_{i:04d}.npy", frame)
```

## Limitations

- JT128 only.
- Raw UDP payloads must be 1100 bytes.
- Dual-return packets are decoded as both block returns. No deduplication is applied.
- `read_pcap_payloads()` reads pcap files, not pcapng files.
- No UDP socket management is provided by the library.
- No point cloud filtering, clustering, tracking, ROS messages, or PCD export is provided.
