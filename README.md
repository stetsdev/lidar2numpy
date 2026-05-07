# lidar2numpy

Pure-Python library that decodes raw Hesai JT128 LiDAR UDP packets into structured NumPy arrays. Forked from [MapIV/hydra4](https://github.com/MapIV/hydra4).

lidar2numpy strips away hydra4's ROS and pypcd4 dependencies. Feed it raw 1100-byte UDP payloads (from a live socket, pcap replay, or any other source); it applies per-channel angle corrections from a calibration CSV, computes XYZ coordinates, and emits one structured NumPy array per complete frame.

## Why This Fork Exists

hydra4 targets the Pandar XT-32 and OT-128 within a ROS ecosystem. lidar2numpy targets the Hesai JT128 in a standalone industrial application with no ROS dependency. Key changes from upstream:

- **JT128 packet decoder** added (the JT128 packet layout differs meaningfully from XT-32/OT-128; offsets, firing-time table, and confidence byte are JT128-specific)
- **pypcd4 dependency removed** — output is plain NumPy structured arrays
- **ROS message layer removed** — input is raw UDP bytes
- **No I/O** — the library decodes bytes you hand it; UDP socket / pcap management is the caller's responsibility

## Supported Hardware

| Model | Channels | FoV | Range | Interface |
|-------|----------|-----|-------|-----------|
| Hesai JT128 | 128 | 360° × 187° (hyper-hemispherical) | Up to 60 m | Gigabit Ethernet (UDP) |

## Requirements

- Python ≥ 3.12
- numpy ≥ 1.22.0

## Installation

```bash
git clone https://github.com/<your-org>/lidar2numpy.git
cd lidar2numpy
uv sync
```

## Usage

```python
from lidar2numpy import Decoder, load_calibration

calibration = load_calibration("angle_corrections.csv")
decoder = Decoder(calibration)

for payload in udp_payloads:  # 1100-byte bytes objects from socket / pcap
    frame = decoder.feed(payload)
    if frame is not None:
        # frame is np.ndarray with POINT_DTYPE; one complete 360° rotation
        ...
```

## Output Contract

Each emitted frame is a NumPy structured array with dtype:

| Field | Type | Description |
|-------|------|-------------|
| `x` | float32 | X coordinate (metres) |
| `y` | float32 | Y coordinate (metres, +Y = 0° azimuth) |
| `z` | float32 | Z coordinate (metres) |
| `intensity` | float32 | Reflectivity 0–255 |
| `ring` | uint16 | Channel number 1–128 |
| `timestamp` | float64 | Per-point Unix epoch seconds (LiDAR hardware clock) |
| `contamination` | uint8 | Lens contamination level 0–3 |
| `noise_level` | uint8 | Discrete noise likelihood 0–63 |

Points with no return (raw distance = 0) are excluded.

## Project Structure

```
src/lidar2numpy/
├── __init__.py            # Public API
├── decoder.py             # decode_packet(): bytes → POINT_DTYPE array
├── frame_assembler.py     # FrameAssembler: azimuth-rollover frame detection
├── structs.py             # POINT_DTYPE, ReturnMode, packet constants
├── calibration.py         # load_calibration() / default_calibration()
├── firing_times.py        # Per-channel firing-time offset table
└── calibrations/
    └── jt128_default.csv  # Nominal Appendix A calibration
docs/
└── jt128-packet-format.md # Authoritative packet format reference
```

## Development

```bash
uv sync

uv run pytest tests/ -x --tb=short
uv run mypy src/ --strict
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

Always invoke tooling via `uv run`. Manage dependencies via `uv add` / `uv add --dev` — do not edit `pyproject.toml` dependency lists by hand.

## Upstream

Forked from [MapIV/hydra4](https://github.com/MapIV/hydra4) (Apache 2.0). hydra4 decodes Hesai Pandar XT-32 and OT-128 packets into pypcd4 PointCloud objects for ROS workflows.

## License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt).
