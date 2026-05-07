# lidar2numpy

Pure-Python module that decodes raw Hesai JT128 LiDAR UDP packets into structured NumPy arrays. Forked from [MapIV/hydra4](https://github.com/MapIV/hydra4).

lidar2numpy strips away hydra4's ROS and pypcd4 dependencies. It reads raw UDP datagrams (from a live socket or pcap replay), applies per-channel angle corrections from a calibration CSV, computes XYZ coordinates, and returns Nx3+ NumPy arrays ready for downstream processing.

## Why This Fork Exists

hydra4 targets the Pandar XT-32 and OT-128 within a ROS ecosystem. lidar2numpy targets the Hesai JT128 in a standalone industrial application with no ROS dependency. Key changes from the upstream:

- **JT128 packet decoder** added (adapted from the OT-128 decoder using the JT128 user manual's packet format specification)
- **pypcd4 dependency removed** — output is plain NumPy arrays, not `PointCloud` objects
- **ROS message layer removed** — input is raw UDP packet bytes, not `pandar_msgs/PandarScan`
- **JT128 calibration CSV** added

## Supported Hardware

| Model | Channels | FoV | Range | Interface |
|-------|----------|-----|-------|-----------|
| Hesai JT128 | 128 | 360° × 187° (hyper-hemispherical) | Up to 60m | Gigabit Ethernet (UDP) |

## Requirements

- Python ≥ 3.12
- numpy ≥ 1.22.0

## Installation

```bash
# Clone and install in development mode
git clone https://github.com/<your-org>/lidar2numpy.git
cd lidar2numpy
uv sync
```

## Output Fields

| Field | Type | Description |
|-------|------|-------------|
| `x` | float32 | X coordinate (metres) |
| `y` | float32 | Y coordinate (metres) |
| `z` | float32 | Z coordinate (metres) |
| `intensity` | float32 | Return intensity (0–255) |
| `ring` | uint16 | Channel / laser ring index |
| `timestamp` | float64 | Per-point timestamp (Unix epoch) |

## Project Structure

```
src/lidar2numpy/
├── __init__.py
├── decoder.py           # Public API — Decoder class
├── structs.py           # ReturnMode enum, Block dataclass
├── calibrations/        # Built-in calibration CSVs
│   └── jt128.csv
└── decoders/
    ├── base.py          # Abstract base decoder
    └── jt128.py         # JT128 packet decoder
```

## Development

```bash
uv sync --group dev

# Tests
uv run pytest tests/ -vv

# Lint and format
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```

## Upstream

Forked from [MapIV/hydra4](https://github.com/MapIV/hydra4) (Apache 2.0). The original hydra4 library decodes Hesai Pandar XT-32 and OT-128 packets into pypcd4 PointCloud objects for ROS workflows.

## License

Apache License 2.0 — see [LICENSE.txt](LICENSE.txt).
