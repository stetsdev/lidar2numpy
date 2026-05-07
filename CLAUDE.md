# CLAUDE.md — lidar2numpy

## Project Overview

Pure-Python library that decodes raw Hesai JT128 LiDAR UDP packets into structured NumPy arrays. Forked from [MapIV/hydra4](https://github.com/MapIV/hydra4) (Pandar XT-32/OT-128 decoder for ROS). No ROS, no pypcd4, no C/C++. Consumed by the Intersection Monitor's `im-perception` process as the LiDAR ingest module.

**Tech stack:** Python 3.12, numpy (only runtime dependency).

**Dev tooling:** uv (package/venv management), ruff (lint + format), mypy (type checking), pytest (testing).

## Commands

Always use `uv run`. Never call bare `python`, `pytest`, `mypy`, or `ruff`.

```bash
uv run pytest tests/ -x --tb=short
uv run pytest tests/test_decoder.py -x --tb=short -v
uv run mypy src/ --strict
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
# All checks (run before committing)
uv run pytest tests/ -x && uv run mypy src/ --strict && uv run ruff check src/ tests/
```

Package management: `uv add` / `uv add --dev`. Never edit `pyproject.toml` dependency lists by hand.

## Project Structure

```
lidar2numpy/
├── src/lidar2numpy/
│   ├── __init__.py              # Public API exports
│   ├── decoder.py               # Raw UDP bytes → structured numpy array
│   ├── frame_assembler.py       # Azimuth rollover → complete frames
│   ├── structs.py               # ReturnMode enum, JT128 packet constants
│   ├── calibration.py           # Angle correction CSV loader
│   ├── calibrations/
│   │   └── jt128_default.csv    # Nominal calibration (Appendix A values)
│   └── firing_times.py          # Per-channel firing time offset table (Appendix B.4)
│       
├── docs/
│   └── jt128-packet-format.md  # Packet format reference
├── tests/
│   ├── test_decoder.py
│   ├── test_frame_assembler.py
│   ├── test_calibration.py
│   ├── test_structs.py
│   └── fixtures/                # Raw packet binaries, calibration CSVs
├── pyproject.toml
├── CLAUDE.md
├── LICENSE.txt                  # Apache 2.0 (from hydra4)
└── .python-version              # 3.12
```

## Reference Documents

`docs/jt128-packet-format.md` — packet structure, field definitions, coordinate formulas. **This is the authoritative reference for the decoder.** When in doubt about a field offset, byte order, or formula, consult this document.

## Development Rules

### 1. Test-Driven Development

Write test → confirm failure → implement → confirm pass. One function or test suite at a time.

**Commit tests before implementing.** If the test file is committed, implementation changes that silently alter expectations show in `git diff`.

### 2. This Is a Library, Not a Service

lidar2numpy decodes bytes it receives. It does not:
- Manage UDP sockets or pcap replay (caller's responsibility)
- Log anything (`raise` with clear messages on parse errors; the consuming app handles logging)
- Process point clouds (filtering, clustering, tracking belong downstream)

### 3. Output Contract

Every decoded frame is a NumPy structured array with dtype:

```python
POINT_DTYPE = np.dtype([
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('intensity', np.float32),    # reflectivity 0–255
    ('ring', np.uint16),          # channel number 1–128
    ('timestamp', np.float64),    # Unix epoch (from LiDAR hardware clock)
    ('contamination', np.uint8),  # bits[7:6] of confidence byte, 0–3
    ('noise_level', np.uint8),    # bits[5:0] of confidence byte, 0–63
])
```

Points with distance = 0 (no valid return) are excluded from the output array.

### 4. Coordinate System

```
x = distance * cos(vert) * sin(horiz)
y = distance * cos(vert) * cos(horiz)    # Y-axis = 0° azimuth
z = distance * sin(vert)
```

Clockwise (top view) is positive azimuth. No optical center offset correction (sub-centimeter error at 5–7m ceiling mount; deferred unless needed).

### 5. Calibration

Per-unit angle correction CSV from Hesai is always preferred over the bundled nominal default. CSV format: `EEFF` header row, column headers, 128 rows of (channel, elevation°, azimuth_offset°), optional SHA-256 checksum. Validate checksum if present, warn if missing, don't fail.

### 6. Frame Boundaries

Detected by azimuth rollover: Block 1 azimuth decreases relative to previous packet → new frame. Discard partial frames at startup.

## Git Workflow

Branch per feature: `feat/constants`, `feat/calibration`, `feat/decoder`, etc.

Commit message format: `feat: implement [module] — tests passing`

Do not push to remote or merge unreviewed work.

## Testing

Tests in `tests/`, fixtures in `tests/fixtures/`.

**Unit tests (highest priority):**
- Packet decoding: synthetic packets with known field values → assert correct extraction
- Calibration: load CSV → assert channel count 128, spot-check known angles
- Coordinate conversion: hand-computed (distance, elevation, azimuth) → XYZ → assert match
- Timestamp: constructed tail bytes → assert correct Unix epoch float64
- Frame assembly: packet sequence with known azimuths → assert correct boundaries

**Integration tests:**
- End-to-end pcap replay → structured NumPy arrays → geometry sanity check

**Fixtures data**:
- Return mode: 0x37 Strongest	
## Code Style

- Type hints on all functions. `from __future__ import annotations`.
- Dataclasses for data structures (packet headers, calibration records).
- No classes for pure logic. Coordinate conversion, calibration loading, timestamp reconstruction are pure functions.
- No bare `except:`. Catch specific exceptions.
- Comments explain *why*, not *what*.
- No logging framework. Raise exceptions with clear messages on parse errors.
- Line length: 100. Indent: 4 spaces.

## Don't Touch

- `docs/jt128-packet-format.md` — reference document. Flag errors; do not edit.
- `src/lidar2numpy/calibrations/` — sensor calibration data. Do not modify.
- `LICENSE.txt` — Apache 2.0, from hydra4.
- `uv.lock` — modify only via `uv add` / `uv remove`.

## What Not To Build

- Any C or C++. Pure Python + numpy.
- ROS messages or pypcd4 PointCloud objects. Output is NumPy structured arrays.
- Point cloud processing (filtering, clustering, tracking). Downstream concern.
- UDP socket management or pcap replay. Library decodes bytes; caller manages I/O.
- Optical center offset correction (deferred; see jt128-packet-format.md).
- Firing time angular correction (v0.1 — stationary ceiling mount; angular error negligible).
- Dual return handling -- not implemented yet by Hesai

## Key Constants

- **UDP payload:** 1100 bytes (6 pre-header + 6 header + 1032 body + 56 tail)
- **Distance unit:** 4mm (`distance_raw * 0.004` → metres)
- **Per-channel record:** 4 bytes (distance uint16, reflectivity uint8, confidence uint8)
- **Return mode codes:** `0x33` First, `0x37` Strongest, `0x38` Last, `0x39` Last+Strongest (default), `0x3B` Last+First, `0x3C` First+Strongest
- **Dual return:** return mode code ≥ `0x39`
- **Channels:** 128 per block, 2 blocks per packet
