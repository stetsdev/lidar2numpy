# JT128 Point Cloud Packet Format — Decoder Reference

Reference for implementing the `lidar2numpy` JT128 decoder. Extracted from the Hesai JT128 User Manual, Section 3 (Data Structure).

---

## Packet Layout

Total point cloud data packet: **1146 bytes** (Ethernet frame). The UDP payload is **1100 bytes**.

All multi-byte fields are **unsigned little-endian** unless noted otherwise.

```
UDP payload (1100 bytes)
├── Pre-Header    6 bytes
├── Header        6 bytes
├── Body       1032 bytes
└── Tail         56 bytes
```

## Pre-Header (6 bytes)

| Field | Bytes | Value |
|-------|-------|-------|
| Start of Packet | 1 | `0xEE` |
| Start of Packet | 1 | `0xFF` |
| Protocol Version Major | 1 | `0x01` |
| Protocol Version Minor | 1 | `0x04` |
| Reserved | 2 | — |

Use `0xEE 0xFF` as the packet sync marker.

## Header (6 bytes)

| Field | Bytes | Value / Notes |
|-------|-------|---------------|
| Channel Num | 1 | Fixed `0x80` (128) |
| Block Num | 1 | Fixed `0x02` (2) |
| Reserved | 1 | — |
| Dis Unit | 1 | Fixed `0x04` → distance unit = **4 mm** |
| Return Num | 1 | `0x01` (single) or `0x02` (dual) |
| Flags | 1 | See below |

**Flags byte:**

| Bits | Field | Notes |
|------|-------|-------|
| [7:6] | Reserved | — |
| [5] | Confidence | 1 = confidence byte present per channel |
| [4:2] | Reserved | — |
| [1] | IMU | 1 = IMU data in tail (fixed on) |
| [0] | UDP Sequence | 1 = sequence number in tail (fixed on) |

The Confidence flag changes the per-channel record size: **3 bytes** without confidence, **4 bytes** with. This determines Body parsing.

## Body (1032 bytes)

Two blocks, each preceded by a 2-byte azimuth field.

```
Body layout:
├── Azimuth 1     2 bytes    (uint16, unit: 0.01°)
├── Block 1     512 bytes    (128 channels × 4 bytes)
├── Azimuth 2     2 bytes    (uint16, unit: 0.01°)
├── Block 2     512 bytes    (128 channels × 4 bytes)
└── CRC           4 bytes    (CRC-32/MPEG-2 over entire Body)
```

512 bytes per block assumes confidence is present (128 × 4). Without confidence: 128 × 3 = 384 bytes per block.

**Per-channel record (with confidence = 4 bytes):**

| Field | Bytes | Description |
|-------|-------|-------------|
| Distance | 2 | uint16. Actual distance = `Distance × Dis Unit` (i.e., × 4 mm → metres: `Distance × 0.004`) |
| Reflectivity | 1 | uint8, 0–255 (maps linearly to 0–255% reflectivity) |
| Confidence | 1 | Bits [7:6] = cover lens contamination (0–3, higher = worse). Bits [5:0] = discrete noise likelihood (0–63, higher = more likely noise). |

Distance = 0 means no return (invalid point — skip or emit as zero).

**Dual return mode:** In dual return, both blocks share the same azimuth. The pairing depends on the Return Mode field in the Tail:

| Return Mode | Block 1 (odd) | Block 2 (even) |
|-------------|---------------|----------------|
| Last and Strongest (default) | Last | Strongest (or 2nd strongest if last = strongest) |
| Last and First | Last | First (identical data if only one return) |
| First and Strongest | First | Strongest (or 2nd strongest if first = strongest) |

For the intersection monitor's use case (binary occupancy detection), single return mode is sufficient. If dual return, apply deduplication threshold per hydra4's approach.

## Tail (56 bytes)

| Field | Bytes | Type | Description |
|-------|-------|------|-------------|
| Reserved | 11 | — | — |
| Working Mode | 1 | uint8 | 0 = operating, 1 = standby |
| Return Mode | 1 | uint8 | See table below |
| Motor Speed | 2 | uint16 | Unit: 0.1 RPM. `RPM = value / 10`. Frame rate = RPM / 60. |
| Date & Time | 6 | 6 × uint8 | Year (value + 1900), Month, Day, Hour, Minute, Second — 1 byte each, standard integers (not BCD) |
| UTC Fractional Seconds | 4 | uint32 | Microseconds (0–999999) |
| Factory Information | 1 | uint8 | Fixed `0x42` |
| UDP Sequence | 4 | uint32 | Packet sequence counter |
| IMU Temperature | 2 | int16 (signed) | Unit: 0.01 °C |
| IMU Acceleration Unit | 2 | uint16 | Currently `0x7A` (122) → 0.122 mg per LSB |
| IMU Angular Velocity Unit | 2 | uint16 | Currently `0xBE9` (3049) → 30.49 mdps per LSB |
| Reserved | 4 | — | — |
| IMU X Accel | 2 | int16 | × accel unit |
| IMU Y Accel | 2 | int16 | × accel unit |
| IMU Z Accel | 2 | int16 | × accel unit |
| IMU X Angular Vel | 2 | int16 | × angular vel unit |
| IMU Y Angular Vel | 2 | int16 | × angular vel unit |
| IMU Z Angular Vel | 2 | int16 | × angular vel unit |
| CRC | 4 | uint32 | CRC-32/MPEG-2 over the Tail |

**Return Mode codes:**

| Code | Mode |
|------|------|
| `0x33` | First |
| `0x37` | Strongest |
| `0x38` | Last |
| `0x39` | Last and Strongest (default) |
| `0x3B` | Last and First |
| `0x3C` | First and Strongest |

**Timestamp reconstruction:** Combine Date & Time (whole seconds as UTC) with UTC Fractional Seconds (microseconds) to produce a Unix epoch float64.

## XYZ Coordinate Computation

Each point requires three inputs: distance, vertical angle (elevation), and horizontal angle (azimuth).

**Vertical angle:** Read from the per-unit angle correction CSV, indexed by channel number (1–128). The CSV contains the accurate vertical angle for each channel. Designed values are in Appendix A of the manual; the correction file is authoritative. Convention: 0° = horizontal, positive = upward.

**Horizontal angle = azimuth_ref + horizontal_offset + firing_time_offset**

Where:
- `azimuth_ref` = the Azimuth field for this block (uint16 × 0.01°)
- `horizontal_offset` = per-channel horizontal angle offset from the correction CSV
- `firing_time_offset` = `firing_time_us × motor_speed_dps` (angular distance the motor rotated during the channel's firing delay)

**Motor speed conversion:** `motor_speed_dps = (Motor Speed × 0.1) × 6.0` (RPM × 0.1 → RPM, × 360/60 → °/s)

**Firing time offsets:** The JT128 uses an interleaved, non-uniform firing sequence. Channel index does NOT equal firing order. Accurate per-channel firing time offsets are in the manual's Appendix B.4. For the intersection monitor's ceiling-mount, low-platform-speed application, the firing time angular offset is negligible and can be **deferred** — apply only the correction CSV offsets initially, add firing time correction later if point cloud quality requires it.

**Spherical to Cartesian conversion:**

```
vert  = vertical_angle  (radians)
horiz = horizontal_angle (radians)

x = distance × cos(vert) × sin(horiz)
y = distance × cos(vert) × cos(horiz)
z = distance × sin(vert)
```

The Y-axis is the 0° azimuth position. Clockwise (top view) is positive.

**Optical center offset correction (Appendix D)**

The formulas above produce coordinates relative to the optical center (Point A), not the LiDAR's coordinate system origin (Point O). The optical center is physically offset from the origin and rotates with the motor, so its position is azimuth-dependent.

To obtain coordinates relative to the origin:

**Optical center position at 0° azimuth (from Figure 5, mm → m)**

Read x_AO_0, y_AO_0, z_AO_0 from the unit's dimension drawing

Rotate x/y by current azimuth:

x_AO = x_AO_0 * cos(horiz) + y_AO_0 * sin(horiz) 
y_AO = -x_AO_0 * sin(horiz) + y_AO_0 * cos(horiz) 
z_AO = z_AO_0 # constant, does not rotate

x = x_AO + x_BA y = y_AO + y_BA z = z_AO + z_BA

The offset magnitude is ~9–14 mm. For the intersection monitor's zone-based occupancy detection at ceiling-mount distances, this error is negligible. **Defer implementation** — apply only if the pipeline is later extended to centimeter-accurate mapping or SLAM.

## Angle Correction File

Per-unit CSV file from Hesai. Each row corresponds to one channel and provides:
- Accurate vertical angle (elevation)
- Accurate horizontal angle offset

These replace the designed/nominal values in the manual's Appendix A. The correction file is the ground truth for angle computation.

## Confidence Field — Contamination Detection

Bits [7:6] of the Confidence byte report cover lens contamination level (0–3). This is directly useful for the intersection monitor's self-diagnostic health reporting: if contamination level rises across channels, flag for cleaning. This is the "self-announcing" contamination signal referenced in the design documents.

## Frame Boundary Detection

The JT128 manual does not specify an explicit frame-boundary marker in the packet stream. Frame boundaries must be inferred from azimuth rollover: when the azimuth value wraps from a high value back toward 0°, a new frame has started. This is the standard approach for spinning LiDARs and is how hydra4 handles it for the OT-128.
