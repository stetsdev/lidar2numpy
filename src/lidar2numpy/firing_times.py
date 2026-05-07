"""Per-channel firing time offset table for the Hesai JT128.

Values are from Appendix B.4 of the JT128 User Manual (also reproduced in
docs/jt128-packet-format.md). The JT128 uses an interleaved, non-uniform
firing sequence — channel index does NOT equal firing order.

These offsets are used to compute per-point timestamps: each point's timestamp
is the block start time (derived from the tail) plus the channel's firing delay.

For the intersection-monitor ceiling-mount application, the firing-time
*angular* correction (the extra horizontal offset caused by motor rotation
during the ~106 µs firing sequence) is negligible and is deferred. Only the
temporal correction (timestamp accuracy) is applied in v0.1.
"""

from __future__ import annotations

import numpy as np

# fmt: off
FIRING_TIME_OFFSETS_US: dict[int, float] = {
      1: 95.18,   2: 23.24,   3: 98.22,   4: 20.20,
      5: 101.26,  6: 17.16,   7: 104.30,  8: 14.12,
      9: 77.28,  10: 92.14,  11: 74.24,  12: 89.10,
     13: 71.20,  14: 86.06,  15: 68.16,  16: 83.02,
     17: 50.26,  18: 11.08,  19: 47.22,  20:  8.04,
     21: 44.18,  22:  5.00,  23: 41.14,  24:  1.96,
     25: 65.12,  26: 105.82, 27: 62.08,  28: 102.78,
     29: 59.04,  30: 99.74,  31: 56.00,  32: 96.70,
     33: 38.10,  34: 24.76,  35: 35.06,  36: 21.72,
     37: 32.02,  38: 18.68,  39: 28.98,  40: 15.64,
     41: 78.80,  42: 93.66,  43: 75.76,  44: 90.62,
     45: 72.72,  46: 87.58,  47: 69.68,  48: 84.54,
     49: 51.78,  50: 12.60,  51: 48.74,  52:  9.56,
     53: 45.70,  54:  6.52,  55: 42.66,  56:  3.48,
     57: 66.64,  58: 103.54, 59: 63.60,  60: 100.50,
     61: 60.56,  62: 97.46,  63: 57.52,  64: 94.42,
     65: 39.62,  66: 22.48,  67: 36.58,  68: 19.44,
     69: 33.54,  70: 16.40,  71: 30.50,  72: 13.36,
     73: 76.52,  74: 91.38,  75: 73.48,  76: 88.34,
     77: 70.44,  78: 85.30,  79: 67.40,  80: 82.26,
     81: 49.50,  82: 10.32,  83: 46.46,  84:  7.28,
     85: 43.42,  86:  4.24,  87: 40.38,  88:  1.20,
     89: 64.36,  90: 105.06, 91: 61.32,  92: 102.02,
     93: 58.28,  94: 98.98,  95: 55.24,  96: 95.94,
     97: 37.34,  98: 24.00,  99: 34.30, 100: 20.96,
    101: 31.26, 102: 17.92, 103: 28.22, 104: 14.88,
    105: 78.04, 106: 92.90, 107: 75.00, 108: 89.86,
    109: 71.96, 110: 86.82, 111: 68.92, 112: 83.78,
    113: 51.02, 114: 11.84, 115: 47.98, 116:  8.80,
    117: 44.94, 118:  5.76, 119: 41.90, 120:  2.72,
    121: 65.88, 122: 62.84, 123: 59.80, 124: 56.76,
    125: 38.86, 126: 35.82, 127: 32.78, 128: 29.74,
}
# fmt: on

# Pre-converted to seconds, indexed 0-based by ring (channel - 1).
# Avoids repeated per-packet multiplication in the decoder's hot path.
FIRING_OFFSETS_S: np.ndarray = np.array(
    [FIRING_TIME_OFFSETS_US[ch] * 1e-6 for ch in range(1, 129)],
    dtype=np.float64,
)
