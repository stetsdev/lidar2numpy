# Cython Decoder Local Benchmark

Date: 2026-07-14

This is development evidence for the Cython spherical decoder candidate. It
does not substitute for the ARM64 Debian `lidar-decode` CPU acceptance test.

## Workload

- PCAP: `jt128-lidar-20260711T114109.pcap`
- Calibration: `jt128-angle-corrections.csv`
- Capture size: 403,860 JT128 payloads
- Measured slice: first 20,000 payloads, with 1,000 warm-up payloads
- Return modes in slice: `0x37` Strongest only
- Repetitions: 3 per backend

## Command

```bash
uv run python scripts/benchmark_lidar_decode.py jt128-lidar-20260711T114109.pcap \
  --calibration jt128-angle-corrections.csv --backend <python|cython> \
  --max-packets 20000 --warmup-packets 1000 --repeat 3 --json-out /tmp/result.json
```

## Results

| Backend | Mean thread CPU time | Mean throughput | Median feed latency |
| --- | ---: | ---: | ---: |
| Python | 0.3561 s | 53,163 packets/s | 17.07 us |
| Cython | 0.0891 s | 212,817 packets/s | 3.92 us |

The Cython result reduces local per-run decoder thread CPU time by about 75%
on this single-return capture. It is a macOS ARM development result; validate
the production CPU reduction on the CM5 ARM64 Debian target before release.
