"""Tests for the raw-packet decoder benchmark CLI."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from _packet_builder import build_packet

from lidar2numpy import _decoder_backend as backend_module
from lidar2numpy import default_calibration


def _load_benchmark_module() -> object:
    script_path = Path(__file__).parents[1] / "scripts" / "benchmark_lidar_decode.py"
    spec = importlib.util.spec_from_file_location("benchmark_lidar_decode", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _payloads() -> list[bytes]:
    channels = {0: (250, 100, 0), 127: (500, 120, 0xC5)}
    return [
        build_packet(block1_az=35_000, block1_channels=channels),
        build_packet(block1_az=100, block1_channels=channels),
        build_packet(block1_az=1_000, block1_channels=channels),
        build_packet(block1_az=35_000, block1_channels=channels),
        build_packet(block1_az=100, block1_channels=channels),
    ]


def test_run_once_reports_raw_decoder_metrics() -> None:
    module = _load_benchmark_module()

    result = module.run_once(
        _payloads(),
        calibration=default_calibration(),
        backend="auto",
        warmup_packets=0,
    )

    assert result["backend"] in {"python", "cython"}
    assert result["packets"] == 5
    assert result["frames"] == 1
    assert result["points"] == 6
    assert result["wall_time_s"] >= 0.0
    assert result["process_cpu_time_s"] >= 0.0
    assert result["thread_cpu_time_s"] >= 0.0
    assert result["packets_per_second"] > 0.0
    assert set(result["feed_latency_us"]) == {"p50", "p95", "p99"}


def test_explicit_unavailable_backend_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_benchmark_module()
    monkeypatch.setattr(backend_module, "_load_compiled_feed", lambda: None)

    with pytest.raises(RuntimeError, match="cython backend is unavailable"):
        module.run_once(
            _payloads(),
            calibration=default_calibration(),
            backend="cython",
            warmup_packets=0,
        )


def test_main_writes_machine_readable_repeated_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_benchmark_module()
    output_path = tmp_path / "benchmark.json"
    monkeypatch.setattr(module, "read_pcap_payloads", lambda _path: iter(_payloads()))

    assert (
        module.main(
            [
                "capture.pcap",
                "--repeat",
                "2",
                "--json-out",
                str(output_path),
            ]
        )
        == 0
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["schema_version"] == 1
    assert result["backend"] in {"python", "cython"}
    assert result["packets_per_run"] == 5
    assert len(result["runs"]) == 2
    assert result["environment"]["python_version"]
    assert result["environment"]["platform"]


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        ([], "pcap is required"),
        (["capture.pcap", "--repeat", "0"], "--repeat must be greater than 0"),
        (["capture.pcap", "--warmup-packets", "-1"], "--warmup-packets must be non-negative"),
    ],
)
def test_parse_args_rejects_invalid_values(
    argv: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_benchmark_module()

    with pytest.raises(SystemExit):
        module.parse_args(argv)

    assert message in capsys.readouterr().err
