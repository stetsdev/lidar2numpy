# ---------------------------------------------------------------------------
# Minimal pcap reader (same approach as test_integration.py)
# ---------------------------------------------------------------------------

import struct
from pathlib import Path

_PCAP_MAGIC_US = 0xA1B2C3D4
_PCAP_MAGIC_NS = 0xA1B23C4D


def read_pcap_payloads(pcap_path: Path, udp_payload_size: int = 1100):
    """Yield UDP payloads from a pcap file, stripping Ethernet+IP+UDP headers."""
    with open(pcap_path, "rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError("Truncated pcap global header")

        magic = struct.unpack("<I", global_header[:4])[0]
        if magic == _PCAP_MAGIC_US:
            endian = "<"
        elif magic == _PCAP_MAGIC_NS:
            endian = "<"
        elif struct.unpack(">I", global_header[:4])[0] in (_PCAP_MAGIC_US, _PCAP_MAGIC_NS):
            endian = ">"
        else:
            raise ValueError(f"Not a pcap file: bad magic {global_header[:4].hex()}")

        while True:
            rec_header = f.read(16)
            if len(rec_header) < 16:
                break
            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", rec_header)
            data = f.read(incl_len)
            if len(data) < incl_len:
                break

            # Strip: 14 Ethernet + 20 IP + 8 UDP = 42 bytes
            if len(data) >= 42 + udp_payload_size:
                payload = data[42 : 42 + udp_payload_size]
                if len(payload) == udp_payload_size:
                    yield payload

