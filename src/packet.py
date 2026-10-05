"""Task 6.1 — binary packet: 4-byte header + d signed INT8 payload bytes.

Header (little-endian): version uint8 | method_id uint8 | config_id uint16
Total size = d + 4 bytes.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

HEADER = struct.Struct("<BBH")
VERSION = 1
METHOD_IDS = {"AE-MSE": 1, "Task-AE": 2, "DCT": 3, "Low-Mel": 4, "INT8": 5}
# config_id encodes d; fold/seed of the model are agreed out-of-band between node and gateway.


@dataclass
class Packet:
    version: int
    method_id: int
    config_id: int
    payload: np.ndarray  # int8 (d,)


def encode(q: np.ndarray, method: str, d: int) -> bytes:
    q = np.asarray(q)
    if q.dtype != np.int8 or q.shape != (d,):
        raise ValueError(f"payload must be int8 of shape ({d},), got {q.dtype} {q.shape}")
    return HEADER.pack(VERSION, METHOD_IDS[method], d) + q.tobytes()


def decode(buf: bytes) -> Packet:
    version, method_id, config_id = HEADER.unpack_from(buf, 0)
    payload = np.frombuffer(buf, dtype=np.int8, offset=HEADER.size)
    if version != VERSION:
        raise ValueError(f"unsupported packet version {version}")
    if len(payload) != config_id:
        raise ValueError(f"payload length {len(payload)} != d {config_id}")
    return Packet(version, method_id, config_id, payload.copy())


def packet_bytes(d: int) -> int:
    return d + HEADER.size
