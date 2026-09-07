"""Payload error-correction codecs."""

from __future__ import annotations

from ..schemas import EccDecodeStats


def _validate_bits(bits: list[int]) -> None:
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Bit sequences may contain only 0 and 1")


def hamming74_encode(bits: list[int]) -> list[int]:
    _validate_bits(bits)
    if len(bits) % 4:
        raise ValueError("Hamming(7,4) input length must be divisible by 4")
    encoded: list[int] = []
    for offset in range(0, len(bits), 4):
        d1, d2, d3, d4 = bits[offset : offset + 4]
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4
        encoded.extend([p1, p2, d1, p4, d2, d3, d4])
    return encoded


def hamming74_decode(bits: list[int]) -> tuple[list[int], EccDecodeStats]:
    _validate_bits(bits)
    block_count = len(bits) // 7
    trailing = len(bits) % 7
    decoded: list[int] = []
    corrected = 0
    for offset in range(0, block_count * 7, 7):
        block = list(bits[offset : offset + 7])
        s1 = block[0] ^ block[2] ^ block[4] ^ block[6]
        s2 = block[1] ^ block[2] ^ block[5] ^ block[6]
        s4 = block[3] ^ block[4] ^ block[5] ^ block[6]
        syndrome = s1 + 2 * s2 + 4 * s4
        if syndrome:
            block[syndrome - 1] ^= 1
            corrected += 1
        decoded.extend([block[2], block[4], block[5], block[6]])
    return decoded, EccDecodeStats(block_count, corrected, trailing)


def ecc_encode(bits: list[int], scheme: str) -> list[int]:
    if scheme == "none":
        _validate_bits(bits)
        return list(bits)
    if scheme == "hamming74":
        return hamming74_encode(bits)
    raise ValueError(f"Unsupported ECC scheme: {scheme}")


def ecc_decode(bits: list[int], scheme: str) -> tuple[list[int], EccDecodeStats]:
    if scheme == "none":
        _validate_bits(bits)
        return list(bits), EccDecodeStats()
    if scheme == "hamming74":
        return hamming74_decode(bits)
    raise ValueError(f"Unsupported ECC scheme: {scheme}")
