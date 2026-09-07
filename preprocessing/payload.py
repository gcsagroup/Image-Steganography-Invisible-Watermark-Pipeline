"""Fixed-header text/byte/bit payload wire protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import zlib

from ..configuration.loader import PreprocessingConfig
from ..errors import PayloadCapacityError
from ..schemas import EccDecodeStats, EncodedPayload, PayloadMetadata
from .ecc import ecc_decode, ecc_encode


def bytes_to_bits(data: bytes) -> list[int]:
    return [(byte >> shift) & 1 for byte in data for shift in range(7, -1, -1)]


def bits_to_bytes(bits: list[int]) -> bytes:
    if any(bit not in (0, 1) for bit in bits):
        raise ValueError("Bit sequences may contain only 0 and 1")
    if len(bits) % 8:
        raise ValueError(f"Bit length {len(bits)} is not byte-aligned")
    return bytes(
        sum(bits[offset + index] << (7 - index) for index in range(8))
        for offset in range(0, len(bits), 8)
    )


_LOGICAL_HEADER_BIT_LENGTH = 67
_MAX_FRAME_BIT_LENGTH = (1 << 24) - 1
_ENCODING_IDS = {"utf-8": 0b00, "gbk": 0b01, "ascii": 0b10, "utf-16": 0b11}
_ENCODING_NAMES = {value: key for key, value in _ENCODING_IDS.items()}
_ECC_IDS = {"none": 0, "hamming74": 1}
_ECC_NAMES = {value: key for key, value in _ECC_IDS.items()}


def _int_to_bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= 1 << width:
        raise ValueError(f"Value {value} does not fit in {width} bits")
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def _bits_to_int(bits: list[int]) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value


def _crc8(bits: list[int]) -> int:
    """CRC-8/ATM over a bit sequence, polynomial 0x07."""
    crc = 0
    for bit in bits:
        feedback = ((crc >> 7) & 1) ^ bit
        crc = (crc << 1) & 0xFF
        if feedback:
            crc ^= 0x07
    return crc


@dataclass(frozen=True)
class DecodedPayload:
    status: str
    ecc_bits: list[int]
    payload_bits: list[int]
    decoded_bytes: bytes
    decoded_text: Optional[str]
    ecc_stats: EccDecodeStats
    error: Optional[str]
    metadata: Optional[PayloadMetadata] = None


class PayloadCodec:
    def __init__(self, config: PreprocessingConfig, text_errors: str = "strict"):
        self.config = config
        self.text_errors = text_errors

    @property
    def logical_header_bit_length(self) -> int:
        return _LOGICAL_HEADER_BIT_LENGTH

    @property
    def header_bit_length(self) -> int:
        return self.logical_header_bit_length * self.config.frame.header_repetitions

    @property
    def ecc_scheme(self) -> str:
        return self.config.ecc.effective_scheme

    def _encode_header(
        self,
        original_bit_length: int,
        payload_crc32: int,
    ) -> list[int]:
        if original_bit_length > _MAX_FRAME_BIT_LENGTH:
            raise PayloadCapacityError("Frame payload length exceeds the 24-bit header field")
        core = (
            _int_to_bits(_ENCODING_IDS[self.config.text_encoding], 2)
            + [_ECC_IDS[self.ecc_scheme]]
            + _int_to_bits(original_bit_length, 24)
            + _int_to_bits(payload_crc32, 32)
        )
        logical_bits = core + _int_to_bits(_crc8(core), 8)
        return logical_bits * self.config.frame.header_repetitions

    def _decode_header(self, raw: list[int]) -> tuple[Optional[dict], Optional[str], Optional[str]]:
        if len(raw) < self.header_bit_length:
            return None, "incomplete_header", (
                f"Frame requires {self.header_bit_length} header bits but only {len(raw)} were extracted"
            )
        length = self.logical_header_bit_length
        copies = [raw[index * length : (index + 1) * length] for index in range(self.config.frame.header_repetitions)]
        threshold = self.config.frame.header_repetitions // 2 + 1
        logical_bits = [
            1 if sum(copy[index] for copy in copies) >= threshold else 0
            for index in range(length)
        ]
        core = logical_bits[:59]
        encoding_id = _bits_to_int(core[:2])
        ecc_id = core[2]
        original_length = _bits_to_int(core[3:27])
        payload_crc32 = _bits_to_int(core[27:59])
        stored_crc = _bits_to_int(logical_bits[59:67])
        calculated_crc = _crc8(core)
        if stored_crc != calculated_crc:
            return None, "header_crc_error", "Header CRC8 validation failed"
        if encoding_id not in _ENCODING_NAMES or ecc_id not in _ECC_NAMES:
            return None, "invalid_header_metadata", "Frame header contains an unknown encoding or ECC identifier"
        ecc_scheme = _ECC_NAMES[ecc_id]
        encoded_length = (
            original_length if ecc_scheme == "none" else (original_length // 4) * 7
        )
        return {
            "text_encoding": _ENCODING_NAMES[encoding_id],
            "ecc_scheme": ecc_scheme,
            "original_bit_length": original_length,
            "encoded_bit_length": encoded_length,
            "payload_crc32": payload_crc32,
        }, None, None

    def encode_text(self, text: str, max_bits: int) -> EncodedPayload:
        data = text.encode(self.config.text_encoding, errors="strict")
        return self.encode_bytes(data, max_bits, source_type="text", original_text=text)

    def encode_bytes(
        self,
        data: bytes,
        max_bits: int,
        source_type: str = "bytes",
        original_text: Optional[str] = None,
    ) -> EncodedPayload:
        return self.encode_bits(
            bytes_to_bits(data),
            max_bits,
            source_type=source_type,
            original_text=original_text,
            original_bytes=data,
        )

    def encode_bits(
        self,
        bits: Iterable[int],
        max_bits: int,
        source_type: str = "bits",
        original_text: Optional[str] = None,
        original_bytes: Optional[bytes] = None,
    ) -> EncodedPayload:
        original = [int(bit) for bit in bits]
        if any(bit not in (0, 1) for bit in original):
            raise ValueError("Secret bits may contain only 0 and 1")
        if len(original) % 8:
            raise ValueError("Secret bit length must be byte-aligned")
        encoded = ecc_encode(original, self.ecc_scheme)
        payload_bytes = original_bytes if original_bytes is not None else bits_to_bytes(original)
        payload_crc32 = zlib.crc32(payload_bytes) & 0xFFFFFFFF
        header = self._encode_header(len(original), payload_crc32)
        final = header + encoded
        if len(final) > max_bits:
            raise PayloadCapacityError(
                f"Final stream requires {len(final)} bits but profile limit is {max_bits} bits"
            )
        metadata = PayloadMetadata(
            source_type=source_type,
            text_encoding=self.config.text_encoding,
            ecc_scheme=self.ecc_scheme,
            original_bit_length=len(original),
            encoded_bit_length=len(encoded),
            header_bit_length=len(header),
            payload_crc32=payload_crc32,
            final_bit_length=len(final),
        )
        return EncodedPayload(
            original_bits=original,
            ecc_bits=encoded,
            final_bits=final,
            metadata=metadata,
            original_text=original_text,
            original_bytes=original_bytes,
        )

    def decode(self, raw_bits: Iterable[int]) -> DecodedPayload:
        raw = [int(bit) for bit in raw_bits]
        header, header_status, header_error = self._decode_header(raw)
        if header is None:
            return DecodedPayload(
                header_status or "invalid_header", [], [], b"", None,
                EccDecodeStats(), header_error, None,
            )
        encoded_length = int(header["encoded_bit_length"])
        original_length = int(header["original_bit_length"])
        frame_length = self.header_bit_length + encoded_length
        metadata = PayloadMetadata(
            source_type="unspecified",
            text_encoding=str(header["text_encoding"]),
            ecc_scheme=str(header["ecc_scheme"]),
            original_bit_length=original_length,
            encoded_bit_length=encoded_length,
            header_bit_length=self.header_bit_length,
            payload_crc32=int(header["payload_crc32"]),
            final_bit_length=frame_length,
        )
        if encoded_length < 0 or original_length < 0 or frame_length > len(raw):
            return DecodedPayload(
                "incomplete_payload", [], [], b"", None, EccDecodeStats(),
                f"Frame declares {frame_length} bits but only {len(raw)} were extracted", metadata,
            )
        ecc_bits = raw[self.header_bit_length : frame_length]
        payload_bits, stats = ecc_decode(ecc_bits, str(header["ecc_scheme"]))
        if stats.trailing_bits:
            return DecodedPayload(
                "ecc_trailing_bits", ecc_bits, payload_bits, b"", None, stats,
                f"ECC stream has {stats.trailing_bits} trailing bits", metadata,
            )
        if len(payload_bits) != original_length:
            return DecodedPayload(
                "payload_length_error", ecc_bits, payload_bits, b"", None, stats,
                f"Header declares {original_length} payload bits but ECC produced {len(payload_bits)}",
                metadata,
            )
        try:
            decoded_bytes = bits_to_bytes(payload_bits)
        except ValueError as exc:
            return DecodedPayload(
                "byte_alignment_error", ecc_bits, payload_bits, b"", None, stats, str(exc), metadata
            )
        calculated_payload_crc = zlib.crc32(decoded_bytes) & 0xFFFFFFFF
        if calculated_payload_crc != int(header["payload_crc32"]):
            return DecodedPayload(
                "payload_crc_error", ecc_bits, payload_bits, decoded_bytes, None, stats,
                "Payload CRC32 validation failed", metadata,
            )
        try:
            decoded_text = decoded_bytes.decode(
                str(header["text_encoding"]), errors=self.text_errors
            )
        except UnicodeDecodeError:
            return DecodedPayload(
                "ok", ecc_bits, payload_bits, decoded_bytes, None, stats, None, metadata
            )
        return DecodedPayload("ok", ecc_bits, payload_bits, decoded_bytes, decoded_text, stats, None, metadata)

    def maximum_payload_bytes(self, max_bits: int, utilization: float = 1.0) -> int:
        target = min(max_bits, int(max_bits * utilization))
        available = target - self.header_bit_length
        if available <= 0:
            return 0
        if self.ecc_scheme == "hamming74":
            payload_bits = (available // 7) * 4
        else:
            payload_bits = available
        return payload_bits // 8
