"""Typed data contracts used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image


@dataclass(frozen=True)
class PayloadMetadata:
    source_type: str
    text_encoding: str
    ecc_scheme: str
    original_bit_length: int
    encoded_bit_length: int
    header_bit_length: int
    payload_crc32: int
    final_bit_length: int


@dataclass
class EncodedPayload:
    original_bits: list[int]
    ecc_bits: list[int]
    final_bits: list[int]
    metadata: PayloadMetadata
    original_text: Optional[str] = None
    original_bytes: Optional[bytes] = None


@dataclass
class PreparedInput:
    profile: Any
    latent: Any
    payload: EncodedPayload
    cover_image: Optional[Image.Image] = None
    source_path: Optional[Path] = None


@dataclass(frozen=True)
class StabilityMapInfo:
    profile_key: str
    path: Path
    shape: tuple[int, int, int]
    matched_samples: int
    algorithm_version: str


@dataclass
class EmbedResult:
    profile: Any
    stego_path: Path
    stego_image: Image.Image
    stego_latent: Any
    embedded_bits: list[int]
    selected_capacity: int
    payload: EncodedPayload


@dataclass(frozen=True)
class EccDecodeStats:
    blocks: int = 0
    corrected_blocks: int = 0
    trailing_bits: int = 0


@dataclass
class ExtractResult:
    profile: Any
    status: str
    raw_bits: list[int]
    framed_ecc_bits: list[int]
    corrected_payload_bits: list[int]
    decoded_bytes: bytes
    decoded_text: Optional[str]
    ecc_stats: EccDecodeStats
    error: Optional[str] = None
    payload_metadata: Optional[PayloadMetadata] = None


@dataclass
class AttackResult:
    attack_name: str
    quality: Optional[int]
    extraction: ExtractResult
    raw_bit_accuracy: Optional[float]
    post_ecc_bit_accuracy: Optional[float]
    text_match: Optional[bool]


@dataclass
class EvaluationResult:
    psnr: Optional[float]
    ssim: Optional[float]
    lpips: Optional[float]
    lossless: AttackResult
    jpeg_results: list[AttackResult]


@dataclass
class TestImageResult:
    filename: str
    profile: Optional[str] = None
    payload_bits: Optional[int] = None
    selected_capacity: Optional[int] = None
    stego_path: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
    error: Optional[str] = None


@dataclass
class TestRunResult:
    report_path: Path
    results: list[TestImageResult]
    successful: int
    failed: int


def to_plain_data(value: Any) -> Any:
    """Convert nested result objects to JSON/report-friendly primitives."""
    if hasattr(value, "key") and hasattr(value, "image_size"):
        return value.key
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Image.Image):
        return {"mode": value.mode, "size": list(value.size)}
    if hasattr(value, "detach"):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if is_dataclass(value):
        return {item.name: to_plain_data(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(k): to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain_data(v) for v in value]
    return value
