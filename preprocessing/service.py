"""Compose image/latent preprocessing with payload encoding."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from PIL import Image

from ..configuration.loader import AppConfig
from ..schemas import EncodedPayload, PreparedInput
from .images import load_validated_image, safe_load_latent
from .payload import PayloadCodec


class PreprocessingService:
    def __init__(self, config: AppConfig, model_provider):
        self.config = config
        self.model_provider = model_provider
        self.payload_codec = PayloadCodec(config.preprocessing, config.extraction.text_errors)

    def _payload(
        self,
        max_bits: int,
        text: Optional[str],
        data: Optional[bytes],
        bits: Optional[Iterable[int]],
    ) -> EncodedPayload:
        provided = sum(value is not None for value in (text, data, bits))
        if provided != 1:
            raise ValueError("Provide exactly one of text, data, or bits")
        if text is not None:
            return self.payload_codec.encode_text(text, max_bits)
        if data is not None:
            return self.payload_codec.encode_bytes(data, max_bits)
        return self.payload_codec.encode_bits(bits or [], max_bits)

    def prepare(
        self,
        *,
        image: str | Path | Image.Image | None = None,
        latent_path: str | Path | None = None,
        text: str | None = None,
        data: bytes | None = None,
        bits: Iterable[int] | None = None,
    ) -> PreparedInput:
        if (image is None) == (latent_path is None):
            raise ValueError("Provide exactly one of image or latent_path")
        registry = self.config.preprocessing.registry
        if image is not None:
            cover, profile, source_path = load_validated_image(
                image, registry, self.config.preprocessing.image_adjustment
            )
            payload = self._payload(profile.max_bits, text, data, bits)
            latent = self.model_provider.original_vae.encode_image(cover).squeeze(0)
            return PreparedInput(profile, latent, payload, cover, source_path)
        latent, profile = safe_load_latent(latent_path, registry)
        payload = self._payload(profile.max_bits, text, data, bits)
        return PreparedInput(profile, latent, payload, None, Path(latent_path).resolve())
