"""Stego-image/latent extraction and payload recovery."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from ..configuration.loader import AppConfig
from ..preprocessing.images import load_validated_image, safe_load_latent
from ..preprocessing.payload import PayloadCodec
from ..schemas import ExtractResult
from ..stability.store import StabilityMapStore
from ..steganography.alpha import calculate_position_alphas
from ..steganography.codec import extract_grouped
from ..steganography.positions import select_dct_positions
from ..steganography.profile_config import resolve_stego_profile_config


class ExtractionService:
    def __init__(self, config: AppConfig, model_provider, store: StabilityMapStore | None = None):
        self.config = config
        self.model_provider = model_provider
        self.store = store or StabilityMapStore(config)
        self.payload_codec = PayloadCodec(config.preprocessing, config.extraction.text_errors)

    def extract(
        self,
        *,
        image: str | Path | Image.Image | None = None,
        latent: torch.Tensor | None = None,
        latent_path: str | Path | None = None,
    ) -> ExtractResult:
        provided = sum(value is not None for value in (image, latent, latent_path))
        if provided != 1:
            raise ValueError("Provide exactly one of image, latent, or latent_path")
        registry = self.config.preprocessing.registry
        if image is not None:
            stego_image, profile, _ = load_validated_image(
                image, registry, self.config.preprocessing.image_adjustment
            )
            encoded = self.model_provider.modified_encoder.encode_image(stego_image)
            latent_tensor = encoded.squeeze(0)
        elif latent_path is not None:
            latent_tensor, profile = safe_load_latent(latent_path, registry)
        else:
            profile = registry.from_latent_shape(latent.shape)
            latent_tensor = latent.squeeze(0) if latent.ndim == 4 else latent
        latent_tensor = latent_tensor.float()
        stability = self.store.load(profile, device=latent_tensor.device)
        profile_cfg = resolve_stego_profile_config(self.config, profile)
        positions = select_dct_positions(
            stability,
            block_size=self.config.stability.block_size,
            positions_per_block=profile_cfg.positions_per_block,
            threshold=profile_cfg.stability_threshold,
        )
        alphas = calculate_position_alphas(
            latent_tensor,
            positions,
            profile_cfg.base_alpha,
            self.config.steganography,
        )
        raw_bits, _, _ = extract_grouped(
            latent_tensor,
            self.config.steganography.groups,
            positions,
            alphas,
            self.config.extraction.tie_bit,
        )
        decoded = self.payload_codec.decode(raw_bits)
        return ExtractResult(
            profile=profile,
            status=decoded.status,
            raw_bits=raw_bits,
            framed_ecc_bits=decoded.ecc_bits,
            corrected_payload_bits=decoded.payload_bits,
            decoded_bytes=decoded.decoded_bytes,
            decoded_text=decoded.decoded_text,
            ecc_stats=decoded.ecc_stats,
            error=decoded.error,
            payload_metadata=decoded.metadata,
        )
