"""High-level latent embedding and stego-PNG creation."""

from __future__ import annotations

from pathlib import Path

from ..configuration.loader import AppConfig
from ..schemas import EmbedResult, PreparedInput
from ..stability.store import StabilityMapStore
from ..steganography.alpha import calculate_position_alphas
from ..steganography.codec import embed_grouped
from ..steganography.positions import select_dct_positions
from ..steganography.positions import group_capacities
from ..steganography.profile_config import resolve_stego_profile_config


def _safe_output_stem(name: str) -> str:
    stem = Path(name).stem.strip()
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in stem)
    return cleaned or "stego"


class EmbeddingService:
    def __init__(self, config: AppConfig, model_provider, store: StabilityMapStore | None = None):
        self.config = config
        self.model_provider = model_provider
        self.store = store or StabilityMapStore(config)

    def selected_capacity(self, profile) -> int:
        stability = self.store.load(profile, device="cpu")
        profile_cfg = resolve_stego_profile_config(self.config, profile)
        positions = select_dct_positions(
            stability,
            block_size=self.config.stability.block_size,
            positions_per_block=profile_cfg.positions_per_block,
            threshold=profile_cfg.stability_threshold,
        )
        return sum(group_capacities(positions, self.config.steganography.groups))

    def embed(
        self,
        prepared: PreparedInput,
        *,
        output_name: str | None = None,
        output_dir: str | Path | None = None,
    ) -> EmbedResult:
        profile = prepared.profile
        latent = prepared.latent.squeeze(0) if prepared.latent.ndim == 4 else prepared.latent
        stability = self.store.load(profile, device=latent.device)
        profile_cfg = resolve_stego_profile_config(self.config, profile)
        positions = select_dct_positions(
            stability,
            block_size=self.config.stability.block_size,
            positions_per_block=profile_cfg.positions_per_block,
            threshold=profile_cfg.stability_threshold,
        )
        alphas = calculate_position_alphas(
            latent,
            positions,
            profile_cfg.base_alpha,
            self.config.steganography,
        )
        stego_latent, selected_capacity, _ = embed_grouped(
            latent,
            prepared.payload.final_bits,
            self.config.steganography.groups,
            positions,
            alphas,
        )
        stego_image = self.model_provider.original_vae.decode_latent(stego_latent)
        if isinstance(stego_image, list):
            stego_image = stego_image[0]
        target_dir = Path(output_dir).resolve() if output_dir else self.config.root / "Stego"
        target_dir.mkdir(parents=True, exist_ok=True)
        if output_name:
            stem = _safe_output_stem(output_name)
        elif prepared.source_path:
            stem = _safe_output_stem(prepared.source_path.stem + "_stego")
        else:
            stem = "stego"
        target = target_dir / f"{stem}.png"
        stego_image.convert("RGB").save(target, format="PNG")
        return EmbedResult(
            profile=profile,
            stego_path=target,
            stego_image=stego_image.convert("RGB"),
            stego_latent=stego_latent,
            embedded_bits=list(prepared.payload.final_bits),
            selected_capacity=selected_capacity,
            payload=prepared.payload,
        )
