"""Generate DCT stability scores from paired latent directories."""

from __future__ import annotations

from pathlib import Path

import torch

from ..configuration.loader import AppConfig
from ..errors import StabilityMapError
from ..preprocessing.images import safe_load_latent
from ..schemas import StabilityMapInfo
from ..steganography.dct import dct2d
from .store import StabilityMapStore


def _base_id(filename: str) -> str:
    return filename.split("_", 1)[0].rsplit(".", 1)[0]


def match_latent_pairs(
    original_dir: str | Path, reconstructed_dir: str | Path, extension: str
) -> list[tuple[Path, Path]]:
    left_dir, right_dir = Path(original_dir).resolve(), Path(reconstructed_dir).resolve()
    if not left_dir.is_dir() or not right_dir.is_dir():
        raise StabilityMapError("Both stability input paths must be directories")
    left = sorted(path for path in left_dir.iterdir() if path.is_file() and path.suffix.lower() == extension)
    right = sorted(path for path in right_dir.iterdir() if path.is_file() and path.suffix.lower() == extension)
    pairs: list[tuple[Path, Path]] = []
    for original in left:
        identifier = _base_id(original.name)
        match = next((candidate for candidate in right if candidate.name.startswith(identifier)), None)
        if match is not None:
            pairs.append((original, match))
    return pairs


class StabilityGenerator:
    def __init__(self, config: AppConfig, store: StabilityMapStore | None = None):
        self.config = config
        self.store = store or StabilityMapStore(config)

    def generate(
        self,
        original_dir: str | Path,
        reconstructed_dir: str | Path,
        *,
        max_samples: int | None = None,
        output_path: str | Path | None = None,
    ) -> StabilityMapInfo:
        cfg = self.config.stability
        pairs = match_latent_pairs(original_dir, reconstructed_dir, cfg.file_extension)
        limit = max_samples if max_samples is not None else cfg.max_samples
        if limit is not None:
            pairs = pairs[:limit]
        if not pairs:
            raise StabilityMapError("No matching latent pairs were found")
        registry = self.config.preprocessing.registry
        first, profile = safe_load_latent(pairs[0][0], registry)
        diff_sum = torch.zeros(profile.latent_shape, dtype=torch.float64)
        count = 0
        for original_path, reconstructed_path in pairs:
            original, original_profile = safe_load_latent(original_path, registry)
            reconstructed, reconstructed_profile = safe_load_latent(reconstructed_path, registry)
            if original_profile.key != profile.key or reconstructed_profile.key != profile.key:
                raise StabilityMapError("All paired latents must have one identical supported shape")
            for channel in range(profile.latent_channels):
                diff_sum[channel] += torch.abs(
                    dct2d(original[channel]) - dct2d(reconstructed[channel])
                ).double().cpu()
            count += 1
        average = diff_sum / max(count, 1)
        scores = 1.0 / (average + cfg.epsilon)
        for channel in range(profile.latent_channels):
            minimum = scores[channel].min()
            maximum = scores[channel].max()
            value_range = maximum - minimum
            if float(value_range) > 0:
                scores[channel] = (scores[channel] - minimum) / value_range
            else:
                scores[channel].zero_()
        return self.store.save(scores.float(), profile, count, output_path)
