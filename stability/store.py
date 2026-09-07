"""Compressed, profile-aware stability-map storage."""

from __future__ import annotations

from math import log
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as functional

from ..configuration.loader import AppConfig
from ..errors import StabilityMapError
from ..profiles import DimensionProfile
from ..schemas import StabilityMapInfo


class StabilityMapStore:
    def __init__(self, config: AppConfig):
        self.config = config
        self._cache: dict[tuple[str, str], torch.Tensor] = {}

    def path_for(self, profile: DimensionProfile) -> Path:
        filename = self.config.stability.filename_template.format(profile=profile.key)
        return self.config.stability.output_dir / filename

    def save(
        self,
        scores: torch.Tensor | np.ndarray,
        profile: DimensionProfile,
        matched_samples: int,
        path: str | Path | None = None,
    ) -> StabilityMapInfo:
        array = (
            scores.detach().float().cpu().numpy()
            if isinstance(scores, torch.Tensor)
            else np.asarray(scores, dtype=np.float32)
        )
        if tuple(array.shape) != profile.latent_shape:
            raise StabilityMapError(
                f"Stability shape {tuple(array.shape)} does not match {profile.key} {profile.latent_shape}"
            )
        target = Path(path).resolve() if path else self.path_for(profile)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            scores=array.astype(np.float32),
            image_width=np.int32(profile.width),
            image_height=np.int32(profile.height),
            latent_channels=np.int32(profile.latent_channels),
            latent_height=np.int32(profile.latent_shape[1]),
            latent_width=np.int32(profile.latent_shape[2]),
            matched_samples=np.int32(matched_samples),
            algorithm_version=np.array(self.config.stability.algorithm_version),
        )
        return StabilityMapInfo(
            profile.key,
            target,
            profile.latent_shape,
            matched_samples,
            self.config.stability.algorithm_version,
        )

    def _read_archive(
        self, path: Path
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int]]:
        try:
            with np.load(path, allow_pickle=False) as archive:
                required = {
                    "scores", "image_width", "image_height", "latent_channels",
                    "latent_height", "latent_width", "matched_samples", "algorithm_version",
                }
                missing = required - set(archive.files)
                if missing:
                    raise StabilityMapError(f"Stability archive is missing: {', '.join(sorted(missing))}")
                scores = np.asarray(archive["scores"], dtype=np.float32)
                image_size = (int(archive["image_width"]), int(archive["image_height"]))
                latent_shape = (
                    int(archive["latent_channels"]),
                    int(archive["latent_height"]),
                    int(archive["latent_width"]),
                )
        except StabilityMapError:
            raise
        except Exception as exc:
            raise StabilityMapError(f"Cannot load stability map {path}: {exc}") from exc
        if tuple(scores.shape) != latent_shape:
            raise StabilityMapError(
                f"Stability score shape does not match metadata in {path}"
            )
        return scores, image_size, latent_shape

    def _load_exact(self, profile: DimensionProfile) -> torch.Tensor:
        path = self.path_for(profile)
        if not path.is_file():
            raise StabilityMapError(
                f"Stability map for {profile.key} not found: {path}. Generate it with the stability command."
            )
        scores, image_size, latent_shape = self._read_archive(path)
        if image_size != profile.image_size or latent_shape != profile.latent_shape:
            raise StabilityMapError(
                f"Stability metadata/shape does not match profile {profile.key}"
            )
        return torch.from_numpy(scores.copy()).float()

    def _available_archives(
        self,
    ) -> list[tuple[Path, np.ndarray, tuple[int, int], tuple[int, int, int]]]:
        pattern = self.config.stability.filename_template.replace("{profile}", "*")
        archives = []
        for path in sorted(self.config.stability.output_dir.glob(pattern)):
            if not path.is_file():
                continue
            try:
                scores, image_size, latent_shape = self._read_archive(path)
            except StabilityMapError:
                continue
            if latent_shape[0] == self.config.model.latent.channels:
                archives.append((path, scores, image_size, latent_shape))
        return archives

    def _interpolate(self, profile: DimensionProfile) -> torch.Tensor:
        exact_path = self.path_for(profile)
        if exact_path.is_file():
            return self._load_exact(profile)
        archives = self._available_archives()
        if not archives:
            raise StabilityMapError(
                f"No stability maps are available for interpolation under {self.config.stability.output_dir}"
            )
        cfg = self.config.stability.map_matching.interpolation
        target_ratio = profile.width / profile.height
        target_area = profile.width * profile.height

        def distance(item) -> tuple[float, str]:
            width, height = item[2]
            ratio_distance = abs(log((width / height) / target_ratio))
            area_distance = abs(log((width * height) / target_area))
            weighted = (
                cfg.aspect_ratio_weight * ratio_distance
                + cfg.area_weight * area_distance
            )
            return weighted, item[0].name

        source_path, scores, source_size, _ = min(archives, key=distance)
        source = torch.from_numpy(scores.copy()).float().unsqueeze(0)
        arguments = {
            "size": profile.latent_shape[1:],
            "mode": cfg.mode,
        }
        if cfg.mode in {"bilinear", "bicubic"}:
            arguments["align_corners"] = cfg.align_corners
        derived = functional.interpolate(source, **arguments).squeeze(0)
        if cfg.renormalize:
            flattened = derived.flatten(1)
            minimum = flattened.min(dim=1).values[:, None, None]
            maximum = flattened.max(dim=1).values[:, None, None]
            span = maximum - minimum
            derived = torch.where(
                span > 0,
                (derived - minimum) / span.clamp_min(1.0e-12),
                torch.zeros_like(derived),
            )
        derived = derived.clamp(0, 1).pow(cfg.score_gamma)
        warnings.warn(
            f"Stability map for {profile.key} was derived from {source_size[0]}x{source_size[1]} "
            f"({source_path.name}) using normalized interpolation.",
            UserWarning,
            stacklevel=3,
        )
        return derived

    def _analytic_band(self, profile: DimensionProfile) -> torch.Tensor:
        cfg = self.config.stability.map_matching.analytic_band
        channels, height, width = profile.latent_shape
        fy = torch.linspace(0.0, 1.0, height).unsqueeze(1)
        fx = torch.linspace(0.0, 1.0, width).unsqueeze(0)
        radius = torch.sqrt(fy.square() + fx.square()) / (2.0 ** 0.5)
        band = torch.exp(-0.5 * ((radius - cfg.center) / cfg.sigma).square())
        valid = (radius >= cfg.low_cutoff) & (radius <= cfg.high_cutoff)
        balance = (
            1.0 - cfg.axis_imbalance_penalty * torch.abs(fx - fy)
        ).clamp(0, 1)
        score = (band * valid * balance).clamp(0, 1)
        score[0, 0] = 0.0
        priors = torch.tensor(cfg.channel_priors, dtype=torch.float32).view(channels, 1, 1)
        return (score.unsqueeze(0) * priors).clamp(0, 1).float()

    def load(
        self, profile: DimensionProfile, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        strategy = self.config.stability.map_matching.strategy
        cache_key = (strategy, profile.key)
        if cache_key not in self._cache:
            if strategy == "strict":
                scores = self._load_exact(profile)
            elif strategy == "normalized_interpolation":
                scores = self._interpolate(profile)
            elif strategy == "analytic_band":
                scores = self._analytic_band(profile)
            else:
                raise StabilityMapError(f"Unsupported stability-map strategy: {strategy}")
            if tuple(scores.shape) != profile.latent_shape:
                raise StabilityMapError(
                    f"Derived stability shape {tuple(scores.shape)} does not match {profile.latent_shape}"
                )
            self._cache[cache_key] = scores.cpu()
        return self._cache[cache_key].to(device)

    def convert_legacy(
        self,
        legacy_path: str | Path,
        profile_key: str,
        output_path: str | Path | None = None,
    ) -> StabilityMapInfo:
        profile = self.config.preprocessing.registry.by_key(profile_key)
        source = Path(legacy_path).resolve()
        try:
            scores = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            scores = torch.load(source, map_location="cpu")
        if not isinstance(scores, torch.Tensor):
            raise StabilityMapError("Legacy stability file must contain one torch.Tensor")
        return self.save(scores.squeeze(0) if scores.ndim == 4 else scores, profile, 0, output_path)
