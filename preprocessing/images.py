"""Image and direct-latent input validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import torch
from PIL import Image, ImageOps

from ..profiles import DimensionProfile, ProfileRegistry


_RESAMPLING = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


def nearest_multiple(value: int, multiple: int) -> int:
    """Round to the nearest positive multiple; exact ties round upward."""
    if multiple <= 0:
        raise ValueError("Image-size multiple must be positive")
    return max(multiple, ((int(value) + multiple // 2) // multiple) * multiple)


def normalize_image_size(
    image: Image.Image,
    adjustment: Any,
    *,
    source_name: str = "input image",
) -> Image.Image:
    """Normalize image dimensions using configured stretch or aspect-preserving crop."""
    rgb = image.convert("RGB")
    target = (
        nearest_multiple(rgb.width, adjustment.multiple),
        nearest_multiple(rgb.height, adjustment.multiple),
    )
    if target == rgb.size:
        return rgb.copy()
    resample = _RESAMPLING[adjustment.resample]
    if adjustment.method == "stretch":
        normalized = rgb.resize(target, resample)
    elif adjustment.method == "scale_crop":
        normalized = ImageOps.fit(rgb, target, method=resample, centering=(0.5, 0.5))
    else:
        raise ValueError(f"Unsupported image adjustment method: {adjustment.method}")
    warnings.warn(
        f"{source_name}: resolution adjusted from {rgb.width}x{rgb.height} to "
        f"{target[0]}x{target[1]} using {adjustment.method}; dimensions must be "
        f"multiples of {adjustment.multiple}.",
        UserWarning,
        stacklevel=2,
    )
    return normalized


def load_validated_image(
    image: str | Path | Image.Image,
    registry: ProfileRegistry,
    adjustment: Any | None = None,
) -> tuple[Image.Image, DimensionProfile, Path | None]:
    source_path: Path | None = None
    if isinstance(image, Image.Image):
        loaded = image.convert("RGB").copy()
        source_name = "in-memory image"
    else:
        source_path = Path(image).resolve()
        with Image.open(source_path) as opened:
            loaded = opened.convert("RGB").copy()
        source_name = source_path.name
    rgb = (
        normalize_image_size(loaded, adjustment, source_name=source_name)
        if adjustment is not None
        else loaded.convert("RGB")
    )
    profile = registry.from_image_size(rgb.size)
    return rgb, profile, source_path


def safe_load_latent(path: str | Path, registry: ProfileRegistry) -> tuple[torch.Tensor, DimensionProfile]:
    latent_path = Path(path).resolve()
    try:
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
    except TypeError:
        latent = torch.load(latent_path, map_location="cpu")
    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"Latent file must contain one torch.Tensor: {latent_path}")
    profile = registry.from_latent_shape(latent.shape)
    if latent.ndim == 4:
        latent = latent.squeeze(0)
    return latent.float(), profile
