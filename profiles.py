"""Supported image and latent dimension profiles."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Mapping, Sequence

from .errors import UnsupportedImageSizeError, UnsupportedLatentShapeError


def calculate_max_bits(width: int, height: int, bits_per_pixel: float) -> int:
    """Return the final framed-stream limit for an image's actual pixel count."""
    return floor(int(width) * int(height) * float(bits_per_pixel))


@dataclass(frozen=True)
class DimensionProfile:
    key: str
    width: int
    height: int
    max_bits: int
    latent_channels: int
    downsample_factor: int

    @property
    def image_size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (
            self.latent_channels,
            self.height // self.downsample_factor,
            self.width // self.downsample_factor,
        )


class ProfileRegistry:
    def __init__(
        self,
        profiles: Mapping[str, DimensionProfile],
        *,
        capacity_bits_per_pixel: float | None = None,
        latent_channels: int | None = None,
        downsample_factor: int | None = None,
        allow_dynamic: bool = False,
    ):
        self._profiles = dict(profiles)
        self.capacity_bits_per_pixel = capacity_bits_per_pixel
        self.latent_channels = latent_channels
        self.downsample_factor = downsample_factor
        self.allow_dynamic = allow_dynamic

    @property
    def profiles(self) -> Mapping[str, DimensionProfile]:
        return dict(self._profiles)

    def by_key(self, key: str) -> DimensionProfile:
        try:
            return self._profiles[key]
        except KeyError as exc:
            raise UnsupportedImageSizeError(f"Unsupported profile: {key}") from exc

    def from_image_size(self, size: Sequence[int]) -> DimensionProfile:
        normalized = tuple(int(v) for v in size)
        for profile in self._profiles.values():
            if profile.image_size == normalized:
                return profile
        if self.allow_dynamic:
            width, height = normalized
            if (
                width > 0
                and height > 0
                and self.downsample_factor is not None
                and width % self.downsample_factor == 0
                and height % self.downsample_factor == 0
                and self.latent_channels is not None
                and self.capacity_bits_per_pixel is not None
            ):
                return DimensionProfile(
                    f"{width}x{height}",
                    width,
                    height,
                    calculate_max_bits(width, height, self.capacity_bits_per_pixel),
                    self.latent_channels,
                    self.downsample_factor,
                )
        supported = ", ".join(p.key for p in self._profiles.values())
        raise UnsupportedImageSizeError(
            f"Unsupported image size {normalized[0]}x{normalized[1]}; supported: {supported}"
        )

    def normalize_latent_shape(self, shape: Sequence[int]) -> tuple[int, int, int]:
        normalized = tuple(int(v) for v in shape)
        if len(normalized) == 4:
            if normalized[0] != 1:
                raise UnsupportedLatentShapeError(
                    f"Only a singleton latent batch is accepted, got {normalized}"
                )
            normalized = normalized[1:]
        if len(normalized) != 3:
            raise UnsupportedLatentShapeError(
                f"Expected latent shape [C,H,W] or [1,C,H,W], got {tuple(shape)}"
            )
        return normalized

    def from_latent_shape(self, shape: Sequence[int]) -> DimensionProfile:
        normalized = self.normalize_latent_shape(shape)
        for profile in self._profiles.values():
            if profile.latent_shape == normalized:
                return profile
        if self.allow_dynamic:
            channels, latent_height, latent_width = normalized
            if (
                channels == self.latent_channels
                and latent_height > 0
                and latent_width > 0
                and self.downsample_factor is not None
                and self.capacity_bits_per_pixel is not None
            ):
                width = latent_width * self.downsample_factor
                height = latent_height * self.downsample_factor
                return DimensionProfile(
                    f"{width}x{height}",
                    width,
                    height,
                    calculate_max_bits(width, height, self.capacity_bits_per_pixel),
                    channels,
                    self.downsample_factor,
                )
        supported = ", ".join(f"{p.key}:{p.latent_shape}" for p in self._profiles.values())
        raise UnsupportedLatentShapeError(
            f"Unsupported latent shape {normalized}; supported: {supported}"
        )
