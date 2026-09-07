"""Resolve embedding parameters for configured and dynamic image profiles."""

from __future__ import annotations

from math import log

from ..configuration.loader import AppConfig, StegoProfileConfig
from ..profiles import DimensionProfile


def resolve_stego_profile_config(
    config: AppConfig, profile: DimensionProfile
) -> StegoProfileConfig:
    exact = config.steganography.profiles.get(profile.key)
    if exact is not None:
        return exact
    interpolation = config.stability.map_matching.interpolation
    target_ratio = profile.width / profile.height
    target_area = profile.width * profile.height

    def distance(key: str) -> tuple[float, str]:
        candidate = config.preprocessing.registry.by_key(key)
        ratio_distance = abs(log((candidate.width / candidate.height) / target_ratio))
        area_distance = abs(log((candidate.width * candidate.height) / target_area))
        return (
            interpolation.aspect_ratio_weight * ratio_distance
            + interpolation.area_weight * area_distance,
            key,
        )

    closest_key = min(config.steganography.profiles, key=distance)
    return config.steganography.profiles[closest_key]
