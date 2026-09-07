"""Legacy content- and frequency-adaptive embedding strengths."""

from __future__ import annotations

import math

import torch

from ..configuration.loader import SteganographyConfig
from .dct import dct2d


def content_adaptive_alpha(
    channel_index: int,
    channel_data: torch.Tensor,
    base_alpha: float,
    config: SteganographyConfig,
) -> float:
    coefficients = dct2d(channel_data).float()
    height, width = coefficients.shape
    rows = torch.arange(height, device=coefficients.device).unsqueeze(1)
    cols = torch.arange(width, device=coefficients.device).unsqueeze(0)
    mask = (torch.sqrt(rows.float() ** 2 + cols.float() ** 2) <= config.low_frequency_radius).float()
    mask[0, 0] = 0.0
    low_energy = torch.mean(torch.abs(coefficients * mask))
    high_energy = torch.mean(torch.abs(coefficients * (1.0 - mask)))
    total = config.low_frequency_weight * low_energy + config.high_frequency_weight * high_energy
    scaled = base_alpha * float(total.item()) / config.channel_energy[channel_index]
    return max(config.minimum_alpha, min(config.maximum_alpha, scaled))


def hvs_weight(row: int, col: int, config: SteganographyConfig) -> float:
    distance = math.sqrt(row * row + col * col)
    for index, bound in enumerate(config.hvs_distance_bounds):
        if distance < bound:
            return config.hvs_weights[index]
    return config.hvs_weights[-1]


def calculate_position_alphas(
    latent: torch.Tensor,
    positions: list[list[tuple[int, int]]],
    base_alpha: tuple[float, ...],
    config: SteganographyConfig,
) -> list[list[float]]:
    if latent.ndim != 3 or latent.shape[0] != len(positions):
        raise ValueError("Latent and position channel counts do not match")
    all_alphas: list[list[float]] = []
    for channel, channel_positions in enumerate(positions):
        channel_alpha = float(base_alpha[channel])
        if config.content_adaptive:
            channel_alpha = content_adaptive_alpha(
                channel, latent[channel], channel_alpha, config
            )
        all_alphas.append(
            [
                channel_alpha * hvs_weight(row, col, config)
                if config.frequency_adaptive
                else channel_alpha
                for row, col in channel_positions
            ]
        )
    return all_alphas
