"""DCT coefficient embedding, extraction, grouping, and voting."""

from __future__ import annotations

import torch

from ..errors import PayloadCapacityError
from .dct import dct2d, idct2d
from .positions import group_capacities


def embed_channel(
    channel_data: torch.Tensor,
    bits: list[int],
    positions: list[tuple[int, int]],
    alphas: list[float],
) -> torch.Tensor:
    if len(bits) > len(positions) or len(alphas) != len(positions):
        raise ValueError("Channel bits, positions, and alpha lengths do not match")
    coefficients = dct2d(channel_data).float()
    modified = coefficients.clone()
    for index, bit in enumerate(bits):
        if bit not in (0, 1):
            raise ValueError("Embedded bits may contain only 0 and 1")
        row, col = positions[index]
        alpha = float(alphas[index])
        current = modified[row, col]
        if bit == 1 and current <= alpha / 2:
            modified[row, col] = alpha
        elif bit == 0 and current >= -alpha / 2:
            modified[row, col] = -alpha
    return idct2d(modified).float()


def extract_channel(
    channel_data: torch.Tensor,
    positions: list[tuple[int, int]],
    alphas: list[float],
) -> list[int]:
    if len(alphas) != len(positions):
        raise ValueError("Channel positions and alpha lengths do not match")
    coefficients = dct2d(channel_data).float()
    bits: list[int] = []
    for (row, col), alpha in zip(positions, alphas):
        value = float(coefficients[row, col])
        if value > alpha / 2:
            bits.append(1)
        elif value < -alpha / 2:
            bits.append(0)
        else:
            bits.append(1 if value >= 0 else 0)
    return bits


def allocate_payload(
    payload_bits: list[int], capacities: list[int]
) -> list[list[int]]:
    total = sum(capacities)
    if len(payload_bits) > total:
        raise PayloadCapacityError(
            f"Selected DCT positions hold {total} bits but payload requires {len(payload_bits)} bits"
        )
    chunks: list[list[int]] = []
    offset = 0
    for capacity in capacities:
        chunks.append(payload_bits[offset : offset + capacity])
        offset += capacity
    return chunks


def embed_grouped(
    latent: torch.Tensor,
    payload_bits: list[int],
    groups: tuple[tuple[int, ...], ...],
    positions: list[list[tuple[int, int]]],
    alphas: list[list[float]],
) -> tuple[torch.Tensor, int, list[list[int]]]:
    if latent.ndim != 3:
        raise ValueError(f"Expected latent [C,H,W], got {tuple(latent.shape)}")
    capacities = group_capacities(positions, groups)
    chunks = allocate_payload(payload_bits, capacities)
    modified = latent.clone().float()
    for group, bits in zip(groups, chunks):
        for channel in group:
            modified[channel] = embed_channel(
                latent[channel], bits, positions[channel], alphas[channel]
            )
    return modified, sum(capacities), chunks


def extract_grouped(
    latent: torch.Tensor,
    groups: tuple[tuple[int, ...], ...],
    positions: list[list[tuple[int, int]]],
    alphas: list[list[float]],
    tie_bit: int = 0,
) -> tuple[list[int], list[list[int]], list[list[int]]]:
    if latent.ndim != 3:
        raise ValueError(f"Expected latent [C,H,W], got {tuple(latent.shape)}")
    capacities = group_capacities(positions, groups)
    voted_groups: list[list[int]] = []
    channel_results: list[list[int]] = []
    for group, capacity in zip(groups, capacities):
        extractions = [
            extract_channel(latent[channel], positions[channel], alphas[channel])[:capacity]
            for channel in group
        ]
        channel_results.extend(extractions)
        voted: list[int] = []
        for bit_index in range(capacity):
            votes = [bits[bit_index] for bits in extractions]
            ones = sum(votes)
            if ones * 2 == len(votes):
                voted.append(tie_bit)
            else:
                voted.append(1 if ones * 2 > len(votes) else 0)
        voted_groups.append(voted)
    flattened = [bit for group_bits in voted_groups for bit in group_bits]
    return flattened, voted_groups, channel_results
