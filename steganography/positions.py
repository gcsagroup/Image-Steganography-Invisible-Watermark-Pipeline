"""Select profile-specific DCT embedding coordinates from stability maps."""

from __future__ import annotations

import torch

from ..errors import StabilityMapError


def select_dct_positions(
    stability_map: torch.Tensor,
    *,
    block_size: int,
    positions_per_block: int,
    threshold: float,
) -> list[list[tuple[int, int]]]:
    if stability_map.ndim != 3:
        raise StabilityMapError(f"Expected stability map [C,H,W], got {tuple(stability_map.shape)}")
    channels, height, width = stability_map.shape
    select_count = min(block_size * block_size, positions_per_block)
    results: list[list[tuple[int, int]]] = []
    scores = stability_map.detach().float().cpu()
    for channel in range(channels):
        positions: list[tuple[int, int]] = []
        block_rows = (height + block_size - 1) // block_size
        block_cols = (width + block_size - 1) // block_size
        for block_row in range(block_rows):
            for block_col in range(block_cols):
                start_row, start_col = block_row * block_size, block_col * block_size
                block_2d = scores[
                    channel,
                    start_row : min(start_row + block_size, height),
                    start_col : min(start_col + block_size, width),
                ]
                block = block_2d.flatten()
                actual_select_count = min(select_count, block.numel())
                indices = torch.argsort(block, descending=True, stable=True)[:actual_select_count]
                for flat_index in indices.tolist():
                    if float(block[flat_index]) < threshold:
                        continue
                    local_row, local_col = divmod(flat_index, block_2d.shape[1])
                    positions.append((start_row + local_row, start_col + local_col))
        results.append(positions)
    return results


def group_capacities(
    positions: list[list[tuple[int, int]]], groups: tuple[tuple[int, ...], ...]
) -> list[int]:
    return [min(len(positions[channel]) for channel in group) for group in groups]
