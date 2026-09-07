import torch

from PipeLine.configuration import load_config
from PipeLine.steganography.alpha import calculate_position_alphas
from PipeLine.steganography.codec import allocate_payload, embed_grouped, extract_grouped
from PipeLine.steganography.dct import dct2d, idct2d
from PipeLine.steganography.positions import select_dct_positions


def test_dct_round_trip():
    tensor = torch.randn(32, 48)
    restored = idct2d(dct2d(tensor))
    assert torch.allclose(restored, tensor, atol=2e-5)


def test_position_selection_is_block_local_and_deterministic():
    scores = torch.ones(2, 16, 16)
    first = select_dct_positions(scores, block_size=8, positions_per_block=3, threshold=0.8)
    second = select_dct_positions(scores, block_size=8, positions_per_block=3, threshold=0.8)
    assert first == second
    assert len(first) == 2
    assert all(len(channel) == 12 for channel in first)


def test_position_selection_supports_partial_edge_blocks():
    scores = torch.ones(1, 53, 80)
    positions = select_dct_positions(
        scores, block_size=8, positions_per_block=3, threshold=0.8
    )
    assert positions[0]
    assert all(0 <= row < 53 and 0 <= col < 80 for row, col in positions[0])


def test_group_embed_extract_without_vae():
    config = load_config()
    latent = torch.randn(16, 16, 16) * 0.01
    scores = torch.ones_like(latent)
    positions = select_dct_positions(scores, block_size=8, positions_per_block=3, threshold=0.8)
    base_alpha = tuple([0.25] * 16)
    alphas = calculate_position_alphas(latent, positions, base_alpha, config.steganography)
    payload = [0, 1, 1, 0, 1, 0, 0, 1] * 4
    modified, capacity, _ = embed_grouped(
        latent, payload, config.steganography.groups, positions, alphas
    )
    extracted, _, _ = extract_grouped(
        modified, config.steganography.groups, positions, alphas, tie_bit=0
    )
    assert capacity == 16 * 12
    assert extracted[: len(payload)] == payload


def test_allocate_payload_preserves_order():
    assert allocate_payload([0, 1, 1, 0, 1], [2, 2, 4]) == [[0, 1], [1, 0], [1]]
