from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image

from PipeLine.configuration import load_config
from PipeLine.evaluation.metrics import calculate_psnr, calculate_ssim
from PipeLine.stability.store import StabilityMapStore
from PipeLine.stability.generator import StabilityGenerator
from PipeLine.errors import StabilityMapError


def test_stability_archive_round_trip(tmp_path):
    config = load_config()
    config = replace(config, stability=replace(config.stability, output_dir=tmp_path))
    profile = config.preprocessing.registry.by_key("1024x1024")
    scores = torch.rand(profile.latent_shape)
    store = StabilityMapStore(config)
    info = store.save(scores, profile, matched_samples=3)
    restored = store.load(profile)
    assert info.path.suffix == ".npz"
    assert torch.allclose(restored, scores)
    with np.load(info.path, allow_pickle=False) as archive:
        assert int(archive["matched_samples"]) == 3


def test_quality_metrics_are_float_safe():
    black = Image.new("RGB", (16, 16), color=(0, 0, 0))
    white = Image.new("RGB", (16, 16), color=(255, 255, 255))
    assert calculate_psnr(black, black) == float("inf")
    assert calculate_psnr(black, white) == 0.0
    assert calculate_ssim(black, black) == 1.0


def test_stability_generation_from_synthetic_pair(tmp_path):
    config = load_config()
    config = replace(config, stability=replace(config.stability, output_dir=tmp_path / "maps"))
    original_dir = tmp_path / "original"
    reconstructed_dir = tmp_path / "reconstructed"
    original_dir.mkdir()
    reconstructed_dir.mkdir()
    profile = config.preprocessing.registry.by_key("1024x1024")
    original = torch.zeros(profile.latent_shape)
    reconstructed = original.clone()
    reconstructed[:, 0, 1] = 0.01
    torch.save(original, original_dir / "00001.pt")
    torch.save(reconstructed, reconstructed_dir / "00001_attack.pt")
    info = StabilityGenerator(config).generate(original_dir, reconstructed_dir)
    scores = StabilityMapStore(config).load(profile)
    assert info.matched_samples == 1
    assert scores.shape == profile.latent_shape
    assert float(scores.min()) >= 0 and float(scores.max()) <= 1


def test_normalized_interpolation_uses_closest_available_map(tmp_path):
    config = load_config()
    config = replace(config, stability=replace(config.stability, output_dir=tmp_path))
    square_profile = config.preprocessing.registry.by_key("1024x1024")
    landscape_profile = config.preprocessing.registry.by_key("1536x1024")
    target_profile = config.preprocessing.registry.from_image_size((640, 424))
    store = StabilityMapStore(config)
    store.save(torch.rand(square_profile.latent_shape), square_profile, matched_samples=1)
    store.save(torch.rand(landscape_profile.latent_shape), landscape_profile, matched_samples=1)
    with pytest.warns(UserWarning, match="1536x1024.*normalized interpolation"):
        derived = store.load(target_profile)
    assert derived.shape == target_profile.latent_shape
    assert float(derived.min()) >= 0
    assert float(derived.max()) <= 1


def test_analytic_band_requires_no_stability_archive(tmp_path):
    config = load_config()
    matching = replace(config.stability.map_matching, strategy="analytic_band")
    config = replace(
        config,
        stability=replace(config.stability, output_dir=tmp_path, map_matching=matching),
    )
    profile = config.preprocessing.registry.from_image_size((640, 424))
    scores = StabilityMapStore(config).load(profile)
    assert scores.shape == profile.latent_shape
    assert float(scores[:, 0, 0].max()) == 0.0


def test_strict_strategy_rejects_missing_exact_map(tmp_path):
    config = load_config()
    matching = replace(config.stability.map_matching, strategy="strict")
    config = replace(
        config,
        stability=replace(config.stability, output_dir=tmp_path, map_matching=matching),
    )
    profile = config.preprocessing.registry.from_image_size((640, 424))
    with pytest.raises(StabilityMapError, match="not found"):
        StabilityMapStore(config).load(profile)
