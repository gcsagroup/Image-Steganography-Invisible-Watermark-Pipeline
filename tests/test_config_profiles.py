from pathlib import Path

import pytest

from PipeLine.configuration import load_config
from PipeLine.errors import UnsupportedImageSizeError, UnsupportedLatentShapeError


def test_default_config_and_profiles():
    config = load_config()
    profiles = config.preprocessing.registry
    assert config.preprocessing.capacity_bits_per_pixel == 0.005
    assert profiles.from_image_size((1024, 1024)).max_bits == 5242
    assert profiles.from_image_size((1536, 1024)).max_bits == 7864
    assert profiles.from_image_size((1536, 1024)).latent_shape == (16, 128, 192)
    assert profiles.from_image_size((1024, 1536)).latent_shape == (16, 192, 128)
    assert config.model.original_vae.config == Path("D:/URIS/PipeLine/Weights/vae_config.json")
    assert config.model.original_vae.weights == Path(
        "D:/URIS/PipeLine/Weights/StableDiffusion3VAE.safetensors"
    )
    assert config.model.modified_encoder.weights == Path(
        "D:/URIS/PipeLine/Weights/FineTurnEncoder.pth"
    )
    assert config.testing.payload_utilization == 0.5
    assert config.testing.max_images is None


def test_capacity_is_calculated_from_actual_pixels():
    preprocessing = load_config().preprocessing
    assert preprocessing.max_bits_for_size(424, 640) == 1356
    assert preprocessing.max_bits_for_size(2240, 1264) == 14156


def test_unsupported_dimensions_are_rejected():
    registry = load_config().preprocessing.registry
    with pytest.raises(UnsupportedImageSizeError):
        registry.from_image_size((1023, 1024))
    with pytest.raises(UnsupportedLatentShapeError):
        registry.from_latent_shape((2, 16, 128, 128))
    dynamic = registry.from_latent_shape((16, 129, 128))
    assert dynamic.key == "1024x1032"
    assert dynamic.max_bits == 5283
