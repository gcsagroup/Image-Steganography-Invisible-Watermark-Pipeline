from dataclasses import replace

import torch
from PIL import Image

from PipeLine.configuration import load_config
from PipeLine.pipeline import StegoPipeline
from PipeLine.stability.store import StabilityMapStore


class FakeOriginalVae:
    def __init__(self):
        self.latest = None

    def encode_image(self, image):
        return torch.zeros(1, 16, image.height // 8, image.width // 8)

    def decode_latent(self, latent):
        self.latest = latent.detach().clone()
        return Image.new("RGB", (latent.shape[2] * 8, latent.shape[1] * 8), color=(127, 127, 127))


class FakeModifiedEncoder:
    def __init__(self, original):
        self.original = original

    def encode_image(self, image):
        return self.original.latest.unsqueeze(0)


class FakeModels:
    def __init__(self):
        self.original_vae = FakeOriginalVae()
        self.modified_encoder = FakeModifiedEncoder(self.original_vae)


def test_pipeline_text_round_trip_with_fake_models(tmp_path):
    config = load_config()
    config = replace(
        config,
        stability=replace(config.stability, output_dir=tmp_path / "weights"),
        evaluation=replace(config.evaluation, enable_lpips=False),
    )
    models = FakeModels()
    pipeline = StegoPipeline(config, model_provider=models)
    profile = config.preprocessing.registry.by_key("1024x1024")
    StabilityMapStore(config).save(torch.ones(profile.latent_shape), profile, 1)
    cover = Image.new("RGB", profile.image_size, color=(127, 127, 127))
    embedded = pipeline.embed(
        image=cover,
        text="标准化 Pipeline",
        output_dir=tmp_path / "stego",
        output_name="sample",
    )
    extracted = pipeline.extract(image=embedded.stego_image)
    assert embedded.stego_path.is_file()
    assert extracted.status == "ok"
    assert extracted.decoded_text == "标准化 Pipeline"
    evaluation = pipeline.evaluate(cover, embedded)
    assert evaluation.lossless.raw_bit_accuracy == 1.0
    assert evaluation.lossless.post_ecc_bit_accuracy == 1.0
    assert all(item.raw_bit_accuracy == 1.0 for item in evaluation.jpeg_results)
