"""Public facade for the standardized VAE steganography workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image

from .configuration import AppConfig, load_config
from .embedding import EmbeddingService
from .evaluation import EvaluationService
from .extraction import ExtractionService
from .models import ModelProvider
from .preprocessing import PreprocessingService
from .preprocessing.images import load_validated_image
from .schemas import EmbedResult, PreparedInput
from .stability import StabilityGenerator, StabilityMapStore


class StegoPipeline:
    def __init__(self, config: AppConfig, model_provider=None):
        self.config = config
        self.models = model_provider or ModelProvider(config)
        self.stability_store = StabilityMapStore(config)
        self.preprocessing = PreprocessingService(config, self.models)
        self.embedding = EmbeddingService(config, self.models, self.stability_store)
        self.extraction = ExtractionService(config, self.models, self.stability_store)
        self.evaluation = EvaluationService(config, self.extraction)
        self.stability = StabilityGenerator(config, self.stability_store)

    @classmethod
    def from_config(cls, config_path: str | Path | None = None, model_provider=None):
        return cls(load_config(config_path), model_provider=model_provider)

    def preprocess(self, **kwargs) -> PreparedInput:
        return self.preprocessing.prepare(**kwargs)

    def embed(
        self,
        *,
        prepared: PreparedInput | None = None,
        image=None,
        latent_path=None,
        text: str | None = None,
        data: bytes | None = None,
        bits: Iterable[int] | None = None,
        output_name: str | None = None,
        output_dir: str | Path | None = None,
    ) -> EmbedResult:
        if prepared is None:
            prepared = self.preprocessing.prepare(
                image=image,
                latent_path=latent_path,
                text=text,
                data=data,
                bits=bits,
            )
        return self.embedding.embed(
            prepared, output_name=output_name, output_dir=output_dir
        )

    def extract(self, **kwargs):
        return self.extraction.extract(**kwargs)

    def evaluate(self, cover_image, embed_result: EmbedResult):
        cover, _, _ = load_validated_image(
            cover_image,
            self.config.preprocessing.registry,
            self.config.preprocessing.image_adjustment,
        )
        return self.evaluation.evaluate(cover, embed_result)

    def evaluate_existing(
        self,
        *,
        cover_image,
        stego_image,
        text: str | None = None,
        data: bytes | None = None,
        bits: Iterable[int] | None = None,
    ):
        cover, profile, _ = load_validated_image(
            cover_image,
            self.config.preprocessing.registry,
            self.config.preprocessing.image_adjustment,
        )
        stego, stego_profile, _ = load_validated_image(
            stego_image,
            self.config.preprocessing.registry,
            self.config.preprocessing.image_adjustment,
        )
        if profile.key != stego_profile.key:
            raise ValueError("Cover and stego images must use the same profile")
        payload = self.preprocessing._payload(profile.max_bits, text, data, bits)
        placeholder = EmbedResult(
            profile=profile,
            stego_path=Path(stego_image).resolve() if not isinstance(stego_image, Image.Image) else Path("<memory>"),
            stego_image=stego,
            stego_latent=None,
            embedded_bits=payload.final_bits,
            selected_capacity=self.embedding.selected_capacity(profile),
            payload=payload,
        )
        return self.evaluation.evaluate(cover, placeholder)

    def generate_stability_map(self, original_dir, reconstructed_dir, **kwargs):
        return self.stability.generate(original_dir, reconstructed_dir, **kwargs)

    def convert_stability_map(self, legacy_path, profile_key, output_path=None):
        return self.stability_store.convert_legacy(legacy_path, profile_key, output_path)

    def run_tests(self):
        from .testing import BatchTestRunner

        return BatchTestRunner(self).run()
