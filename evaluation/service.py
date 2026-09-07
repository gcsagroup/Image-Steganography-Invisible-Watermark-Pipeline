"""Compose quality metrics and attacked extraction evaluations."""

from __future__ import annotations

from PIL import Image

from ..configuration.loader import AppConfig
from ..errors import ExtractionError
from ..schemas import AttackResult, EmbedResult, EvaluationResult
from .attacks import jpeg_compress
from .metrics import LazyLpips, bit_accuracy, calculate_psnr, calculate_ssim


class EvaluationService:
    def __init__(self, config: AppConfig, extraction_service):
        self.config = config
        self.extraction_service = extraction_service
        self._lpips = LazyLpips(config.evaluation.lpips_net, config.model.device)

    def _evaluate_extraction(
        self, name: str, quality: int | None, image: Image.Image, embed_result: EmbedResult
    ) -> AttackResult:
        extraction = self.extraction_service.extract(image=image)
        expected_stream = embed_result.embedded_bits
        raw_prefix = extraction.raw_bits[: len(expected_stream)]
        raw_accuracy = bit_accuracy(expected_stream, raw_prefix)
        post_accuracy = (
            bit_accuracy(
                embed_result.payload.original_bits,
                extraction.corrected_payload_bits,
            )
            if extraction.corrected_payload_bits or not embed_result.payload.original_bits
            else None
        )
        text_match = (
            extraction.decoded_text == embed_result.payload.original_text
            if embed_result.payload.original_text is not None and extraction.decoded_text is not None
            else None
        )
        if self.config.evaluation.fail_on_extraction_error and extraction.status != "ok":
            raise ExtractionError(extraction.error or extraction.status)
        return AttackResult(
            attack_name=name,
            quality=quality,
            extraction=extraction,
            raw_bit_accuracy=raw_accuracy,
            post_ecc_bit_accuracy=post_accuracy,
            text_match=text_match,
        )

    def evaluate(self, cover_image: Image.Image, embed_result: EmbedResult) -> EvaluationResult:
        stego = embed_result.stego_image.convert("RGB")
        cover = cover_image.convert("RGB")
        cfg = self.config.evaluation
        psnr = calculate_psnr(cover, stego) if cfg.enable_psnr else None
        ssim = calculate_ssim(cover, stego) if cfg.enable_ssim else None
        lpips_value = self._lpips.calculate(cover, stego) if cfg.enable_lpips else None
        lossless = self._evaluate_extraction("png", None, stego, embed_result)
        jpeg_results = [
            self._evaluate_extraction(
                "jpeg",
                quality,
                jpeg_compress(stego, quality, cfg.jpeg_subsampling),
                embed_result,
            )
            for quality in cfg.jpeg_qualities
        ]
        return EvaluationResult(psnr, ssim, lpips_value, lossless, jpeg_results)
