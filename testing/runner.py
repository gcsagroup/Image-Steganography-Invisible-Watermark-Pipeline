"""Deterministic folder-based embedding/evaluation runner."""

from __future__ import annotations

import gc
import hashlib
import random
from pathlib import Path

import torch

from ..preprocessing.images import load_validated_image
from ..schemas import TestImageResult, TestRunResult
from .report import write_report


class BatchTestRunner:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.config = pipeline.config

    def _random_bytes(self, filename: str, byte_count: int) -> bytes:
        digest = hashlib.sha256(
            f"{self.config.testing.seed}:{filename}".encode("utf-8")
        ).digest()
        generator = random.Random(int.from_bytes(digest[:8], "big"))
        return bytes(generator.randrange(256) for _ in range(byte_count))

    def run(self) -> TestRunResult:
        cfg = self.config.testing
        cfg.cover_dir.mkdir(parents=True, exist_ok=True)
        cfg.stego_dir.mkdir(parents=True, exist_ok=True)
        candidates = sorted(
            path for path in cfg.cover_dir.iterdir()
            if path.is_file() and path.suffix.lower() in cfg.image_extensions
        )
        if cfg.max_images is not None:
            candidates = candidates[: cfg.max_images]
        results: list[TestImageResult] = []
        codec = self.pipeline.preprocessing.payload_codec
        registry = self.config.preprocessing.registry
        for path in candidates:
            item = TestImageResult(filename=path.name)
            cover = None
            embedded = None
            try:
                cover, profile, _ = load_validated_image(
                    path, registry, self.config.preprocessing.image_adjustment
                )
                item.profile = profile.key
                selected_capacity = self.pipeline.embedding.selected_capacity(profile)
                usable_capacity = min(profile.max_bits, selected_capacity)
                byte_count = codec.maximum_payload_bytes(
                    usable_capacity, cfg.payload_utilization
                )
                if byte_count <= 0:
                    raise ValueError("Configured capacity cannot hold one random payload byte")
                payload = self._random_bytes(path.name, byte_count)
                output_path = cfg.stego_dir / f"{path.stem}_stego.png"
                if output_path.exists() and not cfg.overwrite:
                    raise FileExistsError(f"Output exists and overwrite=false: {output_path}")
                embedded = self.pipeline.embed(
                    image=cover,
                    data=payload,
                    output_name=output_path.stem,
                    output_dir=cfg.stego_dir,
                )
                # Evaluation only needs the decoded image and payload metadata.
                # Release the large device latent before PSNR/SSIM/LPIPS and the
                # four extraction passes, especially for multi-megapixel images.
                embedded.stego_latent = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                evaluation = self.pipeline.evaluate(cover, embedded)
                item.payload_bits = len(embedded.payload.original_bits)
                item.selected_capacity = embedded.selected_capacity
                item.stego_path = str(embedded.stego_path)
                item.evaluation = evaluation
            except Exception as exc:
                item.error = f"{type(exc).__name__}: {exc}"
                if cfg.fail_fast:
                    results.append(item)
                    break
            finally:
                cover = None
                embedded = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            results.append(item)
        report_path = write_report(self.config, results)
        successful = sum(result.evaluation is not None and not result.error for result in results)
        return TestRunResult(report_path, results, successful, len(results) - successful)
