"""Overflow-safe image-quality metrics with lazy LPIPS."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity


def _same_size_rgb(first: Image.Image, second: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    if first.size != second.size:
        raise ValueError(f"Metric images must have identical sizes: {first.size} != {second.size}")
    return (
        np.asarray(first.convert("RGB"), dtype=np.float32),
        np.asarray(second.convert("RGB"), dtype=np.float32),
    )


def calculate_psnr(first: Image.Image, second: Image.Image) -> float:
    first_array, second_array = _same_size_rgb(first, second)
    mse = float(np.mean((first_array - second_array) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(first: Image.Image, second: Image.Image) -> float:
    first_array, second_array = _same_size_rgb(first, second)
    return float(
        structural_similarity(
            first_array,
            second_array,
            channel_axis=2,
            data_range=255.0,
            win_size=11,
        )
    )


class LazyLpips:
    def __init__(self, net: str, device: str = "auto"):
        self.net = net
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self._model = None
        self.load_error: Optional[str] = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self._model is None and self.load_error is None:
            try:
                import lpips

                self._model = lpips.LPIPS(net=self.net, pretrained=True, verbose=False).to(self.device)
                self._model.eval()
            except Exception as exc:
                self.load_error = str(exc)
        return self._model

    def calculate(self, first: Image.Image, second: Image.Image) -> Optional[float]:
        if first.size != second.size:
            raise ValueError("LPIPS images must have identical sizes")
        model = self._load()
        if model is None:
            return None

        def convert(image: Image.Image) -> torch.Tensor:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
            return (tensor * 2.0 - 1.0).to(self.device)

        with torch.inference_mode():
            return float(model(convert(first), convert(second)).item())


def bit_accuracy(expected: list[int], actual: list[int]) -> float:
    denominator = max(len(expected), len(actual))
    if denominator == 0:
        return 1.0
    overlap = min(len(expected), len(actual))
    correct = sum(expected[index] == actual[index] for index in range(overlap))
    return correct / denominator
