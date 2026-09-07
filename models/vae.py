"""Lazy local-only adapters for the original VAE and modified encoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..configuration.loader import AppConfig, ModelConfig
from ..errors import ModelLoadError


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ModelLoadError("model.device=cuda but CUDA is not available")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def image_to_tensor(image: Image.Image, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.0
    return tensor.to(device=device, dtype=dtype)


class OriginalVaeAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype)
        self._vae = None

    @property
    def loaded(self) -> bool:
        return self._vae is not None

    def _load(self):
        if self._vae is not None:
            return self._vae
        cfg = self.config.original_vae
        if not cfg.config.is_file() or not cfg.weights.is_file():
            raise ModelLoadError(f"Original VAE files are missing: {cfg.config}, {cfg.weights}")
        try:
            from diffusers import AutoencoderKL
            from safetensors.torch import load_file

            vae = AutoencoderKL.from_config(str(cfg.config))
            state_dict = load_file(str(cfg.weights), device="cpu")
            vae.load_state_dict(state_dict, strict=True)
            vae.to(device=self.device, dtype=self.dtype)
            vae.eval()
            for parameter in vae.parameters():
                parameter.requires_grad_(False)
            self._vae = vae
        except Exception as exc:
            raise ModelLoadError(f"Failed to load original VAE: {exc}") from exc
        return self._vae

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        vae = self._load()
        tensor = image_to_tensor(image, self.device, self.dtype)
        with torch.inference_mode():
            raw = vae.encode(tensor).latent_dist.mode()
        latent_cfg = self.config.latent
        return (raw - latent_cfg.shift_factor) * latent_cfg.scaling_factor

    def decode_latent(self, latent: torch.Tensor) -> Image.Image | list[Image.Image]:
        vae = self._load()
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)
        if latent.ndim != 4:
            raise ValueError(f"Expected latent [C,H,W] or [B,C,H,W], got {tuple(latent.shape)}")
        latent = latent.to(device=self.device, dtype=self.dtype)
        latent_cfg = self.config.latent
        raw = latent / latent_cfg.scaling_factor + latent_cfg.shift_factor
        with torch.inference_mode():
            decoded = vae.decode(raw).sample
        images_tensor = (decoded.float() / 2 + 0.5).clamp(0, 1).cpu()
        images = [
            Image.fromarray(
                (sample.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8),
                mode="RGB",
            )
            for sample in images_tensor
        ]
        return images[0] if len(images) == 1 else images


class ModifiedEncoderAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.dtype = resolve_dtype(config.dtype)
        self._encoder = None

    @property
    def loaded(self) -> bool:
        return self._encoder is not None

    def _load(self):
        if self._encoder is not None:
            return self._encoder
        model_config_path = self.config.original_vae.config
        weights_path = self.config.modified_encoder.weights
        if not model_config_path.is_file() or not weights_path.is_file():
            raise ModelLoadError(
                f"Modified encoder files are missing: {model_config_path}, {weights_path}"
            )
        try:
            from diffusers.models.autoencoders.vae import Encoder

            architecture = json.loads(model_config_path.read_text(encoding="utf-8"))
            encoder = Encoder(
                in_channels=architecture["in_channels"],
                out_channels=architecture["latent_channels"],
                down_block_types=architecture["down_block_types"],
                block_out_channels=architecture["block_out_channels"],
                layers_per_block=architecture["layers_per_block"],
                act_fn=architecture["act_fn"],
                norm_num_groups=architecture["norm_num_groups"],
                mid_block_add_attention=architecture.get("mid_block_add_attention", True),
            )
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
            if not isinstance(state_dict, dict):
                raise TypeError("Modified encoder checkpoint must contain a state dictionary")
            missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise ModelLoadError(
                    f"Modified encoder has unexpected weight keys: {unexpected[:5]}"
                )
            encoder.to(device=self.device, dtype=self.dtype)
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            self._encoder = encoder
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"Failed to load modified encoder: {exc}") from exc
        return self._encoder

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        encoder = self._load()
        tensor = image_to_tensor(image, self.device, self.dtype)
        with torch.inference_mode():
            raw = encoder(tensor)[:, : self.config.latent.channels]
        latent_cfg = self.config.latent
        return (raw - latent_cfg.shift_factor) * latent_cfg.scaling_factor


class ModelProvider:
    """Own lazy adapters while allowing test doubles to replace either property."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._original_vae = None
        self._modified_encoder = None

    @property
    def original_vae(self) -> OriginalVaeAdapter:
        if self._original_vae is None:
            self._original_vae = OriginalVaeAdapter(self.config.model)
        return self._original_vae

    @original_vae.setter
    def original_vae(self, value):
        self._original_vae = value

    @property
    def modified_encoder(self) -> ModifiedEncoderAdapter:
        if self._modified_encoder is None:
            self._modified_encoder = ModifiedEncoderAdapter(self.config.model)
        return self._modified_encoder

    @modified_encoder.setter
    def modified_encoder(self, value):
        self._modified_encoder = value
