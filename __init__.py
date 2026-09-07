"""Standardized VAE image-steganography pipeline."""

from .errors import PipelineError

__all__ = ["PipelineError", "StegoPipeline"]


def __getattr__(name: str):
    if name == "StegoPipeline":
        from .pipeline import StegoPipeline

        return StegoPipeline
    raise AttributeError(name)
