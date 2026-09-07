"""In-memory image attacks used by robustness evaluation."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


def jpeg_compress(image: Image.Image, quality: int, subsampling: int = 0) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(
        buffer, format="JPEG", quality=int(quality), subsampling=int(subsampling)
    )
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB").copy()
