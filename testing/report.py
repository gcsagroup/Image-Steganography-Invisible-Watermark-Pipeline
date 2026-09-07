"""Markdown batch-test report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from statistics import mean

from ..configuration.loader import AppConfig
from ..schemas import TestImageResult


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _mean(values: list[float | None]) -> str:
    valid = [value for value in values if value is not None]
    return _fmt(mean(valid)) if valid else "N/A"


def render_report(config: AppConfig, results: list[TestImageResult]) -> str:
    qualities = config.evaluation.jpeg_qualities
    headers = [
        "Image", "Profile", "Payload bits", "DCT capacity", "PNG raw acc",
        "PNG ECC acc", "Text match", "PSNR", "SSIM", "LPIPS",
    ]
    for quality in qualities:
        headers.extend([f"JPEG {quality} raw", f"JPEG {quality} ECC"])
    headers.append("Error")
    lines = [
        "# VAE Steganography Batch Test Report",
        "",
        f"- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Seed: {config.testing.seed}",
        f"- ECC: {config.preprocessing.ecc.effective_scheme}",
        f"- Encoding: {config.preprocessing.text_encoding}",
        f"- Frame header: 67 logical bits × {config.preprocessing.frame.header_repetitions} copies",
        f"- Capacity ratio: {config.preprocessing.capacity_bits_per_pixel} bit/pixel",
        f"- Image adjustment: {config.preprocessing.image_adjustment.method}, "
        f"multiple={config.preprocessing.image_adjustment.multiple}",
        f"- Stability-map strategy: {config.stability.map_matching.strategy}",
        f"- JPEG qualities: {', '.join(str(q) for q in qualities)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    successful = [result for result in results if result.evaluation is not None and not result.error]
    for result in results:
        evaluation = result.evaluation
        if evaluation is None:
            row = [
                result.filename, result.profile or "N/A", _fmt(result.payload_bits),
                _fmt(result.selected_capacity), "N/A", "N/A", "N/A", "N/A", "N/A", "N/A",
            ] + ["N/A", "N/A"] * len(qualities) + [result.error or "Unknown error"]
        else:
            jpeg_by_quality = {item.quality: item for item in evaluation.jpeg_results}
            row = [
                result.filename,
                result.profile or "N/A",
                _fmt(result.payload_bits),
                _fmt(result.selected_capacity),
                _fmt(evaluation.lossless.raw_bit_accuracy),
                _fmt(evaluation.lossless.post_ecc_bit_accuracy),
                _fmt(evaluation.lossless.text_match),
                _fmt(evaluation.psnr),
                _fmt(evaluation.ssim),
                _fmt(evaluation.lpips),
            ]
            for quality in qualities:
                attack = jpeg_by_quality.get(quality)
                row.extend([
                    _fmt(attack.raw_bit_accuracy if attack else None),
                    _fmt(attack.post_ecc_bit_accuracy if attack else None),
                ])
            row.append(result.error or "")
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")

    lines.extend([
        "",
        "## Aggregate",
        "",
        f"- Successful images: {len(successful)}",
        f"- Failed images: {len(results) - len(successful)}",
        f"- Mean PSNR: {_mean([r.evaluation.psnr for r in successful])}",
        f"- Mean SSIM: {_mean([r.evaluation.ssim for r in successful])}",
        f"- Mean LPIPS: {_mean([r.evaluation.lpips for r in successful])}",
        f"- Mean PNG raw accuracy: {_mean([r.evaluation.lossless.raw_bit_accuracy for r in successful])}",
        f"- Mean PNG post-ECC accuracy: {_mean([r.evaluation.lossless.post_ecc_bit_accuracy for r in successful])}",
    ])
    for quality in qualities:
        raw_values, ecc_values = [], []
        for result in successful:
            attack = next((item for item in result.evaluation.jpeg_results if item.quality == quality), None)
            if attack:
                raw_values.append(attack.raw_bit_accuracy)
                ecc_values.append(attack.post_ecc_bit_accuracy)
        lines.extend([
            f"- Mean JPEG {quality} raw accuracy: {_mean(raw_values)}",
            f"- Mean JPEG {quality} post-ECC accuracy: {_mean(ecc_values)}",
        ])
    return "\n".join(lines) + "\n"


def write_report(config: AppConfig, results: list[TestImageResult]) -> Path:
    target = config.testing.report_dir / config.testing.report_filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_report(config, results), encoding="utf-8")
    return target
