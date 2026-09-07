"""Command-line interface: python -m PipeLine.cli ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .errors import PipelineError
from .pipeline import StegoPipeline
from .schemas import to_plain_data


def _bits(value: str) -> list[int]:
    normalized = "".join(value.split())
    if not normalized or set(normalized) - {"0", "1"}:
        raise argparse.ArgumentTypeError("bits must be a non-empty sequence of 0 and 1")
    return [int(bit) for bit in normalized]


def _payload_args(parser: argparse.ArgumentParser, required: bool = True) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--text", help="Secret text / 秘密文本")
    group.add_argument("--text-file", type=Path, help="Secret text file / 秘密文本文件")
    group.add_argument("--data-hex", help="Secret bytes as hexadecimal / 十六进制秘密字节")
    group.add_argument("--bits", type=_bits, help="Byte-aligned secret bits / 字节对齐秘密 bit")


def _input_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Supported cover image / 载体图像")
    group.add_argument("--latent", type=Path, help="Direct latent .pt / 直接 latent 文件")


def _resolve_payload(args, pipeline: StegoPipeline) -> dict:
    if getattr(args, "text", None) is not None:
        return {"text": args.text}
    if getattr(args, "text_file", None) is not None:
        return {
            "text": args.text_file.read_text(
                encoding=pipeline.config.preprocessing.text_encoding
            )
        }
    if getattr(args, "data_hex", None) is not None:
        return {"data": bytes.fromhex(args.data_hex)}
    if getattr(args, "bits", None) is not None:
        return {"bits": args.bits}
    raise ValueError("A reference payload is required")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standardized VAE image steganography Pipeline")
    parser.add_argument("--config", type=Path, help="Path to the single config.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    preprocess = commands.add_parser("preprocess", help="Validate and encode an image/latent and payload")
    _input_args(preprocess)
    _payload_args(preprocess)
    preprocess.add_argument("--save-latent", type=Path)

    stability = commands.add_parser("stability", help="Generate a compressed stability map")
    stability.add_argument("--original-dir", type=Path, required=True)
    stability.add_argument("--reconstructed-dir", type=Path, required=True)
    stability.add_argument("--max-samples", type=int)
    stability.add_argument("--output", type=Path)

    convert = commands.add_parser("convert-stability", help="Convert a legacy .pt stability map")
    convert.add_argument("--input", type=Path, required=True)
    convert.add_argument("--profile", choices=["1024x1024", "1536x1024", "1024x1536"], required=True)
    convert.add_argument("--output", type=Path)

    embed = commands.add_parser("embed", help="Embed a secret and save a PNG")
    _input_args(embed)
    _payload_args(embed)
    embed.add_argument("--output-name")
    embed.add_argument("--output-dir", type=Path)

    extract = commands.add_parser("extract", help="Extract a secret from image or latent")
    _input_args(extract)

    evaluate = commands.add_parser("evaluate", help="Evaluate an existing cover/stego pair")
    evaluate.add_argument("--cover", type=Path, required=True)
    evaluate.add_argument("--stego", type=Path, required=True)
    _payload_args(evaluate)

    commands.add_parser("test", help="Run the configured Test/Cover batch")
    return parser


def _print(value) -> None:
    def compact(item):
        if isinstance(item, dict):
            result = {}
            for key, nested in item.items():
                if key in {
                    "embedded_bits", "original_bits", "ecc_bits", "final_bits",
                    "raw_bits", "framed_ecc_bits", "corrected_payload_bits",
                } and isinstance(nested, list):
                    result[key] = {
                        "length": len(nested),
                        "preview": "".join(str(bit) for bit in nested[:64]),
                    }
                else:
                    result[key] = compact(nested)
            return result
        if isinstance(item, list):
            return [compact(nested) for nested in item]
        return item

    print(json.dumps(compact(to_plain_data(value)), ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pipeline = StegoPipeline.from_config(args.config)
        if args.command == "preprocess":
            payload = _resolve_payload(args, pipeline)
            prepared = pipeline.preprocess(
                image=args.image, latent_path=args.latent, **payload
            )
            if args.save_latent:
                args.save_latent.parent.mkdir(parents=True, exist_ok=True)
                torch.save(prepared.latent.detach().cpu(), args.save_latent)
            _print(prepared)
        elif args.command == "stability":
            _print(pipeline.generate_stability_map(
                args.original_dir,
                args.reconstructed_dir,
                max_samples=args.max_samples,
                output_path=args.output,
            ))
        elif args.command == "convert-stability":
            _print(pipeline.convert_stability_map(args.input, args.profile, args.output))
        elif args.command == "embed":
            payload = _resolve_payload(args, pipeline)
            _print(pipeline.embed(
                image=args.image,
                latent_path=args.latent,
                output_name=args.output_name,
                output_dir=args.output_dir,
                **payload,
            ))
        elif args.command == "extract":
            _print(pipeline.extract(image=args.image, latent_path=args.latent))
        elif args.command == "evaluate":
            payload = _resolve_payload(args, pipeline)
            _print(pipeline.evaluate_existing(
                cover_image=args.cover, stego_image=args.stego, **payload
            ))
        elif args.command == "test":
            _print(pipeline.run_tests())
        return 0
    except (PipelineError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
