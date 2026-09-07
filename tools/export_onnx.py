"""Export Pipeline inference graphs with BF16 weight storage to ONNX.

The original VAE is intentionally split into encoder and decoder graphs because
the steganography algorithm modifies the latent between these two stages.  The
fine-tuned extraction encoder is exported as a third graph.  By default, large
initializers are stored as BF16 and explicitly cast to FP32 inside the graph so
the persisted model is smaller while inference inputs, outputs, and operators
retain the Pipeline's FP32 computation convention.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..configuration import load_config
from ..models.vae import ModifiedEncoderAdapter, OriginalVaeAdapter


class OriginalVaeEncoderGraph(nn.Module):
    def __init__(self, vae: nn.Module, shift_factor: float, scaling_factor: float):
        super().__init__()
        self.encoder = vae.encoder
        self.quant_conv = vae.quant_conv if vae.quant_conv is not None else nn.Identity()
        self.shift_factor = float(shift_factor)
        self.scaling_factor = float(scaling_factor)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        moments = self.quant_conv(self.encoder(image))
        latent = moments[:, : moments.shape[1] // 2]
        return (latent - self.shift_factor) * self.scaling_factor


class OriginalVaeDecoderGraph(nn.Module):
    def __init__(self, vae: nn.Module, shift_factor: float, scaling_factor: float):
        super().__init__()
        self.post_quant_conv = (
            vae.post_quant_conv if vae.post_quant_conv is not None else nn.Identity()
        )
        self.decoder = vae.decoder
        self.shift_factor = float(shift_factor)
        self.scaling_factor = float(scaling_factor)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        raw_latent = latent / self.scaling_factor + self.shift_factor
        return self.decoder(self.post_quant_conv(raw_latent))


class ModifiedEncoderGraph(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        latent_channels: int,
        shift_factor: float,
        scaling_factor: float,
    ):
        super().__init__()
        self.encoder = encoder
        self.latent_channels = int(latent_channels)
        self.shift_factor = float(shift_factor)
        self.scaling_factor = float(scaling_factor)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(image)[:, : self.latent_channels]
        return (latent - self.shift_factor) * self.scaling_factor


def _export(
    model: nn.Module,
    example: torch.Tensor,
    output: Path,
    *,
    input_name: str,
    output_name: str,
    opset: int,
    weight_storage: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().float().cpu()
    example = example.float().cpu()
    temporary = (
        output.with_name(f"{output.stem}.float32.tmp.onnx")
        if weight_storage == "bfloat16"
        else output
    )
    try:
        with torch.inference_mode():
            torch.onnx.export(
                model,
                (example,),
                str(temporary),
                export_params=True,
                opset_version=opset,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes={
                    input_name: {0: "batch", 2: "height", 3: "width"},
                    output_name: {0: "batch", 2: "output_height", 3: "output_width"},
                },
                dynamo=False,
            )
        if weight_storage == "bfloat16":
            convert_onnx_initializers_to_bf16(temporary, output)
    finally:
        if temporary != output:
            temporary.unlink(missing_ok=True)
    _check_onnx(output)


def _float32_to_bfloat16_bytes(array: np.ndarray) -> bytes:
    """Round FP32 values to BF16 (round-to-nearest-even) and return raw bytes."""
    values = np.ascontiguousarray(array, dtype=np.float32)
    bits = values.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))
    bfloat16 = ((bits + rounding_bias) >> 16).astype(np.uint16)
    return bfloat16.tobytes()


def convert_onnx_initializers_to_bf16(source: Path, output: Path) -> int:
    """Store FLOAT initializers as BF16 and cast them back to FP32 in-graph.

    The graph's public inputs/outputs and compute tensors remain FLOAT.  This is
    a storage optimization, not a request for BF16 operator execution.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    graph_model = onnx.load(str(source), load_external_data=True)
    cast_nodes = []
    converted = 0
    for initializer in graph_model.graph.initializer:
        if initializer.data_type != TensorProto.FLOAT:
            continue
        original_name = initializer.name
        stored_name = f"{original_name}__bf16_storage"
        values = numpy_helper.to_array(initializer)
        stored = helper.make_tensor(
            name=stored_name,
            data_type=TensorProto.BFLOAT16,
            dims=values.shape,
            vals=_float32_to_bfloat16_bytes(values),
            raw=True,
        )
        initializer.CopyFrom(stored)
        cast_nodes.append(
            helper.make_node(
                "Cast",
                inputs=[stored_name],
                outputs=[original_name],
                name=f"Upcast_{converted}_{original_name}",
                to=TensorProto.FLOAT,
            )
        )
        converted += 1

    original_nodes = list(graph_model.graph.node)
    del graph_model.graph.node[:]
    graph_model.graph.node.extend(cast_nodes)
    graph_model.graph.node.extend(original_nodes)
    properties = {item.key: item.value for item in graph_model.metadata_props}
    properties.update(
        {
            "weight_storage_dtype": "bfloat16",
            "compute_dtype": "float32",
            "weight_upcast": "Cast(BFLOAT16->FLOAT)",
        }
    )
    helper.set_model_props(graph_model, properties)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(graph_model, str(output))
    del graph_model
    gc.collect()
    return converted


def _check_onnx(path: Path) -> None:
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    del model
    gc.collect()


def export_all(
    config_path: Path | None,
    output_dir: Path,
    sample_size: int,
    opset: int,
    weight_storage: str = "bfloat16",
) -> list[Path]:
    if sample_size <= 0 or sample_size % 8:
        raise ValueError("sample-size must be a positive multiple of 8")

    config = load_config(config_path)
    # Export is deliberately CPU/FP32 even if runtime configuration selects CUDA.
    config = replace(config, model=replace(config.model, device="cpu", dtype="float32"))
    latent_cfg = config.model.latent
    image = torch.zeros(1, 3, sample_size, sample_size, dtype=torch.float32)
    latent_size = sample_size // latent_cfg.downsample_factor
    latent = torch.zeros(
        1,
        latent_cfg.channels,
        latent_size,
        latent_size,
        dtype=torch.float32,
    )

    original_adapter = OriginalVaeAdapter(config.model)
    vae = original_adapter._load().float().cpu()
    suffix = "bf16" if weight_storage == "bfloat16" else "fp32"
    original_encoder_path = output_dir / f"original_vae_encoder_{suffix}.onnx"
    original_decoder_path = output_dir / f"original_vae_decoder_{suffix}.onnx"
    _export(
        OriginalVaeEncoderGraph(vae, latent_cfg.shift_factor, latent_cfg.scaling_factor),
        image,
        original_encoder_path,
        input_name="image",
        output_name="latent",
        opset=opset,
        weight_storage=weight_storage,
    )
    _export(
        OriginalVaeDecoderGraph(vae, latent_cfg.shift_factor, latent_cfg.scaling_factor),
        latent,
        original_decoder_path,
        input_name="latent",
        output_name="image",
        opset=opset,
        weight_storage=weight_storage,
    )
    del vae, original_adapter
    gc.collect()

    modified_adapter = ModifiedEncoderAdapter(config.model)
    modified_encoder = modified_adapter._load().float().cpu()
    modified_encoder_path = output_dir / f"modified_encoder_{suffix}.onnx"
    _export(
        ModifiedEncoderGraph(
            modified_encoder,
            latent_cfg.channels,
            latent_cfg.shift_factor,
            latent_cfg.scaling_factor,
        ),
        image,
        modified_encoder_path,
        input_name="image",
        output_name="latent",
        opset=opset,
        weight_storage=weight_storage,
    )

    return [original_encoder_path, original_decoder_path, modified_encoder_path]


def main() -> int:
    parser = argparse.ArgumentParser(description="Export Pipeline inference graphs to ONNX")
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "Weights" / "ONNX",
    )
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--weight-storage",
        choices=["bfloat16", "float32"],
        default="bfloat16",
        help="Initializer storage dtype; BF16 is upcast to FP32 inside the graph",
    )
    args = parser.parse_args()

    outputs = export_all(
        args.config,
        args.output_dir.resolve(),
        args.sample_size,
        args.opset,
        args.weight_storage,
    )
    for output in outputs:
        print(f"exported: {output} ({output.stat().st_size} bytes)")
    print(
        f"weight storage: {args.weight_storage}; public and compute dtype: float32"
    )
    print("ONNX is an intermediate format; convert it for the target accelerator/runtime before deployment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
