# General Robust Image Steganography PipeLine

[简体中文](README.md) | [繁體中文](README_ZH-Traditional.md) | [English](README_EN.md)

URIS is a VAE-based image steganography pipeline that hides secret data in image latent space. It covers cover-image preprocessing, payload framing and error correction, VAE encoding and decoding, stable embedding-position selection, secret extraction, JPEG robustness evaluation, image-quality evaluation, and batch testing. All runtime settings are managed in [`Config/config.yaml`](Config/config.yaml).

## Key Features

- Uses the native VAE from the SD3-family LDM to encode a cover image into a `latent`, then decodes the modified latent into a stego image.
- Accepts `.pt` latents shaped `[16,H,W]` or `[1,16,H,W]` for research and debugging without image encoding.
- Supports UTF-8, GBK, ASCII, and UTF-16 text, as well as raw bytes and byte-aligned bit sequences.
- Uses a compact fixed-length header carrying the text encoding, ECC mode, payload length, and checksums; no fragile end marker is required.
- Supports optional Hamming(7,4) error correction, header CRC8, and payload CRC32.
- Provides strict-map, normalized-interpolation, and analytical-frequency-band position strategies.
- Normalizes irregular image dimensions either by direct stretching or aspect-ratio-preserving scale and center crop.
- Calculates capacity dynamically from the actual cover-image pixel count; the default budget is `0.005 bit/pixel`.
- Includes PNG and JPEG 90/70/50 extraction tests plus PSNR, SSIM, and LPIPS quality metrics.

## Workflow

**Embedding:**

```text
Cover image
   │
   ├─ Size validation and normalization ──> RGB image
   │
   └─ Native VAE (FP32) encoding ──> cover latent
                                            │
Secret text/bytes/bits ──> encoding ──> header ──> ECC ──> final bitstream
                                            │
Stability map/analytical band ──> DCT position selection ──> alpha ──> embedding
                                                                          │
                                                               Native VAE decoding
                                                                          │
                                                                    Stego image
```

**Extraction:**

```text
PNG/JPEG stego image ──> fine-tuned encoder ──> latent ──> matching positions/parameters
                       ──> header majority vote ──> ECC ──> CRC ──> text/bytes
```

Embedding and extraction must use the same model weights, stability strategy, channel groups, and steganography parameters. Otherwise, the selected positions or decision boundaries may differ.

## Project Structure

```text
PipeLine/
├── Config/
│   └── config.yaml                 # Single bilingual configuration file
├── Weights/
│   ├── vae_config.json             # Native VAE architecture
│   ├── diffusion_pytorch_model.safetensors # Native VAE weights
│   ├── encoder_final.pth           # Fine-tuned extraction encoder
│   ├── stability_*.npz             # Compressed stability maps
├── Stego/                          # Default stego PNG output
├── Test/
│   ├── Cover/                      # Batch-test cover images
│   ├── Stego/                      # Batch-test stego images
│   └── Reports/report.md           # Latest batch-test report
├── configuration/                  # YAML loading and validation
├── preprocessing/                  # Image normalization, payload, header, ECC
├── models/                         # Lazy native-VAE and encoder loading
├── stability/                      # Stability-map generation and matching
├── steganography/                  # DCT, positions, alpha, bit codec
├── embedding/                      # Embedding service
├── extraction/                     # Extraction service
├── evaluation/                     # Attacks and quality/accuracy metrics
├── testing/                        # Batch runner and Markdown report
├── tests/                          # Unit and fake-model integration tests
├── pipeline.py                     # Public StegoPipeline facade
├── cli.py                          # Command-line interface
├── position_strategy_debug.py      # Position-strategy debug entry point
├── tools/export_onnx.py            # Reproducible ONNX exporter
├── profiles.py                     # Profiles and dynamic capacity
├── schemas.py                      # Input/output data contracts
└── requirements.txt                # Python dependencies
```

## Model Weights and Responsibilities

The production model files are stored in `PipeLine/Weights/` and loaded from there by default:

```text
PipeLine/Weights/vae_config.json
PipeLine/Weights/diffusion_pytorch_model.safetensors
PipeLine/Weights/encoder_final.pth
```

Paths can be changed under `model.original_vae` and `model.modified_encoder` in `Config/config.yaml`. Relative paths are resolved from `PipeLine/`.

| Model | Files | Responsibility |
| --- | --- | --- |
| Native VAE | `vae_config.json` + `diffusion_pytorch_model.safetensors` | Encodes the cover RGB image into a latent before embedding and decodes the stego latent into a PNG afterward. |
| Fine-tuned standalone encoder | `encoder_final.pth` | Re-encodes PNG/JPEG stego images for DCT decisions, header recovery, ECC decoding, and secret extraction. |

### Stored Precision and FP32 Upcasting

Tensor-by-tensor inspection shows that all 244 tensors in `diffusion_pytorch_model.safetensors` are stored as **BF16**. BF16 is only the checkpoint storage format. The Pipeline constructs an FP32 model, upcasts the weights to FP32 at load time, and runs VAE inference in FP32 for the best currently supported result.

The current `encoder_final.pth` contains 106 **FP32** tensors and is not stored as BF16. It is still loaded explicitly with `model.dtype: float32`. If a future extraction encoder is stored as BF16, it must likewise be upcast to FP32 before the FP32 benchmark claims in this document apply.

The copied weights were checked against the original `Weights/` files with SHA-256 and are byte-identical.

## Installation

A CUDA-capable PyTorch environment is recommended. The project has been tested in the local Conda `base` environment:

```powershell
conda activate base
python -m pip install -r PipeLine/requirements.txt
```

Major dependencies include PyTorch, Diffusers, Safetensors, SciPy, Pillow, PyYAML, scikit-image, and LPIPS. Install a PyTorch build compatible with the local CUDA driver.

Before running, verify that:

1. The native VAE configuration and weights exist.
2. The fine-tuned encoder weights exist.
3. `model.dtype` remains `float32`.
4. The selected stability strategy has a suitable map or a configured fallback.
5. GPU and system memory are sufficient for the target resolution.

## Configuration

All settings are in [`Config/config.yaml`](Config/config.yaml):

| Section | Purpose |
| --- | --- |
| `model` | Device, FP32 dtype, VAE/encoder weights, latent convention |
| `preprocessing` | Text encoding, ECC, header repetition, resizing, capacity |
| `stability` | Stability-map generation, storage, and matching strategies |
| `steganography` | Channel groups, positions, threshold, alpha, adaptation |
| `extraction` | Input normalization, tie handling, text decoding |
| `evaluation` | PSNR, SSIM, LPIPS, and JPEG attack levels |
| `testing` | Test directories, seed, utilization, and report output |

Place a custom global config before the subcommand:

```powershell
python -m PipeLine.cli --config path/to/config.yaml embed --image cover.png --text "secret message"
```

## Image Preprocessing and Size Normalization

Image dimensions must be compatible with VAE downsampling. `preprocessing.image_adjustment.multiple` specifies the required dimension multiple; it defaults to `8` and must be compatible with `model.latent.downsample_factor`.

If either dimension is invalid, the Pipeline emits a warning showing the original and adjusted sizes:

- `stretch`: independently resize width and height to the nearest legal multiples. It is direct but may slightly change the aspect ratio.
- `scale_crop`: preserve aspect ratio while scaling, then center-crop to the nearest legal size. This is the default; it avoids geometric stretching but may remove a small border region.

Supported resamplers are `nearest`, `bilinear`, `bicubic`, and `lanczos`:

```yaml
preprocessing:
  image_adjustment:
    multiple: 8
    method: scale_crop
    resample: lanczos
```

`1024x1024`, `1536x1024`, and `1024x1536` are preset steganography profiles. Other legal dimensions create dynamic profiles and inherit parameters from the geometrically closest preset.

## Payload, Header, and ECC

Exactly one secret input type is accepted:

- `text`: encoded as UTF-8, GBK, ASCII, or UTF-16.
- `data`: arbitrary bytes.
- `bits`: a byte-aligned sequence containing only 0 and 1.

Bytes are converted MSB-first. The default final stream is:

```text
fixed header × 3 + Hamming74(payload_bits)
```

Each logical header is 67 bits:

| Field | Length | Meaning |
| --- | ---: | --- |
| Text encoding | 2 bits | `00=UTF-8`, `01=GBK`, `10=ASCII`, `11=UTF-16` |
| ECC mode | 1 bit | `0=none`, `1=hamming74` |
| Original payload length | 24 bits | Valid bit count before ECC |
| Payload CRC32 | 32 bits | Validates recovered data |
| Header CRC8 | 8 bits | Validates the logical header |

The header is repeated three times and recovered by bitwise majority vote, for 201 bits of overhead. Extraction derives payload length and ECC settings from the header; no end marker is used.

Hamming(7,4) corrects one erroneous bit per 7-bit codeword but cannot reliably identify or recover every multi-bit error. CRC32 is the final payload-integrity check. ECC is not encryption; encrypt sensitive data separately before embedding.

## Capacity Calculation

The pixel budget uses the normalized image size:

```text
pixel_budget = floor(width × height × capacity_bits_per_pixel)
```

The default `capacity_bits_per_pixel` is `0.005`. A 1024×1024 image therefore has a 5,242-bit final-stream budget. This includes repeated headers and ECC redundancy and is not the pure user-payload capacity.

### High-Capacity Mode

When targeting **100% PNG extraction accuracy and LPIPS around 0.05**, payload modifications can be spread over more stable positions by lowering `base_alpha`, recalibrating `stability_threshold` against the actual stability map, and increasing `positions_per_block`. After dataset-specific calibration and per-image validation, `capacity_bits_per_pixel` can be raised to `0.02`:

```text
floor(1024 × 1024 × 0.02) = 20971 bits
```

This gives roughly **20,000 final embedded bits** in a 1024×1024 image. The 20,971 bits include the repeated header and ECC redundancy, so the original user payload is smaller when Hamming(7,4) is enabled.

This is a calibration target, not a guarantee from the default `positions_per_block=3`, `stability_threshold=0.8`, and `base_alpha=0.22`. The selected DCT capacity must be at least as large as the pixel budget, and the stability map must match the target image style, resolution, and aspect ratio. Confirm 100% PNG extraction and approximately 0.05 LPIPS on an independent validation set.

The actual capacity also depends on stable DCT positions:

```text
effective_capacity = min(pixel_budget, selected_DCT_capacity)
```

Capacity may therefore be insufficient even when the pixel budget is large, particularly with a high threshold, too few positions per block, or an unsuitable stability map. Hamming(7,4) expands every four payload bits to seven bits, and the 201-bit header must also be deducted.

## Stability Maps and Position Strategies

A stability score estimates how consistently a latent DCT position preserves its value or decision after VAE decoding and re-encoding. Embedding and extraction reuse the same scores and ordering.

> **Dataset limitation of the bundled 1024×1024 map**
>
> `PipeLine/Weights/stability_1024x1024.npz` was computed from the **Alaska2 dataset**. Those source images have relatively limited resolution and quality, and their content, texture, and compression distributions may differ substantially from production images. The map is an experimental prior and fallback baseline, not a universal stability model.
>
> Applying it directly to another domain—or interpolating it to a different resolution or aspect ratio—may select positions that are not actually stable for the target images. This can reduce PNG/JPEG extraction accuracy, increase header or payload CRC failures, and place stronger modifications in visually sensitive regions, reducing PSNR, SSIM, or LPIPS quality.

For best results, recompute stability maps from a representative batch of production images. Cover expected content styles, texture complexity, sharpness, source/compression quality, resolutions, and aspect ratios. Group data by style or source plus resolution/aspect-ratio tier, then generate and validate a matching map for each group.

Recommended process:

1. Sample representative production data and reserve an independent validation set.
2. Group photography, illustration, low-texture, and high-texture images, and common ratios such as 1:1, 3:2, 2:3, and 16:9.
3. Generate original/reconstructed latent pairs with the production weights, FP32 precision, preprocessing, and latent convention.
4. Compute maps per group and recalibrate `stability_threshold`, `positions_per_block`, and `base_alpha`.
5. Validate PNG/JPEG extraction, CRC, and image quality on held-out images; keep per-image LPIPS near or below 0.05.
6. Select the closest map by data domain, resolution, and aspect ratio at deployment; use interpolation or analytical bands only as fallback.

If very different styles share one resolution, maintain separate scenario-specific `Weights` sets or configs so maps with the same profile name do not overwrite one another. Revalidate whenever map data, model weights, precision, or critical preprocessing changes.

Select a strategy with `stability.map_matching.strategy`:

### `strict`

Requires an exact `stability_<width>x<height>.npz` for the target profile. It is the most controlled option for fixed resolutions with proper precomputation; a missing map is an error.

### `normalized_interpolation`

Uses an exact map when available. Otherwise, it selects the closest source map by aspect-ratio and area distance, then interpolates it to the target latent size. This is the default and supports irregular dimensions, but it is only an engineering fallback—not equivalent to a map computed for the target domain.

### `analytic_band`

Generates scores without precomputed data from normalized radial DCT frequency, band center, bandwidth, and channel priors. It is useful for unknown sizes, cold starts, and comparison experiments, but its assumptions must be evaluated independently.

## Steganography Control Parameters

Each profile defines `positions_per_block`, `stability_threshold`, and `base_alpha`. An “8×8 block” means an **8×8 DCT block in latent space**, not an 8×8 block of image pixels.

| Parameter | Direct role | Typical effect when increased |
| --- | --- | --- |
| `positions_per_block` | Maximum DCT positions per latent 8×8 block and channel | Higher capacity and modification density; image quality is more likely to fall |
| `stability_threshold` | Only positions at or above this score may be used | Fewer positions and lower capacity; positions are usually more reliable |
| `base_alpha` | Base embedding strength/decision margin per latent channel | Usually better robustness, but more distortion and subtle blur |

Relationships:

- `positions_per_block` and `stability_threshold` jointly determine the stable-position capacity ceiling.
- `base_alpha` does not directly add nominal capacity; it controls decision margin and robustness.
- Increasing positions or lowering the threshold generally adds capacity but uses more or less-stable locations.
- Raising alpha may help VAE re-encoding and JPEG robustness but increases latent modification.
- Greater capacity and stronger alpha both harm visual quality; subtle blur or softened local texture is typical.

`base_alpha` contains one value per each of the 16 latent channels. With `content_adaptive` and `frequency_adaptive`, the final per-position alpha is also scaled by content energy and frequency weight and clipped by `minimum_alpha` and `maximum_alpha`.

Default profile example:

```yaml
steganography:
  profiles:
    1024x1024:
      positions_per_block: 3
      stability_threshold: 0.8
      base_alpha: [0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22,
                   0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22, 0.22]
```

## Image Quality and Tuning

Use **LPIPS around `0.05` or lower** as the engineering target for perceptual invisibility. It is not an absolute guarantee for every image, payload, or attack. Inspect cover/stego pairs, local texture, per-image LPIPS, and required compression robustness together.

Recommended tuning order:

1. Keep FP32 and fix model weights, the stability strategy, and a representative cover set.
2. Start from `positions_per_block=3`, `stability_threshold=0.8`, and `base_alpha=0.22`.
3. Confirm the final stream fits both the pixel budget and DCT capacity.
4. Change one parameter at a time and record PNG/JPEG extraction plus PSNR, SSIM, and LPIPS.
5. If LPIPS exceeds 0.05 or blur is visible, reduce payload/position utilization, lower `base_alpha`, or increase the threshold.
6. If quality is acceptable but JPEG extraction is weak, raise alpha in small steps, improve the stability map/selection, and keep ECC enabled.
7. Review worst cases and high percentiles, not only mean LPIPS.

Capacity, robustness, and quality cannot all increase without limit. Determine the payload and attack requirements first, then find the lowest strength that meets them near the LPIPS target.

## Python API

### Embed, Extract, and Evaluate

```python
from PipeLine import StegoPipeline

pipeline = StegoPipeline.from_config()

embedded = pipeline.embed(
    image="cover.png",
    text="secret message",
    output_name="example",
)

extracted = pipeline.extract(image=embedded.stego_path)
print(extracted.status)
print(extracted.decoded_text)

evaluation = pipeline.evaluate("cover.png", embedded)
print(evaluation.psnr, evaluation.ssim, evaluation.lpips)
for attack in evaluation.jpeg_results:
    print(attack.quality, attack.raw_bit_accuracy, attack.post_ecc_bit_accuracy)
```

Stego images are saved as PNG under `PipeLine/Stego/` by default to avoid extra lossy compression.

### Staged Preprocessing

```python
prepared = pipeline.preprocess(image="cover.png", text="secret message")
embedded = pipeline.embed(prepared=prepared, output_name="example")
```

### Direct Latent Input

```python
embedded = pipeline.embed(latent_path="cover.pt", text="secret message")
extracted = pipeline.extract(latent_path="stego_latent.pt")
```

The `.pt` file must contain exactly one `[16,H,W]` or `[1,16,H,W]` tensor. The image profile is inferred with `model.latent.downsample_factor`. Input latents must follow the configured `shifted_scaled` convention.

### Evaluate an Existing Pair

```python
evaluation = pipeline.evaluate_existing(
    cover_image="cover.png",
    stego_image="stego.png",
    text="secret message",
)
```

The normalized cover and stego images must resolve to the same profile, and the reference secret must match the embedded payload.

## Command Line

Run commands from the workspace root.

### Preprocess and Save a Latent

```powershell
python -m PipeLine.cli preprocess `
  --image cover.png `
  --text "secret message" `
  --save-latent cover.pt
```

### Embed

```powershell
python -m PipeLine.cli embed `
  --image cover.png `
  --text "secret message" `
  --output-name example
```

`--text-file`, `--data-hex`, and `--bits` are mutually exclusive alternatives to `--text`:

```powershell
python -m PipeLine.cli embed --image cover.png --data-hex "48656c6c6f"
python -m PipeLine.cli embed --image cover.png --bits "0100100001101001"
```

### Extract

```powershell
python -m PipeLine.cli extract --image PipeLine/Stego/example.png
```

### Evaluate an Existing Pair

```powershell
python -m PipeLine.cli evaluate `
  --cover cover.png `
  --stego PipeLine/Stego/example.png `
  --text "secret message"
```

### Batch Test

```powershell
python -m PipeLine.cli test
```

The runner reads `PipeLine/Test/Cover/`, writes PNGs to `PipeLine/Test/Stego/`, and writes per-image and aggregate metrics to [`Test/Reports/report.md`](Test/Reports/report.md).

```yaml
testing:
  payload_utilization: 0.5  # Use 50% of effective maximum capacity
  max_images: null          # Scan every image under Test/Cover
```

The 50% ratio is applied to `min(pixel_budget, selected_DCT_capacity)`. The 201-bit header is then deducted and the largest byte-aligned pre-ECC payload is calculated. `Payload bits` in the report therefore means original secret bits, not the final framed stream.

## Generate a Stability Map

Prepare two directories containing original and reconstructed `.pt` latents with matching base IDs:

```powershell
python -m PipeLine.cli stability `
  --original-dir path/to/original_latents `
  --reconstructed-dir path/to/reconstructed_latents
```

The generator scores matching latent differences and saves compressed maps by image profile:

```text
PipeLine/Weights/stability_1024x1024.npz
PipeLine/Weights/stability_1536x1024.npz
PipeLine/Weights/stability_1024x1536.npz
```

Use `--max-samples` to limit samples or `--output` to select an output file. Convert a legacy square `.pt` cache with:

```powershell
python -m PipeLine.cli convert-stability `
  --input Weights/Alaska_Dct_stability.pt `
  --profile 1024x1024
```

Maps depend on model precision, weights, encode/decode path, and data distribution. Regenerate or revalidate them after changing any of those inputs.

## Metric Definitions

| Metric | Meaning | Direction |
| --- | --- | --- |
| `raw_bit_accuracy` | Accuracy of bits directly extracted from decision positions | Higher is better |
| `post_ecc_bit_accuracy` | Payload-bit accuracy after header recovery and ECC | Higher is better |
| `text_match` | Whether original and recovered text match exactly | `true` is an exact match |
| `PSNR` | Pixel-domain signal-to-noise ratio | Usually higher is better |
| `SSIM` | Structural similarity | Closer to 1 is better |
| `LPIPS` | Perceptual feature-space distance | Lower is better; target about or below 0.05 |

`N/A` does not mean zero accuracy. It means the value could not be calculated reliably, for example after header/CRC failure, without a valid payload, or when random byte payloads do not support text comparison. Check `status`, `error`, CRC, post-ECC accuracy, and `text_match` together.

> **Precision and benchmark statement**
>
> **Float32 (FP32)** is the reference precision for VAE encoding, decoding, and steganography evaluation. It provides the best and most stable result in the current model and algorithm chain. The native checkpoint is stored as BF16 but must be upcast to FP32 for inference.
>
> All quality and robustness results in this project are guaranteed **only for FP32 execution**. FP16, BF16, FP8, and other reduced-precision modes may change latent values, DCT coefficients, and extraction boundaries. They require new stability-map calibration and a complete benchmark on the target engine and hardware.

## Model Evaluation and Reference Results

[`Test/Reports/report.md`](Test/Reports/report.md) covers all 20 current images under `Test/Cover`, using 50% of effective maximum capacity, Hamming(7,4), normalized stability-map interpolation, and the **PyTorch FP32** model path. All 20 test flows completed.

| Aggregate | Result |
| --- | ---: |
| Successful / failed images | 20 / 0 |
| Mean LPIPS | 0.0433 |
| Mean PNG raw-bit accuracy | 0.9972 |
| Mean PNG post-ECC accuracy | 0.9996 |
| Mean JPEG 90 raw / post-ECC | 0.9933 / 0.9984 |
| Mean JPEG 70 raw / post-ECC | 0.9644 / 0.9968 |
| Mean JPEG 50 raw / post-ECC | 0.9431 / 0.9936 |

Post-ECC JPEG averages omit `N/A` samples whose headers could not be recovered. JPEG 90, 70, and 50 have 19, 14, and 13 comparable post-ECC samples respectively; these means are not full-set CRC success rates.

The results apply only to the listed images, random payloads, weights, config, and environment—not to other datasets, ONNX runtimes, or reduced precision. Mean LPIPS is below 0.05, but only 12 of 20 individual images are below 0.05, so production acceptance must remain per-image.

### Cover / Stego Examples

These are the last five files after sorting `Test/Cover` by name. Click an image to view it at full size. `Origin / ECC` means raw-bit accuracy / post-ECC payload-bit accuracy.

| Cover | Stego | Resolution | Capacity | Accuracy (Origin / ECC) | Visual metrics |
| :---: | :---: | :---: | --- | --- | --- |
| <a href="Test/Cover/131106338932881396_p14.png"><img src="Test/Cover/131106338932881396_p14.png" width="320" alt="p14 Cover"></a><br><sub>p14</sub> | <a href="Test/Stego/131106338932881396_p14_stego.png"><img src="Test/Stego/131106338932881396_p14_stego.png" width="320" alt="p14 Stego"></a><br><sub>p14 stego</sub> | `1536×1024` | Payload<br>`2128 bits`<br>Stream<br>`3925 bits`<br>Ceiling<br>`10190 bits` | PNG<br>`0.9995 / 1.0000`<br>J90<br>`0.9929 / 0.9977`<br>J70<br>`0.9740 / —`<br>J50<br>`0.9592 / —` | PSNR<br>`31.41`<br>SSIM<br>`0.9414`<br>LPIPS<br>`0.0251` |
| <a href="Test/Cover/20260818140720_1563_31.jpg"><img src="Test/Cover/20260818140720_1563_31.jpg" width="320" alt="1563_31 Cover"></a><br><sub>1563_31</sub> | <a href="Test/Stego/20260818140720_1563_31_stego.png"><img src="Test/Stego/20260818140720_1563_31_stego.png" width="320" alt="1563_31 Stego"></a><br><sub>1563_31 stego</sub> | `1440×1920` | Payload<br>`3832 bits`<br>Stream<br>`6907 bits`<br>Ceiling<br>`18157 bits` | PNG<br>`1.0000 / 1.0000`<br>J90<br>`0.9933 / 0.9997`<br>J70<br>`0.8455 / —`<br>J50<br>`0.7963 / —` | PSNR<br>`37.94`<br>SSIM<br>`0.9728`<br>LPIPS<br>`0.0207` |
| <a href="Test/Cover/v2-1a5f6ca2d7303de409237de4a6e70707_r.jpg"><img src="Test/Cover/v2-1a5f6ca2d7303de409237de4a6e70707_r.jpg" width="320" alt="v2-1a5f Cover"></a><br><sub>v2-1a5f…</sub> | <a href="Test/Stego/v2-1a5f6ca2d7303de409237de4a6e70707_r_stego.png"><img src="Test/Stego/v2-1a5f6ca2d7303de409237de4a6e70707_r_stego.png" width="320" alt="v2-1a5f Stego"></a><br><sub>v2-1a5f… stego</sub> | `512×512` | Payload<br>`256 bits`<br>Stream<br>`649 bits`<br>Ceiling<br>`1823 bits` | PNG<br>`0.9553 / 0.9922`<br>J90<br>`0.9399 / 0.9805`<br>J70<br>`0.8814 / —`<br>J50<br>`0.8505 / —` | PSNR<br>`27.53`<br>SSIM<br>`0.9544`<br>LPIPS<br>`0.0165` |
| <a href="Test/Cover/v2-33489711df97a738a22182a331d0cece_r.jpg"><img src="Test/Cover/v2-33489711df97a738a22182a331d0cece_r.jpg" width="320" alt="v2-3348 Cover"></a><br><sub>v2-3348…</sub> | <a href="Test/Stego/v2-33489711df97a738a22182a331d0cece_r_stego.png"><img src="Test/Stego/v2-33489711df97a738a22182a331d0cece_r_stego.png" width="320" alt="v2-3348 Stego"></a><br><sub>v2-3348… stego</sub> | `2240×1264` | Payload<br>`3928 bits`<br>Stream<br>`7075 bits`<br>Ceiling<br>`18393 bits` | PNG<br>`1.0000 / 1.0000`<br>J90<br>`0.9945 / 0.9997`<br>J70<br>`0.8167 / —`<br>J50<br>`0.7405 / —` | PSNR<br>`37.74`<br>SSIM<br>`0.9824`<br>LPIPS<br>`0.0214` |
| <a href="Test/Cover/v2-de6bae99061d02d212a5a22931ebf730_r.jpg"><img src="Test/Cover/v2-de6bae99061d02d212a5a22931ebf730_r.jpg" width="320" alt="v2-de6b Cover"></a><br><sub>v2-de6b…</sub> | <a href="Test/Stego/v2-de6bae99061d02d212a5a22931ebf730_r_stego.png"><img src="Test/Stego/v2-de6bae99061d02d212a5a22931ebf730_r_stego.png" width="320" alt="v2-de6b Stego"></a><br><sub>v2-de6b… stego</sub> | `1776×1152` | Payload<br>`2800 bits`<br>Stream<br>`5101 bits`<br>Ceiling<br>`13455 bits` | PNG<br>`0.9996 / 1.0000`<br>J90<br>`0.9994 / 1.0000`<br>J70<br>`0.9978 / 1.0000`<br>J50<br>`0.9941 / 0.9989` | PSNR<br>`30.02`<br>SSIM<br>`0.9339`<br>LPIPS<br>`0.0593` |

`—` means the header failed recovery or validation, so post-ECC accuracy is unavailable. The stream includes the 201-bit repeated header and Hamming(7,4) redundancy. The final example has LPIPS 0.0593, above the approximate 0.05 target, reinforcing the need for per-image review.

## Testing

Unit and fake-model integration tests:

```powershell
conda run -n base python -m pytest PipeLine/tests -q
```

The current result is **41 passed**. Full batch testing loads the real VAE, extraction encoder, and LPIPS network:

```powershell
conda run -n base python -m PipeLine.cli test
```

Record FP32 precision, weight versions, config, stability-map version, image list, seed, capacity, PNG/JPEG extraction, and all quality metrics for every parameter set.

## Troubleshooting

### The image size was adjusted

This warning is expected when dimensions are not legal multiples. Use `scale_crop` to preserve aspect ratio or `stretch` to preserve all borders while accepting slight geometric distortion.

### `strict` reports a missing stability map

Generate `stability_<width>x<height>.npz` for the target profile, or switch to `normalized_interpolation` / `analytic_band` and re-evaluate.

### The payload exceeds capacity

Capacity is constrained by the bit-per-pixel budget, repeated header, ECC, and available DCT positions. Shorten the secret, disable ECC only if reduced robustness is acceptable, increase position capacity, or use a larger cover. Do not increase strength blindly.

### PNG works but JPEG extraction fails

Confirm that the image was not resized or cropped, then inspect JPEG quality and chroma subsampling. Raise `base_alpha` in small steps if image quality permits, improve stability-map selection, and keep ECC enabled. Recheck LPIPS after every change.

### LPIPS exceeds 0.05 or subtle blur appears

Reduce payload utilization or `positions_per_block`, raise `stability_threshold`, or lower `base_alpha`. Sensitivity varies by image, so evaluate per image.

### Accuracy falls with FP16, BF16, or FP8

Reduced-precision errors affect VAE image/latent conversion and DCT decisions. FP32 is the only currently benchmarked and guaranteed path. Recalibrate and fully test any lower-precision deployment.

## Limitations and Security

- Steganography is not encryption; protect payload confidentiality with independent cryptography.
- Lossy compression, repeated VAE processing, resizing, cropping, rotation, filters, and platform transcoding can damage the payload.
- `normalized_interpolation` and `analytic_band` are fallbacks, not substitutes for measured target-domain stability maps.
- Changes to models, weights, precision, config, or preprocessing may invalidate existing maps and benchmark results.
- Production deployment should benchmark FP32 on its own data and accept results by CRC success, exact text match, and per-image LPIPS.
