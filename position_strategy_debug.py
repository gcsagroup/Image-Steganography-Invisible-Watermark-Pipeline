"""
IDE-debug entry for two stability-map-free position strategies.

策略一：按归一化频率坐标插值已有稳定性得分图。
Strategy 1: interpolate an existing stability map in normalized frequency space.

策略二：根据归一化 DCT 频率构造解析式频带得分图。
Strategy 2: build an analytical band-pass score map from normalized DCT frequencies.

This file intentionally keeps experiment hyperparameters as globals. It reuses the
existing Pipeline preprocessing, VAE adapters, embedding, extraction, ECC, JPEG
attacks, and evaluation services without changing their implementation.
"""

from __future__ import annotations

import gc
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image


# Allow both `python -m PipeLine.position_strategy_debug` and direct IDE launch.
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from PipeLine.configuration import AppConfig, load_config
from PipeLine.embedding import EmbeddingService
from PipeLine.evaluation import EvaluationService
from PipeLine.evaluation.metrics import LazyLpips
from PipeLine.extraction import ExtractionService
from PipeLine.models import ModelProvider
from PipeLine.pipeline import StegoPipeline
from PipeLine.preprocessing import PreprocessingService
from PipeLine.preprocessing.images import normalize_image_size
from PipeLine.profiles import DimensionProfile
from PipeLine.stability import StabilityMapStore


# =============================================================================
# 调试输入 / Debug inputs
# =============================================================================

# Pipeline 单文件配置路径 / Pipeline single YAML configuration path
CONFIG_PATH = Path(__file__).resolve().parent / "Config" / "config.yaml"

# 单图调试载体；PROCESS_TEST_DIRECTORY=false 时使用
# Single-image debug cover; used when PROCESS_TEST_DIRECTORY=false
COVER_IMAGE_PATH = WORKSPACE_ROOT / "DataSet" / "CoverImg" / "00001.jpg"

# 是否批量处理 PipeLine/Test/Cover 下的全部图像
# Whether to process every image under PipeLine/Test/Cover
PROCESS_TEST_DIRECTORY = True

# 批量测试图像目录 / Batch-test cover-image directory
TEST_COVER_DIR = Path(__file__).resolve().parent / "Test" / "Cover"

# 批量扫描的图像扩展名 / Image extensions scanned in batch mode
TEST_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")

# 最大测试图像数；None 表示全部 / Maximum image count; None means all
MAX_TEST_IMAGES = None

# 要嵌入并在提取端恢复的文本 / Text embedded and recovered by extraction
SECRET_TEXT = "Position strategy / 位置策略"

# 两种策略的输出目录 / Output directory for both strategies
OUTPUT_ROOT = Path(__file__).resolve().parent / "StrategyExperiments"

# 按顺序执行的策略；可只保留其中一个以便单步调试
# Strategies executed in order; keep one entry for focused IDE debugging
ENABLED_STRATEGIES = ("interpolated", "analytic_band")

# 是否计算 LPIPS；关闭后调试启动更快且不加载 LPIPS 网络
# Whether to calculate LPIPS; disabling makes debugging faster
ENABLE_LPIPS = True

# JPEG 鲁棒性测试质量等级 / JPEG robustness-test quality levels
JPEG_QUALITIES = (50, 70)

# 每个策略结束后是否主动清理未使用的 CUDA 缓存
# Whether to clear unused CUDA cache after every strategy
EMPTY_CUDA_CACHE_BETWEEN_RUNS = True


# =============================================================================
# 策略一：归一化稳定图插值 / Strategy 1: normalized-map interpolation
# =============================================================================

# 作为插值来源的稳定图规格 / Profile of the source stability map
INTERPOLATION_SOURCE_PROFILE = "1024x1024"

# PyTorch 插值模式；二维连续得分图推荐 bilinear
# PyTorch interpolation mode; bilinear is recommended for continuous 2D scores
INTERPOLATION_MODE = "bilinear"

# 双线性插值是否对齐角点 / Whether bilinear interpolation aligns corners
INTERPOLATION_ALIGN_CORNERS = False

# 插值后是否逐通道重新归一化到 [0,1]
# Whether to renormalize every channel to [0,1] after interpolation
INTERPOLATION_RENORMALIZE = True

# 插值后得分的幂次；大于 1 会强调高置信度位置
# Exponent applied after interpolation; values above 1 emphasize high confidence
INTERPOLATION_SCORE_GAMMA = 1.0

# 插值策略的位置选择阈值 / Position-selection threshold for interpolation
INTERPOLATION_STABILITY_THRESHOLD = 0.80

# 每个频率平面 8x8 分区最多选取的位置数
# Maximum positions selected from each 8x8 frequency-plane region
INTERPOLATION_POSITIONS_PER_BLOCK = 3


# =============================================================================
# 策略二：解析式频带 / Strategy 2: analytical frequency band
# =============================================================================

# 解析频带的归一化径向中心；0 为 DC，1 为频率平面对角角点
# Normalized radial band center; 0 is DC and 1 is the diagonal frequency corner
ANALYTIC_BAND_CENTER = 0.34

# 高斯频带标准差；越大则候选频率范围越宽
# Gaussian band standard deviation; larger values produce a wider candidate band
ANALYTIC_BAND_SIGMA = 0.24

# 强制排除的最低归一化频率，保护 DC/极低频
# Forced minimum normalized frequency, protecting DC and very low frequencies
ANALYTIC_LOW_CUTOFF = 0.04

# 强制排除的最高归一化频率，避开最易被压缩抹除的频率
# Forced maximum normalized frequency, avoiding the most fragile high frequencies
ANALYTIC_HIGH_CUTOFF = 0.82

# 横纵频率不平衡惩罚；0 表示不惩罚，值越大越偏好对角均衡频率
# Horizontal/vertical imbalance penalty; 0 disables it, larger values favor balance
ANALYTIC_AXIS_IMBALANCE_PENALTY = 0.0

# 每通道解析得分先验；必须与 latent 通道数一致
# Per-channel analytical score priors; length must equal the latent channel count
ANALYTIC_CHANNEL_PRIORS = (
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
    1.0, 1.0, 1.0, 1.0,
)

# 解析频带的位置选择阈值 / Position-selection threshold for analytical scores
ANALYTIC_STABILITY_THRESHOLD = 0.55

# 每个频率平面 8x8 分区最多选取的位置数
# Maximum positions selected from each 8x8 frequency-plane region
ANALYTIC_POSITIONS_PER_BLOCK = 3


class InterpolatedStabilityStore:
    """Return a profile-sized map derived from one existing normalized map."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.base_store = StabilityMapStore(config)
        self.source_profile = config.preprocessing.registry.by_key(
            INTERPOLATION_SOURCE_PROFILE
        )
        self._cache: dict[str, torch.Tensor] = {}

    def load(
        self, profile: DimensionProfile, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        if profile.key not in self._cache:
            source = self.base_store.load(self.source_profile, device="cpu").float()
            target_height, target_width = profile.latent_shape[1:]
            if tuple(source.shape[1:]) == (target_height, target_width):
                derived = source.clone()
            else:
                interpolation_arguments = {
                    "size": (target_height, target_width),
                    "mode": INTERPOLATION_MODE,
                }
                if INTERPOLATION_MODE in {"linear", "bilinear", "bicubic", "trilinear"}:
                    interpolation_arguments["align_corners"] = INTERPOLATION_ALIGN_CORNERS
                derived = functional.interpolate(
                    source.unsqueeze(0), **interpolation_arguments
                ).squeeze(0)
            if INTERPOLATION_RENORMALIZE:
                flattened = derived.flatten(1)
                minimum = flattened.min(dim=1).values[:, None, None]
                maximum = flattened.max(dim=1).values[:, None, None]
                value_range = maximum - minimum
                derived = torch.where(
                    value_range > 0,
                    (derived - minimum) / value_range.clamp_min(1.0e-12),
                    torch.zeros_like(derived),
                )
            if INTERPOLATION_SCORE_GAMMA <= 0:
                raise ValueError("INTERPOLATION_SCORE_GAMMA must be positive")
            derived = derived.clamp(0, 1).pow(INTERPOLATION_SCORE_GAMMA)
            if tuple(derived.shape) != profile.latent_shape:
                raise ValueError(
                    f"Interpolated map shape {tuple(derived.shape)} does not match {profile.latent_shape}"
                )
            self._cache[profile.key] = derived.cpu()
        return self._cache[profile.key].to(device)


class AnalyticBandStabilityStore:
    """Generate deterministic, resolution-independent frequency-band scores."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._cache: dict[str, torch.Tensor] = {}
        channels = config.model.latent.channels
        if len(ANALYTIC_CHANNEL_PRIORS) != channels:
            raise ValueError(
                f"ANALYTIC_CHANNEL_PRIORS requires {channels} values, got {len(ANALYTIC_CHANNEL_PRIORS)}"
            )
        if ANALYTIC_BAND_SIGMA <= 0:
            raise ValueError("ANALYTIC_BAND_SIGMA must be positive")
        if not 0 <= ANALYTIC_LOW_CUTOFF < ANALYTIC_HIGH_CUTOFF <= 1:
            raise ValueError("Analytical frequency cutoffs must satisfy 0 <= low < high <= 1")

    def load(
        self, profile: DimensionProfile, device: str | torch.device = "cpu"
    ) -> torch.Tensor:
        if profile.key not in self._cache:
            channels, height, width = profile.latent_shape
            fy = torch.linspace(0.0, 1.0, height).unsqueeze(1)
            fx = torch.linspace(0.0, 1.0, width).unsqueeze(0)

            # Divide by sqrt(2) so the opposite DCT corner has normalized radius 1.
            radius = torch.sqrt(fy.square() + fx.square()) / (2.0 ** 0.5)
            band = torch.exp(
                -0.5
                * ((radius - ANALYTIC_BAND_CENTER) / ANALYTIC_BAND_SIGMA).square()
            )
            valid_band = (
                (radius >= ANALYTIC_LOW_CUTOFF)
                & (radius <= ANALYTIC_HIGH_CUTOFF)
            )
            imbalance = torch.abs(fx - fy)
            balance = (1.0 - ANALYTIC_AXIS_IMBALANCE_PENALTY * imbalance).clamp(0, 1)
            score_2d = (band * balance * valid_band).clamp(0, 1)
            score_2d[0, 0] = 0.0
            priors = torch.tensor(ANALYTIC_CHANNEL_PRIORS).view(channels, 1, 1)
            derived = (score_2d.unsqueeze(0) * priors).clamp(0, 1).float()
            self._cache[profile.key] = derived.cpu()
        return self._cache[profile.key].to(device)


def config_with_dynamic_profile(
    base: AppConfig, width: int, height: int
) -> tuple[AppConfig, DimensionProfile]:
    """Resolve an arbitrary adjusted size through the formal dynamic registry."""
    return base, base.preprocessing.registry.from_image_size((width, height))


def preprocess_cover_resolution(
    path: Path, base_config: AppConfig
) -> tuple[Image.Image, AppConfig, DimensionProfile, tuple[int, int], tuple[int, int]]:
    """Load RGB, stretch dimensions to nearest multiples of eight, and warn."""
    with Image.open(path) as opened:
        image = opened.convert("RGB").copy()
    original_size = image.size
    image = normalize_image_size(
        image,
        base_config.preprocessing.image_adjustment,
        source_name=path.name,
    )
    adjusted_size = image.size
    dynamic_config, profile = config_with_dynamic_profile(
        base_config, adjusted_size[0], adjusted_size[1]
    )
    return image, dynamic_config, profile, original_size, adjusted_size


def strategy_config(base: AppConfig, strategy: str) -> AppConfig:
    """Apply only strategy-specific position hyperparameters in memory."""
    if strategy == "interpolated":
        threshold = INTERPOLATION_STABILITY_THRESHOLD
        positions_per_block = INTERPOLATION_POSITIONS_PER_BLOCK
    elif strategy == "analytic_band":
        threshold = ANALYTIC_STABILITY_THRESHOLD
        positions_per_block = ANALYTIC_POSITIONS_PER_BLOCK
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    profile_configs = {
        key: replace(
            value,
            stability_threshold=threshold,
            positions_per_block=positions_per_block,
        )
        for key, value in base.steganography.profiles.items()
    }
    return replace(
        base,
        steganography=replace(base.steganography, profiles=profile_configs),
        evaluation=replace(
            base.evaluation,
            enable_lpips=ENABLE_LPIPS,
            jpeg_qualities=tuple(JPEG_QUALITIES),
        ),
    )


def pipeline_with_store(
    config: AppConfig, models: ModelProvider, store, shared_lpips: LazyLpips | None = None
) -> StegoPipeline:
    """Reuse the public facade while injecting a position-strategy score store."""
    pipeline = StegoPipeline(config, model_provider=models)
    pipeline.stability_store = store
    pipeline.embedding = EmbeddingService(config, models, store)
    pipeline.extraction = ExtractionService(config, models, store)
    pipeline.evaluation = EvaluationService(config, pipeline.extraction)
    if shared_lpips is not None:
        pipeline.evaluation._lpips = shared_lpips
    return pipeline


def metric_text(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6f}"


def run_strategy(
    strategy: str,
    base_config: AppConfig,
    shared_models: ModelProvider,
    cover_image: Image.Image,
    prepared,
    cover_path: Path,
    original_size: tuple[int, int],
    adjusted_size: tuple[int, int],
    shared_lpips: LazyLpips | None,
) -> dict:
    config = strategy_config(base_config, strategy)
    if strategy == "interpolated":
        store = InterpolatedStabilityStore(config)
    elif strategy == "analytic_band":
        store = AnalyticBandStabilityStore(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    pipeline = pipeline_with_store(config, shared_models, store, shared_lpips)

    strategy_output = OUTPUT_ROOT / strategy
    strategy_output.mkdir(parents=True, exist_ok=True)
    embedded = pipeline.embed(
        prepared=prepared,
        output_name=f"{cover_path.stem}_{strategy}",
        output_dir=strategy_output,
    )
    evaluation = pipeline.evaluate(cover_image, embedded)
    extracted = evaluation.lossless.extraction

    def attack_summary(attack) -> dict:
        header_valid = attack.extraction.payload_metadata is not None
        decoded_text_available = attack.extraction.decoded_text is not None
        return {
            "quality": attack.quality,
            "status": attack.extraction.status,
            "header_valid": header_valid,
            "raw_accuracy": attack.raw_bit_accuracy,
            "post_ecc_accuracy": (
                attack.post_ecc_bit_accuracy if header_valid else None
            ),
            "pipeline_post_ecc_accuracy": attack.post_ecc_bit_accuracy,
            "exact_text_match": (
                attack.text_match if decoded_text_available else None
            ),
            "corrected_blocks": attack.extraction.ecc_stats.corrected_blocks,
        }

    png_header_valid = extracted.payload_metadata is not None

    summary = {
        "image": cover_path.name,
        "original_size": original_size,
        "adjusted_size": adjusted_size,
        "resolution_adjusted": original_size != adjusted_size,
        "strategy": strategy,
        "profile": embedded.profile.key,
        "stego_path": str(embedded.stego_path),
        "payload_bits": len(embedded.payload.original_bits),
        "final_stream_bits": len(embedded.embedded_bits),
        "selected_capacity": embedded.selected_capacity,
        "extraction_status": extracted.status,
        "png_header_valid": png_header_valid,
        "extracted_text": extracted.decoded_text,
        "exact_text_match": (
            extracted.decoded_text == SECRET_TEXT
            if extracted.decoded_text is not None
            else None
        ),
        "psnr": evaluation.psnr,
        "ssim": evaluation.ssim,
        "lpips": evaluation.lpips,
        "png_raw_accuracy": evaluation.lossless.raw_bit_accuracy,
        "png_post_ecc_accuracy": (
            evaluation.lossless.post_ecc_bit_accuracy if png_header_valid else None
        ),
        "jpeg": [attack_summary(attack) for attack in evaluation.jpeg_results],
        "error": None,
    }

    print(f"\n===== {cover_path.name} / {strategy} =====")
    print(
        f"Resolution: {original_size[0]}x{original_size[1]} -> "
        f"{adjusted_size[0]}x{adjusted_size[1]}"
    )
    print(f"Dynamic profile: {summary['profile']}")
    print(
        f"Payload/final/capacity: {summary['payload_bits']} / "
        f"{summary['final_stream_bits']} / {summary['selected_capacity']} bits"
    )
    print(
        f"PNG extraction: {summary['extraction_status']}, "
        f"header_valid={summary['png_header_valid']}, "
        f"raw={metric_text(summary['png_raw_accuracy'])}, "
        f"post-ECC={metric_text(summary['png_post_ecc_accuracy'])}, "
        f"exact_text_match={summary['exact_text_match']}"
    )
    print(
        f"Quality: PSNR={metric_text(summary['psnr'])}, "
        f"SSIM={metric_text(summary['ssim'])}, LPIPS={metric_text(summary['lpips'])}"
    )
    for attack in summary["jpeg"]:
        print(
            f"JPEG {attack['quality']}: status={attack['status']}, "
            f"header_valid={attack['header_valid']}, "
            f"raw={metric_text(attack['raw_accuracy'])}, "
            f"post-ECC={metric_text(attack['post_ecc_accuracy'])}, "
            f"exact_text_match={attack['exact_text_match']}"
        )
    return summary


def write_comparison_report(results: list[dict]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_ROOT / "position_strategy_report.md"
    qualities = tuple(JPEG_QUALITIES)
    headers = [
        "Image",
        "Original size",
        "Adjusted size",
        "Resized",
        "Strategy",
        "Profile",
        "Payload bits",
        "Final bits",
        "Capacity",
        "PNG raw",
        "PNG header",
        "PNG ECC",
        "PNG exact text",
        "PSNR",
        "SSIM",
        "LPIPS",
    ]
    for quality in qualities:
        headers.extend(
            [
                f"JPEG {quality} raw",
                f"JPEG {quality} header",
                f"JPEG {quality} ECC",
                f"JPEG {quality} exact text",
            ]
        )
    headers.append("Error")
    rows = []
    for result in results:
        original_size = result.get("original_size")
        adjusted_size = result.get("adjusted_size")
        row = [
            result.get("image", "N/A"),
            f"{original_size[0]}x{original_size[1]}" if original_size else "N/A",
            f"{adjusted_size[0]}x{adjusted_size[1]}" if adjusted_size else "N/A",
            str(result.get("resolution_adjusted", "N/A")),
            result.get("strategy", "N/A"),
            result.get("profile", "N/A"),
            str(result.get("payload_bits", "N/A")),
            str(result.get("final_stream_bits", "N/A")),
            str(result.get("selected_capacity", "N/A")),
            metric_text(result.get("png_raw_accuracy")),
            str(result.get("png_header_valid", "N/A")),
            metric_text(result.get("png_post_ecc_accuracy")),
            str(result.get("exact_text_match", "N/A")),
            metric_text(result.get("psnr")),
            metric_text(result.get("ssim")),
            metric_text(result.get("lpips")),
        ]
        attacks = {attack["quality"]: attack for attack in result.get("jpeg", [])}
        for quality in qualities:
            attack = attacks.get(quality, {})
            row.extend(
                [
                    metric_text(attack.get("raw_accuracy")),
                    str(attack.get("header_valid", "N/A")),
                    metric_text(attack.get("post_ecc_accuracy")),
                    str(attack.get("exact_text_match", "N/A")),
                ]
            )
        row.append(str(result.get("error") or ""))
        rows.append(row)

    lines = [
        "# Position Strategy Comparison",
        "",
        f"- Input: `{TEST_COVER_DIR if PROCESS_TEST_DIRECTORY else COVER_IMAGE_PATH}`",
        f"- Source stability profile: `{INTERPOLATION_SOURCE_PROFILE}`",
        f"- Secret encoding: `{load_config(CONFIG_PATH).preprocessing.text_encoding}`",
        f"- Capacity ratio: `{load_config(CONFIG_PATH).preprocessing.capacity_bits_per_pixel}` bit/pixel",
        f"- Resolution multiple: `{load_config(CONFIG_PATH).preprocessing.image_adjustment.multiple}`",
        f"- Resolution method: `{load_config(CONFIG_PATH).preprocessing.image_adjustment.method}`",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.extend(
        [
            "",
            "The analytical strategy is deterministic and requires no precomputed stability map. ",
            "The interpolation strategy requires only the configured source-profile map and rescales it in normalized DCT-frequency coordinates.",
            "",
            "`ECC = N/A` means the fixed frame header could not be recovered, so the payload length and ECC parameters were unavailable; it does not mean Hamming corrected zero percent.",
            "`Exact text = True` means the decoded text is exactly equal to SECRET_TEXT. `False` means decoded text exists but differs. `N/A` means no valid text was decoded.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    base_config = load_config(CONFIG_PATH)
    warnings.simplefilter("always", UserWarning)
    if PROCESS_TEST_DIRECTORY:
        if not TEST_COVER_DIR.is_dir():
            raise FileNotFoundError(f"Test cover directory not found: {TEST_COVER_DIR}")
        cover_paths = sorted(
            path
            for path in TEST_COVER_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in TEST_IMAGE_EXTENSIONS
        )
        if MAX_TEST_IMAGES is not None:
            cover_paths = cover_paths[:MAX_TEST_IMAGES]
    else:
        cover_paths = [COVER_IMAGE_PATH]
    if not cover_paths:
        raise FileNotFoundError("No test cover images were found")

    print(f"Images: {len(cover_paths)}")
    shared_models = ModelProvider(base_config)
    shared_lpips = (
        LazyLpips(base_config.evaluation.lpips_net, base_config.model.device)
        if ENABLE_LPIPS
        else None
    )
    results: list[dict] = []
    for cover_path in cover_paths:
        try:
            (
                cover_image,
                image_config,
                profile,
                original_size,
                adjusted_size,
            ) = preprocess_cover_resolution(cover_path, base_config)
            print(
                f"\nPreparing {cover_path.name}: {original_size[0]}x{original_size[1]} "
                f"-> {adjusted_size[0]}x{adjusted_size[1]}, "
                f"latent={profile.latent_shape}, max_bits={profile.max_bits}"
            )
            # Encode the cover once, then reuse the same latent/payload for both strategies.
            prepared = PreprocessingService(image_config, shared_models).prepare(
                image=cover_image,
                text=SECRET_TEXT,
            )
            for strategy in ENABLED_STRATEGIES:
                try:
                    results.append(
                        run_strategy(
                            strategy,
                            image_config,
                            shared_models,
                            cover_image,
                            prepared,
                            cover_path,
                            original_size,
                            adjusted_size,
                            shared_lpips,
                        )
                    )
                except Exception as exc:
                    print(f"ERROR {cover_path.name} / {strategy}: {type(exc).__name__}: {exc}")
                    results.append(
                        {
                            "image": cover_path.name,
                            "original_size": original_size,
                            "adjusted_size": adjusted_size,
                            "resolution_adjusted": original_size != adjusted_size,
                            "strategy": strategy,
                            "profile": profile.key,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                finally:
                    if EMPTY_CUDA_CACHE_BETWEEN_RUNS and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
        except Exception as exc:
            print(f"ERROR preprocessing {cover_path.name}: {type(exc).__name__}: {exc}")
            for strategy in ENABLED_STRATEGIES:
                results.append(
                    {
                        "image": cover_path.name,
                        "strategy": strategy,
                        "error": f"Preprocessing {type(exc).__name__}: {exc}",
                    }
                )
            if EMPTY_CUDA_CACHE_BETWEEN_RUNS and torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    report = write_comparison_report(results)
    print(f"\nComparison report: {report}")


if __name__ == "__main__":
    main()
