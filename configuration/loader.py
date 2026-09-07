"""Load and validate the single grouped Pipeline YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from ..errors import ConfigurationError
from ..profiles import DimensionProfile, ProfileRegistry, calculate_max_bits


PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "Config" / "config.yaml"


@dataclass(frozen=True)
class OriginalVaeConfig:
    config: Path
    weights: Path


@dataclass(frozen=True)
class ModifiedEncoderConfig:
    weights: Path


@dataclass(frozen=True)
class LatentConfig:
    channels: int
    downsample_factor: int
    scaling_factor: float
    shift_factor: float
    convention: str


@dataclass(frozen=True)
class ModelConfig:
    device: str
    dtype: str
    original_vae: OriginalVaeConfig
    modified_encoder: ModifiedEncoderConfig
    latent: LatentConfig


@dataclass(frozen=True)
class EccConfig:
    enabled: bool
    scheme: str

    @property
    def effective_scheme(self) -> str:
        return self.scheme if self.enabled else "none"


@dataclass(frozen=True)
class FrameConfig:
    header_repetitions: int


@dataclass(frozen=True)
class ImageAdjustmentConfig:
    multiple: int
    method: str
    resample: str


@dataclass(frozen=True)
class PreprocessingConfig:
    text_encoding: str
    bit_order: str
    ecc: EccConfig
    frame: FrameConfig
    image_adjustment: ImageAdjustmentConfig
    capacity_bits_per_pixel: float
    registry: ProfileRegistry

    def max_bits_for_size(self, width: int, height: int) -> int:
        return calculate_max_bits(width, height, self.capacity_bits_per_pixel)


@dataclass(frozen=True)
class StabilityConfig:
    block_size: int
    epsilon: float
    file_extension: str
    matching: str
    max_samples: Optional[int]
    output_dir: Path
    filename_template: str
    algorithm_version: str
    map_matching: "StabilityMapMatchingConfig"


@dataclass(frozen=True)
class StabilityInterpolationConfig:
    mode: str
    align_corners: bool
    renormalize: bool
    score_gamma: float
    aspect_ratio_weight: float
    area_weight: float


@dataclass(frozen=True)
class StabilityAnalyticBandConfig:
    center: float
    sigma: float
    low_cutoff: float
    high_cutoff: float
    axis_imbalance_penalty: float
    channel_priors: tuple[float, ...]


@dataclass(frozen=True)
class StabilityMapMatchingConfig:
    strategy: str
    interpolation: StabilityInterpolationConfig
    analytic_band: StabilityAnalyticBandConfig


@dataclass(frozen=True)
class StegoProfileConfig:
    positions_per_block: int
    stability_threshold: float
    base_alpha: tuple[float, ...]


@dataclass(frozen=True)
class SteganographyConfig:
    groups: tuple[tuple[int, ...], ...]
    content_adaptive: bool
    frequency_adaptive: bool
    low_frequency_radius: int
    low_frequency_weight: float
    high_frequency_weight: float
    minimum_alpha: float
    maximum_alpha: float
    channel_energy: tuple[float, ...]
    hvs_distance_bounds: tuple[float, ...]
    hvs_weights: tuple[float, ...]
    profiles: Mapping[str, StegoProfileConfig]


@dataclass(frozen=True)
class ExtractionConfig:
    input_normalization: str
    tie_bit: int
    text_errors: str


@dataclass(frozen=True)
class EvaluationConfig:
    enable_psnr: bool
    enable_ssim: bool
    enable_lpips: bool
    lpips_net: str
    jpeg_qualities: tuple[int, ...]
    jpeg_subsampling: int
    fail_on_extraction_error: bool


@dataclass(frozen=True)
class TestingConfig:
    cover_dir: Path
    stego_dir: Path
    report_dir: Path
    report_filename: str
    seed: int
    payload_utilization: float
    image_extensions: tuple[str, ...]
    fail_fast: bool
    overwrite: bool
    max_images: Optional[int]


@dataclass(frozen=True)
class AppConfig:
    root: Path
    source_path: Path
    model: ModelConfig
    preprocessing: PreprocessingConfig
    stability: StabilityConfig
    steganography: SteganographyConfig
    extraction: ExtractionConfig
    evaluation: EvaluationConfig
    testing: TestingConfig


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{path} must be a YAML mapping")
    return value


def _check_keys(value: Mapping[str, Any], allowed: set[str], path: str) -> None:
    missing = sorted(allowed - set(value))
    unknown = sorted(set(value) - allowed)
    if missing:
        raise ConfigurationError(f"{path} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ConfigurationError(f"{path} has unknown keys: {', '.join(unknown)}")


def _resolve(root: Path, raw_path: Any, path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ConfigurationError(f"{path} must be a non-empty path string")
    candidate = Path(raw_path)
    return (root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _normalize_encoding(value: Any) -> str:
    if not isinstance(value, str):
        raise ConfigurationError("preprocessing.text_encoding must be a string")
    normalized = value.lower().replace("_", "-")
    aliases = {"utf8": "utf-8", "utf16": "utf-16", "ascii": "ascii", "gbk": "gbk"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"utf-8", "gbk", "utf-16", "ascii"}:
        raise ConfigurationError(f"Unsupported text encoding: {value}")
    return normalized


def _parse_profiles(
    raw: Any, channels: int, factor: int, bits_per_pixel: float
) -> ProfileRegistry:
    values = _mapping(raw, "preprocessing.profiles")
    expected = {"1024x1024", "1536x1024", "1024x1536"}
    if set(values) != expected:
        raise ConfigurationError(
            "preprocessing.profiles must contain exactly: " + ", ".join(sorted(expected))
        )
    profiles: dict[str, DimensionProfile] = {}
    for key, item in values.items():
        item = _mapping(item, f"preprocessing.profiles.{key}")
        _check_keys(item, {"width", "height"}, f"preprocessing.profiles.{key}")
        width, height = int(item["width"]), int(item["height"])
        max_bits = calculate_max_bits(width, height, bits_per_pixel)
        if key != f"{width}x{height}" or width % factor or height % factor or max_bits <= 0:
            raise ConfigurationError(f"Invalid dimension profile: {key}")
        profiles[key] = DimensionProfile(key, width, height, max_bits, channels, factor)
    return ProfileRegistry(profiles)


def load_config(config_path: str | Path | None = None) -> AppConfig:
    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise ConfigurationError(f"Configuration file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"Cannot load configuration {path}: {exc}") from exc
    raw = _mapping(raw, "config")
    top_keys = {
        "model", "preprocessing", "stability", "steganography",
        "extraction", "evaluation", "testing",
    }
    _check_keys(raw, top_keys, "config")
    root = PIPELINE_ROOT

    model_raw = _mapping(raw["model"], "model")
    _check_keys(model_raw, {"device", "dtype", "original_vae", "modified_encoder", "latent"}, "model")
    device = str(model_raw["device"]).lower()
    dtype = str(model_raw["dtype"]).lower()
    if device not in {"auto", "cpu", "cuda"}:
        raise ConfigurationError(f"Unsupported model.device: {device}")
    if dtype not in {"float32", "float16", "bfloat16"}:
        raise ConfigurationError(f"Unsupported model.dtype: {dtype}")
    original_raw = _mapping(model_raw["original_vae"], "model.original_vae")
    _check_keys(original_raw, {"config", "weights"}, "model.original_vae")
    modified_raw = _mapping(model_raw["modified_encoder"], "model.modified_encoder")
    _check_keys(modified_raw, {"weights"}, "model.modified_encoder")
    latent_raw = _mapping(model_raw["latent"], "model.latent")
    _check_keys(
        latent_raw,
        {"channels", "downsample_factor", "scaling_factor", "shift_factor", "convention"},
        "model.latent",
    )
    latent = LatentConfig(
        channels=int(latent_raw["channels"]),
        downsample_factor=int(latent_raw["downsample_factor"]),
        scaling_factor=float(latent_raw["scaling_factor"]),
        shift_factor=float(latent_raw["shift_factor"]),
        convention=str(latent_raw["convention"]),
    )
    if latent.channels <= 0 or latent.downsample_factor <= 0 or latent.convention != "shifted_scaled":
        raise ConfigurationError("Invalid model.latent configuration")
    model = ModelConfig(
        device=device,
        dtype=dtype,
        original_vae=OriginalVaeConfig(
            _resolve(root, original_raw["config"], "model.original_vae.config"),
            _resolve(root, original_raw["weights"], "model.original_vae.weights"),
        ),
        modified_encoder=ModifiedEncoderConfig(
            _resolve(root, modified_raw["weights"], "model.modified_encoder.weights")
        ),
        latent=latent,
    )

    prep_raw = _mapping(raw["preprocessing"], "preprocessing")
    _check_keys(
        prep_raw,
        {
            "text_encoding", "bit_order", "ecc", "frame", "image_adjustment",
            "capacity_bits_per_pixel", "profiles",
        },
        "preprocessing",
    )
    if prep_raw["bit_order"] != "msb_first":
        raise ConfigurationError("Only preprocessing.bit_order=msb_first is supported")
    ecc_raw = _mapping(prep_raw["ecc"], "preprocessing.ecc")
    _check_keys(ecc_raw, {"enabled", "scheme"}, "preprocessing.ecc")
    ecc = EccConfig(bool(ecc_raw["enabled"]), str(ecc_raw["scheme"]).lower())
    if ecc.scheme not in {"none", "hamming74"}:
        raise ConfigurationError(f"Unsupported ECC scheme: {ecc.scheme}")
    frame_raw = _mapping(prep_raw["frame"], "preprocessing.frame")
    _check_keys(frame_raw, {"header_repetitions"}, "preprocessing.frame")
    frame = FrameConfig(
        header_repetitions=int(frame_raw["header_repetitions"]),
    )
    if frame.header_repetitions < 1 or frame.header_repetitions % 2 == 0:
        raise ConfigurationError(
            "preprocessing.frame.header_repetitions must be positive and odd"
        )
    adjustment_raw = _mapping(prep_raw["image_adjustment"], "preprocessing.image_adjustment")
    _check_keys(adjustment_raw, {"multiple", "method", "resample"}, "preprocessing.image_adjustment")
    image_adjustment = ImageAdjustmentConfig(
        multiple=int(adjustment_raw["multiple"]),
        method=str(adjustment_raw["method"]).lower(),
        resample=str(adjustment_raw["resample"]).lower(),
    )
    if (
        image_adjustment.multiple <= 0
        or image_adjustment.multiple % latent.downsample_factor != 0
        or image_adjustment.method not in {"stretch", "scale_crop"}
        or image_adjustment.resample not in {"nearest", "bilinear", "bicubic", "lanczos"}
    ):
        raise ConfigurationError(
            "preprocessing.image_adjustment requires a multiple divisible by the VAE downsample factor, "
            "method stretch/scale_crop, and a supported resample mode"
        )
    capacity_bits_per_pixel = float(prep_raw["capacity_bits_per_pixel"])
    if not 0 < capacity_bits_per_pixel <= 1:
        raise ConfigurationError("preprocessing.capacity_bits_per_pixel must be in (0, 1]")
    static_registry = _parse_profiles(
        prep_raw["profiles"], latent.channels, latent.downsample_factor,
        capacity_bits_per_pixel,
    )
    registry = ProfileRegistry(
        static_registry.profiles,
        capacity_bits_per_pixel=capacity_bits_per_pixel,
        latent_channels=latent.channels,
        downsample_factor=latent.downsample_factor,
        allow_dynamic=True,
    )
    preprocessing = PreprocessingConfig(
        _normalize_encoding(prep_raw["text_encoding"]),
        "msb_first",
        ecc,
        frame,
        image_adjustment,
        capacity_bits_per_pixel,
        registry,
    )

    stability_raw = _mapping(raw["stability"], "stability")
    stability_keys = {
        "block_size", "epsilon", "file_extension", "matching", "max_samples",
        "output_dir", "filename_template", "algorithm_version", "map_matching",
    }
    _check_keys(stability_raw, stability_keys, "stability")
    map_matching_raw = _mapping(stability_raw["map_matching"], "stability.map_matching")
    _check_keys(
        map_matching_raw, {"strategy", "interpolation", "analytic_band"},
        "stability.map_matching",
    )
    interpolation_raw = _mapping(
        map_matching_raw["interpolation"], "stability.map_matching.interpolation"
    )
    _check_keys(
        interpolation_raw,
        {
            "mode", "align_corners", "renormalize", "score_gamma",
            "aspect_ratio_weight", "area_weight",
        },
        "stability.map_matching.interpolation",
    )
    interpolation = StabilityInterpolationConfig(
        mode=str(interpolation_raw["mode"]).lower(),
        align_corners=bool(interpolation_raw["align_corners"]),
        renormalize=bool(interpolation_raw["renormalize"]),
        score_gamma=float(interpolation_raw["score_gamma"]),
        aspect_ratio_weight=float(interpolation_raw["aspect_ratio_weight"]),
        area_weight=float(interpolation_raw["area_weight"]),
    )
    analytic_raw = _mapping(
        map_matching_raw["analytic_band"], "stability.map_matching.analytic_band"
    )
    _check_keys(
        analytic_raw,
        {
            "center", "sigma", "low_cutoff", "high_cutoff",
            "axis_imbalance_penalty", "channel_priors",
        },
        "stability.map_matching.analytic_band",
    )
    analytic_band = StabilityAnalyticBandConfig(
        center=float(analytic_raw["center"]),
        sigma=float(analytic_raw["sigma"]),
        low_cutoff=float(analytic_raw["low_cutoff"]),
        high_cutoff=float(analytic_raw["high_cutoff"]),
        axis_imbalance_penalty=float(analytic_raw["axis_imbalance_penalty"]),
        channel_priors=tuple(float(value) for value in analytic_raw["channel_priors"]),
    )
    map_matching = StabilityMapMatchingConfig(
        strategy=str(map_matching_raw["strategy"]).lower(),
        interpolation=interpolation,
        analytic_band=analytic_band,
    )
    stability = StabilityConfig(
        block_size=int(stability_raw["block_size"]),
        epsilon=float(stability_raw["epsilon"]),
        file_extension=str(stability_raw["file_extension"]).lower(),
        matching=str(stability_raw["matching"]),
        max_samples=None if stability_raw["max_samples"] is None else int(stability_raw["max_samples"]),
        output_dir=_resolve(root, stability_raw["output_dir"], "stability.output_dir"),
        filename_template=str(stability_raw["filename_template"]),
        algorithm_version=str(stability_raw["algorithm_version"]),
        map_matching=map_matching,
    )
    if (
        stability.block_size <= 0 or stability.epsilon <= 0 or
        stability.matching != "base_id" or "{profile}" not in stability.filename_template or
        (stability.max_samples is not None and stability.max_samples <= 0)
    ):
        raise ConfigurationError("Invalid stability configuration")
    if (
        map_matching.strategy not in {"strict", "normalized_interpolation", "analytic_band"}
        or interpolation.mode not in {"nearest", "bilinear", "bicubic"}
        or interpolation.score_gamma <= 0
        or interpolation.aspect_ratio_weight < 0
        or interpolation.area_weight < 0
        or interpolation.aspect_ratio_weight + interpolation.area_weight <= 0
        or analytic_band.sigma <= 0
        or not 0 <= analytic_band.low_cutoff < analytic_band.high_cutoff <= 1
        or analytic_band.axis_imbalance_penalty < 0
        or len(analytic_band.channel_priors) != latent.channels
        or any(value < 0 for value in analytic_band.channel_priors)
    ):
        raise ConfigurationError("Invalid stability.map_matching configuration")

    stego_raw = _mapping(raw["steganography"], "steganography")
    stego_keys = {
        "groups", "content_adaptive", "frequency_adaptive", "low_frequency_radius",
        "low_frequency_weight", "high_frequency_weight", "minimum_alpha", "maximum_alpha",
        "channel_energy", "hvs_distance_bounds", "hvs_weights", "profiles",
    }
    _check_keys(stego_raw, stego_keys, "steganography")
    groups = tuple(tuple(int(c) for c in group) for group in stego_raw["groups"])
    if not groups or any(not group for group in groups):
        raise ConfigurationError("steganography.groups may not contain empty groups")
    if any(c < 0 or c >= latent.channels for group in groups for c in group):
        raise ConfigurationError("steganography.groups contains an invalid channel")
    channel_energy = tuple(float(v) for v in stego_raw["channel_energy"])
    bounds = tuple(float(v) for v in stego_raw["hvs_distance_bounds"])
    weights = tuple(float(v) for v in stego_raw["hvs_weights"])
    if len(channel_energy) != latent.channels or len(weights) != len(bounds) + 1:
        raise ConfigurationError("Invalid channel_energy or HVS array length")
    stego_profiles_raw = _mapping(stego_raw["profiles"], "steganography.profiles")
    if set(stego_profiles_raw) != set(registry.profiles):
        raise ConfigurationError("steganography.profiles must match preprocessing.profiles")
    stego_profiles: dict[str, StegoProfileConfig] = {}
    for key, value in stego_profiles_raw.items():
        value = _mapping(value, f"steganography.profiles.{key}")
        _check_keys(value, {"positions_per_block", "stability_threshold", "base_alpha"}, f"steganography.profiles.{key}")
        profile_cfg = StegoProfileConfig(
            positions_per_block=int(value["positions_per_block"]),
            stability_threshold=float(value["stability_threshold"]),
            base_alpha=tuple(float(v) for v in value["base_alpha"]),
        )
        if (
            profile_cfg.positions_per_block <= 0 or
            not 0 <= profile_cfg.stability_threshold <= 1 or
            len(profile_cfg.base_alpha) != latent.channels
        ):
            raise ConfigurationError(f"Invalid steganography profile: {key}")
        stego_profiles[key] = profile_cfg
    steganography = SteganographyConfig(
        groups=groups,
        content_adaptive=bool(stego_raw["content_adaptive"]),
        frequency_adaptive=bool(stego_raw["frequency_adaptive"]),
        low_frequency_radius=int(stego_raw["low_frequency_radius"]),
        low_frequency_weight=float(stego_raw["low_frequency_weight"]),
        high_frequency_weight=float(stego_raw["high_frequency_weight"]),
        minimum_alpha=float(stego_raw["minimum_alpha"]),
        maximum_alpha=float(stego_raw["maximum_alpha"]),
        channel_energy=channel_energy,
        hvs_distance_bounds=bounds,
        hvs_weights=weights,
        profiles=stego_profiles,
    )
    if (
        steganography.low_frequency_radius < 0 or
        steganography.minimum_alpha <= 0 or
        steganography.maximum_alpha < steganography.minimum_alpha or
        any(v <= 0 for v in channel_energy) or
        tuple(sorted(bounds)) != bounds
    ):
        raise ConfigurationError("Invalid steganography adaptive-alpha configuration")

    extraction_raw = _mapping(raw["extraction"], "extraction")
    _check_keys(extraction_raw, {"input_normalization", "tie_bit", "text_errors"}, "extraction")
    extraction = ExtractionConfig(
        input_normalization=str(extraction_raw["input_normalization"]),
        tie_bit=int(extraction_raw["tie_bit"]),
        text_errors=str(extraction_raw["text_errors"]),
    )
    if extraction.input_normalization != "minus_one_one" or extraction.tie_bit not in {0, 1} or extraction.text_errors not in {"strict", "replace"}:
        raise ConfigurationError("Invalid extraction configuration")

    evaluation_raw = _mapping(raw["evaluation"], "evaluation")
    evaluation_keys = {
        "enable_psnr", "enable_ssim", "enable_lpips", "lpips_net", "jpeg_qualities",
        "jpeg_subsampling", "fail_on_extraction_error",
    }
    _check_keys(evaluation_raw, evaluation_keys, "evaluation")
    evaluation = EvaluationConfig(
        enable_psnr=bool(evaluation_raw["enable_psnr"]),
        enable_ssim=bool(evaluation_raw["enable_ssim"]),
        enable_lpips=bool(evaluation_raw["enable_lpips"]),
        lpips_net=str(evaluation_raw["lpips_net"]),
        jpeg_qualities=tuple(int(v) for v in evaluation_raw["jpeg_qualities"]),
        jpeg_subsampling=int(evaluation_raw["jpeg_subsampling"]),
        fail_on_extraction_error=bool(evaluation_raw["fail_on_extraction_error"]),
    )
    if evaluation.lpips_net not in {"alex", "vgg", "squeeze"} or any(q < 1 or q > 100 for q in evaluation.jpeg_qualities):
        raise ConfigurationError("Invalid evaluation configuration")

    testing_raw = _mapping(raw["testing"], "testing")
    testing_keys = {
        "cover_dir", "stego_dir", "report_dir", "report_filename", "seed",
        "payload_utilization", "image_extensions", "fail_fast", "overwrite", "max_images",
    }
    _check_keys(testing_raw, testing_keys, "testing")
    testing = TestingConfig(
        cover_dir=_resolve(root, testing_raw["cover_dir"], "testing.cover_dir"),
        stego_dir=_resolve(root, testing_raw["stego_dir"], "testing.stego_dir"),
        report_dir=_resolve(root, testing_raw["report_dir"], "testing.report_dir"),
        report_filename=str(testing_raw["report_filename"]),
        seed=int(testing_raw["seed"]),
        payload_utilization=float(testing_raw["payload_utilization"]),
        image_extensions=tuple(str(v).lower() for v in testing_raw["image_extensions"]),
        fail_fast=bool(testing_raw["fail_fast"]),
        overwrite=bool(testing_raw["overwrite"]),
        max_images=None if testing_raw["max_images"] is None else int(testing_raw["max_images"]),
    )
    if not 0 < testing.payload_utilization <= 1 or not testing.report_filename.lower().endswith(".md") or (testing.max_images is not None and testing.max_images <= 0):
        raise ConfigurationError("Invalid testing configuration")

    return AppConfig(
        root=root,
        source_path=path,
        model=model,
        preprocessing=preprocessing,
        stability=stability,
        steganography=steganography,
        extraction=extraction,
        evaluation=evaluation,
        testing=testing,
    )
