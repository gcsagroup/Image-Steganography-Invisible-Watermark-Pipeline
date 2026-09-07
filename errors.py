"""Domain errors raised by the standardized pipeline."""


class PipelineError(Exception):
    """Base exception for expected pipeline failures."""


class ConfigurationError(PipelineError):
    pass


class UnsupportedImageSizeError(PipelineError):
    pass


class UnsupportedLatentShapeError(PipelineError):
    pass


class PayloadCapacityError(PipelineError):
    pass


class StabilityMapError(PipelineError):
    pass


class ModelLoadError(PipelineError):
    pass


class ExtractionError(PipelineError):
    pass
