"""unbias-plus: Bias detection and debiasing using a single LLM."""

from importlib.metadata import PackageNotFoundError, version

from unbias_plus.api import serve
from unbias_plus.pipeline import UnBiasPlus
from unbias_plus.schema import BiasedSegment, BiasResult


try:
    __version__ = version("unbias-plus")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["UnBiasPlus", "BiasResult", "BiasedSegment", "serve", "__version__"]
