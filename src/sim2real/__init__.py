"""sim2real-av-bias-audit core package."""

from importlib.metadata import version, PackageNotFoundError

try:  # pragma: no cover - metadata lookup
    __version__ = version("sim2real-av-bias-audit")
except PackageNotFoundError:  # pragma: no cover - fallback during local dev
    __version__ = "0.0.0"

__all__ = ["__version__"]
