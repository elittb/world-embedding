"""DSSDE: daily state-space representation learning (v10 journal pipeline)."""

from .config import DSSConfig
from .expanding_windows import EXPANDING_WINDOWS

__all__ = ["DSSConfig", "EXPANDING_WINDOWS", "__version__"]
__version__ = "0.10.0"
