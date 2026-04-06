"""World Embedding: daily multimodal economic state representation.

Usage:
    from worldembedding import load_embedding, load_regime_labels, get_principal_components
"""

__version__ = "0.1.0"

from worldembedding.core import (
    load_embedding,
    load_regime_labels,
    get_principal_components,
    get_embedding_path,
)

__all__ = [
    "load_embedding",
    "load_regime_labels",
    "get_principal_components",
    "get_embedding_path",
    "__version__",
]
