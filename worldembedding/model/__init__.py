"""DSSDE model architecture, provided for reference and replication.

For most use cases, the pre-computed embedding vectors in data/ are sufficient.
Use this module only if you need to retrain the model or modify the architecture.
"""

from worldembedding.model.dssde import DSSDE

__all__ = ["DSSDE"]
