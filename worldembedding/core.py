"""Core data-loading utilities for pre-computed world embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_embedding_path() -> Path:
    """Return the path to the pre-computed embedding CSV."""
    return _DATA_DIR / "world_embedding_daily.csv"


def load_embedding(path: Optional[str] = None) -> pd.DataFrame:
    """Load the daily 64-dimensional world embedding as a DataFrame.

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. Defaults to the bundled data.

    Returns
    -------
    pd.DataFrame
        Index: ``date`` (DatetimeIndex). Columns: ``dim_0`` … ``dim_63``.
    """
    p = Path(path) if path else get_embedding_path()
    df = pd.read_csv(p, parse_dates=["date"], index_col="date")
    return df


def load_regime_labels(path: Optional[str] = None) -> pd.Series:
    """Load unsupervised VQ regime labels (16 discrete codes).

    Parameters
    ----------
    path : str, optional
        Path to the CSV file. Defaults to the bundled data.

    Returns
    -------
    pd.Series
        Index: ``date`` (DatetimeIndex). Values: integer regime codes 0-15.
    """
    p = Path(path) if path else _DATA_DIR / "world_embedding_regime_labels.csv"
    df = pd.read_csv(p, parse_dates=["date"], index_col="date")
    return df["regime"]


def get_principal_components(
    n_components: int = 5,
    embedding: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Extract principal components from the world embedding.

    Parameters
    ----------
    n_components : int
        Number of PCs to extract. Default 5.
    embedding : pd.DataFrame, optional
        Pre-loaded embedding. If None, loads from the default CSV.

    Returns
    -------
    pd.DataFrame
        Index: ``date``. Columns: ``EPC1`` … ``EPC{n_components}``.
    """
    from sklearn.decomposition import PCA

    if embedding is None:
        embedding = load_embedding()

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embedding.values)
    cols = [f"EPC{i+1}" for i in range(n_components)]
    result = pd.DataFrame(components, index=embedding.index, columns=cols)
    result.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return result
