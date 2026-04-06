"""Quickstart: load the world embedding and add it to your research.

This script demonstrates the most common use cases in ~30 lines.
No GPU or ML knowledge required. The embedding is pre-computed.
"""

import pandas as pd
from worldembedding import load_embedding, load_regime_labels, get_principal_components


# ── 1. Load the daily embedding ─────────────────────────────────────────────

emb = load_embedding()
print(f"Embedding: {emb.shape[0]} business days × {emb.shape[1]} dimensions")
print(f"Date range: {emb.index[0].date()} to {emb.index[-1].date()}")
print()

# ── 2. Extract principal components ─────────────────────────────────────────

epc = get_principal_components(n_components=5)
print("Explained variance (%):")
for i, v in enumerate(epc.attrs["explained_variance_ratio"]):
    print(f"  EPC{i+1}: {v*100:.1f}%")
print()

# ── 3. Use EPCs as controls in a regression ─────────────────────────────────

# Example: merge into your monthly panel
epc_monthly = epc.resample("ME").last()  # end-of-month values
print(f"Monthly EPCs: {epc_monthly.shape[0]} months")
print(epc_monthly.head())
print()

# ── 4. Load regime labels ───────────────────────────────────────────────────

regimes = load_regime_labels()
print(f"\nRegime labels: {regimes.nunique()} unique regimes")
print(f"Most frequent regime: {regimes.mode().iloc[0]}")
print(f"Current regime: {regimes.iloc[-1]} (as of {regimes.index[-1].date()})")
print()

# ── 5. Historical analog: find days most similar to a target ────────────────

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

target_date = "2008-09-15"  # Lehman Brothers collapse
target_vec = emb.loc[target_date].values.reshape(1, -1)
sims = cosine_similarity(target_vec, emb.values).flatten()
sims_series = pd.Series(sims, index=emb.index, name="cosine_similarity")

# Exclude ±21 trading days around target
target_idx = emb.index.get_loc(target_date)
mask = (abs(np.arange(len(emb)) - target_idx) > 21)
sims_filtered = sims_series[mask]

print(f"Top 5 historical analogs to {target_date} (Lehman collapse):")
for date, sim in sims_filtered.nlargest(5).items():
    print(f"  {date.date()}  similarity={sim:.4f}")
