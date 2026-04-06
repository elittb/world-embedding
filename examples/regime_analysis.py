"""Example: Unsupervised regime detection using the world embedding.

Demonstrates how the embedding recovers NBER recession dates through
simple k-means clustering, no supervision on regime labels.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from worldembedding import load_embedding, load_regime_labels


def main():
    emb = load_embedding()

    # Monthly averages for cleaner regime detection
    emb_monthly = emb.resample("ME").mean()

    print("K-means clustering on monthly embeddings\n")
    print(f"{'k':>3}  {'Clusters':>10}  Recession months in dominant cluster")
    print("-" * 60)

    # NBER recession months (approximate)
    recessions = [
        ("1990-07", "1991-03"),
        ("2001-03", "2001-11"),
        ("2007-12", "2009-06"),
        ("2020-02", "2020-04"),
    ]

    is_recession = pd.Series(False, index=emb_monthly.index)
    for start, end in recessions:
        mask = (emb_monthly.index >= start) & (emb_monthly.index <= end)
        is_recession[mask] = True

    for k in [2, 4, 8]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(emb_monthly.values)
        labels_series = pd.Series(labels, index=emb_monthly.index)

        rec_labels = labels_series[is_recession]
        dominant = rec_labels.mode().iloc[0]
        pct_in_dominant = (rec_labels == dominant).mean() * 100

        print(f"{k:>3}  {k:>10}  {pct_in_dominant:.0f}% in cluster {dominant}")

    print()

    # VQ regime labels
    regimes = load_regime_labels()
    regimes_monthly = regimes.resample("ME").apply(lambda x: x.mode().iloc[0])

    is_rec_daily = pd.Series(False, index=regimes.index)
    for start, end in recessions:
        mask = (regimes.index >= start) & (regimes.index <= end)
        is_rec_daily[mask] = True

    rec_regimes = regimes[is_rec_daily]
    print(f"VQ regime distribution during recessions:")
    print(rec_regimes.value_counts().head(5))
    print()

    exp_regimes = regimes[~is_rec_daily]
    print(f"VQ regime distribution during expansions:")
    print(exp_regimes.value_counts().head(5))


if __name__ == "__main__":
    main()
