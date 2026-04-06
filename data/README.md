# Pre-computed World Embedding Data

## Files

### `world_embedding_daily.csv`
Daily 64-dimensional world embedding vectors.

- **Rows:** 9,520 U.S. business days
- **Columns:** `date`, `dim_0` through `dim_63`
- **Look-ahead-free:** The embedding for day *t* depends only on information available through day *t*

**Period breakdown:**

| Period | Dates | Model status | News modality |
|--------|-------|--------------|---------------|
| Training & OOS test | 1985-01-02 to 2017-12-29 | Trained with expanding-window protocol (W1: train 1985–2000, test 2001–2005; W2: train 1985–2005, test 2006–2011; W3: train 1985–2011, test 2012–2017) | Actual (Bybee et al. WSJ topics) |
| Pseudo-OOS extension | 2018-01-02 to 2021-06-30 | All parameters **frozen** at Dec 2017 values; zero retraining | **FAVAR-simulated** (all other 6 modalities use actual observed data) |

### `world_embedding_regime_labels.csv`
Unsupervised regime labels derived from k-means clustering (k=16) on the embedding vectors.

- **Rows:** 9,520 business days
- **Columns:** `date`, `regime` (integer 0–15)

## Usage

```python
import pandas as pd

# Load directly
emb = pd.read_csv("world_embedding_daily.csv", parse_dates=["date"], index_col="date")

# Or use the package
from worldembedding import load_embedding
emb = load_embedding()
```

## Important Notes

1. **The 2018–2021 period is pseudo-out-of-sample.** Model parameters are frozen at December 2017. Six of seven input modalities use actual observed data; news narratives use FAVAR-simulated topic attention (see paper Section 7).
2. **The embedding is NOT a return predictor.** Out-of-sample equity premium R² ≈ 0. Use it as a state control, not a trading signal.
3. **Merge with your data on business days.** Use `pd.merge_asof` or `.reindex(method='ffill')` if your dataset includes non-business days.

## Citation

If you use these data, please cite **the paper** (not this repository):

```bibtex
@article{Tabatabaei2026WorldEmbedding,
  title   = {World Embedding: The Daily Economic State and Bond Risk Premia},
  author  = {Tabatabaei, Elham},
  year    = {2026},
  journal = {SSRN Working Paper},
  number  = {6503446},
  url     = {https://papers.ssrn.com/abstract=6503446}
}
```
