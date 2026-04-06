# World Embedding

**A unified daily measure of the aggregate economic state, replacing ad hoc stacking of VIX, EPU, term spreads, and credit spreads with a single, validated, look-ahead-free vector.**

[![Paper](https://img.shields.io/badge/Paper-SSRN%206503446-blue)](https://papers.ssrn.com/abstract=6503446)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pages](https://img.shields.io/badge/Pages-elittb.github.io-lightgrey)](https://elittb.github.io/world-embedding/)

**Landing page:** [elittb.github.io/world-embedding](https://elittb.github.io/world-embedding/) (mirrors this repo for search engines; enable GitHub Pages from `/docs` in repo Settings, then see [docs/SEARCH_CONSOLE.md](docs/SEARCH_CONSOLE.md) for Google Search Console).

Every event study, asset-pricing test, and macro forecast conditions on the aggregate economic state, yet most researchers approximate it by stacking partial proxies (VIX, EPU, ADS, credit spreads) that each see only one slice of the information environment. The **world embedding** compresses all of them, plus news narratives, geopolitical risk, and 44 international market signals, into a single 64-dimensional daily vector trained under a strict expanding-window protocol.

Think of it as **PCA for the full daily information environment**: just as yield-curve PCs summarize the term structure, world embedding PCs summarize the multimodal economic state.

**Why use it:**
- Raises out-of-sample R² for bond excess returns by **10–34 pp** beyond yield-curve factors ([Tabatabaei 2026](https://papers.ssrn.com/abstract=6503446))
- Recovers NBER recessions via unsupervised clustering with **13–26× higher** alignment than linear PCA
- Produces **near-zero equity-premium R²**, confirming state measurement, not return memorization
- Tracks the **Covid-19** contraction with all parameters frozen at 2017, the largest displacement in 36 years
- **9,520 business days** of pre-computed vectors ready to merge into your dataset in 3 lines of code

---

## Quick Start

### Option 1: Just use the pre-computed daily vectors (no ML needed)

Download the CSV and add embedding principal components to your regressions:

```python
import pandas as pd
from sklearn.decomposition import PCA

# Load pre-computed daily embeddings (see Data Periods section for train/test/simulation splits)
df = pd.read_csv("data/world_embedding_daily.csv", parse_dates=["date"], index_col="date")

# Extract first 5 principal components
pca = PCA(n_components=5)
epc = pd.DataFrame(
    pca.fit_transform(df.values),
    index=df.index,
    columns=[f"EPC{i+1}" for i in range(5)],
)

# Merge into your dataset
your_data = your_data.merge(epc, left_index=True, right_index=True, how="left")
```

### Option 2: Install as a Python package

```bash
pip install worldembedding
```

```python
from worldembedding import load_embedding, load_regime_labels

# Daily 64-dim embedding vectors (pandas DataFrame)
emb = load_embedding()

# Unsupervised regime labels (16 VQ codes)
regimes = load_regime_labels()

# Principal components (convenience function)
from worldembedding import get_principal_components
epc = get_principal_components(n_components=5)
```

---

## What's in the Box

| File | Description |
|------|-------------|
| `data/world_embedding_daily.csv` | 64-dim daily vectors, 9,520 business days (see [Data Periods](#data-periods)) |
| `data/world_embedding_regime_labels.csv` | Unsupervised VQ regime labels (16 codes) |
| `worldembedding/` | Pip-installable Python package for loading and using embeddings |
| `replication/` | Full model code (DSSDE architecture, training, evaluation) |
| `examples/` | Quickstart notebooks and scripts |
| `scripts/` | Spanning puzzle replication, macro forecasting, regime analysis |

---

## Key Properties

- **Look-ahead-free.** All training follows an expanding-window protocol. The embedding for day *t* uses only information available through day *t*.
- **Seven modalities.** News narratives (Bybee et al. 2024), policy uncertainty (Baker, Bloom & Davis 2016), geopolitical risk (Caldara & Iacoviello 2022), news sentiment (Shapiro, Sudhof & Wilson 2022), domestic markets, international markets/FX/commodities, and FRED macro indicators.
- **Validated.** Out-of-sample correlation of 0.50 with the ADS Business Conditions Index (which does not enter the model). Unsupervised clustering recovers NBER recession dates with 13–26× higher alignment than linear PCA.
- **Near-zero equity R².** The embedding does *not* predict stock returns out of sample, confirming it measures the economic state, not exploitable return patterns.
- **Bond spanning.** Embedding PCs raise in-sample R² for bond excess returns by 10–34 pp beyond yield-curve factors. Out-of-sample Clark-West tests significant at 1% for all maturities at the annual horizon.
- **Covid-robust.** With parameters frozen at 2017, the embedding tracks the 2020 pandemic contraction and recovery, the largest displacement in its 36-year history.

---

## Use Cases

The world embedding is designed as **infrastructure** for empirical research:

| Application | How to use |
|---|---|
| **Event studies** | Control for aggregate state using EPC1–EPC5 instead of ad hoc market-return or VIX controls |
| **Asset pricing tests** | Replace separate VIX + term spread + credit spread controls with a single parsimonious set of embedding PCs |
| **Bond return prediction** | Embedding PCs capture unspanned macro risks (Joslin, Priebsch & Singleton 2014) |
| **Regime classification** | Use VQ regime labels or k-means on the embedding for data-driven expansion/recession classification |
| **Macro forecasting** | Embedding carries incremental information for labor-market indicators in crisis periods |
| **Historical analogs** | Cosine similarity retrieval: find the most economically similar days across decades |
| **Rare disaster proxies** | Continuous tracking of crisis narratives as a candidate latent state for disaster models (Barro 2006; Wachter 2013) |

---

## Data Periods

The embedding CSV contains **9,520 business days** spanning three distinct periods:

| Period | Dates | Status | News data | All other modalities |
|--------|-------|--------|-----------|---------------------|
| **Training & OOS test** | 1985-01-02 to 2017-12-29 | Model trained and evaluated with expanding-window protocol | Actual (Bybee et al. WSJ topics) | Actual |
| **Pseudo-OOS extension** | 2018-01-02 to 2021-06-30 | All model parameters **frozen** at Dec 2017 values; no retraining | **FAVAR-simulated** (see paper §7) | Actual |

Within the training period (1985–2017), three expanding windows are used for out-of-sample evaluation:

| Window | Train | Out-of-sample test |
|--------|-------|--------------------|
| W1 | 1985–2000 | 2001–2005 |
| W2 | 1985–2005 | 2006–2011 |
| W3 | 1985–2011 | 2012–2017 |

**Important:** For the 2018–2021 extension, news narratives (360 of 505 features) are FAVAR-simulated from the historical relationship between news topics and observed market/macro data. The remaining six modalities use actual observed data. Any signal in this period is driven by the non-news modalities. See Section 7 of the [paper](https://papers.ssrn.com/abstract=6503446) for full details.

---

## Data Coverage

| Modality | Features | Source |
|----------|----------|--------|
| News narratives | 360 | Bybee et al. (2024) WSJ topic attention |
| Policy uncertainty | 38 | Baker, Bloom & Davis (2016) EPU |
| Geopolitical risk | 5 | Caldara & Iacoviello (2022) GPR |
| News sentiment | 1 | SF Fed Daily News Sentiment |
| Domestic markets | 9 | Yahoo Finance (SPY, VIX, Treasury, commodities, USD) |
| International | 44 | Yahoo Finance + FRED (equities, FX, commodities) |
| Macro indicators | 48 | FRED (yields, spreads, conditions, labor, housing) |
| **Total** | **505** | |

---

## Model Architecture (DSSDE)

The Daily State-Space Deep Embedding implements a simple economic principle: **today's state = yesterday's state + today's news.**

```
                                               Cross-Modal
  News (360) ---> [ Encoder 1 ] ---+          +-----------+         +-------+
  EPU  (38)  ---> [ Encoder 2 ] ---+          |           |         |       |
  GPR  (5)   ---> [ Encoder 3 ] ---+          |  Attention|         |  GRU  |
  Sent (1)   ---> [ Encoder 4 ] ---+--------> |           |-------> |       |---> z_t
  Mkt  (9)   ---> [ Encoder 5 ] ---+          |   Fusion  |         |       |    (64-dim)
  Intl (44)  ---> [ Encoder 6 ] ---+          |           |         +---+---+
  Macro(48)  ---> [ Encoder 7 ] ---+          +-----------+             |
                                                                   z_{t-1}
```

<p align="center">
  <img src="assets/pipeline.png" width="800" alt="DSSDE Architecture: full pipeline with training objectives"/>
</p>

**Econometric analog:** The architecture is a nonlinear state-space model. The encoders + attention = nonlinear observation equation. The GRU = nonlinear state transition (generalized Kalman filter). The expanding-window protocol = real-time econometric discipline.

See the [paper](https://papers.ssrn.com/abstract=6503446) for full details.

---

## Replication

To replicate the paper results from scratch:

```bash
# 1. Clone and install
git clone https://github.com/elittb/world-embedding.git
cd world-embedding
pip install -e ".[replication]"

# 2. Download raw data (requires FRED API key)
export FRED_API_KEY="your_key_here"
python -m replication.data.download_all

# 3. Assemble feature panel
python -m replication.data.assemble

# 4. Train (expanding-window protocol; ~2-4 hours on GPU)
python -m replication.train journal

# 5. Evaluate
python -m replication.evaluate journal
```

Pre-trained model weights for all three expanding windows and the reference model are available under [Releases](https://github.com/elittb/world-embedding/releases).

---

## Citation

If you use the world embedding or its data in your research, please cite **the paper** (not this repository):

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

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

The pre-computed embedding vectors in `data/` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), meaning you can use them freely with attribution.

---

## Contact

**Elham Tabatabaei** · [ORCID](https://orcid.org/0000-0002-9208-1105)
PhD Candidate, Schulich School of Business, York University;
Visiting Scholar, Rady School of Management, UC San Diego
Email: [elliettb@schulich.yorku.ca](mailto:elliettb@schulich.yorku.ca) · [setabatabaei@ucsd.edu](mailto:setabatabaei@ucsd.edu)
