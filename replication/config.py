"""Hyperparameter configuration for the DSSDE model.

**Default journal pipeline** writes to ``output/v14``.

v14 changes from v13:
  - FIXED: Macro contrastive loss was a near-no-op in v13 (0.3% of gradient,
    2% reduction from random baseline).  Three targeted fixes:
      * Global ADS percentile boundaries instead of per-batch quantiles
        (consistent bin assignments across batches).
      * 8 bins (octiles) instead of 4 (halves positives per anchor → harder
        task, stronger gradient signal).
      * Temperature 0.1 instead of 0.5 (5× sharper softmax gradients,
        matching TS2Vec / SupCon best practices).
      * Weight 0.5 instead of 0.1 (combined with above: ~50× stronger signal).
  - REBALANCED: VICReg was too aggressive in v13 (var_weight 0.5 flattened
    PCA spectrum to top-3 = 47.5%, harming t-SNE).  Now:
      * var_weight 0.15 (safety net, not dominant force)
      * cov_weight 0.04 (light decorrelation)
      * gamma 0.5 (allows some dims to be less important, like v4's peaked
        spectrum where top-4 PCs = 97%).
  - IMPROVED: t-SNE visualization uses PCA pre-reduction to 5D (standard
    practice for high intrinsic dimensionality) and lower perplexity.
  - Lessons from v4: peaked PCA spectrum + smooth trajectories gave excellent
    t-SNE; v14 relaxes VICReg to recover this while keeping v13's superior
    nowcasting and regime detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"


MODALITY_SPEC = {
    "news": {
        "prefixes": ["bybee_"],
        "description": "Bybee et al. WSJ topic attention levels + deltas",
    },
    "policy": {
        "prefixes": ["epu_"],
        "description": "Economic Policy Uncertainty (domestic + international)",
    },
    "geopolitical": {
        "prefixes": ["gpr_"],
        "description": "Geopolitical Risk index",
    },
    "sentiment": {
        "prefixes": ["fed_sent_"],
        "description": "SF Fed Daily News Sentiment",
    },
    "market": {
        "prefixes": ["mkt_", "cmd_"],
        "description": "US market returns, levels, commodity futures",
    },
    "international": {
        "prefixes": ["intl_", "fx_"],
        "description": "International equities, FX",
    },
    "macro": {
        "prefixes": ["fred_"],
        "description": "FRED yield curve, spreads, conditions, claims, housing, inflation expectations, etc.",
    },
}


# Macro targets used for the forecasting heads during training
MACRO_FORECAST_TARGETS = ["INDPRO", "UNRATE", "PAYEMS"]


@dataclass
class DSSConfig:
    """All hyperparameters for training and model architecture."""

    # --- Architecture ---
    embed_dim: int = 64
    hidden_dim: int = 128
    n_attention_heads: int = 4
    encoder_dropout: float = 0.1
    transition_type: str = "gru"

    # --- CPC (multi-horizon InfoNCE with stop-gradient on target) ---
    cpc_horizons: List[int] = field(default_factory=lambda: [1, 5, 21, 63, 126])
    cpc_temperature: float = 0.07
    cpc_n_negatives: int = 128

    # --- Loss weights ---
    alpha_rec: float = 1.0
    beta_smo: float = 0.001
    beta_var: float = 0.0
    use_vicreg: bool = True
    vicreg_var_weight: float = 0.15
    vicreg_cov_weight: float = 0.04
    vicreg_gamma: float = 0.5
    gamma_vq: float = 0.1
    lambda_ads: float = 0.5
    lambda_macro: float = 0.5
    use_vq: bool = True

    # --- MAE-style masked macro prediction (disabled in v9) ---
    masked_macro_prob: float = 0.0
    masked_macro_feature_frac: float = 0.0

    # --- Hierarchical temporal aggregation ---
    # v4 used (1,) only (raw daily); v5-v9 tried (1,5,21) which degraded ADS tracking.
    # Reverted to (1,) for v10.
    hierarchical_windows: List[int] = field(default_factory=lambda: [1])

    # --- Spectral normalization ---
    use_spectral_norm: bool = False

    # --- HMM prior (disabled in v9 -- numerically problematic) ---
    use_hmm_prior: bool = False
    hmm_n_states: int = 8
    hmm_weight: float = 0.0

    # --- Reconstruction feature weighting ---
    rec_weight_bybee: float = 0.2
    rec_weight_other: float = 1.0

    # --- Macro forecast heads ---
    n_macro_targets: int = 3
    macro_horizon: int = 21

    # --- Vector quantization (re-enabled: geometric anchor) ---
    vq_codebook_size: int = 16
    vq_ema_decay: float = 0.99
    vq_commitment_weight: float = 0.25

    # --- Macro-informed contrastive + collapse prevention (v14) ---
    regime_aware_smooth: bool = False
    regime_contrastive_weight: float = 0.5
    regime_contrastive_temp: float = 0.1
    macro_contrastive_n_bins: int = 8
    smooth_transition_penalty: float = 0.0

    # --- Training ---
    seq_len: int = 256
    batch_size: int = 32
    max_epochs: int = 300
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 40
    warmup_epochs: int = 15
    grad_clip: float = 1.0
    train_stride: int = 16
    val_stride: int = 1
    seed: int = 42

    # --- Data splits ---
    # Default matches longest expanding-window train cutoff (w3); use
    # ``python -m src.train journal`` (or ``python src/train.py journal``) for ew_w1..w3.
    train_end: str = "2011-12-31"
    val_end: str = "2017-12-31"

    # --- Self-supervised training splits (aligned with train_end for single-phase runs) ---
    selfsup_train_end: str = "2011-12-31"
    macro_supervision_end: str = "2011-12-31"

    # --- v6 two-phase training (unused in v9 single-phase) ---
    phase1_epochs: int = 200
    phase2_epochs: int = 100
    phase2_train_end: str = "1999-12-31"
    phase2_val_start: str = "2000-01-01"
    phase2_val_end: str = "2005-12-31"

    # --- Paths ---
    features_path: str = str(PROCESSED_DIR / "daily_features.csv")
    targets_path: str = str(PROCESSED_DIR / "macro_targets.csv")
    ads_path: str = str(RAW_DIR / "ads" / "ads_latest.csv")
    output_dir: str = str(OUTPUT_DIR / "v14")

    # --- Journal / referee-ready pipeline ---
    # When True, default ``python -m src.evaluate`` / ``python src/evaluate.py`` runs journal
    # eval if all ``ew_*/best_model.pt`` exist. Use ``journal`` / ``standard`` CLI args as needed.
    use_expanding_window_eval: bool = True
    # Stock-Watson-style macro DFM: static factors + lags in the monthly probe
    dfm_n_factors: int = 8
    dfm_n_lags: int = 3

    # Clip standardized daily features to avoid rare OOD spikes (e.g. columns nearly
    # constant in the norm window then active later), which destabilize val reconstruction.
    norm_feature_clip: float = 50.0
    # If True, ``train journal`` retrains even when ``ew_*/best_model.pt`` exists.
    journal_retrain_existing: bool = False
