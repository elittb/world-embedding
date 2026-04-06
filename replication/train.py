"""Training loop for the DSSDE model.

v10 default: single-phase joint training (CPC stop-gradient, VQ); config train end 2011-12-31.
``python -m src.train journal`` (or ``python src/train.py journal``) trains one model per
``expanding_windows.EXPANDING_WINDOWS`` row into ``output_dir/ew_<tag>/``.
Effective rank logged every 5 epochs.
"""

import json
import pathlib
import sys
import time

# Support ``python src/train.py`` (not only ``python -m src.train``).
if __name__ == "__main__" and __package__ is None:
    _root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    import runpy

    runpy.run_module("src.train", run_name="__main__", alter_sys=True)
    raise SystemExit(0)
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import DSSConfig, MODALITY_SPEC, MACRO_FORECAST_TARGETS, OUTPUT_DIR
from .expanding_windows import EXPANDING_WINDOWS
from .model.dssde import DSSDE


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DaySequenceDataset(Dataset):
    """Produces overlapping subsequences from the daily feature matrix.

    Optionally carries aligned ADS targets and macro forecast targets
    for semi-supervised training.
    """

    def __init__(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        col_to_modality: Dict[str, np.ndarray],
        seq_len: int,
        stride: int = 1,
        ads_values: Optional[np.ndarray] = None,
        ads_mask: Optional[np.ndarray] = None,
        macro_values: Optional[np.ndarray] = None,
        macro_mask: Optional[np.ndarray] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.col_to_modality = {
            k: torch.tensor(v, dtype=torch.long) for k, v in col_to_modality.items()
        }
        self.seq_len = seq_len
        self.starts = list(range(0, len(features) - seq_len + 1, stride))

        self.ads_values = (
            torch.tensor(ads_values, dtype=torch.float32) if ads_values is not None else None
        )
        self.ads_mask = (
            torch.tensor(ads_mask, dtype=torch.float32) if ads_mask is not None else None
        )
        self.macro_values = (
            torch.tensor(macro_values, dtype=torch.float32) if macro_values is not None else None
        )
        self.macro_mask = (
            torch.tensor(macro_mask, dtype=torch.float32) if macro_mask is not None else None
        )

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = self.starts[idx]
        e = s + self.seq_len
        x = self.features[s:e]
        m = self.mask[s:e]
        modalities = {k: x[:, v] for k, v in self.col_to_modality.items()}

        ads_v = self.ads_values[s:e] if self.ads_values is not None else torch.zeros(self.seq_len)
        ads_m = self.ads_mask[s:e] if self.ads_mask is not None else torch.zeros(self.seq_len)
        macro_v = (
            self.macro_values[s:e]
            if self.macro_values is not None
            else torch.zeros(self.seq_len, 1)
        )
        macro_m = (
            self.macro_mask[s:e]
            if self.macro_mask is not None
            else torch.zeros(self.seq_len, 1)
        )

        return modalities, x, m, ads_v, ads_m, macro_v, macro_m


def collate_fn(batch):
    modality_keys = batch[0][0].keys()
    mod_inputs = {k: torch.stack([b[0][k] for b in batch]) for k in modality_keys}
    x_full = torch.stack([b[1] for b in batch])
    masks = torch.stack([b[2] for b in batch])
    ads_v = torch.stack([b[3] for b in batch])
    ads_m = torch.stack([b[4] for b in batch])
    macro_v = torch.stack([b[5] for b in batch])
    macro_m = torch.stack([b[6] for b in batch])
    return mod_inputs, x_full, masks, ads_v, ads_m, macro_v, macro_m


def _apply_masked_macro_inputs(
    mod_inputs: Dict[str, torch.Tensor],
    x_full: torch.Tensor,
    meta: dict,
    cfg: DSSConfig,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """MAE-style masking: hide a random subset of macro (fred_) inputs.

    This forces reconstruction / CPC to rely more on non-macro modalities.
    Implemented as input corruption, not label corruption.
    """
    p = float(getattr(cfg, "masked_macro_prob", 0.0))
    frac = float(getattr(cfg, "masked_macro_feature_frac", 0.0))
    if p <= 0 or frac <= 0:
        return mod_inputs, x_full
    if "macro" not in mod_inputs or "macro" not in meta.get("col_to_modality", {}):
        return mod_inputs, x_full

    if torch.rand(()) > p:
        return mod_inputs, x_full

    macro_idx = meta["col_to_modality"]["macro"]
    if len(macro_idx) == 0:
        return mod_inputs, x_full

    macro_dim = mod_inputs["macro"].shape[-1]
    n_mask = max(1, int(round(frac * macro_dim)))
    feat_idx = torch.randperm(macro_dim, device=mod_inputs["macro"].device)[:n_mask]

    mod_inputs = dict(mod_inputs)
    macro_x = mod_inputs["macro"].clone()
    macro_x[..., feat_idx] = 0.0
    mod_inputs["macro"] = macro_x

    # Also mask the corresponding columns in x_full for reconstruction consistency
    x_full = x_full.clone()
    macro_cols = torch.tensor(macro_idx, dtype=torch.long, device=x_full.device)
    x_full.index_copy_(2, macro_cols[feat_idx], x_full[:, :, macro_cols[feat_idx]] * 0.0)

    return mod_inputs, x_full


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _load_ads_aligned(dates: pd.DatetimeIndex, cfg: DSSConfig):
    """Load ADS business conditions index and align to feature dates."""
    ads_path = Path(cfg.ads_path)
    if not ads_path.exists():
        print("  [warn] ADS file not found -- skipping ADS supervision")
        return np.zeros(len(dates)), np.zeros(len(dates))

    ads = pd.read_csv(ads_path, index_col=0, parse_dates=True).squeeze()
    if isinstance(ads, pd.DataFrame):
        ads = ads.iloc[:, -1]
    ads = ads.reindex(dates)

    valid = ~ads.isna()
    values = ads.fillna(0.0).values.astype(np.float32)

    if valid.sum() > 0:
        m, s = values[valid].mean(), values[valid].std()
        if s > 1e-8:
            values = (values - m) / s

    mask_arr = valid.values.astype(np.float32)

    supervision_cutoff = pd.Timestamp(getattr(cfg, "macro_supervision_end", "2005-12-31"))
    mask_arr[dates > supervision_cutoff] = 0.0

    return values, mask_arr


def _load_macro_targets_aligned(dates: pd.DatetimeIndex, cfg: DSSConfig):
    """Load macro targets for forecasting heads and align to feature dates."""
    tgt_path = Path(cfg.targets_path)
    if not tgt_path.exists():
        print("  [warn] Macro targets file not found -- skipping macro supervision")
        return np.zeros((len(dates), len(MACRO_FORECAST_TARGETS))), np.zeros(
            (len(dates), len(MACRO_FORECAST_TARGETS))
        )

    tgt_df = pd.read_csv(tgt_path, index_col=0, parse_dates=True)
    n_tgt = len(MACRO_FORECAST_TARGETS)
    values = np.zeros((len(dates), n_tgt), dtype=np.float32)
    mask = np.zeros((len(dates), n_tgt), dtype=np.float32)

    for i, col in enumerate(MACRO_FORECAST_TARGETS):
        if col not in tgt_df.columns:
            continue
        # Forward-fill raw levels to daily BEFORE computing changes.
        # The CSV has values only on release dates (~1st of month); without
        # this step, pct_change/diff produces all-NaN and the mask stays zero.
        raw = tgt_df[col].ffill(limit=31)
        raw = raw.reindex(dates).ffill(limit=31)
        if col in ("INDPRO", "PAYEMS", "CPIAUCSL"):
            series = raw.pct_change(fill_method=None)
        else:
            series = raw.diff()
        valid = ~series.isna()
        vals = series.fillna(0.0).values.astype(np.float32)
        if valid.sum() > 10:
            m, s = vals[valid].mean(), vals[valid].std()
            if s > 1e-8:
                vals = (vals - m) / s
        values[:, i] = vals
        mask[:, i] = valid.values.astype(np.float32)

    supervision_cutoff = pd.Timestamp(getattr(cfg, "macro_supervision_end", "2005-12-31"))
    mask[dates > supervision_cutoff, :] = 0.0

    return values, mask


def _build_feature_weights(columns: list, cfg: DSSConfig) -> torch.Tensor:
    """Create per-feature reconstruction weights.

    Bybee topic features get lower weight so the reconstruction loss focuses
    on financial and macro features.
    """
    weights = np.full(len(columns), cfg.rec_weight_other, dtype=np.float32)
    for i, c in enumerate(columns):
        if c.startswith("bybee_"):
            weights[i] = cfg.rec_weight_bybee
    return torch.tensor(weights, dtype=torch.float32)


def load_and_prepare(cfg: DSSConfig) -> Tuple[dict, dict]:
    """Load CSV, split by date, build datasets with ADS and macro targets.

    Returns:
        datasets: {"train": Dataset, "val": Dataset, "test": Dataset}
        meta: dict with column info, modality dims, scaler stats
    """
    df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    columns = list(df.columns)

    col_to_modality: Dict[str, list] = {}
    for mod_name, spec in MODALITY_SPEC.items():
        indices = []
        for i, c in enumerate(columns):
            if any(c.startswith(p) for p in spec["prefixes"]):
                indices.append(i)
        if indices:
            col_to_modality[mod_name] = np.array(indices)

    modality_dims = {k: len(v) for k, v in col_to_modality.items()}

    mask_full = (~df.isna()).values.astype(np.float32)
    values_full = df.fillna(0.0).values.astype(np.float32)

    # v5: self-supervised training uses a wider date range than macro supervision
    selfsup_train_end = pd.Timestamp(getattr(cfg, "selfsup_train_end", cfg.train_end))
    norm_end = pd.Timestamp(getattr(cfg, "macro_supervision_end", cfg.train_end))

    val_end = pd.Timestamp(getattr(cfg, "val_end", "2017-12-31"))

    ss_train_mask = df.index <= selfsup_train_end
    ss_val_mask = (df.index > selfsup_train_end) & (df.index <= val_end)

    # Normalization on the macro-supervision period to avoid distributional leakage
    norm_mask = df.index <= norm_end
    norm_vals = values_full[norm_mask]
    norm_msk = mask_full[norm_mask]
    col_mean = np.nanmean(np.where(norm_msk > 0, norm_vals, np.nan), axis=0)
    col_std = np.nanstd(np.where(norm_msk > 0, norm_vals, np.nan), axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    col_std = np.nan_to_num(col_std, nan=1.0)
    col_std = np.where(col_std < 1e-8, 1.0, col_std)

    values_norm = (values_full - col_mean) / col_std
    clip = float(getattr(cfg, "norm_feature_clip", 50.0))
    if clip > 0:
        values_norm = np.clip(values_norm, -clip, clip)

    # Load ADS and macro targets (masks already zeroed after supervision cutoff)
    ads_values, ads_mask = _load_ads_aligned(df.index, cfg)
    macro_values, macro_mask = _load_macro_targets_aligned(df.index, cfg)

    datasets = {}
    for name, bmask, stride in [
        ("train", ss_train_mask, cfg.train_stride),
        ("val", ss_val_mask, cfg.val_stride),
    ]:
        idx = np.where(bmask)[0]
        if len(idx) < cfg.seq_len:
            continue

        ds = DaySequenceDataset(
            features=values_norm[idx],
            mask=mask_full[idx],
            col_to_modality=col_to_modality,
            seq_len=cfg.seq_len,
            stride=stride,
            ads_values=ads_values[idx],
            ads_mask=ads_mask[idx],
            macro_values=macro_values[idx],
            macro_mask=macro_mask[idx],
        )
        datasets[name] = ds

    meta = {
        "columns": columns,
        "modality_dims": modality_dims,
        "col_to_modality": col_to_modality,
        "total_feature_dim": len(columns),
        "col_mean": col_mean,
        "col_std": col_std,
        "dates": df.index,
        "ads_values": ads_values,
        "ads_mask": ads_mask,
    }
    return datasets, meta


def load_and_prepare_phase2(cfg: DSSConfig) -> Tuple[dict, dict]:
    """Load data for Phase 2: train 1985-1999, val 2000-2005.
    Macro/ADS masks active for 1985-2005 (both train and val get labels for validation).
    """
    df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True)
    df = df.sort_index()

    columns = list(df.columns)
    col_to_modality: Dict[str, list] = {}
    for mod_name, spec in MODALITY_SPEC.items():
        indices = [i for i, c in enumerate(columns) if any(c.startswith(p) for p in spec["prefixes"])]
        if indices:
            col_to_modality[mod_name] = np.array(indices)

    modality_dims = {k: len(v) for k, v in col_to_modality.items()}
    mask_full = (~df.isna()).values.astype(np.float32)
    values_full = df.fillna(0.0).values.astype(np.float32)

    norm_end = pd.Timestamp(getattr(cfg, "macro_supervision_end", cfg.train_end))
    norm_mask = df.index <= norm_end
    norm_vals = values_full[norm_mask]
    norm_msk = mask_full[norm_mask]
    col_mean = np.nanmean(np.where(norm_msk > 0, norm_vals, np.nan), axis=0)
    col_std = np.nanstd(np.where(norm_msk > 0, norm_vals, np.nan), axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    col_std = np.nan_to_num(col_std, nan=1.0)
    col_std = np.where(col_std < 1e-8, 1.0, col_std)
    values_norm = (values_full - col_mean) / col_std
    clip = float(getattr(cfg, "norm_feature_clip", 50.0))
    if clip > 0:
        values_norm = np.clip(values_norm, -clip, clip)

    ads_values, ads_mask = _load_ads_aligned(df.index, cfg)
    macro_values, macro_mask = _load_macro_targets_aligned(df.index, cfg)

    phase2_train_end = pd.Timestamp(getattr(cfg, "phase2_train_end", "1999-12-31"))
    phase2_val_start = pd.Timestamp(getattr(cfg, "phase2_val_start", "2000-01-01"))
    phase2_val_end = pd.Timestamp(getattr(cfg, "phase2_val_end", "2005-12-31"))

    train_mask = df.index <= phase2_train_end
    val_mask = (df.index >= phase2_val_start) & (df.index <= phase2_val_end)

    datasets = {}
    for name, bmask, stride in [
        ("train", train_mask, cfg.train_stride),
        ("val", val_mask, cfg.val_stride),
    ]:
        idx = np.where(bmask)[0]
        if len(idx) < cfg.seq_len:
            continue
        ds = DaySequenceDataset(
            features=values_norm[idx],
            mask=mask_full[idx],
            col_to_modality=col_to_modality,
            seq_len=cfg.seq_len,
            stride=stride,
            ads_values=ads_values[idx],
            ads_mask=ads_mask[idx],
            macro_values=macro_values[idx],
            macro_mask=macro_mask[idx],
        )
        datasets[name] = ds

    meta = {
        "columns": columns,
        "modality_dims": modality_dims,
        "col_to_modality": col_to_modality,
        "total_feature_dim": len(columns),
        "col_mean": col_mean,
        "col_std": col_std,
        "dates": df.index,
        "ads_values": ads_values,
        "ads_mask": ads_mask,
    }
    return datasets, meta


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: DSSConfig | None = None):
    if cfg is None:
        cfg = DSSConfig()

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = _get_device()
    print(f"Device: {device}")

    # ---- Data ----
    print("Loading data...")
    datasets, meta = load_and_prepare(cfg)

    feature_weights = _build_feature_weights(meta["columns"], cfg).to(device)

    train_loader = DataLoader(
        datasets["train"], batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Modality dims: {meta['modality_dims']}")
    print(f"Total features: {meta['total_feature_dim']}")
    print(f"Train sequences: {len(datasets['train'])}, Val sequences: {len(datasets.get('val', []))}")

    # ---- Model ----
    model = DSSDE(
        modality_dims=meta["modality_dims"],
        total_feature_dim=meta["total_feature_dim"],
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        n_attention_heads=cfg.n_attention_heads,
        dropout=cfg.encoder_dropout,
        transition_type=cfg.transition_type,
        hierarchical_windows=tuple(getattr(cfg, "hierarchical_windows", [1])),
        use_spectral_norm=getattr(cfg, "use_spectral_norm", False),
        use_vq=cfg.use_vq,
        vq_codebook_size=cfg.vq_codebook_size,
        vq_ema_decay=cfg.vq_ema_decay,
        vq_commitment_weight=cfg.vq_commitment_weight,
        cpc_horizons=tuple(cfg.cpc_horizons),
        cpc_temperature=cfg.cpc_temperature,
        cpc_n_negatives=cfg.cpc_n_negatives,
        alpha_rec=cfg.alpha_rec,
        beta_smo=cfg.beta_smo,
        beta_var=getattr(cfg, "beta_var", 0.0),
        use_vicreg=getattr(cfg, "use_vicreg", False),
        vicreg_var_weight=getattr(cfg, "vicreg_var_weight", 0.0),
        vicreg_cov_weight=getattr(cfg, "vicreg_cov_weight", 0.0),
        vicreg_gamma=getattr(cfg, "vicreg_gamma", 1.0),
        use_hmm_prior=getattr(cfg, "use_hmm_prior", False),
        hmm_n_states=getattr(cfg, "hmm_n_states", 8),
        hmm_weight=getattr(cfg, "hmm_weight", 0.0),
        gamma_vq=cfg.gamma_vq,
        lambda_ads=cfg.lambda_ads,
        lambda_macro=cfg.lambda_macro,
        n_macro_targets=cfg.n_macro_targets,
        macro_horizon=cfg.macro_horizon,
        feature_weights=feature_weights,
        regime_aware_smooth=getattr(cfg, "regime_aware_smooth", False),
        regime_contrastive_weight=getattr(cfg, "regime_contrastive_weight", 0.0),
        regime_contrastive_temp=getattr(cfg, "regime_contrastive_temp", 0.5),
        macro_contrastive_n_bins=getattr(cfg, "macro_contrastive_n_bins", 8),
        smooth_transition_penalty=getattr(cfg, "smooth_transition_penalty", 0.0),
    ).to(device)

    if getattr(cfg, "regime_contrastive_weight", 0.0) > 0:
        model.loss_fn.macro_contrast.set_global_boundaries(
            meta["ads_values"], meta["ads_mask"]
        )
        print(f"  Set global ADS boundaries ({model.loss_fn.macro_contrast.n_bins} bins)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.max_epochs - cfg.warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs],
    )

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = _accumulate_losses()
        for mod_inputs, x_full, mask, ads_v, ads_m, macro_v, macro_m in train_loader:
            mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
            x_full = x_full.to(device)
            mask = mask.to(device)
            ads_v = ads_v.to(device)
            ads_m = ads_m.to(device)
            macro_v = macro_v.to(device)
            macro_m = macro_m.to(device)

            mod_inputs, x_full = _apply_masked_macro_inputs(mod_inputs, x_full, meta, cfg)

            result = model(
                mod_inputs, x_full, mask,
                ads_targets=ads_v, ads_mask=ads_m,
                macro_targets=macro_v, macro_mask=macro_m,
            )
            loss = result["losses"]["total"]

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            _update_losses(train_losses, result["losses"])

        scheduler.step()

        # Validate
        model.eval()
        val_losses = _accumulate_losses()
        with torch.no_grad():
            for mod_inputs, x_full, mask, ads_v, ads_m, macro_v, macro_m in val_loader:
                mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
                x_full = x_full.to(device)
                mask = mask.to(device)
                ads_v = ads_v.to(device)
                ads_m = ads_m.to(device)
                macro_v = macro_v.to(device)
                macro_m = macro_m.to(device)

                result = model(
                    mod_inputs, x_full, mask,
                    ads_targets=ads_v, ads_mask=ads_m,
                    macro_targets=macro_v, macro_mask=macro_m,
                )
                _update_losses(val_losses, result["losses"])

        train_avg = _average_losses(train_losses)
        val_avg = _average_losses(val_losses)

        epoch_time = time.time() - t0
        row = {"epoch": epoch, "time": epoch_time, "lr": scheduler.get_last_lr()[0]}
        row.update({f"train_{k}": v for k, v in train_avg.items()})
        row.update({f"val_{k}": v for k, v in val_avg.items()})
        row["val_selfsup"] = _val_metric_selfsup(val_avg, cfg)

        # Effective rank monitoring every 5 epochs
        eff_rank = float("nan")
        if epoch % 5 == 0 or epoch == 1:
            eff_rank = _compute_effective_rank(model, train_loader, device, max_batches=8)
            row["eff_rank"] = eff_rank

        history.append(row)

        ads_str = f" ads={train_avg.get('ads', 0):.4f}" if train_avg.get("ads", 0) > 0 else ""
        macro_str = f" macro={train_avg.get('macro', 0):.4f}" if train_avg.get("macro", 0) > 0 else ""
        rank_str = f" eff_rank={eff_rank:.2f}" if not np.isnan(eff_rank) else ""

        v_self = row["val_selfsup"]
        print(
            f"Epoch {epoch:3d} | "
            f"train_total={train_avg['total']:.4f} cpc={train_avg['cpc']:.4f} "
            f"rec={train_avg['rec']:.4f}{ads_str}{macro_str} | "
            f"val_selfsup={v_self:.4f} val_total={val_avg['total']:.4f}{rank_str} | "
            f"{epoch_time:.1f}s"
        )

        val_metric = _val_metric_selfsup(val_avg, cfg)
        # Don't save checkpoints or count patience during warmup; LR is still
        # ramping from 1% to 100% and the model hasn't converged enough for the
        # val metric to be meaningful.  This prevents the v10 bug where w3 saved
        # its "best" at epoch 2 (LR = 4e-5, essentially random weights).
        if epoch <= cfg.warmup_epochs:
            if epoch == cfg.warmup_epochs:
                best_val_loss = val_metric
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                print(f"  -> Warmup done. Baseline val_selfsup: {best_val_loss:.4f}")
        elif val_metric < best_val_loss:
            best_val_loss = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> New best val_selfsup: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"Early stopping at epoch {epoch} (patience={cfg.patience})")
                break

    # Save training artifacts
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)

    np.savez(
        out_dir / "norm_stats.npz",
        col_mean=meta["col_mean"],
        col_std=meta["col_std"],
    )

    print(f"\nTraining complete. Best val_selfsup: {best_val_loss:.4f}")
    print(f"Outputs saved to {out_dir}")
    return model, meta, history


def train_journal_expanding_windows(cfg: DSSConfig | None = None) -> None:
    """Train one checkpoint per EXPANDING_WINDOWS cutoff (journal / referee-ready OOS).

    Writes to ``cfg.output_dir / f\"ew_{window_name}\"`` with aligned
    ``train_end``, ``selfsup_train_end``, and ``macro_supervision_end``.
    """
    cfg = cfg or DSSConfig()
    base = Path(cfg.output_dir)
    base.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Journal mode: training one model per expanding-window cutoff")
    print(f"Base output directory: {base}")
    print("=" * 60)
    retrain = bool(getattr(cfg, "journal_retrain_existing", False))
    for _ts, train_end, _te_s, test_end, wname in EXPANDING_WINDOWS:
        sub = replace(
            cfg,
            train_end=train_end,
            selfsup_train_end=train_end,
            macro_supervision_end=test_end,
            val_end=test_end,
            output_dir=str(base / f"ew_{wname}"),
        )
        out_sub = Path(sub.output_dir)
        ckpt = out_sub / "best_model.pt"
        if ckpt.exists() and not retrain:
            print(f"\n>>> Skip {wname} (already have {ckpt.name}; set journal_retrain_existing=True to redo)")
            continue
        print(f"\n>>> Training window {wname} (supervision ends {train_end})")
        train(sub)
    print("\n" + "=" * 60)
    print("All expanding-window models saved under", base)
    print("=" * 60)


# ---------------------------------------------------------------------------
# v6 Two-Phase Training
# ---------------------------------------------------------------------------

def train_phase1(cfg: DSSConfig) -> Tuple[object, dict, list]:
    """Phase 1: Self-supervised pre-train on 1985-2014, val 2015-2017, early stop on val_selfsup."""
    cfg_phase1 = _config_for_phase1(cfg)
    out_dir = Path(cfg_phase1.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg_phase1.seed)
    np.random.seed(cfg_phase1.seed)

    device = _get_device()
    print(f"[Phase 1] Device: {device}")

    print("[Phase 1] Loading data (1985-2014 train, 2015-2017 val)...")
    datasets, meta = load_and_prepare(cfg_phase1)

    feature_weights = _build_feature_weights(meta["columns"], cfg_phase1).to(device)
    train_loader = DataLoader(
        datasets["train"], batch_size=cfg_phase1.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg_phase1.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"[Phase 1] Train sequences: {len(datasets['train'])}, Val: {len(datasets['val'])}")

    model = DSSDE(
        modality_dims=meta["modality_dims"],
        total_feature_dim=meta["total_feature_dim"],
        embed_dim=cfg_phase1.embed_dim,
        hidden_dim=cfg_phase1.hidden_dim,
        n_attention_heads=cfg_phase1.n_attention_heads,
        dropout=cfg_phase1.encoder_dropout,
        transition_type=cfg_phase1.transition_type,
        hierarchical_windows=tuple(getattr(cfg_phase1, "hierarchical_windows", [1])),
        use_spectral_norm=getattr(cfg_phase1, "use_spectral_norm", False),
        use_vq=cfg_phase1.use_vq,
        vq_codebook_size=cfg_phase1.vq_codebook_size,
        vq_ema_decay=cfg_phase1.vq_ema_decay,
        vq_commitment_weight=cfg_phase1.vq_commitment_weight,
        cpc_horizons=tuple(cfg_phase1.cpc_horizons),
        cpc_temperature=cfg_phase1.cpc_temperature,
        cpc_n_negatives=cfg_phase1.cpc_n_negatives,
        alpha_rec=cfg_phase1.alpha_rec,
        beta_smo=cfg_phase1.beta_smo,
        beta_var=getattr(cfg_phase1, "beta_var", 0.0),
        use_vicreg=getattr(cfg_phase1, "use_vicreg", False),
        vicreg_var_weight=getattr(cfg_phase1, "vicreg_var_weight", 0.0),
        vicreg_cov_weight=getattr(cfg_phase1, "vicreg_cov_weight", 0.0),
        vicreg_gamma=getattr(cfg_phase1, "vicreg_gamma", 1.0),
        use_hmm_prior=getattr(cfg_phase1, "use_hmm_prior", False),
        hmm_n_states=getattr(cfg_phase1, "hmm_n_states", 8),
        hmm_weight=getattr(cfg_phase1, "hmm_weight", 0.0),
        gamma_vq=cfg_phase1.gamma_vq,
        lambda_ads=cfg_phase1.lambda_ads,
        lambda_macro=cfg_phase1.lambda_macro,
        n_macro_targets=cfg_phase1.n_macro_targets,
        macro_horizon=cfg_phase1.macro_horizon,
        feature_weights=feature_weights,
        regime_aware_smooth=getattr(cfg_phase1, "regime_aware_smooth", False),
        regime_contrastive_weight=getattr(cfg_phase1, "regime_contrastive_weight", 0.0),
        regime_contrastive_temp=getattr(cfg_phase1, "regime_contrastive_temp", 0.5),
        macro_contrastive_n_bins=getattr(cfg_phase1, "macro_contrastive_n_bins", 8),
        smooth_transition_penalty=getattr(cfg_phase1, "smooth_transition_penalty", 0.0),
    ).to(device)

    if getattr(cfg_phase1, "regime_contrastive_weight", 0.0) > 0:
        model.loss_fn.macro_contrast.set_global_boundaries(
            meta["ads_values"], meta["ads_mask"]
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_phase1.lr, weight_decay=cfg_phase1.weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=cfg_phase1.warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg_phase1.phase1_epochs - cfg_phase1.warmup_epochs,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg_phase1.warmup_epochs],
    )

    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg_phase1.phase1_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = _accumulate_losses()
        for mod_inputs, x_full, mask, ads_v, ads_m, macro_v, macro_m in train_loader:
            mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
            x_full, mask = x_full.to(device), mask.to(device)
            ads_v, ads_m = ads_v.to(device), ads_m.to(device)
            macro_v, macro_m = macro_v.to(device), macro_m.to(device)

            mod_inputs, x_full = _apply_masked_macro_inputs(mod_inputs, x_full, meta, cfg_phase1)

            result = model(mod_inputs, x_full, mask, ads_targets=ads_v, ads_mask=ads_m,
                           macro_targets=macro_v, macro_mask=macro_m)
            loss = result["losses"]["total"]
            optimizer.zero_grad()
            loss.backward()
            if cfg_phase1.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg_phase1.grad_clip)
            optimizer.step()
            _update_losses(train_losses, result["losses"])

        scheduler.step()

        model.eval()
        val_losses = _accumulate_losses()
        with torch.no_grad():
            for mod_inputs, x_full, mask, ads_v, ads_m, macro_v, macro_m in val_loader:
                mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
                x_full, mask = x_full.to(device), mask.to(device)
                ads_v, ads_m = ads_v.to(device), ads_m.to(device)
                macro_v, macro_m = macro_v.to(device), macro_m.to(device)
                result = model(mod_inputs, x_full, mask, ads_targets=ads_v, ads_mask=ads_m,
                              macro_targets=macro_v, macro_mask=macro_m)
                _update_losses(val_losses, result["losses"])

        train_avg = _average_losses(train_losses)
        val_avg = _average_losses(val_losses)
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch, "time": epoch_time, "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
        })

        val_selfsup = (
            val_avg.get("cpc", 0)
            + cfg_phase1.alpha_rec * val_avg.get("rec", 0)
            + cfg_phase1.beta_smo * val_avg.get("smo", 0)
        )
        print(
            f"[Phase 1] Epoch {epoch:3d} | train_total={train_avg['total']:.4f} cpc={train_avg['cpc']:.4f} "
            f"rec={train_avg['rec']:.4f} | val_selfsup={val_selfsup:.4f} | {epoch_time:.1f}s"
        )

        if val_selfsup < best_val_loss:
            best_val_loss = val_selfsup
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "phase1_best.pt")
            print(f"  -> New best val_selfsup: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg_phase1.patience:
                print(f"[Phase 1] Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "phase1_history.csv", index=False)
    print(f"[Phase 1] Complete. Best val_selfsup: {best_val_loss:.4f}")
    return model, meta, history


def train_phase2(cfg: DSSConfig, model: DSSDE, meta: dict) -> Tuple[object, dict, list]:
    """Phase 2: Macro fine-tune on 1985-1999, val 2000-2005, early stop on val_macro."""
    device = next(model.parameters()).device
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[Phase 2] Loading Phase 1 checkpoint...")
    model.load_state_dict(torch.load(out_dir / "phase1_best.pt", map_location=device, weights_only=True))

    print("[Phase 2] Loading data (1985-1999 train, 2000-2005 val)...")
    datasets, meta_p2 = load_and_prepare_phase2(cfg)
    meta = meta_p2 or meta

    train_loader = DataLoader(
        datasets["train"], batch_size=cfg.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, drop_last=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=cfg.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )
    print(f"[Phase 2] Train sequences: {len(datasets['train'])}, Val: {len(datasets['val'])}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr * 0.5, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.phase2_epochs)

    best_val_macro = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, cfg.phase2_epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = _accumulate_losses()
        for mod_inputs, x_full, mask, ads_v, ads_m, macro_v, macro_m in train_loader:
            mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
            x_full, mask = x_full.to(device), mask.to(device)
            ads_v, ads_m = ads_v.to(device), ads_m.to(device)
            macro_v, macro_m = macro_v.to(device), macro_m.to(device)

            mod_inputs, x_full = _apply_masked_macro_inputs(mod_inputs, x_full, meta, cfg)

            result = model(mod_inputs, x_full, mask, ads_targets=ads_v, ads_mask=ads_m,
                           macro_targets=macro_v, macro_mask=macro_m)
            loss = result["losses"]["total"]
            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            _update_losses(train_losses, result["losses"])

        scheduler.step()

        model.eval()
        val_losses = _accumulate_losses()
        with torch.no_grad():
            for mod_inputs, x_full, mask, ads_v, ads_m, macro_v, macro_m in val_loader:
                mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
                x_full, mask = x_full.to(device), mask.to(device)
                ads_v, ads_m = ads_v.to(device), ads_m.to(device)
                macro_v, macro_m = macro_v.to(device), macro_m.to(device)
                result = model(mod_inputs, x_full, mask, ads_targets=ads_v, ads_mask=ads_m,
                              macro_targets=macro_v, macro_mask=macro_m)
                _update_losses(val_losses, result["losses"])

        train_avg = _average_losses(train_losses)
        val_avg = _average_losses(val_losses)
        val_macro = val_avg.get("macro", float("inf"))
        epoch_time = time.time() - t0
        history.append({
            "epoch": epoch, "time": epoch_time, "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
        })

        print(
            f"[Phase 2] Epoch {epoch:3d} | train_total={train_avg['total']:.4f} "
            f"macro={train_avg.get('macro', 0):.4f} | val_macro={val_macro:.4f} | {epoch_time:.1f}s"
        )

        if val_macro < best_val_macro:
            best_val_macro = val_macro
            patience_counter = 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
            print(f"  -> New best val_macro: {best_val_macro:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"[Phase 2] Early stopping at epoch {epoch}")
                break

    pd.DataFrame(history).to_csv(out_dir / "phase2_history.csv", index=False)
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=2, default=str)
    np.savez(out_dir / "norm_stats.npz", col_mean=meta["col_mean"], col_std=meta["col_std"])
    print(f"[Phase 2] Complete. Best val_macro: {best_val_macro:.4f}")
    return model, meta, history


def train_v6(cfg: DSSConfig | None = None):
    """v6: Two-phase training (pre-train + macro fine-tune)."""
    cfg = cfg or DSSConfig()
    print("=" * 60)
    print("DSSDE v6: Two-Phase Training")
    print("=" * 60)
    model, meta, h1 = train_phase1(cfg)
    model, meta, h2 = train_phase2(cfg, model, meta)
    print("\n" + "=" * 60)
    print("v6 Training complete.")
    print(f"Outputs: {Path(cfg.output_dir)}")
    print("=" * 60)
    return model, meta, {"phase1": h1, "phase2": h2}


def _config_for_phase1(cfg: DSSConfig) -> DSSConfig:
    """Return config with phase1 epoch count for early stopping."""
    from copy import copy
    c = copy(cfg)
    c.max_epochs = getattr(cfg, "phase1_epochs", 200)
    return c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_effective_rank(model, loader, device, max_batches: int = 8) -> float:
    """Sample embeddings and compute effective rank of normalized covariance."""
    model.eval()
    z_list = []
    for i, (mod_inputs, x_full, mask, *_rest) in enumerate(loader):
        if i >= max_batches:
            break
        mod_inputs = {k: v.to(device) for k, v in mod_inputs.items()}
        z_seq = model.encoder(mod_inputs)
        z_list.append(z_seq.reshape(-1, z_seq.shape[-1]).cpu().numpy())
    model.train()
    if not z_list:
        return float("nan")
    z = np.concatenate(z_list, axis=0)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    z_norm = z / (norms + 1e-8)
    cov = np.cov(z_norm.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return 1.0
    return float((eigvals.sum() ** 2) / (eigvals ** 2).sum())


def _accumulate_losses():
    return {"sums": {}, "count": 0}


def _update_losses(acc, losses):
    acc["count"] += 1
    for k, v in losses.items():
        val = v.item() if isinstance(v, torch.Tensor) else v
        acc["sums"][k] = acc["sums"].get(k, 0.0) + val


def _average_losses(acc):
    if acc["count"] == 0:
        return {}
    return {k: v / acc["count"] for k, v in acc["sums"].items()}


def _val_metric_selfsup(val_avg: dict, cfg: DSSConfig) -> float:
    """Metric for early stopping / best checkpoint.

    Excludes CPC from the val metric. CPC on out-of-distribution validation
    windows is noisy and dominates the scale (~15 vs rec ~0.3), causing early
    stopping to track CPC noise instead of reconstruction quality.
    v4 (the best-performing version) implicitly had CPC=0 during eval.

    Uses reconstruction + smoothness (+ VQ + ADS + macro if available).
    """
    m = float(cfg.alpha_rec) * float(val_avg.get("rec", 0.0))
    m += float(cfg.beta_smo) * float(val_avg.get("smo", 0.0))
    if cfg.use_vq and cfg.gamma_vq > 0 and "vq" in val_avg:
        v = float(val_avg["vq"])
        if np.isfinite(v):
            m += float(cfg.gamma_vq) * v
    for key, weight_attr in [("ads", "lambda_ads"), ("macro", "lambda_macro")]:
        if key in val_avg:
            v = float(val_avg[key])
            if v > 0 and np.isfinite(v):
                m += float(getattr(cfg, weight_attr, 0.0)) * v
    return m


def train_reference_model(cfg: DSSConfig | None = None) -> None:
    """Train the full-sample reference model used for visualization (ADS plot, t-SNE, trajectory).

    Trained on 1985–2014, validated on 2015–2017 (includes China deval / oil crash for
    strong ADS learning signal).  NOT used for any quantitative OOS statistics; those
    come from the expanding-window models.  Saved to ``cfg.output_dir / reference_model``.
    """
    cfg = cfg or DSSConfig()
    base = Path(cfg.output_dir)
    ref_cfg = replace(
        cfg,
        train_end="2014-12-31",
        selfsup_train_end="2014-12-31",
        macro_supervision_end="2017-12-31",
        val_end="2017-12-31",
        output_dir=str(base / "reference_model"),
    )
    out = Path(ref_cfg.output_dir)
    ckpt = out / "best_model.pt"
    retrain = bool(getattr(cfg, "journal_retrain_existing", False))
    if ckpt.exists() and not retrain:
        print(f"Reference model already trained at {ckpt}. Skipping.")
        return
    print("=" * 60)
    print("Training REFERENCE MODEL for visualization (1985–2014, val 2015–2017)")
    print("=" * 60)
    train(ref_cfg)


def _main_train_cli():
    cfg = DSSConfig()
    if len(sys.argv) > 1 and sys.argv[1] == "journal":
        train_journal_expanding_windows(cfg)
        train_reference_model(cfg)
    elif len(sys.argv) > 1 and sys.argv[1] == "reference":
        train_reference_model(cfg)
    else:
        two_phase_tags = ("v6", "v7", "v8")
        if any(tag in cfg.output_dir for tag in two_phase_tags) and "v9" not in cfg.output_dir:
            train_v6(cfg)
        else:
            train(cfg)


if __name__ == "__main__":
    _main_train_cli()
