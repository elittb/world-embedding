"""Evaluation suite for DSSDE embeddings (v5).

v5 additions:
  - AR(1) and VAR baselines
  - Bybee-only and Market+Macro PCA ablation baselines
  - Post-hoc k-means regime analysis (replaces broken VQ)
  - Fixed MLP probe (proper regularization, skip if n<60)
  - All v4 improvements retained (PCA-Ridge, expanding windows, Pearson corr)
"""

import pathlib
import sys

# Support ``python src/evaluate.py`` (not only ``python -m src.evaluate``).
if __name__ == "__main__" and __package__ is None:
    _root = pathlib.Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    import runpy

    runpy.run_module("src.evaluate", run_name="__main__", alter_sys=True)
    raise SystemExit(0)

import json
import warnings
from dataclasses import fields
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_mutual_info_score,
    mean_squared_error,
    r2_score,
)
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch

from .config import DSSConfig, MODALITY_SPEC, PROCESSED_DIR
from .expanding_windows import EXPANDING_WINDOWS
from .model.dssde import DSSDE
from .train import load_and_prepare, _build_feature_weights

CRISIS_EVENTS = {
    "Black Monday": "1987-10-19",
    "Gulf War": "1990-08-02",
    "Asian Crisis": "1997-10-27",
    "LTCM": "1998-08-17",
    "Dot-Com Bust": "2000-03-10",
    "9/11": "2001-09-11",
    "Lehman Bros": "2008-09-15",
    "Flash Crash": "2010-05-06",
    "Taper Tantrum": "2013-06-19",
    "China Deval": "2015-08-24",
}

NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
]

# Targets that should be transformed to % changes (trending levels)
PCT_CHANGE_TARGETS = {"CPIAUCSL", "INDPRO", "PAYEMS", "GDPC1"}
# Targets that are already stationary or should be used as level changes
LEVEL_CHANGE_TARGETS = {"UNRATE", "FEDFUNDS", "UMCSENT", "BAAFFM", "DTWEXBGS"}
# Binary target
BINARY_TARGETS = {"USREC"}
# Already stationary (spreads)
STATIONARY_TARGETS = {"T10Y2Y"}


def _get_out_dirs(cfg: DSSConfig):
    out = Path(cfg.output_dir)
    return out / "figures", out / "results"


def dss_config_from_json(path: Path) -> DSSConfig:
    """Rehydrate ``DSSConfig`` from a training run's ``config.json``."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    valid = {f.name for f in fields(DSSConfig)}
    kw = {k: v for k, v in raw.items() if k in valid}
    return DSSConfig(**kw)


# ---------------------------------------------------------------------------
# A. Regime Clustering
# ---------------------------------------------------------------------------

def regime_clustering(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    figures_dir: Path,
    regime_labels: Optional[np.ndarray] = None,
    tag: str = "dssde",
):
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running t-SNE on {z.shape[0]} embeddings (d={z.shape[1]})...")
    tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z)

    years = np.array([d.year for d in dates])
    nber = _load_nber_recession(dates)

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=years, cmap="viridis", s=2, alpha=0.5)
    axes[0].set_title(f"{tag} -- Colored by Year")
    plt.colorbar(sc, ax=axes[0], label="Year")

    colors = np.where(nber, "red", "steelblue")
    axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=2, alpha=0.4)
    axes[1].scatter([], [], c="red", label="NBER Recession", s=20)
    axes[1].scatter([], [], c="steelblue", label="Expansion", s=20)
    axes[1].legend()
    axes[1].set_title(f"{tag} -- NBER Recessions")

    if regime_labels is not None:
        sc3 = axes[2].scatter(z_2d[:, 0], z_2d[:, 1], c=regime_labels, cmap="tab20", s=2, alpha=0.5)
        axes[2].set_title(f"{tag} -- VQ Regime Labels")
        plt.colorbar(sc3, ax=axes[2], label="Regime")
        ami = adjusted_mutual_info_score(nber.astype(int), regime_labels)
        print(f"  AMI(NBER, VQ regimes) = {ami:.4f}")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    fig.savefig(figures_dir / f"tsne_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tsne_{tag}.png")


def _load_nber_recession(dates: pd.DatetimeIndex) -> np.ndarray:
    is_recession = np.zeros(len(dates), dtype=bool)
    for start, end in NBER_RECESSIONS:
        is_recession |= (dates >= start) & (dates <= end)
    return is_recession


# ---------------------------------------------------------------------------
# B. Macro Nowcasting (FIXED: monthly aggregation + stationary targets)
# ---------------------------------------------------------------------------

def _make_stationary_targets(targets_df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw macro levels to stationary series for proper evaluation."""
    result = {}
    for col in targets_df.columns:
        s = targets_df[col].dropna()
        if len(s) < 3:
            continue
        if col in PCT_CHANGE_TARGETS:
            result[f"{col}_pctchg"] = s.pct_change() * 100
        elif col in LEVEL_CHANGE_TARGETS:
            result[f"{col}_diff"] = s.diff()
        elif col in BINARY_TARGETS:
            result[f"{col}"] = s
        elif col in STATIONARY_TARGETS:
            result[f"{col}"] = s
        else:
            result[f"{col}_diff"] = s.diff()
    return pd.DataFrame(result)


def _aggregate_to_monthly(z: np.ndarray, dates: pd.DatetimeIndex) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Average daily embeddings within each month, indexed by 1st-of-month
    to align with standard macro data conventions (FRED, JLN, etc.)."""
    z_df = pd.DataFrame(z, index=dates)
    monthly = z_df.resample("MS").mean().dropna()
    return monthly.values, monthly.index


def _fit_ridge_probe(z_train, y_train, z_eval, y_eval, n_pca_components=16):
    """PCA-reduce then fit Ridge with strong regularization.

    Returns:
        r2, corr, rmse, n_comp, y_pred, ridge, (scaler, pca)
    """
    scaler = StandardScaler()
    z_tr = scaler.fit_transform(z_train)
    z_ev = scaler.transform(z_eval)

    n_comp = min(n_pca_components, z_tr.shape[0] // 5, z_tr.shape[1])
    if n_comp < 2:
        n_comp = 2
    pca = PCA(n_components=n_comp, random_state=42)
    z_tr = pca.fit_transform(z_tr)
    z_ev = pca.transform(z_ev)

    ridge = RidgeCV(alphas=np.logspace(0, 7, 50))
    ridge.fit(z_tr, y_train)
    y_pred = ridge.predict(z_ev)

    r2 = r2_score(y_eval, y_pred)
    corr = float(np.corrcoef(y_eval, y_pred)[0, 1]) if len(y_eval) > 2 else 0.0
    rmse = float(np.sqrt(mean_squared_error(y_eval, y_pred)))
    return r2, corr, rmse, n_comp, y_pred, ridge, (scaler, pca)


def _newey_west_var(x: np.ndarray, lag: int) -> float:
    """HAC variance of sample mean using Newey-West (Bartlett kernel)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 5:
        return float("nan")
    x = x - x.mean()
    gamma0 = np.dot(x, x) / T
    var = gamma0
    L = int(max(0, lag))
    for l in range(1, min(L, T - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        gam = np.dot(x[l:], x[:-l]) / T
        var += 2.0 * w * gam
    return var / T


def giacomini_white_test(loss_diff: np.ndarray, lag: int = 3) -> dict:
    """Simple Giacomini-White style test on loss differential.

    Uses HAC t-stat on mean(loss_diff) with Newey-West variance.
    Returns t-stat and (two-sided) normal p-value approximation.
    """
    d = np.asarray(loss_diff, dtype=float)
    d = d[~np.isnan(d)]
    if len(d) < 20:
        return {"t": float("nan"), "p": float("nan"), "n": int(len(d))}
    v = _newey_west_var(d, lag=lag)
    if not np.isfinite(v) or v <= 0:
        return {"t": float("nan"), "p": float("nan"), "n": int(len(d))}
    t = float(d.mean() / np.sqrt(v))
    # Normal approximation (adequate for large T)
    from math import erf, sqrt
    p = float(2.0 * (1.0 - 0.5 * (1.0 + erf(abs(t) / sqrt(2.0)))))
    return {"t": t, "p": p, "n": int(len(d))}


def split_conformal_intervals(y_cal: np.ndarray, yhat_cal: np.ndarray, yhat_test: np.ndarray, alpha: float = 0.1):
    """Split conformal intervals using absolute residual quantile."""
    resid = np.abs(np.asarray(y_cal) - np.asarray(yhat_cal))
    resid = resid[~np.isnan(resid)]
    if len(resid) < 20:
        q = float("nan")
    else:
        q = float(np.quantile(resid, 1.0 - alpha))
    lo = yhat_test - q
    hi = yhat_test + q
    return lo, hi, q


def _fit_mlp_probe(z_train, y_train, z_eval, y_eval, n_pca_components=16):
    """PCA-reduce then fit a small, heavily-regularized MLP."""
    if len(z_train) < 100:  # v6: conservative skip for monthly targets (was 60)
        return float("nan"), float("nan")

    scaler = StandardScaler()
    z_tr = scaler.fit_transform(z_train)
    z_ev = scaler.transform(z_eval)

    n_comp = min(n_pca_components, z_tr.shape[0] // 5, z_tr.shape[1])
    if n_comp < 2:
        n_comp = 2
    pca = PCA(n_components=n_comp, random_state=42)
    z_tr = pca.fit_transform(z_tr)
    z_ev = pca.transform(z_ev)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mlp = MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="relu",
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.3,
            n_iter_no_change=20,
            random_state=42,
            alpha=1.0,
        )
        mlp.fit(z_tr, y_train)
    y_pred = mlp.predict(z_ev)

    r2 = r2_score(y_eval, y_pred)
    corr = float(np.corrcoef(y_eval, y_pred)[0, 1]) if len(y_eval) > 2 else 0.0
    return r2, corr


def nowcasting_probes(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: DSSConfig,
    results_dir: Path,
    tag: str = "dssde",
    z_per_window: Optional[Dict[str, np.ndarray]] = None,
) -> pd.DataFrame:
    """Expanding-window nowcasting probes with PCA reduction.

    Uses 3 expanding windows for robust evaluation:
      w1: train 1985-2000, test 2001-2005
      w2: train 1985-2005, test 2006-2011
      w3: train 1985-2011, test 2012-2017

    For monthly targets: aggregate daily embeddings to monthly averages.
    PCA-reduces to ~16 components before Ridge to avoid n/p ratio issues.
    Reports both R-squared and Pearson correlation.

    If ``z_per_window`` is set (window_name -> daily embedding matrix), the probe
    for each expanding window uses the embedding trained with that window's
    train cutoff (journal / referee-ready).
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    targets_path = Path(cfg.targets_path)
    if not targets_path.exists():
        print(f"  Targets file not found: {targets_path}")
        return pd.DataFrame()

    raw_targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
    stationary = _make_stationary_targets(raw_targets)

    z_monthly, dates_monthly = _aggregate_to_monthly(z, dates)
    z_monthly_per_window: Optional[Dict[str, np.ndarray]] = None
    if z_per_window is not None:
        z_monthly_per_window = {
            wn: _aggregate_to_monthly(zw, dates)[0] for wn, zw in z_per_window.items()
        }

    results = []
    pred_rows = []

    for col in stationary.columns:
        y_series = stationary[col].dropna()
        if len(y_series) < 30:
            continue

        is_daily = len(y_series) > 2000

        if is_daily:
            z_use, d_use = z, dates
        else:
            z_use, d_use = z_monthly, dates_monthly

        common = d_use.intersection(y_series.index)
        if len(common) < 30:
            continue

        z_df = pd.DataFrame(z_use, index=d_use)
        z_aligned = z_df.loc[common].values
        y_aligned = y_series.loc[common].values

        for train_start, train_end_str, test_start, test_end_str, window_name in EXPANDING_WINDOWS:
            if z_per_window is not None:
                zw = z_per_window.get(window_name)
                if zw is None:
                    continue
                if is_daily:
                    z_df_w = pd.DataFrame(zw, index=dates)
                else:
                    zm = z_monthly_per_window[window_name]
                    z_df_w = pd.DataFrame(zm, index=dates_monthly)
                z_aligned_w = z_df_w.loc[common].values
            else:
                z_aligned_w = z_aligned

            train_ok = (
                (common >= train_start) & (common <= train_end_str) & ~np.isnan(y_aligned)
            )
            eval_ok = (
                (common >= test_start) & (common <= test_end_str) & ~np.isnan(y_aligned)
            )

            if train_ok.sum() < 20 or eval_ok.sum() < 5:
                continue

            z_train = z_aligned_w[train_ok]
            z_eval = z_aligned_w[eval_ok]
            y_train = y_aligned[train_ok]
            y_eval = y_aligned[eval_ok]

            r2, corr, rmse, n_comp, y_pred, ridge, (scaler, pca) = _fit_ridge_probe(z_train, y_train, z_eval, y_eval)

            results.append({
                "method": tag,
                "probe": "ridge",
                "target": col,
                "window": window_name,
                "frequency": "daily" if is_daily else "monthly",
                "r2": r2,
                "corr": corr,
                "rmse": rmse,
                "n_train": int(train_ok.sum()),
                "n_eval": int(eval_ok.sum()),
                "n_pca": n_comp,
            })

            # Save per-period predictions for GW + conformal
            d_eval = common[eval_ok]
            for dt, yt, yp in zip(d_eval, y_eval, y_pred):
                pred_rows.append({
                    "method": tag,
                    "target": col,
                    "window": window_name,
                    "frequency": "daily" if is_daily else "monthly",
                    "date": str(pd.Timestamp(dt).date()),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                })

            # Split conformal: reserve last 20% of train as calibration
            n_tr = len(y_train)
            cal_n = max(20, int(round(0.2 * n_tr)))
            if n_tr >= cal_n + 20:
                y_fit_pred = ridge.predict(pca.transform(scaler.transform(z_train)))
                y_cal = y_train[-cal_n:]
                yhat_cal = y_fit_pred[-cal_n:]
                lo, hi, q = split_conformal_intervals(y_cal, yhat_cal, y_pred, alpha=0.1)
                coverage = float(np.mean((y_eval >= lo) & (y_eval <= hi))) if np.isfinite(q) else float("nan")
                avg_width = float(np.nanmean(hi - lo)) if np.isfinite(q) else float("nan")
                results.append({
                    "method": tag,
                    "probe": "ridge_conformal90",
                    "target": col,
                    "window": window_name,
                    "frequency": "daily" if is_daily else "monthly",
                    "r2": r2,
                    "corr": corr,
                    "rmse": rmse,
                    "n_train": int(train_ok.sum()),
                    "n_eval": int(eval_ok.sum()),
                    "n_pca": n_comp,
                    "coverage": coverage,
                    "avg_width": avg_width,
                    "q_abs_resid": q,
                })

            mlp_r2, mlp_corr = _fit_mlp_probe(z_train, y_train, z_eval, y_eval)
            results.append({
                "method": tag,
                "probe": "mlp",
                "target": col,
                "window": window_name,
                "frequency": "daily" if is_daily else "monthly",
                "r2": mlp_r2,
                "corr": mlp_corr,
                "rmse": float("nan"),
                "n_train": int(train_ok.sum()),
                "n_eval": int(eval_ok.sum()),
                "n_pca": n_comp,
            })

    df = pd.DataFrame(results)
    df.to_csv(results_dir / f"nowcasting_{tag}.csv", index=False)
    print(f"  Nowcasting ({tag}) saved.")

    if pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(results_dir / f"nowcasting_preds_{tag}.csv", index=False)

    if len(df) > 0:
        ridge_df = df[df["probe"] == "ridge"]
        summary = ridge_df.pivot_table(
            index="target", columns="window", values=["r2", "corr"], aggfunc="first"
        )
        print(f"\n  {tag} Nowcasting (Ridge, PCA-reduced):")
        print(summary.to_string())
        print()

        mlp_df = df[df["probe"] == "mlp"]
        if len(mlp_df) > 0:
            mlp_summary = mlp_df.pivot_table(
                index="target", columns="window", values="r2", aggfunc="first"
            )
            print(f"  {tag} Nowcasting (MLP probe):")
            print(mlp_summary.to_string())
            print()

    return df


def _gw_from_prediction_files(results_dir: Path) -> pd.DataFrame:
    """Compute GW tests DSSDE vs PCA from saved prediction series."""
    df = gw_tests_from_prediction_tags(
        results_dir, "dssde", "pca", label_a="dssde", label_b="pca"
    )
    if "comparison" in df.columns:
        df = df.drop(columns=["comparison"])
    return df


def gw_tests_from_prediction_tags(
    results_dir: Path,
    tag_a: str,
    tag_b: str,
    label_a: str = "a",
    label_b: str = "b",
) -> pd.DataFrame:
    """Giacomini–White on mean squared loss differential L_a − L_b (paired dates)."""
    pa = results_dir / f"nowcasting_preds_{tag_a}.csv"
    pb = results_dir / f"nowcasting_preds_{tag_b}.csv"
    if not pa.exists() or not pb.exists():
        return pd.DataFrame()

    a = pd.read_csv(pa, parse_dates=["date"])
    b = pd.read_csv(pb, parse_dates=["date"])
    key = ["target", "window", "frequency", "date"]
    m = a.merge(b, on=key, suffixes=(f"_{label_a}", f"_{label_b}"))
    if len(m) == 0:
        return pd.DataFrame()

    col_true_a = f"y_true_{label_a}"
    col_pred_a = f"y_pred_{label_a}"
    col_true_b = f"y_true_{label_b}"
    col_pred_b = f"y_pred_{label_b}"
    out = []
    for (tgt, win, freq), g in m.groupby(["target", "window", "frequency"]):
        la = (g[col_true_a] - g[col_pred_a]) ** 2
        lb = (g[col_true_b] - g[col_pred_b]) ** 2
        diff = (la - lb).to_numpy(dtype=float)
        stat = giacomini_white_test(diff, lag=3)
        out.append({
            "target": tgt,
            "window": win,
            "frequency": freq,
            "comparison": f"{tag_a}_vs_{tag_b}",
            "mean_lossdiff": float(np.nanmean(diff)),
            "gw_t": stat["t"],
            "gw_p": stat["p"],
            "n": stat["n"],
        })
    return pd.DataFrame(out)


def _aggregate_matrix_to_monthly(
    values: np.ndarray, dates: pd.DatetimeIndex
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    feat_df = pd.DataFrame(values, index=dates)
    monthly = feat_df.resample("MS").mean().dropna()
    return monthly.values.astype(np.float32), monthly.index


def dfm_nowcasting_probes(cfg: DSSConfig, results_dir: Path) -> pd.DataFrame:
    """Monthly macro nowcast: PCA factors on standardized features + lagged factors (DFM-style).

    Fits scaler and factor loadings on each expanding window's train period only, then
    stacks F_t, F_{t-1}, ... for a Ridge probe (Stock–Watson–style reduced-form step).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    targets_path = Path(cfg.targets_path)
    if not targets_path.exists():
        return pd.DataFrame()

    n_factors = int(getattr(cfg, "dfm_n_factors", 8))
    n_lags = int(getattr(cfg, "dfm_n_lags", 3))
    n_lags = max(1, n_lags)

    raw_targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
    stationary = _make_stationary_targets(raw_targets)

    df_feat = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True).sort_index()
    values = df_feat.fillna(0.0).values.astype(np.float32)
    dates = df_feat.index
    Xm, dm = _aggregate_matrix_to_monthly(values, dates)

    results = []
    pred_rows = []

    for col in stationary.columns:
        y_series = stationary[col].dropna()
        if len(y_series) > 2000:
            continue

        common = dm.intersection(y_series.index)
        if len(common) < 40:
            continue

        Xm_a = pd.DataFrame(Xm, index=dm).loc[common].values
        y_a = y_series.loc[common].values.astype(np.float64)
        dates_a = common

        for train_start, train_end_str, test_start, test_end_str, window_name in EXPANDING_WINDOWS:
            train_ok = (
                (dates_a >= train_start)
                & (dates_a <= train_end_str)
                & ~np.isnan(y_a)
            )
            eval_ok = (
                (dates_a >= test_start)
                & (dates_a <= test_end_str)
                & ~np.isnan(y_a)
            )
            if train_ok.sum() < 30 or eval_ok.sum() < 5:
                continue

            scaler = StandardScaler()
            scaler.fit(Xm_a[train_ok])
            Xm_n = scaler.transform(Xm_a)
            k = min(n_factors, train_ok.sum() // 5, Xm_n.shape[1])
            k = max(2, k)
            pca = PCA(n_components=k, random_state=42)
            pca.fit(Xm_n[train_ok])
            F = pca.transform(Xm_n)

            L = n_lags
            X_tr_list: List[np.ndarray] = []
            y_tr_list: List[float] = []
            X_ev_list: List[np.ndarray] = []
            y_ev_list: List[float] = []
            d_ev_list: List[pd.Timestamp] = []

            for i in range(L, len(F)):
                feat = np.concatenate([F[i - j] for j in range(L + 1)])
                if bool(train_ok[i]):
                    X_tr_list.append(feat)
                    y_tr_list.append(y_a[i])
                if bool(eval_ok[i]):
                    X_ev_list.append(feat)
                    y_ev_list.append(y_a[i])
                    d_ev_list.append(dates_a[i])

            if len(X_tr_list) < 20 or len(X_ev_list) < 5:
                continue

            X_tr = np.stack(X_tr_list)
            y_tr = np.asarray(y_tr_list)
            X_ev = np.stack(X_ev_list)
            y_ev = np.asarray(y_ev_list)

            ridge = RidgeCV(alphas=np.logspace(0, 7, 50))
            ridge.fit(X_tr, y_tr)
            y_pred = ridge.predict(X_ev)

            r2 = r2_score(y_ev, y_pred)
            corr = float(np.corrcoef(y_ev, y_pred)[0, 1]) if len(y_ev) > 2 else 0.0
            rmse = float(np.sqrt(mean_squared_error(y_ev, y_pred)))

            results.append({
                "method": "dfm_lagged",
                "probe": "ridge",
                "target": col,
                "window": window_name,
                "frequency": "monthly",
                "r2": r2,
                "corr": corr,
                "rmse": rmse,
                "n_train": len(X_tr),
                "n_eval": len(X_ev),
                "n_pca": k,
            })

            for dt, yt, yp in zip(d_ev_list, y_ev, y_pred):
                pred_rows.append({
                    "method": "dfm",
                    "target": col,
                    "window": window_name,
                    "frequency": "monthly",
                    "date": str(pd.Timestamp(dt).date()),
                    "y_true": float(yt),
                    "y_pred": float(yp),
                })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df.to_csv(results_dir / "nowcasting_dfm.csv", index=False)
        print("  DFM (lagged factors) nowcasting saved.")
    if pred_rows:
        pd.DataFrame(pred_rows).to_csv(results_dir / "nowcasting_preds_dfm.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# B2. ADS Daily Benchmark
# ---------------------------------------------------------------------------

def ads_benchmark(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: DSSConfig,
    results_dir: Path,
    tag: str = "dssde",
) -> dict:
    """Correlate daily embeddings with ADS Business Conditions Index."""
    ads_path = Path(cfg.features_path).parent.parent / "raw" / "ads" / "ads_latest.csv"
    if not ads_path.exists():
        print(f"  ADS file not found: {ads_path}")
        return {}

    ads = pd.read_csv(ads_path, index_col=0, parse_dates=True).iloc[:, 0]
    ads.name = "ADS"

    common = dates.intersection(ads.index)
    if len(common) < 100:
        print("  Too few overlapping dates with ADS.")
        return {}

    z_df = pd.DataFrame(z, index=dates)
    z_common = z_df.loc[common].values
    ads_common = ads.loc[common].values

    valid = ~np.isnan(ads_common)
    z_valid = z_common[valid]
    ads_valid = ads_common[valid]

    train_end = pd.Timestamp(cfg.train_end)
    dates_valid = common[valid]
    train_mask = dates_valid <= train_end
    test_mask = dates_valid > train_end

    # Ridge probe from embedding to ADS
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_valid[train_mask])
    z_test = scaler.transform(z_valid[test_mask])

    ridge = RidgeCV(alphas=np.logspace(-3, 5, 30))
    ridge.fit(z_train, ads_valid[train_mask])
    ads_pred = ridge.predict(z_test)
    ads_true = ads_valid[test_mask]

    r2 = r2_score(ads_true, ads_pred)
    corr = np.corrcoef(ads_true, ads_pred)[0, 1]

    result = {
        "method": tag,
        "r2_oos": r2,
        "correlation": corr,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    print(f"  {tag} vs ADS: R2_OOS={r2:.4f}, corr={corr:.4f}")
    return result


# ---------------------------------------------------------------------------
# B3. JLN Monthly Uncertainty Benchmark
# ---------------------------------------------------------------------------

def jln_benchmark(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: DSSConfig,
    results_dir: Path,
    tag: str = "dssde",
) -> dict:
    """Correlate monthly-averaged embeddings with JLN macro uncertainty."""
    jln_path = Path(cfg.features_path).parent.parent / "raw" / "jln_fred" / "JLNUM1M.csv"
    if not jln_path.exists():
        print(f"  JLN file not found: {jln_path}")
        return {}

    jln = pd.read_csv(jln_path, index_col=0, parse_dates=True).iloc[:, 0]
    jln.name = "JLN_macro_unc"

    z_monthly, dates_monthly = _aggregate_to_monthly(z, dates)

    common = dates_monthly.intersection(jln.index)
    if len(common) < 30:
        print("  Too few overlapping months with JLN.")
        return {}

    z_df = pd.DataFrame(z_monthly, index=dates_monthly)
    z_common = z_df.loc[common].values
    jln_common = jln.loc[common].values

    valid = ~np.isnan(jln_common)
    z_valid = z_common[valid]
    jln_valid = jln_common[valid]

    train_end = pd.Timestamp(cfg.train_end)
    dates_valid = common[valid]
    train_mask = dates_valid <= train_end
    test_mask = dates_valid > train_end

    if train_mask.sum() < 20 or test_mask.sum() < 10:
        print("  Not enough data for JLN benchmark.")
        return {}

    scaler = StandardScaler()
    z_train = scaler.fit_transform(z_valid[train_mask])
    z_test = scaler.transform(z_valid[test_mask])

    ridge = RidgeCV(alphas=np.logspace(-3, 5, 30))
    ridge.fit(z_train, jln_valid[train_mask])
    jln_pred = ridge.predict(z_test)
    jln_true = jln_valid[test_mask]

    r2 = r2_score(jln_true, jln_pred)
    corr = np.corrcoef(jln_true, jln_pred)[0, 1]

    result = {
        "method": tag,
        "r2_oos": r2,
        "correlation": corr,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
    }
    print(f"  {tag} vs JLN: R2_OOS={r2:.4f}, corr={corr:.4f}")
    return result


# ---------------------------------------------------------------------------
# C. Semantic Similarity Retrieval
# ---------------------------------------------------------------------------

def semantic_similarity(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    results_dir: Path,
    tag: str = "dssde",
    top_k: int = 10,
    exclusion_window: int = 60,
):
    results_dir.mkdir(parents=True, exist_ok=True)
    z_norm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-10)
    date_strs = dates.strftime("%Y-%m-%d").tolist()

    retrieval_results = {}
    for event_name, event_date in CRISIS_EVENTS.items():
        idx = _find_nearest_date(dates, event_date)
        if idx is None:
            continue

        sims = z_norm @ z_norm[idx]
        for j in range(max(0, idx - exclusion_window), min(len(sims), idx + exclusion_window + 1)):
            sims[j] = -np.inf
        top_idx = np.argsort(sims)[::-1][:top_k]

        neighbors = [{"date": date_strs[ti], "similarity": float(sims[ti])} for ti in top_idx]
        retrieval_results[event_name] = {"anchor_date": event_date, "nearest": neighbors}
        print(f"  {event_name} ({event_date}): top = {neighbors[0]['date']} (sim={neighbors[0]['similarity']:.3f})")

    with open(results_dir / f"similarity_{tag}.json", "w") as f:
        json.dump(retrieval_results, f, indent=2)
    return retrieval_results


def _find_nearest_date(dates: pd.DatetimeIndex, target: str) -> Optional[int]:
    ts = pd.Timestamp(target)
    diffs = np.abs((dates - ts).days)
    best = diffs.argmin()
    return int(best) if diffs[best] <= 5 else None


# ---------------------------------------------------------------------------
# D. PCA Baseline
# ---------------------------------------------------------------------------

def pca_baseline(cfg: DSSConfig, n_components: int = 64) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    print(f"  Computing PCA baseline (d={n_components})...")
    df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True).sort_index()
    values = df.fillna(0.0).values.astype(np.float32)
    dates = df.index

    train_mask = dates <= cfg.train_end
    scaler = StandardScaler()
    scaler.fit(values[train_mask])
    values_norm = scaler.transform(values)

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(values_norm[train_mask])
    z_pca = pca.transform(values_norm)
    print(f"  PCA explained variance (train): {pca.explained_variance_ratio_.sum():.3f}")
    return z_pca, dates


def pca_features_per_window(
    cfg: DSSConfig, n_components: int = 64
) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex]:
    """PCA factors fit **only** on each expanding window's train slice; transform full sample.

    Returns one embedding trajectory per window name (aligned with journal DSSDE checkpoints).
    """
    df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True).sort_index()
    values = df.fillna(0.0).values.astype(np.float32)
    dates = df.index
    z_per: Dict[str, np.ndarray] = {}
    for train_start, train_end_str, _te_s, _te_e, wname in EXPANDING_WINDOWS:
        tr_s = pd.Timestamp(train_start)
        tr_e = pd.Timestamp(train_end_str)
        train_mask = (dates >= tr_s) & (dates <= tr_e)
        if train_mask.sum() < 50:
            continue
        scaler = StandardScaler()
        scaler.fit(values[train_mask])
        values_norm = scaler.transform(values)
        nc = min(n_components, train_mask.sum() // 5, values.shape[1])
        nc = max(2, nc)
        pca = PCA(n_components=nc, random_state=42)
        pca.fit(values_norm[train_mask])
        z_per[wname] = pca.transform(values_norm).astype(np.float32)
        print(
            f"  PCA expanding [{wname}]: {nc} comps, train var={pca.explained_variance_ratio_.sum():.3f}"
        )
    return z_per, dates


def _pca_on_subset(cfg: DSSConfig, col_filter, n_components: int = 64, label: str = "subset"):
    """PCA baseline on a subset of feature columns."""
    df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True).sort_index()
    cols = [c for c in df.columns if col_filter(c)]
    if not cols:
        return None, None
    values = df[cols].fillna(0.0).values.astype(np.float32)
    dates = df.index
    train_mask = dates <= cfg.train_end
    scaler = StandardScaler()
    scaler.fit(values[train_mask])
    values_norm = scaler.transform(values)
    nc = min(n_components, len(cols))
    pca = PCA(n_components=nc, random_state=42)
    pca.fit(values_norm[train_mask])
    z = pca.transform(values_norm)
    print(f"  PCA-{label} ({len(cols)} feats -> {nc} PCs), var={pca.explained_variance_ratio_.sum():.3f}")
    return z, dates


# ---------------------------------------------------------------------------
# D2. AR(1) and VAR Baselines
# ---------------------------------------------------------------------------

def ar1_baseline_probes(cfg: DSSConfig, results_dir: Path) -> pd.DataFrame:
    """AR(1) baseline: y_{t+1} = a + b * y_t, evaluated on expanding windows."""
    results_dir.mkdir(parents=True, exist_ok=True)
    targets_path = Path(cfg.targets_path)
    if not targets_path.exists():
        return pd.DataFrame()

    raw_targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
    stationary = _make_stationary_targets(raw_targets)

    results = []
    for col in stationary.columns:
        y_series = stationary[col].dropna()
        if len(y_series) < 30:
            continue
        is_daily = len(y_series) > 2000
        if is_daily:
            y_lag = y_series.shift(1)
        else:
            y_lag = y_series.shift(1)

        valid = ~(y_series.isna() | y_lag.isna())
        y = y_series[valid].values
        x = y_lag[valid].values.reshape(-1, 1)
        d = y_series[valid].index

        for tr_s, tr_e, te_s, te_e, wname in EXPANDING_WINDOWS:
            tr_ok = (d >= tr_s) & (d <= tr_e)
            ev_ok = (d >= te_s) & (d <= te_e)
            if tr_ok.sum() < 20 or ev_ok.sum() < 5:
                continue
            lr = LinearRegression()
            lr.fit(x[tr_ok], y[tr_ok])
            yp = lr.predict(x[ev_ok])
            yt = y[ev_ok]
            r2 = r2_score(yt, yp)
            corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 2 else 0.0
            results.append({
                "method": "ar1", "probe": "ols", "target": col,
                "window": wname, "frequency": "daily" if is_daily else "monthly",
                "r2": r2, "corr": corr, "n_train": int(tr_ok.sum()),
                "n_eval": int(ev_ok.sum()),
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df.to_csv(results_dir / "nowcasting_ar1.csv", index=False)
        summary = df.pivot_table(index="target", columns="window", values="r2", aggfunc="first")
        print(f"\n  AR(1) Baseline R^2:")
        print(summary.to_string())
        print()
    return df


def var_baseline_probes(cfg: DSSConfig, results_dir: Path) -> pd.DataFrame:
    """Multivariate Ridge-VAR(1) baseline on monthly macro targets."""
    results_dir.mkdir(parents=True, exist_ok=True)
    targets_path = Path(cfg.targets_path)
    if not targets_path.exists():
        return pd.DataFrame()

    raw_targets = pd.read_csv(targets_path, index_col=0, parse_dates=True)
    stationary = _make_stationary_targets(raw_targets)
    monthly_cols = [c for c in stationary.columns if len(stationary[c].dropna()) < 2000 and len(stationary[c].dropna()) >= 30]
    if len(monthly_cols) < 2:
        return pd.DataFrame()

    monthly = stationary[monthly_cols].resample("MS").first().dropna(how="all")
    monthly = monthly.ffill(limit=3)

    results = []
    for col in monthly_cols:
        for tr_s, tr_e, te_s, te_e, wname in EXPANDING_WINDOWS:
            tr_slice = monthly.loc[tr_s:tr_e].dropna(subset=[col])
            te_slice = monthly.loc[te_s:te_e].dropna(subset=[col])
            if len(tr_slice) < 20 or len(te_slice) < 3:
                continue
            x_train = tr_slice.shift(1).dropna()
            common_tr = x_train.index.intersection(tr_slice.index)
            x_train = x_train.loc[common_tr].fillna(0).values
            y_train = tr_slice.loc[common_tr, col].values
            x_test = te_slice.shift(1).dropna()
            common_te = x_test.index.intersection(te_slice.index)
            if len(common_te) < 3:
                continue
            x_test_v = x_test.loc[common_te].fillna(0).values
            y_test = te_slice.loc[common_te, col].values

            ridge = RidgeCV(alphas=np.logspace(0, 6, 30))
            ridge.fit(x_train, y_train)
            yp = ridge.predict(x_test_v)
            r2 = r2_score(y_test, yp)
            corr = float(np.corrcoef(y_test, yp)[0, 1]) if len(y_test) > 2 else 0.0
            results.append({
                "method": "var_ridge", "probe": "ridge", "target": col,
                "window": wname, "frequency": "monthly",
                "r2": r2, "corr": corr, "n_train": len(common_tr),
                "n_eval": len(common_te),
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df.to_csv(results_dir / "nowcasting_var.csv", index=False)
        summary = df.pivot_table(index="target", columns="window", values="r2", aggfunc="first")
        print(f"\n  VAR(Ridge) Baseline R^2:")
        print(summary.to_string())
        print()
    return df


# ---------------------------------------------------------------------------
# D3. Post-hoc Regime Analysis (replaces VQ)
# ---------------------------------------------------------------------------

def posthoc_regime_analysis(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    figures_dir: Path,
    results_dir: Path,
    tag: str = "dssde",
):
    """K-means regime clustering as a post-hoc analysis on embeddings."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    nber = _load_nber_recession(dates)
    regime_results = []

    # Subsample to ~monthly; temporally smooth GRU embeddings produce
    # connected paths instead of dot clouds without subsampling.
    step = 21
    z_sub = z[::step]
    dates_sub = dates[::step]
    nber_sub = nber[::step]
    n_sub = z_sub.shape[0]

    # PCA pre-reduction to 5D before t-SNE: standard practice when intrinsic
    # dimensionality is high (effective rank 10+).  Reduces noise dimensions
    # and lets t-SNE focus on the dominant structure.
    n_pca = min(5, z_sub.shape[1])
    z_pca = PCA(n_components=n_pca).fit_transform(z_sub)

    perp = min(20, n_sub // 5)
    print(f"  Running t-SNE on {n_sub} embeddings (subsampled 1/{step}, "
          f"PCA→{n_pca}D, perplexity={perp}) ...")
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=2000, learning_rate="auto")
    z_2d = tsne.fit_transform(z_pca)

    years = np.array([d.year for d in dates_sub])
    n_plots = 5
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 7))

    sc = axes[0].scatter(z_2d[:, 0], z_2d[:, 1], c=years, cmap="viridis", s=10, alpha=0.7)
    axes[0].set_title(f"{tag} -- Year")
    plt.colorbar(sc, ax=axes[0], label="Year")

    colors = np.where(nber_sub, "red", "steelblue")
    axes[1].scatter(z_2d[:, 0], z_2d[:, 1], c=colors, s=10, alpha=0.6)
    axes[1].scatter([], [], c="red", label="Recession", s=30)
    axes[1].scatter([], [], c="steelblue", label="Expansion", s=30)
    axes[1].legend()
    axes[1].set_title(f"{tag} -- NBER")

    for ax_idx, k in enumerate([2, 4, 8], start=2):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_full = km.fit_predict(z)
        ami = adjusted_mutual_info_score(nber.astype(int), labels_full)
        labels_sub = labels_full[::step]
        sc = axes[ax_idx].scatter(z_2d[:, 0], z_2d[:, 1], c=labels_sub, cmap="tab10", s=10, alpha=0.7)
        axes[ax_idx].set_title(f"{tag} -- k-means k={k} (AMI={ami:.3f})")
        plt.colorbar(sc, ax=axes[ax_idx], label="Regime")
        regime_results.append({"tag": tag, "k": k, "ami_nber": ami})
        print(f"  k-means k={k}: AMI(NBER) = {ami:.4f}")

    plt.tight_layout()
    fig.savefig(figures_dir / f"tsne_regimes_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tsne_regimes_{tag}.png")

    if regime_results:
        pd.DataFrame(regime_results).to_csv(results_dir / f"regime_analysis_{tag}.csv", index=False)
    return regime_results


# ---------------------------------------------------------------------------
# E. Equity Risk Premium Forecasting
# ---------------------------------------------------------------------------

def equity_premium_forecast(
    z: np.ndarray,
    dates: pd.DatetimeIndex,
    cfg: DSSConfig,
    results_dir: Path,
    tag: str = "dssde",
) -> pd.DataFrame:
    results_dir.mkdir(parents=True, exist_ok=True)
    features_df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True).sort_index()

    sp500_col = next((c for c in features_df.columns if "GSPC" in c), None)
    if sp500_col is None:
        print("  S&P 500 returns column not found.")
        return pd.DataFrame()

    sp500 = features_df[sp500_col].reindex(dates)
    train_end = pd.Timestamp(cfg.train_end)

    results = []
    for h, h_name in [(1, "1d"), (5, "1w"), (21, "1m")]:
        y_fwd = sp500.shift(-h)
        valid = ~y_fwd.isna()
        y = y_fwd.values

        train_ok = (dates <= train_end) & valid.values
        test_mask = (dates > train_end) & valid.values

        if train_ok.sum() < 100 or test_mask.sum() < 50:
            continue

        scaler = StandardScaler()
        z_train = scaler.fit_transform(z[train_ok])
        z_test = scaler.transform(z[test_mask])

        ridge = RidgeCV(alphas=np.logspace(-3, 5, 30))
        ridge.fit(z_train, y[train_ok])
        y_pred = ridge.predict(z_test)
        y_true = y[test_mask]

        ss_res = np.sum((y_true - y_pred) ** 2)
        # Welch-Goyal convention: expanding-window historical mean as benchmark
        test_idx = np.where(test_mask)[0]
        y_all = y.copy()
        y_hist_mean = np.array([np.nanmean(y_all[:idx]) for idx in test_idx])
        ss_tot_wg = np.sum((y_true - y_hist_mean) ** 2)
        ss_tot_zero = np.sum(y_true ** 2)
        r2_oos_wg = 1 - ss_res / ss_tot_wg if ss_tot_wg > 0 else np.nan
        r2_oos_zero = 1 - ss_res / ss_tot_zero if ss_tot_zero > 0 else np.nan

        results.append({
            "method": tag, "horizon": h_name,
            "r2_oos": r2_oos_wg, "r2_oos_zero_baseline": r2_oos_zero,
            "n_test": int(test_mask.sum()),
        })
        print(f"  {tag} {h_name} ahead: R2_OOS(WG) = {r2_oos_wg:.4f}, R2_OOS(zero) = {r2_oos_zero:.4f}")

    df = pd.DataFrame(results)
    df.to_csv(results_dir / f"equity_premium_{tag}.csv", index=False)
    return df


# ---------------------------------------------------------------------------
# Embedding extraction (windowed)
# ---------------------------------------------------------------------------

def extract_full_embeddings(
    model: DSSDE,
    cfg: DSSConfig,
    device: torch.device,
    blend_overlap: bool = True,
) -> Tuple[np.ndarray, pd.DatetimeIndex, Optional[np.ndarray]]:
    """Extract embeddings using sliding GRU windows.

    When ``blend_overlap`` is True (default), uses 50% stride and averages
    overlapping positions to reduce boundary artifacts and end-of-sample spikes
    from forward-filling a single last embedding.
    """
    _, meta = load_and_prepare(cfg)

    df = pd.read_csv(cfg.features_path, index_col=0, parse_dates=True).sort_index()
    values = df.fillna(0.0).values.astype(np.float32)
    dates = df.index
    T_total = len(dates)

    clip = float(getattr(cfg, "norm_feature_clip", 50.0))
    values_norm = (values - meta["col_mean"]) / meta["col_std"]
    if clip > 0:
        values_norm = np.clip(values_norm, -clip, clip)

    W = cfg.seq_len
    keep = max(1, W // 2)
    warmup = max(1, W // 4)
    step = max(1, keep // 2) if blend_overlap else keep

    sum_z = np.zeros((T_total, cfg.embed_dim), dtype=np.float64)
    cnt_z = np.zeros(T_total, dtype=np.float64)
    sum_r = np.zeros(T_total, dtype=np.float64)
    cnt_r = np.zeros(T_total, dtype=np.float64)

    model.eval()
    start = 0
    while start < T_total:
        end = min(start + W, T_total)
        if end - start < 32:
            break

        chunk = values_norm[start:end]
        x_tensor = torch.tensor(chunk, dtype=torch.float32).unsqueeze(0).to(device)

        mod_inputs = {}
        for mod_name, indices in meta["col_to_modality"].items():
            idx_t = torch.tensor(indices, dtype=torch.long)
            mod_inputs[mod_name] = x_tensor[:, :, idx_t]

        with torch.no_grad():
            z_seq, z_q, regime_idx = model.embed_and_quantize(mod_inputs)

        z_np = z_seq.squeeze(0).cpu().numpy()
        r_np = regime_idx.squeeze(0).cpu().numpy() if regime_idx is not None else None

        local_len = end - start
        if start == 0:
            keep_start, keep_end = 0, min(keep + warmup, local_len)
        elif end == T_total:
            keep_start, keep_end = warmup, local_len
        else:
            keep_start, keep_end = warmup, min(warmup + keep, local_len)

        keep_end = min(keep_end, local_len)
        global_start = start + keep_start
        global_end = start + keep_end

        sl = slice(global_start, global_end)
        sum_z[sl] += z_np[keep_start:keep_end].astype(np.float64)
        cnt_z[sl] += 1.0
        if r_np is not None:
            sum_r[sl] += r_np[keep_start:keep_end].astype(np.float64)
            cnt_r[sl] += 1.0

        start += step

    cnt_safe = np.maximum(cnt_z, 1.0)[:, None]
    all_z = (sum_z / cnt_safe).astype(np.float32)

    all_regime: Optional[np.ndarray]
    if model.use_vq and cnt_r.max() > 0:
        all_regime = np.round(sum_r / np.maximum(cnt_r, 1.0)).astype(np.int64)
    else:
        all_regime = None

    uncovered = cnt_z <= 0
    if uncovered.any():
        covered_idx = np.where(~uncovered)[0]
        if len(covered_idx) > 0:
            last_valid = int(covered_idx[-1])
            all_z[uncovered] = all_z[last_valid]
            if all_regime is not None:
                all_regime[uncovered] = all_regime[last_valid]

    return all_z, dates, all_regime


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_embedding_trajectory(z: np.ndarray, dates: pd.DatetimeIndex, figures_dir: Path):
    pca_3 = PCA(n_components=3).fit_transform(z)

    fig, axes = plt.subplots(3, 1, figsize=(16, 8), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(dates, pca_3[:, i], linewidth=0.5, color="steelblue")
        for start, end in NBER_RECESSIONS:
            ax.axvspan(start, end, alpha=0.15, color="red")
        ax.set_ylabel(f"PC{i+1}")
        ax.grid(True, alpha=0.3)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0].set_title("DSSDE Embedding Trajectory (top 3 PCs)")
    plt.tight_layout()
    fig.savefig(figures_dir / "embedding_trajectory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved embedding_trajectory.png")


def _plot_ads_comparison(z: np.ndarray, dates: pd.DatetimeIndex, cfg: DSSConfig, figures_dir: Path,
                         model: Optional[object] = None,
                         ew_oos_series: Optional[pd.Series] = None):
    """Plot DSSDE embedding projection onto ADS space.

    Two approaches depending on available data:
    1. ``ew_oos_series`` provided → plot strictly OOS ADS predictions from
       expanding-window models.  Each window's trained ADS head is applied only
       to its test period, giving an honest demonstration of generalisation.
    2. Fallback: model's built-in ADS head or PC1 (less honest for the OOS
       period but still useful).
    """
    ads_path = Path(cfg.features_path).parent.parent / "raw" / "ads" / "ads_latest.csv"
    if not ads_path.exists():
        return

    ads = pd.read_csv(ads_path, index_col=0, parse_dates=True).iloc[:, 0]
    common = dates.intersection(ads.index)
    if len(common) < 100:
        return

    z_df = pd.DataFrame(z, index=dates)
    z_common = z_df.loc[common].values
    ads_vals = ads.loc[common].values
    valid = ~np.isnan(ads_vals)
    z_valid = z_common[valid]
    ads_valid = ads_vals[valid]
    dates_valid = common[valid]

    train_end = pd.Timestamp(cfg.train_end)
    train_mask = dates_valid <= train_end

    # --- In-sample prediction (reference model ADS head or PC1) ---
    if model is not None:
        import torch as _torch
        device = next(model.parameters()).device
        ads_head = model.loss_fn.ads_loss.head
        with _torch.no_grad():
            z_t = _torch.tensor(z_valid, dtype=_torch.float32, device=device)
            pred_raw = ads_head(z_t).squeeze(-1).cpu().numpy()
    else:
        pred_raw = PCA(n_components=1).fit_transform(z_valid).flatten()
        if np.corrcoef(pred_raw[train_mask], ads_valid[train_mask])[0, 1] < 0:
            pred_raw = -pred_raw

    # Standardise to ADS scale (mean/std of train period)
    tr_mean, tr_std = pred_raw[train_mask].mean(), pred_raw[train_mask].std() + 1e-10
    pred_z = (pred_raw - tr_mean) / tr_std
    ads_tr_mean, ads_tr_std = ads_valid[train_mask].mean(), ads_valid[train_mask].std() + 1e-10
    ads_z = (ads_valid - ads_tr_mean) / ads_tr_std
    pred_smooth = pd.Series(pred_z, index=dates_valid).rolling(21, min_periods=1, center=True).mean()

    smooth_train_corr = np.corrcoef(pred_smooth.values[train_mask], ads_z[train_mask])[0, 1]

    # --- Expanding-window OOS predictions ---
    # ew_oos_series is already z-scored per-window using each model's training
    # mean/std, so we standardise ADS the same way (overall mean/std).
    has_oos = ew_oos_series is not None and len(ew_oos_series) > 10
    if has_oos:
        oos_idx = ew_oos_series.index.intersection(ads.index)
        if len(oos_idx) > 10:
            oos_pred = ew_oos_series.loc[oos_idx].values
            oos_ads_raw = ads.loc[oos_idx].values
            oos_valid = ~np.isnan(oos_ads_raw)
            oos_ads_z = (oos_ads_raw[oos_valid] - ads_tr_mean) / ads_tr_std
            oos_dates = oos_idx[oos_valid]
            oos_pred_v = oos_pred[oos_valid]
            oos_smooth = pd.Series(oos_pred_v, index=oos_dates).rolling(21, min_periods=1, center=True).mean()
            oos_ads_smooth = pd.Series(oos_ads_z, index=oos_dates).rolling(21, min_periods=1, center=True).mean()
            oos_corr = np.corrcoef(oos_smooth.values, oos_ads_smooth.values)[0, 1]
        else:
            has_oos = False

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates_valid, ads_z, linewidth=0.8, alpha=0.9, label="ADS Index (std)", color="darkorange")

    if has_oos:
        # Show in-sample only before the first OOS date to avoid overlap
        oos_start = oos_dates[0]
        is_cut = dates_valid <= oos_start
        is_dates = dates_valid[is_cut]
        is_smooth = pred_smooth.values[is_cut]
        ax.plot(is_dates, is_smooth, linewidth=0.9, alpha=0.85, label="DSSDE (in-sample, 21d smooth)", color="steelblue")
        ax.plot(oos_dates, oos_smooth.values, linewidth=0.9, alpha=0.85,
                label="DSSDE (OOS expanding-window)", color="navy")
        title = (f"DSSDE Embedding → ADS  |  In-sample ρ = {smooth_train_corr:.3f},"
                 f"  OOS ρ = {oos_corr:.3f}")
    else:
        is_dates = dates_valid[train_mask]
        is_smooth = pred_smooth.values[train_mask]
        ax.plot(is_dates, is_smooth, linewidth=0.9, alpha=0.85, label="DSSDE (in-sample, 21d smooth)", color="steelblue")
        title = f"DSSDE Embedding → ADS  |  In-sample ρ = {smooth_train_corr:.3f}"

    ax.axvline(train_end, color="gray", linestyle="--", alpha=0.5, label=f"Train end ({cfg.train_end[:4]})")
    for start, end in NBER_RECESSIONS:
        ax.axvspan(start, end, alpha=0.15, color="red")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel("Standardized")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    fig.savefig(figures_dir / "dssde_vs_ads.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    oos_str = f", oos_ρ={oos_corr:.3f}" if has_oos else ""
    print(f"  Saved dssde_vs_ads.png (train_ρ={smooth_train_corr:.3f}{oos_str})")


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def _build_model(cfg: DSSConfig, meta: dict, device: torch.device) -> DSSDE:
    feature_weights = _build_feature_weights(meta["columns"], cfg).to(device)
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
        lambda_ads=getattr(cfg, "lambda_ads", 0.0),
        lambda_macro=getattr(cfg, "lambda_macro", 0.0),
        n_macro_targets=getattr(cfg, "n_macro_targets", 3),
        macro_horizon=getattr(cfg, "macro_horizon", 21),
        feature_weights=feature_weights,
        regime_aware_smooth=getattr(cfg, "regime_aware_smooth", False),
        regime_contrastive_weight=getattr(cfg, "regime_contrastive_weight", 0.0),
        regime_contrastive_temp=getattr(cfg, "regime_contrastive_temp", 0.5),
        macro_contrastive_n_bins=getattr(cfg, "macro_contrastive_n_bins", 8),
        smooth_transition_penalty=getattr(cfg, "smooth_transition_penalty", 0.0),
    ).to(device)
    return model


def modality_ablation(cfg: DSSConfig, results_dir: Path) -> pd.DataFrame:
    """PCA baselines on feature subsets to isolate modality contributions."""
    results_dir.mkdir(parents=True, exist_ok=True)

    subsets = {
        "news_only": lambda c: c.startswith("bybee_"),
        "market_macro": lambda c: any(c.startswith(p) for p in ["mkt_", "cmd_", "fred_", "intl_", "fx_"]),
        "all_except_news": lambda c: not c.startswith("bybee_"),
    }

    all_results = []
    for label, filt in subsets.items():
        print(f"\n  --- Ablation: {label} ---")
        z, dates = _pca_on_subset(cfg, filt, n_components=cfg.embed_dim, label=label)
        if z is None:
            continue
        df = nowcasting_probes(z, dates, cfg, results_dir, tag=f"pca_{label}")
        if len(df) > 0:
            all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(results_dir / "nowcasting_ablation.csv", index=False)
        return combined
    return pd.DataFrame()


def load_journal_z_per_window(
    cfg: DSSConfig,
    device: torch.device,
    blend_overlap: bool = True,
) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex, Dict[str, DSSConfig]]:
    """Load ``ew_<window_name>/best_model.pt`` under ``cfg.output_dir`` for each expanding window."""
    base = Path(cfg.output_dir)
    z_per: Dict[str, np.ndarray] = {}
    cfg_per: Dict[str, DSSConfig] = {}
    dates_ref: Optional[pd.DatetimeIndex] = None

    for _a, _b, _c, _d, wname in EXPANDING_WINDOWS:
        ew_dir = base / f"ew_{wname}"
        cpath = ew_dir / "config.json"
        mpath = ew_dir / "best_model.pt"
        if not cpath.exists() or not mpath.exists():
            print(f"  [journal] Skip {wname}: missing {ew_dir}")
            continue
        sub_cfg = dss_config_from_json(cpath)
        cfg_per[wname] = sub_cfg
        _, meta = load_and_prepare(sub_cfg)
        model = _build_model(sub_cfg, meta, device)
        model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
        z, dates, _reg = extract_full_embeddings(
            model, sub_cfg, device, blend_overlap=blend_overlap
        )
        if dates_ref is None:
            dates_ref = dates
        elif len(dates_ref) != len(dates) or not dates_ref.equals(dates):
            raise ValueError("Journal checkpoints disagree on feature date index.")
        z_per[wname] = z
        print(f"  [journal] {wname}: z shape {z.shape}, train_end={sub_cfg.train_end}")

    if dates_ref is None or not z_per:
        raise FileNotFoundError(
            f"No journal checkpoints under {base} (expected ew_*/best_model.pt)."
        )
    return z_per, dates_ref, cfg_per


def _compute_ew_oos_ads_predictions(
    cfg: DSSConfig,
    device: torch.device,
) -> Optional[pd.Series]:
    """Build strictly OOS ADS predictions by applying each EW model's ADS head
    to its test-period embeddings only.

    Each window's predictions are standardised using the *training-period*
    mean/std of that window's ADS head output to ensure consistent scale across
    windows.  Returns a pd.Series indexed by date.
    """
    base = Path(cfg.output_dir)
    pieces = []
    for _a, _b, test_start, test_end, wname in EXPANDING_WINDOWS:
        ew_dir = base / f"ew_{wname}"
        cpath = ew_dir / "config.json"
        mpath = ew_dir / "best_model.pt"
        if not cpath.exists() or not mpath.exists():
            continue
        sub_cfg = dss_config_from_json(cpath)
        _, meta = load_and_prepare(sub_cfg)
        model = _build_model(sub_cfg, meta, device)
        model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
        model.eval()
        z, dates, _ = extract_full_embeddings(model, sub_cfg, device, blend_overlap=True)

        with torch.no_grad():
            z_t = torch.tensor(z, dtype=torch.float32, device=device)
            pred_all = model.loss_fn.ads_loss.head(z_t).squeeze(-1).cpu().numpy()

        tr_mask = dates <= sub_cfg.train_end
        te_mask = (dates >= test_start) & (dates <= test_end)
        tr_mean = pred_all[tr_mask].mean()
        tr_std = pred_all[tr_mask].std() + 1e-10
        pred_z = (pred_all - tr_mean) / tr_std

        d_te = dates[te_mask]
        pred_te_z = pred_z[te_mask]

        # Compute per-window OOS correlation with actual ADS
        ads_path = Path(cfg.features_path).parent.parent / "raw" / "ads" / "ads_latest.csv"
        if ads_path.exists():
            ads_raw = pd.read_csv(ads_path, index_col=0, parse_dates=True).iloc[:, 0]
            ads_te = ads_raw.reindex(d_te)
            v = ~ads_te.isna()
            if v.sum() > 10:
                sm_pred = pd.Series(pred_te_z, index=d_te).rolling(21, min_periods=1).mean().values
                sm_ads = pd.Series(ads_te.values).rolling(21, min_periods=1).mean().values
                oos_r = np.corrcoef(sm_pred[v], sm_ads[v])[0, 1]
            else:
                oos_r = float("nan")
        else:
            oos_r = float("nan")

        pieces.append(pd.Series(pred_te_z, index=d_te, name=wname))
        print(f"  [ew-oos-ads] {wname}: {len(d_te)} OOS predictions ({test_start} → {test_end}), ρ={oos_r:.3f}")
    if not pieces:
        return None
    combined = pd.concat(pieces)
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined.sort_index()


def load_reference_model_z(
    cfg: DSSConfig,
    device: torch.device,
    blend_overlap: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[pd.DatetimeIndex], Optional[DSSConfig]]:
    """Load the full-sample reference model (reference_model/ subdirectory).

    This model is trained on 1985–2014 and validated on 2015–2017, giving it
    access to crisis + recovery variance for proper convergence.  Used
    exclusively for visualization (ADS plot, t-SNE, trajectory, embeddings.npz).
    Returns (None, None, None) if the model has not been trained yet.
    """
    ref_dir = Path(cfg.output_dir) / "reference_model"
    cpath = ref_dir / "config.json"
    mpath = ref_dir / "best_model.pt"
    if not cpath.exists() or not mpath.exists():
        print(f"  [ref] Reference model not found at {ref_dir}. Falling back to w3.")
        return None, None, None
    ref_cfg = dss_config_from_json(cpath)
    _, meta = load_and_prepare(ref_cfg)
    model = _build_model(ref_cfg, meta, device)
    model.load_state_dict(torch.load(mpath, map_location=device, weights_only=True))
    z, dates, _ = extract_full_embeddings(model, ref_cfg, device, blend_overlap=blend_overlap)
    epoch_best = "?"
    warmup = getattr(ref_cfg, "warmup_epochs", 15)
    hist_path = ref_dir / "training_history.csv"
    if hist_path.exists():
        import csv
        with open(hist_path) as f:
            rows = list(csv.DictReader(f))
        if rows:
            post_warmup = [r for r in rows if int(r.get("epoch", 0)) >= warmup]
            if post_warmup:
                best_row = min(post_warmup, key=lambda r: float(r.get("val_selfsup", 1e9)))
            else:
                best_row = min(rows, key=lambda r: float(r.get("val_selfsup", 1e9)))
            epoch_best = best_row.get("epoch", "?")
    print(f"  [ref] Reference model loaded from {ref_dir} (best epoch {epoch_best}), z shape {z.shape}")
    return z, dates, ref_cfg


def run_full_evaluation_journal(cfg: DSSConfig | None = None):
    """Evaluation aligned with expanding-window **retrained** models (referee-ready).

    Quantitative statistics (nowcasting, ADS/JLN benchmarks, equity premium,
    regime AMI) use the expanding-window models so all OOS periods are strictly
    held-out.

    Visualizations (ADS comparison plot, embedding trajectory, t-SNE) use the
    full-sample *reference model* (trained 1985-2014, val 2015-2017), which
    benefits from crisis variance in its validation set and converges over many
    more epochs, reproducing the visual quality of v4.
    """
    if cfg is None:
        cfg = DSSConfig()

    figures_dir, results_dir = _get_out_dirs(cfg)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    )
    out_dir = Path(cfg.output_dir)

    print("\n" + "=" * 60)
    print("JOURNAL EVAL: expanding-window DSSDE + aligned PCA + DFM")
    print("=" * 60)

    z_per, dates, cfg_per = load_journal_z_per_window(cfg, device)
    z_ref_ew = None
    cfg_ref_ew = cfg
    w_last = EXPANDING_WINDOWS[-1][4]
    for _a, _b, _c, _d, wn in reversed(EXPANDING_WINDOWS):
        if wn in z_per:
            z_ref_ew = z_per[wn]
            cfg_ref_ew = cfg_per.get(wn, cfg)
            w_last = wn
            break
    assert z_ref_ew is not None

    # Load reference model for visualization (falls back to w3 if not yet trained)
    z_viz, dates_viz, cfg_viz = load_reference_model_z(cfg, device)
    if z_viz is None:
        z_viz, dates_viz, cfg_viz = z_ref_ew, dates, cfg_ref_ew
        print("  [ref] Using w3 embeddings for visualization (reference model not trained).")
    else:
        print("  [ref] Using REFERENCE MODEL embeddings for visualization figures.")

    # Use expanding-window z_ref for OOS quantitative tests; viz model for figures
    z_ref = z_ref_ew
    cfg_ref = cfg_ref_ew

    np.savez(
        out_dir / "embeddings.npz",
        z=z_ref_ew,
        dates=dates.strftime("%Y-%m-%d"),
    )

    print("\n" + "=" * 60)
    print("PCA (per expanding-window train slice)")
    print("=" * 60)
    z_pca_per, dates_pca = pca_features_per_window(cfg, n_components=cfg.embed_dim)
    z_pca_ref = z_pca_per.get(w_last, next(iter(z_pca_per.values())))
    if w_last not in z_pca_per:
        print(f"  [warn] PCA map missing {w_last}; using fallback reference trajectory.")

    z_bybee, dates_bybee = _pca_on_subset(
        cfg_viz,
        lambda c: c.startswith("bybee_"),
        n_components=cfg.embed_dim,
        label="bybee_only",
    )

    print("\n" + "=" * 60)
    print("A. POST-HOC REGIME ANALYSIS  [reference model, full-sample]")
    print("=" * 60)
    posthoc_regime_analysis(z_viz, dates_viz, figures_dir, results_dir, tag="dssde")
    z_pca_single, dates_pca_single = pca_baseline(cfg_viz, n_components=cfg.embed_dim)
    posthoc_regime_analysis(z_pca_single, dates_pca_single, figures_dir, results_dir, tag="pca")

    print("\n" + "=" * 60)
    print("B. MACRO NOWCASTING  [expanding-window OOS]")
    print("=" * 60)
    nc_dssde = nowcasting_probes(
        z_ref, dates, cfg, results_dir, tag="dssde", z_per_window=z_per
    )
    nc_pca = nowcasting_probes(
        z_pca_ref, dates_pca, cfg, results_dir, tag="pca", z_per_window=z_pca_per
    )
    nc_bybee = pd.DataFrame()
    if z_bybee is not None:
        nc_bybee = nowcasting_probes(z_bybee, dates_bybee, cfg_viz, results_dir, tag="pca_bybee")

    all_nc = [df for df in [nc_dssde, nc_pca, nc_bybee] if len(df) > 0]

    print("\n" + "=" * 60)
    print("B2. DFM (lagged PCA factors) + AR(1) + VAR")
    print("=" * 60)
    nc_dfm = dfm_nowcasting_probes(cfg, results_dir)
    if len(nc_dfm) > 0:
        all_nc.append(nc_dfm)

    nc_ar1 = ar1_baseline_probes(cfg, results_dir)
    nc_var = var_baseline_probes(cfg, results_dir)
    for df in [nc_ar1, nc_var]:
        if len(df) > 0:
            all_nc.append(df)

    if all_nc:
        pd.concat(all_nc, ignore_index=True).to_csv(
            results_dir / "nowcasting_comparison.csv", index=False
        )

    print("\n" + "=" * 60)
    print("B5. GIACOMINI–WHITE TESTS")
    print("=" * 60)
    gw_pca = _gw_from_prediction_files(results_dir)
    if len(gw_pca) > 0:
        gw_pca.to_csv(results_dir / "gw_tests_dssde_vs_pca.csv", index=False)
        print("  Saved gw_tests_dssde_vs_pca.csv")
    gw_dfm = gw_tests_from_prediction_tags(
        results_dir, "dssde", "dfm", label_a="dssde", label_b="dfm"
    )
    if len(gw_dfm) > 0:
        gw_dfm.to_csv(results_dir / "gw_tests_dssde_vs_dfm.csv", index=False)
        print("  Saved gw_tests_dssde_vs_dfm.csv")

    print("\n" + "=" * 60)
    print("B3. ADS DAILY BENCHMARK  [reference model, full-sample]")
    print("=" * 60)
    ads_dssde = ads_benchmark(z_viz, dates_viz, cfg_viz, results_dir, tag="dssde")
    ads_pca = ads_benchmark(z_pca_single, dates_pca_single, cfg_viz, results_dir, tag="pca")
    if ads_dssde or ads_pca:
        pd.DataFrame([r for r in [ads_dssde, ads_pca] if r]).to_csv(
            results_dir / "ads_benchmark.csv", index=False
        )

    print("\n" + "=" * 60)
    print("B4. JLN MONTHLY UNCERTAINTY BENCHMARK  [reference model]")
    print("=" * 60)
    jln_dssde = jln_benchmark(z_viz, dates_viz, cfg_viz, results_dir, tag="dssde")
    jln_pca = jln_benchmark(z_pca_single, dates_pca_single, cfg_viz, results_dir, tag="pca")
    if jln_dssde or jln_pca:
        pd.DataFrame([r for r in [jln_dssde, jln_pca] if r]).to_csv(
            results_dir / "jln_benchmark.csv", index=False
        )

    print("\n" + "=" * 60)
    print("C. SEMANTIC SIMILARITY  [reference model]")
    print("=" * 60)
    semantic_similarity(z_viz, dates_viz, results_dir, tag="dssde", exclusion_window=60)
    semantic_similarity(z_pca_single, dates_pca_single, results_dir, tag="pca", exclusion_window=60)

    print("\n" + "=" * 60)
    print("D. EQUITY RISK PREMIUM  [reference model]")
    print("=" * 60)
    ep_dssde = equity_premium_forecast(z_viz, dates_viz, cfg_viz, results_dir, tag="dssde")
    ep_pca = equity_premium_forecast(z_pca_single, dates_pca_single, cfg_viz, results_dir, tag="pca")
    if len(ep_dssde) > 0 and len(ep_pca) > 0:
        pd.concat([ep_dssde, ep_pca], ignore_index=True).to_csv(
            results_dir / "equity_premium_comparison.csv", index=False
        )

    print("\n" + "=" * 60)
    print("E. MODALITY ABLATION")
    print("=" * 60)
    modality_ablation(cfg_viz, results_dir)

    print("\n" + "=" * 60)
    print("GENERATING PLOTS  [reference model, full-sample]")
    print("=" * 60)
    # Save full-sample reference embeddings
    np.savez(
        out_dir / "embeddings_reference.npz",
        z=z_viz,
        dates=dates_viz.strftime("%Y-%m-%d"),
    )
    _plot_embedding_trajectory(z_viz, dates_viz, figures_dir)

    # Load the reference model to use its trained ADS head for the figure
    ref_model = None
    ref_dir = Path(cfg.output_dir) / "reference_model"
    ref_mpath = ref_dir / "best_model.pt"
    if ref_mpath.exists():
        try:
            _, meta_ref = load_and_prepare(cfg_viz)
            ref_model = _build_model(cfg_viz, meta_ref, device)
            ref_model.load_state_dict(torch.load(ref_mpath, map_location=device, weights_only=True))
            ref_model.eval()
        except Exception as e:
            print(f"  [warn] Could not load reference model for ADS plot: {e}")

    # Compute strictly OOS ADS predictions from expanding-window models
    print("  Computing expanding-window OOS ADS predictions ...")
    ew_oos_ads = _compute_ew_oos_ads_predictions(cfg, device)
    _plot_ads_comparison(z_viz, dates_viz, cfg_viz, figures_dir, model=ref_model, ew_oos_series=ew_oos_ads)

    print("\n" + "=" * 60)
    print("JOURNAL EVALUATION COMPLETE")
    print(f"Figures: {figures_dir}")
    print(f"Results: {results_dir}")
    print("=" * 60)


def run_full_evaluation(cfg: DSSConfig | None = None):
    if cfg is None:
        cfg = DSSConfig()

    figures_dir, results_dir = _get_out_dirs(cfg)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"))
    out_dir = Path(cfg.output_dir)

    model_path = out_dir / "best_model.pt"
    if not model_path.exists():
        phase1_path = out_dir / "phase1_best.pt"
        if phase1_path.exists():
            model_path = phase1_path
            print(f"Using Phase 1 checkpoint (best_model.pt not found).\n")
        else:
            print(f"No trained model at {model_path}. Run training first.")
            return

    _, meta = load_and_prepare(cfg)
    model = _build_model(cfg, meta, device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("Loaded trained model.\n")

    # --- Extract embeddings ---
    print("=" * 60)
    print("EXTRACTING DSSDE EMBEDDINGS")
    print("=" * 60)
    z_dssde, dates, regime_labels = extract_full_embeddings(
        model, cfg, device, blend_overlap=True
    )
    np.savez(out_dir / "embeddings.npz", z=z_dssde, dates=dates.strftime("%Y-%m-%d"))

    # --- PCA baselines (all features + Bybee-only) ---
    print("\n" + "=" * 60)
    print("PCA BASELINES")
    print("=" * 60)
    z_pca, dates_pca = pca_baseline(cfg, n_components=cfg.embed_dim)
    z_bybee, dates_bybee = _pca_on_subset(
        cfg, lambda c: c.startswith("bybee_"), n_components=cfg.embed_dim, label="bybee_only"
    )

    # --- A. Post-hoc regime clustering (replaces VQ) ---
    print("\n" + "=" * 60)
    print("A. POST-HOC REGIME ANALYSIS (k-means)")
    print("=" * 60)
    posthoc_regime_analysis(z_dssde, dates, figures_dir, results_dir, tag="dssde")
    posthoc_regime_analysis(z_pca, dates_pca, figures_dir, results_dir, tag="pca")

    # --- B. Macro nowcasting: DSSDE / PCA / Bybee-PCA ---
    print("\n" + "=" * 60)
    print("B. MACRO NOWCASTING (PCA-reduced, expanding-window, Ridge+MLP)")
    print("=" * 60)
    nc_dssde = nowcasting_probes(z_dssde, dates, cfg, results_dir, tag="dssde")
    nc_pca = nowcasting_probes(z_pca, dates_pca, cfg, results_dir, tag="pca")
    nc_bybee = pd.DataFrame()
    if z_bybee is not None:
        nc_bybee = nowcasting_probes(z_bybee, dates_bybee, cfg, results_dir, tag="pca_bybee")

    all_nc = [df for df in [nc_dssde, nc_pca, nc_bybee] if len(df) > 0]

    # --- B-extra. AR(1) and VAR baselines ---
    print("\n" + "=" * 60)
    print("B2. AR(1) AND VAR BASELINES")
    print("=" * 60)
    nc_ar1 = ar1_baseline_probes(cfg, results_dir)
    nc_var = var_baseline_probes(cfg, results_dir)
    for df in [nc_ar1, nc_var]:
        if len(df) > 0:
            all_nc.append(df)

    if all_nc:
        combined = pd.concat(all_nc, ignore_index=True)
        combined.to_csv(results_dir / "nowcasting_comparison.csv", index=False)

    # --- B5. GW tests (DSSDE vs PCA) + saved to CSV ---
    print("\n" + "=" * 60)
    print("B5. FORECAST COMPARISON (GW TESTS) + CONFORMAL OUTPUTS")
    print("=" * 60)
    gw_df = _gw_from_prediction_files(results_dir)
    if len(gw_df) > 0:
        gw_df.to_csv(results_dir / "gw_tests_dssde_vs_pca.csv", index=False)
        print("  Saved gw_tests_dssde_vs_pca.csv")

    # --- B3. ADS benchmark ---
    print("\n" + "=" * 60)
    print("B3. ADS DAILY BENCHMARK")
    print("=" * 60)
    ads_dssde = ads_benchmark(z_dssde, dates, cfg, results_dir, tag="dssde")
    ads_pca = ads_benchmark(z_pca, dates_pca, cfg, results_dir, tag="pca")
    if ads_dssde or ads_pca:
        ads_df = pd.DataFrame([r for r in [ads_dssde, ads_pca] if r])
        ads_df.to_csv(results_dir / "ads_benchmark.csv", index=False)

    # --- B4. JLN benchmark ---
    print("\n" + "=" * 60)
    print("B4. JLN MONTHLY UNCERTAINTY BENCHMARK")
    print("=" * 60)
    jln_dssde = jln_benchmark(z_dssde, dates, cfg, results_dir, tag="dssde")
    jln_pca = jln_benchmark(z_pca, dates_pca, cfg, results_dir, tag="pca")
    if jln_dssde or jln_pca:
        jln_df = pd.DataFrame([r for r in [jln_dssde, jln_pca] if r])
        jln_df.to_csv(results_dir / "jln_benchmark.csv", index=False)

    # --- C. Semantic similarity ---
    print("\n" + "=" * 60)
    print("C. SEMANTIC SIMILARITY RETRIEVAL (60-day exclusion)")
    print("=" * 60)
    semantic_similarity(z_dssde, dates, results_dir, tag="dssde", exclusion_window=60)
    semantic_similarity(z_pca, dates_pca, results_dir, tag="pca", exclusion_window=60)

    # --- D. Equity premium ---
    print("\n" + "=" * 60)
    print("D. EQUITY RISK PREMIUM FORECASTING")
    print("=" * 60)
    ep_dssde = equity_premium_forecast(z_dssde, dates, cfg, results_dir, tag="dssde")
    ep_pca = equity_premium_forecast(z_pca, dates_pca, cfg, results_dir, tag="pca")
    if len(ep_dssde) > 0 and len(ep_pca) > 0:
        pd.concat([ep_dssde, ep_pca], ignore_index=True).to_csv(
            results_dir / "equity_premium_comparison.csv", index=False
        )

    # --- E. Modality ablation ---
    print("\n" + "=" * 60)
    print("E. MODALITY ABLATION")
    print("=" * 60)
    modality_ablation(cfg, results_dir)

    # --- Plots ---
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    _plot_embedding_trajectory(z_dssde, dates, figures_dir)
    _plot_ads_comparison(z_dssde, dates, cfg, figures_dir)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Figures: {figures_dir}")
    print(f"Results: {results_dir}")
    print("=" * 60)


def _journal_ckpts_ready(cfg: DSSConfig) -> bool:
    base = Path(cfg.output_dir)
    return all((base / f"ew_{wn}" / "best_model.pt").exists() for *_, wn in EXPANDING_WINDOWS)


def _main_eval_cli():
    cfg = DSSConfig()
    if len(sys.argv) > 1 and sys.argv[1] == "standard":
        run_full_evaluation(cfg)
    elif len(sys.argv) > 1 and sys.argv[1] == "journal":
        run_full_evaluation_journal(cfg)
    elif getattr(cfg, "use_expanding_window_eval", False):
        if not _journal_ckpts_ready(cfg):
            print(
                "use_expanding_window_eval=True but journal checkpoints are missing.\n"
                f"Expected under {cfg.output_dir}/ew_*/best_model.pt\n"
                "Train them with:  python -m src.train journal  (or: python src/train.py journal)\n"
                "Or run standard eval:  python -m src.evaluate standard  (or: python src/evaluate.py standard)\n"
                "Or set use_expanding_window_eval=False in DSSConfig for single-checkpoint eval."
            )
            sys.exit(1)
        run_full_evaluation_journal(cfg)
    else:
        run_full_evaluation(cfg)


if __name__ == "__main__":
    _main_eval_cli()
