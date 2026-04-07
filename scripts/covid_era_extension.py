"""
Covid-Era Embedding Extension: Simulation and Validation
=========================================================

Pipeline:
  1. Download / assemble real data (Jan 2018 - Jun 2021) for all non-news modalities
  2. Simulate 180 news-topic dimensions via conditional factor model (PCA + Ridge)
  3. Concatenate with original 1985-2017 panel → combined CSV
  4. Load trained DSSDE reference model and generate embeddings for the full span
  5. Validate the extension-period embeddings (regime recognition, ADS, conditions)

Window:  data assembly  → Jan 2018 - Jun 2021  (bridge + Covid + recovery)
         analysis focus → Jan 2019 - Jun 2021  (pre-Covid normality + crisis + recovery)

Usage:
    python scripts/covid_era_extension.py              # full pipeline
    python scripts/covid_era_extension.py --skip-download  # reuse cached downloads
"""

import argparse
import json
import sys
import warnings
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR = ROOT / "output" / "v14" / "covid_extension"
FIG_DIR = ROOT / "paper_draft" / "results" / "figures"
REF_MODEL = ROOT / "output" / "v14" / "reference_model"

EXT_START = "2017-12-01"  # overlap for log-return calc
EXT_END = "2021-07-01"
ANALYSIS_START = "2019-01-01"
ANALYSIS_END = "2021-06-30"

NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
    ("2020-02-01", "2020-04-01"),  # Covid
]

N_MC = 50  # Monte Carlo paths for news simulation
N_NEWS_FACTORS = 30  # PCA factors for news topic model
RIDGE_ALPHA = 100.0  # regularisation for conditional model


# ═══════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD EXTENDED DATA
# ═══════════════════════════════════════════════════════════════════════

def _download_yahoo(tickers: dict, start: str, end: str) -> pd.DataFrame:
    """Download close prices from Yahoo Finance, return DataFrame."""
    import yfinance as yf
    all_close = []
    for ticker, desc in tickers.items():
        try:
            data = yf.download(ticker, start=start, end=end,
                               progress=False, auto_adjust=True)
            if len(data) > 0:
                close = data["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.name = ticker
                all_close.append(close)
                print(f"    {ticker}: {len(data)} days")
        except Exception as e:
            print(f"    {ticker}: ERROR {e}")
    if not all_close:
        return pd.DataFrame()
    df = pd.concat(all_close, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def _download_fred_csv(series_ids: list, start: str, end: str) -> pd.DataFrame:
    """Download FRED series via direct CSV (no API key needed)."""
    import requests
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    all_data = {}
    for sid in series_ids:
        try:
            url = (f"{base}?id={sid}&cosd={start}&coed={end}"
                   f"&fq=Daily&fam=avg&fgst=lin&transformation=lin")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100:
                tmp = pd.read_csv(StringIO(resp.text), index_col=0,
                                  parse_dates=True)
                s = pd.to_numeric(tmp.iloc[:, 0], errors="coerce")
                s.name = sid
                if s.dropna().shape[0] > 10:
                    all_data[sid] = s
                    print(f"    {sid}: {s.dropna().shape[0]} obs")
        except Exception as e:
            print(f"    {sid}: ERROR {e}")
    if all_data:
        df = pd.DataFrame(all_data)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df
    return pd.DataFrame()


def download_extended_data(cache_dir: Path) -> dict:
    """Download all non-news data for the extension window. Returns dict of DataFrames."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = {}

    # --- Market prices (US benchmarks) ---
    mkt_cache = cache_dir / "market_ext.csv"
    if mkt_cache.exists():
        frames["market"] = pd.read_csv(mkt_cache, index_col=0, parse_dates=True)
        print(f"  [cache] market: {frames['market'].shape}")
    else:
        print("  Downloading US market data ...")
        from src.data.download_all import MARKET_INDICES
        df = _download_yahoo(MARKET_INDICES, EXT_START, EXT_END)
        if not df.empty:
            df.to_csv(mkt_cache)
        frames["market"] = df

    # --- Global equity indices ---
    intl_cache = cache_dir / "global_indices_ext.csv"
    if intl_cache.exists():
        frames["global_indices"] = pd.read_csv(intl_cache, index_col=0, parse_dates=True)
        print(f"  [cache] global_indices: {frames['global_indices'].shape}")
    else:
        print("  Downloading global equity indices ...")
        from src.data.download_international import GLOBAL_INDICES
        df = _download_yahoo(GLOBAL_INDICES, EXT_START, EXT_END)
        if not df.empty:
            df.to_csv(intl_cache)
        frames["global_indices"] = df

    # --- FX ---
    fx_cache = cache_dir / "fx_pairs_ext.csv"
    if fx_cache.exists():
        frames["fx"] = pd.read_csv(fx_cache, index_col=0, parse_dates=True)
        print(f"  [cache] fx: {frames['fx'].shape}")
    else:
        print("  Downloading FX pairs ...")
        from src.data.download_international import FX_PAIRS
        df = _download_yahoo(FX_PAIRS, EXT_START, EXT_END)
        if not df.empty:
            df.to_csv(fx_cache)
        frames["fx"] = df

    # --- Commodities ---
    cmd_cache = cache_dir / "commodities_ext.csv"
    if cmd_cache.exists():
        frames["commodities"] = pd.read_csv(cmd_cache, index_col=0, parse_dates=True)
        print(f"  [cache] commodities: {frames['commodities'].shape}")
    else:
        print("  Downloading commodity futures ...")
        from src.data.download_international import COMMODITIES
        df = _download_yahoo(COMMODITIES, EXT_START, EXT_END)
        if not df.empty:
            df.to_csv(cmd_cache)
        frames["commodities"] = df

    # --- FRED international ---
    fred_cache = cache_dir / "fred_intl_ext.csv"
    if fred_cache.exists():
        frames["fred_intl"] = pd.read_csv(fred_cache, index_col=0, parse_dates=True)
        print(f"  [cache] fred_intl: {frames['fred_intl'].shape}")
    else:
        print("  Downloading FRED international series ...")
        from src.data.download_international import FRED_INTERNATIONAL
        df = _download_fred_csv(list(FRED_INTERNATIONAL.keys()),
                                EXT_START, EXT_END)
        if not df.empty:
            df.to_csv(fred_cache)
        frames["fred_intl"] = df

    # --- Daily EPU from FRED ---
    epu_cache = cache_dir / "daily_epu_ext.csv"
    if epu_cache.exists():
        frames["daily_epu"] = pd.read_csv(epu_cache, index_col=0, parse_dates=True)
        print(f"  [cache] daily_epu: {frames['daily_epu'].shape}")
    else:
        print("  Downloading daily EPU ...")
        df = _download_fred_csv(["USEPUINDXD"], "2017-12-01", EXT_END)
        if not df.empty:
            df.to_csv(epu_cache)
        frames["daily_epu"] = df

    # --- GPR (already on disk through 2026) ---
    gpr_path = RAW_DIR / "gpr" / "gpr_daily.xls"
    if gpr_path.exists():
        df = pd.read_excel(gpr_path)
        date_col = [c for c in df.columns if "date" in str(c).lower()]
        if date_col:
            df[date_col[0]] = pd.to_datetime(df[date_col[0]], errors="coerce")
            df = df.set_index(date_col[0])
        df.index.name = "date"
        frames["gpr"] = df
        print(f"  [disk] gpr: {df.shape}, ends {df.index.max().date()}")

    # --- Categorical EPU (already on disk through 2026) ---
    cat_path = RAW_DIR / "epu" / "categorical_epu.xlsx"
    if cat_path.exists():
        df = pd.read_excel(cat_path)
        if "Year" in df.columns:
            df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
            df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
            df = df.dropna(subset=["Year", "Month"])
            df["date"] = pd.to_datetime(
                df["Year"].astype(int).astype(str) + "-" +
                df["Month"].astype(int).astype(str) + "-01")
            df = df.set_index("date").drop(columns=["Year", "Month"], errors="ignore")
        df = df.apply(pd.to_numeric, errors="coerce")
        frames["cat_epu"] = df
        print(f"  [disk] categorical_epu: {df.shape}")

    # --- International EPU (already on disk) ---
    intl_epu_path = RAW_DIR / "epu" / "international_epu.xlsx"
    if intl_epu_path.exists():
        try:
            df = pd.read_excel(intl_epu_path)
            if "Year" in df.columns and "Month" in df.columns:
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
                df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
                df = df.dropna(subset=["Year", "Month"])
                df["date"] = pd.to_datetime(
                    df["Year"].astype(int).astype(str) + "-" +
                    df["Month"].astype(int).astype(str) + "-01")
                df = df.set_index("date").drop(columns=["Year", "Month"],
                                               errors="ignore")
            df = df.apply(pd.to_numeric, errors="coerce")
            frames["intl_epu"] = df
            print(f"  [disk] international_epu: {df.shape}")
        except Exception as e:
            print(f"  [warn] international_epu: {e}")

    # --- SF Fed sentiment ---
    sent_path = RAW_DIR / "fed_sentiment" / "news_sentiment.xlsx"
    if sent_path.exists():
        df = pd.read_excel(sent_path, sheet_name="Data", header=0)
        date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).set_index(date_col)
        df.index.name = "date"
        df = df.apply(pd.to_numeric, errors="coerce")
        frames["fed_sentiment"] = df
        print(f"  [disk] fed_sentiment: {df.shape}, ends {df.index.max().date()}")

    return frames


# ═══════════════════════════════════════════════════════════════════════
# 2. ASSEMBLE EXTENDED NON-NEWS PANEL
# ═══════════════════════════════════════════════════════════════════════

def assemble_non_news_extension(frames: dict, orig_cols: list) -> pd.DataFrame:
    """Build the non-news portion of the extension panel, matching original
    column names exactly.  Returns DataFrame indexed by business day."""

    bdays = pd.date_range("2018-01-02", ANALYSIS_END, freq="B")
    panel = pd.DataFrame(index=bdays)
    panel.index.name = "date"

    # --- Market returns + VIX level ---
    if "market" in frames and not frames["market"].empty:
        df = frames["market"]
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={
            c: f"mkt_ret_{c.replace('^', '').replace('=', '_').replace('.', '_')}"
            for c in log_ret.columns
        })
        vix_cols = [c for c in df.columns if "VIX" in c.upper()]
        if vix_cols:
            vix = df[vix_cols].rename(columns={
                c: f"mkt_lvl_{c.replace('^', '').replace('=', '_').replace('.', '_')}"
                for c in vix_cols
            })
            log_ret = log_ret.join(vix)
        panel = panel.join(log_ret, how="left")

    # --- Global equity indices ---
    if "global_indices" in frames and not frames["global_indices"].empty:
        df = frames["global_indices"]
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={
            c: f"intl_eq_{c.replace('^', '').replace('=', '_')}"
            for c in log_ret.columns
        })
        panel = panel.join(log_ret, how="left")

    # --- FX ---
    if "fx" in frames and not frames["fx"].empty:
        df = frames["fx"]
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={
            c: f"fx_{c.replace('=X', '').replace('=', '_').lower()}"
            for c in log_ret.columns
        })
        panel = panel.join(log_ret, how="left")

    # --- Commodities ---
    if "commodities" in frames and not frames["commodities"].empty:
        df = frames["commodities"]
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={
            c: f"cmd_{c.replace('=F', '').replace('=', '_').lower()}"
            for c in log_ret.columns
        })
        panel = panel.join(log_ret, how="left")

    # --- FRED international ---
    if "fred_intl" in frames and not frames["fred_intl"].empty:
        df = frames["fred_intl"].copy()
        df = df.rename(columns={c: f"fred_{c.lower()}" for c in df.columns})
        panel = panel.join(df, how="left")

    # --- Daily EPU ---
    if "daily_epu" in frames and not frames["daily_epu"].empty:
        df = frames["daily_epu"].rename(columns={"USEPUINDXD": "epu_daily"})
        panel = panel.join(df, how="left")

    # --- GPR ---
    if "gpr" in frames and not frames["gpr"].empty:
        df = frames["gpr"]
        keep = [c for c in df.columns if "GPR" in str(c).upper()]
        df = df[keep].apply(pd.to_numeric, errors="coerce")
        df = df.rename(columns={
            c: f"gpr_{c.strip().lower().replace(' ', '_')}"
            for c in df.columns
        })
        panel = panel.join(df, how="left")

    # --- Categorical EPU (monthly → ffill) ---
    if "cat_epu" in frames and not frames["cat_epu"].empty:
        df = frames["cat_epu"].copy()
        df = df.rename(columns={
            c: f"epu_cat_{c.strip().lower().replace(' ', '_')}"
            for c in df.columns
        })
        df = df.reindex(bdays, method="ffill")
        df.index.name = "date"
        panel = panel.join(df, how="left")

    # --- International EPU (monthly → ffill) ---
    if "intl_epu" in frames and not frames["intl_epu"].empty:
        df = frames["intl_epu"].copy()
        df = df.rename(columns={
            c: f"epu_intl_{c.strip().lower().replace(' ', '_')}"
            for c in df.columns
        })
        df = df.reindex(bdays, method="ffill")
        df.index.name = "date"
        panel = panel.join(df, how="left")

    # --- SF Fed sentiment ---
    if "fed_sentiment" in frames and not frames["fed_sentiment"].empty:
        df = frames["fed_sentiment"].copy()
        df = df.rename(columns={
            c: f"fed_sent_{c.strip().lower().replace(' ', '_')}"
            for c in df.columns
        })
        panel = panel.join(df, how="left")

    # Forward-fill gaps (same as assemble.py)
    panel = panel.ffill(limit=5)
    still_null = panel.columns[panel.isnull().any()]
    if len(still_null) > 0:
        panel[still_null] = panel[still_null].ffill(limit=31)

    # Align to original column names (non-news only)
    non_news_orig = [c for c in orig_cols
                     if not c.startswith("bybee_") and c != "date"]
    matched = []
    missing = []
    for c in non_news_orig:
        if c in panel.columns:
            matched.append(c)
        else:
            missing.append(c)

    if missing:
        print(f"\n  WARNING: {len(missing)} original columns not found in extension:")
        for m in missing[:15]:
            print(f"    {m}")
        if len(missing) > 15:
            print(f"    ... and {len(missing)-15} more")

    result = panel.reindex(columns=non_news_orig)
    print(f"\n  Extension non-news panel: {result.shape}")
    print(f"  Coverage: {result.notna().mean().mean()*100:.1f}%")
    print(f"  Matched {len(matched)}/{len(non_news_orig)} columns")
    return result


# ═══════════════════════════════════════════════════════════════════════
# 3. SIMULATE NEWS TOPICS
# ═══════════════════════════════════════════════════════════════════════

def simulate_news_topics(orig_features: pd.DataFrame,
                         ext_non_news: pd.DataFrame,
                         n_mc: int = N_MC) -> pd.DataFrame:
    """Simulate 180 news-topic attention shares for the extension period
    using a Factor-augmented VAR (FAVAR) estimated on pre-2017 data.

    Model (standard in macro-finance, cf. Bernanke-Boivin-Eliasz 2005):

        f_t = A · f_{t-1} + B · x_t + ε_t,    ε_t ~ N(0, Σ)

    where f_t ∈ R^K are PCA factors of the 180 topic levels, x_t are
    contemporaneous observed non-news variables, A captures day-to-day
    persistence (crucial for realistic levels AND deltas), and B conditions
    on the actual market/macro/policy environment.

    Steps:
      1. PCA on 180 topics → K latent factors (pre-2017)
      2. Estimate FAVAR:  f_t = A · f_{t-1} + B · x_t + ε_t  via Ridge
      3. Roll forward day-by-day using actual x_t (2018-2021), drawing ε_t
      4. Reconstruct topics from factors; normalise to valid attention shares
      5. Deltas emerge naturally from the level dynamics
    """
    print("\n" + "=" * 70)
    print("SIMULATING NEWS TOPICS (Factor-VAR)")
    print("=" * 70)

    level_cols = [c for c in orig_features.columns
                  if c.startswith("bybee_level_")]
    non_news_cols = [c for c in orig_features.columns
                     if not c.startswith("bybee_") and c != "date"]

    news_orig = orig_features[level_cols].copy()
    obs_orig = orig_features[non_news_cols].copy()

    # Require both news and non-news observed
    valid_mask = (news_orig.notna().mean(axis=1) > 0.5) & \
                 (obs_orig.notna().mean(axis=1) > 0.5)
    news_train = news_orig.loc[valid_mask].fillna(method="ffill").fillna(0)
    obs_train = obs_orig.loc[valid_mask].fillna(0)

    print(f"  Training sample: {len(news_train)} days, "
          f"{len(level_cols)} topics, {len(non_news_cols)} exog predictors")

    # ── Step 1: PCA on news topics ──
    news_scaler = StandardScaler()
    news_std = news_scaler.fit_transform(news_train.values)
    pca = PCA(n_components=N_NEWS_FACTORS, random_state=42)
    factors_all = pca.fit_transform(news_std)  # (T, K)
    var_explained = pca.explained_variance_ratio_.sum()
    print(f"  PCA: {N_NEWS_FACTORS} factors explain "
          f"{var_explained * 100:.1f}% of topic variance")

    # ── Step 2: Estimate FAVAR  f_t = A · f_{t-1} + B · x_t + ε_t ──
    obs_scaler = StandardScaler()
    obs_std = obs_scaler.fit_transform(obs_train.values)

    # Build design matrix: [f_{t-1}, x_t]  →  target: f_t
    f_lag = factors_all[:-1]       # (T-1, K)
    f_cur = factors_all[1:]        # (T-1, K)
    x_cur = obs_std[1:]            # (T-1, P)
    design = np.hstack([f_lag, x_cur])  # (T-1, K+P)

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(design, f_cur)
    f_fitted = ridge.predict(design)
    residuals = f_cur - f_fitted

    # Diagnostics
    r2_favar = 1 - np.var(residuals, axis=0) / np.var(f_cur, axis=0)
    print(f"  FAVAR R² per factor: median={np.median(r2_favar):.3f}, "
          f"mean={np.mean(r2_favar):.3f}, "
          f"max={np.max(r2_favar):.3f}")

    # Check factor autocorrelation (should be high if VAR is doing its job)
    ac1 = np.array([np.corrcoef(factors_all[:-1, k],
                                factors_all[1:, k])[0, 1]
                    for k in range(N_NEWS_FACTORS)])
    print(f"  Factor AR(1) autocorrelation: median={np.median(ac1):.3f}, "
          f"range=[{ac1.min():.3f}, {ac1.max():.3f}]")

    resid_cov = np.cov(residuals, rowvar=False)
    T_resid = len(residuals)

    K = N_NEWS_FACTORS
    A_hat = ridge.coef_[:, :K]        # (K, K) autoregressive
    B_hat = ridge.coef_[:, K:]        # (K, P) exogenous loading
    intercept = ridge.intercept_      # (K,)

    # ── Idiosyncratic component u_t (topic-level noise not in PCA factors) ──
    # Standard factor decomposition: topics = Λ·f + u
    # news_std and factors_all are both already (T_valid, ·)
    topics_factor_part = pca.inverse_transform(factors_all)
    idio_resid_std = news_std - topics_factor_part
    T_idio = len(idio_resid_std)
    print(f"  Idiosyncratic residuals: {T_idio} days × {idio_resid_std.shape[1]} topics")
    print(f"  Idiosyncratic std (mean across topics): "
          f"{idio_resid_std.std(axis=0).mean():.4f}")

    # ── Step 3: Roll forward with block-bootstrapped residuals ──
    BLOCK_LEN = 21  # ~1 month blocks (Politis & Romano 1994)
    WINSORIZE_Z = 5.0  # clip standardized conditioning variables (McCracken & Ng 2016)

    ext_obs = ext_non_news.reindex(columns=non_news_cols).fillna(0)
    ext_obs_std = obs_scaler.transform(ext_obs.values)

    # Winsorize extreme z-scores in extension period (COVID crash produces
    # values up to 60σ, which makes the linear FAVAR prediction unreliable)
    n_extreme = np.sum(np.abs(ext_obs_std) > WINSORIZE_Z)
    pct_extreme = n_extreme / ext_obs_std.size * 100
    max_z_before = np.abs(ext_obs_std).max()
    ext_obs_std = np.clip(ext_obs_std, -WINSORIZE_Z, WINSORIZE_Z)
    print(f"  Winsorized conditioning variables at ±{WINSORIZE_Z}σ: "
          f"{n_extreme} values ({pct_extreme:.2f}%), max |z| was {max_z_before:.1f}")

    n_ext = len(ext_obs)
    n_topics = len(level_cols)

    f_last = factors_all[-1]
    rng = np.random.default_rng(42)

    def _block_bootstrap(source: np.ndarray, n_needed: int) -> np.ndarray:
        """Moving block bootstrap from a source residual matrix."""
        T_src = len(source)
        out = np.zeros((n_needed, source.shape[1]))
        pos = 0
        while pos < n_needed:
            start = rng.integers(0, T_src - BLOCK_LEN)
            block = source[start:start + BLOCK_LEN]
            take = min(BLOCK_LEN, n_needed - pos)
            out[pos:pos + take] = block[:take]
            pos += take
        return out

    all_topic_paths = np.zeros((n_mc, n_ext, n_topics))
    hist_day_sum = news_train.values.sum(axis=1).mean()

    for mc in range(n_mc):
        # Block-bootstrap factor-level VAR innovations
        eps_seq = _block_bootstrap(residuals, n_ext)
        # Block-bootstrap idiosyncratic topic-level noise
        u_seq = _block_bootstrap(idio_resid_std, n_ext)

        f_prev = f_last.copy()
        f_path = np.zeros((n_ext, K))
        for t in range(n_ext):
            f_t = intercept + A_hat @ f_prev + B_hat @ ext_obs_std[t] + eps_seq[t]
            f_path[t] = f_t
            f_prev = f_t

        # Reconstruct: factor part + idiosyncratic
        topics_std = pca.inverse_transform(f_path) + u_seq
        topics_raw = news_scaler.inverse_transform(topics_std)
        topics_raw = np.clip(topics_raw, 0, None)

        row_sums = topics_raw.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        topics_raw = topics_raw * (hist_day_sum / row_sums)

        all_topic_paths[mc] = topics_raw

    # Single representative path (preserves realistic daily variability)
    topics_single = all_topic_paths[0]

    # Build level DataFrame from single representative path
    ext_levels = pd.DataFrame(topics_single, index=ext_non_news.index,
                              columns=level_cols)

    delta_cols = [c.replace("bybee_level_", "bybee_delta_")
                  for c in level_cols]

    last_train_levels = news_train.iloc[-1].values
    first_delta = topics_single[0] - last_train_levels
    subsequent_deltas = np.diff(topics_single, axis=0)
    all_deltas = np.vstack([first_delta[np.newaxis, :], subsequent_deltas])

    ext_deltas = pd.DataFrame(all_deltas, index=ext_non_news.index,
                              columns=delta_cols)

    result = pd.concat([ext_levels, ext_deltas], axis=1)

    # ── Diagnostics: single path vs historical ──
    sim_ac1 = np.array([
        np.corrcoef(topics_single[:-1, j], topics_single[1:, j])[0, 1]
        for j in range(len(level_cols))
    ])
    hist_ac1 = np.array([
        np.corrcoef(news_train.values[:-1, j],
                     news_train.values[1:, j])[0, 1]
        for j in range(len(level_cols))
    ])
    print(f"\n  Level AR(1) autocorrelation:")
    print(f"    Historical: median={np.nanmedian(hist_ac1):.3f}")
    print(f"    Simulated:  median={np.nanmedian(sim_ac1):.3f}")

    hist_deltas = news_train.diff().iloc[1:]
    sim_delta_std = ext_deltas.std().mean()
    hist_delta_std = hist_deltas.std().mean()
    print(f"  Delta std: historical={hist_delta_std:.6f}, "
          f"simulated={sim_delta_std:.6f}, "
          f"ratio={sim_delta_std / hist_delta_std:.2f}")

    # MC uncertainty across paths (for reporting)
    topic_p10 = np.percentile(all_topic_paths, 10, axis=0)
    topic_p90 = np.percentile(all_topic_paths, 90, axis=0)
    avg_spread = np.mean(topic_p90 - topic_p10)
    avg_level = np.mean(topics_single)
    print(f"  MC uncertainty ({n_mc} paths): avg 80% band = {avg_spread:.6f} "
          f"(avg level = {avg_level:.6f})")
    print(f"  Simulated news panel: {result.shape}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# 4. COMBINE PANELS AND GENERATE EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════

def build_combined_panel(orig_path: Path, ext_news: pd.DataFrame,
                         ext_non_news: pd.DataFrame) -> Path:
    """Concatenate original panel + extension into a combined CSV."""
    print("\n" + "=" * 70)
    print("BUILDING COMBINED PANEL")
    print("=" * 70)

    orig = pd.read_csv(orig_path, index_col=0, parse_dates=True)
    orig_cols = list(orig.columns)

    # Combine news + non-news for extension
    ext = pd.concat([ext_news, ext_non_news], axis=1)
    ext = ext.reindex(columns=orig_cols)

    # Remove overlap (extension starts after original ends)
    ext = ext.loc[ext.index > orig.index.max()]

    combined = pd.concat([orig, ext], axis=0)
    combined = combined.sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]

    out_path = PROC_DIR / "daily_features_extended.csv"
    combined.to_csv(out_path)
    print(f"  Original: {orig.shape[0]} days ending {orig.index.max().date()}")
    print(f"  Extension: {ext.shape[0]} days ending {ext.index.max().date()}")
    print(f"  Combined: {combined.shape[0]} days, {combined.shape[1]} features")
    print(f"  Saved: {out_path}")
    return out_path


def generate_embeddings(combined_path: Path) -> tuple:
    """Load trained reference model and extract embeddings for the full panel."""
    print("\n" + "=" * 70)
    print("GENERATING EMBEDDINGS")
    print("=" * 70)

    import torch
    from src.evaluate import (dss_config_from_json, _build_model,
                              extract_full_embeddings)
    from src.train import load_and_prepare
    from src.config import DSSConfig

    device = torch.device("cpu")
    ckpt_dir = REF_MODEL

    cfg = dss_config_from_json(ckpt_dir / "config.json")
    cfg = cfg.__class__(**{
        **cfg.__dict__,
        "features_path": str(combined_path),
        "output_dir": str(OUT_DIR),
    })

    print(f"  Loading model from {ckpt_dir.name} ...")
    _, meta = load_and_prepare(cfg)

    # Use saved normalization stats from training (pinned to pre-2017 window)
    norm_path = ckpt_dir / "norm_stats.npz"
    if norm_path.exists():
        saved = np.load(norm_path)
        meta["col_mean"] = saved["col_mean"]
        meta["col_std"] = saved["col_std"]
        print("  Using saved norm_stats from training")

    model = _build_model(cfg, meta, device)
    mpath = ckpt_dir / "best_model.pt"
    model.load_state_dict(torch.load(mpath, map_location=device,
                                     weights_only=True))
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    all_z, dates, all_regime = extract_full_embeddings(model, cfg, device,
                                                       blend_overlap=True)
    print(f"  Embeddings: {all_z.shape} over {dates[0]} - {dates[-1]}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "embeddings_extended.npz",
             z=all_z, dates=np.array(dates, dtype="datetime64[D]"))

    return all_z, dates, all_regime


# ═══════════════════════════════════════════════════════════════════════
# 5. VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_embeddings(z: np.ndarray, dates: np.ndarray):
    """Run validation analyses on the extended embeddings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from sklearn.decomposition import PCA as skPCA

    print("\n" + "=" * 70)
    print("VALIDATING EXTENDED EMBEDDINGS")
    print("=" * 70)

    dates = pd.to_datetime(dates)
    df_z = pd.DataFrame(z, index=dates)

    # PCA on embeddings (fit on pre-2017, transform all)
    pre2018_mask = dates < pd.Timestamp("2018-01-01")
    pca = skPCA(n_components=5, random_state=42)
    pca.fit(z[pre2018_mask])
    pcs = pca.transform(z)
    pc_df = pd.DataFrame(pcs, index=dates,
                         columns=[f"PC{i+1}" for i in range(5)])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 5a. Embedding PC trajectory with NBER shading ──
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for i, ax in enumerate(axes):
        pc_name = f"PC{i+1}"
        ax.plot(pc_df.index, pc_df[pc_name], linewidth=0.5, color="black")
        ax.set_ylabel(pc_name)
        ax.axvline(pd.Timestamp("2017-12-29"), color="red", linestyle="--",
                   linewidth=0.8, alpha=0.7)
        ax.text(pd.Timestamp("2018-03-01"), ax.get_ylim()[1] * 0.9,
                "← in-sample | extension →", fontsize=8, color="red",
                va="top")
        for rs, re in NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(rs), pd.Timestamp(re),
                       alpha=0.15, color="gray")
    axes[0].set_title("DSSDE Embedding PCs: Original + Covid Extension")
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_pcs_extended.pdf", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: embedding_pcs_extended.pdf")

    # ── 5b. Covid zoom (2019-2021) ──
    zoom_mask = (dates >= pd.Timestamp(ANALYSIS_START)) & \
                (dates <= pd.Timestamp(ANALYSIS_END))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, ax in enumerate(axes):
        pc_name = f"PC{i+1}"
        ax.plot(pc_df.index[zoom_mask], pc_df[pc_name].values[zoom_mask],
                linewidth=1.0, color="black")
        ax.set_ylabel(pc_name)
        for rs, re in NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(rs), pd.Timestamp(re),
                       alpha=0.2, color="gray")
        ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-01"),
                   alpha=0.3, color="#2d8f47", label="Covid recession")
    axes[0].set_title("Embedding PCs: Covid-Era Focus (2019-2021)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_pcs_covid_zoom.pdf", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: embedding_pcs_covid_zoom.pdf")

    # ── 5c. Compare with ADS index ──
    ads_path = RAW_DIR / "ads" / "ads_latest.csv"
    if ads_path.exists():
        ads = pd.read_csv(ads_path, index_col=0, parse_dates=True)
        ads_col = ads.columns[0] if len(ads.columns) > 0 else None
        if ads_col:
            ads_s = pd.to_numeric(ads[ads_col], errors="coerce").dropna()
            ads_s.index = pd.to_datetime(ads_s.index)

            common = pc_df.index.intersection(ads_s.index)
            if len(common) > 100:
                pc1_aligned = pc_df.loc[common, "PC1"]
                ads_aligned = ads_s.loc[common]

                # Split into pre-2018 and extension
                pre = common < pd.Timestamp("2018-01-01")
                post = common >= pd.Timestamp("2018-01-01")
                corr_pre = pc1_aligned[pre].corr(ads_aligned[pre])
                corr_post = pc1_aligned[post].corr(ads_aligned[post])

                print(f"  ADS correlation: pre-2018={corr_pre:.3f}, "
                      f"extension={corr_post:.3f}")

                fig, ax1 = plt.subplots(figsize=(14, 5))
                ax1.plot(ads_aligned.index, ads_aligned.values,
                         linewidth=0.8, color="steelblue", label="ADS Index")
                ax1.set_ylabel("ADS Index", color="steelblue")
                ax2 = ax1.twinx()
                ax2.plot(pc1_aligned.index, pc1_aligned.values,
                         linewidth=0.8, color="black", alpha=0.7,
                         label="Embedding PC1")
                ax2.set_ylabel("Embedding PC1")
                ax1.axvline(pd.Timestamp("2017-12-29"), color="red",
                            linestyle="--", linewidth=0.8)
                for rs, re in NBER_RECESSIONS:
                    ax1.axvspan(pd.Timestamp(rs), pd.Timestamp(re),
                                alpha=0.15, color="gray")
                ax1.set_title(f"ADS vs Embedding PC1 "
                              f"(corr pre-2018: {corr_pre:.3f}, "
                              f"extension: {corr_post:.3f})")
                lines1, labels1 = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines1 + lines2, labels1 + labels2,
                           loc="lower left", fontsize=8)
                fig.tight_layout()
                fig.savefig(OUT_DIR / "ads_vs_embedding_extended.pdf",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
                print("  Saved: ads_vs_embedding_extended.pdf")

    # ── 5d. Embedding distance from "normal" ──
    pre2018_mean = z[pre2018_mask].mean(axis=0)
    dist_from_normal = np.linalg.norm(z - pre2018_mean, axis=1)
    dist_df = pd.Series(dist_from_normal, index=dates)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dist_df.index, dist_df.values, linewidth=0.5, color="black")
    for rs, re in NBER_RECESSIONS:
        ax.axvspan(pd.Timestamp(rs), pd.Timestamp(re),
                   alpha=0.15, color="gray")
    ax.axvline(pd.Timestamp("2017-12-29"), color="red", linestyle="--",
               linewidth=0.8)
    ax.set_ylabel("‖z(t) − z̄_train‖")
    ax.set_title("Embedding Distance from Training-Period Centroid")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "embedding_distance_from_normal.pdf", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  Saved: embedding_distance_from_normal.pdf")

    # ── 5e. Summary statistics ──
    ext_mask = dates >= pd.Timestamp("2018-01-01")
    covid_mask = (dates >= pd.Timestamp("2020-02-01")) & \
                 (dates <= pd.Timestamp("2020-04-30"))
    recovery_mask = (dates >= pd.Timestamp("2020-05-01")) & \
                    (dates <= pd.Timestamp("2020-12-31"))

    print("\n  Embedding PC1 summary:")
    print(f"    Training mean (pre-2018):  {pc_df.loc[pre2018_mask, 'PC1'].mean():.4f}")
    print(f"    Extension mean (2018+):    {pc_df.loc[ext_mask, 'PC1'].mean():.4f}")
    if covid_mask.any():
        print(f"    Covid crash (Feb-Apr 2020): {pc_df.loc[covid_mask, 'PC1'].mean():.4f}")
    if recovery_mask.any():
        print(f"    Recovery (May-Dec 2020):    {pc_df.loc[recovery_mask, 'PC1'].mean():.4f}")

    # ── 5f. Simple regime classification via cosine similarity to historical crises ──
    gfc_mask = (dates >= pd.Timestamp("2008-09-01")) & \
               (dates <= pd.Timestamp("2009-03-31"))
    expansion_mask = (dates >= pd.Timestamp("2014-01-01")) & \
                     (dates <= pd.Timestamp("2017-12-31"))

    if gfc_mask.any() and expansion_mask.any():
        gfc_centroid = z[gfc_mask].mean(axis=0)
        exp_centroid = z[expansion_mask].mean(axis=0)

        cos_to_gfc = z @ gfc_centroid / (
            np.linalg.norm(z, axis=1) * np.linalg.norm(gfc_centroid) + 1e-10)
        cos_to_exp = z @ exp_centroid / (
            np.linalg.norm(z, axis=1) * np.linalg.norm(exp_centroid) + 1e-10)

        crisis_score = cos_to_gfc - cos_to_exp
        cs_df = pd.Series(crisis_score, index=dates)

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(cs_df.index, cs_df.values, linewidth=0.5, color="black")
        ax.axhline(0, color="gray", linestyle=":", linewidth=0.5)
        for rs, re in NBER_RECESSIONS:
            ax.axvspan(pd.Timestamp(rs), pd.Timestamp(re),
                       alpha=0.15, color="gray")
        ax.axvline(pd.Timestamp("2017-12-29"), color="red", linestyle="--",
                   linewidth=0.8)
        ax.set_ylabel("cos(z, GFC) − cos(z, expansion)")
        ax.set_title("Crisis Proximity Score: GFC-like vs Expansion-like")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "crisis_proximity_score.pdf", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print("  Saved: crisis_proximity_score.pdf")

        if covid_mask.any():
            print(f"\n  Crisis proximity score:")
            print(f"    GFC peak (2008-09 to 2009-03): "
                  f"{cs_df.loc[gfc_mask].mean():.4f}")
            print(f"    Expansion (2014-2017):          "
                  f"{cs_df.loc[expansion_mask].mean():.4f}")
            print(f"    Covid crash (Feb-Apr 2020):     "
                  f"{cs_df.loc[covid_mask].mean():.4f}")

    print(f"\n  All outputs saved to: {OUT_DIR}")


# ═══════════════════════════════════════════════════════════════════════
# 6. MACRO NOWCASTING VALIDATION
# ═══════════════════════════════════════════════════════════════════════

# Stationary transformations matching src/evaluate.py
_PCT_TARGETS = {"CPIAUCSL", "INDPRO", "PAYEMS", "GDPC1"}
_DIFF_TARGETS = {"UNRATE", "FEDFUNDS", "UMCSENT", "BAAFFM", "DTWEXBGS"}
_BINARY_TARGETS = {"USREC"}
_LEVEL_TARGETS = {"T10Y2Y"}

FRED_TARGET_CODES = [
    "CPIAUCSL", "UNRATE", "GDPC1", "UMCSENT", "INDPRO",
    "PAYEMS", "FEDFUNDS", "T10Y2Y", "BAAFFM", "DTWEXBGS",
]


def _stationary(raw: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for c in raw.columns:
        s = raw[c].dropna()
        if len(s) < 3:
            continue
        if c in _PCT_TARGETS:
            out[f"{c}_pctchg"] = s.pct_change() * 100
        elif c in _DIFF_TARGETS:
            out[f"{c}_diff"] = s.diff()
        elif c in _BINARY_TARGETS:
            out[c] = s
        elif c in _LEVEL_TARGETS:
            out[c] = s
        else:
            out[f"{c}_diff"] = s.diff()
    return pd.DataFrame(out)


def download_macro_targets_extended() -> pd.DataFrame:
    """Download macro targets for 2018-2021 from FRED (direct CSV, no API key)
    and combine with original targets."""
    import requests

    cache_path = OUT_DIR / "cache" / "macro_targets_extended.csv"
    if cache_path.exists():
        print("  [cache] macro targets")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    orig = pd.read_csv(PROC_DIR / "macro_targets.csv",
                       index_col=0, parse_dates=True)

    base_url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    ext_frames = {}
    for code in FRED_TARGET_CODES:
        try:
            url = (f"{base_url}?id={code}"
                   f"&cosd=2018-01-01&coed=2021-07-01"
                   f"&fq=Daily&fam=avg&transformation=lin")
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 50:
                s = pd.read_csv(StringIO(resp.text),
                                index_col=0, parse_dates=True).iloc[:, 0]
                s = pd.to_numeric(s, errors="coerce").dropna()
                if len(s) > 0:
                    ext_frames[code] = s
                    print(f"  [FRED] {code}: {len(s)} obs "
                          f"({s.index[0].date()} - {s.index[-1].date()})")
                else:
                    print(f"  [FRED] {code}: empty after parse")
            else:
                print(f"  [FRED] {code}: status {resp.status_code}")
        except Exception as e:
            print(f"  [FRED] {code}: FAILED ({e})")

    if not ext_frames:
        print("  WARNING: No FRED data downloaded. Using original targets only.")
        return orig

    ext_df = pd.DataFrame(ext_frames)

    # Combine original + extension
    combined = pd.concat([orig, ext_df], axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(cache_path)
    print(f"  Combined targets: {combined.shape}")
    return combined


def validate_macro_nowcasting(z: np.ndarray, dates: np.ndarray):
    """Macro nowcasting: train ridge on 1985-2017 embeddings, test on 2019-2021."""
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import r2_score

    print("\n" + "=" * 70)
    print("MACRO NOWCASTING VALIDATION (Covid-era OOS)")
    print("=" * 70)

    dates = pd.to_datetime(dates)

    # Download/load extended targets
    targets_full = download_macro_targets_extended()
    stationary = _stationary(targets_full)

    # Aggregate embeddings to monthly (matching evaluate.py)
    z_df = pd.DataFrame(z, index=dates)
    z_monthly = z_df.resample("MS").mean().dropna()
    z_m_arr = z_monthly.values
    d_m = z_monthly.index

    # Windows: original 3 (for baseline) + new w4 (Covid OOS)
    windows = [
        ("1985-01-01", "2000-12-31", "2001-01-01", "2005-12-31", "w1 (01-05)"),
        ("1985-01-01", "2005-12-31", "2006-01-01", "2011-12-31", "w2 (06-11)"),
        ("1985-01-01", "2011-12-31", "2012-01-01", "2017-12-31", "w3 (12-17)"),
        ("1985-01-01", "2017-12-31", "2019-01-01", "2021-06-30", "w4 (Covid OOS)"),
    ]

    results = []
    for col in stationary.columns:
        y_s = stationary[col].dropna()
        if len(y_s) < 30:
            continue

        is_daily = len(y_s) > 2000
        if is_daily:
            z_use, d_use = z, dates
        else:
            z_use, d_use = z_m_arr, d_m

        common = d_use[np.isin(d_use, y_s.index)]
        common = pd.DatetimeIndex(sorted(set(common) & set(y_s.index)))
        if len(common) < 30:
            continue

        z_aligned = pd.DataFrame(z_use, index=d_use).loc[common].values
        y_aligned = y_s.loc[common].values

        for tr_s, tr_e, te_s, te_e, wname in windows:
            train_ok = (common >= tr_s) & (common <= tr_e) & ~np.isnan(y_aligned)
            test_ok = (common >= te_s) & (common <= te_e) & ~np.isnan(y_aligned)

            if train_ok.sum() < 20 or test_ok.sum() < 3:
                continue

            z_tr = z_aligned[train_ok]
            z_te = z_aligned[test_ok]
            y_tr = y_aligned[train_ok]
            y_te = y_aligned[test_ok]

            # PCA-reduce + RidgeCV (same as evaluate.py)
            scaler = StandardScaler()
            z_tr_s = scaler.fit_transform(z_tr)
            z_te_s = scaler.transform(z_te)

            n_comp = min(16, z_tr_s.shape[0] // 5, z_tr_s.shape[1])
            n_comp = max(n_comp, 2)
            pca_p = PCA(n_components=n_comp, random_state=42)
            z_tr_p = pca_p.fit_transform(z_tr_s)
            z_te_p = pca_p.transform(z_te_s)

            ridge = RidgeCV(alphas=np.logspace(0, 7, 50))
            ridge.fit(z_tr_p, y_tr)
            y_pred = ridge.predict(z_te_p)

            r2 = r2_score(y_te, y_pred)
            corr = float(np.corrcoef(y_te, y_pred)[0, 1]) if len(y_te) > 2 else 0.0

            results.append({
                "target": col, "window": wname,
                "r2": r2, "corr": corr,
                "n_train": int(train_ok.sum()),
                "n_test": int(test_ok.sum()),
            })

    if not results:
        print("  No results (targets may not extend past 2017)")
        return

    rdf = pd.DataFrame(results)
    rdf.to_csv(OUT_DIR / "nowcasting_covid_validation.csv", index=False)

    # Display pivot table
    pivot = rdf.pivot_table(index="target", columns="window",
                            values=["r2", "corr"], aggfunc="first")
    print("\n  Nowcasting R² by target and window:")
    print(pivot.to_string())

    # Focused comparison: w3 vs w4
    w3 = rdf[rdf["window"].str.contains("w3")]
    w4 = rdf[rdf["window"].str.contains("w4")]
    if not w3.empty and not w4.empty:
        comp = w3[["target", "r2", "corr"]].merge(
            w4[["target", "r2", "corr"]],
            on="target", suffixes=("_w3", "_w4"))
        print("\n  Head-to-head comparison (w3 baseline vs w4 Covid OOS):")
        for _, row in comp.iterrows():
            r2_change = row["r2_w4"] - row["r2_w3"]
            arrow = "▲" if r2_change > 0 else "▼"
            print(f"    {row['target']:20s}  R² w3={row['r2_w3']:7.4f}  "
                  f"w4={row['r2_w4']:7.4f}  Δ={r2_change:+7.4f} {arrow}  "
                  f"corr w3={row['corr_w3']:6.3f}  w4={row['corr_w4']:6.3f}")

    print(f"\n  Saved: {OUT_DIR / 'nowcasting_covid_validation.csv'}")


# ═══════════════════════════════════════════════════════════════════════
# 7. BOND SPANNING PUZZLE VALIDATION
# ═══════════════════════════════════════════════════════════════════════

def validate_bond_spanning(z: np.ndarray, dates: np.ndarray):
    """Bond spanning test using extended embeddings + real yields."""
    import statsmodels.api as sm
    from scipy import stats

    print("\n" + "=" * 70)
    print("BOND SPANNING PUZZLE VALIDATION (extended to 2021)")
    print("=" * 70)

    dates = pd.to_datetime(dates)

    # Load extended features (real yields)
    ext_feat_path = PROC_DIR / "daily_features_extended.csv"
    if not ext_feat_path.exists():
        print("  Extended features not found; skipping bond spanning.")
        return

    features = pd.read_csv(ext_feat_path, index_col="date", parse_dates=True)

    yield_names = {1: "fred_dgs1", 2: "fred_dgs2",
                   5: "fred_dgs5", 10: "fred_dgs10", 30: "fred_dgs30"}
    yields_daily = features[list(yield_names.values())].copy()
    yields_daily.columns = list(yield_names.keys())

    emb_df = pd.DataFrame(z, index=dates,
                           columns=[f"z{i}" for i in range(z.shape[1])])

    # End-of-month
    yields_m = yields_daily.resample("ME").last().dropna()
    emb_m = emb_df.resample("ME").last()
    cidx = yields_m.index.intersection(emb_m.index)
    yields_m = yields_m.loc[cidx]
    emb_m = emb_m.loc[cidx]

    print(f"  Monthly sample: {cidx[0].strftime('%Y-%m')} - "
          f"{cidx[-1].strftime('%Y-%m')}  (T = {len(cidx)})")

    # Yield PCs
    yscaler = StandardScaler()
    yields_sc = yscaler.fit_transform(yields_m.values)
    ypca = PCA(n_components=3)
    ypc_vals = ypca.fit_transform(yields_sc)
    ypc = pd.DataFrame(ypc_vals, index=cidx,
                        columns=["YPC1", "YPC2", "YPC3"])

    # Embedding PCs
    N_EPC = 5
    escaler = StandardScaler()
    emb_sc = escaler.fit_transform(emb_m.values)
    epca = PCA(n_components=N_EPC)
    epc_vals = epca.fit_transform(emb_sc)
    epc = pd.DataFrame(epc_vals, index=cidx,
                        columns=[f"EPC{i+1}" for i in range(N_EPC)])

    # Bond excess returns
    def mod_duration(n, y_pct):
        y = y_pct / 100.0
        if y < 1e-8:
            return float(n)
        return (1.0 / (y / 2.0)) * (1.0 - (1.0 + y / 2.0) ** (-2 * n)) / \
               (1.0 + y / 2.0)

    maturities = [2, 5, 10]
    horizons = [1, 3, 12]

    rx1 = pd.DataFrame(index=cidx)
    for n in maturities:
        carry = (yields_m[n] - yields_m[1]) / 12.0 / 100.0
        dur = yields_m[n].apply(lambda y: mod_duration(n, y))
        dy = yields_m[n].diff() / 100.0
        rx1[f"rx{n}"] = carry - dur * dy
    rx1["rx_avg"] = rx1.mean(axis=1)

    rx_all = {}
    for h in horizons:
        rx_h = pd.DataFrame(index=cidx)
        for col in rx1.columns:
            rx_h[col] = rx1[col].rolling(h).sum()
        rx_all[h] = rx_h.iloc[max(h, 1):]

    ypc_lag = ypc.shift(1)
    epc_lag = epc.shift(1)

    def align(rx_df, lag_start=2):
        valid = rx_df.index[lag_start:]
        valid = valid.intersection(ypc_lag.dropna().index).intersection(
                                   epc_lag.dropna().index)
        return rx_df.loc[valid], ypc_lag.loc[valid], epc_lag.loc[valid]

    def nw_ols(y, X, nw_lags):
        Xc = sm.add_constant(X)
        return sm.OLS(y, Xc).fit(cov_type="HAC",
                                  cov_kwds={"maxlags": nw_lags})

    targets = ["rx2", "rx5", "rx10", "rx_avg"]
    all_is = []
    all_oos = []

    # ── In-sample: full period vs pre-2018 ──
    for period_label, cutoff in [("Full 1985-2021", None),
                                  ("Pre-2018 only", "2017-12-31")]:
        print(f"\n  ── In-sample: {period_label} ──")
        for h in [1, 12]:
            rx_h, ypc_h, epc_h = align(rx_all[h])
            if cutoff:
                mask = rx_h.index <= pd.Timestamp(cutoff)
                rx_h, ypc_h, epc_h = rx_h[mask], ypc_h[mask], epc_h[mask]

            nw = max(h + 1, 6)
            for tgt in targets:
                y = rx_h[tgt].values
                m1 = nw_ols(y, ypc_h.values, nw)
                X2 = np.column_stack([ypc_h.values, epc_h.values])
                m2 = nw_ols(y, X2, nw)
                dr2 = m2.rsquared - m1.rsquared
                k1, k2, T = ypc_h.shape[1], epc_h.shape[1], len(y)
                f_stat = (dr2 / k2) / ((1 - m2.rsquared) / (T - k1 - k2 - 1))
                f_pval = 1 - stats.f.cdf(f_stat, k2, T - k1 - k2 - 1)

                stars = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 \
                        else "*" if f_pval < 0.10 else ""
                all_is.append(dict(period=period_label, horizon=h, target=tgt,
                                   R2_yield=m1.rsquared, R2_aug=m2.rsquared,
                                   dR2=dr2, F=f_stat, F_pval=f_pval))
                if h == 1:
                    print(f"    h={h} {tgt:8s}  R²_y={m1.rsquared:.3f}  "
                          f"R²_aug={m2.rsquared:.3f}  ΔR²={dr2:.3f}  "
                          f"F={f_stat:.2f}{stars}")

    # ── Out-of-sample: expanding window through Covid ──
    print("\n  ── OOS Expanding Window (includes Covid) ──")
    MIN_TRAIN = 120

    for h in [1, 12]:
        rx_h, ypc_h, epc_h = align(rx_all[h])
        T = len(rx_h)
        if T <= MIN_TRAIN + 12:
            continue

        for tgt in targets:
            y = rx_h[tgt].values
            Xy = ypc_h.values
            Xe = epc_h.values
            X_aug = np.column_stack([Xy, Xe])

            preds_y, preds_a, actuals, hist_means = [], [], [], []
            for t0 in range(MIN_TRAIN, T):
                y_tr = y[:t0]
                Xc = sm.add_constant(Xy[:t0])
                m1 = sm.OLS(y_tr, Xc).fit()
                Xc_te = np.concatenate([[1.0], Xy[t0]]).reshape(1, -1)
                preds_y.append(m1.predict(Xc_te)[0])

                Xc2 = sm.add_constant(X_aug[:t0])
                m2 = sm.OLS(y_tr, Xc2).fit()
                Xc2_te = np.concatenate([[1.0], X_aug[t0]]).reshape(1, -1)
                preds_a.append(m2.predict(Xc2_te)[0])

                actuals.append(y[t0])
                hist_means.append(y_tr.mean())

            actual = np.array(actuals)
            pred_y = np.array(preds_y)
            pred_a = np.array(preds_a)
            hm = np.array(hist_means)

            sse_y = np.sum((actual - pred_y) ** 2)
            sse_a = np.sum((actual - pred_a) ** 2)
            sst = np.sum((actual - hm) ** 2)
            r2oos_y = 1 - sse_y / sst
            r2oos_a = 1 - sse_a / sst

            # GW test
            loss_y = (actual - pred_y) ** 2
            loss_a = (actual - pred_a) ** 2
            d = loss_y - loss_a
            d_bar = d.mean()
            nw_lag = max(h + 1, 6)

            def _nw_tstat(series):
                s_bar = series.mean()
                n = len(series)
                g0 = np.var(series, ddof=1)
                gs = 0.0
                for j in range(1, nw_lag + 1):
                    w = 1 - j / (nw_lag + 1)
                    gs += 2 * w * np.mean((series[j:] - s_bar) * (series[:-j] - s_bar))
                v = g0 + gs
                return s_bar / np.sqrt(v / n) if v > 0 else np.nan

            t_gw = _nw_tstat(d)
            p_gw = 2 * (1 - stats.norm.cdf(abs(t_gw))) if np.isfinite(t_gw) else np.nan

            # Clark-West (2007) MSPE-adjusted test for nested models
            cw_adj = (pred_y - pred_a) ** 2
            d_cw = d + cw_adj
            t_cw = _nw_tstat(d_cw)
            p_cw = 1.0 - stats.norm.cdf(t_cw) if np.isfinite(t_cw) else np.nan

            gw_star = "***" if p_gw < 0.01 else "**" if p_gw < 0.05 \
                      else "*" if p_gw < 0.10 else ""
            cw_star = "***" if p_cw < 0.01 else "**" if p_cw < 0.05 \
                      else "*" if p_cw < 0.10 else ""

            all_oos.append(dict(horizon=h, target=tgt,
                                R2oos_yield=r2oos_y, R2oos_aug=r2oos_a,
                                dR2oos=r2oos_a - r2oos_y,
                                t_GW=t_gw, p_GW=p_gw,
                                t_CW=t_cw, p_CW=p_cw))

            print(f"    h={h} {tgt:8s}  R²oos_y={r2oos_y:7.4f}  "
                  f"R²oos_aug={r2oos_a:7.4f}  ΔR²={r2oos_a - r2oos_y:7.4f}  "
                  f"t_GW={t_gw:6.2f}{gw_star}  t_CW={t_cw:6.2f}{cw_star}")

    pd.DataFrame(all_is).to_csv(OUT_DIR / "spanning_is_extended.csv", index=False)
    pd.DataFrame(all_oos).to_csv(OUT_DIR / "spanning_oos_extended.csv", index=False)
    print(f"\n  Saved: spanning_is_extended.csv, spanning_oos_extended.csv")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip data download, use cached files")
    args = parser.parse_args()

    print("=" * 70)
    print("COVID-ERA EMBEDDING EXTENSION")
    print(f"Extension window: Jan 2018 - Jun 2021")
    print(f"Analysis focus:   Jan 2019 - Jun 2021")
    print("=" * 70)

    # Load original panel header
    orig_path = PROC_DIR / "daily_features.csv"
    orig_df = pd.read_csv(orig_path, index_col=0, parse_dates=True)
    orig_cols = list(orig_df.columns)
    print(f"\nOriginal panel: {orig_df.shape[0]} days, {len(orig_cols)} features")

    # Step 1: Download extended data
    cache_dir = OUT_DIR / "cache"
    if not args.skip_download:
        print("\n" + "=" * 70)
        print("DOWNLOADING EXTENDED DATA")
        print("=" * 70)
        frames = download_extended_data(cache_dir)
    else:
        print("\n  [skip-download] Loading from cache ...")
        frames = download_extended_data(cache_dir)

    # Step 2: Assemble non-news extension
    print("\n" + "=" * 70)
    print("ASSEMBLING NON-NEWS EXTENSION")
    print("=" * 70)
    ext_non_news = assemble_non_news_extension(frames, orig_cols)

    # Step 3: Simulate news topics
    ext_news = simulate_news_topics(orig_df, ext_non_news)

    # Step 4: Build combined panel
    combined_path = build_combined_panel(orig_path, ext_news, ext_non_news)

    # Step 5: Generate embeddings
    z, dates, regimes = generate_embeddings(combined_path)

    # Step 6: Validate (regime recognition, distance, etc.)
    validate_embeddings(z, dates)

    # Step 7: Macro nowcasting OOS
    validate_macro_nowcasting(z, dates)

    # Step 8: Bond spanning puzzle
    validate_bond_spanning(z, dates)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
