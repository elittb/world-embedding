"""
Assemble all raw data into a single daily feature matrix + macro targets.

Outputs:
    data/processed/daily_features.csv   -- The input matrix (rows=days, cols=features)
    data/processed/macro_targets.csv    -- Validation targets (forward-filled to daily)
    data/processed/data_summary.txt     -- Human-readable summary of the assembled data

Usage:
    python src/data/assemble.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

SAMPLE_START = "1985-01-02"
SAMPLE_END = "2017-12-29"


def load_bybee() -> pd.DataFrame:
    """Load daily topic attention and compute deltas."""
    print("  Loading Bybee et al. topic attention...")
    path = RAW_DIR / "bybee" / "daily_topic_attention.csv"
    df = pd.read_csv(path)

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"

    topic_cols = [c for c in df.columns if c != date_col]
    df = df[topic_cols].apply(pd.to_numeric, errors="coerce")

    # Compute daily delta (change in topic attention)
    df_delta = df.diff()

    # Prefix columns
    df_level = df.rename(columns={c: f"bybee_level_{c}" for c in df.columns})
    df_delta = df_delta.rename(columns={c: f"bybee_delta_{c}" for c in df_delta.columns})

    result = pd.concat([df_level, df_delta], axis=1)
    print(f"    Topics: {len(topic_cols)}, "
          f"Features (level+delta): {result.shape[1]}, "
          f"Date range: {result.index.min().date()} to {result.index.max().date()}")
    return result


def load_epu() -> pd.DataFrame:
    """Load daily aggregate EPU and monthly categorical EPU."""
    print("  Loading EPU indices...")

    # Daily aggregate EPU (from FRED CSV)
    daily_path = RAW_DIR / "epu" / "daily_epu.csv"
    df_daily = pd.read_csv(daily_path)
    # FRED CSV has columns: observation_date, USEPUINDXD
    date_col = df_daily.columns[0]
    val_col = df_daily.columns[1]
    df_daily[date_col] = pd.to_datetime(df_daily[date_col], errors="coerce")
    df_daily = df_daily.dropna(subset=[date_col])
    df_daily = df_daily.set_index(date_col)
    df_daily.index.name = "date"
    df_daily = df_daily.rename(columns={val_col: "epu_daily"})
    df_daily["epu_daily"] = pd.to_numeric(df_daily["epu_daily"], errors="coerce")

    # Monthly categorical EPU
    cat_path = RAW_DIR / "epu" / "categorical_epu.xlsx"
    df_cat = pd.read_excel(cat_path)

    # Drop footer/attribution rows
    if "Year" in df_cat.columns:
        df_cat["Year"] = pd.to_numeric(df_cat["Year"], errors="coerce")
        df_cat = df_cat.dropna(subset=["Year"])
        df_cat["Month"] = pd.to_numeric(df_cat["Month"], errors="coerce")
        df_cat = df_cat.dropna(subset=["Month"])
        df_cat["date"] = pd.to_datetime(
            df_cat["Year"].astype(int).astype(str) + "-" +
            df_cat["Month"].astype(int).astype(str) + "-01"
        )
        df_cat = df_cat.set_index("date")
        df_cat = df_cat.drop(columns=["Year", "Month"], errors="ignore")
    else:
        df_cat.index = pd.to_datetime(df_cat.iloc[:, 0])
        df_cat = df_cat.iloc[:, 1:]

    # Prefix columns
    df_cat = df_cat.rename(columns={c: f"epu_cat_{c.strip().lower().replace(' ', '_')}"
                                    for c in df_cat.columns})
    df_cat = df_cat.apply(pd.to_numeric, errors="coerce")

    # Forward-fill monthly categorical to daily
    daily_idx = pd.date_range(SAMPLE_START, SAMPLE_END, freq="B")
    df_cat = df_cat.reindex(daily_idx, method="ffill")
    df_cat.index.name = "date"

    result = df_daily.join(df_cat, how="outer")
    print(f"    Daily EPU + {len(df_cat.columns)} categorical series, "
          f"Total features: {result.shape[1]}")
    return result



def load_gpr() -> pd.DataFrame:
    """Load daily GPR index."""
    print("  Loading GPR index...")
    path = RAW_DIR / "gpr" / "gpr_daily.xls"
    df = pd.read_excel(path)

    # Find date column
    date_col = None
    for c in df.columns:
        if "date" in str(c).lower():
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.set_index(date_col)
    df.index.name = "date"

    keep_cols = [c for c in df.columns if "GPR" in str(c).upper()]
    if not keep_cols:
        keep_cols = list(df.columns)

    df = df[keep_cols].apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={c: f"gpr_{c.strip().lower().replace(' ', '_')}"
                            for c in df.columns})

    print(f"    Features: {list(df.columns)[:6]}{'...' if len(df.columns) > 6 else ''}")
    return df


def load_fed_sentiment() -> pd.DataFrame:
    """Load SF Fed daily news sentiment."""
    print("  Loading SF Fed News Sentiment...")
    path = RAW_DIR / "fed_sentiment" / "news_sentiment.xlsx"
    df = pd.read_excel(path, sheet_name="Data", header=0)

    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"

    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.rename(columns={c: f"fed_sent_{c.strip().lower().replace(' ', '_')}"
                            for c in df.columns})

    print(f"    Features: {list(df.columns)}, "
          f"Date range: {df.index.min().date()} to {df.index.max().date()}")
    return df


def load_market() -> pd.DataFrame:
    """Load market prices and compute log-returns."""
    print("  Loading market data...")
    path = RAW_DIR / "market" / "market_prices.csv"
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"

    # Compute log-returns
    log_ret = np.log(df / df.shift(1))
    log_ret = log_ret.rename(columns={c: f"mkt_ret_{c.replace('^', '').replace('=', '_').replace('.', '_')}"
                                      for c in log_ret.columns})

    # Also keep VIX level (not return)
    vix_cols = [c for c in df.columns if "VIX" in c.upper()]
    if vix_cols:
        vix = df[vix_cols].rename(columns={c: f"mkt_lvl_{c.replace('^', '').replace('=', '_').replace('.', '_')}"
                                           for c in vix_cols})
        log_ret = log_ret.join(vix)

    print(f"    Market return features: {log_ret.shape[1]}, "
          f"Date range: {log_ret.index.min().date()} to {log_ret.index.max().date()}")
    return log_ret


def load_fred() -> pd.DataFrame:
    """Load FRED macro targets."""
    print("  Loading FRED macro targets...")
    path = RAW_DIR / "fred" / "fred_macro.csv"
    if not path.exists():
        print("    [skip] FRED data not found. Run download_all.py with FRED API key first.")
        return pd.DataFrame()

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    df = df.apply(pd.to_numeric, errors="coerce")

    # Forward-fill monthly/quarterly data to daily
    daily_idx = pd.date_range(SAMPLE_START, SAMPLE_END, freq="B")
    df = df.reindex(daily_idx, method="ffill")
    df.index.name = "date"

    print(f"    Macro series: {list(df.columns)}")
    return df


def load_international() -> tuple:
    """Load all international data (equity, FX, commodities, FRED, EPU)."""
    print("  Loading international data...")
    intl_dir = RAW_DIR / "international"
    frames = {}

    # Global equity indices -> log-returns
    path = intl_dir / "global_indices.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={c: f"intl_eq_{c.replace('^', '').replace('=', '_')}"
                                          for c in log_ret.columns})
        frames["intl_equity"] = log_ret
        print(f"    Global equity indices: {log_ret.shape[1]} series")

    # FX pairs -> log-returns
    path = intl_dir / "fx_pairs.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={c: f"fx_{c.replace('=X', '').replace('=', '_').lower()}"
                                          for c in log_ret.columns})
        frames["fx"] = log_ret
        print(f"    FX pairs: {log_ret.shape[1]} series")

    # Commodities -> log-returns
    path = intl_dir / "commodities.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        log_ret = np.log(df / df.shift(1))
        log_ret = log_ret.rename(columns={c: f"cmd_{c.replace('=F', '').replace('=', '_').lower()}"
                                          for c in log_ret.columns})
        frames["commodities"] = log_ret
        print(f"    Commodities: {log_ret.shape[1]} series")

    # FRED international
    path = intl_dir / "fred_international.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.rename(columns={c: f"fred_{c.lower()}" for c in df.columns})
        frames["fred_intl"] = df
        print(f"    FRED international: {df.shape[1]} series")

    # International EPU (monthly -> ffill to daily)
    path = RAW_DIR / "epu" / "international_epu.xlsx"
    if path.exists():
        try:
            df = pd.read_excel(path)
            if "Year" in df.columns and "Month" in df.columns:
                df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
                df["Month"] = pd.to_numeric(df["Month"], errors="coerce")
                df = df.dropna(subset=["Year", "Month"])
                df["date"] = pd.to_datetime(
                    df["Year"].astype(int).astype(str) + "-" +
                    df["Month"].astype(int).astype(str) + "-01"
                )
                df = df.set_index("date")
                df = df.drop(columns=["Year", "Month"], errors="ignore")
                df = df.apply(pd.to_numeric, errors="coerce")
                df = df.rename(columns={c: f"epu_intl_{c.strip().lower().replace(' ', '_')}"
                                        for c in df.columns})
                frames["epu_intl"] = df
                print(f"    International EPU: {df.shape[1]} countries")
        except Exception as e:
            print(f"    [warn] International EPU: {e}")

    return frames


def assemble():
    """Merge all sources into a single daily feature matrix."""
    print("=" * 70)
    print("ASSEMBLING DAILY FEATURE MATRIX")
    print(f"Sample: {SAMPLE_START} to {SAMPLE_END}")
    print("=" * 70)

    # Load all sources
    bybee = load_bybee()
    epu = load_epu()
    gpr = load_gpr()
    fed_sent = load_fed_sentiment()
    market = load_market()
    fred = load_fred()
    intl_frames = load_international()

    # Create a business-day index for the sample period
    bdays = pd.date_range(SAMPLE_START, SAMPLE_END, freq="B")

    # Merge all feature sources
    print("\n  Merging features on business-day index...")
    features = pd.DataFrame(index=bdays)
    features.index.name = "date"

    all_sources = [("bybee", bybee), ("epu", epu), ("gpr", gpr),
                   ("fed_sent", fed_sent), ("market", market)]
    for name, df in intl_frames.items():
        all_sources.append((name, df))

    for name, df in all_sources:
        if df.empty:
            print(f"    [skip] {name} is empty")
            continue
        df = df[~df.index.duplicated(keep="first")]
        merged = features.join(df, how="left")
        new_cols = [c for c in merged.columns if c not in features.columns]
        print(f"    + {name}: {len(new_cols)} features")
        features = merged

    # Forward-fill: daily/weekly gaps get 5-day limit, then a second
    # pass with 31-day limit catches monthly/weekly FRED series
    # (ICSA, CCSA, WALCL, M2SL, international macro, housing, etc.)
    features = features.ffill(limit=5)
    still_null = features.columns[features.isnull().any()]
    if len(still_null) > 0:
        features[still_null] = features[still_null].ffill(limit=31)

    # Drop columns that are almost entirely NaN (>80% missing even after ffill)
    null_frac = features.isnull().mean()
    high_null = null_frac[null_frac > 0.8].index.tolist()
    if high_null:
        print(f"\n  Dropping {len(high_null)} columns with >80% missing values")
        features = features.drop(columns=high_null)

    # Summary
    n_days, n_features = features.shape
    coverage = features.notna().mean().mean() * 100
    print(f"\n{'=' * 70}")
    print(f"FINAL FEATURE MATRIX")
    print(f"  Days: {n_days}")
    print(f"  Features: {n_features}")
    print(f"  Coverage: {coverage:.1f}%")
    print(f"  Date range: {features.index.min().date()} to {features.index.max().date()}")
    print(f"  Memory: {features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"{'=' * 70}")

    # Feature breakdown
    prefixes = {}
    for c in features.columns:
        prefix = c.split("_")[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    print("\n  Feature breakdown by source:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        print(f"    {prefix}: {count}")

    # Save features
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    features_path = OUT_DIR / "daily_features.csv"
    features.to_csv(features_path)
    print(f"\n  Saved: {features_path}")

    # Save macro targets separately
    if not fred.empty:
        targets_path = OUT_DIR / "macro_targets.csv"
        fred_in_sample = fred.loc[
            (fred.index >= SAMPLE_START) & (fred.index <= SAMPLE_END)]
        fred_in_sample.to_csv(targets_path)
        print(f"  Saved: {targets_path}")

    # Write summary
    summary_path = OUT_DIR / "data_summary.txt"
    with open(summary_path, "w") as f:
        f.write("SEMANTIC DAY EMBEDDING -- DATA SUMMARY\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n")
        f.write(f"{'=' * 60}\n\n")
        f.write(f"Sample period: {SAMPLE_START} to {SAMPLE_END}\n")
        f.write(f"Business days: {n_days}\n")
        f.write(f"Total features: {n_features}\n")
        f.write(f"Overall coverage: {coverage:.1f}%\n\n")
        f.write("Feature breakdown:\n")
        for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
            f.write(f"  {prefix}: {count}\n")
        f.write(f"\nAll features:\n")
        for c in sorted(features.columns):
            f.write(f"  {c}\n")
    print(f"  Saved: {summary_path}")

    return features, fred


if __name__ == "__main__":
    assemble()
