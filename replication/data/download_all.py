"""
Download all raw data sources for the Semantic Day Embedding project.

Sources:
    1. Bybee et al. (JF 2024) - Daily Topic Attention from WSJ (1984-2017)
    2. Baker Bloom Davis (QJE 2016) - Categorical EPU indices (1985-present)
    3. Caldara & Iacoviello (AER 2022) - Geopolitical Risk index (1985-present)
    4. Shapiro et al. (JoE 2022) - SF Fed Daily News Sentiment (1980-present)
    5. Yahoo Finance - Market log-returns for broad asset basket
    6. FRED - Macro validation targets

Usage:
    python src/data/download_all.py
    python src/data/download_all.py --source bybee
    python src/data/download_all.py --source market
"""

import argparse
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

SAMPLE_START = "1985-01-01"
SAMPLE_END = "2017-12-31"


def download_file(url: str, dest: Path, description: str) -> Path:
    """Download a file if it doesn't already exist."""
    if dest.exists():
        print(f"  [skip] {description} already exists: {dest.name}")
        return dest
    print(f"  [download] {description} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(resp.content)
    print(f"  [done] saved {dest.name} ({len(resp.content) / 1024:.0f} KB)")
    return dest


# ---------------------------------------------------------------------------
# 1. Bybee, Kelly, Manela & Xiu -- Daily Topic Attention
# ---------------------------------------------------------------------------

def download_bybee():
    """Download daily topic attention (theta) from structureofnews.com."""
    print("\n=== Bybee et al. (JF 2024): Daily Topic Attention ===")
    out_dir = RAW_DIR / "bybee"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = "https://structureofnews.com/data/download/Daily_Topic_Attention_Theta.csv"
    dest = download_file(url, out_dir / "daily_topic_attention.csv",
                         "Daily Topic Attention (Theta)")

    df = pd.read_csv(dest)
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df.iloc[:, 0].min()} to {df.iloc[:, 0].max()}")
    print(f"  Columns (first 10): {list(df.columns[:10])}")
    return df


# ---------------------------------------------------------------------------
# 2. Baker, Bloom & Davis -- Categorical EPU
# ---------------------------------------------------------------------------

def download_epu():
    """Download categorical EPU indices from policyuncertainty.com + daily from FRED."""
    print("\n=== Baker, Bloom & Davis (QJE 2016): Categorical EPU ===")
    out_dir = RAW_DIR / "epu"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Categorical EPU (monthly, from policyuncertainty.com)
    cat_url = "https://www.policyuncertainty.com/media/Categorical_EPU_Data.xlsx"
    dest_cat = download_file(cat_url, out_dir / "categorical_epu.xlsx",
                             "Categorical EPU (monthly)")

    # Daily aggregate EPU from FRED (more reliable than website download)
    daily_dest = out_dir / "daily_epu.csv"
    if not daily_dest.exists():
        print("  [download] Daily EPU from FRED ...")
        fred_url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
                    "?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans"
                    "&graph_bgcolor=%23ffffff&height=450&mode=fred"
                    "&recession_bars=on&txtcolor=%23444444&ts=12&tts=12"
                    "&width=1168&nt=0&thu=0&trc=0&show_legend=yes"
                    "&show_axis_titles=yes&show_tooltip=yes&id=USEPUINDXD"
                    "&scale=left&cosd=1985-01-01&coed=2018-01-01"
                    "&line_color=%234572a7&link_values=false"
                    "&line_style=solid&mark_type=none&mw=3&lw=2"
                    "&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily"
                    "&fam=avg&fgst=lin&fgsnd=2020-02-01"
                    "&line_index=1&transformation=lin"
                    "&vintage_date=2026-03-17&revision_date=2026-03-17"
                    "&nd=1985-01-01")
        resp = requests.get(fred_url, timeout=60)
        resp.raise_for_status()
        daily_dest.write_bytes(resp.content)
        print(f"  [done] saved {daily_dest.name}")
    else:
        print(f"  [skip] Daily EPU already exists: {daily_dest.name}")

    # Also get Trade Policy Uncertainty (daily, from policyuncertainty.com)
    tpu_url = "https://policyuncertainty.com/media/All_Daily_TPU_Data.csv"
    try:
        download_file(tpu_url, out_dir / "daily_tpu.csv",
                      "Daily Trade Policy Uncertainty")
    except Exception as e:
        print(f"  [warn] Could not download TPU: {e}")

    df_cat = pd.read_excel(dest_cat)
    df_daily = pd.read_csv(daily_dest)
    print(f"  Categorical EPU shape: {df_cat.shape}, columns: {list(df_cat.columns)}")
    print(f"  Daily EPU shape: {df_daily.shape}, columns: {list(df_daily.columns[:5])}")
    return df_cat, df_daily


# ---------------------------------------------------------------------------
# 3. Caldara & Iacoviello -- Geopolitical Risk
# ---------------------------------------------------------------------------

def download_gpr():
    """Download daily GPR index from matteoiacoviello.com."""
    print("\n=== Caldara & Iacoviello (AER 2022): Geopolitical Risk ===")
    out_dir = RAW_DIR / "gpr"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    dest = download_file(url, out_dir / "gpr_daily.xls", "Daily GPR index")

    df = pd.read_excel(dest)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns[:6])}")
    return df


# ---------------------------------------------------------------------------
# 4. SF Fed -- Daily News Sentiment
# ---------------------------------------------------------------------------

def download_fed_sentiment():
    """Download daily news sentiment from SF Fed."""
    print("\n=== Shapiro et al. (JoE 2022): SF Fed Daily News Sentiment ===")
    out_dir = RAW_DIR / "fed_sentiment"
    out_dir.mkdir(parents=True, exist_ok=True)

    url = "https://www.frbsf.org/wp-content/uploads/news_sentiment_data.xlsx"
    dest = download_file(url, out_dir / "news_sentiment.xlsx",
                         "Daily News Sentiment Index")

    df = pd.read_excel(dest)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# 5. Yahoo Finance -- Market Data
# ---------------------------------------------------------------------------

MARKET_TICKERS = {
    # US Equity
    "SPY": "S&P 500 ETF",
    "QQQ": "Nasdaq 100 ETF",
    "IWM": "Russell 2000 ETF",
    "XLF": "Financials Sector",
    "XLE": "Energy Sector",
    "XLK": "Technology Sector",
    "XLV": "Healthcare Sector",
    # Global Equity
    "EFA": "EAFE (Developed ex-US)",
    "EEM": "Emerging Markets",
    # Fixed Income
    "TLT": "20+ Year Treasury Bond",
    "IEF": "7-10 Year Treasury Bond",
    "SHY": "1-3 Year Treasury Bond",
    "HYG": "High Yield Corporate Bond",
    "LQD": "Investment Grade Corp Bond",
    # Commodities
    "GLD": "Gold ETF",
    "USO": "Oil ETF",
    "DBA": "Agriculture ETF",
    # Volatility
    "^VIX": "CBOE Volatility Index",
    # FX (via ETFs for reliable history)
    "UUP": "US Dollar Index ETF",
}

# For longer history (pre-ETF era), use these indices directly
MARKET_INDICES = {
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones Industrial",
    "^IXIC": "Nasdaq Composite",
    "^TNX": "10-Year Treasury Yield",
    "^VIX": "VIX",
    "^GSPC": "S&P 500",
    "GC=F": "Gold Futures",
    "CL=F": "Crude Oil Futures",
    "DX-Y.NYB": "US Dollar Index",
}


def download_market():
    """Download daily market data from Yahoo Finance."""
    import yfinance as yf

    print("\n=== Yahoo Finance: Market Data ===")
    out_dir = RAW_DIR / "market"
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / "market_prices.csv"
    if dest.exists():
        print(f"  [skip] Market data already exists: {dest.name}")
        df = pd.read_csv(dest, index_col=0, parse_dates=True)
        print(f"  Shape: {df.shape}")
        return df

    tickers = list(MARKET_INDICES.keys())
    print(f"  Downloading {len(tickers)} tickers: {tickers}")

    all_close = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start="1980-01-01", end="2018-01-01",
                               progress=False, auto_adjust=True)
            if len(data) > 0:
                close = data["Close"].copy()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                close.name = ticker
                all_close.append(close)
                print(f"    {ticker}: {len(data)} days "
                      f"({data.index.min().date()} to {data.index.max().date()})")
            else:
                print(f"    {ticker}: NO DATA")
        except Exception as e:
            print(f"    {ticker}: ERROR - {e}")

    df = pd.concat(all_close, axis=1)
    df.index.name = "date"
    df.to_csv(dest)
    print(f"  [done] Market prices shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 6. FRED -- Macro Targets
# ---------------------------------------------------------------------------

FRED_SERIES = {
    # Validation targets (slow-moving macro)
    "CPIAUCSL": "CPI All Urban Consumers",
    "UNRATE": "Unemployment Rate",
    "GDPC1": "Real GDP (Quarterly)",
    "USREC": "NBER Recession Indicator",
    "UMCSENT": "U. Michigan Consumer Sentiment",
    "INDPRO": "Industrial Production Index",
    "PAYEMS": "Total Nonfarm Payrolls",
    # Additional financial conditions
    "FEDFUNDS": "Federal Funds Rate",
    "T10Y2Y": "10Y-2Y Treasury Spread",
    "BAAFFM": "Baa Corporate Bond Spread",
    "DTWEXBGS": "Trade-Weighted USD Index",
}


def download_fred():
    """Download macro series from FRED."""
    print("\n=== FRED: Macro Validation Targets ===")
    out_dir = RAW_DIR / "fred"
    out_dir.mkdir(parents=True, exist_ok=True)

    dest = out_dir / "fred_macro.csv"
    if dest.exists():
        print(f"  [skip] FRED data already exists: {dest.name}")
        df = pd.read_csv(dest, index_col=0, parse_dates=True)
        print(f"  Shape: {df.shape}")
        return df

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        api_key_file = Path(__file__).resolve().parents[2] / ".fred_api_key"
        if api_key_file.exists():
            api_key = api_key_file.read_text().strip()

    if not api_key:
        print("  [warn] No FRED API key found.")
        print("         Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("         Then either:")
        print("           export FRED_API_KEY=your_key_here")
        print("           or save it to: .fred_api_key")
        print("")
        print("  Falling back to CSV download from FRED website...")
        return _download_fred_csv_fallback(out_dir, dest)

    from fredapi import Fred
    fred = Fred(api_key=api_key)

    all_data = {}
    for series_id, description in FRED_SERIES.items():
        try:
            s = fred.get_series(series_id, observation_start="1980-01-01",
                                observation_end="2018-01-01")
            all_data[series_id] = s
            print(f"    {series_id} ({description}): {len(s)} obs")
        except Exception as e:
            print(f"    {series_id}: ERROR - {e}")

    df = pd.DataFrame(all_data)
    df.index.name = "date"
    df.to_csv(dest)
    print(f"  [done] FRED data shape: {df.shape}")
    return df


def _download_fred_csv_fallback(out_dir: Path, dest: Path) -> pd.DataFrame:
    """Download key FRED series via direct CSV URLs (no API key needed)."""
    base = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    all_data = {}
    for series_id, description in FRED_SERIES.items():
        try:
            url = f"{base}?bgcolor=%23e1e9f0&chart_type=line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars=on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend=yes&show_axis_titles=yes&show_tooltip=yes&id={series_id}&scale=left&cosd=1980-01-01&coed=2018-01-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw=3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Daily&fam=avg&fgst=lin&fgsnd=2020-02-01&line_index=1&transformation=lin&vintage_date=2026-03-17&revision_date=2026-03-17&nd=1947-01-01"
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100:
                tmp = out_dir / f"{series_id}.csv"
                tmp.write_bytes(resp.content)
                s = pd.read_csv(tmp, index_col=0, parse_dates=True).iloc[:, 0]
                s = pd.to_numeric(s, errors="coerce")
                all_data[series_id] = s
                print(f"    {series_id} ({description}): {s.dropna().shape[0]} obs")
                tmp.unlink()
            else:
                print(f"    {series_id}: could not download (status {resp.status_code})")
        except Exception as e:
            print(f"    {series_id}: ERROR - {e}")

    if all_data:
        df = pd.DataFrame(all_data)
        df.index.name = "date"
        df.to_csv(dest)
        print(f"  [done] FRED data shape: {df.shape}")
        return df
    else:
        print("  [fail] No FRED data downloaded. Please set up API key.")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCES = {
    "bybee": download_bybee,
    "epu": download_epu,
    "gpr": download_gpr,
    "sentiment": download_fed_sentiment,
    "market": download_market,
    "fred": download_fred,
}


def main():
    parser = argparse.ArgumentParser(description="Download raw data for Semantic Day Embedding")
    parser.add_argument("--source", type=str, default=None,
                        choices=list(SOURCES.keys()),
                        help="Download a specific source (default: all)")
    args = parser.parse_args()

    print(f"Raw data directory: {RAW_DIR}")
    print(f"Sample period: {SAMPLE_START} to {SAMPLE_END}")

    if args.source:
        SOURCES[args.source]()
    else:
        for name, fn in SOURCES.items():
            try:
                fn()
            except Exception as e:
                print(f"\n  [ERROR] {name}: {e}")
                import traceback
                traceback.print_exc()

    print("\n=== All downloads complete ===")
    print(f"Data saved to: {RAW_DIR}")


if __name__ == "__main__":
    main()
