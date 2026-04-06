"""
Historical analog case study: cosine similarity of DSSDE embeddings on reference dates.

Loads embeddings from output/v14/reference_model/embeddings.npz (created by evaluation).

The publication table with citations and interpretation notes is maintained at
paper_draft/results/tables/historical_analogs.tex. Run this script to verify numeric
ranks and cosines after regenerating embeddings.

Usage:
  python scripts/historical_analog_case_study.py
  python scripts/historical_analog_case_study.py --gap 252
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

CRISIS_EVENTS: list[tuple[str, str, str]] = [
    ("Gulf War onset (Iraq invades Kuwait)", "Gulf War", "1990-08-02"),
    ("LTCM / autumn 1998 deleveraging", "LTCM", "1998-09-23"),
    ("Lehman Brothers bankruptcy", "Lehman", "2008-09-15"),
]

EXPANSION_EVENTS: list[tuple[str, str, str]] = [
    ("Nasdaq composite peak (dot-com euphoria)", "Nasdaq peak", "2000-03-10"),
    ("Mid-1990s Goldilocks expansion", "Goldilocks", "1996-01-15"),
    ("Mid-2000s credit expansion and housing boom", "Credit boom", "2006-06-15"),
]

DEFAULT_EVENTS = CRISIS_EVENTS + EXPANSION_EVENTS
DEFAULT_GAPS = [21, 252]


def _asof_index(dt: pd.Timestamp, idx: pd.DatetimeIndex) -> int:
    dt = pd.Timestamp(dt).normalize()
    if dt in idx:
        return int(idx.get_loc(dt))
    pos = idx.searchsorted(dt)
    if pos == 0:
        raise ValueError(f"{dt} is before the sample start")
    return int(pos - 1)


def _load_z_dates(repo: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    npz = repo / "output" / "v14" / "reference_model" / "embeddings.npz"
    if not npz.is_file():
        raise FileNotFoundError(f"Missing {npz}. Run evaluation once to save embeddings.npz.")
    d = np.load(npz, allow_pickle=True)
    z = np.asarray(d["z"], dtype=np.float64)
    dates = pd.DatetimeIndex(pd.to_datetime(d["dates"]))
    return z, dates


def top_prior_analogs(
    z: np.ndarray,
    query_i: int,
    gap: int,
    k: int = 2,
) -> list[tuple[int, float]]:
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    q = zn[query_i]
    sims = zn @ q
    end = query_i - gap
    if end < 1:
        return []
    block = sims[:end]
    top = np.argsort(-block)[:k]
    return [(int(j), float(block[j])) for j in top]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--gap",
        type=int,
        nargs="+",
        default=DEFAULT_GAPS,
        help="Minimum trading-day lag(s) (default: 21 252).",
    )
    args = p.parse_args()

    z, dates = _load_z_dates(REPO)

    for section, events in [("CRISIS", CRISIS_EVENTS), ("EXPANSION", EXPANSION_EVENTS)]:
        print(f"\n{'='*60}")
        print(f"  {section} EPISODES")
        print(f"{'='*60}")
        for label, _short, ds in events:
            i = _asof_index(pd.Timestamp(ds), dates)
            used = str(dates[i].date())
            print(f"\n  {label} - event day in data: {used} (index {i})")
            for gap in args.gap:
                results = top_prior_analogs(z, i, gap=gap, k=2)
                print(f"    gap >= {gap} td:")
                for rank, (j, c) in enumerate(results, 1):
                    cal_days = (dates[i] - dates[j]).days
                    print(f"      {rank}. {dates[j].date()}  cosine={c:.4f}  ({cal_days} cal days)")


if __name__ == "__main__":
    main()
