"""Example: Replicate the bond spanning result from the paper.

Shows that world embedding PCs capture unspanned macro risks that improve
bond excess return prediction beyond yield-curve factors.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from worldembedding import load_embedding, get_principal_components


def construct_bond_excess_returns(maturity_years: int = 2) -> pd.Series:
    """Construct monthly bond excess returns from FRED Treasury yields.

    This is a simplified version. See the paper appendix for the full
    construction with continuously compounded returns.

    Parameters
    ----------
    maturity_years : int
        Bond maturity in years (2, 5, or 10).

    Returns
    -------
    pd.Series
        Monthly excess returns, indexed by date.
    """
    try:
        import fredapi
    except ImportError:
        raise ImportError(
            "fredapi is required for this example. "
            "Install with: pip install worldembedding[replication]"
        )

    import os
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "Set FRED_API_KEY environment variable. "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = fredapi.Fred(api_key=api_key)

    series_map = {2: "DGS2", 5: "DGS5", 10: "DGS10"}
    series_id = series_map[maturity_years]

    yields = fred.get_series(series_id, observation_start="1985-01-01")
    yields = yields.dropna()
    monthly = yields.resample("ME").last() / 100

    price_return = -maturity_years * monthly.diff()
    rf = fred.get_series("DGS1MO", observation_start="1985-01-01")
    rf = rf.resample("ME").last() / 100 / 12
    rx = price_return - rf
    rx.name = f"rx_{maturity_years}y"
    return rx.dropna()


def main():
    print("Loading world embedding...")
    epc = get_principal_components(n_components=5)
    epc_monthly = epc.resample("ME").last()

    print(
        "To run the full spanning test, set FRED_API_KEY and use the "
        "replication scripts in scripts/spanning_puzzle.py\n"
    )

    print("Embedding PCs (monthly, end-of-month):")
    print(epc_monthly.describe().round(3))
    print()

    print("Correlation between EPCs:")
    print(epc_monthly.corr().round(3))
    print()

    print(
        "For the full spanning regression results, see:\n"
        "  Table 5 (in-sample) and Table 7 (out-of-sample) in the paper.\n"
        "  https://papers.ssrn.com/abstract=6503446"
    )


if __name__ == "__main__":
    main()
