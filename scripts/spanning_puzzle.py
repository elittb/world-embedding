"""
spanning_puzzle.py
==================
Test whether the DSSDE embedding resolves the interest-rate spanning puzzle.

Literature:
  - Cochrane & Piazzesi (2005, AER): forward rates predict bond excess returns
  - Ludvigson & Ng (2009, RFS): macro factors predict beyond yield curve PCs
  - Duffee (2011, RFS): hidden factor in term structure predicts returns
  - Joslin, Priebsch & Singleton (2014, JF): unspanned macro risks

Methodology:
  1. Construct monthly bond excess returns from FRED constant-maturity yields
     using the duration-approximation (standard for par yields).
  2. Extract yield-curve PCs (level, slope, curvature) and DSSDE embedding PCs.
  3. In-sample: OLS with Newey-West SEs; test ΔR² via partial F-test.
  4. Out-of-sample: expanding-window OLS, Campbell-Thompson OOS R²,
     Giacomini-White tests for predictive comparison.
  5. Clark-West (2007) MSPE-adjusted tests for nested model comparisons.
"""

import warnings, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

# ── paths ─────────────────────────────────────────────────────────────
ROOT = Path("/Users/elittb/Library/CloudStorage/Dropbox/4th Paper_Time-Embedding")
DATA = ROOT / "data" / "processed"
OUT  = ROOT / "output" / "v14"
RES  = OUT / "results"
RES.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════
# 1.  LOAD DATA
# ═══════════════════════════════════════════════════════════════════════
print("Loading data …")
features = pd.read_csv(DATA / "daily_features.csv",
                       index_col="date", parse_dates=True)

emb_data  = np.load(OUT / "embeddings_reference.npz", allow_pickle=True)
emb_z     = emb_data["z"]            # (8608, 64)
emb_dates = pd.to_datetime(emb_data["dates"])

yield_names = {1: "fred_dgs1", 2: "fred_dgs2",
               5: "fred_dgs5", 10: "fred_dgs10", 30: "fred_dgs30"}

yields_daily = features[list(yield_names.values())].copy()
yields_daily.columns = list(yield_names.keys())

emb_df = pd.DataFrame(emb_z, index=emb_dates,
                       columns=[f"z{i}" for i in range(64)])


# ═══════════════════════════════════════════════════════════════════════
# 2.  END-OF-MONTH OBSERVATIONS
# ═══════════════════════════════════════════════════════════════════════
yields_m = yields_daily.resample("ME").last().dropna()
emb_m    = emb_df.resample("ME").last()

cidx = yields_m.index.intersection(emb_m.index)
yields_m = yields_m.loc[cidx]
emb_m    = emb_m.loc[cidx]

print(f"Monthly sample: {cidx[0].strftime('%Y-%m')} – "
      f"{cidx[-1].strftime('%Y-%m')}  (T = {len(cidx)})")


# ═══════════════════════════════════════════════════════════════════════
# 3.  YIELD-CURVE PCs  (level, slope, curvature)
# ═══════════════════════════════════════════════════════════════════════
yscaler = StandardScaler()
yields_sc = yscaler.fit_transform(yields_m.values)
ypca = PCA(n_components=3)
ypc_vals = ypca.fit_transform(yields_sc)
ypc = pd.DataFrame(ypc_vals, index=cidx,
                    columns=["YPC1", "YPC2", "YPC3"])

print(f"Yield-curve PCA var explained: "
      f"{ypca.explained_variance_ratio_.round(3)}")


# ═══════════════════════════════════════════════════════════════════════
# 4.  EMBEDDING PCs  (first N_EPC components)
# ═══════════════════════════════════════════════════════════════════════
N_EPC = 5                               # baseline; robustness with 3, 8
escaler = StandardScaler()
emb_sc = escaler.fit_transform(emb_m.values)
epca = PCA(n_components=N_EPC)
epc_vals = epca.fit_transform(emb_sc)
epc = pd.DataFrame(epc_vals, index=cidx,
                    columns=[f"EPC{i+1}" for i in range(N_EPC)])

print(f"Embedding PCA var explained (first {N_EPC}): "
      f"{epca.explained_variance_ratio_.round(3)}")


# ═══════════════════════════════════════════════════════════════════════
# 5.  BOND EXCESS RETURNS  (duration approximation)
# ═══════════════════════════════════════════════════════════════════════
#   rx_{t→t+1}^{(n)} ≈ carry - D_n · Δy_{t+1}^{(n)}
#   carry  = (y_t^{(n)} − y_t^{(1)}) / 12          (monthly term premium)
#   D_n    = (1/y)[1 − (1+y/2)^{−2n}] / (1+y/2)    (semiannual mod duration)
#
# For h-month horizons we cumulate 1-month returns.

def mod_duration(n, y_pct):
    """Modified duration for a US Treasury paying semiannual coupons at par."""
    y = y_pct / 100.0
    if y < 1e-8:
        return float(n)
    return (1.0 / (y / 2.0)) * (1.0 - (1.0 + y / 2.0) ** (-2 * n)) / (1.0 + y / 2.0)


maturities = [2, 5, 10, 30]
horizons   = [1, 3, 12]              # months

# 1-month excess returns for each maturity
rx1 = pd.DataFrame(index=cidx)
for n in maturities:
    carry = (yields_m[n] - yields_m[1]) / 12.0 / 100.0          # decimal
    dur   = yields_m[n].apply(lambda y: mod_duration(n, y))      # series
    dy    = yields_m[n].diff() / 100.0                           # decimal
    rx1[f"rx{n}"] = carry - dur * dy

rx1["rx_avg"] = rx1[[f"rx{n}" for n in maturities]].mean(axis=1)

# Cumulative h-month excess returns  (overlapping for h > 1)
rx_all = {}
for h in horizons:
    rx_h = pd.DataFrame(index=cidx)
    for col in rx1.columns:
        rx_h[col] = rx1[col].rolling(h).sum()
    rx_all[h] = rx_h

# Drop leading NaN rows from differencing / rolling
for h in horizons:
    rx_all[h] = rx_all[h].iloc[max(h, 1):]


# ═══════════════════════════════════════════════════════════════════════
# 6.  ALIGN PREDICTORS  (lag by 1 month relative to return)
# ═══════════════════════════════════════════════════════════════════════
# Return from month t to t+h is predicted by PCs at end of month t−1.
ypc_lag = ypc.shift(1)
epc_lag = epc.shift(1)

def align(rx_df, lag_start=2):
    """Drop initial NaN rows, return aligned (rx, ypc_l, epc_l)."""
    valid = rx_df.index[lag_start:]
    valid = valid.intersection(ypc_lag.dropna().index).intersection(
                               epc_lag.dropna().index)
    return rx_df.loc[valid], ypc_lag.loc[valid], epc_lag.loc[valid]


# ═══════════════════════════════════════════════════════════════════════
# 7.  IN-SAMPLE SPANNING TESTS
# ═══════════════════════════════════════════════════════════════════════
targets = ["rx2", "rx5", "rx10", "rx_avg"]

def nw_ols(y, X, nw_lags):
    Xc = sm.add_constant(X)
    return sm.OLS(y, Xc).fit(cov_type="HAC",
                              cov_kwds={"maxlags": nw_lags})

all_is = []

print("\n" + "=" * 72)
print("IN-SAMPLE SPANNING TESTS")
print("=" * 72)

for h in horizons:
    rx_h, ypc_h, epc_h = align(rx_all[h])
    nw = max(h + 1, 6)                       # Newey-West lag order

    print(f"\n── Horizon h = {h} month(s)  [T = {len(rx_h)}, NW({nw})] ──")

    for tgt in targets:
        y = rx_h[tgt].values

        # Model 1: yield PCs only
        m1 = nw_ols(y, ypc_h.values, nw)

        # Model 2: yield PCs + embedding PCs
        X2 = np.column_stack([ypc_h.values, epc_h.values])
        m2 = nw_ols(y, X2, nw)

        # Model 3: embedding PCs only
        m3 = nw_ols(y, epc_h.values, nw)

        # Partial F-test for incremental R²
        k1, k2, T = ypc_h.shape[1], epc_h.shape[1], len(y)
        dr2 = m2.rsquared - m1.rsquared
        f_num = dr2 / k2
        f_den = (1.0 - m2.rsquared) / (T - k1 - k2 - 1)
        f_stat = f_num / f_den if f_den > 0 else np.nan
        f_pval = 1.0 - stats.f.cdf(f_stat, k2, T - k1 - k2 - 1)

        # Significant embedding PCs in augmented model
        emb_pvals = m2.pvalues[1 + k1:]       # skip const + yield PCs
        emb_coefs = m2.params[1 + k1:]
        sig_mask  = emb_pvals < 0.10
        sig_str   = ", ".join(
            f"EPC{i+1}(p={emb_pvals[i]:.3f})"
            for i in range(k2) if sig_mask[i])

        row = dict(horizon=h, target=tgt,
                   R2_yield=m1.rsquared, R2_aug=m2.rsquared,
                   R2_emb=m3.rsquared, dR2=dr2,
                   F_stat=f_stat, F_pval=f_pval)
        all_is.append(row)

        stars = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 \
                else "*" if f_pval < 0.10 else ""
        print(f"  {tgt:8s}  R²_y={m1.rsquared:6.3f}  R²_aug={m2.rsquared:6.3f}  "
              f"ΔR²={dr2:6.3f}  F={f_stat:6.2f}{stars:3s}  "
              f"R²_emb={m3.rsquared:6.3f}"
              + (f"  [{sig_str}]" if sig_str else ""))


# ═══════════════════════════════════════════════════════════════════════
# 8.  OUT-OF-SAMPLE EVALUATION  (expanding window)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("OUT-OF-SAMPLE EVALUATION  (expanding window, OLS)")
print("=" * 72)

MIN_TRAIN = 120                               # 10 years initial window

all_oos = []

for h in horizons:
    rx_h, ypc_h, epc_h = align(rx_all[h])
    T = len(rx_h)
    if T <= MIN_TRAIN + 12:
        print(f"\n── h={h}: sample too short for OOS ──")
        continue

    print(f"\n── Horizon h = {h} month(s)  [OOS window: "
          f"{rx_h.index[MIN_TRAIN].strftime('%Y-%m')} – "
          f"{rx_h.index[-1].strftime('%Y-%m')}] ──")

    for tgt in targets:
        y = rx_h[tgt].values
        Xy = ypc_h.values
        Xe = epc_h.values
        X_aug = np.column_stack([Xy, Xe])

        preds_y, preds_a, actuals, hist_means = [], [], [], []

        for t0 in range(MIN_TRAIN, T):
            y_tr = y[:t0]
            # --- yield PCs only ---
            Xc = sm.add_constant(Xy[:t0])
            m1 = sm.OLS(y_tr, Xc).fit()
            Xc_te = np.concatenate([[1.0], Xy[t0]]).reshape(1, -1)
            preds_y.append(m1.predict(Xc_te)[0])

            # --- augmented ---
            Xc2 = sm.add_constant(X_aug[:t0])
            m2  = sm.OLS(y_tr, Xc2).fit()
            Xc2_te = np.concatenate([[1.0], X_aug[t0]]).reshape(1, -1)
            preds_a.append(m2.predict(Xc2_te)[0])

            actuals.append(y[t0])
            hist_means.append(y_tr.mean())

        actual = np.array(actuals)
        pred_y = np.array(preds_y)
        pred_a = np.array(preds_a)
        hm     = np.array(hist_means)

        # Campbell-Thompson OOS R²
        sse_y = np.sum((actual - pred_y) ** 2)
        sse_a = np.sum((actual - pred_a) ** 2)
        sst   = np.sum((actual - hm) ** 2)
        r2oos_y = 1.0 - sse_y / sst
        r2oos_a = 1.0 - sse_a / sst

        # --- Giacomini-White test  (augmented vs yield-only) ---
        loss_y = (actual - pred_y) ** 2
        loss_a = (actual - pred_a) ** 2
        d = loss_y - loss_a          # positive → augmented better
        d_bar = d.mean()
        nw_lag = max(h + 1, 6)

        def _nw_tstat(series):
            """HAC t-statistic for mean(series) = 0 using Bartlett kernel."""
            s_bar = series.mean()
            n = len(series)
            g0 = np.var(series, ddof=1)
            gs = 0.0
            for j in range(1, nw_lag + 1):
                w = 1.0 - j / (nw_lag + 1.0)
                gs += 2 * w * np.mean((series[j:] - s_bar) * (series[:-j] - s_bar))
            v = g0 + gs
            if v > 0:
                return s_bar / np.sqrt(v / n)
            return np.nan

        t_gw = _nw_tstat(d)
        p_gw = 2 * (1.0 - stats.norm.cdf(abs(t_gw))) if np.isfinite(t_gw) else np.nan

        # --- Clark-West (2007) MSPE-adjusted test for nested models ---
        # Adjustment adds (pred_y - pred_a)^2 to the loss differential,
        # correcting for parameter estimation noise under the null.
        cw_adj = (pred_y - pred_a) ** 2
        d_cw = d + cw_adj
        t_cw = _nw_tstat(d_cw)
        p_cw = 1.0 - stats.norm.cdf(t_cw) if np.isfinite(t_cw) else np.nan

        all_oos.append(dict(horizon=h, target=tgt,
                            R2oos_yield=r2oos_y, R2oos_aug=r2oos_a,
                            dR2oos=r2oos_a - r2oos_y,
                            t_GW=t_gw, p_GW=p_gw,
                            t_CW=t_cw, p_CW=p_cw))

        gw_star = "***" if p_gw < 0.01 else "**" if p_gw < 0.05 \
                  else "*" if p_gw < 0.10 else ""
        cw_star = "***" if p_cw < 0.01 else "**" if p_cw < 0.05 \
                  else "*" if p_cw < 0.10 else ""
        print(f"  {tgt:8s}  R²oos_y={r2oos_y:7.4f}  R²oos_aug={r2oos_a:7.4f}  "
              f"ΔR²={r2oos_a - r2oos_y:7.4f}  "
              f"t_GW={t_gw:6.2f}{gw_star:3s} (p={p_gw:.3f})  "
              f"t_CW={t_cw:6.2f}{cw_star:3s} (p={p_cw:.3f})")


# ═══════════════════════════════════════════════════════════════════════
# 9.  ROBUSTNESS: NUMBER OF EMBEDDING PCs
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("ROBUSTNESS: varying number of embedding PCs (h=1, in-sample)")
print("=" * 72)

h = 1
rx_h, ypc_h, _ = align(rx_all[h])
nw = 6
rob_rows = []

for n_epc in [3, 5, 8]:
    epca_r = PCA(n_components=n_epc)
    epc_r  = pd.DataFrame(epca_r.fit_transform(emb_sc),
                           index=cidx,
                           columns=[f"EPC{i+1}" for i in range(n_epc)])
    epc_r_lag = epc_r.shift(1)
    epc_r_h   = epc_r_lag.loc[rx_h.index]

    print(f"\n  N_EPC = {n_epc}")
    for tgt in targets:
        y  = rx_h[tgt].values
        X2 = np.column_stack([ypc_h.values, epc_r_h.values])
        m1 = nw_ols(y, ypc_h.values, nw)
        m2 = nw_ols(y, X2, nw)
        dr2 = m2.rsquared - m1.rsquared
        k1, k2, T = ypc_h.shape[1], n_epc, len(y)
        f_stat = (dr2 / k2) / ((1 - m2.rsquared) / (T - k1 - k2 - 1))
        f_pval = 1 - stats.f.cdf(f_stat, k2, T - k1 - k2 - 1)
        rob_rows.append(dict(n_epc=n_epc, target=tgt,
                             R2_yield=m1.rsquared, R2_aug=m2.rsquared,
                             dR2=dr2, F_stat=f_stat, F_pval=f_pval))
        stars = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 \
                else "*" if f_pval < 0.10 else ""
        print(f"    {tgt:8s}  R²_y={m1.rsquared:6.3f}  "
              f"R²_aug={m2.rsquared:6.3f}  ΔR²={dr2:6.3f}  "
              f"F={f_stat:5.2f}{stars:3s}")


# ═══════════════════════════════════════════════════════════════════════
# 10. SAVE RESULTS
# ═══════════════════════════════════════════════════════════════════════
is_df  = pd.DataFrame(all_is)
oos_df = pd.DataFrame(all_oos)
rob_df = pd.DataFrame(rob_rows)

is_df.to_csv(RES / "spanning_insample.csv",  index=False)
oos_df.to_csv(RES / "spanning_oos.csv",      index=False)
rob_df.to_csv(RES / "spanning_robustness.csv", index=False)

print("\n✓  Results saved to", RES)


# ═══════════════════════════════════════════════════════════════════════
# 11. INDIVIDUAL COEFFICIENT TABLE  (all horizons, full augmented model)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("COEFFICIENT TABLE: augmented model, all horizons")
print("=" * 72)

coef_rows = []

for h in horizons:
    rx_h, ypc_h, epc_h = align(rx_all[h])
    nw = max(h + 1, 6)

    print(f"\n── Horizon h = {h} month(s) ──")

    for tgt in targets:
        y  = rx_h[tgt].values
        X2 = np.column_stack([ypc_h.values, epc_h.values])
        m2 = nw_ols(y, X2, nw)

        names = ["const"] + list(ypc_h.columns) + list(epc_h.columns)
        print(f"\n  {tgt}:")
        for i, nm in enumerate(names):
            coef = m2.params[i]
            se   = m2.bse[i]
            t_   = m2.tvalues[i]
            p_   = m2.pvalues[i]
            st   = "***" if p_ < 0.01 else "**" if p_ < 0.05 \
                   else "*" if p_ < 0.10 else ""
            print(f"    {nm:8s}  β={coef:9.5f}  se={se:9.5f}  "
                  f"t={t_:6.2f}{st:3s}")
            coef_rows.append(dict(horizon=h, target=tgt, var=nm,
                                  coef=coef, se=se, tstat=t_, pval=p_))

coef_df = pd.DataFrame(coef_rows)
coef_df.to_csv(RES / "spanning_coefficients.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════
# 12. ORTHOGONALIZATION ROBUSTNESS
# ═══════════════════════════════════════════════════════════════════════
# Treasury yields enter both the embedding inputs and the bond-return
# construction. Although the incremental test controls for *linear*
# yield-curve PCs, the neural network could learn nonlinear functions of
# yields that are orthogonal to the PCs yet predictive of future returns.
# To rule this out, we orthogonalize each embedding PC against yield-curve
# PCs *and* their squares and pairwise interactions, then re-run the
# spanning test using only the yield-purged residuals.
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 72)
print("ORTHOGONALIZATION ROBUSTNESS  (EPCs purged of linear + nonlinear yield content)")
print("=" * 72)

def build_nonlinear_yield_basis(ypc_arr):
    """Expand 3 yield PCs into 3 + 3 squares + 3 cross-products = 9 regressors."""
    k = ypc_arr.shape[1]
    basis = [ypc_arr]
    basis.append(ypc_arr ** 2)
    for i in range(k):
        for j in range(i + 1, k):
            basis.append((ypc_arr[:, i] * ypc_arr[:, j]).reshape(-1, 1))
    return np.column_stack(basis)

orth_is_rows = []

for h in horizons:
    rx_h, ypc_h, epc_h = align(rx_all[h])
    nw = max(h + 1, 6)

    Y_basis = build_nonlinear_yield_basis(ypc_h.values)
    Y_basis_c = sm.add_constant(Y_basis)

    oepc_arr = np.empty_like(epc_h.values)
    r2_purge = []
    for j in range(epc_h.shape[1]):
        purge_mod = sm.OLS(epc_h.values[:, j], Y_basis_c).fit()
        oepc_arr[:, j] = purge_mod.resid
        r2_purge.append(purge_mod.rsquared)

    oepc = pd.DataFrame(oepc_arr, index=epc_h.index,
                         columns=[f"OEPC{i+1}" for i in range(epc_h.shape[1])])

    print(f"\n── Horizon h = {h} month(s) ──")
    print(f"   Yield content purged from EPCs  "
          f"(R² of purge regressions: {[f'{r:.3f}' for r in r2_purge]})")

    for tgt in targets:
        y = rx_h[tgt].values

        m1 = nw_ols(y, ypc_h.values, nw)

        X2 = np.column_stack([ypc_h.values, oepc_arr])
        m2 = nw_ols(y, X2, nw)

        k1, k2, T = ypc_h.shape[1], oepc.shape[1], len(y)
        dr2 = m2.rsquared - m1.rsquared
        f_num = dr2 / k2
        f_den = (1.0 - m2.rsquared) / (T - k1 - k2 - 1)
        f_stat = f_num / f_den if f_den > 0 else np.nan
        f_pval = 1.0 - stats.f.cdf(f_stat, k2, T - k1 - k2 - 1)

        orth_is_rows.append(dict(
            horizon=h, target=tgt,
            R2_yield=m1.rsquared, R2_aug_orth=m2.rsquared,
            dR2_orth=dr2, F_stat_orth=f_stat, F_pval_orth=f_pval,
        ))

        stars = "***" if f_pval < 0.01 else "**" if f_pval < 0.05 \
                else "*" if f_pval < 0.10 else ""
        print(f"  {tgt:8s}  R²_y={m1.rsquared:6.3f}  R²_aug(orth)={m2.rsquared:6.3f}  "
              f"ΔR²={dr2:6.3f}  F={f_stat:6.2f}{stars:3s}")

orth_df = pd.DataFrame(orth_is_rows)
orth_df.to_csv(RES / "spanning_orthogonalized.csv", index=False)
print(f"\n✓  Orthogonalization results saved to {RES / 'spanning_orthogonalized.csv'}")

print("\nDone.")
