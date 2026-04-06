"""
Generate individual t-SNE regime subplot PDFs with identical axes sizes.

All plots use fixed axes positions so Panel A plots match and Panel B plots match,
regardless of colorbar presence. Colorbar is drawn in a separate axes to avoid
shrinking the main plot.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "v14"
FIG_DIR = PROJECT_ROOT / "paper_draft" / "results" / "figures"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "daily_features.csv"

NBER_RECESSIONS = [
    ("1990-07-01", "1991-03-31"),
    ("2001-03-01", "2001-11-30"),
    ("2007-12-01", "2009-06-30"),
]

# Tighter layout: scatter uses most of figure, minimal margins.
FIG_W, FIG_H = 3.6, 3.0
FIG_W_A, FIG_H_A = FIG_W, FIG_H
FIG_W_B, FIG_H_B = FIG_W, FIG_H
# [left, bottom, width, height] - scatter fills ~85% of figure
AX_RECT = [0.06, 0.08, 0.72, 0.86]       # scatter axes
CBAR_RECT = [0.82, 0.12, 0.03, 0.70]     # colorbar, right edge

REGIME_PALETTE = [
    "#0571B0", "#CA0020", "#2CA02C", "#E08214",
    "#7570B3", "#E7298A", "#1B9E77", "#A6761D",
]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#555555",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.grid": False,
})


def _nber_mask(dates):
    mask = np.zeros(len(dates), dtype=bool)
    for s, e in NBER_RECESSIONS:
        mask |= (dates >= s) & (dates <= e)
    return mask


def _tsne_sub(z, dates, step=21, perp=20):
    z_sub = z[::step]
    dates_sub = dates[::step]
    nber_full = _nber_mask(dates)
    nber_sub = nber_full[::step]
    n_pca = min(5, z_sub.shape[1])
    z_r = PCA(n_components=n_pca).fit_transform(z_sub)
    perp = min(perp, len(z_sub) // 5)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                max_iter=2000, learning_rate="auto")
    z_2d = tsne.fit_transform(z_r)
    years = np.array([d.year for d in dates_sub])
    clusters = {}
    for k in [2, 4, 8]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lf = km.fit_predict(z)
        ami = adjusted_mutual_info_score(nber_full.astype(int), lf)
        clusters[k] = {"labels_sub": lf[::step], "labels_full": lf, "ami": ami}
    return z_2d, years, nber_sub, nber_full, clusters


def _square_ax(ax):
    ax.set_aspect("equal", adjustable="box")
    xl, yl = ax.get_xlim(), ax.get_ylim()
    r = max(xl[1] - xl[0], yl[1] - yl[0]) * 0.52
    xc = (xl[0] + xl[1]) / 2
    yc = (yl[0] + yl[1]) / 2
    ax.set_xlim(xc - r, xc + r)
    ax.set_ylim(yc - r, yc + r)


def _make_fig_ax(panel="A"):
    """Create figure with fixed size; panel 'A' = 1.1x, 'B' = 1.2x."""
    w, h = (FIG_W_A, FIG_H_A) if panel == "A" else (FIG_W_B, FIG_H_B)
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes(AX_RECT)
    return fig, ax


def _save(fig, outname):
    """Save with fixed dimensions so all Panel A (and all Panel B) PDFs match."""
    fig.savefig(FIG_DIR / outname)
    plt.close(fig)
    print(f"  -> {outname}")


def plot_year(z_2d, years, outname):
    fig, ax = _make_fig_ax(panel="A")
    sc = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=years, cmap="viridis",
                    s=12, alpha=0.75, edgecolors="none")
    cax = fig.add_axes(CBAR_RECT)
    fig.colorbar(sc, cax=cax)
    cax.set_ylabel("Year", fontsize=9)
    cax.tick_params(labelsize=7.5)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=7.5)
    _square_ax(ax)
    _save(fig, outname)


def plot_nber(z_2d, nber, outname):
    fig, ax = _make_fig_ax(panel="A")
    exp = ~nber
    ax.scatter(z_2d[exp, 0], z_2d[exp, 1], c="#4393C3", s=12,
               alpha=0.6, edgecolors="none", label="Expansion")
    ax.scatter(z_2d[nber, 0], z_2d[nber, 1], c="#D6604D", s=16,
               alpha=0.85, edgecolors="none", label="Recession", marker="s")
    cax = fig.add_axes(CBAR_RECT)
    cax.axis("off")
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9,
              edgecolor="#CCCCCC", markerscale=1.0,
              handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=7.5)
    _square_ax(ax)
    _save(fig, outname)


def plot_kmeans(z_2d, labels, k, ami, nber, outname, show_colorbar):
    fig, ax = _make_fig_ax(panel="B")
    pal = REGIME_PALETTE[:k]
    cmap = ListedColormap(pal)
    bounds = np.arange(-0.5, k + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    exp_mask = ~nber
    rec_mask = nber
    ax.scatter(z_2d[exp_mask, 0], z_2d[exp_mask, 1],
               c=labels[exp_mask], cmap=cmap, norm=norm,
               s=12, alpha=0.7, edgecolors="none", marker="o")
    ax.scatter(z_2d[rec_mask, 0], z_2d[rec_mask, 1],
               c=labels[rec_mask], cmap=cmap, norm=norm,
               s=22, alpha=0.9, edgecolors="black", linewidths=0.4, marker="s")
    cax = fig.add_axes(CBAR_RECT)
    if show_colorbar:
        from matplotlib.cm import ScalarMappable
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax, ticks=range(k))
        cb.set_label("Regime", fontsize=9)
        cb.ax.set_yticklabels([str(i + 1) for i in range(k)], fontsize=7.5)
    else:
        cax.axis("off")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=5, label="Expansion"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markeredgecolor="black", markeredgewidth=0.4,
               markersize=5, label="Recession"),
    ]
    ax.legend(handles=legend_elements, fontsize=6, loc="upper right",
              framealpha=0.9, edgecolor="#CCCCCC",
              handletextpad=0.3, borderpad=0.3, labelspacing=0.25)
    ax.set_title(f"AMI = {ami:.3f}", fontsize=9, pad=5)
    ax.set_xlabel("t-SNE 1", fontsize=9)
    ax.set_ylabel("t-SNE 2", fontsize=9)
    ax.tick_params(labelsize=7.5)
    _square_ax(ax)
    _save(fig, outname)


def _run_length_encode(labels):
    """Convert label array to (start_idx, end_idx, label) runs."""
    runs = []
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[start]:
            runs.append((start, i, labels[start]))
            start = i
    runs.append((start, len(labels), labels[start]))
    return runs


def plot_regime_timeline(z, dates, nber_full, outname):
    """Cluster monthly-averaged embeddings, then paint each day with its month's label."""
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch
    from collections import Counter

    df_z = pd.DataFrame(z, index=dates)
    monthly = df_z.resample("M").mean().dropna()
    month_keys = monthly.index.to_period("M")

    nber_monthly = pd.Series(nber_full.astype(int), index=dates).resample("M").mean()
    nber_monthly_bool = (nber_monthly > 0.5).reindex(monthly.index).values

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 3.2), sharex=True,
                             gridspec_kw={"hspace": 0.18})

    for ax, k in zip(axes, [2, 4, 8]):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        month_labels = km.fit_predict(monthly.values)

        ami = adjusted_mutual_info_score(nber_monthly_bool.astype(int), month_labels)

        day_periods = dates.to_period("M")
        month_label_map = dict(zip(month_keys, month_labels))
        day_labels = np.array([month_label_map.get(p, -1) for p in day_periods])

        pal = REGIME_PALETTE[:k]

        rec_labels = day_labels[nber_full & (day_labels >= 0)]
        if len(rec_labels) > 0:
            counts = Counter(int(l) for l in rec_labels)
            recession_cluster = counts.most_common(1)[0][0]
        else:
            recession_cluster = 0

        reorder = {}
        next_id = 0
        reorder[recession_cluster] = next_id
        next_id += 1
        for c in range(k):
            if c not in reorder:
                reorder[c] = next_id
                next_id += 1
        labels_reordered = np.array([reorder[l] for l in day_labels])

        date_nums = mdates.date2num(dates)
        runs = _run_length_encode(labels_reordered)
        for s_idx, e_idx, lab in runs:
            if lab < 0:
                continue
            ax.axvspan(date_nums[s_idx],
                       date_nums[min(e_idx, len(date_nums) - 1)],
                       facecolor=pal[lab], alpha=0.85, edgecolor="none")

        for s, e in NBER_RECESSIONS:
            s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
            if e_ts < dates[0] or s_ts > dates[-1]:
                continue
            s_clip = max(s_ts, dates[0])
            e_clip = min(e_ts, dates[-1])
            ax.axvspan(mdates.date2num(s_clip), mdates.date2num(e_clip),
                       facecolor="none", edgecolor="black",
                       linewidth=1.3, linestyle="-", zorder=5)

        ax.set_xlim(date_nums[0], date_nums[-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(f"$k={k}$\nAMI={ami:.2f}", fontsize=8, rotation=0,
                       labelpad=35, va="center")
        ax.tick_params(axis="x", labelsize=7.5)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    legend_elements = [
        Patch(facecolor="white", edgecolor="black", linewidth=1.3,
              label="NBER recession"),
    ]
    leg = fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=1,
        fontsize=6.5,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fancybox=False,
        handletextpad=0.4,
        borderpad=0.35,
    )

    fig.savefig(
        FIG_DIR / outname,
        bbox_inches="tight",
        bbox_extra_artists=(leg,),
        dpi=300,
    )
    plt.close(fig)
    print(f"  -> {outname}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("DSSDE regime t-SNE ...")
    npz = np.load(OUTPUT_DIR / "embeddings_reference.npz", allow_pickle=True)
    z = npz["z"]
    dates = pd.DatetimeIndex(npz["dates"])
    z_2d, years, nber_sub, nber_full, clusters = _tsne_sub(z, dates)
    plot_year(z_2d, years, "regime_dssde_year.pdf")
    plot_nber(z_2d, nber_sub, "regime_dssde_nber.pdf")
    for k in [2, 4, 8]:
        plot_kmeans(z_2d, clusters[k]["labels_sub"], k,
                    clusters[k]["ami"], nber_sub, f"regime_dssde_k{k}.pdf",
                    show_colorbar=(k == 8))
    plot_regime_timeline(z, dates, nber_full, "regime_dssde_timeline.pdf")

    print("PCA regime t-SNE ...")
    df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True).sort_index()
    vals = df.fillna(0.0).values.astype(np.float32)
    dates_pca = df.index
    tm = dates_pca <= pd.Timestamp("2011-12-31")
    scaler = StandardScaler()
    scaler.fit(vals[tm])
    vals_n = scaler.transform(vals)
    pca = PCA(n_components=64, random_state=42)
    pca.fit(vals_n[tm])
    z_pca = pca.transform(vals_n)
    z_2d, years, nber_sub, nber_full_pca, clusters = _tsne_sub(z_pca, dates_pca)
    plot_year(z_2d, years, "regime_pca_year.pdf")
    plot_nber(z_2d, nber_sub, "regime_pca_nber.pdf")
    for k in [2, 4, 8]:
        plot_kmeans(z_2d, clusters[k]["labels_sub"], k,
                    clusters[k]["ami"], nber_sub, f"regime_pca_k{k}.pdf",
                    show_colorbar=(k == 8))
    plot_regime_timeline(z_pca, dates_pca, nber_full_pca, "regime_pca_timeline.pdf")

    print("Done.")


if __name__ == "__main__":
    main()
