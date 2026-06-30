"""
RH2.2: Authentic videos will consistently achieve approximately zero lip closure
during plosive sounds, whereas deepfakes will frequently maintain a measurable
gap between the lips.

Method:
  For bilabial plosives /p/ and /b/ (and nasal /m/), load all individual MAR
  observations from dataset.db.  Define "lip closure" as MAR < threshold θ.
  Compare the closure rate (proportion of near-zero MAR events) between Fake
  and Real using a two-proportion Z-test and Fisher's exact test.
  Also compare the full MAR distributions with Mann–Whitney U and report
  effect size (rank-biserial correlation r).
  Visualise with KDE plots and a bar chart of closure rates.

  Threshold θ is set to the 10th percentile of the Real MAR distribution
  for each phoneme (data-driven, no arbitrary choice).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact
from statsmodels.stats.proportion import proportions_ztest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.rq_utils import DB_PATH, apply_style, save_plot

apply_style()

PLOSIVES = {"p": "/p/", "b": "/b/", "m": "/m/"}


def rank_biserial(u, n1, n2):
    """Effect size r for Mann–Whitney U."""
    return 1 - (2 * u) / (n1 * n2)


def run():
    print("=" * 65)
    print("RH2.2  Lip closure during plosives — Fake vs Real MAR")
    print("=" * 65)

    notes_lines = ["# RH2.2 — Interpretation\n",
                   "## Metric\n"
                   "Lip closure rate = proportion of phoneme observations with "
                   "MAR < θ, where θ is the 10th percentile of the Real MAR "
                   "distribution for each phoneme (data-driven threshold).\n"]

    closure_data = {}   # phoneme → {fake_rate, real_rate, n_fake, n_real}
    all_results  = {}

    for ph, lbl in PLOSIVES.items():
        print(f"\n{lbl} ('{ph}') …", flush=True)
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            "SELECT video_type, dataset, mar FROM lags WHERE phoneme=?",
            conn, params=(ph,))
        conn.close()

        fake_mar = df[df["video_type"] == "Fake"]["mar"].values
        real_mar = df[df["video_type"] == "Real"]["mar"].values
        print(f"  Observations — Fake: {len(fake_mar):,}  Real: {len(real_mar):,}")

        # Data-driven threshold: 10th percentile of Real distribution
        theta = float(np.percentile(real_mar, 10))
        print(f"  Threshold θ (Real p10) = {theta:.5f}")

        # Closure rates
        fake_closed = (fake_mar < theta).sum()
        real_closed = (real_mar < theta).sum()
        fake_rate   = fake_closed / len(fake_mar)
        real_rate   = real_closed / len(real_mar)
        print(f"  Closure rate — Fake: {fake_rate:.4f}  Real: {real_rate:.4f}")

        # Fisher's exact test on closure counts
        table = [[fake_closed,       len(fake_mar) - fake_closed],
                 [real_closed,       len(real_mar) - real_closed]]
        odds, p_fish = fisher_exact(table, alternative="less")
        print(f"  Fisher's exact (Fake rate < Real rate): OR={odds:.4f}  p={p_fish:.4e}")

        # Mann–Whitney U on full MAR distributions
        stat_mw, p_mw = mannwhitneyu(fake_mar, real_mar, alternative="greater")
        r_eff = rank_biserial(stat_mw, len(fake_mar), len(real_mar))
        print(f"  Mann–Whitney U (Fake MAR > Real MAR): U={stat_mw:.1f}  "
              f"p={p_mw:.4e}  r={r_eff:.4f}")

        closure_data[lbl] = dict(fake_rate=fake_rate, real_rate=real_rate,
                                 n_fake=len(fake_mar), n_real=len(real_mar))
        all_results[lbl]  = dict(theta=theta, fake_rate=fake_rate,
                                 real_rate=real_rate, p_fish=p_fish,
                                 p_mw=p_mw, r_eff=r_eff,
                                 fake_mar=fake_mar, real_mar=real_mar)

        notes_lines.append(
            f"### {lbl}  \nθ={theta:.5f}  "
            f"Fake closure={fake_rate:.4f}  Real closure={real_rate:.4f}  "
            f"Fisher p={p_fish:.2e}  Mann–Whitney p={p_mw:.2e}  r={r_eff:.4f}\n")

    supported = all(all_results[lbl]["p_fish"] < 0.05 for lbl in all_results)
    print(f"\n→ RH2.2 {'SUPPORTED' if supported else 'PARTIALLY supported / NOT supported'} "
          f"(Fisher's exact, α=0.05)")

    # ── Notes MD ─────────────────────────────────────────────────────────────
    notes_lines += [
        "\n## Statistical tests\n",
        "**Fisher's exact test (one-sided, Fake < Real):** Tests whether the "
        "proportion of near-zero MAR events is significantly lower in fake "
        "videos, i.e., that deepfakes maintain a residual mouth gap more often.\n\n",
        "**Mann–Whitney U (Fake MAR > Real MAR):** Tests the full distribution "
        "claim — that deepfake MAR values are systematically higher during "
        "plosives (less closure overall). Rank-biserial r is the effect size.\n\n",
        "**Threshold θ:** Set as the 10th percentile of the Real distribution "
        "per phoneme, avoiding an arbitrary absolute cutoff.\n",
    ]

    from scipy.stats import gaussian_kde

    # ── Plot 1: Closure rate bar chart ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    lbls  = list(closure_data.keys())
    x     = np.arange(len(lbls))
    width = 0.35
    ax.bar(x - width/2,
           [closure_data[l]["fake_rate"] for l in lbls],
           width, label="Fake", color="#E69F00", edgecolor="white")
    ax.bar(x + width/2,
           [closure_data[l]["real_rate"] for l in lbls],
           width, label="Real", color="#0072B2", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(lbls)
    ax.set_ylabel("Lip closure rate (MAR < θ)")
    ax.set_title("Lip closure rate during bilabials")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "rh2_2_closure_rate.png")

    # ── Plot 2: KDE overlay (IQR-clipped) ────────────────────────────────────
    def iqr_upper(arr):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        return q3 + 1.5 * (q3 - q1)

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap_fake = ["#E69F00", "#D55E00", "#CC79A7"]
    cmap_real = ["#0072B2", "#009E73", "#56B4E9"]
    x_max = 0.0
    for i, (lbl, res) in enumerate(all_results.items()):
        for mar_arr in [res["fake_mar"], res["real_mar"]]:
            x_max = max(x_max, iqr_upper(mar_arr))
    x_max = min(x_max * 1.05, 1.0)

    for i, (lbl, res) in enumerate(all_results.items()):
        for mar_arr, color, ls, grp in [
            (res["fake_mar"], cmap_fake[i], "-",  f"Fake {lbl}"),
            (res["real_mar"], cmap_real[i], "--", f"Real {lbl}"),
        ]:
            clipped = mar_arr[mar_arr <= x_max]
            xs  = np.linspace(0, x_max, 300)
            kde = gaussian_kde(clipped, bw_method=0.15)
            ax.plot(xs, kde(xs), color=color, ls=ls, lw=1.8, label=grp)
        ax.axvline(res["theta"], color=cmap_fake[i], lw=0.8, ls=":", alpha=0.6)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("MAR value")
    ax.set_ylabel("Density")
    ax.set_title("MAR distribution during bilabials (IQR-clipped)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, "rh2_2_mar_kde.png")


if __name__ == "__main__":
    run()
