"""
RH2.1: Deepfake videos will exhibit a statistically significant decrease in
mouth aspect ratio variance compared to genuine videos.

Rationale: Neural lip-sync models optimise a perceptual loss averaged over
many training examples, producing mouth shapes that cluster around a mean
position rather than expressing the full articulatory range of a real speaker.
This "regression to the mean" should manifest as lower within-video MAR variance
in deepfake content.

Method:
  1. For every video in dataset.db compute the per-video standard deviation
     of MAR (across all phoneme onsets in that video).
  2. Compare the distribution of per-video MAR SD between Fake and Real groups
     using Levene's test (equality of spread, no normality assumption) and a
     one-sided Mann–Whitney U test (H1: Fake SD < Real SD).
  3. Report Cohen's d as an effect-size measure.
  4. Visualise with violin + box plots side-by-side.

References:
  Levene (1960)  Brown & Forsythe (1974)  Mann & Whitney (1947)
  Cohen (1988) — effect size conventions (small=0.2, medium=0.5, large=0.8)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import levene, mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq_utils import DB_PATH, apply_style, save_plot

apply_style()


def cohen_d(a, b):
    """Pooled-SD Cohen's d (a − b)."""
    n_a, n_b = len(a), len(b)
    pooled   = np.sqrt(((n_a-1)*a.std(ddof=1)**2 + (n_b-1)*b.std(ddof=1)**2)
                       / (n_a + n_b - 2))
    return (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan


def run():
    print("=" * 65)
    print("RH2.1  Deepfakes exhibit lower within-video MAR variance")
    print("=" * 65)

    # ── Query: per-video MAR mean and SD ─────────────────────────────────────
    print("\nQuerying per-video MAR statistics …", flush=True)
    query = """
        SELECT filename, video_type, dataset,
               AVG(mar)       AS mar_mean,
               STDEV(mar)     AS mar_sd,
               COUNT(*)       AS n_phonemes
        FROM lags
        GROUP BY filename, video_type, dataset
        HAVING COUNT(*) >= 5
    """
    # SQLite has no built-in STDEV; compute manually with two passes.
    # Use a Python-side aggregation instead.
    query2 = """
        SELECT filename, video_type, dataset, mar
        FROM lags
    """
    conn = sqlite3.connect(DB_PATH)
    # Efficient: pull only the mar column + identifiers
    df_raw = pd.read_sql_query(
        "SELECT filename, video_type, dataset, mar FROM lags", conn)
    conn.close()

    print(f"  Loaded {len(df_raw):,} rows, "
          f"{df_raw['filename'].nunique():,} videos")

    # Per-video MAR statistics
    grp = df_raw.groupby(["filename", "video_type", "dataset"])["mar"]
    per_video = grp.agg(
        mar_mean="mean",
        mar_sd="std",
        n_phonemes="count",
    ).reset_index()
    per_video = per_video[per_video["n_phonemes"] >= 5].copy()
    per_video.dropna(subset=["mar_sd"], inplace=True)

    fake = per_video[per_video["video_type"] == "Fake"]["mar_sd"].values
    real = per_video[per_video["video_type"] == "Real"]["mar_sd"].values

    print(f"\n  Fake videos: {len(fake):,}  |  Real videos: {len(real):,}")
    print(f"  Fake MAR SD — mean={fake.mean():.5f}  median={np.median(fake):.5f}  "
          f"std={fake.std():.5f}")
    print(f"  Real MAR SD — mean={real.mean():.5f}  median={np.median(real):.5f}  "
          f"std={real.std():.5f}")

    # ── Statistical tests ─────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    stat_lev, p_lev = levene(fake, real)
    stat_mw,  p_mw  = mannwhitneyu(fake, real, alternative="less")
    d = cohen_d(fake, real)

    print(f"\nLevene's test (equality of spread):")
    print(f"  W={stat_lev:.4f}  p={p_lev:.4e}")
    print(f"\nMann–Whitney U (Fake SD < Real SD, one-sided):")
    print(f"  U={stat_mw:.1f}  p={p_mw:.4e}")
    print(f"\nCohen's d (Fake − Real): {d:.4f}")
    mag = ("negligible" if abs(d) < 0.2 else
           "small"      if abs(d) < 0.5 else
           "medium"     if abs(d) < 0.8 else "large")
    print(f"  Effect size: {mag}")
    print(f"\n→ RH2.1 {'SUPPORTED' if p_mw < 0.05 else 'NOT supported'} "
          f"(Mann–Whitney, α=0.05)")


    def iqr_clip(arr):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return arr[(arr >= lo) & (arr <= hi)], hi

    fake_c, hi_f = iqr_clip(fake)
    real_c, hi_r = iqr_clip(real)
    y_max = min(max(hi_f, hi_r) * 1.05, 1.0)
    n_out = (len(fake) - len(fake_c)) + (len(real) - len(real_c))
    print(f"  IQR clipping: removed {n_out} outlier points (upper fence)")

    # ── Plot 1: Violin ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    vp = ax.violinplot([fake_c, real_c], positions=[0, 1], showmedians=True,
                       showextrema=False)
    for body, color in zip(vp["bodies"], ["#E69F00", "#0072B2"]):
        body.set_facecolor(color); body.set_alpha(0.7)
    vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Fake", "Real"])
    ax.set_ylabel("Per-video MAR standard deviation")
    ax.set_ylim(0, y_max)
    ax.set_title("MAR variability per video")
    fig.tight_layout()
    save_plot(fig, "rh2_1_mar_violin.png")

    # ── Plot 2: Density histogram ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    bins = np.linspace(0, y_max, 60)
    ax.hist(fake_c, bins=bins, density=True, alpha=0.55, color="#E69F00", label="Fake")
    ax.hist(real_c, bins=bins, density=True, alpha=0.55, color="#0072B2", label="Real")
    ax.axvline(np.median(fake_c), color="#E69F00", lw=2, ls="--")
    ax.axvline(np.median(real_c), color="#0072B2", lw=2, ls="--")
    ax.set_xlabel("Per-video MAR standard deviation")
    ax.set_ylabel("Density")
    ax.set_title("MAR variability per video — density")
    ax.legend()
    fig.tight_layout()
    save_plot(fig, "rh2_1_mar_histogram.png")


if __name__ == "__main__":
    run()
