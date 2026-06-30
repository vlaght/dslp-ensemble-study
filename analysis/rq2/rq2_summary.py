"""
RQ2 Summary: Do specific phoneme classes exhibit measurably different articulatory
             quality between genuine and deepfake speech?

Aggregates results from RH2.1, RH2.2, RH2.3 and produces a composite figure
summarising the three lines of articulatory evidence.

Run after rh2_1.py, rh2_2.py, rh2_3.py have completed.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu

from analysis.rq_utils import DB_PATH, apply_style, save_plot

apply_style()

# Phoneme groups analysed in RH2.1–2.3
PLOSIVES     = {"p", "b", "m"}
APPROXIMANTS = {"ɹ", "l", "w", "j"}


def load_mar(phonemes: set, limit_per_type: int = 200_000) -> pd.DataFrame:
    """Return a sample of MAR values per phoneme and video_type."""
    ph_list = ", ".join(f"'{p}'" for p in sorted(phonemes))
    query = f"""
        SELECT phoneme, video_type, mar
        FROM lags
        WHERE phoneme IN ({ph_list})
          AND mar IS NOT NULL
        LIMIT {limit_per_type * len(phonemes)}
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(query, conn)
    conn.close()
    df["label"] = (df["video_type"] == "Fake").astype(int)
    return df


def run():
    print("=" * 65)
    print("RQ2 Summary — Articulatory quality differences")
    print("=" * 65)

    # ── Load a sample for visualisation ──────────────────────────────────────
    print("\nLoading MAR values for key phonemes …", flush=True)
    df = load_mar(PLOSIVES | APPROXIMANTS)
    print(f"  Loaded {len(df):,} rows")

    # ── Per-phoneme group summary table ───────────────────────────────────────
    print("\n── Per-phoneme MAR summary ──────────────────────────────────────")
    print(f"  {'Phoneme':8s}  {'Group':12s}  {'Fake med':>8s}  {'Real med':>8s}  {'Δ':>8s}")

    summary_rows = []
    for ph in sorted(PLOSIVES | APPROXIMANTS):
        sub = df[df["phoneme"] == ph]
        if sub.empty:
            continue
        fake_med = sub[sub["label"]==1]["mar"].median()
        real_med = sub[sub["label"]==0]["mar"].median()
        group = "Plosive" if ph in PLOSIVES else "Approximant"
        print(f"  {ph:8s}  {group:12s}  {fake_med:8.5f}  {real_med:8.5f}  {fake_med-real_med:+8.5f}")
        summary_rows.append((ph, group, fake_med, real_med))

    # ── RQ2 answer ────────────────────────────────────────────────────────────

    def iqr_clip(arr):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return arr[(arr >= lo) & (arr <= hi)], hi

    # ── All phonemes combined in one figure ───────────────────────────────────
    order = sorted(PLOSIVES) + sorted(APPROXIMANTS)
    order = [p for p in order if p in df["phoneme"].unique()]
    vtype_colors = {"Fake": "#E69F00", "Real": "#0072B2"}
    n_ph = len(order)
    fig, axes = plt.subplots(1, n_ph, figsize=(3.5 * n_ph, 5), sharey=False)
    if n_ph == 1:
        axes = [axes]

    total_out = 0
    for ax, ph in zip(axes, order):
        sub   = df[df["phoneme"] == ph]
        group = "Plosive" if ph in PLOSIVES else "Approx."
        fv = sub[sub["video_type"]=="Fake"]["mar"].dropna().values
        rv = sub[sub["video_type"]=="Real"]["mar"].dropna().values
        fv_c, hi_f = iqr_clip(fv); rv_c, hi_r = iqr_clip(rv)
        total_out += (len(fv) - len(fv_c)) + (len(rv) - len(rv_c))
        y_max = min(max(hi_f, hi_r) * 1.05, 1.0)

        parts = ax.violinplot([fv_c, rv_c], positions=[0, 1],
                              showmedians=True, widths=0.7)
        for body, col in zip(parts["bodies"],
                             [vtype_colors["Fake"], vtype_colors["Real"]]):
            body.set_facecolor(col); body.set_alpha(0.6)
        parts["cmedians"].set_color("black"); parts["cmedians"].set_lw(2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["F", "R"])
        ax.set_ylim(0, y_max)
        ax.set_title(f"/{ph}/", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("MAR (mouth aspect ratio)")

    fig.suptitle("MAR by phoneme class (IQR-clipped)",
                 fontsize=12, fontweight="bold")
    fig.legend(handles=[
        Patch(facecolor=vtype_colors["Fake"], alpha=0.7, label="Fake"),
        Patch(facecolor=vtype_colors["Real"], alpha=0.7, label="Real"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save_plot(fig, "rq2_mar_phonemes_summary.png")


if __name__ == "__main__":
    run()
