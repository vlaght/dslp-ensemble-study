"""
RH2.3: There is a statistically significant difference in MAR distribution
for phonemes /r/, /l/, /w/, and /j/ between authentic videos and deepfakes.

Method:
  Load all individual MAR observations for each target phoneme.
  Apply Mann–Whitney U test (two-sided) comparing Fake vs Real distributions.
  Report rank-biserial r as effect size.
  Apply Bonferroni correction for 4 simultaneous tests.
  Visualise with violin plots.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import sqlite3
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq_utils import DB_PATH, apply_style, save_plot

apply_style()

# /r/ → 'ɹ' in American English espeak
TARGETS = {"ɹ": "/r/", "l": "/l/", "w": "/w/", "j": "/j/"}
ALPHA   = 0.05
N_TESTS = len(TARGETS)


def rank_biserial(u, n1, n2):
    return 1 - (2 * u) / (n1 * n2)


def run():
    print("=" * 65)
    print("RH2.3  MAR distribution shift for /r/, /l/, /w/, /j/")
    print("=" * 65)

    results    = {}
    notes_rows = ["# RH2.3 — Results\n\n"
                  "Bonferroni-corrected α = "
                  f"{ALPHA/N_TESTS:.4f} ({N_TESTS} tests)\n\n"
                  "| Phoneme | n Fake | n Real | Fake median | Real median | "
                  "U | p (raw) | p (Bonf.) | r | Significant? |\n"
                  "|---|---|---|---|---|---|---|---|---|---|\n"]

    fake_mars = {}
    real_mars = {}

    for ph, lbl in TARGETS.items():
        print(f"\n{lbl} ('{ph}') …", flush=True)
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(
            "SELECT video_type, mar FROM lags WHERE phoneme=?",
            conn, params=(ph,))
        conn.close()

        fake_m = df[df["video_type"] == "Fake"]["mar"].values
        real_m = df[df["video_type"] == "Real"]["mar"].values
        fake_mars[lbl] = fake_m
        real_mars[lbl] = real_m

        stat, p_raw = mannwhitneyu(fake_m, real_m, alternative="two-sided")
        p_bonf      = min(p_raw * N_TESTS, 1.0)
        r_eff       = rank_biserial(stat, len(fake_m), len(real_m))
        sig         = p_bonf < ALPHA

        print(f"  n=Fake:{len(fake_m):,}  Real:{len(real_m):,}")
        print(f"  Fake median={np.median(fake_m):.5f}  Real median={np.median(real_m):.5f}")
        print(f"  U={stat:.1f}  p_raw={p_raw:.4e}  p_Bonf={p_bonf:.4e}  r={r_eff:.4f}")
        print(f"  → {'Significant' if sig else 'Not significant'} after Bonferroni")

        results[lbl] = dict(
            ph=ph, n_fake=len(fake_m), n_real=len(real_m),
            fake_med=np.median(fake_m), real_med=np.median(real_m),
            U=stat, p_raw=p_raw, p_bonf=p_bonf, r=r_eff, sig=sig)
        notes_rows.append(
            f"| {lbl} | {len(fake_m):,} | {len(real_m):,} | "
            f"{np.median(fake_m):.5f} | {np.median(real_m):.5f} | "
            f"{stat:.0f} | {p_raw:.2e} | {p_bonf:.2e} | "
            f"{r_eff:.4f} | {'✓' if sig else '✗'} |\n")

    n_sig = sum(r["sig"] for r in results.values())
    print(f"\n→ {n_sig}/{N_TESTS} phonemes significant after Bonferroni correction")
    print(f"  RH2.3 {'SUPPORTED' if n_sig == N_TESTS else 'PARTIALLY supported' if n_sig > 0 else 'NOT supported'}")

    notes_rows += [
        "\n## Statistical tests\n",
        "**Mann–Whitney U (two-sided):** Tests whether the MAR distribution "
        "for a phoneme differs between Fake and Real videos without assuming "
        "a specific direction. Two-sided is appropriate here since the "
        "direction of the effect (Fake higher or lower) is not specified.\n\n",
        "**Bonferroni correction:** With 4 simultaneous tests, the per-test "
        f"significance threshold is adjusted to α' = {ALPHA/N_TESTS:.4f} to "
        "control the family-wise error rate.\n\n",
        "**Rank-biserial r:** Effect size for Mann–Whitney U. |r|=0.1/0.3/0.5 "
        "conventionally small/medium/large. Positive r means Fake MAR "
        "tends to exceed Real MAR.\n",
    ]

    def iqr_clip(arr):
        q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr    = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return arr[(arr >= lo) & (arr <= hi)], hi

    # ── Plot: all phonemes in one multi-panel figure (2×2) ────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    axes = axes.flatten()

    total_out = 0
    for ax, (lbl, res) in zip(axes, results.items()):
        fm = fake_mars[lbl]; rm = real_mars[lbl]
        fm_c, hi_f = iqr_clip(fm); rm_c, hi_r = iqr_clip(rm)
        total_out += (len(fm) - len(fm_c)) + (len(rm) - len(rm_c))
        y_max = min(max(hi_f, hi_r) * 1.05, 1.0)

        vp = ax.violinplot([fm_c, rm_c], positions=[0, 1],
                           showmedians=True, showextrema=False)
        for body, color in zip(vp["bodies"], ["#E69F00", "#0072B2"]):
            body.set_facecolor(color); body.set_alpha(0.7)
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(2)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Fake", "Real"])
        ax.set_ylim(0, y_max)
        sig_sym = "✓" if res["sig"] else "✗"
        ax.set_title(f"/{lbl}/")
        if ax == axes[0]:
            ax.set_ylabel("MAR")

    fig.suptitle("MAR distribution by approximant (IQR-clipped)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save_plot(fig, "rh2_3_mar_all.png")


if __name__ == "__main__":
    run()
