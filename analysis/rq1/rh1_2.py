"""
RH1.2: Phonemes /s/, /sh/, and /f/ will yield significantly higher detection
accuracy than all of the vowels tested in RH1.1.

Primary metric: accuracy from logistic regression on a class-balanced,
  size-controlled per-phoneme subsample (N_PER_CLASS per class).
  Phonemes with fewer than N_PER_CLASS real videos are excluded.
Statistical tests:
  1. Permutation test on group mean accuracy (10k permutations).
  2. Mann-Whitney U on pooled fold-level accuracy scores (robustness check).
Secondary: bootstrap 95% CI on group means + per-group top-feature introspection.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from math import comb
from scipy.stats import mannwhitneyu

from analysis.rq_utils import (apply_style, save_plot,
                                feature_introspection, VISUAL_COLS,
                                phoneme_balanced_auc,
                                permutation_test, bootstrap_mean_ci)

apply_style()

FRICATIVES = {"s": "/s/", "ʃ": "/sh/", "f": "/f/"}
VOWELS     = {"w": "/w/", "uː": "/u/", "ɪ": "/ɪ/", "ɛ": "/e/",
              "æ": "/æ/", "ʌ": "/ʌ/", "ʊ": "/ʊ/", "ɑː": "/ɒ/", "ə": "/ə/"}
ALL_PH     = {**FRICATIVES, **VOWELS}

N_TOP       = 12
MIN_OCC     = 3
N_PERM      = 10_000
N_PER_CLASS = 500


def rank_biserial_r(U, n1, n2):
    return (2 * U) / (n1 * n2) - 1


def run():
    print("=" * 65)
    print("RH1.2  Fricatives vs vowels — accuracy + permutation")
    print(f"       N_PER_CLASS={N_PER_CLASS}")
    print("=" * 65)

    records = {}
    for ph, lbl in ALL_PH.items():
        m = phoneme_balanced_auc(ph, min_occurrences=MIN_OCC,
                                 n_per_class=N_PER_CLASS)
        if m["n"] // 2 < N_PER_CLASS:
            m["bacc"] = np.nan
            m["auc"]  = np.nan
        records[ph] = {"label": lbl, **m}
        if np.isnan(m["bacc"]):
            print(f"  {lbl:12s}  n_real={m['n_real_orig']:5d}  EXCLUDED")
        else:
            print(f"  {lbl:12s}  n_real={m['n_real_orig']:5d}  N_bal={m['n']:5d}  "
                  f"Acc={m['bacc']:.4f} [{m['bacc_lo']:.4f}, {m['bacc_hi']:.4f}]  "
                  f"AUC={m['auc']:.4f}")

    f_scores = [records[p]["bacc"] for p in FRICATIVES
                if p in records and not np.isnan(records[p]["bacc"])]
    v_scores = [records[p]["bacc"] for p in VOWELS
                if p in records and not np.isnan(records[p]["bacc"])]

    # ── Test 1: Permutation test on group mean accuracy ───────────────────────
    print("\n" + "─" * 65)
    obs_diff, p_perm = permutation_test(f_scores, v_scores,
                                        n_perm=N_PERM, alternative="greater")
    f_mean, f_lo, f_hi = bootstrap_mean_ci(f_scores)
    v_mean, v_lo, v_hi = bootstrap_mean_ci(v_scores)
    n_f, n_v = len(f_scores), len(v_scores)
    min_p = 1.0 / comb(n_f + n_v, n_f)
    print(f"\nTest 1 — Permutation test (fricatives > vowels, {N_PERM:,} permutations):")
    print(f"  Observed diff={obs_diff:+.4f}  p={p_perm:.4f}")
    print(f"  Fricatives (n={n_f}): mean Acc={f_mean:.4f}  95% CI [{f_lo:.4f}, {f_hi:.4f}]")
    print(f"  Vowels     (n={n_v}): mean Acc={v_mean:.4f}  95% CI [{v_lo:.4f}, {v_hi:.4f}]")
    print(f"  Min achievable p = 1/C({n_f+n_v},{n_f}) = {min_p:.4f}")
    print(f"  → {'SUPPORTED' if p_perm < 0.05 else 'NOT supported'} at α=0.05")

    # ── Test 2: MWU on fold-level accuracy scores ─────────────────────────────
    f_folds = []
    for p in FRICATIVES:
        if p in records and not np.isnan(records[p]["bacc"]):
            f_folds.extend(records[p]["fold_baccs"])
    v_folds = []
    for p in VOWELS:
        if p in records and not np.isnan(records[p]["bacc"]):
            v_folds.extend(records[p]["fold_baccs"])

    U_stat, p_mwu = mannwhitneyu(f_folds, v_folds, alternative="greater")
    r_mwu = rank_biserial_r(U_stat, len(f_folds), len(v_folds))
    print(f"\nTest 2 — MWU on pooled fold-level accuracy (one-sided, fricatives > vowels):")
    print(f"  Fricatives: n={len(f_folds)} folds  mean={np.mean(f_folds):.4f}")
    print(f"  Vowels:     n={len(v_folds)} folds  mean={np.mean(v_folds):.4f}")
    print(f"  U={U_stat:.0f}  p={p_mwu:.4f}  r={r_mwu:+.4f}")
    print(f"  Effective n ∈ [{n_f}, {len(f_folds)}] (within-phoneme fold correlation)")
    print(f"  → {'SUPPORTED' if p_mwu < 0.05 else 'NOT supported'} at α=0.05")

    # ── Feature introspection ─────────────────────────────────────────────────
    print("\n── Feature introspection (frame-level, r_eff ranked) ────────────")
    intro_f = feature_introspection(list(FRICATIVES.keys()), n_top=N_TOP)
    intro_v = feature_introspection(list(VOWELS.keys()),     n_top=N_TOP)
    for name, intro in [("Fricatives", intro_f), ("Vowels", intro_v)]:
        print(f"\n{name}:")
        for _, row in intro.head(N_TOP).iterrows():
            print(f"  {row['feature']:35s}  r={row['r_eff']:+.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    ph_list = [p for p in ALL_PH if p in records
               and not np.isnan(records[p]["bacc"])]
    labels  = [records[p]["label"] for p in ph_list]
    baccs   = [records[p]["bacc"]  for p in ph_list]
    lows    = [records[p]["bacc"] - records[p]["bacc_lo"] for p in ph_list]
    highs   = [records[p]["bacc_hi"] - records[p]["bacc"] for p in ph_list]
    colors  = ["#009E73" if p in FRICATIVES else "#E69F00" for p in ph_list]
    y_pos   = np.arange(len(ph_list))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(y_pos, baccs, xerr=[lows, highs], color=colors,
            edgecolor="white", height=0.6, capsize=4)
    ax.axvline(0.5, color="grey", lw=1, ls="--")
    ax.set_xlabel(f"Accuracy  (N={N_PER_CLASS} per class)")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_title("RH1.2 — Per-phoneme accuracy (LR baseline)")
    fig.legend(handles=[
        Patch(facecolor="#009E73", label="Fricatives: /s/ /sh/ /f/"),
        Patch(facecolor="#E69F00", label="Vowels"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save_plot(fig, "rh1_2_bacc.png")

    for intro, group_name, fname in [
        (intro_f, "Fricatives: /s/ /sh/ /f/", "rh1_2_features_fricatives.png"),
        (intro_v, "Vowels",                    "rh1_2_features_vowels.png"),
    ]:
        top = intro.head(N_TOP)
        bar_colors = ["#0072B2" if f in VISUAL_COLS else "#E69F00"
                      for f in top["feature"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.arange(len(top))
        ax.barh(y, top["r_eff"].values, color=bar_colors,
                edgecolor="white", height=0.6)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(top["feature"].tolist(), fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Rank-biserial r  (> 0: higher in Fake)")
        ax.set_title(f"RH1.2 — Top discriminative features\n{group_name}")
        fig.legend(handles=[
            Patch(facecolor="#0072B2", label="Visual feature"),
            Patch(facecolor="#E69F00", label="Audio feature"),
        ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        save_plot(fig, fname)


if __name__ == "__main__":
    run()
