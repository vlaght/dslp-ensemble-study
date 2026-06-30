"""
RH1.1: Phonemes /w/ and /u/ will yield significantly higher detection accuracy
than /ɪ/, /e/, /æ/, /ʌ/, /ʊ/, /ɒ/, and /ə/.

Primary metric: balanced accuracy (bACC) from logistic regression on a
  class-balanced, size-controlled per-phoneme subsample (N_PER_CLASS per class).
  Phonemes with fewer than N_PER_CLASS real videos are excluded to ensure
  bACC differences reflect signal, not training-set size.
Statistical tests:
  1. Permutation test on group mean bACC (10k permutations, phoneme-level n).
  2. Mann-Whitney U on pooled fold-level bACC scores (10 vs 25 values).
     Fold scores within a phoneme are correlated; effective n is between
     n_phonemes and n_phonemes*n_folds. Reported as a robustness check.
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
from scipy.stats import mannwhitneyu

from analysis.rq_utils import (apply_style, save_plot,
                                feature_introspection, VISUAL_COLS,
                                phoneme_balanced_auc,
                                permutation_test, bootstrap_mean_ci)

apply_style()

GROUP_A = {"w": "/w/", "uː": "/u/"}
GROUP_B = {"ɪ": "/ɪ/", "ɛ": "/e/", "æ": "/æ/",
           "ʌ": "/ʌ/", "ʊ": "/ʊ/", "ɑː": "/ɒ/ (≈ɑː)", "ə": "/ə/"}
ALL_PH  = {**GROUP_A, **GROUP_B}

N_TOP       = 12
MIN_OCC     = 3
N_PERM      = 10_000
N_PER_CLASS = 500


def rank_biserial_r(U, n1, n2):
    return (2 * U) / (n1 * n2) - 1


def run():
    print("=" * 65)
    print("RH1.1  /w/, /u/ vs other vowels — balanced accuracy + permutation")
    print(f"       N_PER_CLASS={N_PER_CLASS} (excluded if n_real < {N_PER_CLASS})")
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
            print(f"  {lbl:18s}  n_real={m['n_real_orig']:5d}  EXCLUDED (n_real < {N_PER_CLASS})")
        else:
            print(f"  {lbl:18s}  n_real={m['n_real_orig']:5d}  N_bal={m['n']:5d}  "
                  f"bACC={m['bacc']:.4f} [{m['bacc_lo']:.4f}, {m['bacc_hi']:.4f}]  "
                  f"AUC={m['auc']:.4f}")

    a_scores = [records[p]["bacc"] for p in GROUP_A
                if p in records and not np.isnan(records[p]["bacc"])]
    b_scores = [records[p]["bacc"] for p in GROUP_B
                if p in records and not np.isnan(records[p]["bacc"])]

    # ── Test 1: Permutation test on group mean bACC ───────────────────────────
    print("\n" + "─" * 65)
    obs_diff, p_perm = permutation_test(a_scores, b_scores,
                                        n_perm=N_PERM, alternative="greater")
    a_mean, a_lo, a_hi = bootstrap_mean_ci(a_scores)
    b_mean, b_lo, b_hi = bootstrap_mean_ci(b_scores)
    n_A = len(a_scores)
    n_tot = n_A + len(b_scores)
    from math import comb
    min_p = 1.0 / comb(n_tot, n_A)
    print(f"\nTest 1 — Permutation test (Group A > Group B, {N_PERM:,} permutations):")
    print(f"  Observed diff={obs_diff:+.4f}  p={p_perm:.4f}")
    print(f"  Group A: n={len(a_scores)}  mean bACC={a_mean:.4f}  95% CI [{a_lo:.4f}, {a_hi:.4f}]")
    print(f"  Group B: n={len(b_scores)}  mean bACC={b_mean:.4f}  95% CI [{b_lo:.4f}, {b_hi:.4f}]")
    print(f"  Min achievable p = 1/C({n_tot},{n_A}) = {min_p:.4f}")
    print(f"  → {'SUPPORTED' if p_perm < 0.05 else 'NOT supported'} at α=0.05")

    # ── Test 2: MWU on pooled fold-level bACC scores ──────────────────────────
    a_folds = []
    for p in GROUP_A:
        if p in records and not np.isnan(records[p]["bacc"]):
            a_folds.extend(records[p]["fold_baccs"])
    b_folds = []
    for p in GROUP_B:
        if p in records and not np.isnan(records[p]["bacc"]):
            b_folds.extend(records[p]["fold_baccs"])

    U_stat, p_mwu = mannwhitneyu(a_folds, b_folds, alternative="greater")
    r_mwu = rank_biserial_r(U_stat, len(a_folds), len(b_folds))
    print(f"\nTest 2 — MWU on pooled fold-level bACC (one-sided, A > B):")
    print(f"  Group A: n={len(a_folds)} folds  mean={np.mean(a_folds):.4f}")
    print(f"  Group B: n={len(b_folds)} folds  mean={np.mean(b_folds):.4f}")
    print(f"  U={U_stat:.0f}  p={p_mwu:.4f}  r={r_mwu:+.4f}")
    print(f"  Effective n ∈ [{n_A}, {len(a_folds)}] (within-phoneme fold correlation)")
    print(f"  → {'SUPPORTED' if p_mwu < 0.05 else 'NOT supported'} at α=0.05")

    # ── Feature introspection ─────────────────────────────────────────────────
    print("\n── Feature introspection (frame-level, r_eff ranked) ────────────")
    intro_a = feature_introspection(list(GROUP_A.keys()), n_top=N_TOP)
    intro_b = feature_introspection(list(GROUP_B.keys()), n_top=N_TOP)
    for name, intro in [("Group A (/w/, /u/)", intro_a),
                         ("Group B (other vowels)", intro_b)]:
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
    colors  = ["#0072B2" if p in GROUP_A else "#E69F00" for p in ph_list]
    y_pos   = np.arange(len(ph_list))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(y_pos, baccs, xerr=[lows, highs], color=colors,
            edgecolor="white", height=0.6, capsize=4)
    ax.axvline(0.5, color="grey", lw=1, ls="--")
    ax.set_xlabel(f"Balanced accuracy  (N={N_PER_CLASS} per class)")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_title("RH1.1 — Per-phoneme balanced accuracy (LR baseline)")
    fig.legend(handles=[
        Patch(facecolor="#0072B2", label="Group A: /w/, /u/"),
        Patch(facecolor="#E69F00", label="Group B: other vowels"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    save_plot(fig, "rh1_1_bacc.png")

    for intro, group_name, fname in [
        (intro_a, "Group A: /w/, /u/",  "rh1_1_features_groupA.png"),
        (intro_b, "Group B: other vowels", "rh1_1_features_groupB.png"),
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
        ax.set_title(f"RH1.1 — Top discriminative features\n{group_name}")
        fig.legend(handles=[
            Patch(facecolor="#0072B2", label="Visual feature"),
            Patch(facecolor="#E69F00", label="Audio feature"),
        ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        save_plot(fig, fname)


if __name__ == "__main__":
    run()
