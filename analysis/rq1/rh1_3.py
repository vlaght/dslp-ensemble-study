"""
RH1.3: Bilabial phonemes (/m/, /p/, /b/) will yield the highest overall
detection accuracy across all phoneme classes.

Statistical tests:
  1. Permutation 3-group variance of means (10k perm).
  2. Directed permutation: bilabials > (fricatives + vowels combined).
  3. MWU on pooled fold-level accuracy: bilabials vs all others (robustness check).
Secondary: bootstrap 95% CI on each group mean + per-group top-feature introspection.
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

BILABIALS  = {"m": "/m/", "p": "/p/", "b": "/b/"}
FRICATIVES = {"s": "/s/", "ʃ": "/sh/", "f": "/f/"}
VOWELS     = {"w": "/w/", "uː": "/u/", "ɪ": "/ɪ/", "ɛ": "/e/",
              "æ": "/æ/", "ʌ": "/ʌ/", "ʊ": "/ʊ/", "ɑː": "/ɒ/", "ə": "/ə/"}
ALL_PH     = {**BILABIALS, **FRICATIVES, **VOWELS}

N_TOP       = 12
MIN_OCC     = 3
N_PERM      = 10_000
N_PER_CLASS = 500


def permutation_test_3groups(scores_a, scores_b, scores_c,
                              n_perm: int = 10_000, seed: int = 42) -> tuple:
    a, b, c = (np.array(x, dtype=float) for x in (scores_a, scores_b, scores_c))
    observed = np.var([a.mean(), b.mean(), c.mean()])
    combined = np.concatenate([a, b, c])
    na, nb = len(a), len(b)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm = rng.permutation(combined)
        pa, pb, pc = perm[:na], perm[na:na+nb], perm[na+nb:]
        null[i] = np.var([pa.mean(), pb.mean(), pc.mean()])
    return float(observed), float((null >= observed).mean())


def rank_biserial_r(U, n1, n2):
    return (2 * U) / (n1 * n2) - 1


def run():
    print("=" * 65)
    print("RH1.3  Bilabials highest — accuracy + permutation")
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

    def valid_scores(group):
        return [records[p]["bacc"] for p in group
                if p in records and not np.isnan(records[p]["bacc"])]

    def valid_folds(group):
        folds = []
        for p in group:
            if p in records and not np.isnan(records[p]["bacc"]):
                folds.extend(records[p]["fold_baccs"])
        return folds

    bil_scores  = valid_scores(BILABIALS)
    fric_scores = valid_scores(FRICATIVES)
    vowl_scores = valid_scores(VOWELS)
    others      = fric_scores + vowl_scores

    n_bil = len(bil_scores)
    n_oth = len(others)
    min_p_dir = 1.0 / comb(n_bil + n_oth, n_bil)

    # ── Test 1: 3-group permutation ───────────────────────────────────────────
    print("\n" + "─" * 65)
    obs_3g, p_3g = permutation_test_3groups(bil_scores, fric_scores, vowl_scores,
                                             n_perm=N_PERM)
    bil_mean,  bil_lo,  bil_hi  = bootstrap_mean_ci(bil_scores)
    fric_mean, fric_lo, fric_hi = bootstrap_mean_ci(fric_scores)
    vowl_mean, vowl_lo, vowl_hi = bootstrap_mean_ci(vowl_scores)
    print(f"\nGroup means:")
    print(f"  Bilabials  (n={n_bil}): Acc={bil_mean:.4f}  [{bil_lo:.4f}, {bil_hi:.4f}]")
    print(f"  Fricatives (n={len(fric_scores)}): Acc={fric_mean:.4f}  [{fric_lo:.4f}, {fric_hi:.4f}]")
    print(f"  Vowels     (n={len(vowl_scores)}): Acc={vowl_mean:.4f}  [{vowl_lo:.4f}, {vowl_hi:.4f}]")
    print(f"\nTest 1 — Permutation 3-group (variance of group means, {N_PERM:,} perm):")
    print(f"  stat={obs_3g:.6f}  p={p_3g:.4f}")
    print(f"  → {'SIGNIFICANT' if p_3g < 0.05 else 'not significant'} at α=0.05")

    # ── Test 2: Directed permutation (bilabials > others) ────────────────────
    obs_dir, p_dir = permutation_test(bil_scores, others,
                                      n_perm=N_PERM, alternative="greater")
    print(f"\nTest 2 — Directed permutation (bilabials > all others, {N_PERM:,} perm):")
    print(f"  diff={obs_dir:+.4f}  p={p_dir:.4f}  min_p={min_p_dir:.4f}")
    print(f"  → {'SUPPORTED' if p_dir < 0.05 else 'NOT supported'} at α=0.05")

    # ── Test 3: MWU on fold-level accuracy ───────────────────────────────────
    bil_folds  = valid_folds(BILABIALS)
    oth_folds  = valid_folds(FRICATIVES) + valid_folds(VOWELS)
    U_stat, p_mwu = mannwhitneyu(bil_folds, oth_folds, alternative="greater")
    r_mwu = rank_biserial_r(U_stat, len(bil_folds), len(oth_folds))
    print(f"\nTest 3 — MWU on fold-level accuracy (bilabials > others, one-sided):")
    print(f"  Bilabials: n={len(bil_folds)} folds  mean={np.mean(bil_folds):.4f}")
    print(f"  Others:    n={len(oth_folds)} folds  mean={np.mean(oth_folds):.4f}")
    print(f"  U={U_stat:.0f}  p={p_mwu:.4f}  r={r_mwu:+.4f}")
    print(f"  Effective n ∈ [{n_bil}, {len(bil_folds)}] (within-phoneme fold correlation)")
    print(f"  → {'SUPPORTED' if p_mwu < 0.05 else 'NOT supported'} at α=0.05")

    # ── Feature introspection ─────────────────────────────────────────────────
    print("\n── Feature introspection (frame-level, r_eff ranked) ────────────")
    intro_bil  = feature_introspection(list(BILABIALS.keys()),  n_top=N_TOP)
    intro_fric = feature_introspection(list(FRICATIVES.keys()), n_top=N_TOP)
    intro_vowl = feature_introspection(list(VOWELS.keys()),     n_top=N_TOP)
    for name, intro in [("Bilabials", intro_bil),
                         ("Fricatives", intro_fric),
                         ("Vowels", intro_vowl)]:
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
    n_bil_  = len([p for p in ph_list if p in BILABIALS])
    n_fric_ = len([p for p in ph_list if p in FRICATIVES])
    colors  = (["#CC79A7"] * n_bil_ +
               ["#009E73"] * n_fric_ +
               ["#E69F00"] * (len(ph_list) - n_bil_ - n_fric_))
    y_pos   = np.arange(len(ph_list))

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.barh(y_pos, baccs, xerr=[lows, highs], color=colors,
            edgecolor="white", height=0.6, capsize=4)
    ax.axvline(0.5, color="grey", lw=1, ls="--")
    ax.set_xlabel(f"Accuracy  (N={N_PER_CLASS} per class)")
    ax.set_yticks(y_pos); ax.set_yticklabels(labels)
    ax.set_title("RH1.3 — Per-phoneme accuracy (LR baseline)")
    fig.legend(handles=[
        Patch(facecolor="#CC79A7", label="Bilabials: /m/ /p/ /b/"),
        Patch(facecolor="#009E73", label="Fricatives: /s/ /sh/ /f/"),
        Patch(facecolor="#E69F00", label="Vowels"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save_plot(fig, "rh1_3_bacc.png")

    for intro, group_name, fname in [
        (intro_bil,  "Bilabials: /m/ /p/ /b/",  "rh1_3_features_bilabials.png"),
        (intro_fric, "Fricatives: /s/ /sh/ /f/", "rh1_3_features_fricatives.png"),
        (intro_vowl, "Vowels",                    "rh1_3_features_vowels.png"),
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
        ax.set_title(f"RH1.3 — Top discriminative features\n{group_name}")
        fig.legend(handles=[
            Patch(facecolor="#0072B2", label="Visual feature"),
            Patch(facecolor="#E69F00", label="Audio feature"),
        ], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout()
        save_plot(fig, fname)


if __name__ == "__main__":
    run()
