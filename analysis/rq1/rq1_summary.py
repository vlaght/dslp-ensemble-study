"""
RQ1 Summary: To what extent do specific phonemes vary in their capacity to
provide discriminative forensic features for deepfake detection?

Aggregates results from RH1.1, RH1.2, RH1.3 and produces a full-phoneme
ranking plot using DSMILP learned attention priors.

Run after all three RH scripts and train_save_priors.py have completed.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq_utils import apply_style, save_plot

apply_style()

BILABIALS  = {"m", "p", "b"}
FRICATIVES = {"s", "ʃ", "f"}
HYPOTHESIS_VOWELS = {"w", "uː", "ɪ", "ɛ", "æ", "ʌ", "ʊ", "ɑː", "ə"}

GROUP_COLORS = {
    "bilabial":  "#CC79A7",
    "fricative": "#009E73",
    "h-vowel":   "#E69F00",   # hypothesis vowels
    "other":     "#AAAAAA",
}


def run():
    print("=" * 65)
    print("RQ1 Summary — Phoneme informativeness (full ranking)")
    print("=" * 65)

    with open(".tmp/dsmilp_phoneme_priors.json", encoding="utf-8") as f:
        priors = json.load(f)

    names    = priors["phoneme_names"]
    vis_p    = np.array(priors["vis_prior"])
    aud_p    = np.array(priors["aud_prior"])
    combined = (vis_p + aud_p) / 2   # simple average of both streams

    # Sort all phonemes by combined prior (descending)
    order = np.argsort(combined)[::-1]

    print(f"\nModel AUC at training: {priors['best_auc']:.4f}")
    print(f"\nTop 20 phonemes by combined (vis+aud)/2 prior:")
    print(f"  {'Phoneme':12s}  {'Combined':>9s}  {'Visual':>9s}  {'Audio':>9s}")
    for i in order[:20]:
        ph = names[i]
        print(f"  {ph:12s}  {combined[i]:+9.4f}  {vis_p[i]:+9.4f}  {aud_p[i]:+9.4f}")

    print(f"\nBottom 10 phonemes by combined prior:")
    for i in order[-10:]:
        ph = names[i]
        print(f"  {ph:12s}  {combined[i]:+9.4f}  {vis_p[i]:+9.4f}  {aud_p[i]:+9.4f}")

    # ── Group summary for hypothesis phonemes ─────────────────────────────────
    print("\n── Hypothesis phoneme group summary ────────────────────────────")
    for group_name, members in [
        ("Bilabials (/m/ /p/ /b/)",     BILABIALS),
        ("Fricatives (/s/ /sh/ /f/)",   FRICATIVES),
        ("Hyp. vowels (/w/ /u/ /ɪ/ …)", HYPOTHESIS_VOWELS),
    ]:
        idxs = [i for i, ph in enumerate(names) if ph in members]
        if not idxs:
            continue
        cv = combined[idxs]; vv = vis_p[idxs]; av = aud_p[idxs]
        print(f"\n  {group_name}")
        print(f"    Combined prior:  mean={cv.mean():+.4f}  range=[{cv.min():+.4f}, {cv.max():+.4f}]")
        print(f"    Visual  prior:   mean={vv.mean():+.4f}")
        print(f"    Audio   prior:   mean={av.mean():+.4f}")



    # ── Full ranking plot (top 40 by |combined prior|) ────────────────────────
    top_n    = min(40, len(names))
    top_idxs = order[:top_n]
    ph_names = [names[i] for i in top_idxs]
    cv_vals  = combined[top_idxs]
    vv_vals  = vis_p[top_idxs]
    av_vals  = aud_p[top_idxs]

    def group_color(ph):
        if ph in BILABIALS:   return GROUP_COLORS["bilabial"]
        if ph in FRICATIVES:  return GROUP_COLORS["fricative"]
        if ph in HYPOTHESIS_VOWELS: return GROUP_COLORS["h-vowel"]
        return GROUP_COLORS["other"]

    colors  = [group_color(ph) for ph in ph_names]
    y_pos   = np.arange(top_n)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=GROUP_COLORS["bilabial"],  label="Bilabials /m/ /p/ /b/"),
        Patch(facecolor=GROUP_COLORS["fricative"], label="Fricatives /s/ /sh/ /f/"),
        Patch(facecolor=GROUP_COLORS["h-vowel"],   label="Hypothesis vowels"),
        Patch(facecolor=GROUP_COLORS["other"],     label="Other phonemes"),
    ]
    for vals, stream, fname in [
        (cv_vals, "Combined (vis+aud)/2 prior", "rq1_ranking_combined.png"),
        (vv_vals, "Visual stream prior",        "rq1_ranking_visual.png"),
        (av_vals, "Audio stream prior",         "rq1_ranking_audio.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 12))
        ax.barh(y_pos, vals, color=colors, edgecolor="white", height=0.7)
        ax.axvline(0, color="grey", lw=1, ls="--")
        ax.set_xlabel("Learned attention prior")
        ax.set_yticks(y_pos); ax.set_yticklabels(ph_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"RQ1 — Top {top_n} phonemes: {stream}\n"
                     f"(DSMILP AUC = {priors['best_auc']:.4f})",
                     fontsize=12, fontweight="bold")
        fig.legend(handles=legend_handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.04))
        fig.tight_layout()
        save_plot(fig, fname)


if __name__ == "__main__":
    run()
