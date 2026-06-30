"""
RQ3 Summary: Which sequence-modelling architecture best captures the temporal
             dynamics of phoneme articulation for deepfake detection?

Aggregates RH3.1 (sequence vs ML), RH3.2 (BiLSTM vs UniLSTM),
RH3.3 (multimodal vs unimodal) and produces an architecture comparison plot.

Run after rh3_1.py, rh3_2.py, rh3_3.py have completed.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from analysis.rq_utils import apply_style, save_plot

apply_style()

# Single-model architectures to include in the ranking
SEQ_SINGLE = {"bilstm", "attention", "dsmil", "dsmilp", "bigru",
              "cnn", "tcn", "transformer", "mscnn", "crossmodal",
              "phoneme_gated"}

ARCH_LABELS = {
    "bilstm":       "BiLSTM",
    "attention":    "Temporal-Attention BiLSTM",
    "bigru":        "BiGRU",
    "cnn":          "Conv1D",
    "tcn":          "TCN",
    "transformer":  "Transformer",
    "dsmil":        "DualStreamMIL",
    "dsmilp":       "DualStreamMIL+Phoneme",
    "mscnn":        "MultiScale-CNN",
    "crossmodal":   "CrossModal LSTM",
    "phoneme_gated": "PhonemeGated LSTM",
}

# Colorblind-safe palette (Okabe-Ito)
ARCH_COLORS = {
    "bilstm":       "#0072B2",
    "attention":    "#009E73",
    "bigru":        "#56B4E9",
    "cnn":          "#E69F00",
    "tcn":          "#D55E00",
    "transformer":  "#CC79A7",
    "dsmil":        "#F0E442",
    "dsmilp":       "#000000",
    "mscnn":        "#999999",
    "crossmodal":   "#E0C080",
    "phoneme_gated": "#8080FF",
}


def run():
    print("=" * 65)
    print("RQ3 Summary — Architecture comparison")
    print("=" * 65)

    with open(".tmp/all_results.json") as f:
        all_res = json.load(f)

    # Per-architecture AUC distributions (feature_set="all", single models only)
    arch_aucs = {}
    for r in all_res:
        model = r.get("model", "")
        fset  = r.get("feature_set", "")
        auc   = r.get("auc", 0)
        if model in SEQ_SINGLE and fset == "all" and auc > 0:
            arch_aucs.setdefault(model, []).append(auc)

    print("\nArchitecture AUC summary (feature_set='all', single models):")
    print(f"  {'Architecture':30s}  {'n':>3s}  {'mean AUC':>9s}  {'max AUC':>8s}  {'std':>6s}")
    summary_rows = []
    for arch, aucs in sorted(arch_aucs.items(), key=lambda x: -np.mean(x[1])):
        m = np.mean(aucs); mx = max(aucs); s = np.std(aucs)
        print(f"  {ARCH_LABELS.get(arch, arch):30s}  {len(aucs):3d}  {m:9.4f}  {mx:8.4f}  {s:6.4f}")
        summary_rows.append((arch, aucs, m, mx, s))

    # RQ3 answer text
    best_arch = summary_rows[0][0] if summary_rows else "dsmilp"
    best_mean = summary_rows[0][2] if summary_rows else 0.0


    # ── Architecture comparison box-plot ──────────────────────────────────────
    if not summary_rows:
        print("  No data — skipping plot"); return

    # Sort by mean AUC descending
    sorted_rows = sorted(summary_rows, key=lambda x: x[2])  # ascending for horizontal
    archs   = [r[0] for r in sorted_rows]
    aucs_l  = [r[1] for r in sorted_rows]
    cols    = [ARCH_COLORS.get(a, "#AAAAAA") for a in archs]
    ylabels = [ARCH_LABELS.get(a, a) for a in archs]

    fig, ax = plt.subplots(figsize=(10, 6))
    rng = np.random.RandomState(42)
    for i, (aucs, col) in enumerate(zip(aucs_l, cols)):
        jit = rng.uniform(-0.15, 0.15, len(aucs))
        ax.scatter(aucs, np.full(len(aucs), i) + jit,
                   color=col, alpha=0.5, s=25, zorder=3)
    ax.boxplot(aucs_l, vert=False, positions=range(len(archs)),
               widths=0.4, patch_artist=True,
               boxprops=dict(alpha=0.3),
               medianprops=dict(color="black", lw=2))
    for patch, col in zip(ax.patches, cols):
        patch.set_facecolor(col)

    ax.set_yticks(range(len(archs)))
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("AUC")
    ax.set_title("RQ3 — Architecture comparison (feature_set='all', single models)\n"
                 "Higher = better deepfake detection", fontsize=12)
    ax.axvline(0.9, color="grey", lw=1, ls="--", alpha=0.5)
    fig.tight_layout()
    save_plot(fig, "rq3_architecture_comparison.png")


if __name__ == "__main__":
    run()
