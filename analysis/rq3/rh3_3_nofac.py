"""
RH3.3 re-run excluding FakeAVCeleb_v1.2 videos.

FakeAVCeleb contains face-swap videos with authentic audio, which creates
a conflicting training signal for the audio stream (real audio, Fake label).
This re-run tests whether the confound was masking a genuine multimodal benefit.

Identical architecture and CV protocol to rh3_3.py — only the video set differs.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import math, sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, cohen_kappa_score
from scipy.stats import mannwhitneyu, kruskal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.rq_utils import (apply_style, save_plot,
                                DB_PATH, FEATURE_COLS, VISUAL_COLS, AUDIO_COLS)

apply_style()

N_FOLDS  = 10
N_SEEDS  = 1
SEED     = 42
EPOCHS   = 30
BATCH    = 64
LR_RATE  = 1e-3
HIDDEN   = 64
N_LAYERS = 2

FEATURE_SETS = {
    "multimodal":  FEATURE_COLS,
    "visual-only": VISUAL_COLS,
    "audio-only":  AUDIO_COLS,
}

PHON_EMBED_DIM = 8


# ── Model ─────────────────────────────────────────────────────────────────────

class PlainLSTM(nn.Module):
    def __init__(self, n_features, num_phonemes=0, hidden=HIDDEN, n_layers=N_LAYERS, dropout=0.3):
        super().__init__()
        self.phon_embed = (nn.Embedding(num_phonemes, PHON_EMBED_DIM)
                           if num_phonemes > 0 else None)
        lstm_in = n_features + (PHON_EMBED_DIM if num_phonemes > 0 else 0)
        self.lstm = nn.LSTM(lstm_in, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths, phon_ids=None):
        if self.phon_embed is not None and phon_ids is not None:
            x = torch.cat([x, self.phon_embed(phon_ids)], dim=-1)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1]).squeeze(1)


# ── Dataset / collate ─────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, seqs, phon_ids, labels):
        self.seqs, self.phon_ids, self.labels = seqs, phon_ids, labels
    def __len__(self): return len(self.seqs)
    def __getitem__(self, i):
        return self.seqs[i], self.phon_ids[i], float(self.labels[i])


def collate_fn(batch):
    seqs, phon_ids, labels = zip(*batch)
    lengths  = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded   = pad_sequence(seqs, batch_first=True)
    padded_p = pad_sequence(phon_ids, batch_first=True)
    return padded, padded_p, lengths, torch.tensor(labels, dtype=torch.float32)


# ── Training / eval ───────────────────────────────────────────────────────────

def train_eval(model, train_idx, val_idx, sequences, phon_seqs, labels, pos_weight, device):
    train_ds = SeqDataset([sequences[i] for i in train_idx],
                          [phon_seqs[i]  for i in train_idx], labels[train_idx])
    val_ds   = SeqDataset([sequences[i] for i in val_idx],
                          [phon_seqs[i]  for i in val_idx],   labels[val_idx])
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, collate_fn=collate_fn)

    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt  = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    model.train()
    for _ in range(EPOCHS):
        for feats, pids, lengths, labs in train_ld:
            feats, pids, lengths, labs = (feats.to(device), pids.to(device),
                                          lengths.to(device), labs.to(device))
            opt.zero_grad()
            crit(model(feats, lengths, pids), labs).backward()
            opt.step()

    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for feats, pids, lengths, labs in val_ld:
            feats, pids, lengths = feats.to(device), pids.to(device), lengths.to(device)
            logits_all.append(model(feats, lengths, pids).cpu().numpy())
            labels_all.append(labs.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(labels_all).astype(int)
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    return {
        "auc":   float(roc_auc_score(y_true, probs)),
        "f1":    float(f1_score(y_true, preds, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, preds)),
    }


def rank_biserial_r(a, b):
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 1 - (2 * u) / (len(a) * len(b))


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("RH3.3  Multimodal vs Visual-only vs Audio-only")
    print("       (excluding FakeAVCeleb_v1.2)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    import pickle
    from pathlib import Path
    _cache = Path(".tmp/rh3_3_seqcache_nofac.pkl")

    if _cache.exists():
        print(f"\nLoading sequences from cache {_cache} …", flush=True)
        with open(_cache, "rb") as f:
            _c = pickle.load(f)
        video_names   = _c["video_names"]
        video_labels  = _c["video_labels"]
        raw_seqs      = _c["raw_seqs"]
        phon_idx_seqs = _c["phon_idx_seqs"]
        phon_to_idx   = _c["phon_to_idx"]
        num_phonemes  = _c["num_phonemes"]
        print(f"  {num_phonemes - 1} unique phoneme types", flush=True)
        print(f"  {len(video_labels):,} videos loaded from cache", flush=True)
    else:
        print(f"\nLoading sequences from {DB_PATH} (no FakeAVCeleb) …", flush=True)
        feat_sql = ", ".join(FEATURE_COLS)
        query = f"""
            SELECT filename, video_type, phoneme, {feat_sql}
            FROM lags
            WHERE dataset != 'FakeAVCeleb_v1.2'
            ORDER BY filename, timestamp
        """
        conn = sqlite3.connect(DB_PATH)
        df   = pd.read_sql_query(query, conn)
        conn.close()
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

        all_phonemes  = sorted(df["phoneme"].dropna().unique())
        phon_to_idx   = {p: i + 1 for i, p in enumerate(all_phonemes)}
        num_phonemes  = len(phon_to_idx) + 1
        print(f"  {num_phonemes - 1} unique phoneme types")

        video_names, video_labels, raw_seqs, phon_idx_seqs = [], [], [], []
        for fname, grp in df.groupby("filename", sort=False):
            video_names.append(fname)
            video_labels.append(1 if grp["video_type"].iloc[0] == "Fake" else 0)
            raw_seqs.append(grp[FEATURE_COLS].values.astype(np.float32))
            phon_idx_seqs.append(np.array(
                [phon_to_idx.get(p, 0) for p in grp["phoneme"]], dtype=np.int32
            ))
        video_labels = np.array(video_labels)

        Path(".tmp").mkdir(exist_ok=True)
        print(f"  Saving cache to {_cache} …", flush=True)
        with open(_cache, "wb") as f:
            pickle.dump({"video_names": video_names, "video_labels": video_labels,
                         "raw_seqs": raw_seqs, "phon_idx_seqs": phon_idx_seqs,
                         "phon_to_idx": phon_to_idx, "num_phonemes": num_phonemes}, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  {num_phonemes - 1} unique phoneme types  ({_cache.stat().st_size/1e6:.0f} MB cached)")

    video_labels = np.array(video_labels)
    n_videos     = len(video_labels)
    n_fake       = video_labels.sum()
    n_real       = n_videos - n_fake
    pos_weight   = n_real / n_fake
    print(f"  {n_videos:,} videos  Fake={n_fake:,}  Real={n_real:,}")

    phon_tensors = [torch.tensor(p.astype(np.int64), dtype=torch.long) for p in phon_idx_seqs]

    all_cols = FEATURE_COLS
    vis_idx  = [all_cols.index(c) for c in VISUAL_COLS]
    aud_idx  = [all_cols.index(c) for c in AUDIO_COLS]
    mod_idx  = {"multimodal":  list(range(len(all_cols))),
                "visual-only": vis_idx,
                "audio-only":  aud_idx}

    USE_PHON_EMB = {"multimodal": True, "visual-only": False, "audio-only": True}

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = list(cv.split(np.arange(n_videos), video_labels))

    METRICS = ["auc", "f1", "kappa"]
    results = {m: {k: [] for k in METRICS} for m in FEATURE_SETS}

    for modality, feat_cols_m in FEATURE_SETS.items():
        col_idx   = mod_idx[modality]
        use_phon  = USE_PHON_EMB[modality]
        n_phon    = num_phonemes if use_phon else 0
        n_feats   = len(feat_cols_m)
        n_feats_display = n_feats + (PHON_EMBED_DIM if use_phon else 0)
        print(f"\n{'─'*65}")
        print(f"  {modality.upper()}  ({n_feats_display} features  phoneme_emb={use_phon})", flush=True)

        for fold_i, (train_idx, val_idx) in enumerate(folds):
            scaler = StandardScaler()
            train_rows = np.concatenate([raw_seqs[i][:, col_idx] for i in train_idx])
            chunk = 50_000
            for s in range(0, len(train_rows), chunk):
                scaler.partial_fit(train_rows[s:s+chunk])
            del train_rows

            scaled_seqs = [
                torch.tensor(scaler.transform(raw_seqs[i][:, col_idx]), dtype=torch.float32)
                for i in range(n_videos)
            ]

            for seed_i in range(N_SEEDS):
                torch.manual_seed(SEED + fold_i * N_SEEDS + seed_i)
                model = PlainLSTM(n_feats, num_phonemes=n_phon).to(device)
                m = train_eval(model, train_idx, val_idx,
                               scaled_seqs, phon_tensors, video_labels, pos_weight, device)
                for k in METRICS:
                    results[modality][k].append(m[k])
                print(f"    fold {fold_i+1}/{N_FOLDS}  seed {seed_i+1}/{N_SEEDS}"
                      f"  AUC={m['auc']:.4f}  F1={m['f1']:.4f}  Kappa={m['kappa']:.4f}",
                      flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    for mod, met_dict in results.items():
        for k in METRICS:
            a = np.array(met_dict[k])
            print(f"  {mod:14s}  {k.upper():6s}  n={len(a)}  mean={a.mean():.4f}  "
                  f"std={a.std():.4f}  max={a.max():.4f}")

    def get(mod, metric):
        return np.array(results[mod][metric])

    print("\nKruskal-Wallis omnibus:")
    kw_results = {}
    for k in METRICS:
        H, p_kw = kruskal(get("multimodal", k), get("visual-only", k), get("audio-only", k))
        kw_results[k] = (H, p_kw)
        print(f"  {k.upper():6s}  H={H:.2f}  p={p_kw:.4e}")

    def mwu(a, b, label, k):
        stat, p = mannwhitneyu(a, b, alternative="greater")
        r = rank_biserial_r(a, b)
        sig = "✓" if p < 0.05 else "✗"
        print(f"  {k.upper():6s}  {label}: U={stat:.0f}  p={p:.4e}  r={r:+.4f}  {sig}")
        return stat, p, r

    mwu_results = {}
    for k in METRICS:
        print(f"\nMann-Whitney U (one-sided) — {k.upper()}:")
        u_mv, p_mv, r_mv = mwu(get("multimodal",k), get("visual-only",k), "Multimodal > Visual-only", k)
        u_ma, p_ma, r_ma = mwu(get("multimodal",k), get("audio-only",k),  "Multimodal > Audio-only ", k)
        u_va, p_va, r_va = mwu(get("visual-only",k),get("audio-only",k),  "Visual-only > Audio-only", k)
        mwu_results[k] = dict(u_mv=u_mv, p_mv=p_mv, r_mv=r_mv,
                               u_ma=u_ma, p_ma=p_ma, r_ma=r_ma,
                               u_va=u_va, p_va=p_va, r_va=r_va)

    multi = get("multimodal", "auc")
    vis   = get("visual-only", "auc")
    aud   = get("audio-only",  "auc")
    H_auc, p_kw_auc = kw_results["auc"]

    # ── Verdict ───────────────────────────────────────────────────────────────
    p_mv_auc = mwu_results["auc"]["p_mv"]
    p_ma_auc = mwu_results["auc"]["p_ma"]
    supported = p_mv_auc < 0.05 and p_ma_auc < 0.05
    verdict = "SUPPORTED" if supported else "NOT SUPPORTED"
    print(f"\n→ RH3.3 (no FakeAVCeleb) {verdict} at α=0.05")

    # ── Notes ─────────────────────────────────────────────────────────────────

    # ── Plots ─────────────────────────────────────────────────────────────────
    mod_labels = ["Multimodal\n(visual+audio+phon)", "Visual-only\n(58 feat)", "Audio-only\n(audio+phon)"]
    colors     = ["#0072B2", "#009E73", "#E69F00"]
    rng        = np.random.RandomState(SEED)

    metric_meta = {
        "auc":   ("AUC",         "rh3_3_nofac_auc.png"),
        "f1":    ("F1 score",    "rh3_3_nofac_f1.png"),
        "kappa": ("Cohen Kappa", "rh3_3_nofac_kappa.png"),
    }

    for k, (ylabel, fname) in metric_meta.items():
        groups = [get("multimodal", k), get("visual-only", k), get("audio-only", k)]
        r      = mwu_results[k]
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, (g, col) in enumerate(zip(groups, colors)):
            jit = rng.uniform(-0.12, 0.12, len(g))
            ax.scatter(np.full(len(g), i) + jit, g, color=col, alpha=0.5, s=22, zorder=3)
            ax.boxplot([g], positions=[i], widths=0.3, patch_artist=True,
                       boxprops=dict(facecolor=col, alpha=0.4),
                       medianprops=dict(color="black", lw=2),
                       flierprops=dict(marker=""))
        ax.set_xticks(range(3))
        ax.set_xticklabels(mod_labels)
        ax.set_ylabel(ylabel)
        all_vals = np.concatenate(groups)
        lo, hi = all_vals.min(), all_vals.max()
        pad = max((hi - lo) * 0.20, 0.005)
        ax.set_ylim(lo - pad, min(1.0, hi + pad))
        annot = (f"Multi > Visual: p={r['p_mv']:.2e}\n"
                 f"Multi > Audio:  p={r['p_ma']:.2e}\n"
                 f"Visual > Audio: p={r['p_va']:.2e}")
        ax.text(0.98, 0.04, annot, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        ax.set_title(f"RH3.3 (no FakeAVCeleb) — {ylabel} comparison")
        fig.tight_layout()
        save_plot(fig, fname)
        plt.close(fig)

    print(f"\nPlots saved to analysis/plots/")


if __name__ == "__main__":
    run()
