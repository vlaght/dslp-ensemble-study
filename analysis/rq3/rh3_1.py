"""
RH3.1: Base sequence-modelling architectures (LSTM, RNN, Transformer) will
significantly outperform base ML classifiers (LR, RF, SVM).

Two experiments, both using 5-fold stratified CV (split by video):

  Exp 1 — Frame-level (same data for both groups):
    Seq models  : ordered per-video phoneme-frame sequences
    ML classifiers: individual phoneme frames as independent samples;
                    predictions averaged per video for evaluation.
    Phoneme type is included as a normalised label-encoded feature for all models.

  Exp 2 — Phoneme-averaged ML vs sequences:
    Seq models  : same ordered sequences as Exp 1
    ML classifiers: per-(video, phoneme_type) mean feature row;
                    predictions averaged per video for evaluation.

Both experiments use identical train/val video splits. No feature engineering
beyond a per-fold StandardScaler.

References:
  Hochreiter & Schmidhuber (1997) — LSTM
  Elman (1990) — Simple RNN
  Vaswani et al. (2017) — Transformer
  Cortes & Vapnik (1995) — SVM | Breiman (2001) — Random Forest
  Fisher (1925) — one-way ANOVA | Cohen (1988) — effect size d
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              cohen_kappa_score, precision_score, recall_score)
from scipy.stats import f_oneway
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCALER_CHUNK = 50_000   # rows per partial_fit chunk
PREDICT_CHUNK = 100_000 # rows per prediction chunk

from analysis.rq_utils import DB_PATH, apply_style, save_plot, FEATURE_COLS

apply_style()

N_FOLDS  = 5
SEED     = 42
EPOCHS   = 30
BATCH    = 64
LR_RATE  = 1e-3
HIDDEN   = 64
N_LAYERS = 2
D_MODEL  = 64
NHEAD    = 4
SHARED_METRICS = ["auc", "acc", "f1", "kappa"]


# ── Base sequence architectures (phoneme_id included as last feature col) ─────

class PlainLSTM(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(h[-1]).squeeze(1)


class PlainRNN(nn.Module):
    def __init__(self, n_features, hidden=HIDDEN, n_layers=N_LAYERS, dropout=0.3):
        super().__init__()
        self.rnn = nn.RNN(n_features, hidden, n_layers, batch_first=True,
                          dropout=dropout if n_layers > 1 else 0.0,
                          nonlinearity="tanh")
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, h = self.rnn(packed)
        return self.fc(h[-1]).squeeze(1)


class PlainTransformer(nn.Module):
    def __init__(self, n_features, d_model=D_MODEL, nhead=NHEAD,
                 n_layers=N_LAYERS, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        max_len = 1024
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        B, T, _ = x.shape
        x = self.proj(x) + self.pe[:T]
        mask = (torch.arange(T, device=x.device).unsqueeze(0)
                >= lengths.unsqueeze(1).to(x.device))
        out  = self.encoder(x, src_key_padding_mask=mask)
        valid = ~mask
        pooled = (out * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).float()
        return self.fc(pooled).squeeze(1)


# ── Dataset / collate ─────────────────────────────────────────────────────────

class VideoSeqDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs   = seqs
        self.labels = labels

    def __len__(self):  return len(self.seqs)

    def __getitem__(self, i):
        return self.seqs[i], float(self.labels[i])


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.float32)


# ── Training / eval helpers ───────────────────────────────────────────────────

def train_eval_seq(model, train_idx, val_idx, sequences, labels,
                   pos_weight, device):
    train_ds = VideoSeqDataset([sequences[i] for i in train_idx],
                               [labels[i]    for i in train_idx])
    val_ds   = VideoSeqDataset([sequences[i] for i in val_idx],
                               [labels[i]    for i in val_idx])
    train_ld = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          collate_fn=collate_fn)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                          collate_fn=collate_fn)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device))
    opt  = torch.optim.Adam(model.parameters(), lr=LR_RATE)

    model.train()
    for _ in range(EPOCHS):
        for feats, lengths, labs in train_ld:
            feats, lengths, labs = (feats.to(device), lengths.to(device),
                                    labs.to(device))
            opt.zero_grad()
            crit(model(feats, lengths), labs).backward()
            opt.step()

    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for feats, lengths, labs in val_ld:
            feats, lengths = feats.to(device), lengths.to(device)
            logits_all.append(model(feats, lengths).cpu().numpy())
            labels_all.append(labs.numpy())
    logits = np.concatenate(logits_all)
    y_true = np.concatenate(labels_all).astype(int)
    probs  = 1 / (1 + np.exp(-logits))
    preds  = (probs >= 0.5).astype(int)
    return dict(auc   = roc_auc_score(y_true, probs),
                acc   = accuracy_score(y_true, preds),
                f1    = f1_score(y_true, preds, zero_division=0),
                kappa = cohen_kappa_score(y_true, preds))


def fit_scaler_chunked(X: np.ndarray) -> StandardScaler:
    """Fit StandardScaler on X in chunks to avoid large float64 allocation."""
    scaler = StandardScaler()
    for start in range(0, len(X), SCALER_CHUNK):
        scaler.partial_fit(X[start:start + SCALER_CHUNK])
    return scaler


def transform_chunked(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Transform X in chunks and return float32 result."""
    out = np.empty_like(X, dtype=np.float32)
    for start in range(0, len(X), PREDICT_CHUNK):
        end = start + PREDICT_CHUNK
        out[start:end] = scaler.transform(X[start:end]).astype(np.float32)
    return out


def predict_proba_chunked(clf, X: np.ndarray) -> np.ndarray:
    """Predict probabilities in chunks to avoid large temporary allocations."""
    probs = np.empty(len(X), dtype=np.float32)
    for start in range(0, len(X), PREDICT_CHUNK):
        end = start + PREDICT_CHUNK
        probs[start:end] = clf.predict_proba(X[start:end])[:, 1]
    return probs


def eval_ml_video_level(clf, X_test_rows, y_test_video, video_idx_test):
    """
    Predict per row (in chunks), aggregate mean probability per video,
    then compute metrics.
    video_idx_test: array same length as X_test_rows, giving the video index.
    """
    probs_row = predict_proba_chunked(clf, X_test_rows)
    unique_vids = np.unique(video_idx_test)
    probs_vid, labels_vid = [], []
    for v in unique_vids:
        mask = video_idx_test == v
        probs_vid.append(probs_row[mask].mean())
        labels_vid.append(y_test_video[v])
    probs_vid  = np.array(probs_vid)
    labels_vid = np.array(labels_vid)
    preds_vid  = (probs_vid >= 0.5).astype(int)
    return dict(auc       = roc_auc_score(labels_vid, probs_vid),
                acc       = accuracy_score(labels_vid, preds_vid),
                f1        = f1_score(labels_vid, preds_vid, zero_division=0),
                kappa     = cohen_kappa_score(labels_vid, preds_vid),
                precision = precision_score(labels_vid, preds_vid, zero_division=0),
                recall    = recall_score(labels_vid, preds_vid, zero_division=0))


def cohen_d(a, b):
    na, nb = len(a), len(b)
    s = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2)
                / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / s if s > 0 else np.nan


def run_anova(seq_pooled, ml_pooled, label=""):
    print(f"\n{'Metric':10s}  {'Seq mean':>9s}  {'ML mean':>9s}  "
          f"{'F':>7s}  {'p':>10s}  {'d':>8s}  {'Sig?':>5s}")
    rows = []
    for m in SHARED_METRICS:
        F, p = f_oneway(seq_pooled[m], ml_pooled[m])
        d    = cohen_d(seq_pooled[m], ml_pooled[m])
        sig  = p < 0.05
        print(f"  {m:8s}  {seq_pooled[m].mean():9.4f}  {ml_pooled[m].mean():9.4f}  "
              f"{F:7.2f}  {p:10.4e}  {d:8.4f}  {'✓' if sig else '✗'}")
        rows.append(dict(metric=m, seq_mean=seq_pooled[m].mean(),
                         ml_mean=ml_pooled[m].mean(), F=F, p=p, d=d, sig=sig))
    n_sig = sum(r["sig"] for r in rows)
    verdict = ("SUPPORTED" if n_sig == len(rows)
               else f"PARTIALLY supported ({n_sig}/{len(rows)})" if n_sig
               else "NOT supported")
    print(f"\n  → RH3.1 {label}: {verdict} at α=0.05")
    return rows, verdict


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("RH3.1  Sequence models vs ML classifiers (two experiments)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load raw frame data ───────────────────────────────────────────────────
    print("\nLoading frame data …", flush=True)
    feat_sql = ", ".join(FEATURE_COLS)
    query = f"""
        SELECT filename, video_type, phoneme, {feat_sql}
        FROM lags
        ORDER BY filename, timestamp
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(query, conn)
    conn.close()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df["label"] = (df["video_type"] == "Fake").astype(int)

    # Encode phoneme → integer, then normalise to [0,1]
    le = LabelEncoder()
    df["phoneme_id"] = le.fit_transform(df["phoneme"])
    n_phonemes = df["phoneme_id"].nunique()
    df["phoneme_norm"] = df["phoneme_id"] / (n_phonemes - 1)

    ALL_FEAT_COLS = FEATURE_COLS + ["phoneme_norm"]  # 99 + 1 = 100 features
    n_feats = len(ALL_FEAT_COLS)

    print(f"  {len(df):,} frame rows  |  {n_phonemes} unique phonemes")

    # ── Build per-video data structures ───────────────────────────────────────
    video_names, video_labels = [], []
    sequences = []     # for seq models: ordered (T, n_feats) tensors (unscaled)
    frame_vidx = []    # video index for each frame row
    phon_mean_rows = []  # for Exp 2 ML: list of (phoneme_norm, mean_feats)

    for vidx, (fname, grp) in enumerate(df.groupby("filename", sort=False)):
        video_names.append(fname)
        video_labels.append(int(grp["label"].iloc[0]))
        seq = grp[ALL_FEAT_COLS].values.astype(np.float32)
        sequences.append(seq)
        frame_vidx.extend([vidx] * len(grp))
        # Per-(video, phoneme_type) mean rows for Exp 2
        for phon_id, pgrp in grp.groupby("phoneme_id"):
            row = pgrp[ALL_FEAT_COLS].mean().values.astype(np.float32)
            phon_mean_rows.append((vidx, row))

    video_labels = np.array(video_labels)
    frame_vidx   = np.array(frame_vidx, dtype=np.int32)
    n_videos     = len(video_labels)
    n_fake       = video_labels.sum()
    n_real       = n_videos - n_fake
    pos_weight   = n_real / n_fake

    # Raw frame matrix
    frame_X = df[ALL_FEAT_COLS].values.astype(np.float32)
    frame_y_vid = frame_vidx   # video label per frame = video_labels[frame_vidx]

    # Per-(video, phoneme) matrix for Exp 2
    phon_vidx_arr = np.array([r[0] for r in phon_mean_rows], dtype=np.int32)
    phon_X        = np.vstack([r[1] for r in phon_mean_rows]).astype(np.float32)

    print(f"  {n_videos:,} videos (Fake={n_fake:,} Real={n_real:,})  "
          f"|  {len(frame_X):,} frames  |  {len(phon_X):,} (video×phoneme) rows")

    # ── CV splits (by video index) ────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    vid_indices = np.arange(n_videos)
    folds = list(cv.split(vid_indices, video_labels))

    # ── ML classifiers ────────────────────────────────────────────────────────
    # LinearSVC replaces RBF SVM: O(n) memory vs O(n²), scales to millions of rows.
    # CalibratedClassifierCV wraps it to produce probability estimates.
    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=SEED,
            solver="saga"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            max_samples=0.3,   # bootstrap 30% to keep memory manageable
            random_state=SEED, n_jobs=-1),
        "Linear SVM": CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", max_iter=2000,
                      random_state=SEED), cv=3),
    }

    # ── Seq model definitions ─────────────────────────────────────────────────
    seq_model_defs = {
        "LSTM":        lambda: PlainLSTM(n_feats),
        "RNN":         lambda: PlainRNN(n_feats),
        "Transformer": lambda: PlainTransformer(n_feats),
    }

    # ══ EXP 1: Frame-level (same rows) ═══════════════════════════════════════
    print("\n" + "═"*65)
    print("Experiment 1 — Frame-level (ML on individual frames, same as seq)")
    print("═"*65)

    ml1_results = {}
    for clf_name, clf in classifiers.items():
        print(f"\n  [ML-Exp1] {clf_name} …", flush=True)
        fold_scores = {m: [] for m in SHARED_METRICS + ["precision","recall"]}
        for train_vid, val_vid in folds:
            train_mask = np.isin(frame_vidx, train_vid)
            val_mask   = np.isin(frame_vidx, val_vid)

            X_tr = frame_X[train_mask]
            X_va = frame_X[val_mask]
            y_tr_frame = video_labels[frame_vidx[train_mask]]

            scaler = fit_scaler_chunked(X_tr)
            X_tr_s = transform_chunked(scaler, X_tr)
            X_va_s = transform_chunked(scaler, X_va)
            del X_tr  # free memory before fit

            clf.fit(X_tr_s, y_tr_frame)
            del X_tr_s
            scores = eval_ml_video_level(clf, X_va_s, video_labels,
                                         frame_vidx[val_mask])
            del X_va_s
            for m in SHARED_METRICS + ["precision","recall"]:
                fold_scores[m].append(scores[m])

        ml1_results[clf_name] = {m: np.array(v) for m, v in fold_scores.items()}
        print(f"    AUC={ml1_results[clf_name]['auc'].mean():.4f}  "
              f"Acc={ml1_results[clf_name]['acc'].mean():.4f}  "
              f"F1={ml1_results[clf_name]['f1'].mean():.4f}  "
              f"Kappa={ml1_results[clf_name]['kappa'].mean():.4f}")

    ml1_pooled = {m: np.concatenate([ml1_results[n][m] for n in ml1_results])
                  for m in SHARED_METRICS}

    # ══ EXP 2: Phoneme-averaged ML ══════════════════════════════════════════
    print("\n" + "═"*65)
    print("Experiment 2 — Phoneme-averaged ML (per video×phoneme mean rows)")
    print("═"*65)

    ml2_results = {}
    for clf_name, clf in classifiers.items():
        print(f"\n  [ML-Exp2] {clf_name} …", flush=True)
        fold_scores = {m: [] for m in SHARED_METRICS + ["precision","recall"]}
        for train_vid, val_vid in folds:
            train_mask = np.isin(phon_vidx_arr, train_vid)
            val_mask   = np.isin(phon_vidx_arr, val_vid)

            X_tr = phon_X[train_mask]
            X_va = phon_X[val_mask]
            y_tr_phon = video_labels[phon_vidx_arr[train_mask]]

            scaler = fit_scaler_chunked(X_tr)
            X_tr_s = transform_chunked(scaler, X_tr)
            X_va_s = transform_chunked(scaler, X_va)
            del X_tr

            clf.fit(X_tr_s, y_tr_phon)
            del X_tr_s
            scores = eval_ml_video_level(clf, X_va_s, video_labels,
                                         phon_vidx_arr[val_mask])
            del X_va_s
            for m in SHARED_METRICS + ["precision","recall"]:
                fold_scores[m].append(scores[m])

        ml2_results[clf_name] = {m: np.array(v) for m, v in fold_scores.items()}
        print(f"    AUC={ml2_results[clf_name]['auc'].mean():.4f}  "
              f"Acc={ml2_results[clf_name]['acc'].mean():.4f}  "
              f"F1={ml2_results[clf_name]['f1'].mean():.4f}  "
              f"Kappa={ml2_results[clf_name]['kappa'].mean():.4f}")

    ml2_pooled = {m: np.concatenate([ml2_results[n][m] for n in ml2_results])
                  for m in SHARED_METRICS}

    # ══ Sequence models (same for both exps) ═════════════════════════════════
    print("\n" + "═"*65)
    print("Sequence models (ordered per-video sequences, used in both exps)")
    print("═"*65)

    seq_results = {}
    for arch_name, model_fn in seq_model_defs.items():
        print(f"\n  [{arch_name}] …", flush=True)
        fold_scores = {m: [] for m in SHARED_METRICS}
        for fold_i, (train_vid, val_vid) in enumerate(folds):
            print(f"    fold {fold_i+1}/{N_FOLDS}", flush=True)
            # Fit scaler on training frames (chunked to avoid float64 OOM)
            train_frames = np.concatenate([sequences[i] for i in train_vid])
            scaler = fit_scaler_chunked(train_frames)
            del train_frames
            scaled_seqs = [torch.tensor(scaler.transform(sequences[i]))
                           for i in range(n_videos)]
            torch.manual_seed(SEED)
            model  = model_fn().to(device)
            scores = train_eval_seq(model, train_vid, val_vid,
                                    scaled_seqs, video_labels, pos_weight, device)
            for m in SHARED_METRICS:
                fold_scores[m].append(scores[m])
            print(f"      AUC={scores['auc']:.4f}  F1={scores['f1']:.4f}")
        seq_results[arch_name] = {m: np.array(v) for m, v in fold_scores.items()}

    seq_pooled = {m: np.concatenate([seq_results[n][m] for n in seq_results])
                  for m in SHARED_METRICS}

    # ── Per-architecture summary ───────────────────────────────────────────────
    print("\n─── Per-architecture results ───────────────────────────────────")
    print("Sequence models:")
    for name in seq_results:
        r = seq_results[name]
        print(f"  {name:12s}  AUC={r['auc'].mean():.4f}  Acc={r['acc'].mean():.4f}  "
              f"F1={r['f1'].mean():.4f}  Kappa={r['kappa'].mean():.4f}")
    print("ML Exp1 (frame-level):")
    for name in ml1_results:
        r = ml1_results[name]
        print(f"  {name:22s}  AUC={r['auc'].mean():.4f}  Acc={r['acc'].mean():.4f}  "
              f"F1={r['f1'].mean():.4f}  Kappa={r['kappa'].mean():.4f}  "
              f"Prec={r['precision'].mean():.4f}  Rec={r['recall'].mean():.4f}")
    print("ML Exp2 (phoneme-averaged):")
    for name in ml2_results:
        r = ml2_results[name]
        print(f"  {name:22s}  AUC={r['auc'].mean():.4f}  Acc={r['acc'].mean():.4f}  "
              f"F1={r['f1'].mean():.4f}  Kappa={r['kappa'].mean():.4f}  "
              f"Prec={r['precision'].mean():.4f}  Rec={r['recall'].mean():.4f}")

    # ── ANOVA ─────────────────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("Exp 1 ANOVA: Seq models vs frame-level ML")
    rows1, verdict1 = run_anova(seq_pooled, ml1_pooled, "Exp1")

    print("\n" + "─"*65)
    print("Exp 2 ANOVA: Seq models vs phoneme-averaged ML")
    rows2, verdict2 = run_anova(seq_pooled, ml2_pooled, "Exp2")

    # ── Notes ─────────────────────────────────────────────────────────────────
    def arch_table(results):
        header = "| Architecture | AUC | Accuracy | F1 | Cohen's κ |"
        sep    = "|---|---|---|---|---|"
        lines  = [header, sep]
        for name, r in results.items():
            lines.append(f"| {name} | {r['auc'].mean():.4f} | "
                         f"{r['acc'].mean():.4f} | {r['f1'].mean():.4f} | "
                         f"{r['kappa'].mean():.4f} |")
        return "\n".join(lines)

    def ml_table(results):
        header = "| Classifier | AUC | Accuracy | F1 | Cohen's κ | Precision | Recall |"
        sep    = "|---|---|---|---|---|---|---|"
        lines  = [header, sep]
        for name, r in results.items():
            lines.append(f"| {name} | {r['auc'].mean():.4f} | "
                         f"{r['acc'].mean():.4f} | {r['f1'].mean():.4f} | "
                         f"{r['kappa'].mean():.4f} | "
                         f"{r['precision'].mean():.4f} | {r['recall'].mean():.4f} |")
        return "\n".join(lines)

    def anova_table(rows):
        header = "| Metric | Seq mean | ML mean | F | p | Cohen's d | Sig |"
        sep    = "|---|---|---|---|---|---|---|"
        lines  = [header, sep]
        for r in rows:
            lines.append(f"| {r['metric']} | {r['seq_mean']:.4f} | "
                         f"{r['ml_mean']:.4f} | {r['F']:.2f} | {r['p']:.4e} | "
                         f"{r['d']:.4f} | {'✓' if r['sig'] else '✗'} |")
        return "\n".join(lines)


    # ── Plots: one per metric, two boxes per plot (seq vs ml1 and ml2) ────────
    metric_labels = {"auc":"AUC","acc":"Accuracy","f1":"F1","kappa":"Cohen's κ"}
    c_seq  = "#0072B2"
    c_ml1  = "#E69F00"
    c_ml2  = "#009E73"

    for m in SHARED_METRICS:
        fig, ax = plt.subplots(figsize=(7, 5))
        sv  = seq_pooled[m]
        m1v = ml1_pooled[m]
        m2v = ml2_pooled[m]
        rng = np.random.RandomState(SEED)

        pos = [0, 1, 2]
        for vals, color, p in [(sv, c_seq, 0), (m1v, c_ml1, 1), (m2v, c_ml2, 2)]:
            jitter = rng.uniform(-0.12, 0.12, len(vals))
            ax.scatter(np.full(len(vals), p) + jitter, vals,
                       color=color, alpha=0.55, s=28, zorder=3)
        bp = ax.boxplot([sv, m1v, m2v], positions=pos, widths=0.3,
                        patch_artist=True,
                        boxprops=dict(alpha=0.35),
                        medianprops=dict(color="black", lw=2),
                        flierprops=dict(marker=""))
        colors_list = [c_seq, c_ml1, c_ml2]
        for box, c in zip(bp["boxes"], colors_list):
            box.set_facecolor(c)

        r1 = next(r for r in rows1 if r["metric"] == m)
        r2 = next(r for r in rows2 if r["metric"] == m)
        ax.set_xticks(pos)
        ax.set_xticklabels(["Sequence\n(LSTM/RNN/Transformer)",
                            "ML frame-level\n(LR/RF/SVM)",
                            "ML phon-averaged\n(LR/RF/SVM)"])
        ax.set_ylabel(metric_labels[m])
        ax.set_title(
            f"RH3.1 — {metric_labels[m]}\n"
            f"Exp1: F={r1['F']:.2f} p={r1['p']:.2e} d={r1['d']:.3f}  |  "
            f"Exp2: F={r2['F']:.2f} p={r2['p']:.2e} d={r2['d']:.3f}")
        fig.tight_layout()
        save_plot(fig, f"rh3_1_{m}.png")

    # ── ML bar chart (Exp2, 6 metrics) ────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    all_ml_metrics = ["auc","acc","f1","kappa","precision","recall"]
    ml_labels_str  = ["AUC","Accuracy","F1","Kappa","Precision","Recall"]
    clf_colors     = ["#CC79A7","#009E73","#56B4E9"]
    x = np.arange(len(all_ml_metrics))
    width = 0.25

    for ax, ml_res, title in [(axes[0], ml1_results, "Exp1: frame-level"),
                               (axes[1], ml2_results, "Exp2: phoneme-averaged")]:
        for i, (name, r) in enumerate(ml_res.items()):
            means = [r[mm].mean() for mm in all_ml_metrics]
            errs  = [r[mm].std()  for mm in all_ml_metrics]
            ax.bar(x + (i-1)*width, means, width, yerr=errs,
                   label=name, color=clf_colors[i], edgecolor="white",
                   capsize=3, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(ml_labels_str, fontsize=9)
        ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
        ax.set_title(f"RH3.1 — ML baselines ({title})")
        ax.legend(fontsize=9)
    fig.tight_layout()
    save_plot(fig, "rh3_1_ml_metrics.png")


if __name__ == "__main__":
    run()
