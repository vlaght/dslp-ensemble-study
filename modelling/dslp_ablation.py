"""
DSLP leave-one-out feature ablation.
5-fold CV over all data (dataset.db, no dataset filter, prefix='').
Each iteration excludes one feature (including 'phoneme_embedding').
Saves per-feature checkpoint after each fold; fully recoverable.

Usage:
    python -u modelling/dslp_ablation.py > .tmp/dslp_ablation.log 2>&1
"""
import sys, os, gc, ctypes
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MALLOC_ARENA_MAX", "1")
_libc = ctypes.CDLL("libc.so.6")
_libc.mallopt(ctypes.c_int(-3), ctypes.c_int(256 * 1024))
_libc.mallopt(ctypes.c_int(-1), ctypes.c_int(32 * 1024))

import sqlite3, json, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, cohen_kappa_score,
                              accuracy_score, f1_score, precision_score,
                              recall_score, confusion_matrix)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONEME_DB = os.path.join(SCRIPT_DIR, "data/dataset.db")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, ".tmp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RESULTS_FILE = os.path.join(OUTPUT_DIR, "dslp_ablation_results.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_FOLDS    = 5
BATCH_SIZE = 64
EPOCHS     = 30
PATIENCE   = 7
EMB_DIM    = 8
HIDDEN     = 256

# ── Modality sets ─────────────────────────────────────────────────────────────
N_MFCC = 13
MFCC_COLS = set(
    [f"mfcc_{i}"        for i in range(N_MFCC)]
    + [f"mfcc_delta_{i}"  for i in range(N_MFCC)]
    + [f"mfcc_delta2_{i}" for i in range(N_MFCC)]
    + ["audio_magnitude", "mfcc_energy"]
)
MOUTH_VISUAL = {
    "mar", "mouth_area", "mouth_width", "lip_curvature", "mouth_circularity",
    "jaw_forward", "jaw_left", "jaw_open", "jaw_right",
    "mouth_close", "mouth_dimple_left", "mouth_dimple_right",
    "mouth_frown_left", "mouth_frown_right", "mouth_funnel",
    "mouth_left", "mouth_lower_down_left", "mouth_lower_down_right",
    "mouth_press_left", "mouth_press_right", "mouth_pucker",
    "mouth_right", "mouth_roll_lower", "mouth_roll_upper",
    "mouth_shrug_lower", "mouth_shrug_upper", "mouth_smile_left",
    "mouth_smile_right", "mouth_stretch_left", "mouth_stretch_right",
    "mouth_upper_up_left", "mouth_upper_up_right",
}
_EXTRA_FACE = {
    "lip_distance", "_neutral",
    "brow_down_left", "brow_down_right", "brow_inner_up",
    "brow_outer_up_left", "brow_outer_up_right",
    "cheek_puff", "cheek_squint_left", "cheek_squint_right",
    "eye_blink_left", "eye_blink_right",
    "eye_look_down_left", "eye_look_down_right",
    "eye_look_in_left", "eye_look_in_right",
    "eye_look_out_left", "eye_look_out_right",
    "eye_look_up_left", "eye_look_up_right",
    "eye_squint_left", "eye_squint_right",
    "eye_wide_left", "eye_wide_right",
    "nose_sneer_left", "nose_sneer_right",
}
EXTENDED_VISUAL = MOUTH_VISUAL | _EXTRA_FACE
AUDIO_FEATURES  = MFCC_COLS

PHONEME_FEAT = "phoneme_embedding"


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t   = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


class AblationDSLP(nn.Module):
    """UniDSLP with configurable embedding_dim; embedding_dim=0 disables phoneme."""

    def __init__(self, num_visual, num_audio, num_phonemes,
                 embedding_dim=EMB_DIM, hidden=HIDDEN):
        super().__init__()
        self.emb_dim = embedding_dim
        if embedding_dim > 0:
            self.phoneme_embedding = nn.Embedding(num_phonemes, embedding_dim)

        self.vis_lstm = nn.LSTM(num_visual + embedding_dim, hidden, 2,
                                batch_first=True, dropout=0.3)
        self.vis_inst = nn.Linear(hidden, 1)

        self.aud_lstm = nn.LSTM(num_audio + embedding_dim, hidden, 2,
                                batch_first=True, dropout=0.3)
        self.aud_inst = nn.Linear(hidden, 1)

        self.fusion = nn.Sequential(
            nn.Linear(2, 16), nn.GELU(), nn.Dropout(0.3), nn.Linear(16, 1)
        )

    def _stream(self, lstm, inst, feats, phon_emb, lengths):
        T = feats.shape[1]
        x = torch.cat([feats, phon_emb], dim=-1) if self.emb_dim > 0 else feats
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=T)
        scores = inst(out)
        valid  = (torch.arange(T, device=feats.device).unsqueeze(0)
                  < lengths.to(feats.device).unsqueeze(1))
        scores = scores.masked_fill(~valid.unsqueeze(-1), 0.0)
        return scores.sum(1) / lengths.float().unsqueeze(-1).to(feats.device)

    def forward(self, visual, audio, phons, lengths):
        phon_emb = self.phoneme_embedding(phons) if self.emb_dim > 0 else None
        vis = self._stream(self.vis_lstm, self.vis_inst, visual, phon_emb, lengths)
        aud = self._stream(self.aud_lstm, self.aud_inst, audio, phon_emb, lengths)
        return self.fusion(torch.cat([vis, aud], dim=-1)).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset / collate
# ═══════════════════════════════════════════════════════════════════════════════

class CrossModalDataset(Dataset):
    def __init__(self, sequences, labels, visual_idx, audio_idx):
        self.sequences  = sequences
        self.labels     = labels
        self.visual_idx = visual_idx
        self.audio_idx  = audio_idx

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        feats, phons = self.sequences[idx]
        return (
            torch.tensor(feats[:, self.visual_idx], dtype=torch.float32),
            torch.tensor(feats[:, self.audio_idx],  dtype=torch.float32),
            torch.tensor(phons, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def collate_fn(batch):
    vis, aud, phon, lbl = zip(*batch)
    lengths  = torch.tensor([v.shape[0] for v in vis], dtype=torch.long)
    vis_pad  = nn.utils.rnn.pad_sequence(vis,  batch_first=True)
    aud_pad  = nn.utils.rnn.pad_sequence(aud,  batch_first=True)
    phon_pad = nn.utils.rnn.pad_sequence(phon, batch_first=True)
    return vis_pad, aud_pad, phon_pad, lengths, torch.stack(lbl)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_data():
    """Stream lags table sorted by filename; emit sequences without full-df in RAM."""
    print(f"Loading {PHONEME_DB} ...")
    META = {"filename", "video_type", "dataset", "phoneme", "frame_position",
            "timestamp", "label_encoded", "phoneme_encoded", "phoneme_duration"}

    conn = sqlite3.connect(PHONEME_DB)

    # Pass 1: schema + phoneme vocabulary (tiny)
    sample = pd.read_sql(
        "SELECT * FROM lags WHERE frame_position = 'onset' LIMIT 1", conn)
    feat_cols = [c for c in sample.columns
                 if c not in META and pd.api.types.is_numeric_dtype(sample[c])]

    all_phonemes = pd.read_sql(
        "SELECT DISTINCT phoneme FROM lags WHERE frame_position='onset'", conn
    )["phoneme"].tolist()
    le = LabelEncoder().fit(sorted(all_phonemes))

    # Pass 2: chunked stream ORDER BY filename,timestamp
    # Each chunk processed with groupby; carry-over buffer for split filenames.
    sequences, labels, filenames = [], [], []
    buf: pd.DataFrame = pd.DataFrame()
    total_rows = 0

    def _emit_grp(grp):
        ts    = grp["timestamp"].values
        feats = grp[feat_cols].values.astype(np.float32)
        phons = le.transform(grp["phoneme"].values).astype(np.int64)
        # IOI
        diffs       = np.diff(ts, append=np.nan)
        mean_dur    = float(np.nanmean(diffs)) if diffs.size > 1 else 0.0
        diffs       = np.where(np.isnan(diffs), mean_dur, diffs).astype(np.float32)
        feats_ioi   = np.hstack([feats, diffs[:, None]])
        label       = 1 if grp["video_type"].iloc[0] == "Fake" else 0
        sequences.append((feats_ioi, phons))
        labels.append(label)
        filenames.append(grp["filename"].iloc[0])

    sql = (
        "SELECT filename, video_type, timestamp, phoneme, "
        + ", ".join(feat_cols)
        + " FROM lags WHERE frame_position='onset' ORDER BY filename, timestamp"
    )
    for chunk in pd.read_sql(sql, conn, chunksize=50_000):
        total_rows += len(chunk)
        if not buf.empty:
            chunk = pd.concat([buf, chunk], ignore_index=True)

        last_fname = chunk["filename"].iloc[-1]
        complete   = chunk[chunk["filename"] != last_fname]
        buf        = chunk[chunk["filename"] == last_fname].copy()

        for fname, grp in complete.groupby("filename", sort=False):
            _emit_grp(grp)

        del chunk, complete
        gc.collect()

    # flush final buffer
    if not buf.empty:
        for fname, grp in buf.groupby("filename", sort=False):
            _emit_grp(grp)
    conn.close()

    feat_cols_with_ioi = feat_cols + ["phoneme_duration"]
    print(f"  {total_rows:,} rows -> {len(sequences)} videos | "
          f"{len(feat_cols_with_ioi)} feats | {len(le.classes_)} phonemes | "
          f"Fake={sum(labels)} Real={sum(l==0 for l in labels)}")
    return sequences, np.array(labels), filenames, feat_cols_with_ioi, le


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def apply_delta(seqs):
    out = []
    for feats, phons in seqs:
        delta = np.zeros_like(feats)
        if len(feats) > 1:
            delta[1:] = feats[1:] - feats[:-1]
        out.append((np.hstack([feats, delta]).astype(np.float32), phons))
    return out


def apply_global_scale(train_seqs, test_seqs):
    scaler = StandardScaler()
    for f, _ in train_seqs:
        scaler.partial_fit(f)
    def _s(seqs):
        return [(scaler.transform(f).astype(np.float32), p) for f, p in seqs]
    return _s(train_seqs), _s(test_seqs)


def apply_vstats_std(seqs):
    out = []
    for feats, phons in seqs:
        std = feats.std(axis=0, keepdims=True).repeat(len(feats), axis=0)
        std = np.where(std < 1e-8, 1e-8, std)
        out.append((np.hstack([feats, std]).astype(np.float32), phons))
    return out


def build_pipeline(tr, te):
    tr = apply_delta(tr)
    te = apply_delta(te)
    tr, te = apply_global_scale(tr, te)
    tr = apply_vstats_std(tr)
    te = apply_vstats_std(te)
    return tr, te


def build_modality_indices(feat_cols):
    """Return (visual_idx, audio_idx) into the post-pipeline feature array."""
    col_mod = []
    for c in feat_cols:
        if c in EXTENDED_VISUAL:
            col_mod.append("v")
        elif c in AUDIO_FEATURES:
            col_mod.append("a")
        else:
            col_mod.append("o")
    col_mod = col_mod + col_mod   # delta
    col_mod = col_mod + col_mod   # vstats_std
    visual_idx = [i for i, m in enumerate(col_mod) if m == "v"]
    audio_idx  = [i for i, m in enumerate(col_mod) if m == "a"]
    return visual_idx, audio_idx


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════

def make_focal_loss(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    return FocalLoss(gamma=2.0, pos_weight=pw)


def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for vis, aud, phon, lengths, lbls in loader:
        vis, aud, phon, lbls = vis.to(device), aud.to(device), phon.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(vis, aud, phon, lengths), lbls)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, lbls = [], []
    for vis, aud, phon, lengths, labels in loader:
        vis, aud, phon = vis.to(device), aud.to(device), phon.to(device)
        p = torch.sigmoid(model(vis, aud, phon, lengths)).cpu().numpy()
        probs.extend(p)
        lbls.extend(labels.numpy())
    return np.array(probs), np.array(lbls)


def train_model(model, tr_loader, va_loader, y_train):
    criterion = make_focal_loss(y_train)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    best_auc, best_state, no_imp = 0.0, None, 0
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, tr_loader, criterion, optimizer)
        probs, labels = predict(model, va_loader)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5
        scheduler.step(auc)
        if epoch % 5 == 0:
            print(f"      epoch {epoch:3d}/{EPOCHS}  loss={loss:.4f}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print(f"      early stop at epoch {epoch}")
                break
    if best_state:
        model.load_state_dict(best_state)
    return model


def compute_metrics(probs, labels, thr=0.5):
    auc   = roc_auc_score(labels, probs)
    preds = (probs >= thr).astype(int)
    acc   = accuracy_score(labels, preds)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    f1    = f1_score(labels, preds, zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc":         round(float(auc),   4),
        "accuracy":    round(float(acc),   4),
        "precision":   round(float(prec),  4),
        "recall":      round(float(rec),   4),
        "f1":          round(float(f1),    4),
        "kappa":       round(float(kappa), 4),
        "specificity": round(float(spec),  4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Ablation loop
# ═══════════════════════════════════════════════════════════════════════════════

def safe_key(feat_name):
    """Filesystem-safe key for checkpoint filenames."""
    return re.sub(r"[^a-zA-Z0-9_]", "_", feat_name)


def ckpt_path(feat_name):
    return os.path.join(OUTPUT_DIR, f"dslp_ablation_{safe_key(feat_name)}_ckpt.json")


def load_ckpt(feat_name):
    p = ckpt_path(feat_name)
    if os.path.exists(p):
        with open(p) as f:
            return {int(k): v for k, v in json.load(f).items()}
    return {}


def save_ckpt(feat_name, data):
    with open(ckpt_path(feat_name), "w") as f:
        json.dump(data, f)


def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


def summarise(fold_data):
    """Compute mean/std over folds for each metric."""
    metrics = [k for k in next(iter(fold_data.values()))["metrics"].keys()
               if k not in ("TP", "TN", "FP", "FN")]
    mean = {m: round(float(np.mean([fold_data[fi]["metrics"][m] for fi in fold_data])), 4)
            for m in metrics}
    std  = {m: round(float(np.std( [fold_data[fi]["metrics"][m] for fi in fold_data])), 4)
            for m in metrics}
    return {"mean": mean, "std": std}


def run_ablation(excluded_feat, active_feat_cols, visual_idx, audio_idx,
                 sequences, labels, fold_splits, num_phonemes, emb_dim):
    """Run N_FOLDS-fold CV with one feature excluded. Returns per-fold dict."""
    fold_data = load_ckpt(excluded_feat)
    n_done    = len(fold_data)
    print(f"\n  Excluded: {excluded_feat!r}  "
          f"({len(active_feat_cols)} raw feats, emb_dim={emb_dim}, "
          f"vis={len(visual_idx)} aud={len(audio_idx)}, resuming fold {n_done+1})")

    for fold_i, (tr_idx, te_idx) in fold_splits:
        if fold_i <= n_done:
            continue
        print(f"\n    Fold {fold_i}/{N_FOLDS}  train={len(tr_idx)} test={len(te_idx)}")

        tr_seqs = [sequences[i] for i in tr_idx]
        te_seqs = [sequences[i] for i in te_idx]
        y_tr    = labels[tr_idx]
        y_te    = labels[te_idx]

        tr_seqs, te_seqs = build_pipeline(tr_seqs, te_seqs)

        torch.manual_seed(SEED + fold_i)
        model = AblationDSLP(len(visual_idx), len(audio_idx),
                              num_phonemes, embedding_dim=emb_dim).to(device)
        tr_loader = DataLoader(
            CrossModalDataset(tr_seqs, y_tr, visual_idx, audio_idx),
            batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        te_loader = DataLoader(
            CrossModalDataset(te_seqs, y_te, visual_idx, audio_idx),
            batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        model     = train_model(model, tr_loader, te_loader, y_tr)
        probs, gt = predict(model, te_loader)

        del model, tr_loader, te_loader, tr_seqs, te_seqs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        _libc.malloc_trim(0)

        m = compute_metrics(probs, gt)
        print(f"    fold {fold_i}: {m}")
        fold_data[fold_i] = {"metrics": m}
        save_ckpt(excluded_feat, fold_data)
        print(f"  Checkpoint saved ({fold_i}/{N_FOLDS} folds done)", flush=True)

    return fold_data


def drop_feature(sequences, feat_cols, exclude):
    """Return sequences and feat_cols with one column removed."""
    idx = feat_cols.index(exclude)
    new_cols = [c for c in feat_cols if c != exclude]
    new_seqs = [(f[:, [i for i in range(f.shape[1]) if i != idx]], p)
                for f, p in sequences]
    return new_seqs, new_cols


def main():
    sequences, labels, filenames, feat_cols, le = load_data()
    num_phonemes = len(le.classes_)

    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    indices     = np.arange(len(sequences))
    fold_splits = list(enumerate(skf.split(indices, labels), 1))

    # Feature list: all raw feature columns + the phoneme embedding as a special entry
    all_feats = feat_cols + [PHONEME_FEAT]

    results = load_results()
    print(f"\nTotal features to ablate: {len(all_feats)}")
    print(f"Already done: {len(results)}")

    # ── Baseline: full feature set ────────────────────────────────────────────
    BASELINE_KEY = "__baseline__"
    if not (BASELINE_KEY in results and len(results[BASELINE_KEY].get("folds", {})) == N_FOLDS):
        print("\n[Baseline] Running full-feature CV ...")
        visual_idx_b, audio_idx_b = build_modality_indices(feat_cols)
        fold_data_b = run_ablation(
            BASELINE_KEY, feat_cols, visual_idx_b, audio_idx_b,
            sequences, labels, fold_splits, num_phonemes, EMB_DIM,
        )
        summary_b = summarise(fold_data_b)
        results[BASELINE_KEY] = {"folds": {str(fi): fd for fi, fd in fold_data_b.items()}, **summary_b}
        save_results(results)
        print(f"  Baseline AUC: {summary_b['mean']['auc']:.4f} +/- {summary_b['std']['auc']:.4f}")
    else:
        print(f"  Baseline AUC: {results[BASELINE_KEY]['mean']['auc']:.4f} "
              f"+/- {results[BASELINE_KEY]['std']['auc']:.4f} (cached)")

    for excluded in all_feats:
        if excluded in results and len(results[excluded].get("folds", {})) == N_FOLDS:
            print(f"  Skipping {excluded!r} (already complete)")
            continue

        if excluded == PHONEME_FEAT:
            active_seqs   = sequences
            active_cols   = feat_cols
            emb_dim       = 0
        else:
            active_seqs, active_cols = drop_feature(sequences, feat_cols, excluded)
            emb_dim = EMB_DIM

        visual_idx, audio_idx = build_modality_indices(active_cols)

        fold_data = run_ablation(
            excluded, active_cols, visual_idx, audio_idx,
            active_seqs, labels, fold_splits, num_phonemes, emb_dim,
        )

        summary = summarise(fold_data)
        results[excluded] = {
            "folds":   {str(fi): fd for fi, fd in fold_data.items()},
            **summary,
        }
        save_results(results)
        print(f"\n  Feature {excluded!r} done: "
              f"AUC {summary['mean']['auc']:.4f} +/- {summary['std']['auc']:.4f}")

        if excluded != PHONEME_FEAT:
            del active_seqs
        gc.collect()
        _libc.malloc_trim(0)

    print(f"\nAblation complete. Results saved to {RESULTS_FILE}")

    # Print summary sorted by AUC drop
    print(f"\n{'='*60}")
    print("Feature importance (sorted by mean AUC, descending drop = more important)")
    print(f"{'='*60}")
    baseline = results.get("__baseline__", {}).get("mean", {}).get("auc", None)
    rows = [(feat, results[feat]["mean"]["auc"]) for feat in results if feat != "__baseline__"]
    for feat, auc in sorted(rows, key=lambda x: x[1]):
        print(f"  excl {feat:40s}  AUC={auc:.4f}")


if __name__ == "__main__":
    main()
