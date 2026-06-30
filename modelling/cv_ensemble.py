"""
10-fold stratified cross-validation of the best 3-model ensemble:
    DSLP h=256 (phoneme) + TA-BiLSTM (freq) + TA-BiLSTM (noise)

Data used (intersection across all three feature sets):
    Phoneme : exported_phonemes.db  (deepspeakv2 + FakeAVCeleb_v1.2 + augmented)
    Freq    : freq_features_dsv2.db + freq_features.db
    Noise   : noise_features_dsv2.db + noise_features.db

Only videos present in ALL three feature sets are included (~21,600 expected).

Per-fold metrics + aggregate (mean±std) + pooled (all predictions concatenated).

Usage:
    python.exe -u modelling/cv_ensemble.py > .tmp/cv_ensemble.log 2>&1
"""
import sys, os, gc, ctypes
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force glibc to mmap large allocs (>= 256 KB) so freed memory returns to OS immediately,
# preventing heap fragmentation from large numpy/pandas allocations across folds.
# M_MMAP_THRESHOLD=-3 ; M_TRIM_THRESHOLD=-1 ; M_ARENA_MAX via env var
os.environ.setdefault("MALLOC_ARENA_MAX", "1")   # single arena → malloc_trim more effective
if sys.platform != "win32":
    _libc = ctypes.CDLL("libc.so.6")
    _libc.mallopt(ctypes.c_int(-3), ctypes.c_int(256 * 1024))   # M_MMAP_THRESHOLD = 256 KB
    _libc.mallopt(ctypes.c_int(-1), ctypes.c_int(32 * 1024))    # M_TRIM_THRESHOLD = 32 KB
else:
    _libc = None

import sqlite3, json
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
from modelling.dslp_arch import UniDSLP

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

SCRIPT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHONEME_DB      = os.path.join(SCRIPT_DIR, "data/dataset.db")       # relpaths for EXT datasets
PHONEME_DB_DSV2 = os.path.join(SCRIPT_DIR, "data", "deepspeakv2.db")   # basenames for deepspeakv2
FREQ_DB_DSV2    = os.path.join(SCRIPT_DIR, "data", "freq_features_dsv2.db")
FREQ_DB_EXT   = os.path.join(SCRIPT_DIR, "data", "freq_features.db")
NOISE_DB_DSV2 = os.path.join(SCRIPT_DIR, "data", "noise_features_dsv2.db")
NOISE_DB_EXT  = os.path.join(SCRIPT_DIR, "data", "noise_features.db")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, ".tmp")
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

N_FOLDS    = 10
BATCH_SIZE = 64

# ── Feature set definitions (snake_case to match deepspeakv2.db) ─────────────
N_MFCC = 13
MFCC_COLS = set(
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + [f"mfcc_delta_{i}" for i in range(N_MFCC)]
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

PHONEME_DATASETS = {"deepspeakv2", "FakeAVCeleb_v1.2", "augmented"}


# ═══════════════════════════════════════════════════════════════════════════════
# Models
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
        p_t = probs * targets + (1 - probs) * (1 - targets)
        return ((1 - p_t) ** self.gamma * bce).mean()


# Phoneme model: UniDSLP imported from modelling.dslp_arch


class TemporalAttentionLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_score = nn.Linear(hidden_dim * 2, 1)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, feats, lengths):
        packed  = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _  = self.lstm(packed)
        out, _  = pad_packed_sequence(out, batch_first=True)
        B, T, H = out.shape
        mask    = torch.arange(T, device=out.device).unsqueeze(0) < lengths.to(out.device).unsqueeze(1)
        scores  = self.attn_score(out).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1).unsqueeze(1)
        context = torch.bmm(weights, out).squeeze(1)
        return self.classifier(context).squeeze(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Datasets / collate
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
            torch.tensor(phons,  dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )


def collate_fn_crossmodal(batch):
    vis_list, aud_list, phon_list, label_list = zip(*batch)
    lengths  = torch.tensor([v.shape[0] for v in vis_list], dtype=torch.long)
    vis_pad  = nn.utils.rnn.pad_sequence(vis_list,  batch_first=True)
    aud_pad  = nn.utils.rnn.pad_sequence(aud_list,  batch_first=True)
    phon_pad = nn.utils.rnn.pad_sequence(phon_list, batch_first=True)
    return vis_pad, aud_pad, phon_pad, lengths, torch.stack(label_list)


class FrameDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels    = labels

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx],    dtype=torch.float32),
        )


def collate_fn_frame(batch):
    feats_list, labels_list = zip(*batch)
    lengths   = torch.tensor([f.shape[0] for f in feats_list], dtype=torch.long)
    feats_pad = nn.utils.rnn.pad_sequence(feats_list, batch_first=True)
    return feats_pad, lengths, torch.stack(labels_list)


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_phoneme_data(db_specs, filenames_filter=None):
    """Load phoneme sequences from one or more DBs.

    db_specs: list of (db_path, dataset_filter_set_or_None, filename_prefix) tuples.
      - db_path: path to SQLite DB with a 'lags' table
      - dataset_filter: set of dataset names to keep (None = keep all rows)
      - filename_prefix: string prepended to every filename (e.g. "deepspeakv2/")
    filenames_filter: optional set of filenames to keep (post-prefix), avoids OOM on large DBs.
    """
    all_dfs = []
    for db_path, ds_filter, prefix in db_specs:
        print(f"  Loading {os.path.basename(db_path)} (prefix={prefix!r}) ...")
        conn   = sqlite3.connect(db_path)
        chunks = []
        for chunk in pd.read_sql(
            "SELECT * FROM lags WHERE frame_position = 'onset'", conn, chunksize=50000
        ):
            if ds_filter is not None and "dataset" in chunk.columns:
                chunk = chunk[chunk["dataset"].isin(ds_filter)]
            if prefix:
                chunk = chunk.copy()
                chunk["filename"] = prefix + chunk["filename"]
            if filenames_filter is not None:
                chunk = chunk[chunk["filename"].isin(filenames_filter)]
            if not chunk.empty:
                chunks.append(chunk)
        conn.close()
        if chunks:
            all_dfs.append(pd.concat(chunks, ignore_index=True))
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.sort_values(["filename", "timestamp"]).reset_index(drop=True)
    print(f"  {len(df):,} rows, {df['filename'].nunique():,} videos")

    # IOI
    def _ioi(s):
        diffs = np.diff(s.values, append=np.nan)
        valid = diffs[~np.isnan(diffs)]
        mean_dur = float(valid.mean()) if len(valid) > 0 else 0.0
        diffs[np.isnan(diffs)] = mean_dur
        return pd.Series(diffs, index=s.index)
    df["phoneme_duration"] = df.groupby("filename", sort=False)["timestamp"].transform(_ioi)

    META = {"filename", "video_type", "dataset", "phoneme", "frame_position",
            "timestamp", "label_encoded", "phoneme_encoded"}
    HARMFUL_FEATURES = {
        'mfcc_4', 'mouth_upper_up_left', 'mfcc_11', 'eye_look_down_right', 'audio_magnitude',
        'mfcc_delta2_10', 'mfcc_delta2_2', 'mfcc_delta_3', 'mfcc_delta2_0', 'mfcc_delta_11',
        'brow_down_left', 'mouth_stretch_right', 'eye_look_up_left', '_neutral', 'eye_blink_left',
        'mfcc_delta2_3', 'mfcc_7', 'mouth_frown_right', 'cheek_puff', 'mfcc_delta2_12',
        'mouth_upper_up_right', 'mouth_lower_down_right', 'mfcc_delta_8', 'mfcc_delta_7',
        'mfcc_delta2_11', 'mfcc_delta2_7', 'mfcc_delta2_4', 'lip_curvature', 'jaw_open',
        'brow_inner_up', 'jaw_left', 'mouth_shrug_lower', 'mfcc_12', 'mouth_left',
        'mfcc_delta_4', 'mfcc_8', 'mouth_press_left', 'mouth_frown_left', 'nose_sneer_left',
        'brow_outer_up_right', 'mouth_pucker',
    }
    feat_cols = [c for c in df.columns
                 if c not in META and c not in HARMFUL_FEATURES
                 and pd.api.types.is_numeric_dtype(df[c])]

    # LabelEncoder fitted globally on all phonemes (phoneme types are fixed)
    le_phoneme = LabelEncoder()
    df["phoneme_encoded"] = le_phoneme.fit_transform(df["phoneme"])
    df["label_encoded"]   = (df["video_type"] == "Fake").astype(int)

    sequences, labels, filenames = [], [], []
    for fname, grp in df.groupby("filename", sort=False):
        feats = grp[feat_cols].values.astype(np.float32)
        phons = grp["phoneme_encoded"].values.astype(np.int64)
        sequences.append((feats, phons))
        labels.append(int(grp["label_encoded"].iloc[0]))
        filenames.append(fname)

    print(f"  {len(sequences)} sequences | {len(feat_cols)} features | {len(le_phoneme.classes_)} phonemes")
    print(f"  Fake={sum(labels)}  Real={sum(l==0 for l in labels)}")
    return sequences, np.array(labels), filenames, feat_cols, le_phoneme


def load_combined_frame_data(*dbs, filenames_filter=None):
    """Load frame-level features from one or more SQLite DBs.

    dbs elements: plain path string OR (path, prefix) tuple.
    filenames_filter: optional set of filenames to keep (post-prefix).
    """
    dfs = []
    for spec in dbs:
        if isinstance(spec, tuple):
            db, prefix = spec
        else:
            db, prefix = spec, ""
        if not os.path.exists(db):
            print(f"  [WARN] {db} not found, skipping")
            continue
        print(f"  Loading {os.path.basename(db)} (prefix={prefix!r}) ...")
        conn = sqlite3.connect(db)
        df   = pd.read_sql("SELECT * FROM features ORDER BY filename, frame_idx", conn)
        conn.close()
        if prefix:
            df["filename"] = prefix + df["filename"]
        if filenames_filter is not None:
            df = df[df["filename"].isin(filenames_filter)]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(["filename", "frame_idx"]).reset_index(drop=True)
    print(f"  Combined: {len(df):,} rows, {df['filename'].nunique():,} unique paths")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    sequences, labels, filenames = [], [], []
    for fname, grp in df.groupby("filename", sort=False):
        grp   = grp.sort_values("frame_idx")
        feats = grp[feat_cols].values.astype(np.float32)
        vtype = grp["video_type"].iloc[0]
        label = 1 if vtype == "Fake" else 0
        sequences.append(feats)
        labels.append(label)
        filenames.append(fname)

    print(f"  {len(sequences)} sequences  Fake={sum(labels)}  Real={sum(l==0 for l in labels)}")
    return sequences, np.array(labels), filenames


# ═══════════════════════════════════════════════════════════════════════════════
# Transforms
# ═══════════════════════════════════════════════════════════════════════════════

def apply_av_interaction(sequences, feature_cols):
    col_idx  = {c: i for i, c in enumerate(feature_cols)}
    av_pairs = [
        ("jaw_open",    "audio_magnitude"),
        ("jaw_open",    "mfcc_energy"),
        ("mouth_area",  "mfcc_energy"),
        ("mouth_area",  "audio_magnitude"),
        ("mar",         "mfcc_energy"),
        ("mouth_close", "audio_magnitude"),
    ]
    valid = [(col_idx[a], col_idx[b]) for a, b in av_pairs
             if col_idx.get(a, -1) >= 0 and col_idx.get(b, -1) >= 0]
    if not valid:
        return sequences, 0
    result = []
    for feats, phons in sequences:
        products = np.stack([feats[:, ai] * feats[:, bi] for ai, bi in valid], axis=1)
        result.append((np.hstack([feats, products]).astype(np.float32), phons))
    return result, len(valid)


def apply_delta_phoneme(sequences):
    result = []
    for feats, phons in sequences:
        delta = np.zeros_like(feats)
        if len(feats) > 1:
            delta[1:] = feats[1:] - feats[:-1]
        result.append((np.hstack([feats, delta]).astype(np.float32), phons))
    return result


def apply_global_scale_phoneme(train_seqs, test_seqs):
    scaler = StandardScaler()
    for f, _ in train_seqs:
        scaler.partial_fit(f)
    def _scale(seqs):
        return [(scaler.transform(f).astype(np.float32), p) for f, p in seqs]
    return _scale(train_seqs), _scale(test_seqs)


def apply_vstats_std_phoneme(sequences):
    result = []
    for feats, phons in sequences:
        std = feats.std(axis=0, keepdims=True).repeat(len(feats), axis=0)
        std = np.where(std < 1e-8, 1e-8, std)
        result.append((np.hstack([feats, std]).astype(np.float32), phons))
    return result


def build_col_mod(feature_cols):
    col_mod = []
    for c in feature_cols:
        if c in EXTENDED_VISUAL:
            col_mod.append("v")
        elif c in AUDIO_FEATURES:
            col_mod.append("a")
        else:
            col_mod.append("o")
    col_mod = col_mod + col_mod  # delta doubles
    col_mod = col_mod + col_mod  # vstats_std doubles
    return col_mod


def fit_global_scaler_frame(train_seqs):
    all_rows  = np.concatenate(train_seqs, axis=0)
    col_means = np.nanmean(all_rows, axis=0)
    nan_mask  = np.isnan(all_rows)
    all_rows[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
    scaler = StandardScaler().fit(all_rows)
    return scaler, col_means


def apply_global_scale_frame(sequences, scaler, col_means):
    result = []
    for s in sequences:
        s = s.copy()
        nan_mask = np.isnan(s)
        if nan_mask.any():
            s[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        result.append(scaler.transform(s).astype(np.float32))
    return result


def apply_delta_frame(sequences):
    result = []
    for s in sequences:
        delta = np.zeros_like(s)
        if len(s) > 1:
            delta[1:] = s[1:] - s[:-1]
        result.append(np.concatenate([s, delta], axis=1))
    return result


def apply_vstats_full_frame(sequences):
    result = []
    for s in sequences:
        mean = np.mean(s, axis=0, keepdims=True).repeat(len(s), axis=0)
        std  = np.std(s,  axis=0, keepdims=True).repeat(len(s), axis=0)
        result.append(np.concatenate([s, mean, std], axis=1))
    return result


def build_frame_pipeline(train_seqs, test_seqs):
    scaler, col_means = fit_global_scaler_frame(train_seqs)
    train_seqs = apply_global_scale_frame(train_seqs, scaler, col_means)
    test_seqs  = apply_global_scale_frame(test_seqs,  scaler, col_means)
    train_seqs = apply_delta_frame(train_seqs)
    test_seqs  = apply_delta_frame(test_seqs)
    train_seqs = apply_vstats_full_frame(train_seqs)
    test_seqs  = apply_vstats_full_frame(test_seqs)
    return train_seqs, test_seqs


# ═══════════════════════════════════════════════════════════════════════════════
# Training helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_focal_loss(y_train):
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pw = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=device)
    print(f"    pos_weight={pw.item():.3f}  (neg={n_neg}, pos={n_pos})")
    return FocalLoss(gamma=2.0, pos_weight=pw)


def train_epoch_crossmodal(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for vis, aud, phons, lengths, labels in loader:
        vis, aud, phons, labels = vis.to(device), aud.to(device), phons.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(vis, aud, phons, lengths), labels)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict_crossmodal(model, loader):
    model.eval()
    probs, lbls = [], []
    for vis, aud, phons, lengths, labels in loader:
        vis, aud, phons = vis.to(device), aud.to(device), phons.to(device)
        p = torch.sigmoid(model(vis, aud, phons, lengths)).cpu().numpy()
        probs.extend(p); lbls.extend(labels.numpy())
    return np.array(probs), np.array(lbls)


def train_epoch_frame(model, loader, criterion, optimizer):
    model.train()
    total = 0.0
    for feats, lengths, labels in loader:
        feats, labels = feats.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(feats, lengths), labels)
        loss.backward(); optimizer.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict_frame(model, loader):
    model.eval()
    probs, lbls = [], []
    for feats, lengths, labels in loader:
        feats = feats.to(device)
        p = torch.sigmoid(model(feats, lengths)).cpu().numpy()
        probs.extend(p); lbls.extend(labels.numpy())
    return np.array(probs), np.array(lbls)


def train_model(model, train_fn, pred_fn, train_loader, val_loader, y_train,
                epochs=30, lr=0.001, patience=7, name="model"):
    criterion = make_focal_loss(y_train)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    best_auc, best_state, no_imp = 0.0, None, 0
    for epoch in range(1, epochs + 1):
        loss = train_fn(model, train_loader, criterion, optimizer)
        probs, labels = pred_fn(model, val_loader)
        try:
            auc = roc_auc_score(labels, probs)
        except Exception:
            auc = 0.5
        scheduler.step(auc)
        if epoch % 5 == 0:
            print(f"      [{name}] epoch {epoch:3d}/{epochs}  loss={loss:.4f}  val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}; no_imp = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"      [{name}] early stop at epoch {epoch}"); break
    if best_state:
        model.load_state_dict(best_state)
    return model, best_auc


def compute_metrics(probs, labels, thr=0.5):
    auc   = roc_auc_score(labels, probs)
    preds = (probs >= thr).astype(int)
    kappa = cohen_kappa_score(labels, preds)
    acc   = accuracy_score(labels, preds)
    f1    = f1_score(labels, preds, zero_division=0)
    prec  = precision_score(labels, preds, zero_division=0)
    rec   = recall_score(labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    best_k, best_thr = kappa, thr
    for t in np.arange(0.05, 0.96, 0.05):
        try:
            k = cohen_kappa_score(labels, (probs >= t).astype(int))
        except Exception:
            k = 0.0
        if k > best_k:
            best_k, best_thr = k, t

    # metrics at optimal threshold
    preds_opt = (probs >= best_thr).astype(int)
    prec_opt  = precision_score(labels, preds_opt, zero_division=0)
    rec_opt   = recall_score(labels, preds_opt, zero_division=0)
    tn_o, fp_o, fn_o, tp_o = confusion_matrix(labels, preds_opt, labels=[0, 1]).ravel()
    spec_opt  = tn_o / (tn_o + fp_o) if (tn_o + fp_o) > 0 else 0.0

    return {
        "auc":        round(float(auc),    4),
        "kappa":      round(float(kappa),  4),
        "kappa_opt":  round(float(best_k), 4),
        "sum_opt":    round(float(auc + best_k), 4),
        "acc":        round(float(acc),    4),
        "precision":  round(float(prec),   4),
        "recall":     round(float(rec),    4),
        "specificity":round(float(spec),   4),
        "f1":         round(float(f1),     4),
        "best_thr":   round(float(best_thr), 2),
        # at optimal threshold
        "precision_opt":   round(float(prec_opt),  4),
        "recall_opt":      round(float(rec_opt),   4),
        "specificity_opt": round(float(spec_opt),  4),
        # confusion matrix at default thr=0.5
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

EXPERIMENTS = [
    {
        "name":       "EXT",
        "label":      "FakeAVCeleb + augmented + TalkVid",
        # (db_path, dataset_filter, filename_prefix)
        "phoneme_db_specs": [(PHONEME_DB, {"FakeAVCeleb_v1.2", "augmented", "TalkVid"}, "")],
        "freq_dbs":   [FREQ_DB_EXT],
        "noise_dbs":  [NOISE_DB_EXT],
        "out_json":   "dslp_ensemble_ext.json",
        "out_log_prefix": "cv_ext",
    },
    {
        "name":       "DSV2",
        "label":      "deepspeakv2 only",
        # deepspeakv2.db has plain basenames; freq/noise DSV2 also plain basenames → direct match
        "phoneme_db_specs": [(PHONEME_DB_DSV2, None, "")],
        "freq_dbs":   [FREQ_DB_DSV2],
        "noise_dbs":  [NOISE_DB_DSV2],
        "out_json":   "dslp_ensemble_dsv2.json",
        "out_log_prefix": "cv_dsv2",
    },
    {
        "name":       "FAKEAVCELEB",
        "label":      "FakeAVCeleb v1.2 only",
        "phoneme_db_specs": [(PHONEME_DB, {"FakeAVCeleb_v1.2"}, "")],
        "freq_dbs":   [FREQ_DB_EXT],
        "noise_dbs":  [NOISE_DB_EXT],
        "out_json":   "dslp_ensemble_fakeavceleb.json",
        "out_log_prefix": "cv_fakeavceleb",
    },
    {
        "name":       "ALL",
        "label":      "All datasets",
        # DSV2 basenames prefixed with "deepspeakv2/" to form a shared namespace with EXT relpaths
        "phoneme_db_specs": [
            (PHONEME_DB, None, ""),
        ],
        "freq_dbs":   [(FREQ_DB_DSV2, "deepspeakv2/"), FREQ_DB_EXT],
        "noise_dbs":  [(NOISE_DB_DSV2, "deepspeakv2/"), NOISE_DB_EXT],
        "out_json":   "dslp_ensemble_all.json",
        "out_log_prefix": "cv_all",
    },
]


def _run_phase(phase_name, model_key, seqs, labels_arr, fold_splits,
               train_fn, pred_fn, build_model_fn, build_pipeline_fn,
               ckpt_path):
    """Run one model's N-fold CV. Returns dict {fold_i: (probs, lbls)}."""
    saved = {}
    if os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            saved = {int(k): v for k, v in json.load(f).items()}
    start = len(saved)
    print(f"\n{'─' * 60}")
    print(f"  Phase: {phase_name}  ({len(seqs)} videos, resuming fold {start+1})")
    print(f"{'─' * 60}")

    for fold_i, (train_idx, test_idx) in fold_splits:
        if fold_i <= start:
            continue
        print(f"\n  Fold {fold_i}/{len(fold_splits)}  train={len(train_idx)} test={len(test_idx)}")
        tr, te, y_tr, y_te = (
            [seqs[i] for i in train_idx],
            [seqs[i] for i in test_idx],
            labels_arr[train_idx],
            labels_arr[test_idx],
        )
        tr, te = build_pipeline_fn(tr, te)
        tr_loader, te_loader = build_model_fn(tr, te, y_tr, y_te)

        torch.manual_seed(SEED + fold_i)
        model, _, probs, lbls = train_fn(tr_loader, te_loader, y_tr, te)
        del model, tr_loader, te_loader, tr, te
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()
        if _libc is not None: _libc.malloc_trim(0)

        m = compute_metrics(probs, lbls)
        print(f"    {phase_name} fold {fold_i}: {m}")
        saved[fold_i] = {"probs": probs.tolist(), "lbls": lbls.tolist(), "metrics": m}
        with open(ckpt_path, "w") as f:
            json.dump(saved, f)
        print(f"  Checkpoint saved ({fold_i}/{len(fold_splits)} folds done)", flush=True)

    return saved


def run_cv(exp):
    """Run N_FOLDS-fold CV for one experiment (three sequential phases)."""
    print(f"\n{'#' * 70}")
    print(f"EXPERIMENT: {exp['name']} — {exp['label']}")
    print(f"{'#' * 70}")

    # ── 1. Get freq/noise filenames first (lightweight) to pre-filter phoneme load
    def _get_fnames(dbs):
        fnames = set()
        for spec in dbs:
            db, prefix = (spec if isinstance(spec, tuple) else (spec, ""))
            if not os.path.exists(db):
                continue
            conn = sqlite3.connect(db)
            rows = pd.read_sql("SELECT DISTINCT filename FROM features", conn)
            conn.close()
            fnames |= {(prefix + f) for f in rows["filename"]}
        return fnames

    print("\n[1] Getting freq/noise filename sets ...")
    freq_fnames_set  = _get_fnames(exp["freq_dbs"])
    noise_fnames_set = _get_fnames(exp["noise_dbs"])
    phoneme_prefilter = freq_fnames_set & noise_fnames_set
    print(f"  freq={len(freq_fnames_set)}  noise={len(noise_fnames_set)}  prefilter={len(phoneme_prefilter)}")

    # ── 2. Load phoneme data filtered to intersection candidates only
    print("\n[2] Loading phoneme data ...")
    phon_seqs, phon_labels, phon_fnames, feat_cols, le_phoneme = \
        load_all_phoneme_data(exp["phoneme_db_specs"], filenames_filter=phoneme_prefilter)
    num_phonemes = len(le_phoneme.classes_)

    # ── 3. Final intersection + fold splits ───────────────────────────────────
    phon_set = set(phon_fnames)
    common   = sorted(phon_set & freq_fnames_set & noise_fnames_set)
    print(f"\n[3] Intersection: {len(common)} videos "
          f"(phon={len(phon_set)} freq={len(freq_fnames_set)} noise={len(noise_fnames_set)})")
    del freq_fnames_set, noise_fnames_set, phoneme_prefilter

    phon_map     = {f: i for i, f in enumerate(phon_fnames)}
    common_seqs  = [phon_seqs[phon_map[f]] for f in common]
    common_lbls  = np.array([phon_labels[phon_map[f]] for f in common])
    del phon_seqs, phon_labels, phon_fnames, phon_map
    gc.collect()
    if _libc is not None: _libc.malloc_trim(0)

    # Drop phoneme_duration if present
    if "phoneme_duration" in feat_cols:
        keep_idx  = [i for i, c in enumerate(feat_cols) if c != "phoneme_duration"]
        feat_cols = [c for c in feat_cols if c != "phoneme_duration"]
        common_seqs = [(f[:, keep_idx], p) for f, p in common_seqs]
        print(f"  Dropped phoneme_duration")

    col_mod    = build_col_mod(feat_cols)
    visual_idx = [i for i, m in enumerate(col_mod) if m == "v"]
    audio_idx  = [i for i, m in enumerate(col_mod) if m == "a"]
    print(f"  Pipeline: {len(feat_cols)} feats -> delta(x2) -> vstats_std(x2) = {len(col_mod)}")
    print(f"  Visual: {len(visual_idx)}  Audio: {len(audio_idx)}")
    print(f"  Final: {len(common)} videos  Fake={common_lbls.sum()}  Real={(common_lbls==0).sum()}")

    skf         = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    indices     = np.arange(len(common))
    fold_splits = list(enumerate(skf.split(indices, common_lbls), 1))  # [(fold_i, (tr, te)), ...]
    common_set  = set(common)

    def _ckpt(name): return os.path.join(OUTPUT_DIR,
                                         exp["out_json"].replace(".json", f"_{name}_ckpt.json"))

    # ── Phase 1: DSLP CV ─────────────────────────────────────────────────────
    print(f"\n[4] Phase 1/3: DSLP CV ...")

    def _dslp_pipeline_correct(tr, te):
        tr = apply_delta_phoneme(tr)
        te = apply_delta_phoneme(te)
        tr, te = apply_global_scale_phoneme(tr, te)
        tr = apply_vstats_std_phoneme(tr)
        te = apply_vstats_std_phoneme(te)
        return tr, te

    def _dslp_loaders(tr, te, y_tr, y_te):
        tr_l = DataLoader(CrossModalDataset(tr, y_tr, visual_idx, audio_idx),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_crossmodal)
        te_l = DataLoader(CrossModalDataset(te, y_te, visual_idx, audio_idx),
                          batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_crossmodal)
        return tr_l, te_l

    def _dslp_train(tr_loader, te_loader, y_tr, _te):
        model = UniDSLP(len(visual_idx), len(audio_idx), num_phonemes).to(device)
        model, _ = train_model(model, train_epoch_crossmodal, predict_crossmodal,
                               tr_loader, te_loader, y_tr,
                               epochs=30, lr=0.001, patience=7, name="DSLP")
        probs, lbls = predict_crossmodal(model, te_loader)
        return model, None, probs, lbls

    dslp_saved = _run_phase("DSLP", "DSLP", common_seqs, common_lbls, fold_splits,
                             _dslp_train, None, _dslp_loaders, _dslp_pipeline_correct,
                             _ckpt("dslp"))

    del common_seqs
    gc.collect()
    if _libc is not None: _libc.malloc_trim(0)

    # ── Phase 2: TA-BiLSTM freq CV ───────────────────────────────────────────
    print(f"\n[5] Phase 2/3: TA-BiLSTM freq CV ...")
    print("  Loading freq sequences (intersection only) ...")
    freq_seqs, freq_labels, freq_fnames_loaded = load_combined_frame_data(
        *exp["freq_dbs"], filenames_filter=common_set)
    freq_map  = {f: i for i, f in enumerate(freq_fnames_loaded)}
    freq_seqs_ord  = [freq_seqs[freq_map[f]] for f in common]
    freq_lbls_ord  = np.array([freq_labels[freq_map[f]] for f in common])
    del freq_seqs, freq_labels, freq_fnames_loaded, freq_map
    gc.collect()
    if _libc is not None: _libc.malloc_trim(0)

    freq_input_dim = [None]

    def _frame_pipeline(tr, te):
        return build_frame_pipeline(tr, te)

    def _freq_loaders(tr, te, y_tr, y_te):
        freq_input_dim[0] = tr[0].shape[1]
        tr_l = DataLoader(FrameDataset(tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_frame)
        te_l = DataLoader(FrameDataset(te, y_te),
                          batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_frame)
        return tr_l, te_l

    def _freq_train(tr_loader, te_loader, y_tr, _te):
        model = TemporalAttentionLSTM(freq_input_dim[0], 128, 2, 0.3).to(device)
        model, _ = train_model(model, train_epoch_frame, predict_frame,
                               tr_loader, te_loader, y_tr,
                               epochs=30, lr=0.001, patience=7, name="TA_freq")
        probs, lbls = predict_frame(model, te_loader)
        return model, None, probs, lbls

    freq_saved = _run_phase("TA_freq", "TA_freq", freq_seqs_ord, freq_lbls_ord, fold_splits,
                             _freq_train, None, _freq_loaders, _frame_pipeline,
                             _ckpt("ta_freq"))

    del freq_seqs_ord, freq_lbls_ord
    gc.collect()
    if _libc is not None: _libc.malloc_trim(0)

    # ── Phase 3: TA-BiLSTM noise CV ─────────────────────────────────────────
    print(f"\n[6] Phase 3/3: TA-BiLSTM noise CV ...")
    print("  Loading noise sequences (intersection only) ...")
    noise_seqs, noise_labels, noise_fnames_loaded = load_combined_frame_data(
        *exp["noise_dbs"], filenames_filter=common_set)
    noise_map  = {f: i for i, f in enumerate(noise_fnames_loaded)}
    noise_seqs_ord = [noise_seqs[noise_map[f]] for f in common]
    noise_lbls_ord = np.array([noise_labels[noise_map[f]] for f in common])
    del noise_seqs, noise_labels, noise_fnames_loaded, noise_map
    gc.collect()
    if _libc is not None: _libc.malloc_trim(0)

    noise_input_dim = [None]

    def _noise_loaders(tr, te, y_tr, y_te):
        noise_input_dim[0] = tr[0].shape[1]
        tr_l = DataLoader(FrameDataset(tr, y_tr),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_frame)
        te_l = DataLoader(FrameDataset(te, y_te),
                          batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_frame)
        return tr_l, te_l

    def _noise_train(tr_loader, te_loader, y_tr, _te):
        model = TemporalAttentionLSTM(noise_input_dim[0], 128, 2, 0.3).to(device)
        model, _ = train_model(model, train_epoch_frame, predict_frame,
                               tr_loader, te_loader, y_tr,
                               epochs=30, lr=0.001, patience=7, name="TA_noise")
        probs, lbls = predict_frame(model, te_loader)
        return model, None, probs, lbls

    noise_saved = _run_phase("TA_noise", "TA_noise", noise_seqs_ord, noise_lbls_ord, fold_splits,
                              _noise_train, None, _noise_loaders, _frame_pipeline,
                              _ckpt("ta_noise"))

    del noise_seqs_ord, noise_lbls_ord
    gc.collect()
    if _libc is not None: _libc.malloc_trim(0)

    # ── Phase 4: Ensemble (load saved predictions, no model in memory) ────────
    print(f"\n[7] Assembling ensemble from saved fold predictions ...")
    fold_results   = []
    all_fold_probs = {"DSLP": [], "TA_freq": [], "TA_noise": [], "ensemble": []}
    all_fold_lbls  = []

    for fold_i, _ in fold_splits:
        d_probs = np.array(dslp_saved[fold_i]["probs"])
        f_probs = np.array(freq_saved[fold_i]["probs"])
        n_probs = np.array(noise_saved[fold_i]["probs"])
        lbls    = np.array(dslp_saved[fold_i]["lbls"])
        ens     = (d_probs + f_probs + n_probs) / 3.0

        ens_m = compute_metrics(ens, lbls)
        print(f"  Fold {fold_i}: DSLP={dslp_saved[fold_i]['metrics']['auc']:.4f} "
              f"TA_freq={freq_saved[fold_i]['metrics']['auc']:.4f} "
              f"TA_noise={noise_saved[fold_i]['metrics']['auc']:.4f} "
              f"Ensemble={ens_m['auc']:.4f}")
        fold_results.append({
            "fold":     fold_i,
            "DSLP":     dslp_saved[fold_i]["metrics"],
            "TA_freq":  freq_saved[fold_i]["metrics"],
            "TA_noise": noise_saved[fold_i]["metrics"],
            "ensemble": ens_m,
        })
        all_fold_probs["DSLP"].extend(d_probs.tolist())
        all_fold_probs["TA_freq"].extend(f_probs.tolist())
        all_fold_probs["TA_noise"].extend(n_probs.tolist())
        all_fold_probs["ensemble"].extend(ens.tolist())
        all_fold_lbls.extend(lbls.tolist())

    # ── 4. Aggregate ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Aggregate results across all folds")
    print("=" * 70)

    agg = {}
    for model_name in ["DSLP", "TA_freq", "TA_noise", "ensemble"]:
        metric_keys = fold_results[0][model_name].keys()
        mean_std = {}
        for k in metric_keys:
            vals = [fr[model_name][k] for fr in fold_results]
            mean_std[k] = {"mean": round(float(np.mean(vals)), 4),
                           "std":  round(float(np.std(vals)),  4)}
        agg[model_name] = mean_std
        print(f"\n  {model_name}:")
        for k, v in mean_std.items():
            print(f"    {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    # Pooled metrics (all fold predictions concatenated)
    print(f"\n{'─' * 60}")
    print("Pooled metrics (all folds concatenated):")
    pooled = {}
    all_lbls_arr = np.array(all_fold_lbls)
    for model_name in ["DSLP", "TA_freq", "TA_noise", "ensemble"]:
        p = np.array(all_fold_probs[model_name])
        m = compute_metrics(p, all_lbls_arr)
        pooled[model_name] = m
        print(f"  {model_name}: {m}")

    # ── 5. Print pooled confusion matrices ────────────────────────────────────
    print(f"\n{'─' * 60}")
    print("Pooled confusion matrices (Real=0, Fake=1) at thr=best_thr:")
    all_lbls_arr = np.array(all_fold_lbls)
    for model_name in ["DSLP", "TA_freq", "TA_noise", "ensemble"]:
        p    = np.array(all_fold_probs[model_name])
        thr  = pooled[model_name]["best_thr"]
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(all_lbls_arr, pred, labels=[0, 1]).ravel()
        print(f"\n  {model_name} (thr={thr}):")
        print(f"    Predicted:   Real    Fake")
        print(f"    Real (act):  TN={tn:<6} FP={fp}")
        print(f"    Fake (act):  FN={fn:<6} TP={tp}")

    # ── 6. Save ───────────────────────────────────────────────────────────────
    output = {
        "experiment":   exp["name"],
        "label":        exp["label"],
        "n_folds":      N_FOLDS,
        "n_videos":     len(common),
        "label_dist":   {"Fake": int(common_lbls.sum()),
                         "Real": int((common_lbls == 0).sum())},
        "fold_results": fold_results,
        "aggregate":    agg,
        "pooled":       pooled,
    }
    out_path = os.path.join(OUTPUT_DIR, exp["out_json"])
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return output


def main():
    all_results = {}
    for exp in EXPERIMENTS:
        out_path = os.path.join(OUTPUT_DIR, exp["out_json"])
        if os.path.exists(out_path):
            with open(out_path, encoding="utf-8") as f:
                cached = json.load(f)
            if cached.get("fold_results") and len(cached["fold_results"]) == N_FOLDS:
                print(f"\n[SKIP] {exp['name']} — already complete ({N_FOLDS} folds found)")
                all_results[exp["name"]] = cached
                continue
        result = run_cv(exp)
        all_results[exp["name"]] = result

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY — Ensemble pooled metrics across all experiments")
    print("=" * 70)
    header = f"{'Experiment':<30} {'AUC':>7} {'Kappa_opt':>10} {'Sum_opt':>9} {'Acc':>7} {'Prec':>7} {'Recall':>8} {'Spec':>7} {'F1':>7}"
    print(header)
    print("─" * len(header))
    for name, res in all_results.items():
        m = res["pooled"]["ensemble"]
        print(f"  {name:<28} {m['auc']:>7.4f} {m['kappa_opt']:>10.4f} {m['sum_opt']:>9.4f} "
              f"{m['acc']:>7.4f} {m['precision_opt']:>7.4f} {m['recall_opt']:>8.4f} "
              f"{m['specificity_opt']:>7.4f} {m['f1']:>7.4f}")

    combined_path = os.path.join(OUTPUT_DIR, "cv_all_experiments.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {combined_path}")


if __name__ == "__main__":
    main()
