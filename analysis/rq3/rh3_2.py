"""
RH3.2: Bidirectional LSTMs will significantly outperform unidirectional LSTMs
       because they can capture both past and future phoneme context.

Approach: train both architectures from scratch under identical conditions
  (same data split, same hyperparameters, same 100 raw features) and compare
  AUC values across N_SEEDS seeds.

Feature pipeline: 99 numeric features from dataset.db + phoneme identity as a
  normalised scalar (phoneme_enc / (num_phonemes - 1)) = 100 features total.
  Only a StandardScaler is applied (no AV-interaction, no delta, no vstats).

Statistical test: one-sided Mann-Whitney U (BiLSTM AUC > UniLSTM AUC).
Cohen's d as effect size.

References:
  Schuster & Paliwal (1997) - Bidirectional Recurrent Neural Networks
  Hochreiter & Schmidhuber (1997) - LSTM
  Mann & Whitney (1947)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, ".")

import sqlite3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, cohen_kappa_score
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modelling.experiments_deepspeakv2 import device, BATCH_SIZE, DEFAULT_EPOCHS, LEARNING_RATE, SEED
from analysis.rq_utils import apply_style, save_plot, DB_PATH, FEATURE_COLS

apply_style()

N_SEEDS  = 5
HIDDEN   = 128
N_LAYERS = 2
DROPOUT  = 0.3
NUM_FEATURES = 100   # 99 numeric + phoneme_norm scalar


# ── Models ────────────────────────────────────────────────────────────────────

class UniLSTM(nn.Module):
    """Unidirectional LSTM — forward context only."""
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=HIDDEN,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            num_features, hidden_dim, num_layers,
            batch_first=True, bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, feats, lengths):
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.fc(hidden[-1]).squeeze(1)


class BiLSTM(nn.Module):
    """Bidirectional LSTM — forward + backward context."""
    def __init__(self, num_features=NUM_FEATURES, hidden_dim=HIDDEN,
                 num_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            num_features, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, feats, lengths):
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        final = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(final).squeeze(1)


# ── Dataset / collate ─────────────────────────────────────────────────────────

class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs   = sequences    # list of np.ndarray (T, F)
        self.labels = labels       # np.ndarray (N,)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return torch.tensor(self.seqs[idx], dtype=torch.float32), \
               torch.tensor(self.labels[idx], dtype=torch.float32)


def collate_fn(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)   # (B, T_max, F)
    return padded, lengths, torch.stack(labels)


# ── Training helper ───────────────────────────────────────────────────────────

def evaluate_all(model, loader):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for feats, lengths, labels_b in loader:
            feats = feats.to(device)
            probs = torch.sigmoid(model(feats, lengths)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels_b.numpy())
    all_labels = np.array(all_labels)
    all_preds  = (np.array(all_probs) >= 0.5).astype(int)
    return {
        "auc":   float(roc_auc_score(all_labels, all_probs)),
        "acc":   float(accuracy_score(all_labels, all_preds)),
        "f1":    float(f1_score(all_labels, all_preds)),
        "kappa": float(cohen_kappa_score(all_labels, all_preds)),
    }


def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    train_ds = SeqDataset(X_train, y_train)
    test_ds  = SeqDataset(X_test,  y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn)

    num_neg   = int((y_train == 0).sum())
    num_pos   = int((y_train == 1).sum())
    pw        = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    for epoch in range(DEFAULT_EPOCHS):
        model.train()
        for feats, lengths, labels_b in train_loader:
            feats    = feats.to(device)
            labels_b = labels_b.to(device)
            optimizer.zero_grad()
            out  = model(feats, lengths)
            loss = criterion(out, labels_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    return evaluate_all(model, test_loader)


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na-1)*np.std(a, ddof=1)**2 + (nb-1)*np.std(b, ddof=1)**2)
                     / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else np.nan


def run():
    print("=" * 65)
    print("RH3.2  BiLSTM vs UniLSTM (bare, 100 raw features)")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading data from {DB_PATH} ...", flush=True)
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

    # Phoneme as normalised scalar (feature 100)
    le_phoneme = LabelEncoder()
    df["phoneme_enc"] = le_phoneme.fit_transform(df["phoneme"])
    num_phonemes = df["phoneme_enc"].nunique()
    df["phoneme_norm"] = df["phoneme_enc"] / (num_phonemes - 1)

    input_cols = FEATURE_COLS + ["phoneme_norm"]   # 100 columns

    # Build per-video sequences
    sequences, labels = [], []
    for _, grp in df.groupby("filename", sort=False):
        feats = grp[input_cols].values.astype(np.float32)
        sequences.append(feats)
        labels.append(1 if grp["video_type"].iloc[0] == "Fake" else 0)
    labels = np.array(labels)

    print(f"  {len(sequences):,} videos  "
          f"Fake={labels.sum():,}  Real={(labels==0).sum():,}  "
          f"Phonemes={num_phonemes}  Features={len(input_cols)}")

    # 80/20 train/test split (video-level, stratified)
    idx = np.arange(len(sequences))
    idx_tr, idx_te = train_test_split(idx, test_size=0.2,
                                       random_state=SEED, stratify=labels)

    X_tr_raw = [sequences[i] for i in idx_tr]
    X_te_raw = [sequences[i] for i in idx_te]
    y_train  = labels[idx_tr]
    y_test   = labels[idx_te]

    print(f"  Train: {len(X_tr_raw)}  Test: {len(X_te_raw)}", flush=True)

    # StandardScaler — fit on train rows only
    print("  Fitting StandardScaler on training data ...", flush=True)
    scaler     = StandardScaler()
    CHUNK      = 50_000
    train_rows = np.vstack(X_tr_raw)
    for start in range(0, len(train_rows), CHUNK):
        scaler.partial_fit(train_rows[start:start + CHUNK])
    del train_rows

    X_tr = [scaler.transform(s) for s in X_tr_raw]
    X_te = [scaler.transform(s) for s in X_te_raw]

    # ── Train N_SEEDS of each architecture ────────────────────────────────────
    METRICS = ["auc", "acc", "f1", "kappa"]
    bi_results  = {m: [] for m in METRICS}
    uni_results = {m: [] for m in METRICS}

    for seed in range(N_SEEDS):
        torch.manual_seed(SEED + seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + seed)

        print(f"\n  Seed {seed+1}/{N_SEEDS} — BiLSTM ...", flush=True)
        bi_model = BiLSTM().to(device)
        bi_m     = train_and_evaluate(bi_model, X_tr, X_te, y_train, y_test)
        for m in METRICS:
            bi_results[m].append(bi_m[m])
        print(f"    BiLSTM  AUC={bi_m['auc']:.4f}  Acc={bi_m['acc']:.4f}  "
              f"F1={bi_m['f1']:.4f}  Kappa={bi_m['kappa']:.4f}")

        torch.manual_seed(SEED + seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED + seed)

        print(f"  Seed {seed+1}/{N_SEEDS} — UniLSTM ...", flush=True)
        uni_model = UniLSTM().to(device)
        uni_m     = train_and_evaluate(uni_model, X_tr, X_te, y_train, y_test)
        for m in METRICS:
            uni_results[m].append(uni_m[m])
        print(f"    UniLSTM AUC={uni_m['auc']:.4f}  Acc={uni_m['acc']:.4f}  "
              f"F1={uni_m['f1']:.4f}  Kappa={uni_m['kappa']:.4f}")

    bi_results  = {m: np.array(v) for m, v in bi_results.items()}
    uni_results = {m: np.array(v) for m, v in uni_results.items()}

    print("\n" + "-" * 65)
    mwu_results = {}
    for m in METRICS:
        stat, p = mannwhitneyu(bi_results[m], uni_results[m], alternative="greater")
        d       = cohen_d(bi_results[m], uni_results[m])
        mwu_results[m] = {"U": float(stat), "p": float(p), "d": float(d)}
        verdict_m = "yes" if p < 0.05 else "no"
        print(f"{m.upper():6s}  BiLSTM={bi_results[m].mean():.4f}+/-{bi_results[m].std():.4f}"
              f"  UniLSTM={uni_results[m].mean():.4f}+/-{uni_results[m].std():.4f}"
              f"  U={stat:.0f}  p={p:.2e}  d={d:.3f}  sig={verdict_m}")

    # overall verdict: supported if AUC is significant (primary metric)
    verdict = "SUPPORTED" if mwu_results["auc"]["p"] < 0.05 else "NOT supported"
    print(f"\n  RH3.2 {verdict} at alpha=0.05 (primary metric: AUC)")

    # ── Plot (AUC) ────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    rng = np.random.RandomState(SEED)
    jit = rng.uniform(-0.08, 0.08, N_SEEDS)
    ax.scatter(np.zeros(N_SEEDS) + jit, bi_results["auc"],
               color="#0072B2", alpha=0.85, s=60, zorder=3)
    ax.scatter(np.ones(N_SEEDS) + jit, uni_results["auc"],
               color="#E69F00", alpha=0.85, s=60, zorder=3)
    ax.boxplot([bi_results["auc"], uni_results["auc"]], positions=[0, 1], widths=0.3,
               patch_artist=True,
               boxprops=dict(alpha=0.3),
               medianprops=dict(color="black", lw=2))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BiLSTM\n(bidirectional)", "UniLSTM\n(unidirectional)"])
    ax.set_ylabel("AUC")
    p_auc = mwu_results["auc"]["p"]
    d_auc = mwu_results["auc"]["d"]
    ax.set_title(f"RH3.2 — Bidirectional vs Unidirectional LSTM\n"
                 f"Mann-Whitney p={p_auc:.2e}  |  Cohen's d={d_auc:.3f}")
    fig.tight_layout()
    save_plot(fig, "rh3_2_bilstm_vs_unilstm.png")

    # ── Save results ──────────────────────────────────────────────────────────
    import json, os
    out = {
        "bi":  {m: bi_results[m].tolist()  for m in METRICS},
        "uni": {m: uni_results[m].tolist() for m in METRICS},
        "mwu": mwu_results,
        "verdict": verdict,
    }
    os.makedirs(".tmp", exist_ok=True)
    with open(".tmp/rh3_2_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nResults saved to .tmp/rh3_2_results.json")


if __name__ == "__main__":
    run()
