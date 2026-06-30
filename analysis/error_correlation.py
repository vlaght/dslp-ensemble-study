"""
Error correlation analysis for the three-model ensemble.

Trains DSMILP, TA_freq, TA_noise once on 80% of DSV2, predicts on the
held-out 20%, then tests pairwise error independence using:
  - McNemar's test (exact binomial, N<25 in discordant cells; chi-squared otherwise)
  - Yule's Q statistic (pairwise error agreement)
  - Pearson r on binary error vectors

McNemar's test null hypothesis: the two classifiers make errors independently
(i.e. P(A wrong, B right) == P(A right, B wrong)).  A significant result
(p < 0.05) means their error patterns differ systematically -- which is
what we WANT for a good ensemble (models fail on different samples).

Usage:
    cd t:\\thesis
    python.exe -u analysis/error_correlation.py > .tmp/error_correlation.log 2>&1
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, cohen_kappa_score
from scipy.stats import pearsonr, binomtest
from scipy.stats import chi2

from modelling.cv_ensemble import (
    # constants
    SEED, device, BATCH_SIZE,
    PHONEME_DB_DSV2, FREQ_DB_DSV2, NOISE_DB_DSV2,
    # models
    DualStreamMILPhoneme, TemporalAttentionLSTM,
    # data loading
    load_all_phoneme_data, load_combined_frame_data,
    # transforms
    apply_av_interaction, apply_delta_phoneme,
    apply_global_scale_phoneme, apply_vstats_full_phoneme,
    build_frame_pipeline, build_col_mod,
    # datasets & collate
    CrossModalDataset, FrameDataset,
    collate_fn_crossmodal, collate_fn_frame,
    # training & prediction
    train_epoch_crossmodal, predict_crossmodal,
    train_epoch_frame, predict_frame, train_model,
)

torch.manual_seed(SEED)
np.random.seed(SEED)


# ── Statistics ────────────────────────────────────────────────────────────────

def mcnemar_test(errors_a, errors_b):
    """McNemar's test on binary error vectors (1=wrong, 0=correct).
    Returns (statistic, p_value, b, c)."""
    b = int(((errors_a == 1) & (errors_b == 0)).sum())
    c = int(((errors_a == 0) & (errors_b == 1)).sum())
    n = b + c
    if n == 0:
        return 0.0, 1.0, b, c
    if n < 25:
        p = 2 * min(binomtest(b, n, 0.5), 1.0)
        stat = float(b)
    else:
        stat = (abs(b - c) - 1) ** 2 / (b + c)
        p = 1 - chi2.cdf(stat, df=1)
    return stat, p, b, c


def yule_q(errors_a, errors_b):
    """Yule's Q: 0=independent, >0=correlated errors, <0=complementary (ideal)."""
    a = int(((errors_a == 1) & (errors_b == 1)).sum())
    b = int(((errors_a == 1) & (errors_b == 0)).sum())
    c = int(((errors_a == 0) & (errors_b == 1)).sum())
    d = int(((errors_a == 0) & (errors_b == 0)).sum())
    denom = a * d + b * c
    return float("nan") if denom == 0 else (a * d - b * c) / denom


def best_threshold(probs, labels):
    best_k, best_t = -1.0, 0.5
    for t in np.arange(0.1, 0.91, 0.05):
        k = cohen_kappa_score(labels, (probs >= t).astype(int))
        if k > best_k:
            best_k, best_t = k, t
    return best_t


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("Loading phoneme data ...")
    phon_seqs, phon_labels, phon_fnames, feat_cols, le_phoneme = \
        load_all_phoneme_data([(PHONEME_DB_DSV2, None, "")])
    num_phonemes = len(le_phoneme.classes_)

    print("\nLoading freq features ...")
    freq_seqs, freq_labels, freq_fnames = load_combined_frame_data(FREQ_DB_DSV2)

    print("\nLoading noise features ...")
    noise_seqs, noise_labels, noise_fnames = load_combined_frame_data(NOISE_DB_DSV2)

    # ── 2. Intersection ───────────────────────────────────────────────────────
    common = sorted(set(phon_fnames) & set(freq_fnames) & set(noise_fnames))
    print(f"\nIntersection: {len(common)} videos")
    phon_map  = {f: i for i, f in enumerate(phon_fnames)}
    freq_map  = {f: i for i, f in enumerate(freq_fnames)}
    noise_map = {f: i for i, f in enumerate(noise_fnames)}

    c_phon  = [phon_seqs[phon_map[f]]   for f in common]
    c_freq  = [freq_seqs[freq_map[f]]   for f in common]
    c_noise = [noise_seqs[noise_map[f]] for f in common]
    c_labels = np.array([phon_labels[phon_map[f]] for f in common])
    print(f"Fake={c_labels.sum()}  Real={(c_labels==0).sum()}")

    # ── 3. Train/test split (80/20, stratified) ───────────────────────────────
    idx = np.arange(len(common))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, stratify=c_labels,
                                       random_state=SEED)
    y_train = c_labels[tr_idx]
    y_test  = c_labels[te_idx]
    print(f"Train: {len(tr_idx)}  Test: {len(te_idx)}")

    # ── 4. Phoneme pipeline ───────────────────────────────────────────────────
    tr_phon = [c_phon[i] for i in tr_idx]
    te_phon = [c_phon[i] for i in te_idx]
    tr_phon, n_av = apply_av_interaction(tr_phon, feat_cols)
    te_phon, _    = apply_av_interaction(te_phon, feat_cols)
    tr_phon = apply_delta_phoneme(tr_phon)
    te_phon = apply_delta_phoneme(te_phon)
    tr_phon, te_phon = apply_global_scale_phoneme(tr_phon, te_phon)
    tr_phon = apply_vstats_full_phoneme(tr_phon)
    te_phon = apply_vstats_full_phoneme(te_phon)

    col_mod   = build_col_mod(feat_cols, n_av)
    vis_idx   = [i for i, m in enumerate(col_mod) if m == "v"]
    aud_idx   = [i for i, m in enumerate(col_mod) if m == "a"]
    print(f"Phoneme: vis={len(vis_idx)} aud={len(aud_idx)} features")

    phon_tr_loader = DataLoader(
        CrossModalDataset(tr_phon, y_train, vis_idx, aud_idx),
        batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_crossmodal)
    phon_te_loader = DataLoader(
        CrossModalDataset(te_phon, y_test,  vis_idx, aud_idx),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_crossmodal)

    # ── 5. Frame pipelines ────────────────────────────────────────────────────
    tr_freq,  te_freq  = build_frame_pipeline(
        [c_freq[i]  for i in tr_idx], [c_freq[i]  for i in te_idx])
    tr_noise, te_noise = build_frame_pipeline(
        [c_noise[i] for i in tr_idx], [c_noise[i] for i in te_idx])

    freq_tr_loader  = DataLoader(FrameDataset(tr_freq,  y_train), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_frame)
    freq_te_loader  = DataLoader(FrameDataset(te_freq,  y_test),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_frame)
    noise_tr_loader = DataLoader(FrameDataset(tr_noise, y_train), batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn_frame)
    noise_te_loader = DataLoader(FrameDataset(te_noise, y_test),  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_frame)

    # ── 6. Train ──────────────────────────────────────────────────────────────
    print("\n── Training DSMILP ──")
    torch.manual_seed(SEED)
    dsmilp = DualStreamMILPhoneme(
        num_visual=len(vis_idx), num_audio=len(aud_idx),
        num_phonemes=num_phonemes, embedding_dim=8,
        hidden_dim=256, num_layers=2, dropout=0.3,
    ).to(device)
    dsmilp, _ = train_model(dsmilp, train_epoch_crossmodal, predict_crossmodal,
                             phon_tr_loader, phon_te_loader, y_train,
                             epochs=30, lr=0.001, patience=7, name="DSMILP")
    dsmilp_probs, _ = predict_crossmodal(dsmilp, phon_te_loader)
    del dsmilp; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n── Training TA_freq ──")
    torch.manual_seed(SEED)
    ta_freq = TemporalAttentionLSTM(input_dim=tr_freq[0].shape[1],
                                    hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    ta_freq, _ = train_model(ta_freq, train_epoch_frame, predict_frame,
                              freq_tr_loader, freq_te_loader, y_train,
                              epochs=30, lr=0.001, patience=7, name="TA_freq")
    freq_probs, _ = predict_frame(ta_freq, freq_te_loader)
    del ta_freq; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print("\n── Training TA_noise ──")
    torch.manual_seed(SEED)
    ta_noise = TemporalAttentionLSTM(input_dim=tr_noise[0].shape[1],
                                     hidden_dim=128, num_layers=2, dropout=0.3).to(device)
    ta_noise, _ = train_model(ta_noise, train_epoch_frame, predict_frame,
                               noise_tr_loader, noise_te_loader, y_train,
                               epochs=30, lr=0.001, patience=7, name="TA_noise")
    noise_probs, _ = predict_frame(ta_noise, noise_te_loader)
    del ta_noise; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── 7. Error vectors ──────────────────────────────────────────────────────
    thr_d = best_threshold(dsmilp_probs, y_test)
    thr_f = best_threshold(freq_probs,   y_test)
    thr_n = best_threshold(noise_probs,  y_test)

    err_d = ((dsmilp_probs >= thr_d).astype(int) != y_test).astype(int)
    err_f = ((freq_probs   >= thr_f).astype(int) != y_test).astype(int)
    err_n = ((noise_probs  >= thr_n).astype(int) != y_test).astype(int)

    print(f"\n── Error rates ──")
    print(f"  DSMILP  (thr={thr_d:.2f}): {err_d.sum()} / {len(y_test)} ({100*err_d.mean():.1f}%)")
    print(f"  TA_freq (thr={thr_f:.2f}): {err_f.sum()} / {len(y_test)} ({100*err_f.mean():.1f}%)")
    print(f"  TA_noise(thr={thr_n:.2f}): {err_n.sum()} / {len(y_test)} ({100*err_n.mean():.1f}%)")

    # ── 8. Pairwise statistics ────────────────────────────────────────────────
    pairs = [
        ("DSMILP",   "TA_freq",  err_d, err_f),
        ("DSMILP",   "TA_noise", err_d, err_n),
        ("TA_freq",  "TA_noise", err_f, err_n),
    ]

    print("\n── Pearson r (error vectors) ──")
    print(f"  {'Pair':<25}  r       p-value    note")
    for na, nb, ea, eb in pairs:
        r, p = pearsonr(ea, eb)
        note = "correlated" if p < 0.05 and r > 0 else \
               "anti-correlated" if p < 0.05 and r < 0 else "independent"
        print(f"  {na+' vs '+nb:<25}  {r:+.4f}  {p:.4e}  {note}")

    print("\n── McNemar's test ──")
    print(f"  H0: errors are independent between models")
    print(f"  {'Pair':<25}  stat      p-value    b(A✗B✓)  c(A✓B✗)  result")
    for na, nb, ea, eb in pairs:
        stat, p, b, c = mcnemar_test(ea, eb)
        result = "REJECT H0" if p < 0.05 else "fail to reject H0"
        print(f"  {na+' vs '+nb:<25}  {stat:8.3f}  {p:.4e}  {b:7d}  {c:7d}  {result}")

    print("\n── Yule's Q (Q=0: independent | Q<0: complementary) ──")
    for na, nb, ea, eb in pairs:
        q = yule_q(ea, eb)
        print(f"  {na+' vs '+nb:<25}  Q={q:+.4f}")

    print("\n── AUC ──")
    print(f"  DSMILP  : {roc_auc_score(y_test, dsmilp_probs):.4f}")
    print(f"  TA_freq : {roc_auc_score(y_test, freq_probs):.4f}")
    print(f"  TA_noise: {roc_auc_score(y_test, noise_probs):.4f}")
    ens = (dsmilp_probs + freq_probs + noise_probs) / 3.0
    print(f"  Ensemble: {roc_auc_score(y_test, ens):.4f}")


if __name__ == "__main__":
    main()
