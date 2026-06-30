"""
Train and persist the three FINAL ensemble components (DSLP, TAFreq, TANoise) on the
full EXT dataset, saving everything needed for stand-alone inference (weights, fitted
StandardScalers, phoneme label encoder, feature-column metadata, input dims).

Reuses the exact data loaders, feature pipelines, model classes and training loop of
modelling/cv_ensemble.py, so the saved models match the cross-validated configuration.

Usage:
    python -u modelling/train_and_save_final.py            # EXT (default)
    python -u modelling/train_and_save_final.py --dataset DSV2

Artifacts are written to trained/final/.
"""
import os, sys, gc, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from modelling import cv_ensemble as cve
from modelling.dslp_arch import UniDSLP

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trained", "final")
os.makedirs(OUT_DIR, exist_ok=True)
device = cve.device
VAL_FRAC = 0.10


def _split(seqs, labels):
    idx = np.arange(len(seqs))
    tr, va = train_test_split(idx, test_size=VAL_FRAC, stratify=labels, random_state=cve.SEED)
    return ([seqs[i] for i in tr], labels[tr]), ([seqs[i] for i in va], labels[va])


def train_dslp(common_seqs, common_lbls, feat_cols, num_phonemes):
    print("\n=== DSLP ===")
    col_mod    = cve.build_col_mod(feat_cols)
    visual_idx = [i for i, m in enumerate(col_mod) if m == "v"]
    audio_idx  = [i for i, m in enumerate(col_mod) if m == "a"]
    (tr_s, y_tr), (va_s, y_va) = _split(common_seqs, common_lbls)

    # pipeline: delta -> global z-score (fit on train) -> vstats_std
    tr_s = cve.apply_delta_phoneme(tr_s); va_s = cve.apply_delta_phoneme(va_s)
    scaler = StandardScaler()
    for f, _ in tr_s:
        scaler.partial_fit(f)
    tr_s = [(scaler.transform(f).astype(np.float32), p) for f, p in tr_s]
    va_s = [(scaler.transform(f).astype(np.float32), p) for f, p in va_s]
    tr_s = cve.apply_vstats_std_phoneme(tr_s); va_s = cve.apply_vstats_std_phoneme(va_s)

    tr_l = DataLoader(cve.CrossModalDataset(tr_s, y_tr, visual_idx, audio_idx),
                      batch_size=cve.BATCH_SIZE, shuffle=True, collate_fn=cve.collate_fn_crossmodal)
    va_l = DataLoader(cve.CrossModalDataset(va_s, y_va, visual_idx, audio_idx),
                      batch_size=cve.BATCH_SIZE, shuffle=False, collate_fn=cve.collate_fn_crossmodal)

    model = UniDSLP(len(visual_idx), len(audio_idx), num_phonemes).to(device)
    model, auc = cve.train_model(model, cve.train_epoch_crossmodal, cve.predict_crossmodal,
                                 tr_l, va_l, y_tr, epochs=30, lr=0.001, patience=7, name="DSLP")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "dslp.pth"))
    joblib.dump({"scaler": scaler, "feat_cols": feat_cols, "visual_idx": visual_idx,
                 "audio_idx": audio_idx, "num_phonemes": num_phonemes,
                 "phoneme_classes": list(LE_CLASSES)},
                os.path.join(OUT_DIR, "dslp_artifacts.pkl"))
    print(f"  saved dslp.pth  (val AUC {auc:.4f})")


def train_frame(seqs, labels, name):
    print(f"\n=== {name} ===")
    (tr_s, y_tr), (va_s, y_va) = _split(seqs, labels)
    scaler, col_means = cve.fit_global_scaler_frame(tr_s)
    tr_s = cve.apply_global_scale_frame(tr_s, scaler, col_means)
    va_s = cve.apply_global_scale_frame(va_s, scaler, col_means)
    tr_s = cve.apply_delta_frame(tr_s); va_s = cve.apply_delta_frame(va_s)
    tr_s = cve.apply_vstats_full_frame(tr_s); va_s = cve.apply_vstats_full_frame(va_s)
    input_dim = tr_s[0].shape[1]

    tr_l = DataLoader(cve.FrameDataset(tr_s, y_tr), batch_size=cve.BATCH_SIZE, shuffle=True,
                      collate_fn=cve.collate_fn_frame)
    va_l = DataLoader(cve.FrameDataset(va_s, y_va), batch_size=cve.BATCH_SIZE, shuffle=False,
                      collate_fn=cve.collate_fn_frame)
    model = cve.TemporalAttentionLSTM(input_dim, 128, 2, 0.3).to(device)
    model, auc = cve.train_model(model, cve.train_epoch_frame, cve.predict_frame,
                                 tr_l, va_l, y_tr, epochs=30, lr=0.001, patience=7, name=name)
    key = name.lower()
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{key}.pth"))
    joblib.dump({"scaler": scaler, "col_means": col_means, "input_dim": input_dim},
                os.path.join(OUT_DIR, f"{key}_artifacts.pkl"))
    print(f"  saved {key}.pth  (val AUC {auc:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ALL", choices=["EXT", "DSV2", "FAKEAVCELEB", "ALL"])
    args = ap.parse_args()
    exp = next(e for e in cve.EXPERIMENTS if e["name"] == args.dataset)

    # filename intersection across the three feature sources
    fseqs, flab, ffn = cve.load_combined_frame_data(*exp["freq_dbs"])
    nseqs, nlab, nfn = cve.load_combined_frame_data(*exp["noise_dbs"])
    freq_set, noise_set = set(ffn), set(nfn)
    pre = freq_set & noise_set

    phon_seqs, phon_lbls, phon_fn, feat_cols, le = \
        cve.load_all_phoneme_data(exp["phoneme_db_specs"], filenames_filter=pre)
    global LE_CLASSES; LE_CLASSES = le.classes_
    num_phonemes = len(le.classes_)

    common = sorted(set(phon_fn) & freq_set & noise_set)
    pmap = {f: i for i, f in enumerate(phon_fn)}
    common_seqs = [phon_seqs[pmap[f]] for f in common]
    common_lbls = np.array([phon_lbls[pmap[f]] for f in common])
    if "phoneme_duration" in feat_cols:
        keep = [i for i, c in enumerate(feat_cols) if c != "phoneme_duration"]
        feat_cols = [c for c in feat_cols if c != "phoneme_duration"]
        common_seqs = [(f[:, keep], p) for f, p in common_seqs]
    print(f"[data] {len(common)} videos  Fake={int(common_lbls.sum())} Real={int((common_lbls==0).sum())}")

    train_dslp(common_seqs, common_lbls, feat_cols, num_phonemes)
    del common_seqs, phon_seqs; gc.collect()

    fmap = {f: i for i, f in enumerate(ffn)}
    train_frame([fseqs[fmap[f]] for f in common],
                np.array([flab[fmap[f]] for f in common]), "TAFreq")
    del fseqs; gc.collect()

    nmap = {f: i for i, f in enumerate(nfn)}
    train_frame([nseqs[nmap[f]] for f in common],
                np.array([nlab[nmap[f]] for f in common]), "TANoise")

    json.dump({"dataset": args.dataset, "n_videos": len(common), "num_phonemes": num_phonemes,
               "components": ["dslp", "tafreq", "tanoise"], "ensemble": "unweighted mean of sigmoids"},
              open(os.path.join(OUT_DIR, "ensemble_manifest.json"), "w"), indent=2)
    print(f"\nDone. Artifacts in {OUT_DIR}")


if __name__ == "__main__":
    main()
