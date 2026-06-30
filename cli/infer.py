"""
Inference engine for the final detector. Extracts per-video features for each stream,
applies the saved pipelines, and runs the trained models.

Reuses the project's own extraction + pipeline code so features match training:
  - DSLP   : processing.py (visual landmarks + MFCC + phoneme alignment)
  - TAFreq : extract_freq_features._process_video_core -> 36-d frequency features
  - TANoise: extract_noise_features._process_video_core -> 19-d noise-residual features

Loads weights/scalers from trained/final/ (produced by modelling/train_and_save_final.py).
"""
import os, sys
import numpy as np
import torch
import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "feat-extraction"))

from modelling import cv_ensemble as cve
from modelling.dslp_arch import UniDSLP

FINAL_DIR = os.path.join(ROOT, "trained", "final")
device = cve.device


# -- feature extraction --------------------------------------------------------
def extract_dslp(video_path):
    """Return (raw_feature_df, phoneme_list) for the DSLP stream, or (None, None)."""
    import processing
    from extract_features import BLENDSHAPE_NAMES, to_snake_case
    vis = processing.extract_visual_lip_data(video_path, processing.face_landmarker)
    lip_d, ts, blends, mars, areas, widths, curvs, circs = vis
    if not lip_d:
        return None, None
    phonemes = processing.extract_phonemes(video_path)
    if not phonemes:
        return None, None
    mfcc_feats, mfcc_ts = processing.extract_mfcc(video_path)
    feats = processing.align_features(lip_d, ts, phonemes, blends, mars, areas, widths,
                                      curvs, circs, mfcc_features=mfcc_feats, mfcc_timestamps=mfcc_ts)
    if not feats:
        return None, None
    df = pd.DataFrame(feats)
    for col in [to_snake_case(n) for n in BLENDSHAPE_NAMES]:
        if col not in df.columns:
            df[col] = 0.0
    return df, df["phoneme"].tolist() if "phoneme" in df.columns else None


def extract_freq_seq(video_path):
    """Per-video 36-d frequency features (matches data/freq_features.db)."""
    import extract_freq_features as eff
    rows = eff._process_video_core(video_path, "cli", "Fake")
    if not rows:
        return None
    return np.array([r[4:] for r in rows], dtype=np.float32)


def extract_noise_seq(video_path):
    """Per-video 19-d noise-residual features (matches data/noise_features.db)."""
    import extract_noise_features as enf
    rows = enf._process_video_core(video_path, "cli", "Fake")
    if not rows:
        return None
    return np.array([r[4:] for r in rows], dtype=np.float32)


# -- per-stream prediction ------------------------------------------------------
def predict_dslp(video_path):
    art = joblib.load(os.path.join(FINAL_DIR, "dslp_artifacts.pkl"))
    df, phonemes = extract_dslp(video_path)
    if df is None:
        raise RuntimeError("DSLP feature extraction failed (no face/speech).")
    feat_cols = art["feat_cols"]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0
    X = df[feat_cols].to_numpy(np.float32)
    classes = list(art["phoneme_classes"]); cmap = {c: i for i, c in enumerate(classes)}
    phon_ids = np.array([cmap.get(p, 0) for p in phonemes], dtype=np.int64)

    seq = [(X, phon_ids)]
    seq = cve.apply_delta_phoneme(seq)
    seq = [(art["scaler"].transform(f).astype(np.float32), p) for f, p in seq]
    seq = cve.apply_vstats_std_phoneme(seq)

    from torch.utils.data import DataLoader
    dl = DataLoader(cve.CrossModalDataset(seq, np.array([0]), art["visual_idx"], art["audio_idx"]),
                    batch_size=1, shuffle=False, collate_fn=cve.collate_fn_crossmodal)
    model = UniDSLP(len(art["visual_idx"]), len(art["audio_idx"]), art["num_phonemes"]).to(device)
    model.load_state_dict(torch.load(os.path.join(FINAL_DIR, "dslp.pth"), map_location=device))
    probs, _ = cve.predict_crossmodal(model, dl)
    return float(probs[0])


def _predict_frame(video_path, which):  # which in {"tafreq","tanoise"}
    art = joblib.load(os.path.join(FINAL_DIR, f"{which}_artifacts.pkl"))
    X = extract_freq_seq(video_path) if which == "tafreq" else extract_noise_seq(video_path)
    if X is None:
        raise RuntimeError(f"{which} feature extraction failed.")
    seq = [X]
    seq = cve.apply_global_scale_frame(seq, art["scaler"], art["col_means"])
    seq = cve.apply_delta_frame(seq)
    seq = cve.apply_vstats_full_frame(seq)

    from torch.utils.data import DataLoader
    dl = DataLoader(cve.FrameDataset(seq, np.array([0])),
                    batch_size=1, shuffle=False, collate_fn=cve.collate_fn_frame)
    model = cve.TemporalAttentionLSTM(art["input_dim"], 128, 2, 0.3).to(device)
    model.load_state_dict(torch.load(os.path.join(FINAL_DIR, f"{which}.pth"), map_location=device))
    probs, _ = cve.predict_frame(model, dl)
    return float(probs[0])


def predict_tafreq(video_path):  return _predict_frame(video_path, "tafreq")
def predict_tanoise(video_path): return _predict_frame(video_path, "tanoise")


def predict_ensemble(video_path):
    p = {"dslp": predict_dslp(video_path),
         "tafreq": predict_tafreq(video_path),
         "tanoise": predict_tanoise(video_path)}
    p["ensemble"] = float(np.mean([p["dslp"], p["tafreq"], p["tanoise"]]))
    return p


def report(name, prob):
    verdict = "FAKE" if prob >= 0.5 else "REAL"
    print(f"{name}: P(fake) = {prob:.4f}  ->  {verdict}")
