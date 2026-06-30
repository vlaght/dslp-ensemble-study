"""
Shared utilities for RQ1–RQ3 analysis scripts.
All scripts assume they are run from t:\\thesis with PYTHONPATH=t:\\thesis.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import sqlite3
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, json

# ── Paths ─────────────────────────────────────────────────────────────────────
DB_PATH   = "data/dataset.db"
PLOTS_DIR = "analysis/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

SEED = 42

# ── Feature columns (all numeric, excluding identifiers) ─────────────────────
VISUAL_COLS = [
    "lip_distance", "mar", "mouth_area", "mouth_width",
    "lip_curvature", "mouth_circularity",
    "_neutral",
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
    "jaw_forward", "jaw_left", "jaw_open", "jaw_right",
    "mouth_close", "mouth_dimple_left", "mouth_dimple_right",
    "mouth_frown_left", "mouth_frown_right", "mouth_funnel",
    "mouth_left", "mouth_lower_down_left", "mouth_lower_down_right",
    "mouth_press_left", "mouth_press_right", "mouth_pucker", "mouth_right",
    "mouth_roll_lower", "mouth_roll_upper",
    "mouth_shrug_lower", "mouth_shrug_upper",
    "mouth_smile_left", "mouth_smile_right",
    "mouth_stretch_left", "mouth_stretch_right",
    "mouth_upper_up_left", "mouth_upper_up_right",
    "nose_sneer_left", "nose_sneer_right",
]

AUDIO_COLS = (
    [f"mfcc_{i}"        for i in range(13)] +
    [f"mfcc_delta_{i}"  for i in range(13)] +
    [f"mfcc_delta2_{i}" for i in range(13)] +
    ["audio_magnitude", "mfcc_energy"]
)

FEATURE_COLS = VISUAL_COLS + AUDIO_COLS


# ── Database helpers ──────────────────────────────────────────────────────────

def load_phoneme_data(phoneme: str, min_occurrences: int = 3) -> pd.DataFrame:
    """
    Return a per-video DataFrame: for each video that contains `phoneme`
    at least `min_occurrences` times, compute the mean of every feature
    across those occurrences.

    Columns: filename, video_type, dataset, label (1=Fake), phoneme_count,
             + all FEATURE_COLS.
    """
    agg_exprs = ", ".join(f"AVG({c}) AS {c}" for c in FEATURE_COLS)
    query = f"""
        SELECT filename, video_type, dataset,
               COUNT(*) AS phoneme_count,
               {agg_exprs}
        FROM lags
        WHERE phoneme = ?
        GROUP BY filename, video_type, dataset
        HAVING COUNT(*) >= ?
    """
    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql_query(query, conn, params=(phoneme, min_occurrences))
    conn.close()
    df["label"] = (df["video_type"] == "Fake").astype(int)
    return df


def load_phoneme_raw(phoneme: str) -> pd.DataFrame:
    """
    Return all individual rows for a given phoneme (not aggregated per video).
    Suitable for distribution-level analyses (RQ2).
    """
    query = "SELECT * FROM lags WHERE phoneme = ?"
    conn  = sqlite3.connect(DB_PATH)
    df    = pd.read_sql_query(query, conn, params=(phoneme,))
    conn.close()
    df["label"] = (df["video_type"] == "Fake").astype(int)
    return df


def feature_introspection(phonemes, n_top: int = 15, alpha: float = 0.05,
                          feature_cols=None) -> pd.DataFrame:
    """
    Compute Mann–Whitney U (Fake vs Real) per phoneme separately, then average
    r_eff across phonemes with equal weight.  For single-phoneme calls this is
    identical to the pooled computation; for groups it prevents high-frequency
    phonemes from dominating the aggregate r_eff.

    Returns DataFrame: feature, r_eff (mean across phonemes), abs_r.
    r_eff > 0: higher in Fake; r_eff < 0: higher in Real.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if isinstance(phonemes, str):
        phonemes = [phonemes]

    placeholders = ",".join("?" * len(phonemes))
    feat_sql = ", ".join(feature_cols)
    query = (f"SELECT phoneme, video_type, {feat_sql} FROM lags "
             f"WHERE phoneme IN ({placeholders})")
    conn = sqlite3.connect(DB_PATH)
    df_all = pd.read_sql_query(query, conn, params=list(phonemes))
    conn.close()
    df_all["label"] = (df_all["video_type"] == "Fake").astype(int)

    ph_r_effs = {}
    for ph in phonemes:
        df = df_all[df_all["phoneme"] == ph]
        r_map = {}
        for feat in feature_cols:
            fake_v = df[df["label"] == 1][feat].dropna().values
            real_v = df[df["label"] == 0][feat].dropna().values
            if len(fake_v) < 10 or len(real_v) < 10:
                r_map[feat] = np.nan
                continue
            stat, _ = mannwhitneyu(fake_v, real_v, alternative="two-sided")
            r_map[feat] = 1.0 - 2.0 * stat / (len(fake_v) * len(real_v))
        ph_r_effs[ph] = r_map

    rows = []
    for feat in feature_cols:
        vals = [ph_r_effs[ph].get(feat, np.nan) for ph in phonemes]
        vals_clean = [v for v in vals if not np.isnan(v)]
        if not vals_clean:
            continue
        mean_r = float(np.mean(vals_clean))
        rows.append(dict(feature=feat, r_eff=mean_r, abs_r=abs(mean_r)))

    return (pd.DataFrame(rows)
            .sort_values("abs_r", ascending=False)
            .reset_index(drop=True))


# ── Video-level discriminability + permutation / bootstrap helpers ────────────

def phoneme_discriminability(phoneme: str, min_occurrences: int = 3,
                              feature_cols=None) -> dict:
    """
    Video-level discriminability score for one phoneme.

    Uses load_phoneme_data (per-video aggregated features), runs MWU (Fake vs Real)
    on each feature independently, and returns mean |r_eff| across all features as
    a single discriminability score.  Because data is already aggregated per video,
    observations are independent — no within-video clustering bias.

    Returns dict: phoneme, n, n_fake, n_real, mean_abs_r, r_df (sorted DataFrame).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df = load_phoneme_data(phoneme, min_occurrences=min_occurrences)
    fake_df = df[df["label"] == 1]
    real_df = df[df["label"] == 0]
    n_fake, n_real = len(fake_df), len(real_df)
    empty = dict(phoneme=phoneme, n=len(df), n_fake=n_fake,
                 n_real=n_real, mean_abs_r=np.nan, r_df=None)
    if n_fake < 5 or n_real < 5:
        return empty
    n_tests = len(feature_cols)
    rows = []
    for feat in feature_cols:
        f_vals = fake_df[feat].dropna().values
        r_vals = real_df[feat].dropna().values
        if len(f_vals) < 5 or len(r_vals) < 5:
            continue
        stat, p_raw = mannwhitneyu(f_vals, r_vals, alternative="two-sided")
        r_eff  = 1.0 - 2.0 * stat / (len(f_vals) * len(r_vals))
        p_bonf = min(p_raw * n_tests, 1.0)
        rows.append(dict(feature=feat, r_eff=r_eff, abs_r=abs(r_eff),
                         p_bonf=p_bonf, sig=p_bonf < 0.05))
    if not rows:
        return empty
    r_df = (pd.DataFrame(rows)
              .sort_values("abs_r", ascending=False)
              .reset_index(drop=True))
    return dict(phoneme=phoneme, n=len(df), n_fake=n_fake, n_real=n_real,
                mean_abs_r=float(r_df["abs_r"].mean()), r_df=r_df)


def phoneme_balanced_auc(phoneme: str, min_occurrences: int = 3,
                         n_per_class: int = None,
                         feature_cols=None, seed: int = SEED) -> dict:
    """
    Per-phoneme bACC/AUC on a single class-balanced subsample evaluated with
    5-fold CV (see compute_metrics_ci). Bootstrap CIs use concatenated OOF
    predictions (n_per_class * 2 samples × 5 folds).
    Returns same dict as compute_metrics_ci plus n_fake_orig/n_real_orig.
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    df      = load_phoneme_data(phoneme, min_occurrences=min_occurrences)
    fake_df = df[df["label"] == 1]
    real_df = df[df["label"] == 0]
    n_each  = min(len(fake_df), len(real_df))
    if n_per_class is not None:
        n_each = min(n_each, n_per_class)
    empty = dict(phoneme=phoneme, n=n_each * 2,
                 n_fake_orig=len(fake_df), n_real_orig=len(real_df),
                 auc=np.nan, auc_lo=np.nan, auc_hi=np.nan,
                 bacc=np.nan, bacc_lo=np.nan, bacc_hi=np.nan)
    if n_each < 20:
        return empty
    rng    = np.random.RandomState(seed)
    df_bal = pd.concat([
        fake_df.sample(n=n_each, random_state=rng.randint(1 << 31)),
        real_df.sample(n=n_each, random_state=rng.randint(1 << 31)),
    ]).reset_index(drop=True)
    m = compute_metrics_ci(df_bal, feature_cols=feature_cols, seed=seed)
    m["n_fake_orig"] = len(fake_df)
    m["n_real_orig"] = len(real_df)
    return m


def permutation_test(scores_a, scores_b, n_perm: int = 10_000,
                     alternative: str = "greater", seed: int = SEED) -> tuple:
    """
    Permutation test on difference of group means (mean_A - mean_B).

    alternative: 'greater' | 'less' | 'two-sided'
    Returns (observed_diff, p_value).
    """
    a = np.array(scores_a, dtype=float)
    b = np.array(scores_b, dtype=float)
    observed = a.mean() - b.mean()
    combined = np.concatenate([a, b])
    na = len(a)
    rng  = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm    = rng.permutation(combined)
        null[i] = perm[:na].mean() - perm[na:].mean()
    if alternative == "greater":
        p = (null >= observed).mean()
    elif alternative == "less":
        p = (null <= observed).mean()
    else:
        p = (np.abs(null) >= np.abs(observed)).mean()
    return float(observed), float(p)


def bootstrap_mean_ci(scores, n_bootstrap: int = 2000,
                      seed: int = SEED) -> tuple:
    """
    Bootstrap 95% percentile CI on the mean of scores.
    Returns (mean, lo, hi).
    """
    arr = np.array(scores, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, np.nan
    rng   = np.random.RandomState(seed)
    means = [rng.choice(arr, len(arr), replace=True).mean()
             for _ in range(n_bootstrap)]
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ── AUC with bootstrap CI ─────────────────────────────────────────────────────

def compute_metrics_ci(df: pd.DataFrame,
                       feature_cols=None,
                       n_bootstrap: int = 500,
                       seed: int = SEED,
                       n_folds: int = 5):
    """
    Train LR via stratified n_folds-fold CV, report mean AUC/bACC.
    Bootstrap CIs use concatenated out-of-fold predictions (n*n_folds samples).
    Returns dict with keys:
      auc, auc_lo, auc_hi,
      bacc, bacc_lo, bacc_hi,
      n, n_fake, n_real,
      fold_baccs, fold_aucs   (raw per-fold scores for multi-seed aggregation)
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X = df[feature_cols].values
    y = df["label"].values
    n_fake = int(y.sum())
    n_real = int((1 - y).sum())
    empty  = dict(auc=np.nan, auc_lo=np.nan, auc_hi=np.nan,
                  bacc=np.nan, bacc_lo=np.nan, bacc_hi=np.nan,
                  n=len(df), n_fake=n_fake, n_real=n_real,
                  fold_baccs=[], fold_aucs=[])

    if n_fake < 10 or n_real < 10:
        return empty

    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_baccs: list = []
    fold_aucs:  list = []
    all_y, all_probs, all_preds = [], [], []

    for tr_idx, te_idx in kf.split(X, y):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_te)[:, 1]
        preds = clf.predict(X_te)
        fold_aucs.append(float(roc_auc_score(y_te, probs)))
        fold_baccs.append(float(balanced_accuracy_score(y_te, preds)))
        all_y.extend(y_te); all_probs.extend(probs); all_preds.extend(preds)

    auc  = float(np.mean(fold_aucs))
    bacc = float(np.mean(fold_baccs))

    all_y     = np.array(all_y)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    rng = np.random.RandomState(seed)
    n   = len(all_y)
    boot_aucs:  list = []
    boot_baccs: list = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(all_y[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(all_y[idx], all_probs[idx]))
        boot_baccs.append(balanced_accuracy_score(all_y[idx], all_preds[idx]))

    return dict(
        auc=auc,
        auc_lo=float(np.percentile(boot_aucs, 2.5)),
        auc_hi=float(np.percentile(boot_aucs, 97.5)),
        bacc=bacc,
        bacc_lo=float(np.percentile(boot_baccs, 2.5)),
        bacc_hi=float(np.percentile(boot_baccs, 97.5)),
        n=len(df), n_fake=n_fake, n_real=n_real,
        fold_baccs=fold_baccs, fold_aucs=fold_aucs,
    )


# Keep a simpler alias for scripts that only need AUC
def compute_auc_ci(df, feature_cols=None, n_bootstrap=500, seed=SEED):
    m = compute_metrics_ci(df, feature_cols, n_bootstrap, seed)
    return m["auc"], m["auc_lo"], m["auc_hi"], m["n"], m["n_fake"], m["n_real"]


# ── Plot helpers ──────────────────────────────────────────────────────────────

THESIS_STYLE = {
    "figure.dpi":        150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "legend.fontsize":   11,
}

def apply_style():
    plt.rcParams.update(THESIS_STYLE)

def save_plot(fig, name: str):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


