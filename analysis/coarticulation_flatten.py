"""
Reads exported_phonemes.db, reconstructs per-phoneme-instance triplets (onset/mid/offset),
and writes one flattened row per instance to a new SQLite database using the
symmetric / coarticulation decomposition (Option D):

    baseline    = (onset + offset) / 2   ← coarticulation context (influence of neighbours)
    peak_excess = mid − baseline          ← phoneme-specific peak deviation from context
    drift       = offset − onset          ← net directional movement across the phoneme

Output columns per feature f:
    {f}_baseline, {f}_peak_excess, {f}_drift

Plus metadata columns: filename, video_type, dataset, phoneme, timestamp (onset frame)
"""

import os
import sqlite3

import numpy as np
import pandas as pd

# --- Configuration ---
INPUT_DB  = os.path.join(".", "exported_phonemes.db")
OUTPUT_DB = os.path.join(".", "phonemes_coarticulation.db")

FRAME_ORDER = {"onset": 0, "mid": 1, "offset": 2}

META_COLS = {"filename", "video_type", "dataset", "phoneme", "frame_position", "timestamp"}


def load_and_group(db_path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    Load the full exported DB, sort frames, and assign instance IDs.

    Returns the annotated DataFrame and the list of numeric feature column names.
    """
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM lags", conn)
    conn.close()

    feature_cols = [c for c in df.columns if c not in META_COLS]

    df["_fp_ord"] = df["frame_position"].map(FRAME_ORDER)
    df = df.sort_values(["filename", "timestamp", "_fp_ord"]).reset_index(drop=True)

    # Each 'onset' row marks the start of a new phoneme instance within a video
    df["_inst"] = df.groupby("filename", sort=False)["frame_position"].transform(
        lambda s: (s == "onset").cumsum()
    )

    return df, feature_cols


def build_flat_rows(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Reconstruct triplets and compute the coarticulation decomposition.

    For each feature f:
        {f}_baseline    = (onset_f + offset_f) / 2
        {f}_peak_excess = mid_f - baseline_f
        {f}_drift       = offset_f - onset_f

    Note: features that are constant across frames (e.g. audio_magnitude,
    which is computed over the full phoneme span) will produce peak_excess ≈ 0
    and drift ≈ 0 — those columns carry no information and can be dropped
    downstream.
    """
    rows = []
    skipped = 0

    for (filename, _inst), grp in df.groupby(["filename", "_inst"], sort=False):
        if len(grp) != 3:
            skipped += 1
            continue

        grp = grp.sort_values("_fp_ord")
        onset  = grp[feature_cols].iloc[0].values.astype(np.float64)
        mid    = grp[feature_cols].iloc[1].values.astype(np.float64)
        offset = grp[feature_cols].iloc[2].values.astype(np.float64)

        baseline    = (onset + offset) / 2.0
        peak_excess = mid - baseline
        drift       = offset - onset

        meta = grp.iloc[0][["filename", "video_type", "dataset", "phoneme"]].to_dict()
        # Preserve the onset timestamp so downstream code can sort phonemes temporally
        meta["timestamp"] = float(grp.iloc[0]["timestamp"])
        row  = dict(meta)

        for i, f in enumerate(feature_cols):
            row[f"{f}_baseline"]    = baseline[i]
            row[f"{f}_peak_excess"] = peak_excess[i]
            row[f"{f}_drift"]       = drift[i]

        rows.append(row)

    if skipped:
        print(f"  Warning: skipped {skipped:,} incomplete triplets")

    return pd.DataFrame(rows)


def main():
    print(f"Reading {INPUT_DB}...")
    df, feature_cols = load_and_group(INPUT_DB)
    print(f"  {len(df):,} rows | {len(feature_cols)} feature columns | "
          f"{df['filename'].nunique():,} unique videos")

    print("Building flattened rows...")
    flat = build_flat_rows(df, feature_cols)
    print(f"  {len(flat):,} instances produced")

    # Drop videos with fewer than 3 phoneme instances — too short for reliable sequence modelling
    instances_per_video = flat.groupby("filename")["filename"].transform("count")
    before = flat["filename"].nunique()
    flat = flat[instances_per_video >= 3].reset_index(drop=True)
    after = flat["filename"].nunique()
    print(f"  Dropped {before - after:,} videos with <3 instances "
          f"({after:,} videos / {len(flat):,} instances remaining)")

    print(f"  Output columns: {len(flat.columns)} "
          f"(5 meta + {len(feature_cols)} × 3 derived = {5 + len(feature_cols) * 3})")

    # Label distribution
    if "video_type" in flat.columns:
        dist = flat["video_type"].value_counts()
        print(f"  Label distribution:\n{dist.to_string()}")

    print(f"\nWriting to {OUTPUT_DB}...")
    if os.path.exists(OUTPUT_DB):
        os.remove(OUTPUT_DB)

    with sqlite3.connect(OUTPUT_DB) as conn:
        flat.to_sql("instances", conn, index=False, if_exists="replace")

    print(f"Done. {len(flat):,} rows written to '{OUTPUT_DB}' (table: instances)")
    print(f"\nSample row (first instance):")
    # Show a few columns to verify
    sample_cols = (
        ["filename", "video_type", "phoneme"]
        + [f"{feature_cols[0]}_baseline",
           f"{feature_cols[0]}_peak_excess",
           f"{feature_cols[0]}_drift"]
    )
    print(flat[sample_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()
