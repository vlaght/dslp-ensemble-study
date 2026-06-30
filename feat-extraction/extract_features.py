import os
import glob
import sqlite3
import pandas as pd
import re

from processing import (
    face_landmarker,
    extract_phonemes,
    extract_visual_lip_data,
    extract_mfcc,
    align_features,
    get_video_type,
)


LOG_ONCE_PER_N = 50
LIP_THRESH = 0.113
BATCH_SIZE = 50

N_MFCC = 13
MFCC_COLS = (
    [f"mfcc_{i}" for i in range(N_MFCC)]
    + [f"mfcc_delta_{i}" for i in range(N_MFCC)]
    + [f"mfcc_delta2_{i}" for i in range(N_MFCC)]
    + ["mfcc_energy"]
)  # 40 columns total

BLENDSHAPE_NAMES = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]


def to_snake_case(name):
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


BLENDSHAPE_COLS = [to_snake_case(b) for b in BLENDSHAPE_NAMES]


if __name__ == "__main__":
    # --- Paths ---
    import argparse
    _ap = argparse.ArgumentParser(
        description="Extract visual + MFCC + phoneme features into dataset.db")
    _ap.add_argument("--dataset-dir", default=os.path.join(".", "datasets"),
                     help="root directory of source videos (default: ./datasets)")
    _ap.add_argument("--db-path", default=os.path.join(".", "data", "dataset.db"),
                     help="output SQLite DB path (default: ./data/dataset.db)")
    _args = _ap.parse_args()
    base_dataset_path = _args.dataset_dir
    db_path = _args.db_path

    # --- 1. Ensure Dataset Exists ---
    print("Checking dataset availability...")
    video_pattern = os.path.join(base_dataset_path, "**", "*.mp4")
    found_videos = glob.glob(video_pattern, recursive=True)

    # --- 2. Identify Processed Videos ---
    print("Checking database for existing records...")
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        columns_sql = [
            "filename TEXT",
            "video_type TEXT",
            "dataset TEXT",
            "phoneme TEXT",
            "frame_position TEXT",
            "timestamp REAL",
            "lip_distance REAL",
            "mar REAL",
            "mouth_area REAL",
            "mouth_width REAL",
            "lip_curvature REAL",
            "mouth_circularity REAL",
            "audio_magnitude REAL",
        ]
        columns_sql.extend([f"{col} REAL" for col in BLENDSHAPE_COLS])
        columns_sql.extend([f"{col} REAL" for col in MFCC_COLS])
        columns_concat = ", ".join(columns_sql)

        cursor.execute(f"CREATE TABLE IF NOT EXISTS lags ({columns_concat});")

        # Migrate existing DB: add MFCC columns and dataset column if they don't exist yet
        existing_cols = {row[1] for row in cursor.execute("PRAGMA table_info(lags);")}

        # Add dataset column if missing
        if "dataset" not in existing_cols:
            cursor.execute("ALTER TABLE lags ADD COLUMN dataset TEXT;")

        # Add frame_position column if missing
        if "frame_position" not in existing_cols:
            cursor.execute("ALTER TABLE lags ADD COLUMN frame_position TEXT;")

        # Add MFCC columns if missing
        for col in MFCC_COLS:
            if col not in existing_cols:
                cursor.execute(f"ALTER TABLE lags ADD COLUMN {col} REAL;")
        conn.commit()

        processed_files_q = conn.execute("SELECT filename FROM lags;")
        processed_filenames = {row[0] for row in processed_files_q}  # now stores rel_paths
        conn.commit()
    finally:
        conn.close()

    # --- 3. Filter Videos ---
    videos_to_process = [
        vp for vp in found_videos
        if os.path.relpath(vp, base_dataset_path).replace("\\", "/") not in processed_filenames
    ]
    print(f"Videos remaining to process: {len(videos_to_process)}")

    # --- 4. Resume Processing ---
    batch_lags_to_insert = []
    processed_count = 0

    if videos_to_process:
        print("Starting processing loop...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            for i, vp in enumerate(videos_to_process):
                try:
                    # Extract dataset name from path (top-level folder in datasets/)
                    rel_path = os.path.relpath(vp, base_dataset_path)
                    dataset_name = rel_path.split(os.sep)[0]

                    video_type = get_video_type(vp)
                    if video_type == "Unknown":
                        continue

                    ld, vt, bs, mars, areas, widths, curvatures, circularities = (
                        extract_visual_lip_data(vp, face_landmarker)
                    )
                    if not ld:
                        continue

                    ph = extract_phonemes(vp)
                    if not ph:
                        continue

                    mfcc_feats, mfcc_ts = extract_mfcc(vp)

                    features = align_features(
                        ld,
                        vt,
                        ph,
                        bs,
                        mars,
                        areas,
                        widths,
                        curvatures,
                        circularities,
                        mfcc_features=mfcc_feats,
                        mfcc_timestamps=mfcc_ts,
                    )

                    n_cols = 13 + len(BLENDSHAPE_NAMES) + len(MFCC_COLS)
                    for f in features:
                        bs_values = [
                            f["blendshapes"].get(name, 0.0) for name in BLENDSHAPE_NAMES
                        ]
                        # Convert MFCC numpy values to Python floats to ensure proper storage
                        if "mfcc" in f:
                            mfcc_values = [float(x) for x in f["mfcc"]]
                        else:
                            mfcc_values = [0.0] * len(MFCC_COLS)
                        batch_lags_to_insert.append(
                            (
                                os.path.relpath(vp, base_dataset_path).replace("\\", "/"),
                                video_type,
                                dataset_name,
                                f["phoneme"],
                                f["frame_position"],
                                f["timestamp"],
                                f["lip_distance"],
                                f["mar"],
                                f["mouth_area"],
                                f["mouth_width"],
                                f["lip_curvature"],
                                f["mouth_circularity"],
                                f["audio_magnitude"],
                                *bs_values,
                                *mfcc_values,
                            )
                        )

                    if len(batch_lags_to_insert) >= BATCH_SIZE:
                        placeholders = ", ".join(["?"] * n_cols)
                        cursor.executemany(
                            f"INSERT INTO lags VALUES ({placeholders})",
                            batch_lags_to_insert,
                        )
                        conn.commit()
                        batch_lags_to_insert = []

                    processed_count += 1
                    if processed_count % LOG_ONCE_PER_N == 0:
                        print(f"Processed {i + 1}/{len(videos_to_process)} videos...")

                except Exception as e:
                    print(f"Error processing {os.path.basename(vp)}: {e}")
                    continue

        except KeyboardInterrupt:
            print("\nProcessing interrupted by user.")

        finally:
            if batch_lags_to_insert:
                print(
                    f"Inserting final batch of {len(batch_lags_to_insert)} records..."
                )
                placeholders = ", ".join(
                    ["?"] * (13 + len(BLENDSHAPE_NAMES) + len(MFCC_COLS))
                )
                cursor.executemany(
                    f"INSERT INTO lags VALUES ({placeholders})", batch_lags_to_insert
                )
                conn.commit()
            conn.close()
            print(
                f"Processing finished. Successfully processed {processed_count} new videos."
            )
    else:
        print("No new videos to process.")

    # --- 5. Verify Data ---
    conn_check = sqlite3.connect(db_path)
    print("\nLast 5 DB Entries:")
    print(
        pd.read_sql_query("SELECT * FROM lags ORDER BY rowid DESC LIMIT 5", conn_check)
    )
    conn_check.close()
