"""
extract_noise_features.py
--------------------------
Extracts per-frame noise-residual features from .mp4 videos.

Two modes via --mode argument:
  --mode ext   : EXT datasets (FakeAVCeleb_v1.2 + augmented)
                 Labels derived from directory structure (real/ subdir → Real)
                 Output: t:\\thesis\\data\\noise_features.db
  --mode dsv2  : DeepSpeakV2 dataset
                 Labels looked up from deepspeakv2.db (ground truth)
                 Output: t:\\thesis\\data\\noise_features_dsv2.db

19 features per frame:
  0-3  : Laplacian mean-abs, std, kurtosis, entropy
  4-9  : DoG mean-abs + std at sigma=0.5/1.0/2.0
  10-13: per-quadrant mean-abs Laplacian
  14   : high-pass energy ratio
  15-16: multi-scale (32x32) Laplacian mean-abs, std
  17   : LBP-inspired texture fraction
  18   : temporal consistency (NaN for first frame)

Usage:
    python.exe -u feat-extraction/extract_noise_features.py --mode ext  > .tmp/noise_extract.log 2>&1
    python.exe -u feat-extraction/extract_noise_features.py --mode dsv2 > .tmp/noise_extract_dsv2.log 2>&1
"""

import sys
import os
import json
import sqlite3
import time
import argparse
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import kurtosis as scipy_kurtosis

# OpenCV Haar cascade face detector
_HAAR_XML = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_HAAR_XML)
if _face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from {_HAAR_XML}")
print("Face detector: OpenCV Haar cascade")

# ---------------------------------------------------------------------------
# Paths (SCRIPT_DIR is now feat-extraction/, so root is one level up)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.join(SCRIPT_DIR, "..")   # t:\thesis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_FRAMES      = 40
FACE_SIZE     = 64
SMALL_SIZE    = 32
N_TOTAL_FEATS = 19

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
FEAT_COLS  = ", ".join(f"feat_{i} REAL" for i in range(N_TOTAL_FEATS))
CREATE_SQL = f"""
CREATE TABLE IF NOT EXISTS features (
    filename   TEXT,
    video_type TEXT,
    frame_idx  INTEGER,
    timestamp  REAL,
    {FEAT_COLS}
)
"""
INDEX_SQL = "CREATE INDEX IF NOT EXISTS idx_filename ON features (filename)"

# ---------------------------------------------------------------------------
# Helpers: label
# ---------------------------------------------------------------------------

def get_video_type_from_path(video_path: str) -> str:
    """EXT mode: label from directory structure (real/ subdir → Real)."""
    parts = video_path.replace("\\", "/").split("/")
    return "Real" if "real" in parts else "Fake"


def load_label_lookup(db_path: str) -> dict:
    """DSV2 mode: load label lookup from deepspeakv2.db."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT filename, video_type FROM lags").fetchall()
    conn.close()
    return {r[0]: r[1] for r in rows}


# ---------------------------------------------------------------------------
# Helpers: face detection
# ---------------------------------------------------------------------------

def _center_crop_gray(frame_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    s  = min(h, w)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    crop = frame_bgr[y0:y0 + s, x0:x0 + s]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size))


def detect_face_gray(frame_bgr: np.ndarray, size: int = FACE_SIZE) -> np.ndarray:
    gray_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray_full, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
    )
    if len(faces) > 0:
        areas = [w * h for (x, y, w, h) in faces]
        x, y, w, h = faces[int(np.argmax(areas))]
        crop = gray_full[y:y + h, x:x + w]
        return cv2.resize(crop, (size, size))
    return _center_crop_gray(frame_bgr, size)


# ---------------------------------------------------------------------------
# Feature extraction: noise residual (19 features)
# ---------------------------------------------------------------------------

def _safe_entropy(arr: np.ndarray) -> float:
    a = np.abs(arr).ravel() + 1e-12
    p = a / a.sum()
    return float(-np.sum(p * np.log2(p)))


def extract_noise_features(
    gray_patch: np.ndarray,
    prev_lap:   np.ndarray | None,
) -> tuple:
    patch_f32 = gray_patch.astype(np.float32)

    # 1. Laplacian residual on 64×64 patch
    lap = cv2.Laplacian(patch_f32, cv2.CV_32F)
    lap_abs  = np.abs(lap)
    lap_mean = float(lap_abs.mean())
    lap_std  = float(lap_abs.std())
    lap_kurt = float(scipy_kurtosis(lap.ravel(), fisher=True))
    lap_ent  = _safe_entropy(lap)
    laplacian_feats = np.array([lap_mean, lap_std, lap_kurt, lap_ent], dtype=np.float32)

    # 2. Gaussian difference (DoG) at σ = 0.5, 1.0, 2.0
    dog_feats = np.zeros(6, dtype=np.float32)
    for k, sigma in enumerate([0.5, 1.0, 2.0]):
        smooth = ndimage.gaussian_filter(patch_f32, sigma=sigma)
        dog    = patch_f32 - smooth
        dog_feats[2 * k]     = float(np.abs(dog).mean())
        dog_feats[2 * k + 1] = float(dog.std())

    # 3. Per-quadrant mean-absolute Laplacian
    half = FACE_SIZE // 2
    quad_feats = np.array([
        float(np.abs(lap[:half, :half ]).mean()),
        float(np.abs(lap[:half, half: ]).mean()),
        float(np.abs(lap[half:, :half ]).mean()),
        float(np.abs(lap[half:, half: ]).mean()),
    ], dtype=np.float32)

    # 4. High-pass energy ratio
    total_energy = float((patch_f32 ** 2).sum()) + 1e-12
    hp_ratio     = float((lap ** 2).sum()) / total_energy

    # 5. Multi-scale: Laplacian on 32×32 resized patch
    small    = cv2.resize(gray_patch, (SMALL_SIZE, SMALL_SIZE)).astype(np.float32)
    lap_s    = cv2.Laplacian(small, cv2.CV_32F)
    ms_feats = np.array([float(np.abs(lap_s).mean()), float(lap_s.std())], dtype=np.float32)

    # 6. LBP-inspired texture fraction
    local_mean = ndimage.uniform_filter(patch_f32, size=3)
    lbp_frac   = float((patch_f32 > local_mean).mean())

    # 7. Temporal consistency
    if prev_lap is None:
        temporal_diff = float("nan")
    else:
        temporal_diff = float(np.abs(lap - prev_lap).mean())

    feats = np.concatenate([
        laplacian_feats,
        dog_feats,
        quad_feats,
        [hp_ratio],
        ms_feats,
        [lbp_frac],
        [temporal_diff],
    ]).astype(np.float32)

    return feats, lap


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _process_video_core(video_path: str, filename: str, video_type: str) -> list:
    rows     = []
    prev_lap = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [SKIP] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0

    if total_frames < 1:
        cap.release()
        return []

    frame_indices  = np.linspace(0, total_frames - 1, N_FRAMES, dtype=int)
    prev_frame_idx = -1

    for frame_idx in frame_indices:
        if frame_idx != prev_frame_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            continue
        prev_frame_idx = frame_idx

        timestamp = frame_idx / fps

        try:
            gray_patch      = detect_face_gray(frame_bgr)
            feats, curr_lap = extract_noise_features(gray_patch, prev_lap)
            prev_lap        = curr_lap
        except Exception:
            feats    = np.full(N_TOTAL_FEATS, np.nan, dtype=np.float32)
            prev_lap = None

        # Replace NaN with 0, except feat_18 (temporal) preserves NaN
        feats_safe = np.where(np.isnan(feats), 0.0, feats)
        if np.isnan(feats[18]):
            feats_safe[18] = float("nan")
        feats_safe = np.where(np.isinf(feats_safe), 0.0, feats_safe)

        row = (filename, video_type, int(frame_idx), float(timestamp)) + tuple(feats_safe.tolist())
        rows.append(row)

    cap.release()
    return rows


def process_video_ext(video_path: str, dataset_root: str) -> list:
    """EXT mode: uses path-based label."""
    filename   = os.path.relpath(video_path, dataset_root).replace("\\", "/")
    video_type = get_video_type_from_path(video_path)
    return _process_video_core(video_path, filename, video_type)


def process_video_dsv2(video_path: str, video_type: str) -> list:
    """DSV2 mode: uses DB-provided label."""
    filename = os.path.basename(video_path)
    return _process_video_core(video_path, filename, video_type)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_all_videos(root: str) -> list:
    videos = []
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            if fname.lower().endswith(".mp4"):
                videos.append(os.path.join(dirpath, fname))
    return sorted(videos)


def load_progress(progress_file: str) -> set:
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            pass
    return set()


def save_progress(done: set, progress_file: str) -> None:
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(sorted(done), f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract noise-residual features from videos")
    parser.add_argument("--mode", choices=["ext", "dsv2"], required=True,
                        help="ext: FakeAVCeleb+augmented (dir labels); dsv2: DeepSpeakV2 (DB labels)")
    parser.add_argument("--dataset-root", default=None,
                        help="override the dataset root directory")
    parser.add_argument("--output-db", default=None,
                        help="override the output feature DB path")
    parser.add_argument("--label-db", default=None,
                        help="override the DSV2 label DB path (dsv2 mode)")
    parser.add_argument("--progress-file", default=None,
                        help="override the resume-progress JSON path")
    args = parser.parse_args()

    if args.mode == "ext":
        dataset_root  = os.path.join(ROOT_DIR, "datasets")
        dataset_dirs  = ["FakeAVCeleb_v1.2", "augmented"]
        output_db     = os.path.join(ROOT_DIR, "data", "noise_features.db")
        progress_file = os.path.join(ROOT_DIR, ".tmp", "noise_extract_progress.json")
        label_lookup  = None
    else:  # dsv2
        dataset_root  = os.path.join(ROOT_DIR, "datasets", "deepspeakv2")
        dataset_dirs  = None
        label_db      = args.label_db or os.path.join(ROOT_DIR, "data", "deepspeakv2.db")
        output_db     = os.path.join(ROOT_DIR, "data", "noise_features_dsv2.db")
        progress_file = os.path.join(ROOT_DIR, ".tmp", "noise_extract_dsv2_progress.json")
        label_lookup  = load_label_lookup(label_db)
        print(f"Label lookup : {len(label_lookup)} entries from deepspeakv2.db")

    # CLI path overrides (fall back to the per-mode defaults above)
    if args.dataset_root:
        dataset_root = args.dataset_root
    if args.output_db:
        output_db = args.output_db
    if args.progress_file:
        progress_file = args.progress_file

    print(f"=== extract_noise_features.py --mode {args.mode} ===")
    print(f"Dataset root : {dataset_root}")
    print(f"Output DB    : {output_db}")
    print(f"Progress file: {progress_file}")

    con = sqlite3.connect(output_db)
    cur = con.cursor()
    cur.execute(CREATE_SQL)
    con.commit()

    # Collect videos
    if dataset_dirs is not None:
        all_videos = []
        for d in dataset_dirs:
            all_videos.extend(collect_all_videos(os.path.join(dataset_root, d)))
    else:
        all_videos = collect_all_videos(dataset_root)

    print(f"Total videos found: {len(all_videos)}")

    done = load_progress(progress_file)

    # Progress key: relpath for EXT, basename for DSV2
    if args.mode == "ext":
        remaining = [v for v in all_videos
                     if os.path.relpath(v, dataset_root).replace("\\", "/") not in done]
    else:
        remaining = [v for v in all_videos if os.path.basename(v) not in done]

    print(f"Already processed : {len(done)}")
    print(f"Remaining         : {len(remaining)}")

    if not remaining:
        print("Nothing to do – building index and exiting.")
        cur.execute(INDEX_SQL)
        con.commit()
        con.close()
        return

    # DSV2 sanity check
    if args.mode == "dsv2":
        print("\n[sanity] Label check on first 5 videos:")
        for v in remaining[:5]:
            fname = os.path.basename(v)
            lbl   = label_lookup.get(fname, "UNKNOWN")
            print(f"  {fname}  ->  {lbl}")

    insert_sql = (
        "INSERT INTO features VALUES ("
        + ", ".join(["?"] * (4 + N_TOTAL_FEATS))
        + ")"
    )

    batch_rows = []
    videos_since_commit = 0
    COMMIT_EVERY = 50

    t_start   = time.time()
    processed = 0
    errors    = 0

    for i, vpath in enumerate(remaining):
        if args.mode == "ext":
            vname = os.path.relpath(vpath, dataset_root).replace("\\", "/")
            try:
                rows = process_video_ext(vpath, dataset_root)
                if rows:
                    batch_rows.extend(rows)
                done.add(vname)
                processed += 1
            except Exception:
                print(f"  [ERROR] {vpath}")
                traceback.print_exc()
                done.add(vname)
                errors += 1
        else:  # dsv2
            fname      = os.path.basename(vpath)
            video_type = label_lookup.get(fname)
            if video_type is None:
                errors += 1
                done.add(fname)
                continue
            try:
                rows = process_video_dsv2(vpath, video_type)
                if rows:
                    batch_rows.extend(rows)
                done.add(fname)
                processed += 1
            except Exception:
                print(f"  [ERROR] {vpath}")
                traceback.print_exc()
                done.add(fname)
                errors += 1

        videos_since_commit += 1

        if videos_since_commit >= COMMIT_EVERY:
            if batch_rows:
                cur.executemany(insert_sql, batch_rows)
                con.commit()
                batch_rows = []
            save_progress(done, progress_file)
            videos_since_commit = 0

        if (i + 1) % 100 == 0 or (i + 1) == len(remaining):
            elapsed = time.time() - t_start
            rate    = processed / max(elapsed, 1e-6)
            left    = len(remaining) - (i + 1)
            eta_s   = left / max(rate, 1e-6)
            pct     = 100.0 * (i + 1) / len(remaining)
            print(
                f"  [{i+1}/{len(remaining)}] {pct:.1f}% | "
                f"rate={rate:.1f} vid/s | ETA={eta_s/60:.1f} min | errors={errors}"
            )

    if batch_rows:
        cur.executemany(insert_sql, batch_rows)
        con.commit()
    save_progress(done, progress_file)

    print("Building index on filename ...")
    cur.execute(INDEX_SQL)
    con.commit()
    con.close()

    elapsed = time.time() - t_start
    print(
        f"\nDone. {processed} videos processed ({errors} errors) "
        f"in {elapsed/60:.1f} min."
    )
    print(f"Output: {output_db}")


if __name__ == "__main__":
    main()
