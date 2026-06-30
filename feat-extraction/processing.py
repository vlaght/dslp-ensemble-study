import logging

import sqlite3
import pandas as pd
import glob
import os
import io
import subprocess
import cv2
import numpy as np
import torch
import librosa
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

TARGET_PHONEMES = ["p", "b", "m", "f", "v", "a", "e", "i", "o", "u", "k", "n"]

# eSpeak NG for the phonemizer backend. Configure with the PHONEMIZER_ESPEAK_LIBRARY /
# PHONEMIZER_ESPEAK_PATH environment variables. On Linux/macOS eSpeak NG is auto-detected
# on PATH; the Windows defaults below apply only when the env vars are unset and the files
# exist (forward slashes are accepted by Windows).
_ESPEAK_DEFAULTS = {
    "PHONEMIZER_ESPEAK_LIBRARY": "C:/Program Files/eSpeak NG/libespeak-ng.dll",
    "PHONEMIZER_ESPEAK_PATH": "C:/Program Files/eSpeak NG/espeak-ng.exe",
}
for _k, _v in _ESPEAK_DEFAULTS.items():
    if _k not in os.environ and os.path.exists(_v):
        os.environ[_k] = _v

# --- 1. Model Download & Initialization ---
PRETRAINED_FOLDER =  os.path.join(os.path.dirname(os.path.dirname(__file__)), "pretrained_models")
LANDMARKER_PATH = os.path.join(PRETRAINED_FOLDER, "face_landmarker.task")
# Download MediaPipe face landmarker if not present
if not os.path.exists(LANDMARKER_PATH):
    logging.debug("Downloading face_landmarker.task...")
    os.makedirs(PRETRAINED_FOLDER, exist_ok=True)
    import urllib.request
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        LANDMARKER_PATH,
    )

# Initialize MediaPipe FaceLandmarker
logging.debug("Initializing MediaPipe FaceLandmarker...")
base_options = python.BaseOptions(
    model_asset_path=LANDMARKER_PATH,
    # delegate=python.BaseOptions.Delegate.GPU,
)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
    num_faces=1,
)
face_landmarker = vision.FaceLandmarker.create_from_options(options)

# Initialize Wav2Vec2 (Speech-to-Text)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.debug(f"PyTorch running on: {device}")

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
)
model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-lv-60-espeak-cv-ft",
).to(device)
# --- 2. Helper Functions ---


def calculate_mar(landmarks):
    # Inner lip indices for MAR
    # Corners: 78, 308; Vertical pairs: (81, 178), (13, 14), (311, 402)
    p_corners = [landmarks[78], landmarks[308]]
    p_vertical = [
        (landmarks[81], landmarks[178]),
        (landmarks[13], landmarks[14]),
        (landmarks[311], landmarks[402]),
    ]

    dist_h = np.linalg.norm(
        np.array([p_corners[0].x, p_corners[0].y])
        - np.array([p_corners[1].x, p_corners[1].y])
    )
    if dist_h == 0:
        return 0.0

    dist_v_sum = 0
    for top, bot in p_vertical:
        dist_v_sum += np.linalg.norm(
            np.array([top.x, top.y]) - np.array([bot.x, bot.y])
        )

    return dist_v_sum / (2.0 * dist_h)


def calculate_mouth_area(landmarks):
    # Inner lip polygon indices (ordered)
    indices = [
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
    ]
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    # Shoelace formula for polygon area
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def calculate_circularity(landmarks):
    # Inner lip indices (same as area)
    indices = [
        78,
        191,
        80,
        81,
        82,
        13,
        312,
        311,
        310,
        415,
        308,
        324,
        318,
        402,
        317,
        14,
        87,
        178,
        88,
        95,
    ]
    points = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])

    # Area (Shoelace)
    x = points[:, 0]
    y = points[:, 1]
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Perimeter
    perimeter = np.sum(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))

    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter**2)


def extract_visual_lip_data(video_path, landmarker):
    lip_distances = []
    timestamps = []
    blendshapes = []
    mars = []
    areas = []
    widths = []
    curvatures = []
    circularities = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], [], [], [], [], [], [], []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

        if detection_result.face_landmarks:
            landmarks = detection_result.face_landmarks[0]
            # Lip landmarks: 13 (upper), 14 (lower)
            p1 = np.array([landmarks[13].x, landmarks[13].y])
            p2 = np.array([landmarks[14].x, landmarks[14].y])
            # Normalization (IOD): 33, 263
            eye_left = np.array([landmarks[33].x, landmarks[33].y])
            eye_right = np.array([landmarks[263].x, landmarks[263].y])
            iod = np.linalg.norm(eye_left - eye_right)

            dist = np.linalg.norm(p1 - p2) / (iod if iod > 0 else 1.0)
            lip_distances.append(dist)
            timestamps.append(frame_count / fps)

            if detection_result.face_blendshapes:
                blendshapes.append(
                    {
                        cat.category_name: cat.score
                        for cat in detection_result.face_blendshapes[0]
                    }
                )
            else:
                blendshapes.append({})

            mars.append(calculate_mar(landmarks))
            areas.append(calculate_mouth_area(landmarks))

            # Mouth Width (Normalized)
            p_left = np.array([landmarks[78].x, landmarks[78].y])
            p_right = np.array([landmarks[308].x, landmarks[308].y])
            widths.append(np.linalg.norm(p_left - p_right) / (iod if iod > 0 else 1.0))

            # Lip Curvature (avg corner y - avg center y)
            curvatures.append(
                ((landmarks[78].y + landmarks[308].y) / 2.0)
                - ((landmarks[13].y + landmarks[14].y) / 2.0)
            )

            circularities.append(calculate_circularity(landmarks))
        frame_count += 1
    cap.release()
    return (
        lip_distances,
        timestamps,
        blendshapes,
        mars,
        areas,
        widths,
        curvatures,
        circularities,
    )


def extract_mfcc(
    video_path, sr=16000, n_mfcc=13, n_fft=512, hop_length=160, win_length=320
):
    """
    Extract MFCC features from video audio.

    Parameters follow the paper:
    - 13 MFCCs + delta + double-delta + log-energy = 40 features total
    - 20ms windows (win_length=320 at 16kHz), 10ms hop (hop_length=160 at 16kHz)
    - Pre-emphasis coefficient 0.97, Hamming window
    - Mean normalization (subtract signal mean)
    - 512-point FFT

    Returns:
        features: np.ndarray of shape (40, T) — one column per frame
        timestamps: np.ndarray of shape (T,) — center time of each frame in seconds
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
            "-f",
            "wav",
            "-loglevel",
            "quiet",
            "-",
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0 or not process.stdout:
            return None, None

        audio_memory = io.BytesIO(process.stdout)
        speech, sr = librosa.load(audio_memory, sr=sr)

        if len(speech) == 0:
            return None, None

        # Mean normalization
        speech = speech - np.mean(speech)

        # Pre-emphasis (coefficient 0.97)
        speech = np.append(speech[0], speech[1:] - 0.97 * speech[:-1])

        # 13 MFCCs
        mfccs = librosa.feature.mfcc(
            y=speech,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hamming",
        )  # (13, T)

        # Delta and double-delta
        delta = librosa.feature.delta(mfccs)  # (13, T)
        delta2 = librosa.feature.delta(mfccs, order=2)  # (13, T)

        # Log-energy per frame
        rms = librosa.feature.rms(
            y=speech, frame_length=win_length, hop_length=hop_length
        )  # (1, T)
        log_energy = np.log(rms + 1e-9)  # (1, T)

        # Stack: 13 + 13 + 13 + 1 = 40
        features = np.vstack([mfccs, delta, delta2, log_energy])  # (40, T)

        n_frames = features.shape[1]
        timestamps = librosa.frames_to_time(
            np.arange(n_frames), sr=sr, hop_length=hop_length
        )

        return features, timestamps
    except Exception:
        return None, None


def extract_phonemes(video_path):
    try:
        # ffmpeg command to stream audio to stdout
        cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "-loglevel",
            "quiet",
            "-",
        ]
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if process.returncode != 0 or not process.stdout:
            return []

        audio_memory = io.BytesIO(process.stdout)
        speech, sr = librosa.load(audio_memory, sr=16000)

        # Normalize audio for magnitude calculation
        speech_norm = speech / (np.max(np.abs(speech)) + 1e-9)

        input_values = processor(
            speech, return_tensors="pt", sampling_rate=16000
        ).input_values.to(device)
        with torch.no_grad():
            logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)[0]
        phonemes = []
        prev_id = -1
        pad_id = processor.tokenizer.pad_token_id
        time_stride = 0.02

        for idx, token_id in enumerate(predicted_ids):
            if token_id != pad_id and token_id != prev_id:
                char = processor.tokenizer.convert_ids_to_tokens(token_id.item())

                # Count consecutive frames with this same token to get actual phoneme duration
                # (excludes any following pause/blank frames)
                span = 1
                while idx + span < len(predicted_ids) and predicted_ids[idx + span] == token_id:
                    span += 1
                duration = span * time_stride

                # Calculate magnitude over the actual phoneme span
                start_sample = int(idx * time_stride * 16000)
                end_sample = int((idx + span) * time_stride * 16000)
                segment = speech_norm[start_sample:end_sample]
                magnitude = np.sqrt(np.mean(segment**2)) if len(segment) > 0 else 0.0

                phonemes.append(
                    {
                        "phoneme": char,
                        "time": idx * time_stride,
                        "duration": duration,
                        "magnitude": float(magnitude),
                    }
                )
            prev_id = token_id
        return phonemes
    except Exception as e:
        return []


def align_features(
    lip_distances,
    video_timestamps,
    phonemes_data,
    blendshapes,
    mars,
    areas,
    widths,
    curvatures,
    circularities,
    mfcc_features=None,
    mfcc_timestamps=None,
):
    video_timestamps = np.array(video_timestamps)
    aligned_data = []

    if len(video_timestamps) == 0:
        return []

    has_mfcc = mfcc_features is not None and mfcc_timestamps is not None

    for p in phonemes_data:
        onset_time = p["time"]

        # Use actual phoneme span from Wav2Vec2 output (excludes following pauses)
        duration = p["duration"]

        for frame_position, p_time in [("onset", onset_time)]:
            # Find closest video frame to this time point
            idx = (np.abs(video_timestamps - p_time)).argmin()

            entry = {
                "phoneme": p["phoneme"],
                "frame_position": frame_position,
                "timestamp": p_time,
                "audio_magnitude": p["magnitude"],
                "lip_distance": lip_distances[idx],
                "mouth_area": areas[idx],
                "mar": mars[idx],
                "mouth_width": widths[idx],
                "lip_curvature": curvatures[idx],
                "mouth_circularity": circularities[idx],
                "blendshapes": blendshapes[idx],
            }

            if has_mfcc:
                mfcc_idx = (np.abs(mfcc_timestamps - p_time)).argmin()
                mfcc_vec = mfcc_features[:, mfcc_idx]  # (40,)
                entry["mfcc"] = mfcc_vec

            aligned_data.append(entry)

    return aligned_data


def get_video_type(video_path):
    path_parts = video_path.split(os.sep)
    if "real" in path_parts:
        return "Real"
    return "Fake"
