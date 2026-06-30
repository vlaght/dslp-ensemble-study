# Phoneme-Based Audio-Visual Face-Forgery Detection (DSLP Ensemble)

Code, experiments, and pretrained models for the paper *"On Phoneme-Based
Audio-Visual Face Forgery Detection."* The detector is an unweighted-average ensemble of
three models: **DSLP** (dual-stream phoneme-aligned LSTM), **TAFreq** and **TANoise**
(temporal-attention BiLSTMs on frequency / noise-residual features).

This repo lets you either **(A) try the pretrained detector** on your own video, or
**(B) reproduce everything** — every experiment, figure, and table in the paper, plus
feature extraction, cross-validation, and training from scratch.

- **Pretrained weights:** https://huggingface.co/vl4gh7/dslp-ensemble
- **Paper resources (datasets, feature DBs):** supplement Appendix A

---

## Results

10-fold stratified cross-validation. The detector is strongest on the modern DeepSpeak v2
generators; FakeAVCeleb is harder because many of its forgeries leave mouth motion largely
intact.

**Ensemble**

| Dataset | AUC | F1 | Accuracy |
|---|---|---|---|
| FakeAVCeleb (21,544 videos)  | 0.9593 | 0.9599 | 0.9248 |
| DeepSpeak v2 (16,465 videos) | 0.9984 | 0.9783 | 0.9810 |

**Per-component AUC**

| Model | FakeAVCeleb | DeepSpeak v2 |
|---|---|---|
| DSLP    | 0.8606 | 0.9579 |
| TAFreq  | 0.9674 | 0.9954 |
| TANoise | 0.8100 | 0.9356 |
| **Ensemble** | **0.9593** | **0.9984** |

---

## System dependencies

Install these at the OS level **before** the Python packages:

- **Python 3.12**
- **FFmpeg** (`ffmpeg` + `ffprobe`) — audio/video decoding and segmentation
- **eSpeak NG** — phonemizer backend for the wav2vec 2.0 phoneme recogniser
- **libsndfile** — audio I/O for librosa
- **CUDA 12.6** GPU recommended (CPU works, slower)

```bash
# Debian / Ubuntu / WSL
sudo apt install ffmpeg espeak-ng libsndfile1
# Arch
sudo pacman -S ffmpeg espeak-ng libsndfile
# macOS (Homebrew)
brew install ffmpeg espeak-ng libsndfile
```

On Linux/macOS eSpeak NG is found automatically on `PATH`. On **Windows**, install
[eSpeak NG](https://github.com/espeak-ng/espeak-ng/releases) and
[FFmpeg](https://ffmpeg.org/) manually, then point the phonemizer backend at them via the
`PHONEMIZER_ESPEAK_LIBRARY` (path to `libespeak-ng.dll`) and `PHONEMIZER_ESPEAK_PATH`
(path to `espeak-ng.exe`) environment variables. If eSpeak NG is installed at the default
`C:\Program Files\eSpeak NG\`, it is picked up automatically and no env vars are needed.

Python dependencies are installed per path: a lightweight CPU set for **A. Try the
pretrained detector**, or the exact pinned GPU set for **B. Reproduce from scratch**.

Two third-party **models** download automatically on first use (not committed):
- [**MediaPipe Face Landmarker**](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) (`face_landmarker.task`) — facial landmarks + blendshapes.
- [**wav2vec 2.0** phoneme recogniser](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) (`facebook/wav2vec2-lv-60-espeak-cv-ft`, [paper](https://arxiv.org/abs/2109.11680)) — phoneme alignment.

---

## A. Try the pretrained detector

Install the **System dependencies** above (`ffmpeg`, `espeak-ng`, `libsndfile`), then:

```bash
git clone https://github.com/vlaght/dslp-ensemble-study.git
cd dslp-ensemble-study

# CPU-only Python deps — installs the CPU build of PyTorch (no ~2.8 GB CUDA download)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -U huggingface_hub

# pretrained weights from the Hugging Face Hub -> trained/final/
hf download vl4gh7/dslp-ensemble --local-dir trained/final

# run on the bundled sample clip (or pass your own short mp4 with a speaking face)
python -u cli/classify_ensemble.py --video assets/sample.mp4   # full detector  -> P(fake)
python -u cli/classify_dslp.py     --video assets/sample.mp4   # DSLP component only
python -u cli/classify_tafreq.py   --video assets/sample.mp4   # TAFreq component only
python -u cli/classify_tanoise.py  --video assets/sample.mp4   # TANoise component only
```

The ensemble CLI runs live feature extraction (MediaPipe landmarks, wav2vec 2.0 phonemes,
MFCC, DCT/STFT, noise residuals), runs the three models, and averages their sigmoid
probabilities. Needs a visible speaking face and audible speech.

---

## B. Reproduce from scratch (modelling)

Full pipeline: raw videos → feature databases → analysis / cross-validation / training.

### B.0 Setup (full GPU environment)

```bash
git clone https://github.com/vlaght/dslp-ensemble-study.git
cd dslp-ensemble-study
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-full.txt          # exact pinned versions used in the paper (CUDA 12.6)
```

### B.1 Datasets

The study combines four sources of talking-face video (licenses are the providers' — most
are research-only / non-commercial; obtain each from its provider):

- **FakeAVCeleb** (Khalid et al., NeurIPS 2021 D&B) —
  https://github.com/DASH-Lab/FakeAVCeleb
  Audio-visual multimodal deepfake dataset: real celebrity clips plus fakes spanning all
  real/fake video×audio combinations. The primary forgery source.
- **DeepSpeak v2** (Barrington et al., 2025) —
  https://huggingface.co/datasets/faridlab/deepspeak_v2
  100+ hours of authentic and deepfake audiovisual content from consenting participants,
  generated with 14 video-synthesis and 3 voice-cloning engines. Adds modern generators.
- **TalkVid-Bench** (Chen et al., 2025) —
  https://github.com/FreedomIntelligence/TalkVid
  A stratified 500-clip evaluation subset of TalkVid (a large, demographically diverse
  talking-head dataset). Used here as a source of authentic, diverse talking-head clips.
- **Augmented** authentic clips — real YouTube talking-head videos downloaded from the URL
  list in `feat-extraction/videos.txt` (subject to YouTube Terms of Service; next step).

Place raw videos under `raw_videos/` (gitignored). Extracted feature databases (SQLite)
live in `data/`: `dataset.db` (landmark/MFCC/phoneme), `freq_features.db`,
`noise_features.db`, plus the DeepSpeak counterparts `deepspeakv2.db`,
`freq_features_dsv2.db`, `noise_features_dsv2.db` (all large, gitignored).

### B.2 Download the augmented YouTube clips

```bash
python -u feat-extraction/download_yt_videos.py     # URL list: feat-extraction/videos.txt
                                                    # --urls / --output-dir to override
```

### B.3 Pre-processing

```bash
python -u feat-extraction/process_raw_videos.py     # segment into <=10s clips, drop <4s
python -u feat-extraction/load_deepspeach.py        # fetch DeepSpeak v2 (HuggingFace)
```

### B.4 Feature extraction

```bash
# Visual (52 MediaPipe blendshapes + 6 geometry) + 41 audio (MFCC/Δ/Δ²/mag/energy)
# + phoneme alignment  ->  data/dataset.db
python -u feat-extraction/extract_features.py

# 36 frequency features (14 visual DCT + 22 audio STFT)  ->  data/freq_features.db
python -u feat-extraction/extract_freq_features.py --mode ext

# 19 noise-residual features (Laplacian / DoG / quadrant)  ->  data/noise_features.db
python -u feat-extraction/extract_noise_features.py --mode ext
```

Use `--mode dsv2` for the DeepSpeak v2 databases. Every script takes `--help` for path
overrides (`--dataset-root`, `--output-db`, `--db-path`, …).

### B.5 Per-hypothesis analysis (figures + tables)

```bash
# RQ1 (phoneme informativeness)
python -u analysis/rq1/rh1_1.py ; python -u analysis/rq1/rh1_2.py ; python -u analysis/rq1/rh1_3.py
python -u analysis/rq1/rq1_summary.py
# RQ2 (MAR variance / closure / approximants)
python -u analysis/rq2/rh2_1.py ; python -u analysis/rq2/rh2_2.py ; python -u analysis/rq2/rh2_3.py
python -u analysis/rq2/rq2_summary.py
# RQ3 (architecture comparison; rh3_3_nofac = FakeAVCeleb-excluded re-run)
python -u analysis/rq3/rh3_1.py ; python -u analysis/rq3/rh3_2.py
python -u analysis/rq3/rh3_3.py ; python -u analysis/rq3/rh3_3_nofac.py
python -u analysis/rq3/rq3_summary.py
```

Supporting analyses: `analysis/dataset_analysis.py`, `analysis/phoneme_analysis.py`,
`analysis/coarticulation_flatten.py`, `analysis/annotate_landmarks.py`.

### B.6 Cross-validation (the ensemble result)

```bash
python -u modelling/cv_ensemble.py            # 10-fold CV: DSLP + TAFreq + TANoise + ensemble
```
Produces the per-configuration table and ensemble AUCs reported in the paper
(writes `.tmp/dslp_ensemble_*.json`).

### B.7 Train + save the final models

```bash
python -u modelling/train_and_save_final.py                  # default: --dataset ALL
python -u modelling/train_and_save_final.py --dataset EXT    # FakeAVCeleb + augmented + TalkVid
python -u modelling/train_and_save_final.py --dataset DSV2   # DeepSpeak v2 only
```
Trains the three components (90% train / 10% held out for early stopping) and writes
`trained/final/`: `dslp.pth`, `tafreq.pth`, `tanoise.pth`, `*_artifacts.pkl`,
`ensemble_manifest.json`. These are the weights published on Hugging Face.

### B.8 Ablations / component analysis

```bash
python -u modelling/dslp_ablation.py          # leave-one-out feature pruning (-> 58-feature DSLP)
python -u analysis/error_correlation.py       # pairwise error-correlation (ensemble diversity)
```

---

## Repository layout

```
feat-extraction/   raw-video -> feature-DB extraction (visual, freq, noise, phonemes, YT download)
modelling/         cv_ensemble.py (CV), train_and_save_final.py (train+save), dslp_arch.py,
                   dslp_ablation.py
analysis/          rq1/ rq2/ rq3/ per-hypothesis scripts + summaries; dataset/phoneme/error analyses
cli/               per-component + ensemble inference
trained/final/     final model weights (downloaded from Hugging Face)
data/              feature databases incl. dataset.db (gitignored)
```

## Reproducibility notes

Full architecture, training, initialisation, normalisation, hyperparameter policy, CV
arrangement, hardware, and exact software versions are in the paper's supplement (Appendix
C) and `requirements-full.txt`. No normalisation layers; PyTorch-default initialisation;
per-fold z-score feature standardisation; 10-fold stratified CV (random_state=42); single
workstation (Ryzen 5 7600X, RTX 4070 Ti Super 16 GB).

## Citation

If you use this code or the pretrained models, please cite the repository (and the paper
once published). Please also cite the source datasets (see the Hugging Face model card).

**BibTeX**
```bibtex
@software{boiko_dslp_ensemble_2026,
  author = {Boiko, Vladislav},
  title  = {Phoneme-Based Audio-Visual Face-Forgery Detection (DSLP Ensemble)},
  year   = {2026},
  url    = {https://github.com/vlaght/dslp-ensemble-study}
}
```

**APA**
> Boiko, V. (2026). *Phoneme-Based Audio-Visual Face-Forgery Detection (DSLP Ensemble)*
> [Computer software]. https://github.com/vlaght/dslp-ensemble-study

**IEEE**
> V. Boiko, "Phoneme-Based Audio-Visual Face-Forgery Detection (DSLP Ensemble)," 2026.
> [Online]. Available: https://github.com/vlaght/dslp-ensemble-study
