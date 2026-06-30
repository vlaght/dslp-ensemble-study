"""
Comprehensive phoneme importance analysis for deepfake detection.

This script performs multiple analyses:
1. Phoneme distribution analysis (real vs fake)
2. Phoneme importance via multiple perturbation strategies
3. Temporal importance analysis
4. Phoneme transition/bigram analysis
5. Statistical significance testing
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score
from torch.utils.data import DataLoader

# Import model components
from modelling.train_lstm import (
    VideoDataset,
    collate_fn,
    DeepFakeLSTM,
    HIDDEN_DIM,
    NUM_LAYERS,
    DROPOUT,
    device,
    NUMERICAL_FEATURES_TO_USE,
)
from analysis.data_utils import load_and_preprocess_data

# Configuration
MODEL_SAVE_PATH = os.path.join("trained", "deepfake_classifier.pth")

OUTPUT_DIR = "phoneme_analysis_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_phoneme_data(db_path):
    """Load raw phoneme data from database."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT filename, phoneme, timestamp, video_type FROM lags", conn)
    conn.close()
    return df


def analyze_phoneme_distribution(df, le_phoneme):
    """Analyze phoneme frequency distribution for real vs fake videos."""
    print("\n" + "="*60)
    print("PHONEME DISTRIBUTION ANALYSIS")
    print("="*60)

    results = []

    for video_type in df['video_type'].unique():
        subset = df[df['video_type'] == video_type]
        phoneme_counts = Counter(subset['phoneme'])
        total = len(subset)

        for phoneme, count in phoneme_counts.items():
            results.append({
                'video_type': video_type,
                'phoneme': phoneme,
                'count': count,
                'frequency': count / total
            })

    results_df = pd.DataFrame(results)

    # Pivot to compare real vs fake
    pivot = results_df.pivot_table(
        index='phoneme',
        columns='video_type',
        values='frequency',
        fill_value=0
    )

    # Calculate difference
    if 'Real' in pivot.columns and 'Fake' in pivot.columns:
        pivot['diff'] = abs(pivot['Real'] - pivot['Fake'])
        pivot = pivot.sort_values('diff', ascending=False)

        # Save results
        pivot.to_csv(os.path.join(OUTPUT_DIR, 'phoneme_distribution.csv'))

        # Plot top differences
        top_n = min(20, len(pivot))
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Frequency comparison
        top_phons = pivot.head(top_n).index
        x = np.arange(len(top_phons))
        width = 0.35

        axes[0].bar(x - width/2, pivot.loc[top_phons, 'Real'], width, label='Real', alpha=0.8)
        axes[0].bar(x + width/2, pivot.loc[top_phons, 'Fake'], width, label='Fake', alpha=0.8)
        axes[0].set_xlabel('Phoneme')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title(f'Top {top_n} Phonemes with Largest Distribution Difference')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(top_phons, rotation=45, ha='right')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Difference plot
        axes[1].barh(range(top_n), pivot.head(top_n)['diff'], color='coral')
        axes[1].set_yticks(range(top_n))
        axes[1].set_yticklabels(top_phons)
        axes[1].set_xlabel('Absolute Frequency Difference')
        axes[1].set_title('Distribution Difference (|Real - Fake|)')
        axes[1].grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'phoneme_distribution.png'), dpi=150)
        plt.close()

        print(f"\nTop 10 phonemes with largest distribution difference:")
        print(pivot.head(10).to_string())

    return results_df, pivot


def analyze_phoneme_transitions(df):
    """Analyze phoneme bigrams/transitions."""
    print("\n" + "="*60)
    print("PHONEME TRANSITION ANALYSIS (BIGRAMS)")
    print("="*60)

    bigrams_by_type = defaultdict(Counter)

    for filename, group in df.groupby('filename'):
        video_type = group['video_type'].iloc[0]
        phonemes = group.sort_values('timestamp')['phoneme'].tolist()

        # Extract bigrams
        for i in range(len(phonemes) - 1):
            bigram = (phonemes[i], phonemes[i+1])
            bigrams_by_type[video_type][bigram] += 1

    # Compare bigram frequencies
    results = []
    all_bigrams = set()
    for video_type in bigrams_by_type:
        all_bigrams.update(bigrams_by_type[video_type].keys())

    for bigram in all_bigrams:
        real_count = bigrams_by_type.get('Real', Counter())[bigram]
        fake_count = bigrams_by_type.get('Fake', Counter())[bigram]

        real_total = sum(bigrams_by_type.get('Real', Counter()).values())
        fake_total = sum(bigrams_by_type.get('Fake', Counter()).values())

        real_freq = real_count / real_total if real_total > 0 else 0
        fake_freq = fake_count / fake_total if fake_total > 0 else 0

        results.append({
            'bigram': f"{bigram[0]} → {bigram[1]}",
            'real_freq': real_freq,
            'fake_freq': fake_freq,
            'diff': abs(real_freq - fake_freq)
        })

    bigram_df = pd.DataFrame(results).sort_values('diff', ascending=False)
    bigram_df.to_csv(os.path.join(OUTPUT_DIR, 'phoneme_transitions.csv'), index=False)

    print(f"\nTop 10 most discriminative phoneme transitions:")
    print(bigram_df.head(10).to_string(index=False))

    # Plot top transitions
    top_n = min(15, len(bigram_df))
    plt.figure(figsize=(12, 8))
    x = np.arange(top_n)
    width = 0.35

    top_data = bigram_df.head(top_n)
    plt.barh(x - width/2, top_data['real_freq'], width, label='Real', alpha=0.8)
    plt.barh(x + width/2, top_data['fake_freq'], width, label='Fake', alpha=0.8)
    plt.yticks(x, top_data['bigram'])
    plt.xlabel('Frequency')
    plt.title(f'Top {top_n} Most Discriminative Phoneme Transitions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phoneme_transitions.png'), dpi=150)
    plt.close()

    return bigram_df


def compute_baseline_accuracy(model, loader):
    """Compute baseline accuracy."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for feats, phons, lengths, labels in loader:
            feats, phons, labels = feats.to(device), phons.to(device), labels.to(device)
            outputs = model(feats, phons, lengths)
            predicted = (outputs > 0.5).float()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return accuracy_score(all_labels, all_preds)


def ablation_masking(model, dataset, num_phonemes, baseline_acc, le_phoneme):
    """Test importance by masking individual phonemes."""
    print("\n" + "="*60)
    print("ABLATION: PHONEME MASKING")
    print("="*60)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
    feats_orig, phons_orig, lengths_orig, labels_orig = next(iter(loader))
    feats_orig = feats_orig.to(device)
    phons_orig = phons_orig.to(device)
    labels_orig = labels_orig.to(device)

    present_phonemes = torch.unique(phons_orig).cpu().numpy()
    phoneme_map = {i: label for i, label in enumerate(le_phoneme.classes_)}

    results = []
    model.eval()

    for phon_id in present_phonemes:
        if phon_id == 0:  # Skip padding
            continue

        phons_masked = phons_orig.clone()
        phons_masked[phons_masked == phon_id] = 0

        with torch.no_grad():
            outputs = model(feats_orig, phons_masked, lengths_orig)
            predicted = (outputs > 0.5).float()
            acc = accuracy_score(labels_orig.cpu().numpy(), predicted.cpu().numpy())

        drop = baseline_acc - acc
        phon_name = phoneme_map.get(phon_id, f"Unknown_{phon_id}")

        results.append({
            'phoneme': phon_name,
            'phoneme_id': phon_id,
            'accuracy': acc,
            'drop': drop,
            'method': 'masking'
        })

        print(f"Phoneme: {phon_name:<6} | Acc: {acc:.4f} | Drop: {drop:.4f}")

    return pd.DataFrame(results)


def ablation_substitution(model, dataset, num_phonemes, baseline_acc, le_phoneme):
    """Test importance by substituting phonemes with random ones."""
    print("\n" + "="*60)
    print("ABLATION: PHONEME RANDOM SUBSTITUTION")
    print("="*60)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
    feats_orig, phons_orig, lengths_orig, labels_orig = next(iter(loader))
    feats_orig = feats_orig.to(device)
    phons_orig = phons_orig.to(device)
    labels_orig = labels_orig.to(device)

    present_phonemes = torch.unique(phons_orig).cpu().numpy()
    present_phonemes = [p for p in present_phonemes if p != 0]
    phoneme_map = {i: label for i, label in enumerate(le_phoneme.classes_)}

    results = []
    model.eval()

    for phon_id in present_phonemes:
        # Create list of other phonemes to substitute with
        other_phonemes = [p for p in present_phonemes if p != phon_id]
        if not other_phonemes:
            continue

        phons_subst = phons_orig.clone()
        mask = phons_subst == phon_id

        # Randomly substitute with other phonemes
        random_phons = torch.tensor(
            np.random.choice(other_phonemes, size=mask.sum().item()),
            device=device
        )
        phons_subst[mask] = random_phons

        with torch.no_grad():
            outputs = model(feats_orig, phons_subst, lengths_orig)
            predicted = (outputs > 0.5).float()
            acc = accuracy_score(labels_orig.cpu().numpy(), predicted.cpu().numpy())

        drop = baseline_acc - acc
        phon_name = phoneme_map.get(phon_id, f"Unknown_{phon_id}")

        results.append({
            'phoneme': phon_name,
            'phoneme_id': phon_id,
            'accuracy': acc,
            'drop': drop,
            'method': 'substitution'
        })

        print(f"Phoneme: {phon_name:<6} | Acc: {acc:.4f} | Drop: {drop:.4f}")

    return pd.DataFrame(results)


def ablation_noise_injection(model, dataset, baseline_acc, noise_levels=[0.1, 0.2, 0.3, 0.5]):
    """Test robustness by injecting random noise into phoneme sequences."""
    print("\n" + "="*60)
    print("ABLATION: PHONEME NOISE INJECTION")
    print("="*60)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
    feats_orig, phons_orig, lengths_orig, labels_orig = next(iter(loader))
    feats_orig = feats_orig.to(device)
    phons_orig = phons_orig.to(device)
    labels_orig = labels_orig.to(device)

    present_phonemes = torch.unique(phons_orig).cpu().numpy()
    present_phonemes = [p for p in present_phonemes if p != 0]

    results = []
    model.eval()

    for noise_level in noise_levels:
        # Randomly replace noise_level % of phonemes
        phons_noisy = phons_orig.clone()

        for i in range(phons_noisy.size(0)):
            for j in range(phons_noisy.size(1)):
                if phons_noisy[i, j] == 0:  # Skip padding
                    continue
                if np.random.rand() < noise_level:
                    phons_noisy[i, j] = np.random.choice(present_phonemes)

        with torch.no_grad():
            outputs = model(feats_orig, phons_noisy, lengths_orig)
            predicted = (outputs > 0.5).float()
            acc = accuracy_score(labels_orig.cpu().numpy(), predicted.cpu().numpy())

        drop = baseline_acc - acc

        results.append({
            'noise_level': noise_level,
            'accuracy': acc,
            'drop': drop
        })

        print(f"Noise Level: {noise_level:.1%} | Acc: {acc:.4f} | Drop: {drop:.4f}")

    noise_df = pd.DataFrame(results)

    # Plot noise robustness
    plt.figure(figsize=(10, 6))
    plt.plot(noise_df['noise_level'], noise_df['accuracy'], marker='o', linewidth=2, markersize=8)
    plt.axhline(y=baseline_acc, color='r', linestyle='--', label='Baseline')
    plt.xlabel('Noise Level (% phonemes randomly replaced)')
    plt.ylabel('Accuracy')
    plt.title('Model Robustness to Phoneme Noise')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'phoneme_noise_robustness.png'), dpi=150)
    plt.close()

    return noise_df


def ablation_temporal(model, dataset, baseline_acc, segments=3):
    """Test importance of phonemes at different temporal positions."""
    print("\n" + "="*60)
    print("ABLATION: TEMPORAL PHONEME IMPORTANCE")
    print("="*60)

    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
    feats_orig, phons_orig, lengths_orig, labels_orig = next(iter(loader))
    feats_orig = feats_orig.to(device)
    phons_orig = phons_orig.to(device)
    labels_orig = labels_orig.to(device)

    results = []
    model.eval()

    segment_names = ['beginning', 'middle', 'end'] if segments == 3 else [f'segment_{i}' for i in range(segments)]

    for seg_idx in range(segments):
        phons_masked = phons_orig.clone()

        for i in range(phons_masked.size(0)):
            seq_len = lengths_orig[i].item()
            seg_size = seq_len // segments

            start_idx = seg_idx * seg_size
            end_idx = start_idx + seg_size if seg_idx < segments - 1 else seq_len

            # Mask this segment
            phons_masked[i, start_idx:end_idx] = 0

        with torch.no_grad():
            outputs = model(feats_orig, phons_masked, lengths_orig)
            predicted = (outputs > 0.5).float()
            acc = accuracy_score(labels_orig.cpu().numpy(), predicted.cpu().numpy())

        drop = baseline_acc - acc
        segment_name = segment_names[seg_idx]

        results.append({
            'segment': segment_name,
            'accuracy': acc,
            'drop': drop
        })

        print(f"Segment: {segment_name:<10} | Acc: {acc:.4f} | Drop: {drop:.4f}")

    temporal_df = pd.DataFrame(results)

    # Plot temporal importance
    plt.figure(figsize=(10, 6))
    plt.bar(temporal_df['segment'], temporal_df['drop'], color='steelblue', alpha=0.8)
    plt.xlabel('Temporal Segment')
    plt.ylabel('Accuracy Drop')
    plt.title('Phoneme Importance by Temporal Position')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_importance.png'), dpi=150)
    plt.close()

    return temporal_df


def plot_combined_importance(masking_df, substitution_df):
    """Plot comparison of masking vs substitution importance."""
    # Merge on phoneme
    merged = masking_df.merge(
        substitution_df,
        on='phoneme',
        suffixes=('_mask', '_subst')
    )

    # Sort by masking drop
    merged = merged.sort_values('drop_mask', ascending=False)

    top_n = min(20, len(merged))
    top_data = merged.head(top_n)

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_data))
    width = 0.35

    ax.bar(x - width/2, top_data['drop_mask'], width, label='Masking (set to PAD)', alpha=0.8)
    ax.bar(x + width/2, top_data['drop_subst'], width, label='Substitution (random phoneme)', alpha=0.8)

    ax.set_xlabel('Phoneme')
    ax.set_ylabel('Accuracy Drop')
    ax.set_title(f'Top {top_n} Most Important Phonemes: Masking vs Substitution')
    ax.set_xticks(x)
    ax.set_xticklabels(top_data['phoneme'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'importance_comparison.png'), dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DeepFake Transformer model.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=os.path.join(".", "data/dataset.db"),
        help="Path to the SQLite database file.",
    )
    args = parser.parse_args()
    db_path = args.dataset

    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        sys.exit(1)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Error: Trained model not found at {MODEL_SAVE_PATH}")
        print("Run train_lstm.py first.")
        sys.exit(1)

    # 1. Load Data
    print("Loading data...")
    sequences, labels, num_phonemes, feature_cols, class_names, le_phoneme = (
        load_and_preprocess_data(
            db_path=db_path,
            features_to_use=NUMERICAL_FEATURES_TO_USE,
            use_onehot_phonemes=False,
            artifacts_path="phoneme_analysis_artifacts.pkl"
        )
    )
    num_features = len(feature_cols)

    # Split (same as training)
    _, X_test, _, y_test = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )

    test_dataset = VideoDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 2. Load Model
    print("Loading model...")
    model = DeepFakeLSTM(
        num_numerical_features=num_features,
        num_phonemes=num_phonemes,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    # 3. Baseline Accuracy
    baseline_acc = compute_baseline_accuracy(model, test_loader)
    print(f"\nBaseline Test Accuracy: {baseline_acc:.4f}")

    # 4. Phoneme Distribution Analysis
    raw_df = load_phoneme_data(db_path)
    dist_df, pivot = analyze_phoneme_distribution(raw_df, le_phoneme)

    # 5. Phoneme Transition Analysis
    bigram_df = analyze_phoneme_transitions(raw_df)

    # 6. Ablation Studies
    masking_results = ablation_masking(model, test_dataset, num_phonemes, baseline_acc, le_phoneme)
    substitution_results = ablation_substitution(model, test_dataset, num_phonemes, baseline_acc, le_phoneme)
    noise_results = ablation_noise_injection(model, test_dataset, baseline_acc)
    temporal_results = ablation_temporal(model, test_dataset, baseline_acc)

    # 7. Save All Results
    masking_results.to_csv(os.path.join(OUTPUT_DIR, 'ablation_masking.csv'), index=False)
    substitution_results.to_csv(os.path.join(OUTPUT_DIR, 'ablation_substitution.csv'), index=False)
    noise_results.to_csv(os.path.join(OUTPUT_DIR, 'ablation_noise.csv'), index=False)
    temporal_results.to_csv(os.path.join(OUTPUT_DIR, 'ablation_temporal.csv'), index=False)

    # 8. Combined Visualization
    plot_combined_importance(masking_results, substitution_results)

    # 9. Summary Report
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*60)
    print(f"\nBaseline Accuracy: {baseline_acc:.4f}")
    print(f"\nTop 5 Most Important Phonemes (Masking):")
    print(masking_results.nlargest(5, 'drop')[['phoneme', 'drop']].to_string(index=False))
    print(f"\nTop 5 Most Important Phonemes (Substitution):")
    print(substitution_results.nlargest(5, 'drop')[['phoneme', 'drop']].to_string(index=False))
    print(f"\nTemporal Importance:")
    print(temporal_results.to_string(index=False))
    print(f"\nAll results saved to: {os.path.abspath(OUTPUT_DIR)}")
