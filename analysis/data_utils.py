"""
Common data loading and preprocessing utilities for deepfake detection models.
"""

import os
import sqlite3
import struct
import pandas as pd
import numpy as np
import joblib
import torch
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import StandardScaler, LabelEncoder



def load_and_preprocess_data(
    db_path,
    features_to_use,
    max_seq_len=None,
    max_samples=None,
    use_onehot_phonemes=False,
    artifacts_path=None,
    chunksize=50000,
    phonemes_to_include=None,
    classes_ratio=1.0,
    table_name="lags",
):
    """
    Load and preprocess data from SQLite database for deepfake detection.

    Args:
        db_path: Path to SQLite database file
        features_to_use: Set or list of feature column names to use
        max_seq_len: Maximum sequence length (for DTW, truncates long sequences)
        max_samples: Maximum number of samples to use (for DTW performance)
        use_onehot_phonemes: If True, use one-hot encoding for phonemes (DTW),
                            if False, use LabelEncoder (deep learning models)
        artifacts_path: Path to save model artifacts (scaler, encoders, etc.)
        chunksize: Number of rows to read per chunk (default: 50000)
        phonemes_to_include: Set of phonemes to include (default: {'n','b','p','m','f','v','w','u','o'})
        classes_ratio: Ratio of majority class samples to minority class samples (default: 1.0)
                      E.g., 1.0 = balanced, 2.0 = 2x more majority samples than minority
        table_name: Name of the table to query (default: 'lags'; use 'instances' for
                   phonemes_coarticulation.db)

    Returns:
        For deep learning models (use_onehot_phonemes=False):
            - sequences: List of (features, phonemes) tuples
            - labels: Array of encoded labels
            - num_phonemes: Number of unique phonemes
            - feature_cols: List of feature column names
            - class_names: List of class names
            - le_phoneme: LabelEncoder for phonemes

        For DTW models (use_onehot_phonemes=True):
            - sequences: List of feature arrays (phonemes one-hot encoded)
            - labels: Array of encoded labels
            - all_feature_cols: List of all feature column names (including phoneme dummies)
            - class_names: List of class names
    """
    print("Loading data from database...")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database not found at {db_path}")

    # Default phonemes to include
    if phonemes_to_include is None:
        phonemes_to_include = {"n", "b", "p", "m", "f", "v", "w", "u", "o"}

    conn = sqlite3.connect(db_path)

    # Build SQL query with optional phoneme filter
    if phonemes_to_include == "all":
        query = f"SELECT * FROM {table_name}"
        print("Filtering phonemes: all")
    else:
        phoneme_list = "', '".join(phonemes_to_include)
        query = f"SELECT * FROM {table_name} WHERE phoneme IN ('{phoneme_list}')"
        print(f"Filtering phonemes: {phonemes_to_include}")

    # Load table in chunks to save memory
    print("Reading table from database in chunks...")
    try:
        chunks = pd.read_sql_query(query, conn, chunksize=chunksize)
    except Exception as e:
        conn.close()
        raise ValueError(f"Error reading from database: {e}")

    processed_chunks = []
    total_rows = 0
    chunk_num = 0

    for chunk_df in chunks:
        chunk_num += 1
        total_rows += len(chunk_df)

        if not chunk_df.empty:
            processed_chunks.append(chunk_df)

        if chunk_num % 10 == 0:
            print(f"  Processed {total_rows} rows ({len(processed_chunks)} chunks)...")

    conn.close()

    if not processed_chunks:
        raise ValueError("Database is empty or all data was filtered out.")

    print(f"Combining {len(processed_chunks)} chunks...")
    df = pd.concat(processed_chunks, ignore_index=True)

    print(f"Final dataset: {len(df)} rows")

    if df.empty:
        raise ValueError(
            "No valid data remaining after filtering. Check your database."
        )

    # 1. Encode Labels (Real vs Fake)
    le_label = LabelEncoder()
    df["label_encoded"] = le_label.fit_transform(df["video_type"])
    print(
        f"Label classes: {dict(zip(le_label.classes_, le_label.transform(le_label.classes_)))}"
    )

    # 2. Handle Phonemes
    if use_onehot_phonemes:
        # One-hot encode phonemes (for DTW)
        print("One-hot encoding phonemes...")
        phoneme_dummies = pd.get_dummies(df["phoneme"], prefix="ph")
        df = pd.concat([df, phoneme_dummies], axis=1)
        phoneme_cols = phoneme_dummies.columns.tolist()
        le_phoneme = None
        num_phonemes = len(phoneme_cols)
    else:
        # Label encode phonemes (for deep learning)
        le_phoneme = LabelEncoder()
        df["phoneme_encoded"] = le_phoneme.fit_transform(df["phoneme"])
        num_phonemes = len(le_phoneme.classes_)
        phoneme_cols = []
        print(f"Number of unique phonemes: {num_phonemes}")

    # 3. Identify Feature Columns
    target_col = "video_type"
    exclude_cols = {
        "filename",
        target_col,
        "phoneme",
        "frame_position",
        "timestamp",
        "dataset",
        "label_encoded",
        "phoneme_encoded",
    }

    # Numerical features
    # features_to_use=None means "use every numeric column not in exclude_cols"
    if features_to_use is None:
        numerical_feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]
    else:
        numerical_feature_cols = [
            c for c in df.columns if c in features_to_use and c not in exclude_cols
        ]

    if use_onehot_phonemes:
        # For DTW: combine numerical + phoneme dummy columns
        all_feature_cols = numerical_feature_cols + phoneme_cols
        print(
            f"Using {len(numerical_feature_cols)} numerical features and {len(phoneme_cols)} phoneme features."
        )
    else:
        # For deep learning: only numerical features (phonemes handled separately)
        all_feature_cols = numerical_feature_cols
        print(f"Using {len(all_feature_cols)} numerical features: {all_feature_cols}")

    # 4. Normalize Numerical Features
    scaler = StandardScaler()
    df[numerical_feature_cols] = scaler.fit_transform(df[numerical_feature_cols])

    # 5. Group by Video to create sequences
    df = df.sort_values(by=["filename", "timestamp"])
    grouped = df.groupby("filename")

    sequences = []
    labels = []

    print("Grouping data into sequences...")
    for filename, group in grouped:
        if use_onehot_phonemes:
            # DTW: single feature array including phonemes
            feats = group[all_feature_cols].values

            # Truncate if too long
            if max_seq_len and len(feats) > max_seq_len:
                feats = feats[:max_seq_len]

            sequences.append(feats)
        else:
            # Deep learning: separate features and phonemes
            feats = group[numerical_feature_cols].values.astype(np.float32)
            phons = group["phoneme_encoded"].values.astype(np.int64)
            sequences.append((feats, phons))

        labels.append(group["label_encoded"].iloc[0])

    # 6. Balance Data with configurable class ratio
    labels_arr = np.array(labels)
    unique_classes, counts = np.unique(labels_arr, return_counts=True)
    print(f"Class distribution before balancing: {dict(zip(unique_classes, counts))}")

    if classes_ratio is None:
        # Use all rows — no balancing
        print("classes_ratio=None: using all samples without balancing")
        balanced_sequences = sequences
        balanced_labels = labels_arr
    else:
        min_count = counts.min()
        min_class = unique_classes[np.argmin(counts)]
        maj_class = unique_classes[np.argmax(counts)]

        print(f"Minority class: {min_class} ({min_count} samples)")
        print(f"Majority class: {maj_class} ({counts.max()} samples)")

        # Calculate number of samples to take from majority class
        maj_count = int(min_count * classes_ratio)
        print(
            f"Sampling with classes_ratio={classes_ratio}: {min_count} minority + {maj_count} majority"
        )

        balanced_indices = []
        for cls in unique_classes:
            cls_indices = np.where(labels_arr == cls)[0]
            if cls == min_class:
                sampled_indices = np.random.choice(cls_indices, min_count, replace=False)
            else:
                sampled_indices = np.random.choice(cls_indices, maj_count, replace=False)
            balanced_indices.extend(sampled_indices)

        np.random.shuffle(balanced_indices)
        balanced_sequences = [sequences[i] for i in balanced_indices]
        balanced_labels = labels_arr[balanced_indices]

        # Print final distribution
        final_unique, final_counts = np.unique(balanced_labels, return_counts=True)
        print(
            f"Class distribution after balancing: {dict(zip(final_unique, final_counts))}"
        )

    # 7. Downsample further if needed (for DTW performance)
    if max_samples and len(balanced_sequences) > max_samples:
        print(
            f"Downsampling from {len(balanced_sequences)} to {max_samples} random samples."
        )
        indices = np.random.choice(len(balanced_sequences), max_samples, replace=False)
        balanced_sequences = [balanced_sequences[i] for i in indices]
        balanced_labels = balanced_labels[indices]

    # 8. Save artifacts if path provided
    if artifacts_path:
        artifacts = {
            "scaler": scaler,
            "le_label": le_label,
            "feature_cols": all_feature_cols
            if use_onehot_phonemes
            else numerical_feature_cols,
            "num_phonemes": num_phonemes,
        }

        if use_onehot_phonemes:
            artifacts["phoneme_cols"] = phoneme_cols
            artifacts["max_seq_len"] = max_seq_len
        else:
            artifacts["le_phoneme"] = le_phoneme

        joblib.dump(artifacts, artifacts_path)
        print(f"Saved model artifacts to {artifacts_path}")

    # 9. Return appropriate data based on model type
    class_names = list(le_label.classes_)

    if use_onehot_phonemes:
        # DTW model
        return balanced_sequences, balanced_labels, all_feature_cols, class_names
    else:
        # Deep learning models
        return (
            balanced_sequences,
            balanced_labels,
            num_phonemes,
            numerical_feature_cols,
            class_names,
            le_phoneme,
        )


# --- Collate Functions for DataLoader ---




