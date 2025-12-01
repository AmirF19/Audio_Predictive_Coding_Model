"""
Data loading functions for the PC model.
Handles lexicon, semantic features, and phoneme vectors.
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple

from config import (
    WORD_LIST_FILE,
    SEMANTIC_JSON_FILE,
    PHONEME_VECTOR_DIR,
    N_PHONEME_SLOTS,
    PHONEME_VECTOR_DIM,
    INPUT_DIM,
    DEVICE,
)

_debug_printed = False


def load_lexicon() -> List[str]:
    """Load word list from CSV."""
    if not WORD_LIST_FILE.exists():
        print(f"Lexicon not found: {WORD_LIST_FILE}")
        return []

    df = pd.read_csv(WORD_LIST_FILE)
    
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
    
    return sorted(list(set([w for w in words if len(w) > 0])))


def load_semantic_matrix(lexicon: List[str]) -> Tuple[torch.Tensor, List[str]]:
    """
    Build V_SL matrix from Llama-generated semantic features.
    Returns (semantic_dim x lexical_dim) binary matrix.
    """
    if not SEMANTIC_JSON_FILE.exists():
        print(f"Semantic file not found: {SEMANTIC_JSON_FILE}")
        return torch.empty(0), []

    with open(SEMANTIC_JSON_FILE, 'r') as f:
        data = json.load(f)

    # Collect all unique features
    all_features = set()
    for features in data.values():
        all_features.update(features)
    feature_list = sorted(list(all_features))
    
    # Build index mappings
    feat_to_idx = {f: i for i, f in enumerate(feature_list)}
    word_to_idx = {w: i for i, w in enumerate(lexicon)}

    # Construct matrix
    V_SL = np.zeros((len(feature_list), len(lexicon)), dtype=np.float32)

    for word, features in data.items():
        word = word.strip().lower()
        if word in word_to_idx:
            w_idx = word_to_idx[word]
            for feat in features:
                if feat in feat_to_idx:
                    V_SL[feat_to_idx[feat], w_idx] = 1.0

    print(f"Semantic matrix: {len(feature_list)} features x {len(lexicon)} words")
    return torch.from_numpy(V_SL).to(DEVICE), feature_list


def load_phoneme_input(word: str, condition: str) -> torch.Tensor:
    """
    Load Wav2Vec phoneme vector for a word.
    
    Args:
        word: Target word
        condition: 'clear' or 'noisy'
    
    Returns:
        Flattened, normalized phoneme vector (INPUT_DIM,)
    """
    global _debug_printed
    
    word = word.strip().lower()
    
    if condition == 'clear':
        filename = f"{word}_pc_phoneme_input.npy"
    elif condition == 'noisy':
        filename = f"{word}_noisy_pc_phoneme_input.npy"
    else:
        raise ValueError("condition must be 'clear' or 'noisy'")

    filepath = PHONEME_VECTOR_DIR / filename

    if not filepath.exists():
        # Print debug info once for missing files
        if not _debug_printed:
            print(f"\nMissing file: {filename}")
            print(f"Directory: {PHONEME_VECTOR_DIR}")
            files = list(PHONEME_VECTOR_DIR.glob("*.npy"))
            print(f"Found {len(files)} .npy files total\n")
            _debug_printed = True
        return torch.zeros(INPUT_DIM, dtype=torch.float32, device=DEVICE)

    try:
        data = np.load(filepath)
        rows, cols = data.shape
        
        # Pad or truncate to expected size
        if rows > N_PHONEME_SLOTS:
            data = data[:N_PHONEME_SLOTS, :]
        elif rows < N_PHONEME_SLOTS:
            padding = np.zeros((N_PHONEME_SLOTS - rows, cols), dtype=data.dtype)
            data = np.vstack([data, padding])

        vec = torch.from_numpy(data.flatten()).float()
        
        # L2 normalize
        norm = torch.norm(vec, p=2)
        if norm > 0:
            vec = vec / norm
        
        # Add noise for noisy condition
        # (Wav2Vec representations of vocoded speech are too clean)
        if condition == 'noisy':
            noise = torch.randn_like(vec) * 0.2
            vec = vec + noise
            vec = vec / torch.norm(vec, p=2)
            
        return vec.to(DEVICE)

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return torch.zeros(INPUT_DIM, dtype=torch.float32, device=DEVICE)
