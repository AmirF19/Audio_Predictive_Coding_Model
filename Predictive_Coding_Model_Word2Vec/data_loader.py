"""
Data loading functions for the PC model (Word2Vec version).
Uses Word2Vec embeddings instead of Llama semantic features.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple

from config import (
    WORD_LIST_FILE,
    PHONEME_VECTOR_DIR,
    N_PHONEME_SLOTS,
    PHONEME_VECTOR_DIM,
    INPUT_DIM,
    WORD2VEC_DIM,
    DEVICE,
)

_debug_printed = False
_word2vec_model = None


def load_word2vec():
    """
    Load pre-trained Word2Vec model.
    Uses gensim's word2vec-google-news-300.
    """
    global _word2vec_model
    
    if _word2vec_model is not None:
        return _word2vec_model
    
    try:
        import gensim.downloader as api
        print("Loading Word2Vec model (this may take a minute on first run)...")
        _word2vec_model = api.load("word2vec-google-news-300")
        print(f"Word2Vec loaded: {len(_word2vec_model)} words, {WORD2VEC_DIM} dimensions")
        return _word2vec_model
    except Exception as e:
        print(f"Error loading Word2Vec: {e}")
        print("Install gensim: pip install gensim")
        return None


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
    Build V_SL matrix from Word2Vec embeddings.
    Returns (WORD2VEC_DIM x lexical_dim) matrix of dense embeddings.
    
    Unlike the Llama version, this uses continuous-valued embeddings
    instead of binary features.
    """
    w2v = load_word2vec()
    if w2v is None:
        return torch.empty(0), []

    # Build embedding matrix
    V_SL = np.zeros((WORD2VEC_DIM, len(lexicon)), dtype=np.float32)
    
    found = 0
    missing = []
    
    for i, word in enumerate(lexicon):
        # Try different forms of the word
        forms = [word, word.capitalize(), word.upper()]
        
        for form in forms:
            if form in w2v:
                V_SL[:, i] = w2v[form]
                found += 1
                break
        else:
            missing.append(word)
    
    print(f"Word2Vec matrix: {WORD2VEC_DIM} dims x {len(lexicon)} words")
    print(f"  Found embeddings for {found}/{len(lexicon)} words")
    if len(missing) <= 10:
        print(f"  Missing: {missing}")
    else:
        print(f"  Missing: {len(missing)} words (e.g., {missing[:5]})")
    
    return torch.from_numpy(V_SL).to(DEVICE), [f"dim_{i}" for i in range(WORD2VEC_DIM)]


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
        if condition == 'noisy':
            noise = torch.randn_like(vec) * 0.2
            vec = vec + noise
            vec = vec / torch.norm(vec, p=2)
            
        return vec.to(DEVICE)

    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return torch.zeros(INPUT_DIM, dtype=torch.float32, device=DEVICE)

