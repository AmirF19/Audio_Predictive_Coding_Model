"""
Empirical calibration tool: Find the right noise level to match vocoded speech.

We measured cosine similarity between clean and vocoded speech in our pilot (0.575).
This script tests different noise mixing levels (N=1,2,3,4,5) to see which one
produces the same similarity when we average target + N random words.

The goal: match our computational noise to real acoustic degradation.

Author: Muhammad Fusenig
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# Import paths from run_simulation
PROJECT_ROOT = Path("C:/Users/Muhammad/OneDrive/Desktop/comp_ling_project")
WORD_LIST_FILE = PROJECT_ROOT / "my_800_words.csv"
COCHLEAR_VECTOR_DIR = PROJECT_ROOT / "audio_phonemes" / "Cochlear_Input_Vectors"

# Noise levels to test
NOISE_LEVELS = [1, 2, 3, 4, 5]

# Target similarity (empirically measured vocoded speech)
TARGET_SIMILARITY = 0.575


def load_lexicon():
    """Load word list from CSV."""
    df = pd.read_csv(WORD_LIST_FILE)
    
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
    
    return sorted(list(set([w for w in words if w])))


def load_cochlear_vectors(lexicon):
    """Load pre-computed cochlear feature vectors."""
    vectors = {}
    for word in lexicon:
        filepath = COCHLEAR_VECTOR_DIR / f"{word}.npy"
        if filepath.exists():
            vec = np.load(filepath)
            vectors[word] = vec.flatten().astype(np.float32)
    
    print(f"Loaded {len(vectors)}/{len(lexicon)} cochlear vectors")
    return vectors


def create_noisy_version(target_vec, all_vectors, target_word, n_mixes):
    """
    Mix target with N random words (same as run_simulation.py).
    We use this to measure how similar the mixed version is to the original.
    """
    available_words = [w for w in all_vectors.keys() if w != target_word]
    
    if len(available_words) < n_mixes:
        n_mixes = len(available_words)  # Can't mix with more than we have
    
    if n_mixes == 0:
        return target_vec  # No mixing possible
    
    # Pick N random words and average them all together with target
    random_words = np.random.choice(available_words, n_mixes, replace=False)
    random_vecs = [all_vectors[w] for w in random_words]
    all_vecs = [target_vec] + random_vecs
    mixed_vec = np.mean(all_vecs, axis=0)
    
    return mixed_vec


def calculate_similarity_for_noise_level(audio_vectors, n_mixes, n_samples=100):
    """
    Test one noise level: mix each word with N random words and measure similarity.
    
    We do this multiple times per word (n_samples) because the random selection varies.
    Returns the average similarity and how much it varies.
    """
    similarities = []
    words = list(audio_vectors.keys())
    
    # Sample words for faster computation
    if len(words) > 200:
        sample_words = np.random.choice(words, 200, replace=False)
    else:
        sample_words = words
    
    print(f"  Testing {len(sample_words)} words with {n_samples} samples each...")
    
    for word in tqdm(sample_words, desc=f"  N={n_mixes}", leave=False):
        clean_vec = audio_vectors[word]
        
        # Create multiple noisy versions (different random words each time)
        for _ in range(n_samples):
            noisy_vec = create_noisy_version(clean_vec, audio_vectors, word, n_mixes)
            
            # Calculate cosine similarity
            sim = cosine_similarity(clean_vec.reshape(1, -1), 
                                   noisy_vec.reshape(1, -1))[0, 0]
            similarities.append(sim)
    
    return np.mean(similarities), np.std(similarities)


def main():
    """Main calculation routine."""
    print("="*60)
    print("COSINE SIMILARITY CALCULATION")
    print("="*60)
    print(f"Target similarity (vocoded speech): {TARGET_SIMILARITY:.3f}")
    print(f"Testing noise levels: {NOISE_LEVELS}")
    print()
    
    # Load data
    print("Loading lexicon and vectors...")
    lexicon = load_lexicon()
    audio_vectors = load_cochlear_vectors(lexicon)
    
    # Filter to words with vectors
    audio_vectors = {w: v for w, v in audio_vectors.items() if w in lexicon}
    print(f"Using {len(audio_vectors)} words for analysis\n")
    
    # Calculate similarities for each noise level
    results = []
    
    for n_mixes in NOISE_LEVELS:
        print(f"Noise level N={n_mixes} (target weight: {100/(n_mixes+1):.1f}%)")
        
        mean_sim, std_sim = calculate_similarity_for_noise_level(
            audio_vectors, n_mixes, n_samples=50
        )
        
        # Calculate distance from target
        distance_from_target = abs(mean_sim - TARGET_SIMILARITY)
        
        results.append({
            'noise_level': n_mixes,
            'target_weight_pct': 100 / (n_mixes + 1),
            'mean_similarity': mean_sim,
            'std_similarity': std_sim,
            'target_similarity': TARGET_SIMILARITY,
            'distance_from_target': distance_from_target
        })
        
        print(f"  Mean similarity: {mean_sim:.3f} +/- {std_sim:.3f}")
        print(f"  Distance from target (0.575): {distance_from_target:.3f}")
        print()
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Find best match
    best_idx = results_df['distance_from_target'].idxmin()
    best_n = results_df.loc[best_idx, 'noise_level']
    best_sim = results_df.loc[best_idx, 'mean_similarity']
    best_dist = results_df.loc[best_idx, 'distance_from_target']
    
    # Print summary
    print("="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(results_df.to_string(index=False))
    print()
    
    print("="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"Best match: N={best_n}")
    print(f"  Achieved similarity: {best_sim:.3f}")
    print(f"  Target similarity: {TARGET_SIMILARITY:.3f}")
    print(f"  Distance: {best_dist:.3f}")
    print(f"  Target weight: {100/(best_n+1):.1f}%")
    print()
    print(f"Use NOISE_MIX_LEVEL = {best_n} for empirically calibrated noise.")
    print("="*60)
    
    # Save results
    output_dir = Path("C:/Users/Muhammad/OneDrive/Desktop/comp_ling_project/Final_Predictive_Coding_Model_Aligned")
    output_file = output_dir / "cosine_similarity_analysis.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    main()
