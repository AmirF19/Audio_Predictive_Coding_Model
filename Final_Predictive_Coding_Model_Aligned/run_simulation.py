"""
PROJECT OVERVIEW:
A predictive coding account of noisy word recognition
Muhammad Fusenig, Alba Jorquera, William Zumchak
LING848 - Predictive Processing and Sentence Structure
Dr. Philip Resnik


EXPERIMENT DESIGN:
    Prime-Target paradigm with 2x2 factorial design:
    - Identity: same word vs different word
    - Clarity: clean audio vs vocoded (noisy) audio
    N400 measurement: lexico-semantic prediction error during target phase
    
PIPELINE:
    1. Load lexicon, semantic features, and auditory vectors
    2. Build weight matrices (audio-lexical-semantic-contextual)
    3. Apply frequency bias to bottom-up weights
    4. Extract N400 metrics and recognition accuracy
    6. Generate visualizations and export results
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json

from pc_model_gpu import BatchedAuditoryPCModelGPU, get_device
from analysis import print_summary, plot_results, save_results


PROJECT_ROOT = Path("C:/Users/Muhammad/OneDrive/Desktop/comp_ling_project")
WORD_LIST_FILE = PROJECT_ROOT / "my_800_words.csv"
SEMANTIC_JSON_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "output" / "all_words_model_input.json"
COCHLEAR_VECTOR_DIR = PROJECT_ROOT / "audio_phonemes" / "Cochlear_Input_Vectors"
EXPERIMENTAL_PAIRS_FILE = PROJECT_ROOT / "experimental_pairs" / "conditions_words.csv"
OUTPUT_DIR = PROJECT_ROOT / "Final_Predictive_Coding_Model_Aligned" / "results_aligned"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# PARAMETERS: (same as Samer's model, except for the batch size)
NUM_ITERS = 20               
TARGET_ITERS = 20            
BLANKS_BEFORE_TARGET = 5     
POST_TARGET_BLANKS = 5       
BATCH_SIZE = 256             

# CONFIGURATION:
# (SAMER_MODE = True: baseline (no additional noise), False: experimental features)
# I wanted to see if the model worked on its own, without the additional noise. It does work as expected, but the N400 responses are still generally high for the different condition pairs. If you want to check as well, feel free to change it to True.
SAMER_MODE = False
# Concept (cpt) clamp: fixes the top conceptual layer to the prime during target
USE_CONCEPT_CLAMP = False
APPLY_FREQUENCY_BIAS = True
APPLY_NOISE = False if SAMER_MODE else True

# INPUTS
#Cochlear vectors are 10-hot (10 phoneme slots)
# Structure: 10 slots × 40 features = 400 dims (vs original model: 4 slots × 26 letters = 104 dims)

INPUT_SCALE = 1.0
AUTO_SCALE_INPUT = True     #(When set to False, the peaks for the same condition are basically double compared to True. Visually, it looks like the model performs as expected, but 2x the magnitude? Maybe False, INPUT_SCALE = 0.5 is the optimal config?)
TARGET_INPUT_NORM = 2.00  # (Don't touch, this was in Samer's original model)
"""
AUTO_SCALE_INPUT set to False:

N400 by Condition:
  same/noisy: mean=40.00, peak=185.17, peak_iter=25.1
  same/clear: mean=38.63, peak=139.81, peak_iter=25.0
  different/noisy: mean=558.18, peak=902.05, peak_iter=28.0
  different/clear: mean=368.78, peak=639.60, peak_iter=27.4

Recognition Accuracy:
  noisy: 81.4%
  clear: 98.6%

Output if we set AUTO_SCALE_INPUT to True:

N400 by Condition:
  same/noisy: mean=40.63, peak=179.22, peak_iter=25.0
  same/clear: mean=38.91, peak=129.78, peak_iter=25.0
  different/noisy: mean=582.43, peak=937.76, peak_iter=28.1
  different/clear: mean=372.08, peak=638.83, peak_iter=27.6

Recognition Accuracy:
  noisy: 84.0%
  clear: 98.6%
"""

def calibrate_input_scale(vectors, target_norm=2.0, eps=1e-6):
    """
    Calibrate global input scaling to match Samer's model's drive.
    
    MOTIVATION:
        Samer's model uses 4-hot orthographic input with L2 norm ~2.
        Our cochlear vectors are 10-hot.
        We compute a scaling factor to match the target norm.
    
    Args:
        vectors: Iterable of audio vectors
        target_norm: Desired mean L2 norm (2.0 matches reference)
        eps: Minimum norm to avoid division by zero
    
    Returns:
        scale: Multiplicative factor to apply to all inputs
    """
    norms = []
    for vec in vectors:
        norm = np.linalg.norm(vec)
        if norm > eps:
            norms.append(norm)
    
    if len(norms) == 0:
        return 1.0
    
    mean_norm = np.mean(norms)
    return target_norm / mean_norm

# DATA
def load_lexicon():
    """
    Load word list from CSV.
    
    Returns:
        Sorted list of unique words (lowercase, stripped)
    """
    df = pd.read_csv(WORD_LIST_FILE)
    
    # Try 'word' column first, else use first column
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
    
    return sorted(list(set([w for w in words if w])))


def load_semantic_matrix(lexicon):
    """
    Load semantic feature matrix from JSON.
    
    FORMAT:
        JSON dictionary: {word: [feature1, feature2, ...], ...}
        Converts to binary matrix: (n_features x n_words)
    
    Args:
        lexicon: List of words in order
    
    Returns:
        matrix: (n_features x n_words) binary matrix
        feature_list: Sorted list of all features
    """
    with open(SEMANTIC_JSON_FILE, 'r') as f:
        data = json.load(f)
    
    # Collect all unique features
    all_features = set()
    for features in data.values():
        all_features.update(features)
    
    feature_list = sorted(list(all_features))
    feat_to_idx = {f: i for i, f in enumerate(feature_list)}
    word_to_idx = {w: i for i, w in enumerate(lexicon)}
    
    # Build binary feature matrix
    matrix = np.zeros((len(feature_list), len(lexicon)), dtype=np.float32)
    for word, features in data.items():
        word = word.strip().lower()
        if word in word_to_idx:
            w_idx = word_to_idx[word]
            for feat in features:
                if feat in feat_to_idx:
                    matrix[feat_to_idx[feat], w_idx] = 1.0
    
    print(f"Semantic matrix: {len(feature_list)} features × {len(lexicon)} words")
    return matrix, feature_list


def load_cochlear_vectors(lexicon):
    """
    Load pre-computed cochlear feature vectors.
    
    FORMAT:
        One .npy file per word in COCHLEAR_VECTOR_DIR
        Each file contains a phoneme-slot encoded vector (400 dims)
        Structure: 10 phoneme slots * 40 features per slot
        Each slot is 1-hot → words are 10-hot (up to 10 active features)
    
    Args:
        lexicon: List of words to load
    
    Returns:
        Dictionary {word: vector (400,)} for words that have vectors
    """
    vectors = {}
    for word in lexicon:
        filepath = COCHLEAR_VECTOR_DIR / f"{word}.npy"
        if filepath.exists():
            vec = np.load(filepath)
            vectors[word] = vec.flatten().astype(np.float32)
    
    if len(vectors) != len(lexicon):
        missing = [w for w in lexicon if w not in vectors]
        print(f"[WARN] Missing cochlear vectors for {len(missing)} words")
    else:
        print(f"Loaded {len(vectors)}/{len(lexicon)} cochlear vectors")
    
    return vectors


def load_experimental_pairs(filepath, lexicon, available_words):
    """
    Load prime-target pairs from CSV.
    
    FORMAT:
        Columns: prime_word, target_word, degradation (0=clean, 1=noisy), identity
        
    FILTERING:
        Only includes pairs where both prime and target have cochlear vectors
    
    Args:
        filepath: Path to experimental pairs CSV
        lexicon: List of all words (for indexing)
        available_words: Set of words with cochlear vectors
    
    Returns:
        List of dicts with keys: prime, target, clarity, condition, prime_idx, target_idx
    """
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    word_to_idx = {w: i for i, w in enumerate(lexicon)}
    pairs = []
    
    # Find column names (flexible column naming)
    prime_col = next((c for c in df.columns if c in ['prime', 'prime_word']), None)
    target_col = next((c for c in df.columns if c in ['target', 'target_word']), None)
    degradation_col = next((c for c in df.columns if c in ['degradation']), None)
    identity_col = next((c for c in df.columns if 'identity' in c), None)
    
    for _, row in df.iterrows():
        prime = str(row[prime_col]).lower().strip()
        target = str(row[target_col]).lower().strip()
        
        # Remove .wav extension if present
        if prime.endswith('.wav'):
            prime = prime[:-4]
        if target.endswith('.wav'):
            target = target[:-4]
        
        # Skip if either word missing cochlear vector
        if prime not in available_words or target not in available_words:
            continue
        
        # Determine clarity and condition
        if degradation_col:
            clarity = 'noisy' if row[degradation_col] == 1 else 'clear'
        else:
            clarity = 'clear'
        
        if identity_col:
            condition = 'same' if row[identity_col] == 1 else 'different'
        else:
            condition = 'same' if prime == target else 'different'
        
        pairs.append({
            'prime': prime,
            'target': target,
            'clarity': clarity,
            'condition': condition,
            'prime_idx': word_to_idx[prime],
            'target_idx': word_to_idx[target],
        })
    
    print(f"Loaded {len(pairs)} experimental pairs")
    return pairs


def load_frequency_bias(lexicon):
    """
    Load SUBTLEX word frequency and convert to bias weights.
    
    NORMALIZATION:
        Log frequency is linearly scaled to [0, 0.1] range
        This range was tuned in the reference model for optimal behavior
    
    Args:
        lexicon: List of words in order
    
    Returns:
        Array of shape (n_words,) with frequency bias weights
        Returns None if frequency file not found
    """
    freq_file = PROJECT_ROOT / "samer_model" / "helper_txt_files" / "SUBTLEXus2007.csv"
    if not freq_file.exists():
        print("Frequency file not found; skipping frequency bias.")
        return None
    
    df = pd.read_csv(freq_file)
    df['Word'] = df['Word'].astype(str).str.lower()
    df = df[df['Word'].isin(lexicon)]
    df = df.set_index('Word')
    
    # Get log frequency for each word (missing words get min value)
    logFreq = []
    for w in lexicon:
        if w in df.index:
            logFreq.append(df.loc[w, 'Lg10WF'])
        else:
            logFreq.append(df['Lg10WF'].min())
    logFreq = np.array(logFreq)
    
    # Linear scaling to [0, 0.1]
    min_freq_score, max_freq_score = 0, 0.1
    scale_shift = np.linalg.inv(
        np.array([[logFreq.min(), 1], [logFreq.max(), 1]])
    ) @ np.array([min_freq_score, max_freq_score])
    freq_scaled = scale_shift[0] * logFreq + scale_shift[1]
    
    print("Loaded frequency bias (SUBTLEX) scaled to [0, 0.1]")
    return freq_scaled.astype(np.float32)


# MAIN SIMULATION
def run_experiment():
    """
    Execute complete auditory priming experiment.
    
    PIPELINE:
        1. Load data (lexicon, semantics, audio vectors, experimental pairs)
        2. Build audio matrix with optional input scaling
        3. Initialize GPU model with weight matrices
        4. Process experimental pairs in batches
        5. Extract N400 metrics and recognition accuracy
        6. Generate visualizations and save results
    """
    print("=" * 60)
    print("Auditory N400 Priming Simulation")
    print("=" * 60)
    
    # 1: Load Data
    device = get_device()
    
    lexicon_full = load_lexicon()
    cochlear_vectors = load_cochlear_vectors(lexicon_full)
    
    # Filter lexicon to words with cochlear vectors
    lexicon = sorted(set(lexicon_full) & set(cochlear_vectors.keys()))
    if len(lexicon) < len(lexicon_full):
        missing = sorted(set(lexicon_full) - set(lexicon))
        print(f"Filtering lexicon: using {len(lexicon)}/{len(lexicon_full)} words")
        print(f"Missing (first 20): {missing[:20]}")
    
    # 2: Build Semantic Matrix
    sem_matrix, feature_list = load_semantic_matrix(lexicon)
    
    # 3: Build Audio Matrix
    audio_vectors = {w: cochlear_vectors[w] for w in lexicon}
    if len(audio_vectors) == 0:
        raise ValueError("No cochlear vectors available after lexicon filtering.")
    
    effective_scale = INPUT_SCALE
    if AUTO_SCALE_INPUT:
        effective_scale = calibrate_input_scale(
            audio_vectors.values(), 
            target_norm=TARGET_INPUT_NORM
        )
        print(f"Auto input scale -> {effective_scale:.3f} (target norm {TARGET_INPUT_NORM})")
    
    # Construct audio matrix: where each column is one word's phoneme vector
    sample_vec = next(iter(audio_vectors.values()))
    audio_dim = len(sample_vec)
    audio_matrix = np.zeros((audio_dim, len(lexicon)), dtype=np.float32)
    
    for i, word in enumerate(lexicon):
        vec = audio_vectors[word]
        # L2 normalize then scale
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        vec = vec * effective_scale
        audio_matrix[:, i] = vec
    
    print(f"Audio matrix: {audio_dim} dims x {len(lexicon)} words")
    
    # STEP 4: Frequency Bias
    freq_bias = load_frequency_bias(lexicon) if APPLY_FREQUENCY_BIAS else None
    
    # 5: GPU go brrrrr
    print("\nInitializing GPU model...")
    model = BatchedAuditoryPCModelGPU(
        lexicon_words=lexicon,
        audio_matrix=audio_matrix,
        semantic_matrix=sem_matrix,
        frequency_bias=freq_bias,
        device=device
    )
    
    # 6: Load Experimental Pairs (Loveboat style)
    pairs = load_experimental_pairs(EXPERIMENTAL_PAIRS_FILE, lexicon, set(lexicon))
    
    if not pairs:
        raise ValueError(f"No experimental pairs found in {EXPERIMENTAL_PAIRS_FILE}. Please ensure the experimental pairs file exists and contains valid data.")
    
    # RUN (Forest RUN) SIMULATION
    print(f"\nRunning {len(pairs)} trials in batches of {BATCH_SIZE}...")
    all_results = []
    
    n_batches = (len(pairs) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_idx in tqdm(range(n_batches), desc="Processing batches"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, len(pairs))
        batch_pairs = pairs[batch_start:batch_end]
        batch_size = len(batch_pairs)
        
        prime_vecs = np.zeros((audio_dim, batch_size), dtype=np.float32)
        target_vecs = np.zeros((audio_dim, batch_size), dtype=np.float32)
        prime_indices = []
        
        # Fill batch arrays
        for i, pair in enumerate(batch_pairs):
            prime_vec = audio_vectors.get(pair['prime'])
            target_vec = audio_vectors.get(pair['target'])
            
            noisy_processed = False
            
            # README:APPLIES NOISE WHEN SAMER=_MODE IS SET TO FALSE: inject noise into noisy targets 
            if APPLY_NOISE and pair['clarity'] == 'noisy' and target_vec is not None:
                target_vec = target_vec.copy()
                active_mask = target_vec > 0
                
                if np.any(active_mask):
                    # Normalize
                    norm_t = np.linalg.norm(target_vec)
                    if norm_t > 0:
                        target_vec = target_vec / norm_t
                    
                    # DEFAULT NOISE (RECOGNITION ACCURACY TURNS TO: noisy: 68.6% & clear: 89.7%; it seems like we get the N400 responses that we're looking for)
                    base_noise = 1.0
                    
                    # PROPORTIONAL NOISE (PROPORTIONAL TO COSINE DISSIMILARITY (0.575 similarity -> 0.425 dissimilarity) noisy = 76.3% & clear = 89.7%; This produces very similar N400 responses for the same and different condition pairs)
                    # base_noise = 1.0 * (1 - 0.575)  # = 0.425
                    
                    target_vec[active_mask] = (
                        target_vec[active_mask] +
                        np.random.randn(active_mask.sum()).astype(np.float32) * 
                        base_noise
                    )
                    target_vec = np.maximum(target_vec, 0)  # Clip negative values
                
                target_vecs[:, i] = target_vec * INPUT_SCALE
                noisy_processed = True
            
            # Store prime vector (always clean)
            if prime_vec is not None:
                norm_p = np.linalg.norm(prime_vec)
                vec_p = prime_vec / norm_p if norm_p > 0 else prime_vec
                prime_vecs[:, i] = vec_p * INPUT_SCALE
            
            # Store target vector
            if target_vec is not None and not noisy_processed:
                norm_t = np.linalg.norm(target_vec)
                vec_t = target_vec / norm_t if norm_t > 0 else target_vec
                target_vecs[:, i] = vec_t * INPUT_SCALE
            
            prime_indices.append(pair['prime_idx'])
        
        # Run GPU brrrrrrr
        batch_results = model.run_batch_trials(
            prime_vecs,
            target_vecs,
            prime_indices,
            prime_iters=NUM_ITERS,
            blank_iters=BLANKS_BEFORE_TARGET,
            target_iters=TARGET_ITERS,
            post_target_iters=POST_TARGET_BLANKS,
            use_cpt_clamp=USE_CONCEPT_CLAMP
        )
        
        #EXTRACT METRICS
        target_start = NUM_ITERS + BLANKS_BEFORE_TARGET
        target_end = target_start + TARGET_ITERS
        post_end = target_end + POST_TARGET_BLANKS
        
        for i, pair in enumerate(batch_pairs):
            # Extract traces
            trace_lexsem = batch_results['total_lexsem_err'][i]
            trace_lex = batch_results['total_lex_err'][i]
            trace_sem = batch_results['total_sem_err'][i]
            
            # N400 metrics: window 2-11 during target
            target_window = slice(target_start + 1, min(target_start + 11, len(trace_lexsem)))
            n400_mean = np.mean(trace_lexsem[target_window])
            n400_peak = np.max(trace_lexsem[target_start:target_end])
            n400_peak_iter = target_start + int(np.argmax(trace_lexsem[target_start:target_end]))
            
            # Final value (after settling)
            n400_final = trace_lexsem[post_end - 1] if post_end - 1 < len(trace_lexsem) else trace_lexsem[-1]
            
            # Recognition: did model select correct word?
            max_activation = batch_results['max_lex_state_activation'][i, -1]
            winner_idx = int(batch_results['max_lex_state_identity'][i, -1])
            target_correct = (winner_idx == pair['target_idx'])
            
            # Store results
            all_results.append({
                'prime': pair['prime'],
                'target': pair['target'],
                'clarity': pair['clarity'],
                'condition': pair['condition'],
                'n400_mean': n400_mean,
                'n400_peak': n400_peak,
                'n400_peak_iter': n400_peak_iter,
                'n400_final': n400_final,
                'max_activation': max_activation,
                'winner_word': lexicon[winner_idx],
                'target_correct': target_correct,
                'trace_lexsem_err': trace_lexsem.tolist(),
                'trace_lex_err': trace_lex.tolist(),
                'trace_sem_err': trace_sem.tolist(),
                'trace_max_activation': batch_results['max_lex_state_activation'][i].tolist(),
            })
    
    df = pd.DataFrame(all_results)
    
    print_summary(df)
    save_results(df, OUTPUT_DIR)
    plot_results(df, OUTPUT_DIR, 
                num_iters=NUM_ITERS, 
                blanks_before=BLANKS_BEFORE_TARGET, 
                target_iters=TARGET_ITERS)
    
    return df

if __name__ == "__main__":
    run_experiment()
