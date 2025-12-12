"""
Calibrate noise level by comparing clean vs noisy audio through cochlear model.

This script:
1. Processes clean and noisy audio through the cochlear model
2. Uses the same phoneme boundaries from MFA TextGrids
3. Computes cosine similarity between clean and noisy vectors
4. Outputs calibration data to determine appropriate noise level for one-hot encoding
"""

import numpy as np
import soundfile as sf
from pathlib import Path
from scipy.signal import resample
from scipy.spatial.distance import cosine
import pandas as pd

# Import the cochlear model
from strfpy import wav2aud

# Paths
PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
CLEAN_AUDIO_DIR = PROJECT_ROOT / "audio_phonemes" / "All_Recordings"
NOISY_AUDIO_DIR = PROJECT_ROOT / "audio_phonemes" / "Noisy_Recordings"
TEXTGRID_DIR = PROJECT_ROOT / "audio_phonemes" / "MFA_Output_TextGrids"
COCHBA_PATH = Path(__file__).parent / "cochba.txt"
WORDS_LIST = PROJECT_ROOT / "my_800_words.csv"
OUTPUT_FILE = PROJECT_ROOT / "audio_phonemes" / "cochlear_processing" / "noise_calibration.csv"

# Cochlear model parameters
TARGET_SR = 16000
FRMLEN = 8
TIME_CONSTANT = 8
FAC = -2
OCTAVE_SHIFT = 0
FIXED_SLOTS = 10


def parse_textgrid(path):
    """Minimal TextGrid parser for MFA outputs."""
    intervals = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    for i, line in enumerate(lines):
        if line.startswith("intervals ["):
            try:
                xmin = float(lines[i+1].split("=")[1])
                xmax = float(lines[i+2].split("=")[1])
                text = lines[i+3].split("=")[1].strip().strip('"')
                intervals.append((text, xmin, xmax))
            except Exception:
                continue
    return intervals


def compute_cochlear(audio, sr):
    """Compute cochlear spectrogram."""
    if sr != TARGET_SR:
        n_samples = int(len(audio) * TARGET_SR / sr)
        audio = resample(audio, n_samples)
    
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val
    
    aud_spec = wav2aud(audio, FRMLEN, TIME_CONSTANT, FAC, OCTAVE_SHIFT,
                       cochba_path=str(COCHBA_PATH))
    return aud_spec.T.astype(np.float32)  # (time, 128)


def pool_time_slice(coch, t_start, t_end, frame_rate):
    """Pool cochleagram between times."""
    start_idx = max(0, int(t_start * frame_rate))
    end_idx = max(start_idx + 1, int(t_end * frame_rate))
    start_idx = min(start_idx, coch.shape[0] - 1)
    end_idx = min(end_idx, coch.shape[0])
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    slice_ = coch[start_idx:end_idx]
    if len(slice_) == 0:
        return np.zeros(coch.shape[1], dtype=np.float32)
    return slice_.mean(axis=0)


def process_word_cochlear(word, audio_dir):
    """Process a word through cochlear model and return phoneme-pooled vector."""
    wav_path = audio_dir / f"{word}.wav"
    tg_path = TEXTGRID_DIR / f"{word}.TextGrid"
    
    if not wav_path.exists() or not tg_path.exists():
        return None
    
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    
    coch = compute_cochlear(audio, sr)
    frame_rate = 1000.0 / FRMLEN
    
    intervals = parse_textgrid(tg_path)
    phoneme_vecs = []
    for phn, start, end in intervals:
        if phn in ("", "sp", "sil"):
            continue
        vec = pool_time_slice(coch, start, end, frame_rate)
        phoneme_vecs.append(vec)
    
    if not phoneme_vecs:
        return None
    
    freq_dim = coch.shape[1]
    if len(phoneme_vecs) < FIXED_SLOTS:
        pad = [np.zeros(freq_dim, dtype=np.float32) for _ in range(FIXED_SLOTS - len(phoneme_vecs))]
        phoneme_vecs.extend(pad)
    phoneme_vecs = phoneme_vecs[:FIXED_SLOTS]
    
    stacked = np.stack(phoneme_vecs, axis=0)
    flat = stacked.reshape(-1).astype(np.float32)
    return flat


def cosine_similarity(a, b):
    """Compute cosine similarity."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def load_word_list(path):
    df = pd.read_csv(path)
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:,0].astype(str).str.strip().str.lower().tolist()
    return sorted(set([w for w in words if w]))


def main():
    words = load_word_list(WORDS_LIST)
    print(f"Processing {len(words)} words...")
    print(f"Clean audio: {CLEAN_AUDIO_DIR}")
    print(f"Noisy audio: {NOISY_AUDIO_DIR}")
    print()
    
    results = []
    similarities = []
    
    for i, word in enumerate(words):
        # Check if both clean and noisy exist
        clean_path = CLEAN_AUDIO_DIR / f"{word}.wav"
        noisy_path = NOISY_AUDIO_DIR / f"{word}.wav"
        
        if not clean_path.exists() or not noisy_path.exists():
            continue
        
        # Process both versions
        clean_vec = process_word_cochlear(word, CLEAN_AUDIO_DIR)
        noisy_vec = process_word_cochlear(word, NOISY_AUDIO_DIR)
        
        if clean_vec is None or noisy_vec is None:
            continue
        
        # Compute similarity
        sim = cosine_similarity(clean_vec, noisy_vec)
        similarities.append(sim)
        
        results.append({
            'word': word,
            'cosine_similarity': sim,
            'clean_norm': np.linalg.norm(clean_vec),
            'noisy_norm': np.linalg.norm(noisy_vec),
        })
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1} words, avg similarity so far: {np.mean(similarities):.4f}")
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nResults saved to: {OUTPUT_FILE}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Words processed: {len(results)}")
    print(f"\nCosine Similarity (Clean vs Noisy):")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Std:  {np.std(similarities):.4f}")
    print(f"  Min:  {np.min(similarities):.4f}")
    print(f"  Max:  {np.max(similarities):.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    
    # Interpretation
    avg_sim = np.mean(similarities)
    degradation = 1 - avg_sim
    print(f"\n  Average degradation: {degradation*100:.1f}%")
    print(f"  (Noisy signal retains ~{avg_sim*100:.1f}% of clean signal)")
    
    # Recommended noise parameters
    print("\n" + "=" * 60)
    print("RECOMMENDED NOISE PARAMETERS FOR ONE-HOT ENCODING")
    print("=" * 60)
    
    # For one-hot vectors, we want to match this degradation
    # Options:
    # 1. Reduce activation by (1 - similarity)
    # 2. Add noise proportional to degradation
    
    reduction_factor = avg_sim  # Multiply active positions by this
    noise_std = degradation * 0.5  # Add noise with this std to active positions
    
    print(f"  Activation reduction factor: {reduction_factor:.3f}")
    print(f"  (Multiply active positions by {reduction_factor:.3f})")
    print(f"  Noise std to add: {noise_std:.4f}")
    print(f"  (Add randn * {noise_std:.4f} to active positions)")
    
    print("\nCode snippet for audio_hcp_model_gpu.py:")
    print("-" * 40)
    print(f"""
if pair['clarity'] == 'noisy' and target_vec is not None:
    target_vec = target_vec.copy()
    active_mask = target_vec > 0
    # Reduce activation (calibrated from clean/noisy similarity)
    target_vec[active_mask] *= {reduction_factor:.3f}
    # Add calibrated noise
    target_vec[active_mask] += np.random.randn(active_mask.sum()).astype(np.float32) * {noise_std:.4f}
""")
    
    return df


if __name__ == "__main__":
    main()





