"""
Build phoneme-slot vectors for each word.

Following Samer's approach:
- Samer uses 4 letter positions × 26 letters = 104 dimensions (sparse one-hot)
- We use 10 phoneme slots × 40 phonemes = 400 dimensions (sparse one-hot)

This creates sparse vectors where similarity is directly proportional to 
phoneme overlap, just like Samer's orthographic vectors.
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Paths
PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
AUDIO_DIR = PROJECT_ROOT / "audio_phonemes" / "All_Recordings"
TEXTGRID_DIR = PROJECT_ROOT / "audio_phonemes" / "MFA_Output_TextGrids"
COCHLEAR_OUTPUT_DIR = PROJECT_ROOT / "audio_phonemes" / "Cochlear_Input_Vectors"
WORDS_LIST = PROJECT_ROOT / "my_800_words.csv"

# Settings - analogous to Samer's 4 positions × 26 letters = 104 dims
FIXED_SLOTS = 10  # Max phoneme positions (like 4 letter positions in Samer)

# ARPAbet phoneme inventory from MFA (like 26 letters in Samer)
# Core phonemes without stress markers
PHONEME_LIST = [
    # Vowels
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 
    'IY', 'OW', 'OY', 'UH', 'UW',
    # Consonants  
    'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L',
    'M', 'N', 'NG', 'P', 'R', 'S', 'SH', 'T', 'TH', 'V',
    'W', 'Y', 'Z', 'ZH',
    # Special
    'spn',  # spoken noise
]
N_PHONEMES = len(PHONEME_LIST)  # ~40 phonemes (like 26 letters)
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_LIST)}

# Total dimensions: FIXED_SLOTS × N_PHONEMES (like 4 × 26 = 104 in Samer)
TOTAL_DIMS = FIXED_SLOTS * N_PHONEMES  # 10 × 40 = 400


def normalize_phoneme(phoneme):
    """
    Normalize phoneme label by removing stress markers.
    MFA outputs like 'AH0', 'AH1', 'AH2' -> 'AH'
    """
    # Remove trailing digits (stress markers)
    phoneme = phoneme.upper().strip()
    while phoneme and phoneme[-1].isdigit():
        phoneme = phoneme[:-1]
    return phoneme


def parse_textgrid(path):
    """Minimal TextGrid parser for MFA outputs; returns list of (phoneme, start, end)."""
    intervals = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    
    # Find the phones tier (usually the first tier with phoneme labels)
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


def phoneme_to_onehot(phoneme_list):
    """
    Convert a list of phonemes to a sparse one-hot vector.
    
    Analogous to Samer's wordlist_to_orth():
    - Each position (slot) gets a one-hot encoding of which phoneme is there
    - Slots × Phonemes = dimensions (like 4 × 26 = 104)
    
    Args:
        phoneme_list: List of phoneme strings (e.g., ['AE', 'P', 'AH', 'L'] for "apple")
    
    Returns:
        1D numpy array of shape (FIXED_SLOTS * N_PHONEMES,) with sparse one-hot encoding
    """
    # Initialize all zeros (like Samer's onehots = np.zeros(...))
    onehot = np.zeros(TOTAL_DIMS, dtype=np.float32)
    
    # Truncate to FIXED_SLOTS if needed
    phoneme_list = phoneme_list[:FIXED_SLOTS]
    
    # Set ones at appropriate positions (like Samer's indices = np.add(wordids[i], np.array([0,1,2,3])*26))
    for slot_idx, phoneme in enumerate(phoneme_list):
        phoneme_normalized = normalize_phoneme(phoneme)
        
        if phoneme_normalized in PHONEME_TO_IDX:
            phoneme_idx = PHONEME_TO_IDX[phoneme_normalized]
            # Index = slot_offset + phoneme_idx (like position*26 + letter_idx)
            final_idx = slot_idx * N_PHONEMES + phoneme_idx
            onehot[final_idx] = 1.0
    
    return onehot


def process_word(word):
    """
    Process a word to create its phoneme-based one-hot vector.
    
    Analogous to how Samer creates orthographic vectors from word spellings.
    """
    tg_path = TEXTGRID_DIR / f"{word}.TextGrid"
    wav_path = AUDIO_DIR / f"{word}.wav"
    
    if not tg_path.exists():
        raise FileNotFoundError(f"Missing TextGrid for {word}")
    if not wav_path.exists():
        raise FileNotFoundError(f"Missing audio for {word}")
    
    # Parse TextGrid to get phoneme sequence
    intervals = parse_textgrid(tg_path)
    
    # Extract phonemes (skip silence/empty)
    phonemes = []
    for phn, start, end in intervals:
        phn = phn.strip()
        if phn and phn not in ("", "sp", "sil", ""):
            phonemes.append(phn)
    
    if not phonemes:
        raise ValueError(f"No phonemes found for {word}")
    
    # Convert to one-hot vector (like Samer's wordlist_to_orth)
    onehot = phoneme_to_onehot(phonemes)
    
    # Save
    COCHLEAR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COCHLEAR_OUTPUT_DIR / f"{word}.npy"
    np.save(out_path, onehot)
    
    return out_path, onehot.shape, len(phonemes)


def load_word_list(path):
    import pandas as pd
    df = pd.read_csv(path)
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:,0].astype(str).str.strip().str.lower().tolist()
    words = [w for w in words if w]
    return sorted(set(words))


def main():
    words = load_word_list(WORDS_LIST)
    print(f"Words to process: {len(words)}")
    print(f"Vector dimensions: {FIXED_SLOTS} slots × {N_PHONEMES} phonemes = {TOTAL_DIMS}")
    print(f"(Analogous to Samer's 4 positions × 26 letters = 104)")
    print()
    
    ok, fail = 0, 0
    phoneme_counts = []
    
    for w in words:
        try:
            _, shape, n_phonemes = process_word(w)
            ok += 1
            phoneme_counts.append(n_phonemes)
            if ok % 50 == 0:
                print(f"Processed {ok} words; vector shape {shape}")
        except Exception as e:
            fail += 1
            print(f"[WARN] {w}: {e}")
    
    print(f"\nDone. Success: {ok}, Fail: {fail}")
    
    if phoneme_counts:
        print(f"\nPhoneme statistics:")
        print(f"  Mean phonemes per word: {np.mean(phoneme_counts):.1f}")
        print(f"  Max phonemes: {max(phoneme_counts)}")
        print(f"  Min phonemes: {min(phoneme_counts)}")
        print(f"  Words with >{FIXED_SLOTS} phonemes (truncated): {sum(1 for c in phoneme_counts if c > FIXED_SLOTS)}")


if __name__ == "__main__":
    main()
