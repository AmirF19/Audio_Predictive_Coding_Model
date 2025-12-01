import numpy as np
import pandas as pd
from pathlib import Path
import textgrid # Requires: pip install textgrid
import sys
import os

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
WAV2VEC_DIR = BASE_DIR / "Wav2Vec"
ALIGNMENT_DIR = BASE_DIR / "MFA_Output_TextGrids" 
FINAL_PC_INPUT_DIR = BASE_DIR / "PC_Input_Vectors"
FINAL_PC_INPUT_DIR.mkdir(exist_ok=True)

# Input words - Pointing to the main 800 word list
INPUT_WORDS_FILE = BASE_DIR.parent / "my_800_words.csv"

# Wav2Vec Frame Rate
SAMPLE_RATE = 16000 
FRAME_LENGTH_S = 0.02 # Wav2Vec output frame rate (~20ms per logit vector)

# --- Robust Word Loading Logic ---
def load_user_words(filepath):
    """Loads the list of words from the user's CSV."""
    df = None
    try:
        df = pd.read_csv(filepath, sep='\t')
    except:
        pass 

    if df is None or len(df.columns) < 2: 
        try:
            df = pd.read_csv(filepath, sep=',')
        except:
             print(f"FATAL ERROR: Could not read CSV: {filepath}")
             sys.exit(1)

    df.columns = df.columns.str.strip().str.lower()
    
    word_col_name = None
    if 'word' in df.columns:
        word_col_name = 'word'
    elif 'word,column_2' in df.columns:
         word_col_name = 'word,column_2'

    if word_col_name is None:
        # Fallback to first column
        word_col_name = df.columns[0]
    
    words = df[word_col_name].astype(str).str.strip().str.lower().tolist()
    return [w for w in sorted(list(set(words))) if w]


def load_alignment_data(word, alignment_dir):
    """
    Loads time boundaries from the TextGrid file produced by MFA.
    """
    textgrid_path = alignment_dir / f"{word.lower()}.TextGrid"
    
    if not textgrid_path.exists():
        # Try checking for capitalized filename just in case
        textgrid_path = alignment_dir / f"{word.upper()}.TextGrid"
        if not textgrid_path.exists():
            return None

    try:
        tg = textgrid.TextGrid.fromFile(str(textgrid_path))
        # Find the phoneme tier (usually the one with the most intervals)
        phoneme_tier = max(tg, key=len)
        
        alignment_data = []
        for interval in phoneme_tier:
            # Skip silence/boundary markers
            if interval.mark.strip() and interval.mark.lower() not in ['sil', 'sp', '']:
                alignment_data.append({
                    "phoneme": interval.mark,
                    "start_s": interval.minTime,
                    "end_s": interval.maxTime,
                })
        return alignment_data
    except Exception as e:
        print(f"Error reading TextGrid for {word}: {e}")
        return None

def generate_pc_vector(word):
    """
    Loads raw Wav2Vec time series and extracts vectors using Center-Weighted Slicing.
    """
    # 1. Load Raw Wav2Vec Data (T x N_PHONEMES)
    wav2vec_path = WAV2VEC_DIR / f"{word.lower()}_wav2vec_timeseries_activation.npy"
    if not wav2vec_path.exists():
        return
        
    raw_activations = np.load(wav2vec_path) # Shape: (T, N_PHONEMES)
    
    # 2. Load Phoneme Alignment (Time Boundaries)
    alignment_data = load_alignment_data(word, ALIGNMENT_DIR)
    if not alignment_data:
        return

    pc_input_vectors = []
    
    for segment in alignment_data:
        # --- SOLUTION 3: Center-Weighted / Safe Slicing ---
        
        # A. Calculate Center and Duration
        duration_s = segment['end_s'] - segment['start_s']
        center_s = segment['start_s'] + (duration_s / 2)
        
        # B. Calculate "Ideal" Indices
        center_frame = int(round(center_s / FRAME_LENGTH_S))
        width_frames = duration_s / FRAME_LENGTH_S
        
        if width_frames < 1.0:
            # CASE: Short Consonant (< 20ms)
            # It's too short for a range. Grab the single frame at the center.
            idx = min(max(0, center_frame), raw_activations.shape[0] - 1)
            mean_vector = raw_activations[idx]
        else:
            # CASE: Standard Phoneme
            # Use ceiling for end index to capture the "tail" of the sound
            start_idx = int(np.floor(segment['start_s'] / FRAME_LENGTH_S))
            end_idx = int(np.ceil(segment['end_s'] / FRAME_LENGTH_S))
            
            # Clamp to bounds
            start_idx = max(0, start_idx)
            end_idx = min(raw_activations.shape[0], end_idx)
            
            if end_idx > start_idx:
                mean_vector = raw_activations[start_idx:end_idx].mean(axis=0)
            else:
                # Fallback (should be covered by short consonant logic, but just in case)
                idx = min(max(0, start_idx), raw_activations.shape[0] - 1)
                mean_vector = raw_activations[idx]

        pc_input_vectors.append(mean_vector)

    # 4. Final Output
    if len(pc_input_vectors) > 0:
        final_input_array = np.stack(pc_input_vectors)
        output_path = FINAL_PC_INPUT_DIR / f"{word}_pc_phoneme_input.npy"
        np.save(output_path, final_input_array)
        # print(f"Processed {word}: {final_input_array.shape[0]} phonemes.")
    else:
        print(f"Warning: No valid segments for {word}")


def main():
    print("\n--- Starting Final PC Input Vector Generation ---")
    
    if not ALIGNMENT_DIR.exists():
        print(f"ERROR: Alignment directory not found: {ALIGNMENT_DIR}")
        sys.exit(1)
        
    words = load_user_words(INPUT_WORDS_FILE)
    print(f"Processing {len(words)} words...")

    for word in words:
        generate_pc_vector(word)

    print("\nProcessing complete. Inputs saved to PC_Input_Vectors.")

if __name__ == '__main__':
    main()