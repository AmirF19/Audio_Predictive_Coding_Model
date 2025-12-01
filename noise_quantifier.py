import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import textgrid # Requires: pip install textgrid

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
PC_INPUT_DIR = BASE_DIR / "PC_Input_Vectors"
ALIGNMENT_DIR = BASE_DIR / "MFA_Output_TextGrids" 
INPUT_WORDS_FILE = BASE_DIR.parent / "my_800_words.csv"

# Constants for Slice Calculation (Must match pc_input_generator.py)
FRAME_LENGTH_S = 0.02 

# Single Output File: Granular Phoneme Data
OUTPUT_RESULTS_FILE = BASE_DIR / "noise_similarity_phoneme_level.csv"

# --- Robust Word Loading Logic ---
def load_user_words(filepath):
    """Loads the list of words from the user's CSV."""
    print(f"Loading word list from: {filepath}")
    df = None
    try:
        df = pd.read_csv(filepath, sep='\t')
    except:
        pass 

    if df is None or len(df.columns) < 2: 
        try:
            df = pd.read_csv(filepath, sep=',')
        except:
             return []

    df.columns = df.columns.str.strip().str.lower()
    word_col_name = 'word' if 'word' in df.columns else df.columns[0]
    words = df[word_col_name].astype(str).str.strip().str.lower().tolist()
    return [w for w in sorted(list(set(words))) if w]

def load_alignment_details(word):
    """
    Loads detailed alignment info (Label, Start, End) from the TextGrid.
    Returns a list of dictionaries.
    """
    tg_path = ALIGNMENT_DIR / f"{word.lower()}.TextGrid"
    
    if not tg_path.exists():
        # Try checking for capitalized filename just in case
        tg_path = ALIGNMENT_DIR / f"{word.upper()}.TextGrid"
        if not tg_path.exists():
            return None

    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        phoneme_tier = max(tg, key=len)
        
        details = []
        for interval in phoneme_tier:
            if interval.mark.strip() and interval.mark.lower() not in ['sil', 'sp', '']:
                details.append({
                    "phoneme": interval.mark,
                    "start": interval.minTime,
                    "end": interval.maxTime
                })
        return details
    except Exception as e:
        print(f"Error reading TextGrid for {word}: {e}")
        return None

def calculate_slice_count(start_s, end_s):
    """
    Replicates the 'Safe Slicing' logic to report how many frames were used.
    """
    duration = end_s - start_s
    width_frames = duration / FRAME_LENGTH_S
    
    if width_frames < 1.0:
        return 1 # Short consonant logic (Center frame)
    else:
        start_idx = int(np.floor(start_s / FRAME_LENGTH_S))
        end_idx = int(np.ceil(end_s / FRAME_LENGTH_S))
        count = end_idx - start_idx
        return max(1, count) # Ensure at least 1 frame is reported

def quantify_noise_similarity():
    print("\n--- STARTING PHONEME-LEVEL NOISE QUANTIFICATION ---")
    
    if not PC_INPUT_DIR.exists():
        print(f"FATAL ERROR: Vector directory not found: {PC_INPUT_DIR}")
        sys.exit(1)
    
    words = load_user_words(INPUT_WORDS_FILE)
    print(f"Processing {len(words)} words...")

    all_phoneme_data = []
    
    for word in words:
        clear_path = PC_INPUT_DIR / f"{word}_pc_phoneme_input.npy"
        noisy_path = PC_INPUT_DIR / f"{word}_noisy_pc_phoneme_input.npy"
        
        if not clear_path.exists() or not noisy_path.exists():
            continue
            
        try:
            V_clear = np.load(clear_path) 
            V_noisy = np.load(noisy_path) 
            
            if V_noisy.shape != V_clear.shape:
                print(f"Shape mismatch for {word}: {V_clear.shape} vs {V_noisy.shape}")
                continue

            # Load alignment details for timestamps
            alignment = load_alignment_details(word)
            
            # Verify lengths match
            if alignment and len(alignment) != V_clear.shape[0]:
                print(f"Warning: Alignment count ({len(alignment)}) != Vector count ({V_clear.shape[0]}) for {word}. Check logic.")
                alignment = None 
                
            # --- PHONEME-BY-PHONEME COMPARISON ---
            n_phonemes = V_clear.shape[0]
            
            for i in range(n_phonemes):
                p_clear = V_clear[i].reshape(1, -1)
                p_noisy = V_noisy[i].reshape(1, -1)
                
                sim = cosine_similarity(p_clear, p_noisy)[0][0]
                
                # Get Details
                if alignment:
                    label = alignment[i]['phoneme']
                    start = alignment[i]['start']
                    end = alignment[i]['end']
                    slices = calculate_slice_count(start, end)
                else:
                    label = f"unk_{i}"
                    start = 0.0
                    end = 0.0
                    slices = 0

                all_phoneme_data.append({
                    'word': word,
                    'phoneme_position': i, 
                    'phoneme': label,
                    'cosine_similarity': sim,
                    'start_time': start,
                    'end_time': end,
                    'slice_count': slices
                })
            
        except Exception as e:
            print(f"Error processing {word}: {e}")

    if not all_phoneme_data:
        print("FATAL ERROR: No valid pairs found.")
        return

    # Save Detailed Breakdown
    df_detail = pd.DataFrame(all_phoneme_data)
    df_detail.to_csv(OUTPUT_RESULTS_FILE, index=False)
    
    mean_sim = df_detail['cosine_similarity'].mean()
    min_sim = df_detail['cosine_similarity'].min()
    avg_slices = df_detail['slice_count'].mean()
    
    print("\n" + "="*40)
    print(f"FINAL ANALYSIS (Phoneme-Level)")
    print("="*40)
    print(f"Total Phonemes Analyzed: {len(df_detail)}")
    print(f"Mean Phoneme Similarity: {mean_sim:.4f}")
    print(f"Lowest Phoneme Similarity: {min_sim:.4f}")
    print(f"Avg Frames per Phoneme:  {avg_slices:.2f}")
    print(f"Results saved to: {OUTPUT_RESULTS_FILE}")

if __name__ == '__main__':
    quantify_noise_similarity()