import torch
import numpy as np
import pandas as pd
import librosa
import textgrid
from pathlib import Path
from transformers import AutoProcessor, Wav2Vec2ForCTC
import sys
import os

# --- CONFIGURATION (Uses the same base paths as pc_input_generator.py) ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
NOISY_AUDIO_DIR = BASE_DIR / "Noisy_Recordings" # Input: Folder with noisy .wav files
ALIGNMENT_DIR = BASE_DIR / "MFA_Output_TextGrids" # Input: Existing clean alignments
FINAL_PC_INPUT_DIR = BASE_DIR / "PC_Input_Vectors" # Output destination

# Input words (Pointing to the main 800 word list)
INPUT_WORDS_FILE = BASE_DIR.parent / "my_800_words.csv"
MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000 
FRAME_LENGTH_S = 0.02 

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_user_words(filepath):
    """Robustly loads the word list."""
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

def process_noisy_word(word, model, processor):
    # 1. Load Noisy Audio
    audio_path = NOISY_AUDIO_DIR / f"{word}.wav"
    if not audio_path.exists():
        # print(f"Skipping {word}: Noisy audio not found.")
        return

    try:
        waveform, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return

    # 2. Get Raw Wav2Vec Logits (T x V)
    inputs = processor(waveform, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values.to(DEVICE)
    with torch.no_grad():
        logits = model(inputs).logits.squeeze(0).cpu().numpy()

    # 3. Load Existing Alignment (TextGrid)
    # Uses the CLEAN alignment data (MFA)
    tg_path = ALIGNMENT_DIR / f"{word}.TextGrid"
    if not tg_path.exists():
        # Try capitalized filename
        tg_path = ALIGNMENT_DIR / f"{word.upper()}.TextGrid"
        if not tg_path.exists():
            return

    try:
        tg = textgrid.TextGrid.fromFile(str(tg_path))
        phoneme_tier = max(tg, key=len)
        alignment_data = []
        for interval in phoneme_tier:
            if interval.mark.strip() and interval.mark.lower() not in ['sil', 'sp', '']:
                alignment_data.append({'s': interval.minTime, 'e': interval.maxTime})
    except:
        return

    # 4. Center-Weighted Slicing (Solution 3) - Uses logic from pc_input_generator.py
    pc_vectors = []
    
    for seg in alignment_data:
        duration = seg['e'] - seg['s']
        center = seg['s'] + (duration / 2)
        
        center_frame = int(round(center / FRAME_LENGTH_S))
        width_frames = duration / FRAME_LENGTH_S
        
        # Slicing logic to handle short consonants vs long vowels
        if width_frames < 1.0:
            idx = min(max(0, center_frame), logits.shape[0] - 1)
            raw_vec = logits[idx]
        else:
            start_idx = max(0, int(np.floor(seg['s'] / FRAME_LENGTH_S)))
            end_idx = min(logits.shape[0], int(np.ceil(seg['e'] / FRAME_LENGTH_S)))
            
            if end_idx > start_idx:
                raw_vec = logits[start_idx:end_idx].mean(axis=0)
            else:
                idx = min(max(0, start_idx), logits.shape[0] - 1)
                raw_vec = logits[idx]

        # Normalize to probability-like distribution
        vec = np.maximum(0, raw_vec)
        if vec.sum() > 0: vec /= vec.sum()
        pc_vectors.append(vec)

    # 5. Save with _noisy Suffix
    if pc_vectors:
        final_array = np.stack(pc_vectors)
        out_path = FINAL_PC_INPUT_DIR / f"{word}_noisy_pc_phoneme_input.npy"
        np.save(out_path, final_array)
        # print(f"Generated NOISY input for {word}: {final_array.shape}")

def main():
    print("--- Processing Noisy Condition ---")
    if not NOISY_AUDIO_DIR.exists():
        print(f"FATAL ERROR: Noisy recordings folder not found: {NOISY_AUDIO_DIR}")
        print("Please ensure your vocoded audio files are in this folder.")
        sys.exit(1)

    # Initialize Model
    print(f"Loading Wav2Vec on {DEVICE}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(DEVICE)

    words = load_user_words(INPUT_WORDS_FILE)
    print(f"Processing {len(words)} words...")

    for word in words:
        process_noisy_word(word, model, processor)

    print("\nDone. Noisy vectors saved to PC_Input_Vectors.")

if __name__ == '__main__':
    main()