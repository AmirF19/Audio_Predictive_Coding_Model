import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
USER_WORDS_FILE = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project") / "my_800_words.csv"
SAMPLE_RATE = 16000 

# Sample phoneme set (replace with your actual phoneme inventory if known)
PHONEME_INVENTORY = ['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 'm', 'n', 'l', 'r', 'iy', 'ih', 'eh', 'ae', 'aa', 'ao', 'uh', 'uw', 'ah', 'er']
PHONEME_MAP = {p: i for i, p in enumerate(PHONEME_INVENTORY)}
N_PHONEMES = len(PHONEME_INVENTORY)

def load_user_words(filepath):
    """Loads the list of words from the user's CSV."""
    try:
        # Tries reading with tab, then comma separator for robustness
        try:
            df = pd.read_csv(filepath, sep='\t')
        except:
            df = pd.read_csv(filepath, sep=',')
        
        df.columns = df.columns.str.strip().str.lower()
        word_col_name = 'word' if 'word' in df.columns else None
        
        if word_col_name is None:
            print(f"FATAL ERROR: CSV must contain a column named 'word'. Columns found: {df.columns.tolist()}")
            sys.exit(1)
            
        words = df[word_col_name].astype(str).str.strip().str.lower().tolist()
        return [w for w in sorted(list(set(words))) if w]
        
    except FileNotFoundError:
        print(f"FATAL ERROR: User word list not found at '{filepath}'.")
        sys.exit(1)


def load_audio_file(word, audio_dir=BASE_DIR):
    """Loads an audio file and returns the waveform."""
    filepath = audio_dir / f"{word}.wav"
    if not filepath.exists():
        return None
    
    try:
        # librosa handles robust loading and resampling to 16kHz
        waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        return waveform
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

if __name__ == '__main__':
    # Ensure this file is only run as a module, not standalone
    print("This file contains shared helper functions and configuration.")