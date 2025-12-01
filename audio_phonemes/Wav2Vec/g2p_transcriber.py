import pandas as pd
from pathlib import Path
from g2p_en import G2p
import sys

# --- CONFIGURATION ---
# Base directory for all data
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
# Input: Your main word list (or test list)
INPUT_WORDS_FILE = BASE_DIR / "my_5_words.csv" # Use "my_800_words.csv" for final run
# Output: Transcribed phonetic sequence
OUTPUT_TRANSCRIPTIONS_FILE = BASE_DIR / "phonetic_transcriptions.csv"

# --- Shared Loading Logic (For Consistency) ---
def load_user_words(filepath):
    """Loads the list of words from the user's CSV, robustly handling separators."""
    df = None
    separators = ['\t', ',', ' '] 
    
    for sep in separators:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if len(df.columns) > 1 or 'word' in df.columns.str.lower().str.strip().tolist():
                 break
        except Exception:
            df = None

    if df is None:
        print(f"FATAL ERROR: Could not read CSV or identify separator for {filepath}")
        sys.exit(1)

    df.columns = df.columns.str.strip().str.lower()
    word_col_name = 'word' if 'word' in df.columns else None
    
    if word_col_name is None:
        if 'word, column_2' in df.columns: # Last resort fix for reported error
             word_col_name = 'word, column_2'
        else:
             print(f"FATAL ERROR: CSV must contain a column named 'word'.")
             sys.exit(1)
    
    words = df[word_col_name].astype(str).str.strip().str.lower().tolist()
    return [w for w in sorted(list(set(words))) if w]

# --- Main G2P Conversion ---
def run_g2p_transcription():
    
    words = load_user_words(INPUT_WORDS_FILE)
    if not words:
        print("No words loaded for transcription.")
        return

    print(f"\n--- Starting Grapheme-to-Phoneme (G2P) Transcription for {len(words)} words ---")
    
    # Initialize the G2P model (using CMU Pronouncing Dictionary format)
    g2p = G2p()
    
    results = []
    
    for word in words:
        try:
            # Convert the word to a list of phonemes (e.g., ['AE1', 'T', 'AH0', 'M'])
            phonemes = g2p(word)
            
            # CMU outputs a list. We join them with spaces for readability and MFA compatibility.
            transcription = ' '.join(phonemes)
            
            results.append({
                'word': word,
                'transcription': transcription,
                'phoneme_count': len(phonemes)
            })
            print(f"  {word.upper():<10} -> {transcription}")
        
        except Exception as e:
            print(f"  [ERROR] Could not transcribe {word}: {e}")
            results.append({'word': word, 'transcription': 'ERROR', 'phoneme_count': 0})

    # Save output to CSV
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_TRANSCRIPTIONS_FILE, index=False)
    
    print(f"\n--- Transcription Complete ---")
    print(f"Phonetic transcriptions saved to: {OUTPUT_TRANSCRIPTIONS_FILE}")


if __name__ == '__main__':
    run_g2p_transcription()