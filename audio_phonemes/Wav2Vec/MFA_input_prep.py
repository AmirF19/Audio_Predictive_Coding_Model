import pandas as pd
import shutil
import sys
from pathlib import Path

# --- Configuration (ABSOLUTE PATHS) ---
# This forces the script to always find the project root correctly
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")

# 1. Point to the FULL 800 word list
CSV_PATH = BASE_DIR / "audio_phonemes" / "my_800_words.csv"

# 2. Define Audio Source and Staging Output
SOURCE_AUDIO_DIR = BASE_DIR / "audio_phonemes" / "All_Recordings"
MFA_INPUT_DIR = BASE_DIR / "audio_phonemes" / "MFA_Ready" 

# Create output directory if it doesn't exist
MFA_INPUT_DIR.mkdir(parents=True, exist_ok=True)

def prep_mfa_data():
    print(f"--- Starting MFA Prep ---")
    print(f"Project Root: {BASE_DIR}")
    print(f"Reading CSV: {CSV_PATH}")
    
    # 1. Load the Word List
    if not CSV_PATH.exists():
        print(f"FATAL ERROR: CSV not found at {CSV_PATH}")
        print("Please check that 'my_800_words.csv' is in the 'audio_phonemes' folder.")
        return
    
    # Robust CSV Loading (Handles 'word' or 'word, column_2' issues)
    try:
        df = pd.read_csv(CSV_PATH)
        # normalize headers
        df.columns = df.columns.astype(str).str.strip().str.lower()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Find the word column
    target_col = None
    if 'word' in df.columns:
        target_col = 'word'
    elif 'word, column_2' in df.columns:
        target_col = 'word, column_2'
    else:
        # Fallback to first column
        target_col = df.columns[0]
        print(f"Warning: 'word' header not found. Using first column: '{target_col}'")

    target_words = df[target_col].astype(str).str.strip().tolist()
    
    # 2. Index existing audio files (Case Insensitive Map)
    # Creates a map: {'atom': 'atom.wav', 'word2': 'Word2.WAV'}
    print(f"Scanning audio files in {SOURCE_AUDIO_DIR}...")
    if not SOURCE_AUDIO_DIR.exists():
        print(f"Error: Source audio directory does not exist.")
        return

    audio_files_map = {f.stem.lower(): f.name for f in SOURCE_AUDIO_DIR.glob("*.wav")}
    
    processed_count = 0

    # 3. Match and Generate
    for word in target_words:
        clean_word = str(word).strip()
        word_key = clean_word.lower()
        
        if not word_key: continue # skip empty lines

        if word_key in audio_files_map:
            actual_filename = audio_files_map[word_key]
            source_file = SOURCE_AUDIO_DIR / actual_filename
            
            # Destination paths
            dest_wav = MFA_INPUT_DIR / actual_filename
            dest_txt = MFA_INPUT_DIR / f"{source_file.stem}.txt"
            
            # A. Copy audio to staging folder
            shutil.copy2(source_file, dest_wav)
            
            # B. Write the text transcription file
            # MFA expects the orthographic word (e.g., "ATOM") in the text file
            with open(dest_txt, "w", encoding="utf-8") as f:
                f.write(clean_word.upper())
                
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Prepared {processed_count} files...")
        else:
            # Optional: Uncomment to see missing files
            # print(f"WARNING: Audio file for '{clean_word}' not found.")
            pass

    print(f"--- Complete. Prepared {processed_count} pairs in {MFA_INPUT_DIR} ---")

if __name__ == "__main__":
    prep_mfa_data()