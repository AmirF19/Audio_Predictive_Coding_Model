import pandas as pd
import shutil
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "my_5_words.csv"
SOURCE_AUDIO_DIR = BASE_DIR / "audio_phonemes" / "All_Recordings"
# We create a specific staging folder where WAV and TXT will live together
MFA_INPUT_DIR = BASE_DIR / "audio_phonemes" / "MFA_Ready" 

# Create output directory if it doesn't exist
MFA_INPUT_DIR.mkdir(parents=True, exist_ok=True)

def prep_mfa_data():
    print(f"--- Starting MFA Prep ---")
    
    # 1. Load the Word List
    if not CSV_PATH.exists():
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    # Assuming the column name is 'word', verify this matches your CSV
    # If your CSV has no header, use: target_words = df.iloc[:, 0].tolist()
    target_words = df['word'].tolist() 

    # 2. Index existing audio files (Case Insensitive Map)
    # Creates a map: {'atom': 'atom.wav', 'word2': 'Word2.WAV'}
    audio_files_map = {f.stem.lower(): f.name for f in SOURCE_AUDIO_DIR.glob("*.wav")}
    
    processed_count = 0

    # 3. Match and Generate
    for word in target_words:
        clean_word = str(word).strip()
        word_key = clean_word.lower()
        
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
                
            print(f"Matched: {clean_word} -> {actual_filename}")
            processed_count += 1
        else:
            print(f"WARNING: Audio file for '{clean_word}' not found in {SOURCE_AUDIO_DIR}")

    print(f"--- Complete. Prepared {processed_count} pairs in {MFA_INPUT_DIR} ---")

if __name__ == "__main__":
    prep_mfa_data()