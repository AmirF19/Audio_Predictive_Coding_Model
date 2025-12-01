import shutil
import os
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
SOURCE_DIR = BASE_DIR / "All_Recordings"
DEST_DIR = BASE_DIR / "Noisy_Recordings"

# The prefix used in your dataset for the noisy condition
NOISY_PREFIX = "vocoded_8band_"

def organize_files():
    print("--- Starting Organization of Vocoded Files ---")
    
    if not SOURCE_DIR.exists():
        print(f"FATAL ERROR: Source directory not found: {SOURCE_DIR}")
        return

    # Create (or clean) the destination folder
    if DEST_DIR.exists():
        print(f"Warning: Destination folder {DEST_DIR} already exists.")
        print("Existing files with the same name will be overwritten.")
    DEST_DIR.mkdir(exist_ok=True)

    # Find all vocoded files
    all_files = list(SOURCE_DIR.glob(f"{NOISY_PREFIX}*.wav"))
    print(f"Found {len(all_files)} vocoded files in source directory.")

    if len(all_files) == 0:
        print(f"No files found starting with '{NOISY_PREFIX}'. Please check the filenames.")
        return

    success_count = 0
    
    for src_file in tqdm(all_files, desc="Copying and Renaming"):
        try:
            # Determine new filename (remove the prefix)
            # e.g., "vocoded_8band_bishop.wav" -> "bishop.wav"
            new_filename = src_file.name.replace(NOISY_PREFIX, "")
            dst_file = DEST_DIR / new_filename
            
            # Copy and rename
            shutil.copy2(src_file, dst_file)
            success_count += 1
            
        except Exception as e:
            print(f"Error processing {src_file.name}: {e}")

    print("\n--- Organization Complete ---")
    print(f"Successfully prepared {success_count} noisy audio files.")
    print(f"Location: {DEST_DIR}")
    print("\nNEXT STEP: Run 'python process_noisy_condition.py'")

if __name__ == '__main__':
    organize_files()