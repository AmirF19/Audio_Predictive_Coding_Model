import torch
import numpy as np
import sys
import pandas as pd 
import librosa 
from pathlib import Path
from transformers import AutoProcessor, Wav2Vec2ForCTC

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
SAMPLE_RATE = 16000 

# REMOVED: PHONEME_INVENTORY and N_PHONEMES (We don't need to force a size)

WAV2VEC_OUTPUT_DIR = BASE_DIR / "Wav2Vec"
WAV2VEC_OUTPUT_DIR.mkdir(exist_ok=True)
WAV2VEC_MODEL_ID = "facebook/wav2vec2-base-960h" 

# --- Point to the Full 800 Word Dataset ---
INPUT_WORDS_FILE = BASE_DIR / "my_800_words.csv"

# Check GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- IMPROVED LOADER ---
def load_user_words(filepath):
    print(f"Loading words from: {filepath}")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    df.columns = df.columns.astype(str).str.strip().str.lower()
    
    target_col = None
    if 'word' in df.columns:
        target_col = 'word'
    elif 'word, column_2' in df.columns:
        target_col = 'word, column_2'
    else:
        print(f"Warning: Column 'word' not found. Defaulting to first column: '{df.columns[0]}'")
        target_col = df.columns[0]

    words = df[target_col].astype(str).str.strip().str.lower().tolist()
    clean_words = sorted(list(set([w for w in words if w and w != 'nan'])))
    print(f"Successfully loaded {len(clean_words)} unique words.")
    return clean_words

def load_audio_file(word, audio_dir):
    # Try different capitalizations
    candidates = [f"{word}.wav", f"{word.upper()}.wav", f"{word.capitalize()}.wav"]
    filepath = None
    for c in candidates:
        if (audio_dir / c).exists():
            filepath = audio_dir / c
            break   
    if not filepath:
        return None
    try:
        waveform, sr = librosa.load(filepath, sr=SAMPLE_RATE)
        return waveform
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_wav2vec(waveform, word, model, processor):
    """
    Extracts raw logits and outputs the FULL high-resolution time series.
    No artificial slicing or inventory matching.
    """
    
    # 1. Processing and Inference
    input_values = processor(waveform, return_tensors="pt", sampling_rate=SAMPLE_RATE).input_values
    input_values = input_values.to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_values).logits 
    
    # Shape: (Time_Steps, Vocab_Size) -> Usually (T, 32) for this model
    logits = logits.squeeze(0).cpu().numpy()
    
    # 2. Normalize (Optional but recommended for PC input)
    # We turn the raw logits (negative/positive numbers) into probabilities (0.0 to 1.0)
    # This makes it much easier for the PC model to process.
    # We use Softmax for this.
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    probabilities = softmax(logits)

    # 3. Save the result
    output_path = WAV2VEC_OUTPUT_DIR / f"{word}_wav2vec_timeseries_activation.npy"
    np.save(output_path, probabilities)
    return probabilities

def main():
    if not BASE_DIR.exists():
        print(f"FATAL ERROR: Base directory not found: {BASE_DIR}")
        sys.exit(1)
        
    WAV2VEC_OUTPUT_DIR.mkdir(exist_ok=True)
    
    words = load_user_words(INPUT_WORDS_FILE)
    if not words:
        sys.exit(1)
        
    print(f"\n--- Starting Wav2Vec 2.0 Feature Processing for {len(words)} words ---")
    
    print(f"\nInitializing Wav2Vec 2.0 model on {DEVICE}...")
    try:
        processor = AutoProcessor.from_pretrained(WAV2VEC_MODEL_ID)
        model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC_MODEL_ID)
        model.to(DEVICE)
    except Exception as e:
        print(f"WARNING: Could not initialize Wav2Vec model: {e}")
        return

    audio_sub_dir = BASE_DIR / "All_Recordings"
    if not audio_sub_dir.exists():
         print(f"FATAL ERROR: Audio folder not found: {audio_sub_dir}")
         sys.exit(1)

    processed_count = 0
    for i, word in enumerate(words):
        if i % 50 == 0: print(f"Progress: {i}/{len(words)}...")
        
        waveform = load_audio_file(word, audio_dir=audio_sub_dir) 
        if waveform is None:
            continue
            
        process_wav2vec(waveform, word, model, processor)
        processed_count += 1

    print("\n--- All Wav2Vec Processing Complete ---")
    print(f"Processed {processed_count}/{len(words)} words.")
    print(f"Wav2Vec activation vectors saved to: {WAV2VEC_OUTPUT_DIR}")

if __name__ == '__main__':
    main()