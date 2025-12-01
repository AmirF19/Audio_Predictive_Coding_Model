import numpy as np
import sys
from phoneme_data_loader import load_user_words, load_audio_file, BASE_DIR, PHONEME_INVENTORY, N_PHONEMES, SAMPLE_RATE, USER_WORDS_FILE
import librosa

# --- CONFIGURATION ---
MFCC_OUTPUT_DIR = BASE_DIR / "MFCC"
MFCC_OUTPUT_DIR.mkdir(exist_ok=True)

def process_mfcc(waveform, word):
    """
    Method 1: MFCC Feature Extraction (Acoustic Transparency)
    Calculates MFCCs and derives a simulated phoneme activation vector. 
    
    """
    
    # 1. MFCC Extraction 
    # n_mfcc=13 is standard; hop_length determines time resolution
    mfccs = librosa.feature.mfcc(y=waveform, sr=SAMPLE_RATE, n_mfcc=13, hop_length=160)
    
    # 2. Simplification: Calculate mean MFCC over time 
    mean_mfcc = mfccs.mean(axis=1)

    # 3. CRITICAL: Simulated Phoneme Activation from MFCC
    # *REPLACE THIS SECTION with your actual prototype similarity calculation*
    # (e.g., Euclidean distance between mean_mfcc and a prototype MFCC for each phoneme)
    
    # Simulation: Generate a graded, random vector.
    simulated_activation = np.random.uniform(0.1, 0.8, N_PHONEMES) 
    
    # Normalize the vector to sum to 1.0 (optional, but clean)
    activation_vector = simulated_activation / simulated_activation.sum()
    
    # 4. Save the result
    output_path = MFCC_OUTPUT_DIR / f"{word}_mfcc_activation.npy"
    np.save(output_path, activation_vector)
    
    print(f"  [MFCC] Processed and saved activation for '{word}' to {output_path.name} (Vector Size: {activation_vector.shape[0]})")
    return activation_vector


def main():
    if not BASE_DIR.exists():
        print(f"FATAL ERROR: Base directory not found: {BASE_DIR}")
        sys.exit(1)
        
    words = load_user_words(USER_WORDS_FILE)
    if not words:
        print("No words found to process.")
        sys.exit(1)
        
    print(f"\n--- Starting MFCC Feature Processing for {len(words)} words ---")
    
    for i, word in enumerate(words):
        print(f"\nProcessing word {i+1}/{len(words)}: '{word}'")
        
        # 1. Load Audio
        # We assume audio files are in the BASE_DIR: .../audio_phonemes/word.wav
        waveform = load_audio_file(word) 
        if waveform is None:
            print(f"Skipping '{word}': Audio file not found in {BASE_DIR}.")
            continue
            
        # 2. Run MFCC Processing
        process_mfcc(waveform, word)

    print("\n--- All MFCC Processing Complete ---")
    print(f"MFCC activation vectors saved to: {MFCC_OUTPUT_DIR}")

if __name__ == '__main__':
    main()