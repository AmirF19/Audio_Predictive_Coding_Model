import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from pathlib import Path
import textgrid
import sys

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes")
AUDIO_DIR = BASE_DIR / "All_Recordings"
MFA_DIR = BASE_DIR / "MFA_Output_TextGrids"
OUTPUT_PLOT_DIR = BASE_DIR / "Alignment_Plots"
OUTPUT_PLOT_DIR.mkdir(exist_ok=True)

WORDS_TO_CHECK = ["atom", "cherry", "supper"] # Add words you want to inspect

def plot_alignment(word):
    audio_path = AUDIO_DIR / f"{word}.wav"
    tg_path = MFA_DIR / f"{word}.TextGrid"
    
    if not audio_path.exists() or not tg_path.exists():
        print(f"Missing data for {word}")
        return

    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=16000)
    
    # 2. Load TextGrid
    tg = textgrid.TextGrid.fromFile(str(tg_path))
    phoneme_tier = max(tg, key=len) # Assuming the granular tier is phonemes

    # 3. Plot
    plt.figure(figsize=(14, 6))
    
    # Waveform
    ax1 = plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr, alpha=0.6)
    plt.title(f"Alignment Check: {word.upper()}")
    plt.xlabel("")
    
    # Spectrogram (Optional but helpful for consonants)
    # D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    # librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', alpha=0.3)

    # Draw Boundaries
    for interval in phoneme_tier:
        if interval.mark: # Skip blanks if desired, or keep them
            plt.axvline(x=interval.minTime, color='r', linestyle='--', alpha=0.8)
            plt.text(interval.minTime + 0.01, 0.5, interval.mark, color='black', 
                     fontweight='bold', fontsize=12, rotation=0, 
                     transform=ax1.get_xaxis_transform())

    plt.tight_layout()
    save_path = OUTPUT_PLOT_DIR / f"{word}_alignment.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    plt.close()

if __name__ == '__main__':
    for word in WORDS_TO_CHECK:
        plot_alignment(word)
    print(f"Check the plots in {OUTPUT_PLOT_DIR}")