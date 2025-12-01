import numpy as np
from pathlib import Path

# Adjust filename if you want to check a different word
file_path = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes\PC_Input_Vectors\atom_pc_phoneme_input.npy")

if file_path.exists():
    data = np.load(file_path)
    print(f"\n--- Checking {file_path.name} ---")
    print(f"Shape: {data.shape} (Phonemes x Features)")
    
    # Check for zero vectors (the "vanishing consonant" bug)
    # A zero vector means the slice was empty
    zero_vectors = np.where(~data.any(axis=1))[0]
    
    if len(zero_vectors) > 0:
        print(f"WARNING: Indices {zero_vectors} are zero vectors! The consonant logic failed.")
    else:
        print("SUCCESS: No zero vectors found. All phonemes have acoustic data!")
        print("First phoneme vector sample (first 5 vals):", data[0][:5])
else:
    print(f"File not found: {file_path}")