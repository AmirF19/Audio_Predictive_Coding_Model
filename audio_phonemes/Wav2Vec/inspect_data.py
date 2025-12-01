import numpy as np
from pathlib import Path

# Load the file for 'atom'
path = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\audio_phonemes\PC_Input_Vectors\atom_pc_input.npz")
data = np.load(path)

print("--- File: atom_pc_input.npz ---")
print(f"Phonemes: {data['labels']}")
print(f"Vector Matrix Shape: {data['vectors'].shape}")
# Expected: (4, 32) or (4, 768) depending on your specific wav2vec model configuration