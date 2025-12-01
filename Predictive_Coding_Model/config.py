"""
Configuration for the Predictive Coding N400 Model
"""

import torch
from pathlib import Path

# Paths
PROJECT_ROOT = Path("C:\\Users\\Muhammad\\OneDrive\\Desktop\\comp_ling_project")
PHONEME_VECTOR_DIR = PROJECT_ROOT / "audio_phonemes" / "PC_Input_Vectors"
SEMANTIC_JSON_FILE = PROJECT_ROOT / "outputs" / "semantic_features_model_input.json"
WORD_LIST_FILE = PROJECT_ROOT / "my_800_words.csv"

OUTPUT_DIR = PROJECT_ROOT / "Predictive_Coding_Model" / "results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
RESULTS_CSV_FILE = OUTPUT_DIR / "simulation_results.csv"
RESULTS_PLOT_FILE = OUTPUT_DIR / "n400_results.png"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model dimensions
N_PHONEME_SLOTS = 15
PHONEME_VECTOR_DIM = 32
INPUT_DIM = N_PHONEME_SLOTS * PHONEME_VECTOR_DIM

# Dynamics
DT = 0.01
MAX_STEPS = 500

# Precision weights
PRECISION_CLEAR = 1.0
PRECISION_NOISY = 1.0
PRECISION_LEXICAL = 0.1
PRECISION_SEMANTIC = 1.0

# Learning rates
LEARNING_RATE_LEXICAL = 0.1
LEARNING_RATE_SEMANTIC = 0.1

# Decay
SEMANTIC_DECAY_RATE = 0.0

# Priming parameters
PRIME_SETTLE_STEPS = 200
SEMANTIC_PERSISTENCE = 0.8
RESET_LEXICAL_ON_TARGET = True
