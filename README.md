# Predictive Coding Model of Audio Input & N400

A computational model simulating the N400 ERP component during auditory word recognition, based on hierarchical predictive coding principles.

## Overview

This project implements a predictive coding model adapted from Nour Eddine et al. (2024) for auditory priming experiments. The model simulates how the brain processes spoken words and generates prediction errors that correspond to the N400 ERP component.

### Research Question

Can a predictive coding model explain the pattern of N400 responses in auditory priming, particularly when speech is degraded (noisy)?

### Key Finding

The model successfully demonstrates:
- **Repetition priming** reduces N400 response (Same word: lowest prediction error)
- **Semantic mismatch** increases N400 response (Different words: higher prediction error)
- **Noise disrupts priming** (Degraded speech abolishes the N400 difference across conditions)

## Project Structure

```
comp_ling_project/
├── Predictive_Coding_Model/           # Main model (Llama semantic features)
│   ├── config.py                      # Model parameters
│   ├── data_loader.py                 # Data loading functions
│   ├── pc_model.py                    # Predictive coding model
│   ├── simulation.py                  # Main experiment script
│   └── results/                       # Output files
│
├── Predictive_Coding_Model_Word2Vec/  # Alternative model (Word2Vec semantics)
│   ├── config.py
│   ├── data_loader.py
│   ├── pc_model.py
│   ├── simulation.py
│   └── results/
│
├── audio_phonemes/                    # Audio processing pipeline
│   ├── PC_Input_Vectors/              # Wav2Vec phoneme embeddings (.npy)
│   ├── All_Recordings/                # Original audio files
│   ├── Noisy_Recordings/              # Vocoded/degraded audio
│   └── MFA_Output_TextGrids/          # Forced alignment data
│
├── outputs/                           # Semantic feature data
│   └── semantic_features_model_input.json  # Llama-generated features
│
└── my_800_words.csv                   # Lexicon
```

## Model Architecture

```
Input Layer (Phoneme)     →    Hidden Layer (Lexical)    →    Top Layer (Semantic)
   480 units                      ~800 units                    ~13,000 units
   (Wav2Vec features)             (One per word)                (Semantic features)
```

### Dynamics

1. **Top-down predictions**: Higher layers predict lower layer states
2. **Prediction errors**: Mismatch between prediction and actual state
3. **State updates**: Gradient descent minimizes prediction error
4. **N400**: Operationalized as semantic prediction error

## Experimental Design

2x2 factorial design simulating an auditory priming experiment:

| Factor | Levels |
|--------|--------|
| Semantic Similarity | Same (repetition), Similar, Dissimilar |
| Auditory Clarity | Clear, Noisy (degraded) |

### Trial Structure

1. **Prime phase**: Present prime word (always clear), let model settle
2. **Store expectation**: Save semantic state as prediction
3. **Target phase**: Present target word (clear or noisy)
4. **Measure N400**: Compute mismatch between expected and actual semantics

## Installation

### Requirements

- Python 3.10+
- PyTorch (with CUDA for GPU acceleration)
- NumPy, Pandas, Matplotlib
- tqdm
- gensim (for Word2Vec version only)

### Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch numpy pandas matplotlib tqdm

# For Word2Vec version
pip install gensim
```

## Usage

### Run Llama Semantics Version

```bash
cd Predictive_Coding_Model
python simulation.py
```

### Run Word2Vec Version

```bash
cd Predictive_Coding_Model_Word2Vec
python simulation.py
```

### Output

- `results/simulation_results.csv`: Trial-level data
- `results/n400_results.png`: Visualization of results

## Results Summary

### Llama Semantics Model

| Condition | Clear N400 | Noisy N400 | Clear Accuracy |
|-----------|------------|------------|----------------|
| Same | 0.133 | 0.184 | 90% |
| Similar | 0.327 | 0.204 | 84% |
| Dissimilar | 0.303 | 0.200 | 90% |

### Word2Vec Model

| Condition | Clear N400 | Noisy N400 | Clear Accuracy |
|-----------|------------|------------|----------------|
| Same | 0.013 | 0.019 | 90% |
| Similar | 0.026 | 0.021 | 91% |
| Dissimilar | 0.024 | 0.020 | 90% |

Both models show:
- Priming effect in clear speech (Same < Similar/Dissimilar)
- Abolished priming in noisy speech (flat pattern)

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| DT | 0.01 | Integration timestep |
| MAX_STEPS | 500 | Simulation steps per word |
| PRIME_SETTLE_STEPS | 200 | Steps for prime processing |
| SEMANTIC_PERSISTENCE | 0.8 | Priming strength (0-1) |
| PRECISION_CLEAR | 1.0 | Input precision for clear speech |
| PRECISION_NOISY | 1.0 | Input precision for noisy speech |

## Data Sources

- **Phoneme vectors**: Wav2Vec 2.0 embeddings of audio recordings
- **Semantic features (Llama)**: Generated by Llama 3.1 (binary features)
- **Semantic features (Word2Vec)**: Google News Word2Vec (300-dim embeddings)
- **Lexicon**: 829 English words

## References

Nour Eddine, S., Brothers, T., Wang, L., Spratling, M., & Kuperberg, G. R. (2024). A predictive coding model of the N400. *Cognition*, 246, 105755.

## Authors

Alba Jorquera, Muhammad Fusenig, William Zumchak  
University of Maryland, Ling 848A - Computational Linguistics Seminar w/ Philip Resnik

## License

MIT License

