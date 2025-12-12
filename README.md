# Auditory N400 Predictive Coding Model

A hierarchical predictive coding implementation for auditory word recognition and N400 priming experiments, adapted from Nour Eddine et al. (2024).

## Overview

This repository contains an implementation of an auditory predictive coding model adapted from Nour Eddine et al. (2024). The model simulates N400 responses during auditory priming, particularly examining how acoustic degradation affects semantic prediction errors.

### Research Questions

1. How does acoustic degradation affect N400 priming in predictive coding models?
2. Can the model reproduce the compensatory top-down hypothesis vs. early degradation hypothesis?

### Key Results

Current implementation shows:
- **Enhanced priming under noise** (supports compensatory hypothesis)
- **N400 peaks ~60-640** (biologically plausible range after semantic scaling)
- **Recognition accuracy**: 98.6% clear, 84.0% noisy (with semantic scaling)
- **Priming effect**: ~10x ratio (different/same conditions)

#### Scaling Experiment Results:

**AUTO_SCALE_INPUT = False (Raw phoneme magnitudes):**
```
N400 by Condition:
  same/noisy: mean=40.42, peak=194.58, peak_iter=25.0
  same/clear: mean=38.63, peak=139.81, peak_iter=25.0
  different/noisy: mean=585.79, peak=934.37, peak_iter=28.0
  different/clear: mean=368.78, peak=639.60, peak_iter=27.4

Recognition Accuracy: noisy: 85.7%, clear: 98.6%
```

**AUTO_SCALE_INPUT = True (Scaled to TARGET_INPUT_NORM = 2.0):**
```
N400 by Condition:
  same/noisy: mean=40.63, peak=179.22, peak_iter=25.0
  same/clear: mean=38.91, peak=129.78, peak_iter=25.0
  different/noisy: mean=582.43, peak=937.76, peak_iter=28.1
  different/clear: mean=372.08, peak=638.83, peak_iter=27.6

Recognition Accuracy: noisy: 84.0%, clear: 98.6%
```

**AUTO_SCALE_INPUT = True + Semantic Scaling (3700/19450 ≈ 0.190):**
```
N400 by Condition:
  same/noisy: mean=11.48, peak=60.95, peak_iter=25.1
  same/clear: mean=10.24, peak=43.37, peak_iter=25.0
  different/noisy: mean=235.22, peak=642.89, peak_iter=26.3
  different/clear: mean=152.80, peak=426.37, peak_iter=26.2

Recognition Accuracy: noisy: 81.9%, clear: 98.6%
```

## Repository Structure

```
comp_ling_project/
├── Final_Predictive_Coding_Model_Aligned/    # Main implementation
│   ├── pc_model_gpu.py                       # GPU-accelerated predictive coding model
│   ├── run_simulation.py                     # Complete simulation pipeline
│   ├── requirements.txt                      # Python dependencies
│   └── analysis.py                           # Results visualization (optional)
│
├── audio_phonemes/                           # Cochlear vector representations
│   ├── Cochlear_Input_Vectors/               # 10-slot phoneme vectors (.npy)
│   └── [additional audio processing files]
│
├── semantics_alignment/                      # Semantic feature generation pipeline
│   ├── category_prompts/output/              # Final semantic features (JSON)
│   └── [semantic processing scripts]
│
├── experimental_pairs/                       # Experimental stimuli
│   └── conditions_words.csv                  # Prime-target pairs
│
└── [supporting files]
    ├── my_800_words.csv                      # Lexicon (805 words)
    ├── semantic_validator.py                 # Feature validation
    ├── noise_quantifier.py                   # Acoustic degradation tools
    └── SUBTLEX_frequency_importer.py         # Word frequency data
```

## Model Architecture

**4-Layer Hierarchical Predictive Coding:**

```
Audio Layer -> Lexical Layer -> Semantic Layer -> Conceptual Layer
   400 units    ~800 units      ~19,450 units     ~800 units
(10×40 phonemes) (words)        (binary features)  (word concepts)
```

### Dynamics

1. **Bottom-up processing**: Audio input drives lexical and semantic activation
2. **Top-down predictions**: Higher layers predict what lower layers should see
3. **Prediction errors**: Computed as ratio between actual and predicted states
4. **State updates**: Multiplicative updates minimize prediction errors
5. **N400**: Operationalized as lexical + semantic prediction error magnitude

### Key Parameters
- **Semantic scaling**: Optional scaling by 3700/19450 ≈ 0.190 for feature space normalization
- **Input scaling**: Configurable normalization of phoneme vectors
- **Noise injection**: Gaussian corruption for noisy trials

## Experimental Design

**2×2 Factorial Design** examining auditory repetition priming:

| Factor | Levels |
|--------|--------|
| Identity | Same word, Different word |
| Clarity | Clear speech, Vocoded noise |

### Trial Structure

1. **Prime phase** (20 iterations): Present prime word (always clear), model settles
2. **ISI** (5 iterations): Blank period for expectation decay
3. **Target phase** (20 iterations): Present target word (clear or noisy)
4. **N400 window**: Iterations 2-11 of target phase (~300-500ms analog)
5. **Measure N400**: Lexical + semantic prediction error magnitude

### Hypotheses Tested

- **Compensatory Top-Down**: Noise enhances priming (Hypothesis 1)
- **Early Degradation**: Noise abolishes priming (Hypothesis 2)

**Current results support Hypothesis 1**: Enhanced N400 priming under acoustic degradation.

## Installation & Setup

### Requirements
- Python 3.8+
- PyTorch (CUDA recommended for GPU acceleration)
- NumPy, Pandas, Matplotlib, tqdm

### Quick Start

```bash
# Clone repository
git clone https://github.com/AmirF19/Audio_Predictive_Coding_Model.git
cd comp_ling_project

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r Final_Predictive_Coding_Model_Aligned/requirements.txt
```

## Usage

### Run Simulation

```bash
cd Final_Predictive_Coding_Model_Aligned
python run_simulation.py
```

### Output Files

Results are saved to `results_aligned/`:
- **`simulation_results.csv`**: Trial-by-trial data
- **`n400_timecourse_all.png`**: Complete timecourse across all 4 conditions
- **`n400_timecourse_same.png`**: N400 timecourse for same-word trials only
- **`n400_timecourse_different.png`**: N400 timecourse for different-word trials only
- **`n400_bars.png`**: Mean N400 comparison across all conditions
- **`recognition_accuracy.png`**: Recognition accuracy by clarity condition


### Key Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Phoneme slots | 10 | Fixed-width input representation |
| Semantic features | ~19,450 | Binary LLM-generated features |
| Lexicon size | 805 | Words with available phoneme data |
| Simulation iterations | 20 per phase | Prime + target phases |
| Semantic scaling | 0.190 | Optional: 3700/19450 feature normalization |

### Audio Processing
- **Phoneme extraction**: Montreal Forced Aligner (MFA) for segmentation
- **Feature representation**: Cochlear-inspired 10×40 dimensional vectors
- **Noise simulation**: Additive Gaussian corruption for vocoded speech

### Semantic Processing
- **Feature generation**: LLM-based semantic attribute extraction
- **Feature selection**: Category-specific prompting with deduplication

## Data Sources

- **Audio recordings**: Clean speech corpus with MFA alignment
- **Phoneme vectors**: Cochlear preprocessing pipeline output
- **Semantic features**: LLM-generated binary attribute vectors
- **Experimental pairs**: Controlled prime-target combinations
- **Lexicon**: 805 words with complete phoneme/semantic data

## Contact
**Authors**: Muhammad Fusenig, Alba Jorquera, William Zumchak  
**Institution**: University of Maryland, Computational Linguistics Seminar  
**Course**: LING848A with Professor Philip Resnik