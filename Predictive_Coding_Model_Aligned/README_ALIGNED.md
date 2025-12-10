# Auditory N400 Predictive Coding Model - Aligned Implementation

## Overview

This directory contains a GPU-accelerated implementation of a hierarchical predictive coding model for auditory word recognition and N400 priming experiments. The implementation is aligned with Samer Nour Eddine's reference model (2024) with documented adaptations for auditory input.

## Quick Start

```bash
# Activate virtual environment (if using one)
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Run the simulation
cd Predictive_Coding_Model_Aligned
python run_simulation.py
```

**Output**: Results saved to `results_aligned/` directory (5 PNG figures + CSV with trial data)

**Configuration**: Edit parameters in `run_simulation.py` (lines 61-95):
- `SAMER_MODE`: Toggle reference alignment (True) vs experimental features (False)
- `TARGET_INPUT_NORM`: Input scaling (0.25 tuned for ~400 peak N400)
- `APPLY_PRECISION_SCALING`: Enable/disable noisy audio precision weighting

---

## File Structure

```
Predictive_Coding_Model_Aligned/
├── pc_model_gpu.py          # Core predictive coding model (GPU-accelerated)
├── run_simulation.py        # Main simulation script
├── analysis.py              # Results visualization and export
├── results_aligned/         # Output directory (created automatically)
│   ├── simulation_results.csv
│   ├── n400_timecourse_all.png
│   ├── n400_timecourse_same.png
│   ├── n400_timecourse_different.png
│   ├── n400_bars.png
│   └── recognition_accuracy.png
└── README_ALIGNED.md        # This file
```

### File Descriptions

**`pc_model_gpu.py`**
- Implements the 4-level hierarchical predictive coding model
- Layers: Audio → Lexical → Semantic → Contextual
- GPU-accelerated via PyTorch (CUDA)
- Faithful adaptation of reference model dynamics
- Batch processing for parallel trial execution

**`run_simulation.py`**
- Main execution script for priming experiments
- Loads lexicon, semantics, and auditory vectors
- Runs 2×2 factorial design (identity × clarity)
- Extracts N400 metrics and recognition accuracy
- Calls analysis module for visualization

**`analysis.py`**
- Modular visualization and results export
- Generates 5 separate figures:
  - N400 timecourse (all 4 conditions combined)
  - N400 timecourse (same condition: clear vs noisy)
  - N400 timecourse (different condition: clear vs noisy)
  - N400 bar plot (2×2 grouped bars)
  - Recognition accuracy by clarity
- Prints summary statistics to console
- Exports trial-level CSV with full traces

---

## Experimental Design

### 2×2 Factorial Design

| Factor | Levels | Description |
|--------|--------|-------------|
| **Identity** | Same / Different | Prime and target are same or different words |
| **Clarity** | Clean / Noisy | Target is clean or vocoded (noise-degraded) |

### Timeline (per trial)

1. **Prime phase** (20 iterations): Present prime word
2. **ISI** (5 blank iterations): Inter-stimulus interval
3. **Target phase** (20 iterations): Present target word
4. **Settling** (5 blank iterations): Post-target decay

### N400 Measurement

- **Metric**: Lexico-semantic prediction error (sum of lexical + semantic PE)
- **Window**: Iterations 2-11 of target phase (~300-500ms in biological time)
- **Interpretation**: High PE = poor match between input and predictions

---

## Model Architecture

### Four-Level Hierarchy

```
Contextual Layer (top)
    ↕
Semantic Layer (semantic features)
    ↕
Lexical Layer (word representations)
    ↕
Audio Layer (phoneme features)
```

### Each Level Maintains

- **State** [0]: Current activation
- **Reconstruction** [1]: Top-down prediction from higher level
- **Top-down Bias** [2]: Reconstruction / State
- **Prediction Error** [3]: State / Reconstruction

### Dynamics (per iteration)

1. **Bottom-up**: Sensory input propagates prediction errors upward
2. **State updates**: Multiplicative updates minimize prediction error
3. **Top-down**: Higher levels generate predictions (reconstructions)
4. **Convergence**: Process iterates until stable or max iterations

---

## Alignment with Reference Model

### What Matches Exactly

✅ **State initialization**: Uniform distributions, zero biases  
✅ **Weight construction**: Normalized block matrices, frequency bias  
✅ **Update dynamics**: Epsilon-guarded multiplicative updates  
✅ **Reconstruction order**: Bottom-up PE → State update → Top-down prediction  
✅ **Epsilon values**: EPSILON1=0.005, EPSILON2=0.0001  

### Documented Adaptations

#### 1. Audio Input (instead of Orthographic)

- **Reference**: 4-letter slot coding (4 slots × 26 letters = 104 dims, 4-hot)
- **Ours**: Phoneme-slot encoding (10 slots × 40 features = 400 dims, 10-hot)
- **Structure**: Each phoneme slot is 1-hot; words with <10 phonemes have trailing padding
- **Rationale**: Auditory modality, but preserves sparse slot-based encoding

#### 2. Adaptive Momentum for State Updates

- **Location**: Lines 125-131, 360-365, 417-424, 447-452, 476-482 in `pc_model_gpu.py`
- **Formula**: 
  - During input: `new_state = 0.7 × old_state + 0.3 × proposed_state` (stable processing)
  - During blanks: `new_state = 0.3 × old_state + 0.7 × proposed_state` (fast decay)
- **Rationale**: 
  1. Pure multiplicative updates oscillate during same-word repetition
  2. High baseline PE during ISI (blank input vs strong predictions)
  3. Adaptive momentum: slow during input (stability), fast during blanks (decay to baseline)
  4. Biologically: implements neural adaptation/inertia
  5. Common in recurrent predictive coding (Rao & Ballard, 1999; Spratling, 2017)

#### 3. Precision Scaling for Noisy Trials

- **Location**: Line 378 in `pc_model_gpu.py`
- **Formula**: `audio_PE = audio_PE × precision_weight`
- **Weights**: `w_clean=1.0`, `w_noisy=1.21` (based on 0.575 measured similarity)
- **Rationale**: Operationalizes perceptual difficulty of vocoded speech

#### 4. Semantic PE Normalization

- **Location**: Lines 502-509 in `pc_model_gpu.py`
- **Formula**: `semantic_PE × (3715 / 20533)`
- **Rationale**: Our semantic space (20,533 features) is 5.5× larger than reference (3,715). Without normalization, semantic PE dominates. Scaling brings N400 peaks to reference range (~400).

---

## Configuration Parameters

### Key Settings (in `run_simulation.py`)

```python
# Timeline
NUM_ITERS = 20              # Prime phase iterations
TARGET_ITERS = 20           # Target phase iterations
BLANKS_BEFORE_TARGET = 5    # ISI
POST_TARGET_BLANKS = 5      # Settling phase

# Experimental manipulations
APPLY_NOISE = True          # Inject Gaussian noise into noisy targets
APPLY_PRECISION_SCALING = True  # Weight PE by clean/noisy similarity
USE_CTX_CLAMP = False       # Clamp context to prime (False = natural dynamics)

# Input scaling
TARGET_INPUT_NORM = 0.25    # Calibrated to produce ~400 peak N400
```

### Tuning Notes

- **TARGET_INPUT_NORM**: Controls overall PE magnitude. 0.25 produces peaks ~400-750 (matching reference).
- **Dampening (in `precision_weights`)**: 0.5 moderates noisy PE amplification (0.5 = 50% of full dissimilarity effect).
- **Semantic normalization**: Fixed at 3715/20533 to match reference scale.

---

## Running the Simulation

### Prerequisites

```bash
# Python 3.8+
pip install torch numpy pandas matplotlib tqdm

# CUDA-enabled GPU recommended (tested on RTX 3090)
```

### Execution

```bash
cd Predictive_Coding_Model_Aligned
python run_simulation.py
```

### Expected Output

```
============================================================
GPU-Accelerated Auditory N400 Priming Simulation
============================================================
Using GPU: NVIDIA GeForce RTX 3090
Loaded 829/829 cochlear vectors
Semantic matrix: 20533 features × 829 words
Audio matrix: 400 dims × 829 words
Loaded frequency bias (SUBTLEX) scaled to [0, 0.1]
Loaded 744 experimental pairs

Running 744 trials in batches of 256...
Processing batches: 100%|██████████| 3/3 [00:28<00:00,  9.4s/it]

============================================================
RESULTS SUMMARY
============================================================

N400 by Condition:
  same/noisy: mean=25.07, peak=42.12, peak_iter=25.4
  same/clear: mean=20.08, peak=28.72, peak_iter=25.4
  different/noisy: mean=306.42, peak=751.01, peak_iter=26.6
  different/clear: mean=159.62, peak=338.17, peak_iter=27.1

Recognition Accuracy:
  noisy: 64.6%
  clear: 89.7%

Results saved to: results_aligned/simulation_results.csv
Plot saved to: results_aligned/n400_results.png
```

---

## Interpreting Results

### Key Effects

1. **N400 Priming Effect**: `different > same`
   - Different words produce higher PE (poor match)
   - Same words produce lower PE (good match)

2. **Clarity Effect**: `noisy > clean`
   - Noisy targets produce higher PE (degraded input)
   - Clean targets produce lower PE (clear input)

3. **Recognition Accuracy**: `clean > noisy`
   - Clean: ~90% correct word identification
   - Noisy: ~65% correct word identification

### Expected Magnitudes

- **Same condition**: N400 peak ~30-50
- **Different/clean**: N400 peak ~300-400
- **Different/noisy**: N400 peak ~600-800

These magnitudes match the reference model's range and are biologically plausible when interpreted as normalized PE.

---

## Visualization Output

### 3-Panel Figure (`results_aligned/n400_results.png`)

**Panel 1: N400 Timecourse (All 4 Conditions)**
- X-axis: Iterations from target onset (0 = target starts)
- Y-axis: Lexico-semantic PE (N400)
- Lines: Blue=same, Red=different; Solid=clear, Dashed=noisy
- Shows temporal dynamics and peak-and-settle pattern

**Panel 2: N400 Mean (2×2 Grouped Bars)**
- X-axis: Same vs Different
- Bars: Clear (blue) vs Noisy (orange)
- Shows condition × clarity interaction

**Panel 3: Recognition Accuracy**
- Bars: Clear vs Noisy
- Shows intelligibility difference

---

## Technical Details

### GPU Acceleration

- **Batch size**: 256 trials processed in parallel
- **Speed**: ~800 trials in ~30 seconds (RTX 3090)
- **Memory**: ~2GB VRAM for typical lexicon (800 words)

### Numerical Stability

- **Epsilon guards**: Prevent division by zero and state collapse
- **Multiplicative updates**: Enforce positivity (no negative activations)
- **Normalized weights**: Prevent runaway activation growth

### Reproducibility

- **Random seed**: Set in script for noise injection reproducibility
- **Deterministic ops**: PyTorch CUDA operations are deterministic when possible
- **Input normalization**: L2 normalization ensures consistent drive

---

## Comparison to Reference Model

| Aspect | Reference (Samer 2024) | Current Implementation |
|--------|----------------------|------------------------|
| **Input modality** | Orthographic (4-hot) | Auditory (5-hot) |
| **Input dims** | 104 | 400 |
| **Lexicon size** | 1568 words | 829 words |
| **Semantic features** | 3,715 | 20,533 |
| **Weight construction** | ✅ Matched | ✅ Matched |
| **State updates** | ✅ Matched | ✅ Matched |
| **Epsilon guards** | ✅ Matched | ✅ Matched |
| **Frequency bias** | ✅ Matched | ✅ Matched |
| **N400 formula** | sum(lex PE + sem PE) | sum(lex PE + 0.181×sem PE) |
| **Precision scaling** | N/A | ⭐ Added (for noisy) |
| **Context clamp** | Optional | Optional (off by default) |

---

## Citation

If using this implementation, please cite:

1. **Reference model**: Samer Nour Eddine's predictive coding model (2024)
2. **Cochlear processing**: Our auditory feature extraction pipeline
3. **Semantic features**: LLaMA-derived semantic representations

---

## Contact & Support

For questions about the implementation or experimental design, contact the lab.

**Last updated**: December 2025

