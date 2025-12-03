 Methods: Hierarchical Predictive Coding Model of Auditory Word Recognition and N400 Simulation

**Authors**: Alba Jorquera, Muhammad Fusenig, William Zumchak  
**Institution**: University of Maryland, Ling 848A - Computational Linguistics Seminar  
**Advisor**: Philip Resnik  
**Date**: December 2025

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Theoretical Foundation: The Nour Eddine et al. (2024) Model](#2-theoretical-foundation-the-nour-eddine-et-al-2024-model)
3. [Modifications to the Original Architecture](#3-modifications-to-the-original-architecture)
4. [Methodology Part I: Phoneme Input Layer](#4-methodology-part-i-phoneme-input-layer)
5. [Methodology Part II: Semantic Feature Engineering](#5-methodology-part-ii-semantic-feature-engineering)
6. [Model Implementation](#6-model-implementation)
7. [Experimental Design](#7-experimental-design)
8. [Results Summary](#8-results-summary)
9. [Technical Specifications](#9-technical-specifications)
10. [File Manifest](#10-file-manifest)

---

## 1. Project Overview

### 1.1 Research Question

This project addresses the following question: **Can a predictive coding model explain the pattern of N400 event-related potential (ERP) responses in auditory semantic priming, particularly when speech is degraded?**

The N400 is a negative-going ERP component peaking approximately 400ms after stimulus onset, widely interpreted as reflecting the ease or difficulty of semantic integration. Our goal was to create a computational model that:

1. Processes realistic acoustic input (not idealized symbolic representations)
2. Maps phonetic representations (Layer 1) to lexical units (Layer 2) to semantic features (Layer 3)
3. Generates prediction errors that correlate with observed N400 amplitudes
4. Demonstrates how noise disrupts semantic priming effects

### 1.2 Connection to Experimental Work

This model was developed to simulate findings from Zumchak et al.'s (2025) auditory priming study, which employed a 2×2 factorial design crossing:
- **Semantic Similarity**: Same word (repetition) vs. Semantically similar vs. Dissimilar
- **Auditory Clarity**: Clear speech vs. Vocoded/degraded speech

The key empirical finding: In clear speech, N400 amplitude varies with semantic relatedness (reduced N400 for repeated/related words). In degraded speech, this N400 differentiation is abolished, presumably because the system cannot reliably identify the word to benefit from priming.

### 1.3 Summary of Results

The Llama-generated semantic features achieved the following alignment with human semantic norms (McRae et al., 2005):
- **Mean Cosine Similarity (Recall)**: **0.693**
- **Mean F1 Score**: 0.653 (Precision: 0.619, Recall: 0.693)

The predictive coding model qualitatively reproduces the expected N400 pattern: priming effects in clear speech that are abolished under noisy conditions.

---

## 2. Theoretical Foundation: The Nour Eddine et al. (2024) Model

### 2.1 Original Architecture

Our model is adapted from Nour Eddine, S., Brothers, T., Wang, L., Spratling, M., & Kuperberg, G. R. (2024). "A predictive coding model of the N400." *Cognition*, 246, 105755.

The original model implements a three-layer hierarchical predictive coding network for visual word processing:

```
Orthographic Layer → Lexical Layer → Semantic Layer
```

Key principles from the original:
1. **Hierarchical Prediction**: Higher layers predict lower layer states
2. **Prediction Error**: Mismatch between prediction and actual state drives learning
3. **N400 Operationalization**: N400 amplitude corresponds to semantic prediction error

### 2.2 Original Input Representation

Samer Nour Eddine's original model used:
- **Idealized letter inputs**: Binary activation vectors representing letter identity at each position
- **Perfect input**: No ambiguity—each letter was represented with full activation (1.0)
- **Orthographic-to-Lexical mapping**: Direct, deterministic connections

### 2.3 Limitations for Auditory Processing

The original model could not handle:
- Graded acoustic evidence (phonemes are not binary in real speech)
- Acoustic ambiguity (multiple phonemes may be partially activated)
- Degraded speech (noise, vocoding, coarticulation effects)

---

## 3. Modifications to the Original Architecture

### 3.1 Phoneme Layer Replacement

**Original**: Orthographic input (letter vectors)  
**Modified**: Phonemic input derived from Wav2Vec 2.0 acoustic processing

We replaced the idealized binary letter representations with probabilistic phoneme activation vectors extracted from real audio recordings using a pretrained acoustic model.

### 3.2 Architecture Overview

```
Input Layer (Phoneme)     →    Hidden Layer (Lexical)    →    Top Layer (Semantic)
   480 units                      ~800 units                    ~13,000 units
   (Wav2Vec features)             (One per word)                (Semantic features)
   15 slots × 32 dims
```

### 3.3 Key Modifications to the PC Dynamics

The core predictive coding dynamics remained faithful to Nour Eddine et al. (2024):

```python
# Top-down predictions
pred_lexical = V_SL.T @ state_semantic
pred_phoneme = V_LP @ state_lexical

# Prediction errors (precision-weighted)
error_phoneme = (input_u - pred_phoneme) * pi_input
error_lexical = (state_lexical - pred_lexical) * PRECISION_LEXICAL
error_semantic = (state_semantic - SEMANTIC_DECAY_RATE) * PRECISION_SEMANTIC

# State updates via gradient descent
drive_lex = V_LP.T @ error_phoneme
delta_lex = drive_lex - error_lexical
state_lexical += DT * LEARNING_RATE_LEXICAL * delta_lex

drive_sem = V_SL @ error_lexical
delta_sem = drive_sem - error_semantic
state_semantic += DT * LEARNING_RATE_SEMANTIC * delta_sem
```

### 3.4 Priming Mechanism

We implemented semantic persistence to model priming effects:

```python
# After prime processing, store semantic expectation
semantic_expectation = state_semantic.clone()

# Before target processing, partially decay semantic state (not fully reset)
state_semantic = state_semantic * SEMANTIC_PERSISTENCE  # Default: 0.8
```

The N400 is computed as the mismatch between expected and actual semantic states:
```python
N400 = |semantic_expectation - state_semantic|.sum()
```

---

## 4. Methodology Part I: Phoneme Input Layer

### 4.1 Design Rationale

We rejected manual weighting approaches (e.g., "set incorrect phonemes to 0.3") and instead used Wav2Vec 2.0 to extract acoustic feature vectors that preserve natural phonetic ambiguity.

### 4.2 The Pipeline Architecture

#### Step 1: Audio Recording Collection

- **Source**: ~800 English words recorded by a native speaker
- **Format**: WAV files at 16kHz sampling rate
- **Conditions**: Clear recordings + Vocoded/degraded versions

#### Step 2: Acoustic Feature Extraction (Wav2Vec 2.0)

**Tool**: `facebook/wav2vec2-base-960h` (Hugging Face Transformers)

**Rationale**: Wav2Vec produces probability distributions over the phonemic inventory. Unlike binary representations, these probabilities allow the model to represent graded acoustic evidence—important for handling coarticulation and degraded speech. Note: We use these vectors as static phoneme-position features, not as contextualized representations for the downstream PC model.

**Process**: Audio waveforms are passed through the Wav2Vec model to extract logits, which are then converted to probabilities via softmax.

**Output**: High-resolution time series of phoneme probability vectors (~20ms per frame, 32 dimensions per frame)

**Script**: `audio_phonemes/Wav2Vec/wav2vec_processor.py`

#### Step 3: Grapheme-to-Phoneme Transcription (G2P)

**Tool**: `g2p_en` (Python library using CMU Pronouncing Dictionary)

**Purpose**: Generate ground-truth phoneme sequences for alignment

**Example**:
```
"ATOM" → ['AE1', 'T', 'AH0', 'M']
"CHERRY" → ['CH', 'EH1', 'R', 'IY0']
```

**Script**: `audio_phonemes/Wav2Vec/g2p_transcriber.py`

#### Step 4: Forced Alignment (Montreal Forced Aligner)

**Tool**: Montreal Forced Aligner (MFA)

**Purpose**: Raw Wav2Vec output is a continuous time series. The PC model requires discrete input slots (P₁, P₂, ..., Pₙ) corresponding to phoneme positions. MFA provides precise time boundaries for each phoneme in the audio recording.

**Output**: TextGrid files containing start and end times for each phoneme segment

**Script**: `MFA_input_prep.py` (prepares audio and transcription files for MFA)

#### Step 5: PC Input Vector Generation

**Method**: Combined Wav2Vec time series (Step 2) with MFA time boundaries (Step 4). For each phoneme segment, we averaged the Wav2Vec frames within that time window to produce a single activation vector. Short consonants (< 20ms) used a single center frame instead of averaging.

**Output**: `.npy` files with shape `(N_phonemes, 32)` for each word

**Script**: `audio_phonemes/Wav2Vec/pc_input_generator.py` (contains the center-weighted averaging algorithm)

### 4.3 Noise Quantification

To validate that the degraded audio produced meaningfully different representations:

**Metric**: Cosine similarity between clear and noisy phoneme vectors

**Script**: `noise_quantifier.py`

**Results**: Phoneme-level similarity analysis stored in `noise_similarity_phoneme_level.csv`

### 4.4 Final Input Representation

Each word is represented as a matrix of shape `(15, 32)`:
- **15**: Maximum phoneme slots (padded if shorter)
- **32**: Wav2Vec vocabulary dimension

This is flattened to a 480-dimensional vector for input to the PC model, then L2-normalized (to ensure all words have unit length vectors, making comparisons based on direction rather than magnitude).

For noisy conditions, additional Gaussian noise (σ=0.2) is added post-normalization to further degrade the signal, simulating the perceptual difficulty of processing vocoded speech.

---

## 5. Methodology Part II: Semantic Feature Engineering

### 5.1 Design Rationale

The semantic layer requires a feature-based representation mapping each word to a set of semantic properties. We chose to generate these using a large language model (LLM) rather than relying solely on existing norms (which cover a limited vocabulary) or purely distributional embeddings (which lack interpretability).

### 5.2 Validation Against Human Norms

**Source**: McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). "Semantic feature production norms for a large set of living and nonliving things." *Behavior Research Methods*, 37(4), 547-559.

We extracted features for 98 words overlapping between our lexicon and the McRae norms to use as our validation benchmark (`outputs/mcrae_gold_standard.json`).

### 5.3 The Iterative Optimization Process

#### Attempt 1: Baseline Prompting

**Prompt Strategy**: Asked the LLM for "visual scenes and objects associated with the concept"

**Result**: F1 Score ≈ 0.04

**Issue**: The model generated contextual associations ("leash on hook") while humans generate intrinsic properties ("has a tail").

#### Attempt 2: Embedding-Based Validation

**Upgrade**: Used Sentence-Transformers (`all-MiniLM-L6-v2`) to compute semantic similarity between LLM-generated features and human features. This allows synonymous features to be recognized as matches.

**Metrics**: 
- **Precision**: Average max similarity from each LLM feature to any human feature
- **Recall**: Average max similarity from each human feature to any LLM feature
- **F1**: Harmonic mean of precision and recall

**Result**: F1 jumped to ~0.58, confirming the LLM was generating semantically relevant content.

**Script**: `semantics_alignment/semantic_validator.py`

#### Attempt 3: Model and Temperature Sweep

**Models Compared**:
- Llama 3.1 8B Instruct
- Qwen 2.5 (alternative)

**Temperatures Tested**: [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

**Results** (Llama 3.1, 50-word validation subset):

| Temperature | Mean F1 | Mean Precision | Mean Recall |
|-------------|---------|----------------|-------------|
| **0.5**     | **0.656** | 0.617        | **0.702**   |
| 0.7         | 0.647   | 0.607          | 0.695       |
| 0.9         | 0.646   | 0.603          | 0.697       |
| 0.3         | 0.645   | 0.607          | 0.690       |
| 1.1         | 0.643   | 0.601          | 0.694       |
| 1.3         | 0.635   | 0.587          | 0.692       |
| 1.5         | 0.611   | 0.570          | 0.665       |

**Selection**: Temperature 0.5 was optimal for balancing diversity and accuracy.

#### Attempt 4: McRae-Style Prompt Engineering

**Strategy**: Replaced generic instructions with the exact instructions from McRae et al. (2005).

**System Prompt**:
```
You are a participant in a psycholinguistic study focused on semantic associations.
Your task is to generate semantic properties for a given concept, as if you were 
a human participant providing data for psycholinguistic norms.

Instructions (based on McRae et al., 2005):
1. List specific properties of the concept
2. Include: physical properties (parts, appearance, sounds, smells, feels, tastes);
   functional properties (uses, users, locations); categorical membership;
   behavioral properties; origins
3. Treat all words as nouns only
4. Generate exactly 25 distinct properties
5. Respond ONLY with valid JSON: {"features": ["property1", "property2", ...]}

Examples:
duck: is a bird, is an animal, waddles, flies, migrates, lays eggs, quacks...
cucumber: is a vegetable, has green skin, has seeds inside, is cylindrical...
stove: is an appliance, produces heat, has elements, made of metal...
```

#### Attempt 5: Few-Shot Example Optimization

**Method**: We ran `optimize_prompt_examples.py` to mathematically select the best few-shot examples.

**Algorithm**:
1. Generate 75 features per word (3 runs × 25 features)
2. Score each feature against McRae norms using embedding similarity
3. Select top 4 words where the model achieved highest alignment:
   - **Kettle** (Avg Score: 0.834)
   - **Apple** (Avg Score: 0.795)
   - **Chicken** (Avg Score: 0.780)
   - **Lemon** (Avg Score: 0.749)

These high-performing examples were used as few-shot demonstrations in the final prompt.

#### Attempt 6: Feature Deduplication and Selection

**Problem**: Raw LLM output contained redundant features ("is-red", "red-color", "has-red-color").

**Solution**: Implemented greedy deduplication that iteratively selects features only if their embedding similarity to already-selected features is below a threshold (0.75). This ensures the final feature set is diverse.

**Script**: `semantics_alignment/feature_selector.py`

### 5.4 Final Generation Pipeline

**Model**: `meta-llama/Llama-3.1-8B-Instruct`

**Hardware**: NVIDIA RTX 3090 (24GB VRAM)

**Precision**: bfloat16 (native, no quantization)

**Parameters**:
- Temperature: 0.5
- Runs per word: 3
- Raw features per run: 25
- Final features per word: 20 (after deduplication)
- Token limit: 512

**Script**: `semantics_alignment/feature_generator_final.py`

### 5.5 Final Validation Results

**Validation Set**: 98 overlapping words between our lexicon and McRae norms

**Final Metrics**:
| Metric | Score |
|--------|-------|
| Mean F1 Score | **0.653** |
| Mean Precision | 0.619 |
| Mean Recall | **0.693** |

**Top-Performing Words** (F1 > 0.75):
- kettle (0.803)
- oven (0.777)
- lobster (0.777)
- hammer (0.744)
- bedroom (0.743)
- cellar (0.739)
- lemon (0.736)
- orange (0.731)
- eagle (0.730)

**Lower-Performing Words** (F1 < 0.50):
- cigar (0.475)
- football (0.479)
- mirror (0.488)
- rocket (0.489)

The lower scores for certain words reflect either:
- Cultural/regional differences in conceptualization (football = American vs. Soccer)
- Abstract or context-dependent properties (mirror reflections)
- Ambiguous category membership

### 5.6 Final Semantic Matrix

**Output**: `outputs/semantic_features_model_input.json`

**Structure**:
```json
{
  "apple": ["is-a-fruit", "grows-on-trees", "is-red", "is-round", ...],
  "kettle": ["is-an-appliance", "used-for-boiling-water", "has-a-spout", ...],
  ...
}
```

**Matrix Dimensions**: 13,009 unique features × 806 words (binary matrix V_SL)

---

## 6. Model Implementation

### 6.1 Weight Matrices

#### V_LP: Phoneme-to-Lexical Mapping
- **Shape**: (480, 806) - INPUT_DIM × LEXICAL_DIM
- **Construction**: Each column contains the flattened, normalized phoneme vector for one word
- **Source**: Clear audio PC input vectors

#### V_SL: Lexical-to-Semantic Mapping
- **Shape**: (13009, 806) - SEMANTIC_DIM × LEXICAL_DIM
- **Construction**: Binary matrix where V_SL[i,j] = 1 if word j has feature i
- **Source**: Llama-generated semantic features

### 6.2 PC Dynamics Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| DT | 0.01 | Integration timestep |
| MAX_STEPS | 500 | Simulation steps per word |
| PRIME_SETTLE_STEPS | 200 | Steps for prime processing |
| PRECISION_CLEAR | 1.0 | Input precision for clear speech |
| PRECISION_NOISY | 1.0 | Input precision for noisy speech |
| PRECISION_LEXICAL | 0.1 | Lexical layer precision |
| PRECISION_SEMANTIC | 1.0 | Semantic layer precision |
| LEARNING_RATE_LEXICAL | 0.1 | Lexical state update rate |
| LEARNING_RATE_SEMANTIC | 0.1 | Semantic state update rate |
| SEMANTIC_PERSISTENCE | 0.8 | Priming strength (0-1) |
| RESET_LEXICAL_ON_TARGET | True | Reset lexical layer between prime/target |

### 6.3 N400 Computation

The N400 is operationalized as the total semantic prediction error:

```python
def get_semantic_prediction_error(self):
    if self.semantic_expectation is None:
        return self.state_semantic.abs().sum().item()
    
    mismatch = torch.abs(self.semantic_expectation - self.state_semantic)
    return mismatch.sum().item()
```

We also compute semantic similarity (cosine) as a complementary measure:
```python
def get_semantic_similarity(self):
    dot = torch.dot(self.semantic_expectation, self.state_semantic)
    norm_exp = torch.norm(self.semantic_expectation)
    norm_act = torch.norm(self.state_semantic)
    return (dot / (norm_exp * norm_act)).item()
```

---

## 7. Experimental Design

### 7.1 Trial Structure

1. **Prime Phase** (200 steps):
   - Present prime word (always clear audio)
   - Let model settle to stable state
   - Store semantic expectation

2. **Target Phase** (500 steps):
   - Partially reset state (maintain SEMANTIC_PERSISTENCE of semantic activation)
   - Present target word (clear or noisy)
   - Track N400 trace over time

3. **Metrics Collection**:
   - N400 (mean, peak, final)
   - Semantic similarity
   - Recognition accuracy (winner-take-all)

### 7.2 Pair Selection

**Same Condition**: Each available word paired with itself (repetition priming)

**Similar Condition**: Words with cosine similarity 0.2 < sim < 0.9 in semantic space

**Dissimilar Condition**: Words with cosine similarity < 0.1

### 7.3 Conditions Crossed

| Factor | Levels |
|--------|--------|
| Semantic Condition | Same, Similar, Dissimilar |
| Auditory Clarity | Clear, Noisy |

Total: 6 cells (3 × 2), with ~800 pairs per semantic condition

---

## 8. Results Summary

### 8.1 Llama Semantics Model Results

| Condition | Clear N400 | Noisy N400 | Clear Accuracy |
|-----------|------------|------------|----------------|
| Same | 0.133 | 0.184 | 90% |
| Similar | 0.327 | 0.203 | 84% |
| Dissimilar | 0.303 | 0.200 | 90% |

### 8.2 Key Findings

1. **Priming Effect (Clear Speech)**:
   - Same word: Lowest N400 (repetition benefit)
   - Different words: Higher N400 (prediction error)

2. **Noise Abolishes Priming**:
   - All conditions converge to similar N400 (~0.19-0.20)
   - Recognition accuracy drops to ~2.5%

3. **Similar > Dissimilar** (unexpected):
   - Semantically similar words show slightly higher N400 than dissimilar
   - Possible interpretation: Lexical competition from partial semantic overlap

### 8.3 Comparison: Llama vs. Word2Vec

| Model | Clear N400 Range | Noisy N400 Range | Pattern |
|-------|------------------|------------------|---------|
| Llama (sparse) | 0.13 - 0.33 | 0.18 - 0.20 | Strong priming effect |
| Word2Vec (dense) | 0.013 - 0.026 | 0.019 - 0.020 | Weaker but qualitatively similar |

Both models show the same qualitative pattern, validating the robustness of the predictive coding architecture.

---

## 9. Technical Specifications

### 9.1 Hardware

- **GPU**: NVIDIA RTX 3090 (24GB VRAM)
- **Runtime**: ~13-15 minutes per full simulation

### 9.2 Software Dependencies

```
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
gensim>=4.3.0  # Word2Vec version only
sentence-transformers>=2.2.0  # Validation only
textgrid>=1.5  # Phoneme processing
librosa>=0.10.0  # Audio loading
g2p_en>=2.1.0  # Grapheme-to-phoneme
```

### 9.3 Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install torch numpy pandas matplotlib tqdm
pip install transformers sentence-transformers
pip install gensim textgrid librosa g2p_en

# For MFA (separate conda environment)
conda create -n mfa_env -c conda-forge montreal-forced-aligner
```

---

## 10. File Manifest

### 10.1 Core Model Files

| File | Purpose |
|------|---------|
| `Predictive_Coding_Model/config.py` | Model parameters and paths |
| `Predictive_Coding_Model/pc_model.py` | PCModel class with dynamics |
| `Predictive_Coding_Model/data_loader.py` | Data loading utilities |
| `Predictive_Coding_Model/simulation.py` | Main experiment script |

### 10.2 Phoneme Processing Pipeline

| File | Purpose |
|------|---------|
| `audio_phonemes/Wav2Vec/wav2vec_processor.py` | Extract Wav2Vec features from audio |
| `audio_phonemes/Wav2Vec/g2p_transcriber.py` | Grapheme-to-phoneme conversion |
| `audio_phonemes/Wav2Vec/pc_input_generator.py` | Generate final PC input vectors |
| `MFA_input_prep.py` | Prepare files for Montreal Forced Aligner |
| `noise_quantifier.py` | Quantify clear vs. noisy vector differences |

### 10.3 Semantic Feature Generation

| File | Purpose |
|------|---------|
| `semantics_alignment/feature_generator_final.py` | Main Llama feature generation |
| `semantics_alignment/feature_selector.py` | Deduplication and selection |
| `semantics_alignment/optimize_prompt_examples.py` | Find optimal few-shot examples |
| `semantics_alignment/semantic_validator.py` | Validate against McRae norms |
| `semantics_alignment/sweep_validator.py` | Temperature sweep validation |

### 10.4 Output Files

| File | Contents |
|------|----------|
| `outputs/semantic_features_model_input.json` | Final semantic features (806 words × 20 features) |
| `outputs/mcrae_gold_standard.json` | Human norms for validation (98 words) |
| `outputs/semantic_validation_optimized_results.csv` | Per-word validation scores |
| `audio_phonemes/PC_Input_Vectors/*.npy` | Phoneme vectors for each word |
| `Predictive_Coding_Model/results/simulation_results.csv` | Simulation output |

### 10.5 Data Sources

| Resource | Description |
|----------|-------------|
| `my_800_words.csv` | Master lexicon (806 words) |
| `audio_phonemes/All_Recordings/` | Clear speech recordings |
| `audio_phonemes/Noisy_Recordings/` | Vocoded/degraded recordings |
| `audio_phonemes/MFA_Output_TextGrids/` | Forced alignment output |

---

## Acknowledgments

This project adapts the predictive coding model architecture from Eddine et al. (2024) that was additionally covered in Dr. Philip Resnik's Computational Linguistics Seminar (2025). We thank Dr. Resnik for his suggestions regarding semantic similarity measurement. Additionally, we thank Dr. Eddine for making the original model available and for guidance on implementing the core PC dynamics. The semantic feature engineering methodology was developed through extensive experimentation by the project team.

---

## References

McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. *Behavior Research Methods*, 37(4), 547-559.

Nour Eddine, S., Brothers, T., Wang, L., Spratling, M., & Kuperberg, G. R. (2024). A predictive coding model of the N400. *Cognition*, 246, 105755.

Zumchak, W., et al. (2025). *Auditory semantic priming under degraded speech conditions*. [Manuscript in preparation].

