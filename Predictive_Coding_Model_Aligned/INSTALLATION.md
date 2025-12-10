# Installation and Setup

## Prerequisites

- Python 3.8 or higher
- NVIDIA GPU with CUDA support (recommended for speed)
- 8GB+ RAM

## Installation Steps

### 1. Clone Repository

```bash
git clone <repository_url>
cd comp_ling_project
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib tqdm
```

**Note**: Adjust PyTorch installation for your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for details.

### 4. Verify CUDA (Optional but Recommended)

```python
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA available: True`

If False, model will run on CPU (slower but functional).

---

## Quick Test Run

```bash
cd Predictive_Coding_Model_Aligned
python run_simulation.py
```

**Expected output:**
- Progress bar showing batch processing
- Results summary printed to console
- 5 PNG files saved to `results_aligned/`
- `simulation_results.csv` with trial-level data

**Typical runtime:**
- GPU (RTX 3090): ~30 seconds for 800 trials
- CPU: ~5-10 minutes for 800 trials

---

## Troubleshooting

### ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'matplotlib'`

**Solution**: Activate virtual environment and install dependencies

```bash
venv\Scripts\activate  # Windows
pip install matplotlib numpy pandas torch tqdm
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**: Reduce `BATCH_SIZE` in `run_simulation.py`

```python
BATCH_SIZE = 128  # Reduce from 256
```

### Missing Cochlear Vectors

**Problem**: `[WARN] Missing cochlear vectors for N words`

**Solution**: Ensure `audio_phonemes/Cochlear_Input_Vectors/` contains `.npy` files for your words. Model will skip missing words automatically.

### Slow Performance on CPU

**Problem**: Model running very slowly

**Solution**: 
1. Check if CUDA is available (see verification step above)
2. If no GPU available, reduce problem size:
   ```python
   # In run_simulation.py, limit number of trials
   pairs = pairs[:100]  # Add after pairs are loaded
   ```

---

## Configuration

Edit `run_simulation.py` to customize:

```python
# Line 71: Reference alignment mode
SAMER_MODE = True  # True = baseline, False = experimental features

# Line 95: Input scaling (affects N400 magnitude)
TARGET_INPUT_NORM = 0.25  # Tuned for ~400 peak N400

# Line 67: Batch size (adjust for your GPU memory)
BATCH_SIZE = 256  # Reduce if out of memory

# Line 83-87: Experimental manipulations
APPLY_NOISE = False if SAMER_MODE else True
APPLY_PRECISION_SCALING = False if SAMER_MODE else True
```

---

## File Paths

The model expects the following directory structure:

```
comp_ling_project/
├── my_800_words.csv                      # Lexicon
├── semantics_alignment/...               # Semantic features
├── audio_phonemes/
│   └── Cochlear_Input_Vectors/          # Auditory vectors (.npy files)
├── experimental_pairs/
│   └── conditions_words.csv             # Experimental design
├── samer_model/
│   └── helper_txt_files/
│       └── SUBTLEXus2007.csv           # Word frequency data
└── Predictive_Coding_Model_Aligned/
    ├── run_simulation.py
    ├── pc_model_gpu.py
    ├── analysis.py
    └── README_ALIGNED.md
```

**Note**: Paths are hard-coded in `run_simulation.py` (lines 53-58). Update if your structure differs.

---

## Minimal Dependencies (requirements.txt)

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

Save as `requirements.txt` and install with:

```bash
pip install -r requirements.txt
```

---

## Next Steps

1. ✅ Run default configuration (SAMER_MODE=True) to validate setup
2. ✅ Examine output figures in `results_aligned/`
3. ✅ Review CSV for trial-level data
4. ⚙️ Adjust configuration for your experiments
5. 📊 Analyze results with your own statistical tests

For more details, see `README_ALIGNED.md`.

