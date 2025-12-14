# Changelog - Auditory N400 Predictive Coding Model

## Version 1.0 - GitHub Release (December 2025)

### Major Refactoring
- **Modular Architecture**: Split monolithic script into three files
  - `pc_model_gpu.py`: Core predictive coding model implementation
  - `run_simulation.py`: Main execution script for priming experiments  
  - `analysis.py`: Visualization and results export module

### Model Enhancements

#### Adaptive Momentum (v1.0)
- **Problem**: Oscillations in same-word condition + high baseline PE during ISI
- **Solution**: Context-dependent momentum for state updates
  - During input (prime/target): momentum = 0.7 (stable processing)
  - During blanks (ISI): momentum = 0.3 (fast decay to baseline)
- **Result**: Smooth N400 timecourses, biologically plausible baseline

#### Semantic Feature Normalization (v1.0)
- **Problem**: N400 peaks too high (~2000) due to larger semantic space
- **Solution**: Scale semantic PE by (reference_n_features / current_n_features)
  - Reference: 3,715 features
  - Current: 20,533 features
  - Scaling factor: 0.181
- **Result**: N400 peaks in biologically plausible range (~300-750)

#### Precision Weighting for Noisy Audio (v1.0)
- **Motivation**: Model perceptual difficulty of vocoded speech
- **Implementation**: Weight audio PE by acoustic similarity
  - Clean: w = 1.0 (baseline)
  - Noisy: w = 1.21 (21% harder, based on 0.575 cosine similarity)
- **Control**: Enabled/disabled via `SAMER_MODE` flag

### Alignment with Reference Model
- N400 metric: Lexico-semantic PE (lexical + semantic, not just lexical)
- Context handling: Natural evolution (no hard clamp to prime)
- Input scaling: Calibrated to match reference model's drive
- Post-target settling: 5 blank iterations for decay phase

### Output Improvements
- **5 Separate Figures** (publication-ready):
  1. `n400_timecourse_all.png` - All 4 conditions on one plot
  2. `n400_timecourse_same.png` - Same condition: clear vs noisy
  3. `n400_timecourse_different.png` - Different condition: clear vs noisy
  4. `n400_bars.png` - 2×2 grouped bar plot
  5. `recognition_accuracy.png` - Word recognition by clarity

### Configuration
- `SAMER_MODE`: Toggle reference alignment (default: True)
- `TARGET_INPUT_NORM`: Input scaling factor (tuned to 0.25)
- `APPLY_PRECISION_SCALING`: Enable precision weighting (conditional on SAMER_MODE)
- `APPLY_NOISE`: Inject Gaussian noise to noisy targets (conditional on SAMER_MODE)

### Documentation
- Comprehensive `README_ALIGNED.md` with:
  - Quick start guide
  - Architecture overview
  - Alignment documentation
  - Parameter descriptions
- In-code comments: Casual, thinking-through style (not overly polished)
- Function docstrings: Clear explanations of purpose and logic

### Performance
- GPU-accelerated: ~800 trials in ~30 seconds (NVIDIA RTX 3090)
- Batch processing: 256 trials in parallel
- Memory efficient: Uses PyTorch for all matrix operations

---

## Future Work
- [ ] Explore alternative momentum schedules
- [ ] Test with larger lexicons (current: 800 words)
- [ ] Add statistical significance tests to analysis output
- [ ] Compare with human N400 ERP data
- [ ] Investigate semantic feature space compression techniques


