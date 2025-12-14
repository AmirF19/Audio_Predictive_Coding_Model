# Noise Comparison Results

Three noise configurations tested.

## Overview

Noise mixing: averaging target phoneme vector with N random word vectors.

---

## N=2

**Target contribution:** 33.3% (1 target + 2 random words)
**Cosine similarity:** 0.604 (vocoded speech similarity: 0.575)

**Results:**
- Clear accuracy: 98.6%
- Noisy accuracy: 37.5%
- Clear N400 MEAN (different): 372.1
- Noisy N400 MEAN (different): 968.7
- Priming effect ratio: 1.00
- N400 timecourse: spike then settle

---

## N=3

**Target contribution:** 25% (1 target + 3 random words)
**Cosine similarity:** 0.537 (vocoded speech similarity: 0.575)

**Results:**
- Clear accuracy: 98.6%
- Noisy accuracy: 31.9%
- Clear N400 MEAN (different): 372.1
- Noisy N400 MEAN (different): 1062.4
- Priming effect ratio: 0.76
- N400 timecourse: spike then settle

---

## N=40

**Target contribution:** 2.4% (1 target + 40 random words)
**Cosine similarity:** 0.106

**Results:**
- Clear accuracy: 98.6%
- Noisy accuracy: 33.6%
- Noisy accuracy (different-word): 0.0%
- Clear N400 MEAN (different): 372.1
- Noisy N400 MEAN (different): 1381.9
- Priming effect ratio: 0.11
- N400 timecourse: plateau (does not settle)

First N to achieve 0% different-word accuracy (Samer's goal).

---

## Files in Each Folder

- `simulation_results.csv` - Trial-by-trial data with N400 timecourses
- `n400_timecourse_all.png` - All conditions
- `n400_timecourse_same.png` - Same-word priming
- `n400_timecourse_different.png` - Different-word priming
- `n400_bars.png` - Mean N400 by condition
- `recognition_accuracy.png` - Accuracy breakdown

---

## Related Files

- `../cosine_similarity_analysis.csv` - Calibration data (N=1 to N=5)
- `../condition_analysis/` - Systematic testing (N=3 to N=100)
- `calculate_cosine_similarity.py` - Calibration script
