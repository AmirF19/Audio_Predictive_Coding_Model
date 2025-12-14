# Archive: Old Baseline Results

These folders contain output from when we were aligning the model with Samer's original implementation, before implementing the proper vector averaging noise method.

## Folders:

### results_aligned/
Initial alignment test with Samer's original parameters (no additional noise beyond model baseline).

### results_auto_scale_false/
Test run with AUTO_SCALE_INPUT = False to check magnitude differences.

### results_auto_scale_true/
Test run with AUTO_SCALE_INPUT = True (our standard configuration).

### resultsauto_scale_true_plus_reduced_semantic_fspace/
Early test considering semantic feature space scaling (before Samer clarified this wasn't necessary).

---

**These are kept for reference and are not part of the current model (noise implementation is improved).**

For current results, see:
- `../noise_comparison/` - Main comparison (N=2, 3, 40)
- `../condition_analysis/` - Systematic noise testing
