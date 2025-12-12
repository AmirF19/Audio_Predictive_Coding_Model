"""
Compare category-specific vs general prompt strategies per category.
Determines optimal strategy for each category based on McRae validation.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
CATEGORY_RESULTS_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "output" / "category_validation_results.csv"
GENERAL_RESULTS_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "output" / "general_validation_results.csv"
STRATEGY_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "category_strategy.json"

THRESHOLD = 0.7  # Categories below this will be tested with general prompt


def load_results(filepath):
    """Load validation results and compute per-category F1."""
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    
    # Group by category
    category_f1 = df.groupby('category')['f1'].agg(['mean', 'count', 'std']).reset_index()
    category_f1.columns = ['category', 'mean_f1', 'count', 'std_f1']
    
    return category_f1, df


def main():
    print("=" * 70)
    print("STRATEGY COMPARISON: Category-Specific vs General Prompts")
    print("=" * 70)
    
    # Load category-specific results
    if not CATEGORY_RESULTS_FILE.exists():
        print(f"\nError: Run category validation first.")
        print(f"  python category_batch_generator.py --all")
        print(f"  python validate_by_category.py")
        return
    
    cat_stats, cat_df = load_results(CATEGORY_RESULTS_FILE)
    
    print("\n" + "-" * 70)
    print("CATEGORY-SPECIFIC PROMPT RESULTS")
    print("-" * 70)
    print(f"{'Category':<25} {'Count':>6} {'Mean F1':>10} {'Status':>15}")
    print("-" * 70)
    
    underperforming = []
    for _, row in cat_stats.sort_values('mean_f1', ascending=False).iterrows():
        status = "OK" if row['mean_f1'] >= THRESHOLD else "NEEDS TEST"
        print(f"{row['category']:<25} {row['count']:>6} {row['mean_f1']:>10.4f} {status:>15}")
        if row['mean_f1'] < THRESHOLD:
            underperforming.append(row['category'])
    
    print(f"\nCategories below {THRESHOLD} threshold: {len(underperforming)}")
    if underperforming:
        print(f"  {', '.join(underperforming)}")
    
    # Check if general results exist
    if not GENERAL_RESULTS_FILE.exists():
        if underperforming:
            print(f"\n" + "=" * 70)
            print("NEXT STEP: Generate general prompt results for underperforming categories")
            print("=" * 70)
            
            # Save underperforming categories for the generator
            underperform_file = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "underperforming_categories.json"
            with open(underperform_file, 'w') as f:
                json.dump({'threshold': THRESHOLD, 'categories': underperforming}, f, indent=2)
            
            print(f"\nSaved underperforming categories to: {underperform_file}")
            print(f"\nRun:")
            print(f"  python category_batch_generator.py --test-general-for-categories")
            print(f"Then re-run this script to compare.")
        return
    
    # Load general results and compare
    gen_stats, gen_df = load_results(GENERAL_RESULTS_FILE)
    
    print("\n" + "-" * 70)
    print("COMPARISON: Category-Specific vs General (for underperforming categories)")
    print("-" * 70)
    print(f"{'Category':<20} {'Cat-Specific':>12} {'General':>12} {'Winner':>15}")
    print("-" * 70)
    
    strategy = {}
    
    for _, row in cat_stats.iterrows():
        cat = row['category']
        cat_f1 = row['mean_f1']
        
        if cat in underperforming:
            # Get general F1 for this category
            gen_row = gen_stats[gen_stats['category'] == cat]
            if len(gen_row) > 0:
                gen_f1 = gen_row['mean_f1'].values[0]
                winner = "GENERAL" if gen_f1 > cat_f1 else "CATEGORY"
                strategy[cat] = 'general' if gen_f1 > cat_f1 else 'category'
                print(f"{cat:<20} {cat_f1:>12.4f} {gen_f1:>12.4f} {winner:>15}")
            else:
                strategy[cat] = 'category'
                print(f"{cat:<20} {cat_f1:>12.4f} {'N/A':>12} {'CATEGORY':>15}")
        else:
            strategy[cat] = 'category'
    
    # Save strategy
    with open(STRATEGY_FILE, 'w') as f:
        json.dump(strategy, f, indent=2, sort_keys=True)
    
    print(f"\n" + "=" * 70)
    print("FINAL STRATEGY SAVED")
    print("=" * 70)
    print(f"Saved to: {STRATEGY_FILE}")
    
    cat_count = sum(1 for v in strategy.values() if v == 'category')
    gen_count = sum(1 for v in strategy.values() if v == 'general')
    print(f"\nCategory-specific: {cat_count} categories")
    print(f"General: {gen_count} categories")
    
    if gen_count > 0:
        print(f"\nCategories using general prompt:")
        for cat, strat in strategy.items():
            if strat == 'general':
                print(f"  - {cat}")


if __name__ == "__main__":
    main()


