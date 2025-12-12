"""
Compare GPT 5.1 semantic features against original Llama model.
Shows per-word F1 scores and overall improvement.
"""

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")

# Load both validation results
LLAMA_FILE = PROJECT_ROOT / "outputs" / "semantic_validation_embeddings_results_FULL.csv"
GPT51_FILE = PROJECT_ROOT / "semantics_alignment" / "gpt51_features" / "validation_results.csv"

# Example words to exclude from comparison (used in GPT prompt)
EXAMPLE_WORDS = {'tiger', 'hammer', 'lion', 'piano', 'sofa', 'shovel'}

def main():
    # Load Llama results
    llama_df = pd.read_csv(LLAMA_FILE)
    llama_df['word'] = llama_df['word'].str.lower()
    llama_scores = dict(zip(llama_df['word'], llama_df['alignment_f1']))
    
    # Load GPT 5.1 results
    if not GPT51_FILE.exists():
        print(f"GPT 5.1 results not found: {GPT51_FILE}")
        print("Run gpt51_batch_generator.py and validate_gpt51_batch.py first")
        return
    
    gpt_df = pd.read_csv(GPT51_FILE)
    gpt_df['word'] = gpt_df['word'].str.lower()
    
    # Compare
    print("=" * 80)
    print("COMPARISON: GPT 5.1 vs Llama Semantic Features")
    print("=" * 80)
    print(f"{'Word':<15} {'Llama F1':>10} {'GPT5.1 F1':>10} {'Delta':>10} {'Winner':>10}")
    print("-" * 80)
    
    improvements = []
    gpt_wins = 0
    llama_wins = 0
    ties = 0
    
    for _, row in gpt_df.iterrows():
        word = row['word']
        if word in EXAMPLE_WORDS:
            continue
            
        gpt_f1 = row['f1']
        llama_f1 = llama_scores.get(word, None)
        
        if llama_f1 is None:
            continue
        
        delta = gpt_f1 - llama_f1
        improvements.append(delta)
        
        if delta > 0.01:
            winner = "GPT5.1"
            gpt_wins += 1
        elif delta < -0.01:
            winner = "Llama"
            llama_wins += 1
        else:
            winner = "Tie"
            ties += 1
        
        print(f"{word:<15} {llama_f1:>10.4f} {gpt_f1:>10.4f} {delta:>+10.4f} {winner:>10}")
    
    print("-" * 80)
    
    if improvements:
        avg_improvement = sum(improvements) / len(improvements)
        llama_mean = sum(llama_scores.get(row['word'].lower(), 0) 
                        for _, row in gpt_df.iterrows() 
                        if row['word'].lower() not in EXAMPLE_WORDS and row['word'].lower() in llama_scores) / len(improvements)
        gpt_mean = gpt_df[~gpt_df['word'].str.lower().isin(EXAMPLE_WORDS)]['f1'].mean()
        
        print(f"\nSUMMARY ({len(improvements)} words compared)")
        print(f"  Llama Mean F1:     {llama_mean:.4f}")
        print(f"  GPT 5.1 Mean F1:   {gpt_mean:.4f}")
        print(f"  Average Delta:     {avg_improvement:+.4f}")
        print(f"\n  GPT 5.1 Wins: {gpt_wins}")
        print(f"  Llama Wins:   {llama_wins}")
        print(f"  Ties:         {ties}")
        
        if avg_improvement > 0:
            print(f"\n  GPT 5.1 shows {avg_improvement*100:.2f}% improvement over Llama")
        else:
            print(f"\n  Llama shows {-avg_improvement*100:.2f}% better performance than GPT 5.1")

if __name__ == "__main__":
    main()
