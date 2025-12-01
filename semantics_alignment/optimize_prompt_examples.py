import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch

# --- CONFIGURATION ---
BASE_DIR = Path("outputs_test")

# 1. Input: The RAW output containing multiple runs
# We use the output from your best performing temperature (e.g., 0.5)
RAW_INPUT_FILE = BASE_DIR / "temp_0_5" / "features_raw.json" 

# 2. Input: The Gold Standard
GOLD_STANDARD_FILE = BASE_DIR / "mcrae_gold_standard.json"

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- OPTIMIZATION PARAMETERS ---
# 1. Deduplication: Any feature with similarity > 0.75 to an already selected feature is skipped
REDUNDANCY_THRESHOLD = 0.75 

# 2. Individual Quality: Minimum score for a SINGLE feature to be considered
# LOWERED to 0.60 to ensure we can find enough features to fill the list of 15
MIN_INDIVIDUAL_QUALITY = 0.60

# 3. Group Quality: The AVERAGE score of the word's features must meet this
# LOWERED slightly to 0.80 to accommodate the larger list size
MIN_AVG_SCORE = 0.80

# 4. Count Limit: We want exactly this many attributes
TARGET_EXAMPLE_COUNT = 15

def optimize_word_features(word, raw_runs, human_features, model):
    """
    Pools all raw features, scores them against human features, 
    and selects the Top 15 distinct semantic matches.
    """
    # 1. Flatten all runs into one large pool of unique strings
    pool = list(set([feat for run in raw_runs for feat in run]))
    
    if not pool or not human_features:
        return [], 0.0

    # 2. Encode
    pool_embeddings = model.encode(pool, convert_to_tensor=True)
    human_embeddings = model.encode(human_features, convert_to_tensor=True)

    # 3. Calculate Similarity to Human Norms (Quality Score)
    cos_scores_human = util.cos_sim(pool_embeddings, human_embeddings)
    
    # Score each pool item by its max similarity to ANY human feature
    quality_scores, _ = torch.max(cos_scores_human, dim=1)
    
    # 4. Greedy Selection with Deduplication
    sorted_indices = torch.argsort(quality_scores, descending=True)
    
    selected_features = []
    selected_embeddings = []
    final_scores = []
    
    for idx in sorted_indices:
        # Stop if we hit the target count
        if len(selected_features) >= TARGET_EXAMPLE_COUNT:
            break
            
        candidate_feat = pool[idx]
        candidate_emb = pool_embeddings[idx]
        candidate_score = quality_scores[idx].item()
        
        # --- FILTER: Individual Quality Control ---
        if candidate_score < MIN_INDIVIDUAL_QUALITY:
            continue
        
        # If this is the first item, accept it
        if not selected_features:
            selected_features.append(candidate_feat)
            selected_embeddings.append(candidate_emb)
            final_scores.append(candidate_score)
            continue
            
        # Check similarity against ALREADY SELECTED features (Deduplication)
        stack = torch.stack(selected_embeddings)
        sims_to_selected = util.cos_sim(candidate_emb.unsqueeze(0), stack)
        max_redundancy = torch.max(sims_to_selected).item()
        
        if max_redundancy < REDUNDANCY_THRESHOLD:
            selected_features.append(candidate_feat)
            selected_embeddings.append(candidate_emb)
            final_scores.append(candidate_score)
    
    avg_score = np.mean(final_scores) if final_scores else 0.0
    
    return selected_features, avg_score

def main():
    print(f"--- Starting Prompt Optimization ---")
    print(f"Criteria: Avg Score >= {MIN_AVG_SCORE}, Target Count = {TARGET_EXAMPLE_COUNT}")
    
    if not RAW_INPUT_FILE.exists():
        print(f"Error: File not found at {RAW_INPUT_FILE}")
        return

    with open(RAW_INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
        
    if not GOLD_STANDARD_FILE.exists():
        print(f"Error: Gold standard not found at {GOLD_STANDARD_FILE}")
        return

    with open(GOLD_STANDARD_FILE, 'r') as f:
        human_data = json.load(f)
        # Normalize human data
        for w in human_data:
            human_data[w] = [f.replace('_', '-') for f in human_data[w]]

    print("Loading Embedding Model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Find intersection
    common_words = sorted(list(set(raw_data.keys()).intersection(set(human_data.keys()))))
    print(f"Optimizing features for {len(common_words)} common words...")

    scores = []

    for word in common_words:
        best_feats, score = optimize_word_features(word, raw_data[word], human_data[word], model)
        
        # --- FILTER: Require Minimum Average Score ---
        if score >= MIN_AVG_SCORE:
            scores.append({
                "word": word,
                "score": score,
                "count": len(best_feats),
                "optimized_features": best_feats
            })

    # --- SORTING LOGIC ---
    # 1. Sort by Count (We want words that successfully found 15 features)
    # 2. Sort by Score (Among those, which have the highest average)
    sorted_scores = sorted(scores, key=lambda x: (x['count'], x['score']), reverse=True)
    
    if not sorted_scores:
        print("\nWARNING: No words met the strict criteria.")
        print("Try lowering MIN_AVG_SCORE or MIN_INDIVIDUAL_QUALITY.")
        return

    print("\n" + "="*60)
    # UPDATED: Select Top 4 instead of 3
    print(f"TOP 4 WORDS (Prioritizing Count={TARGET_EXAMPLE_COUNT}, then Highest Avg Score)")
    print("="*60)
    
    top_4 = sorted_scores[:4]
    
    formatted_prompt_text = ""
    
    for item in top_4:
        word = item['word']
        feats = item['optimized_features']
        
        print(f"\nWORD: {word.upper()} (Avg Score: {item['score']:.4f} | Count: {item['count']})")
        print(f"Selected Features ({len(feats)}): {feats}")
        
        # Format for the python prompt string
        feat_string = ", ".join(feats)
        formatted_prompt_text += f'{word}: {feat_string}\n'

    print("\n" + "="*60)
    print("COPY THIS INTO YOUR PYTHON SCRIPT (SYSTEM_PROMPT):")
    print("="*60)
    print(formatted_prompt_text)
    print("="*60)

if __name__ == '__main__':
    main()