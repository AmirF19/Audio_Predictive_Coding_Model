import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import sys
import os
import torch 

# --- CONFIGURATION ---
# BASE_DIR is the project root 
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")

# 1. LLM Input: Pointing to the NEW OPTIMIZED output from feature_selector.py
LLM_INPUT_FILE = BASE_DIR / "outputs" / "llama_temp5_optimized_features_model_input.json" 

# 2. Human Input: Gold Standard (McRae norms)
GOLD_STANDARD_FILE = BASE_DIR / "outputs" / "mcrae_gold_standard.json" 

# 3. Output: Final Results CSV
OUTPUT_RESULTS_FILE = BASE_DIR / "outputs" / "semantic_validation_optimized_results.csv"

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def calculate_alignment(llm_features, human_features, model):
    """
    Calculates the semantic alignment between two lists of features using embeddings.
    Returns: Precision (LLM->Human), Recall (Human->LLM), and F1 Score.
    """
    if not llm_features or not human_features:
        return 0.0, 0.0, 0.0, torch.tensor([])

    # 1. Encode all features into vectors
    embeddings_llm = model.encode(llm_features, convert_to_tensor=True)
    embeddings_human = model.encode(human_features, convert_to_tensor=True)

    # 2. Compute Cosine Similarity Matrix (LLM_count x Human_count)
    cosine_scores = util.cos_sim(embeddings_llm, embeddings_human)

    # 3. Precision: Max similarity for each LLM feature to ANY human feature
    # "Is the LLM outputting valid human-like concepts?"
    max_sim_per_llm_feature, _ = torch.max(cosine_scores, dim=1)
    precision_score = torch.mean(max_sim_per_llm_feature).item()

    # 4. Recall: Max similarity for each Human feature to ANY LLM feature
    # "Did the LLM miss any core concepts?"
    max_sim_per_human_feature, _ = torch.max(cosine_scores, dim=0)
    recall_score = torch.mean(max_sim_per_human_feature).item()

    # 5. F1 Score (Harmonic Mean)
    if (precision_score + recall_score) > 0:
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    else:
        f1_score = 0.0

    return precision_score, recall_score, f1_score, cosine_scores

def run_semantic_validation():
    
    if not LLM_INPUT_FILE.exists():
        print(f"FATAL ERROR: Optimized input file not found: {LLM_INPUT_FILE}")
        print("Please ensure you ran 'feature_selector.py' successfully.")
        sys.exit(1)
        
    if not GOLD_STANDARD_FILE.exists():
        print(f"FATAL ERROR: Gold standard file not found: {GOLD_STANDARD_FILE}")
        sys.exit(1)

    print(f"\n--- Loading Embedding Model: {EMBEDDING_MODEL_NAME} ---")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 1. Load Data
    print(f"Loading Optimized LLM data from: {LLM_INPUT_FILE.name}")
    with open(LLM_INPUT_FILE, 'r') as f:
        llm_data = json.load(f)
    
    print(f"Loading Human data from: {GOLD_STANDARD_FILE.name}")
    with open(GOLD_STANDARD_FILE, 'r') as f:
        human_data = json.load(f)

    # --- STANDARDIZATION FIX ---
    print("Standardizing feature separators (converting '_' to '-')...")
    for word in llm_data:
        llm_data[word] = [f.replace('_', '-') for f in llm_data[word]]
    for word in human_data:
        human_data[word] = [f.replace('_', '-') for f in human_data[word]]
    # ---------------------------

    # 2. Find Common Words for Validation
    llm_words = set(llm_data.keys())
    human_words = set(human_data.keys())
    common_words = sorted(list(llm_words.intersection(human_words)))
    
    if not common_words:
        print("ERROR: No common words found between LLM output and Gold Standard.")
        sys.exit(1)
        
    print(f"Validating {len(common_words)} common words using Semantic Embeddings.")
    
    results = []
    
    # Process words
    for i, word in enumerate(common_words):
        llm_feats = llm_data.get(word, [])
        human_feats = human_data.get(word, [])
        
        precision, recall, f1, sim_matrix = calculate_alignment(llm_feats, human_feats, model)
        
        results.append({
            'word': word,
            'alignment_f1': f1,
            'precision_llm_to_human': precision,
            'recall_human_to_llm': recall
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(common_words)}: {word.upper()} -> F1: {f1:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_RESULTS_FILE, index=False)
    
    print("\n" + "="*40)
    print(f"FINAL VALIDATION RESULTS (OPTIMIZED)")
    print("="*40)
    print(f"Mean F1 Score:    {df_results['alignment_f1'].mean():.4f}")
    print(f"Mean Precision:   {df_results['precision_llm_to_human'].mean():.4f}")
    print(f"Mean Recall:      {df_results['recall_human_to_llm'].mean():.4f}")
    print(f"Results saved to: {OUTPUT_RESULTS_FILE}")

if __name__ == '__main__':
    run_semantic_validation()