import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import sys
import os
import torch # MOVED TO TOP LEVEL

# --- CONFIGURATION FOR FULL RUN ---
# Pointing to the main project root
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")

# Input Files (Pointing to the MAIN 'outputs' folder)
LLM_INPUT_FILE = BASE_DIR / "outputs" / "semantic_features_model_input.json" 
GOLD_STANDARD_FILE = BASE_DIR / "outputs" / "mcrae_gold_standard.json" 

# Output Results
OUTPUT_RESULTS_FILE = BASE_DIR / "outputs" / "semantic_validation_embeddings_results_FULL.csv"

# Embedding Model (Small, fast, and effective for short text similarity)
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

    # 3. Calculate Precision: For each LLM feature, find the max similarity to ANY human feature
    # "How meaningful/accurate is the LLM output?"
    max_sim_per_llm_feature, _ = torch.max(cosine_scores, dim=1)
    precision_score = torch.mean(max_sim_per_llm_feature).item()

    # 4. Calculate Recall: For each Human feature, find the max similarity to ANY LLM feature
    # "Did the LLM capture the gold standard concepts?"
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
        print(f"FATAL ERROR: LLM input file not found: {LLM_INPUT_FILE}")
        sys.exit(1)
        
    if not GOLD_STANDARD_FILE.exists():
        print(f"FATAL ERROR: Gold standard file not found: {GOLD_STANDARD_FILE}")
        print("Please run 'mcrae_prep.py' to generate this.")
        sys.exit(1)

    print(f"\n--- Loading Embedding Model: {EMBEDDING_MODEL_NAME} ---")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 1. Load Data
    print(f"Loading LLM data from: {LLM_INPUT_FILE.name}")
    with open(LLM_INPUT_FILE, 'r') as f:
        llm_data = json.load(f)
    
    print(f"Loading Human data from: {GOLD_STANDARD_FILE.name}")
    with open(GOLD_STANDARD_FILE, 'r') as f:
        human_data = json.load(f)

    # --- STANDARDIZATION FIX ---
    # Normalize separators to hyphens to aid embedding tokenization
    print("Standardizing feature separators (converting '_' to '-')...")
    for word in llm_data:
        llm_data[word] = [f.replace('_', '-') for f in llm_data[word]]
    for word in human_data:
        human_data[word] = [f.replace('_', '-') for f in human_data[word]]
    # ---------------------------

    # 2. Find Common Words
    llm_words = set(llm_data.keys())
    human_words = set(human_data.keys())
    common_words = sorted(list(llm_words.intersection(human_words)))
    
    if not common_words:
        print("ERROR: No common words found between LLM output and Gold Standard.")
        print("Check capitalization or file contents.")
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
        
        # Print progress every 10 words to reduce clutter
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(common_words)}: {word.upper()} -> F1: {f1:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_RESULTS_FILE, index=False)
    
    print(f"\nMean Alignment F1 Score: {df_results['alignment_f1'].mean():.4f}")
    print(f"Mean Precision: {df_results['precision_llm_to_human'].mean():.4f}")
    print(f"Mean Recall: {df_results['recall_human_to_llm'].mean():.4f}")
    print(f"Validation results saved to {OUTPUT_RESULTS_FILE}")

if __name__ == '__main__':
    run_semantic_validation()