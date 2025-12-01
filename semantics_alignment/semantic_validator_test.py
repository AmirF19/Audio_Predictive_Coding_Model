import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import sys
import os
import torch # MOVED TO TOP LEVEL

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\semantics_alignment")

# Files
LLM_INPUT_FILE = BASE_DIR / "outputs_test" / "semantic_features_model_input_mcrae_prompt_test.json" 
GOLD_STANDARD_FILE = BASE_DIR / "outputs_test" / "mcrae_gold_standard.json" 
OUTPUT_RESULTS_FILE = BASE_DIR / "outputs_test" / "semantic_validation_embeddings_results.csv"

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
    
    if not LLM_INPUT_FILE.exists() or not GOLD_STANDARD_FILE.exists():
        print("FATAL ERROR: Missing input files.")
        sys.exit(1)

    print(f"\n--- Loading Embedding Model: {EMBEDDING_MODEL_NAME} ---")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # 1. Load Data
    with open(LLM_INPUT_FILE, 'r') as f:
        llm_data = json.load(f)
    with open(GOLD_STANDARD_FILE, 'r') as f:
        human_data = json.load(f)

    # --- STANDARDIZATION FIX ---
    # Normalize separators: Convert all underscores to hyphens to ensure consistent tokenization
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
    
    print(f"Validating {len(common_words)} common words using Semantic Embeddings.")
    
    results = []
    
    for word in common_words:
        llm_feats = llm_data.get(word, [])
        human_feats = human_data.get(word, [])
        
        precision, recall, f1, sim_matrix = calculate_alignment(llm_feats, human_feats, model)
        
        # --- DEBUGGING OUTPUT (Show best matches for first word) ---
        print(f"\nAnalysis for: {word.upper()}")
        print(f"  Alignment Score (F1): {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")
        
        # Show the top matches to verify it's working
        if sim_matrix.numel() > 0:
            print("  Top 3 Semantic Matches:")
            # Get indices of top correlations
            # Flatten matrix, get top k, then unravel indices
            k = min(3, sim_matrix.numel())
            flat_indices = torch.topk(sim_matrix.flatten(), k=k).indices
            for idx in flat_indices:
                # Manually calculate row and col since divmod behavior on tensors can be tricky
                r = idx.item() // sim_matrix.shape[1]
                c = idx.item() % sim_matrix.shape[1]
                score = sim_matrix[r, c].item()
                print(f"    LLM: '{llm_feats[r]}' <--> Human: '{human_feats[c]}' (Sim: {score:.4f})")
        # ------------------------

        results.append({
            'word': word,
            'alignment_f1': f1,
            'precision_llm_to_human': precision,
            'recall_human_to_llm': recall
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(OUTPUT_RESULTS_FILE, index=False)
    
    print(f"\nMean Alignment F1 Score: {df_results['alignment_f1'].mean():.4f}")
    print(f"Validation results saved to {OUTPUT_RESULTS_FILE}")

if __name__ == '__main__':
    run_semantic_validation()