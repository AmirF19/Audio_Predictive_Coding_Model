import json
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

# --- CONFIGURATION ---
# Point to your MAIN project root
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")

# 1. Input: The RAW output with ~75 attributes per word
RAW_INPUT_FILE = BASE_DIR / "outputs" / "llama_temp5_semantic_features_raw_mcrae_prompt_test.json"

# 2. Output: The optimized file to use for Validation and the Model
FINAL_OPTIMIZED_JSON = BASE_DIR / "outputs" / "llama_temp5_optimized_features_model_input.json"
FINAL_OPTIMIZED_CSV = BASE_DIR / "outputs" / "llama_temp5_optimized_features.csv"

# Parameters
TARGET_FEATURE_COUNT = 20  # Select top 20 distinct features
REDUNDANCY_THRESHOLD = 0.75 # Skip if > 75% similar to an already selected feature
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def select_best_features(features_pool, model):
    """
    Selects the top N distinct features from a large pool using greedy deduplication.
    """
    # 1. Basic string cleanup and deduplication (preserving order)
    clean_pool = []
    seen_strings = set()
    for f in features_pool:
        # Ensure hyphenated format
        clean_f = f.replace('_', '-')
        if clean_f not in seen_strings:
            seen_strings.add(clean_f)
            clean_pool.append(clean_f)
            
    if not clean_pool:
        return []

    # 2. Encode the pool
    pool_embeddings = model.encode(clean_pool, convert_to_tensor=True)
    
    selected_features = []
    selected_embeddings = []
    
    # 3. Greedy Selection Loop
    for i, candidate_feat in enumerate(clean_pool):
        if len(selected_features) >= TARGET_FEATURE_COUNT:
            break
            
        candidate_emb = pool_embeddings[i]
        
        # Always accept the first feature (highest LLM probability)
        if not selected_features:
            selected_features.append(candidate_feat)
            selected_embeddings.append(candidate_emb)
            continue
            
        # Check similarity against ALL currently selected features
        # Stack embeddings to compare candidate against the whole selected group at once
        stack = torch.stack(selected_embeddings)
        
        # Calculate similarity: (1, Vector_Dim) x (Selected_Count, Vector_Dim).T
        sims_to_selected = util.cos_sim(candidate_emb.unsqueeze(0), stack)
        
        # Find the maximum similarity this candidate has to ANY existing feature
        max_redundancy = torch.max(sims_to_selected).item()
        
        # If it's unique enough, keep it
        if max_redundancy < REDUNDANCY_THRESHOLD:
            selected_features.append(candidate_feat)
            selected_embeddings.append(candidate_emb)
            
    return selected_features

def main():
    print(f"--- Starting Smart Feature Selection (Target: {TARGET_FEATURE_COUNT}) ---")
    print(f"Loading Raw Data: {RAW_INPUT_FILE}")
    
    if not RAW_INPUT_FILE.exists():
        print(f"FATAL ERROR: Raw input file not found.")
        return

    with open(RAW_INPUT_FILE, 'r') as f:
        raw_data = json.load(f)
        
    print(f"Loading Embedding Model: {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    optimized_data = {}
    csv_rows = []
    
    print(f"Processing {len(raw_data)} words...")
    
    for word, runs in tqdm(raw_data.items()):
        # Flatten runs into one single list (preserving order: Run 1 -> Run 2...)
        flat_pool = [feat for run in runs for feat in run]
        
        # Perform Smart Selection
        selected = select_best_features(flat_pool, model)
        optimized_data[word] = selected
        
        # Prepare CSV row
        row = {'word': word}
        for i, feat in enumerate(selected):
            row[f'feature_{i+1}'] = feat
        csv_rows.append(row)

    # Save Outputs
    print(f"\nSaving optimized JSON to: {FINAL_OPTIMIZED_JSON}")
    with open(FINAL_OPTIMIZED_JSON, 'w') as f:
        json.dump(optimized_data, f, indent=2)
        
    print(f"Saving CSV to: {FINAL_OPTIMIZED_CSV}")
    pd.DataFrame(csv_rows).to_csv(FINAL_OPTIMIZED_CSV, index=False)
    
    print("\n--- Selection Complete ---")
    print("Next Step: Update 'semantic_validator.py' to point to this new optimized JSON file and run it.")

if __name__ == '__main__':
    main()