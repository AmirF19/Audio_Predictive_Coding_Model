import numpy as np
import pandas as pd
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import sys
import torch
import glob

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\semantics_alignment")

# 1. The Human Gold Standard (Fixed Location)
GOLD_STANDARD_FILE = BASE_DIR / "outputs_test" / "mcrae_gold_standard.json"

# 2. The Folder to Validate (Change this to switch between Llama and Qwen)
# Options: "outputs_test" (Llama) or "outputs_test_qwen" (Qwen)
TARGET_SWEEP_DIR = BASE_DIR / "outputs_test" 

# Embedding Model
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

def calculate_alignment(llm_features, human_features, model):
    """Calculates Precision, Recall, and F1 using semantic embeddings."""
    if not llm_features or not human_features:
        return 0.0, 0.0, 0.0

    embeddings_llm = model.encode(llm_features, convert_to_tensor=True)
    embeddings_human = model.encode(human_features, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings_llm, embeddings_human)

    # Precision (LLM -> Human)
    max_sim_per_llm, _ = torch.max(cosine_scores, dim=1)
    precision = torch.mean(max_sim_per_llm).item()

    # Recall (Human -> LLM)
    max_sim_per_human, _ = torch.max(cosine_scores, dim=0)
    recall = torch.mean(max_sim_per_human).item()

    # F1 Score
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return precision, recall, f1

def find_model_input_file(folder_path):
    """Finds the JSON file ending in '_model_input.json' inside a folder."""
    # Matches both 'model_input.json' (Llama) and 'qwen_model_input.json' (Qwen)
    files = list(folder_path.glob("*model_input.json"))
    if files:
        return files[0]
    return None

def run_sweep_validation():
    if not GOLD_STANDARD_FILE.exists():
        print(f"FATAL ERROR: Gold standard not found at {GOLD_STANDARD_FILE}")
        sys.exit(1)

    print(f"\n--- Loading Embedding Model: {EMBEDDING_MODEL_NAME} ---")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Load Human Data Once
    with open(GOLD_STANDARD_FILE, 'r') as f:
        human_data = json.load(f)
    
    # Standardize Human Data (Hyphens)
    for word in human_data:
        human_data[word] = [f.replace('_', '-') for f in human_data[word]]

    print(f"\n--- Validating Sweep Directory: {TARGET_SWEEP_DIR.name} ---")
    
    # Find all temperature subfolders
    temp_folders = sorted(list(TARGET_SWEEP_DIR.glob("temp_*")))
    
    if not temp_folders:
        print(f"No 'temp_*' folders found in {TARGET_SWEEP_DIR}")
        sys.exit(1)

    sweep_results = []

    for temp_folder in temp_folders:
        temperature = temp_folder.name.replace("temp_", "").replace("_", ".")
        json_file = find_model_input_file(temp_folder)
        
        if not json_file:
            print(f"Skipping {temp_folder.name}: No model_input.json found.")
            continue

        print(f"Processing Temperature {temperature}...")
        
        with open(json_file, 'r') as f:
            llm_data = json.load(f)

        # Find Common Words
        llm_words = set(llm_data.keys())
        human_words = set(human_data.keys())
        common_words = sorted(list(llm_words.intersection(human_words)))

        if not common_words:
            print(f"  WARNING: No matching words found for Temp {temperature}")
            continue

        # Calculate Aggregate Scores for this Temperature
        f1_scores = []
        prec_scores = []
        rec_scores = []

        for word in common_words:
            # Standardize LLM Data (Hyphens)
            llm_feats = [f.replace('_', '-') for f in llm_data.get(word, [])]
            human_feats = human_data.get(word, [])
            
            p, r, f1 = calculate_alignment(llm_feats, human_feats, model)
            f1_scores.append(f1)
            prec_scores.append(p)
            rec_scores.append(r)

        # Average for this Temp
        avg_f1 = np.mean(f1_scores)
        avg_prec = np.mean(prec_scores)
        avg_rec = np.mean(rec_scores)

        print(f"  -> Mean F1: {avg_f1:.4f}")

        sweep_results.append({
            "Temperature": temperature,
            "Mean_F1": avg_f1,
            "Mean_Precision": avg_prec,
            "Mean_Recall": avg_rec,
            "Words_Validated": len(common_words)
        })

    # Save Summary Report
    output_csv = TARGET_SWEEP_DIR / "sweep_validation_summary.csv"
    df = pd.DataFrame(sweep_results)
    # Sort by F1 score descending to see winner at top
    df = df.sort_values(by="Mean_F1", ascending=False)
    df.to_csv(output_csv, index=False)

    print("\n==================================================")
    print("               VALIDATION COMPLETE                ")
    print("==================================================")
    print(f"Summary saved to: {output_csv}")
    print("\nTop Results:")
    print(df.head().to_string(index=False))

if __name__ == '__main__':
    run_sweep_validation()