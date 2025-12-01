import torch
import json
import re
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import time
import sys
import gc

# --- 0. Setup and Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Base output directory
BASE_OUTPUT_DIR = Path("outputs_test")
BASE_OUTPUT_DIR.mkdir(exist_ok=True)

# Input: Your 50-word test list
USER_WORDS_FILE = "50_words_test.csv"

# --- EXPERIMENT PARAMETERS ---
# The specific temperatures you requested
TEMPERATURE_LIST = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

RUNS_PER_WORD = 4      # How many times to ask the model per word per temp
TARGET_FEATURE_COUNT = 25 
BATCH_SAVE_SIZE = 10   # Save more frequently for the test

# --- 1. Model Loading ---
# We load the model ONCE globally to avoid reloading 7 times
if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA AVAILABLE. Runtime Version: {torch.version.cuda}")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("WARNING: CUDA is not detected by PyTorch.")
    MAX_CPU_WORDS = 5

try:
    print("\nLoading model into memory (this happens once)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    if device == "cuda":
        print("FORCING model parameters to CUDA device...")
        model.to(device)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

except Exception as e:
    print(f"\nFATAL ERROR during model loading: {e}")
    sys.exit(1)


# --- Llama 3 System Prompt (McRae Replication) ---
SYSTEM_PROMPT = f"""
You are a participant in a psycholinguistic study focused on semantic associations in a natural environment. Your task is to generate a list of distinct semantic properties and facts for a given concept, as if you were a human participant providing data for psycholinguistic norms.

To help us conduct this work, we need information on what people know about different things in the world. Your instructions are based on the McRae et al. (2005) feature production task:
1. For a given concept, list the specific properties of the concept to which the word refers
2. Examples of different types of properties would be: physical properties, such as internal and external parts, and how it looks, sounds, smells, feels, or tastes; functional properties, such as what it is used for; where, when and by whom it is used; things that the concept is related to, such as the category that it belongs in; and other facts, such as how it behaves, or where it comes from.
3. All words are meant to be considered as **nouns only**.
4. Generate exactly {TARGET_FEATURE_COUNT} distinct properties.
5. **Respond ONLY with a valid JSON object in the format: {{"features": ["property1", "property2", ...]}}**

Examples of what constitutes a property:
duck: is a bird, is an animal, waddles, flies, migrates, lays eggs, quacks, swims, has wings, has a beak, has webbed feet, has feathers, lives in ponds, lives in water, hunted by people, is edible
cucumber: is a vegetable, has green skin, has a white inside, has seeds inside, is cylindrical, is long, grows in gardens, grows on vines, is edible, is crunchy, used for making pickles, eaten in salads
stove: is an appliance, produces heat, has elements, has an oven, made of metal, is hot, is electrical, runs on wood, runs on gas, found in kitchens, used for baking, used for cooking food"""

# --- 2. Helper Functions ---

def load_user_words(filepath):
    """Robustly loads the word list."""
    try:
        # Try reading with standard CSV first
        df = pd.read_csv(filepath, sep=None, engine='python')
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        word_col = None
        if 'word' in df.columns:
            word_col = 'word'
        elif len(df.columns) > 0:
            # Fallback to first column
            word_col = df.columns[0]
            
        if not word_col:
            raise ValueError("Could not identify word column")
            
        words = df[word_col].astype(str).str.strip().str.lower().tolist()
        return sorted(list(set([w for w in words if w])))
    except Exception as e:
        print(f"Error reading word list {filepath}: {e}")
        sys.exit(1)

def generate_features_for_word(word, pipe, num_runs):
    """Generates features for a single word using the provided pipeline."""
    all_features_for_word = []
    
    for i in range(num_runs):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"The concept is: {word}"}
        ]

        try:
            outputs = pipe(messages)
            response = outputs[0]['generated_text'].strip()
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            
            if json_match:
                data = json.loads(json_match.group(0))
                features = data.get("features", []) 
                
                if isinstance(features, list):
                    # Clean and normalize
                    cleaned = [re.sub(r'\s+', '-', f.lower().strip()) for f in features if f.strip()]
                    all_features_for_word.append(cleaned)
                else:
                    all_features_for_word.append([])
            else:
                all_features_for_word.append([])

        except Exception:
            all_features_for_word.append([])
            
    return all_features_for_word

def run_experiment_for_temperature(temp, words):
    """Runs the full generation loop for ONE temperature setting."""
    
    # 1. Setup Output Directory for this specific temperature
    temp_str = str(temp).replace('.', '_')
    current_output_dir = BASE_OUTPUT_DIR / f"temp_{temp_str}"
    current_output_dir.mkdir(exist_ok=True)
    
    print(f"\n>>> STARTING EXPERIMENT: Temperature {temp}")
    print(f"    Output Folder: {current_output_dir}")

    # Define specific output files for this run
    raw_json_path = current_output_dir / "features_raw.json"
    final_json_path = current_output_dir / "model_input.json"
    final_csv_path = current_output_dir / "features_final.csv"

    # 2. Initialize Pipeline (New pipeline per temp to apply settings)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=1024,      
        do_sample=True,
        temperature=temp, # <--- APPLIED HERE
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id
    )

    # 3. Load Existing Progress (Resume Capability)
    if raw_json_path.exists():
        with open(raw_json_path, 'r') as f:
            db = json.load(f)
    else:
        db = {}

    words_to_do = [w for w in words if w not in db or len(db[w]) < RUNS_PER_WORD]
    
    if not words_to_do:
        print("    All words already completed for this temperature.")
    else:
        # 4. Generation Loop
        session_count = 0
        for word in tqdm(words_to_do, desc=f"Gen Temp {temp}"):
            existing = db.get(word, [])
            needed = RUNS_PER_WORD - len(existing)
            
            if needed > 0:
                new_feats = generate_features_for_word(word, pipe, needed)
                db[word] = existing + new_feats
                session_count += 1
            
            if session_count > 0 and session_count % BATCH_SAVE_SIZE == 0:
                with open(raw_json_path, 'w') as f:
                    json.dump(db, f, indent=2)

        # Final Raw Save
        with open(raw_json_path, 'w') as f:
            json.dump(db, f, indent=2)

    # 5. Post-Processing (Flatten & Deduplicate)
    final_model_data = {}
    csv_rows = []
    
    for word, runs in db.items():
        # Flatten and deduplicate while preserving order
        combined = []
        seen = set()
        for run in runs:
            for feat in run:
                if feat not in seen:
                    seen.add(feat)
                    combined.append(feat)
        
        # Cut to target count
        final_list = combined[:TARGET_FEATURE_COUNT]
        final_model_data[word] = final_list
        
        # CSV format
        row = {'word': word}
        for i, f in enumerate(final_list):
            row[f'feature_{i+1}'] = f
        csv_rows.append(row)

    # Save Clean Outputs
    with open(final_json_path, 'w') as f:
        json.dump(final_model_data, f, indent=2)
        
    pd.DataFrame(csv_rows).to_csv(final_csv_path, index=False)
    print(f"    Finished Temperature {temp}. Data saved.")
    
    # Cleanup to free memory
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# --- 3. Main Execution ---
def main():
    print("\n==================================================")
    print("      STARTING SEMANTIC PARAMETER SWEEP           ")
    print("==================================================")
    
    words = load_user_words(USER_WORDS_FILE)
    print(f"Loaded {len(words)} words from {USER_WORDS_FILE}")
    print(f"Testing Temperatures: {TEMPERATURE_LIST}")
    
    for temp in TEMPERATURE_LIST:
        run_experiment_for_temperature(temp, words)
        # Small pause to let GPU cool
        time.sleep(1)

    print("\n==================================================")
    print("               SWEEP COMPLETE                     ")
    print("==================================================")
    print(f"All outputs located in: {BASE_OUTPUT_DIR}")

if __name__ == '__main__':
    main()