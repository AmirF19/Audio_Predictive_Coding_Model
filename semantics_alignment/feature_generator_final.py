import torch
import json
import re
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import time
import sys

# --- 0. Setup and Configuration ---
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Output Directory
OUTPUT_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Input: The FULL 800-word list
USER_WORDS_FILE = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\my_800_words.csv")

# Output Files (Specific filenames requested)
RAW_JSON_OUTPUT = OUTPUT_DIR / "llama_temp5_semantic_features_raw_mcrae_prompt_test.json"
FINAL_CSV_OUTPUT = OUTPUT_DIR / "llama_temp5_semantic_features_final_mcrae_prompt_test.csv"
FINAL_MODEL_INPUT_JSON = OUTPUT_DIR / "llama_temp5_semantic_features_model_input_mcrae_prompt_test.json"

# --- OPTIMIZED PARAMETERS ---
RUNS_PER_WORD = 3 
TARGET_FEATURE_COUNT = 25 
BATCH_SAVE_SIZE = 50 
TEMPERATURE = 0.5 # Optimized based on sweep results

# --- 1. Model Loading ---

if torch.cuda.is_available():
    device = "cuda"
    print(f"CUDA AVAILABLE. Runtime Version: {torch.version.cuda}")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    device = "cpu"
    print("WARNING: CUDA is not detected by PyTorch, using CPU (this will be extremely slow!)")
    MAX_CPU_WORDS = 10
    
try:
    # Load model with bfloat16 (Non-quantized as requested)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # --- CRITICAL FIX: EXPLICITLY MOVE MODEL TO GPU ---
    if device == "cuda":
        print("FORCING model parameters to CUDA device...")
        model.to(device)
        
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

except Exception as e:
    print(f"\nFATAL ERROR during model loading. Ensure you are authenticated and have sufficient memory.")
    print(f"Error: {e}")
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
stove: is an appliance, produces heat, has elements, has an oven, made of metal, is hot, is electrical, runs on wood, runs on gas, found in kitchens, used for baking, used for cooking food
"""

# --- Pipeline Setup ---
print(f"Setting up text-generation pipeline with Temperature {TEMPERATURE}...")
# --- CRITICAL FIX: Pass the device ID to the pipeline ---
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1, # Use device index 0 for CUDA, -1 for CPU
    # LOWERED TO 512: Safe for 25 features, reduces max VRAM spike
    max_new_tokens=512,      
    do_sample=True,
    temperature=TEMPERATURE, # Optimized Temperature
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id
)
print(f"Pipeline device set to: {'CUDA:0' if device == 'cuda' else 'CPU'}")
print("Pipeline ready.")

# --- 2. Function Definitions ---

def load_user_words(filepath):
    """Loads the list of words from the user's CSV."""
    
    # --- START ROBUST CSV READING ---
    df = None
    
    try:
        df = pd.read_csv(filepath, sep='\t')
    except:
        pass 

    if df is None or len(df.columns) < 2: 
        try:
            df = pd.read_csv(filepath, sep=',')
        except:
             print(f"FATAL ERROR: Could not read CSV: {filepath}")
             sys.exit(1)

    df.columns = df.columns.str.strip().str.lower()
    
    word_col_name = None
    if 'word' in df.columns:
        word_col_name = 'word'
    elif 'word,column_2' in df.columns and len(df.columns) == 1:
        try:
             df = pd.read_csv(filepath, sep=',')
             df.columns = df.columns.str.strip().str.lower()
             if 'word' in df.columns:
                 word_col_name = 'word'
        except Exception as e:
            pass 

    if word_col_name is None:
        # Fallback: Assume first column is the word column if named ambiguously
        word_col_name = df.columns[0]
        print(f"Warning: 'word' column not found. Using first column: '{word_col_name}'")
    
    # --- END ROBUST CSV READING ---
    
    words = df[word_col_name].astype(str).str.strip().str.lower().tolist()
    words = sorted(list(set([w for w in words if w])))

    print(f"Successfully loaded {len(words)} unique words from {filepath}.")
    
    global device
    if device == "cpu" and 'MAX_CPU_WORDS' in globals() and len(words) > MAX_CPU_WORDS:
        print(f"WARNING: Limiting words to {MAX_CPU_WORDS} for CPU demo.")
        words = words[:MAX_CPU_WORDS]
    
    return words


def generate_and_parse_features(word, num_runs):
    """
    Calls the local LLM 'num_runs' independent times, extracting and validating JSON.
    """
    all_features_for_word = []
    
    for i in range(num_runs):
        user_prompt = f"The concept is: {word}"
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Generate the response
            outputs = text_generator(messages)
            assistant_response = outputs[0]['generated_text'].strip()

            # Robust JSON extraction
            json_match = re.search(r'\{.*\}', assistant_response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Looks for 'features' key 
                features = data.get("features", []) 
                
                if isinstance(features, list):
                    # Basic cleaning: remove capitalization, replace spaces
                    cleaned_features = [re.sub(r'\s+', '-', f.lower().strip()) for f in features if f.strip()]
                    all_features_for_word.append(cleaned_features)
                    # Real-time update
                    # print(f"\n  > RUN {i+1} for '{word}': Generated {len(cleaned_features)} features.")
                else:
                    all_features_for_word.append([])
                    # print(f"\n  > RUN {i+1} for '{word}': FAILED (Valid JSON, but 'features' key missing or not a list)")
            else:
                all_features_for_word.append([])
                # print(f"\n  > RUN {i+1} for '{word}': FAILED (No JSON found in response)")

        except Exception as e:
            # print(f"\n  > RUN {i+1} for '{word}': FAILED with Python error: {e}")
            all_features_for_word.append([])
            
        time.sleep(0.5) # Small delay to stabilize GPU memory usage

    return all_features_for_word


def save_final_csv(raw_data, output_path):
    """
    Converts the raw JSON data (word -> [run1_features], [run2_features]...) 
    into a flat CSV format where each word has its features listed horizontally.
    """
    print(f"\n--- Saving Final CSV Output ---")
    
    # Identify all unique features generated across all words/runs
    all_features = set()
    for runs in raw_data.values():
        for run in runs:
            all_features.update(run)

    # Create a dense feature matrix for the CSV
    data = []
    all_unique_features_list = sorted(list(all_features))
    
    for word, runs in raw_data.items():
        # Combine all unique features from all runs (flattening list of lists)
        combined_features = []
        seen = set()
        # Preserve order by iterating
        for run in runs:
            for f in run:
                if f not in seen:
                    seen.add(f)
                    combined_features.append(f)
                    
        # Store features as columns for the CSV output
        row = {'word': word}
        
        # Max out to TARGET_FEATURE_COUNT
        final_list = combined_features[:TARGET_FEATURE_COUNT] 
        
        # Add the features list to the CSV data
        for i, feat in enumerate(final_list):
            row[f'feature_{i+1}'] = feat
            
        data.append(row)

    df_output = pd.DataFrame(data)
    df_output.to_csv(output_path, index=False)
    
    print(f"Final feature list saved to: {output_path}")
    print(f"Total unique features generated for the entire corpus: {len(all_unique_features_list)}")


# --- 3. Main Execution ---
def main():
    print("\n==================================================")
    print("      STARTING FINAL SEMANTIC GENERATION          ")
    print(f"      Model: Llama 3.1 | Temp: {TEMPERATURE}")
    print("==================================================")
    
    # Load words 
    WORDS_TO_PROCESS = load_user_words(USER_WORDS_FILE)
    
    # Handle resume-friendly loading and saving
    if RAW_JSON_OUTPUT.exists():
        print(f"Loading existing raw data from {RAW_JSON_OUTPUT}")
        with open(RAW_JSON_OUTPUT, 'r') as f:
            generated_features_db = json.load(f)
    else:
        print(f"Creating new feature database at {RAW_JSON_OUTPUT}")
        generated_features_db = {}

    words_to_do = [
        w for w in WORDS_TO_PROCESS 
        if w not in generated_features_db or 
        len(generated_features_db.get(w, [])) < RUNS_PER_WORD
    ]

    if not words_to_do:
        print("All words are already processed. Skipping generation.")
    else:
        print(f"Processing {len(words_to_do)} new/incomplete words...")
        
        # Batch counters for periodic saving
        words_processed_in_session = 0
        
        for word in tqdm(words_to_do, desc="Generating Features"):
            # Determine how many runs are still needed for this word
            existing_runs = generated_features_db.get(word, [])
            runs_needed = RUNS_PER_WORD - len(existing_runs)
            
            if runs_needed > 0:
                new_features = generate_and_parse_features(word, num_runs=runs_needed)
                generated_features_db[word] = existing_runs + new_features
                words_processed_in_session += 1
            
            # Save progress every BATCH_SAVE_SIZE words
            if words_processed_in_session > 0 and words_processed_in_session % BATCH_SAVE_SIZE == 0:
                print(f"\n[AUTO-SAVE] Saving progress after {words_processed_in_session} words...")
                with open(RAW_JSON_OUTPUT, 'w') as f:
                    json.dump(generated_features_db, f, indent=2)
                print("[AUTO-SAVE] Save complete.")

        # Final save for any remaining words
        with open(RAW_JSON_OUTPUT, 'w') as f:
            json.dump(generated_features_db, f, indent=2)
        
    # 4. Final Output Conversion
    save_final_csv(generated_features_db, FINAL_CSV_OUTPUT)
    
    # 5. Outputting Final JSON 
    final_model_data = {}
    for word, runs in generated_features_db.items():
        # Combine and deduplicate
        combined_features = []
        seen = set()
        for run in runs:
            for f in run:
                if f not in seen:
                    seen.add(f)
                    combined_features.append(f)
        # Trim/pad this combined list to the required TARGET_FEATURE_COUNT 
        final_model_data[word] = combined_features[:TARGET_FEATURE_COUNT]
        
    with open(FINAL_MODEL_INPUT_JSON, 'w') as f:
        json.dump(final_model_data, f, indent=2)
        
    print(f"\nDONE! Final model input saved to: {FINAL_MODEL_INPUT_JSON}")

if __name__ == '__main__':
    main()