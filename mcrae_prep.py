import pandas as pd
import json
import re
from pathlib import Path
import sys

# --- CONFIGURATION (NOW RELATIVE TO PROJECT ROOT) ---
# BASE_DIR is now the main C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\
BASE_DIR = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project") 
# Input: The raw tab-delimited file (Moved to Project Root)
RAW_MCRAE_FILE = BASE_DIR / "CONCS_FEATS_concstats_brm.txt"
# Input: Your original list of words (Remains in Project Root)
USER_WORDS_FILE = BASE_DIR / "my_800_words.csv" 
# Output: The JSON file required by semantic_validator.py (Placed in the 'outputs' folder next to the LLM JSON)
OUTPUT_JSON_FILE = BASE_DIR / "outputs" / "mcrae_gold_standard.json"
OUTPUT_JSON_FILE.parent.mkdir(exist_ok=True) # Ensure outputs folder exists

# --- Main Processing Logic ---

def load_user_words(filepath):
    """Loads the list of words from the user's CSV for filtering."""
    try:
        # Use robust reading logic (tab or comma)
        df = pd.read_csv(filepath, sep='\t')
        if len(df.columns) < 2 or 'word' not in df.columns.str.lower().str.strip().tolist():
            df = pd.read_csv(filepath, sep=',')
            
        word_col_name = 'word' if 'word' in df.columns.str.lower().str.strip().tolist() else df.columns[0]
        
        words = df[word_col_name].astype(str).str.strip().str.lower().tolist()
        return set([w for w in words if w])
    except Exception as e:
        print(f"FATAL ERROR loading user words: {e}")
        return set()

def process_mcrae_norms():
    if not RAW_MCRAE_FILE.exists():
        print(f"FATAL ERROR: Raw McRae file not found. Ensure '{RAW_MCRAE_FILE.name}' is in the project root folder.")
        sys.exit(1)
        
    user_words = load_user_words(USER_WORDS_FILE)
    if not user_words:
        print("FATAL ERROR: User word list could not be loaded for filtering.")
        sys.exit(1)

    print(f"\n--- Starting McRae Norms Processing ---")
    print(f"Filtering features for {len(user_words)} concepts...")

    # 1. Load the raw tab-delimited file
    # We explicitly define the separator and low_memory=False for large files
    df = pd.read_csv(RAW_MCRAE_FILE, sep='\t', low_memory=False)

    # 2. Clean and Filter
    df.columns = df.columns.str.strip()
    
    # We only need 'Concept' (the word) and 'Feature'
    # Note: McRae's raw files usually have 'Concept' and 'Feature' columns.
    if 'Concept' not in df.columns or 'Feature' not in df.columns:
        print("FATAL ERROR: McRae file missing 'Concept' or 'Feature' column. Check headers.")
        sys.exit(1)
        
    df_features = df[['Concept', 'Feature']].copy()
    
    # Clean the concept name (word) and filter to only include words in your study
    df_features['Concept'] = df_features['Concept'].astype(str).str.strip().str.lower()
    df_filtered = df_features[df_features['Concept'].isin(user_words)]
    
    # 3. Clean Features
    def clean_feature(feature):
        # Apply the same cleaning rules as the LLM output: lowercase, replace spaces with hyphens
        return re.sub(r'\s+', '-', str(feature).lower().strip())
    
    df_filtered['Feature_Clean'] = df_filtered['Feature'].apply(clean_feature)

    # 4. Aggregate: Group by concept and collect all cleaned features into a list
    mcrae_json_dict = df_filtered.groupby('Concept')['Feature_Clean'].apply(list).to_dict()
    
    # 5. Save the JSON file
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(mcrae_json_dict, f, indent=2)
        
    concepts_found = len(mcrae_json_dict)
    
    print(f"\nProcessed {concepts_found} concepts that overlap with your list.")
    print(f"Gold Standard JSON saved to: {OUTPUT_JSON_FILE}")
    print("\nNext step: Run the semantic_validator.py script.")
    
if __name__ == '__main__':
    process_mcrae_norms()