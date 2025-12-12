"""
GPT 5.1 Semantic Feature Generator - Batch Mode

Processes words in batches of 25, starting with McRae-verifiable words.
Saves after each batch for verification and resume capability.
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# Configuration
PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
PROMPT_FILE = PROJECT_ROOT / "semantics_alignment" / "chatgpt_optimized_prompt.txt"
WORD_LIST_FILE = PROJECT_ROOT / "my_800_words.csv"
MCRAE_FILE = PROJECT_ROOT / "outputs" / "mcrae_gold_standard.json"

OUTPUT_DIR = PROJECT_ROOT / "semantics_alignment" / "gpt51_features"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Output files
RAW_JSON = OUTPUT_DIR / "gpt51_features_raw.json"
FINAL_JSON = OUTPUT_DIR / "gpt51_features_model_input.json"
PROGRESS_FILE = OUTPUT_DIR / "progress.json"

# Model settings
MODEL = "gpt-5.1"
TEMPERATURE = 0.7
BATCH_SIZE = 25


def load_api_key():
    """Load API key from file."""
    with open(API_KEY_FILE, 'r') as f:
        return f.read().strip()


def load_prompt_template():
    """Load the optimized prompt."""
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        return f.read()


def load_word_list():
    """Load full word list."""
    df = pd.read_csv(WORD_LIST_FILE)
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
    return sorted(list(set([w for w in words if w])))


def load_mcrae_words():
    """Load McRae gold standard words."""
    with open(MCRAE_FILE, 'r') as f:
        mcrae = json.load(f)
    return list(mcrae.keys())


def load_progress():
    """Load progress from previous runs."""
    if RAW_JSON.exists():
        with open(RAW_JSON, 'r') as f:
            return json.load(f)
    return {}


def save_progress(data):
    """Save current progress."""
    with open(RAW_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def generate_features(client, prompt_template, word):
    """Generate features for a single word."""
    prompt = prompt_template.replace('[WORD_TO_TEST]', word)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_completion_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_content = content[json_start:json_end]
            parsed = json.loads(json_content)
            if 'features' in parsed and isinstance(parsed['features'], list):
                return parsed['features']
        
        return None
        
    except Exception as e:
        print(f"  Error for '{word}': {e}")
        return None


def run_batch(client, prompt_template, words, existing_data, batch_num):
    """Run a single batch of words."""
    print(f"\n{'='*60}")
    print(f"BATCH {batch_num}: Processing {len(words)} words")
    print(f"{'='*60}")
    
    results = {}
    
    for i, word in enumerate(words, 1):
        if word in existing_data:
            print(f"  [{i}/{len(words)}] {word}: already processed, skipping")
            continue
            
        print(f"  [{i}/{len(words)}] {word}: generating...", end=" ")
        
        features = generate_features(client, prompt_template, word)
        
        if features:
            results[word] = features
            existing_data[word] = features
            print(f"{len(features)} features")
        else:
            print("FAILED")
        
        # Rate limiting
        time.sleep(1.5)
    
    # Save after batch
    save_progress(existing_data)
    print(f"\nBatch {batch_num} complete. Progress saved.")
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate semantic features with GPT 5.1')
    parser.add_argument('--batch', type=int, default=1, help='Batch number to run (1-indexed)')
    parser.add_argument('--mcrae-first', action='store_true', default=True, 
                        help='Process McRae words first for validation')
    parser.add_argument('--list-batches', action='store_true', help='List all batches and exit')
    args = parser.parse_args()
    
    # Load data
    all_words = load_word_list()
    mcrae_words = load_mcrae_words()
    existing_data = load_progress()
    
    print(f"Total words: {len(all_words)}")
    print(f"McRae words (for validation): {len(mcrae_words)}")
    print(f"Already processed: {len(existing_data)}")
    
    # Order words: McRae first, then remaining
    if args.mcrae_first:
        mcrae_in_lexicon = [w for w in mcrae_words if w in all_words]
        other_words = [w for w in all_words if w not in mcrae_words]
        ordered_words = mcrae_in_lexicon + other_words
    else:
        ordered_words = all_words
    
    # Split into batches
    batches = []
    for i in range(0, len(ordered_words), BATCH_SIZE):
        batches.append(ordered_words[i:i + BATCH_SIZE])
    
    print(f"Total batches: {len(batches)} (batch size: {BATCH_SIZE})")
    
    if args.list_batches:
        print("\nBatch contents:")
        for i, batch in enumerate(batches, 1):
            done = sum(1 for w in batch if w in existing_data)
            print(f"  Batch {i}: {batch[0]} ... {batch[-1]} ({done}/{len(batch)} done)")
        return
    
    # Validate batch number
    if args.batch < 1 or args.batch > len(batches):
        print(f"Error: batch must be 1-{len(batches)}")
        return
    
    # Initialize client with US regional endpoint
    api_key = load_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://us.api.openai.com/v1"
    )
    prompt_template = load_prompt_template()
    
    # Run selected batch
    batch_words = batches[args.batch - 1]
    run_batch(client, prompt_template, batch_words, existing_data, args.batch)
    
    # Generate final output after each batch
    print("\nGenerating model input file...")
    with open(FINAL_JSON, 'w') as f:
        json.dump(existing_data, f, indent=2)
    print(f"Saved to: {FINAL_JSON}")


if __name__ == '__main__':
    main()
