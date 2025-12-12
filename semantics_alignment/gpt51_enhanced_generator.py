"""
GPT 5.1 Enhanced Semantic Feature Generator

Combines multiple strategies:
1. Multiple runs per word (3x)
2. Generate more features (35 per run)
3. Consensus selection (features in 2+ runs)
4. Final trim to 25 features
"""

import os
import json
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI
from collections import Counter

# Configuration
PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
PROMPT_FILE = PROJECT_ROOT / "semantics_alignment" / "chatgpt_optimized_prompt.txt"
WORD_LIST_FILE = PROJECT_ROOT / "my_800_words.csv"
MCRAE_FILE = PROJECT_ROOT / "outputs" / "mcrae_gold_standard.json"

OUTPUT_DIR = PROJECT_ROOT / "semantics_alignment" / "gpt51_enhanced"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

RAW_JSON = OUTPUT_DIR / "gpt51_enhanced_raw.json"
FINAL_JSON = OUTPUT_DIR / "gpt51_enhanced_model_input.json"

# Model settings
MODEL = "gpt-5.1"
TEMPERATURE = 0.7
RUNS_PER_WORD = 3
FEATURES_PER_RUN = 30
FINAL_FEATURE_COUNT = 25
BATCH_SIZE = 25


def load_api_key():
    with open(API_KEY_FILE, 'r') as f:
        return f.read().strip()


def load_prompt_template():
    with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
        template = f.read()
    # Prompt already says 30 features, no modification needed
    return template


def load_word_list():
    df = pd.read_csv(WORD_LIST_FILE)
    if 'word' in df.columns:
        words = df['word'].astype(str).str.strip().str.lower().tolist()
    else:
        words = df.iloc[:, 0].astype(str).str.strip().str.lower().tolist()
    return sorted(list(set([w for w in words if w])))


def load_mcrae_words():
    with open(MCRAE_FILE, 'r') as f:
        mcrae = json.load(f)
    return list(mcrae.keys())


def load_progress():
    if RAW_JSON.exists():
        with open(RAW_JSON, 'r') as f:
            return json.load(f)
    return {}


def save_progress(data):
    with open(RAW_JSON, 'w') as f:
        json.dump(data, f, indent=2)


def generate_features_single_run(client, prompt_template, word):
    """Generate features for a single run."""
    prompt = prompt_template.replace('[WORD_TO_TEST]', word)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_completion_tokens=1500,  # More tokens for 35 features
        )
        
        content = response.choices[0].message.content.strip()
        
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end != 0:
            json_content = content[json_start:json_end]
            parsed = json.loads(json_content)
            if 'features' in parsed and isinstance(parsed['features'], list):
                return parsed['features']
        
        return None
        
    except Exception as e:
        print(f"    Error: {e}")
        return None


def consensus_selection(all_runs):
    """
    Select features that appear in 2+ runs.
    Returns features sorted by frequency (most common first).
    """
    # Normalize features for comparison
    def normalize(f):
        return f.lower().strip().replace(' ', '_')
    
    # Count feature occurrences across runs
    feature_counts = Counter()
    feature_original = {}  # Keep original formatting
    
    for run in all_runs:
        if run:
            seen_in_run = set()
            for feat in run:
                norm = normalize(feat)
                if norm not in seen_in_run:
                    feature_counts[norm] += 1
                    seen_in_run.add(norm)
                    # Keep first seen version
                    if norm not in feature_original:
                        feature_original[norm] = feat
    
    # Select features appearing in 2+ runs (consensus)
    consensus_features = []
    for norm_feat, count in feature_counts.most_common():
        if count >= 2:  # Appears in at least 2 runs
            consensus_features.append(feature_original[norm_feat])
    
    # If not enough consensus features, add top single-run features
    if len(consensus_features) < FINAL_FEATURE_COUNT:
        for norm_feat, count in feature_counts.most_common():
            if count == 1 and len(consensus_features) < FINAL_FEATURE_COUNT:
                if feature_original[norm_feat] not in consensus_features:
                    consensus_features.append(feature_original[norm_feat])
    
    return consensus_features[:FINAL_FEATURE_COUNT]


def process_word(client, prompt_template, word, existing_data):
    """Process a single word with multiple runs and consensus."""
    
    # Check if already processed
    if word in existing_data and 'final_features' in existing_data[word]:
        return existing_data[word]['final_features']
    
    print(f"  Processing '{word}'...")
    
    all_runs = []
    
    for run_num in range(RUNS_PER_WORD):
        print(f"    Run {run_num + 1}/{RUNS_PER_WORD}...", end=" ")
        features = generate_features_single_run(client, prompt_template, word)
        
        if features:
            print(f"{len(features)} features")
            all_runs.append(features)
        else:
            print("FAILED")
            all_runs.append([])
        
        time.sleep(1)  # Rate limiting
    
    # Apply consensus selection
    final_features = consensus_selection(all_runs)
    
    # Store results
    existing_data[word] = {
        'all_runs': all_runs,
        'final_features': final_features,
        'consensus_count': len([f for f in final_features if sum(1 for run in all_runs if f in run) >= 2])
    }
    
    print(f"    Final: {len(final_features)} features ({existing_data[word]['consensus_count']} consensus)")
    
    return final_features


def run_batch(batch_num, words_to_process):
    """Run a batch of words."""
    print(f"\n{'='*60}")
    print(f"ENHANCED BATCH {batch_num}: {len(words_to_process)} words")
    print(f"Settings: {RUNS_PER_WORD} runs x {FEATURES_PER_RUN} features, consensus selection")
    print(f"{'='*60}")
    
    # Load existing data
    existing_data = load_progress()
    
    # Initialize client
    api_key = load_api_key()
    client = OpenAI(
        api_key=api_key,
        base_url="https://us.api.openai.com/v1"
    )
    prompt_template = load_prompt_template()
    
    for i, word in enumerate(words_to_process, 1):
        print(f"\n[{i}/{len(words_to_process)}] ", end="")
        process_word(client, prompt_template, word, existing_data)
        
        # Save progress after each word
        save_progress(existing_data)
    
    # Generate final output
    print("\nGenerating final model input...")
    final_data = {word: data['final_features'] for word, data in existing_data.items() if 'final_features' in data}
    with open(FINAL_JSON, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"Saved to: {FINAL_JSON}")
    return existing_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Enhanced GPT 5.1 feature generator')
    parser.add_argument('--batch', type=int, default=1, help='Batch number (1-indexed)')
    parser.add_argument('--list-batches', action='store_true', help='List batches and exit')
    args = parser.parse_args()
    
    # Load words
    all_words = load_word_list()
    mcrae_words = load_mcrae_words()
    
    # Order: McRae first
    mcrae_in_lexicon = [w for w in mcrae_words if w in all_words]
    other_words = [w for w in all_words if w not in mcrae_words]
    ordered_words = mcrae_in_lexicon + other_words
    
    # Split into batches
    batches = []
    for i in range(0, len(ordered_words), BATCH_SIZE):
        batches.append(ordered_words[i:i + BATCH_SIZE])
    
    print(f"Total words: {len(ordered_words)}")
    print(f"McRae words: {len(mcrae_in_lexicon)}")
    print(f"Total batches: {len(batches)}")
    
    if args.list_batches:
        for i, batch in enumerate(batches, 1):
            print(f"  Batch {i}: {batch[0]} ... {batch[-1]}")
        return
    
    if args.batch < 1 or args.batch > len(batches):
        print(f"Error: batch must be 1-{len(batches)}")
        return
    
    batch_words = batches[args.batch - 1]
    run_batch(args.batch, batch_words)


if __name__ == '__main__':
    main()
