"""
Generate semantic features for ALL 800 words using optimal strategy per category.
Uses the strategy determined by strategy_comparison.py
"""

import json
import time
import argparse
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
WORDS_FILE = PROJECT_ROOT / "my_800_words.csv"
CATEGORY_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "word_categories.json"
MCRAE_FILE = PROJECT_ROOT / "outputs" / "mcrae_gold_standard.json"
STRATEGY_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "category_strategy.json"
OUTPUT_DIR = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "output"

MODEL = "gpt-5.1"
TEMPERATURE = 0.7
BATCH_SIZE = 30

# Import from category_batch_generator
from category_batch_generator import (
    CATEGORY_EXAMPLES, GENERAL_EXAMPLES,
    parse_json_response, build_prompt, get_examples_for_category
)


def load_api_key():
    with open(API_KEY_FILE, 'r') as f:
        return f.read().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch number (1-based)')
    parser.add_argument('--all', action='store_true', help='Process all words')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "all_words_features.json"
    
    # Load data
    print("Loading data...")
    
    import pandas as pd
    words_df = pd.read_csv(WORDS_FILE)
    all_words = sorted(words_df['word'].str.strip().str.lower().tolist())
    
    with open(CATEGORY_FILE, 'r') as f:
        word_categories = json.load(f)
    
    with open(MCRAE_FILE, 'r') as f:
        mcrae_data = json.load(f)
    
    # Load strategy (default to category-specific if no strategy file)
    strategy = {}
    if STRATEGY_FILE.exists():
        with open(STRATEGY_FILE, 'r') as f:
            strategy = json.load(f)
        print(f"Loaded strategy for {len(strategy)} categories")
        general_cats = [c for c, s in strategy.items() if s == 'general']
        if general_cats:
            print(f"Using GENERAL prompt for: {', '.join(general_cats)}")
    else:
        print("No strategy file found. Using category-specific for all.")
    
    # All words can be generated (examples are dynamically swapped out per word)
    words_to_generate = all_words
    
    print(f"\nTotal words: {len(all_words)}")
    
    # Load existing results and skip already done
    results = {}
    if output_file.exists():
        with open(output_file, 'r') as f:
            results = json.load(f)
        print(f"Already generated: {len(results)} words (will skip)")
        words_to_generate = [w for w in words_to_generate if w not in results]
        print(f"Remaining to generate: {len(words_to_generate)}")
    
    # Select batch
    if not args.all:
        start_idx = (args.batch - 1) * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        words_to_generate = words_to_generate[start_idx:end_idx]
    
    if not words_to_generate:
        print("No words to process!")
        return
    
    print(f"\nProcessing {len(words_to_generate)} words...")
    
    # Initialize client
    client = OpenAI(api_key=load_api_key(), base_url="https://us.api.openai.com/v1")
    
    # Generate
    for i, word in enumerate(words_to_generate):
        category = word_categories.get(word, 'OTHER')
        use_general = strategy.get(category, 'category') == 'general'
        prompt_type = "GENERAL" if use_general else category
        
        print(f"\n[{i+1}/{len(words_to_generate)}] {word.upper()} ({prompt_type})")
        
        # Get examples
        if use_general:
            examples = [(w, mcrae_data[w]) for w in GENERAL_EXAMPLES 
                       if w in mcrae_data and w != word][:5]
        else:
            examples = get_examples_for_category(category, mcrae_data, exclude_word=word)
        
        # Build prompt and generate
        prompt = build_prompt(word, examples)
        
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_completion_tokens=1000,
            )
            content = response.choices[0].message.content.strip()
            features = parse_json_response(content)
        except Exception as e:
            print(f"  Error: {e}")
            features = []
        
        results[word] = {
            "category": category,
            "prompt_type": prompt_type,
            "features": features
        }
        
        print(f"  Generated {len(features)} features")
        
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        
        time.sleep(0.5)
    
    print(f"\n\nResults saved to: {output_file}")
    print(f"Total words processed: {len(results)}")


if __name__ == "__main__":
    main()


