"""
Category-specific semantic feature generator.
Uses WordNet-derived categories to select relevant McRae examples.
"""

import json
import time
import argparse
from pathlib import Path
from openai import OpenAI

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
API_KEY_FILE = PROJECT_ROOT / "api_key.txt"
CATEGORY_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "word_categories.json"
MCRAE_FILE = PROJECT_ROOT / "outputs" / "mcrae_gold_standard.json"
OUTPUT_DIR = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "output"

MODEL = "gpt-5.1"
TEMPERATURE = 0.7
BATCH_SIZE = 30
FEATURE_COUNT = 30

# McRae examples for each category (5 per category)
# These are manually selected from McRae gold standard
CATEGORY_EXAMPLES = {
    'ANIMAL': ['tiger', 'lion', 'rabbit', 'squirrel', 'deer'],
    'BIRD': ['eagle', 'robin', 'sparrow', 'pigeon', 'crow'],
    'INSECT': ['beetle', 'butterfly', 'ant', 'bee', 'fly'],
    'FISH': ['salmon', 'trout', 'tuna', 'cod', 'bass'],
    'FRUIT': ['apple', 'orange', 'lemon', 'cherry', 'banana'],
    'VEGETABLE': ['carrot', 'cabbage', 'onion', 'radish', 'celery'],
    'FOOD': ['bread', 'cheese', 'butter', 'chicken', 'pizza'],
    'TOOL': ['hammer', 'shovel', 'chisel', 'pliers', 'screwdriver'],
    'WEAPON': ['sword', 'gun', 'dagger', 'rifle', 'knife'],
    'VEHICLE': ['car', 'airplane', 'bicycle', 'train', 'truck'],
    'CONTAINER': ['bucket', 'barrel', 'basket', 'bottle', 'jar'],
    'FURNITURE': ['sofa', 'table', 'dresser', 'cradle', 'chair'],
    'BUILDING': ['church', 'barn', 'hotel', 'chapel', 'house'],
    'ROOM': ['bedroom', 'kitchen', 'bathroom', 'closet', 'basement'],
    'CLOTHING': ['jacket', 'pants', 'shirt', 'dress', 'coat'],
    'APPLIANCE': ['refrigerator', 'stove', 'oven', 'toaster', 'blender'],
    'MUSICAL_INSTRUMENT': ['piano', 'guitar', 'trumpet', 'violin', 'drum'],
    'PLANT': ['oak', 'rose', 'daisy', 'tulip', 'pine'],
    'BODY_PART': ['arm', 'leg', 'heart', 'brain', 'hand'],
    'MATERIAL': ['cotton', 'leather', 'rubber', 'metal', 'wood'],
    'NATURAL_FEATURE': ['mountain', 'river', 'ocean', 'valley', 'lake'],
    'PLACE': ['city', 'village', 'farm', 'beach', 'park'],
    'PERSON': ['doctor', 'teacher', 'soldier', 'artist', 'nurse'],
    'EMOTION': ['anger', 'fear', 'joy', 'sadness', 'love'],
    'COGNITION': ['idea', 'memory', 'knowledge', 'belief', 'thought'],
    'COMMUNICATION': ['letter', 'message', 'story', 'novel', 'book'],
    'EVENT': ['party', 'concert', 'wedding', 'funeral', 'game'],
    'STATE': ['freedom', 'danger', 'safety', 'peace', 'comfort'],
    'TIME': ['morning', 'evening', 'summer', 'winter', 'night'],
    'QUANTITY': ['dozen', 'million', 'gallon', 'pound', 'inch'],
    'OTHER': ['hammer', 'apple', 'sofa', 'car', 'bottle'],  # Generic fallback
}


def load_api_key():
    with open(API_KEY_FILE, 'r') as f:
        return f.read().strip()


def load_categories():
    """Load word -> category mapping."""
    with open(CATEGORY_FILE, 'r') as f:
        return json.load(f)


def load_mcrae():
    """Load McRae gold standard."""
    with open(MCRAE_FILE, 'r') as f:
        return json.load(f)


def get_examples_for_category(category, mcrae_data, exclude_word=None):
    """Get up to 5 McRae examples for a category."""
    example_words = CATEGORY_EXAMPLES.get(category, CATEGORY_EXAMPLES['OTHER'])
    examples = []
    
    for word in example_words:
        if word == exclude_word:
            continue
        if word in mcrae_data:
            examples.append((word, mcrae_data[word]))
        if len(examples) >= 5:
            break
    
    return examples


def build_prompt(target_word, examples):
    """Build prompt with category-specific examples."""
    
    prompt = f"""Persona: You are a computational research scientist with a joint appointment from UC Berkeley and MIT, specializing in computational psycholinguistics. You are tasked with recovering lost data from a semantic attribution task. You have access to the data and instructions provided below. You will be provided one word at a time. You are to provide {FEATURE_COUNT} total features in JSON format that accord with the instructions below.

Generate semantic attributes for a given word for the McRae et al. (2005) feature production task:
1. For a given concept, list the specific properties of the concept to which the word refers
2. Types of properties include: physical properties (internal/external parts, appearance, sounds, smells, taste); functional properties (what it is used for, where/when/by whom it is used); categorical relations; and behavioral facts
3. All words are considered as **nouns only**
4. Generate exactly {FEATURE_COUNT} distinct properties
5. Use underscores between words (no spaces)
6. **Respond ONLY with a valid JSON object in the format: {{"features": ["property1", "property2", ...]}}**

Below are examples of recovered participant responses:

"""
    
    # Add examples (up to 5)
    for word, features in examples[:5]:
        features_str = '", "'.join(features)
        prompt += f'Word: {word}\n'
        prompt += f'Participant: {{"features": ["{features_str}"]}}\n\n'
    
    # Target word
    prompt += f'Word: {target_word}\nParticipant:'
    
    return prompt


def parse_json_response(content):
    """Parse JSON from GPT response with error handling."""
    import re
    
    # Try direct parse
    try:
        if content.startswith('{'):
            return json.loads(content).get('features', [])
    except:
        pass
    
    # Extract JSON from response
    try:
        start = content.find('{')
        end = content.rfind('}') + 1
        if start != -1 and end > start:
            json_str = content[start:end]
            return json.loads(json_str).get('features', [])
    except:
        pass
    
    # Try to fix common issues (unescaped quotes in features)
    try:
        # Find the features array
        match = re.search(r'"features"\s*:\s*\[(.*?)\]', content, re.DOTALL)
        if match:
            features_str = match.group(1)
            # Split by ", " pattern and clean up
            features = re.findall(r'"([^"]*)"', features_str)
            return features
    except:
        pass
    
    return []


def generate_features(client, word, category, mcrae_data):
    """Generate features for a single word."""
    
    examples = get_examples_for_category(category, mcrae_data, exclude_word=word)
    prompt = build_prompt(word, examples)
    
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_completion_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        return parse_json_response(content)
        
    except Exception as e:
        print(f"  Error for {word}: {e}")
        return []


LOW_PERFORMERS_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "low_performers.json"
UNDERPERFORMING_CATS_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "underperforming_categories.json"
STRATEGY_FILE = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "category_strategy.json"

# General examples (used for low performers or fallback)
# Diverse set: animal, tool, furniture, fruit, container
GENERAL_EXAMPLES = ['tiger', 'hammer', 'sofa', 'apple', 'bottle']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1, help='Batch number (1-based)')
    parser.add_argument('--all', action='store_true', help='Process all McRae words')
    parser.add_argument('--use-general-for-low', action='store_true', 
                        help='Re-run low performers with general prompt')
    parser.add_argument('--test-general-for-categories', action='store_true',
                        help='Test general prompt for underperforming categories')
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    word_categories = load_categories()
    mcrae_data = load_mcrae()
    mcrae_words = sorted(mcrae_data.keys())
    
    # Check if re-running low performers
    low_performer_words = set()
    underperforming_categories = set()
    
    if args.use_general_for_low:
        if LOW_PERFORMERS_FILE.exists():
            with open(LOW_PERFORMERS_FILE, 'r') as f:
                low_data = json.load(f)
                low_performer_words = set(low_data.get('words', []))
            print(f"Re-running {len(low_performer_words)} low performers with GENERAL prompt")
        else:
            print("No low_performers.json found. Run identify_low_performers.py first.")
            return
    
    if args.test_general_for_categories:
        if UNDERPERFORMING_CATS_FILE.exists():
            with open(UNDERPERFORMING_CATS_FILE, 'r') as f:
                cat_data = json.load(f)
                underperforming_categories = set(cat_data.get('categories', []))
            print(f"Testing GENERAL prompt for {len(underperforming_categories)} underperforming categories:")
            print(f"  {', '.join(underperforming_categories)}")
        else:
            print("No underperforming_categories.json found. Run strategy_comparison.py first.")
            return
    
    # ALL McRae words are available (examples are dynamically swapped out per word)
    test_words = mcrae_words
    
    print(f"Total McRae words: {len(mcrae_words)}")
    
    # Select batch
    if args.use_general_for_low:
        # Only process low performers
        words_to_process = [w for w in test_words if w in low_performer_words]
    elif args.test_general_for_categories:
        # Only process words from underperforming categories
        words_to_process = [w for w in test_words 
                           if word_categories.get(w, 'OTHER') in underperforming_categories]
    elif args.all:
        words_to_process = test_words
    else:
        start_idx = (args.batch - 1) * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        words_to_process = test_words[start_idx:end_idx]
    
    if not words_to_process:
        print(f"No words in batch {args.batch}")
        return
    
    print(f"\nProcessing batch {args.batch}: {len(words_to_process)} words")
    print(f"Words: {', '.join(words_to_process[:10])}...")
    
    # Load existing results (use different file for general testing)
    if args.test_general_for_categories:
        output_file = OUTPUT_DIR / "general_features_raw.json"
    else:
        output_file = OUTPUT_DIR / "category_features_raw.json"
    
    if output_file.exists():
        with open(output_file, 'r') as f:
            results = json.load(f)
    else:
        results = {}
    
    # Skip words already generated
    already_done = set(results.keys())
    if already_done:
        print(f"Already generated: {len(already_done)} words (will skip)")
        words_to_process = [w for w in words_to_process if w not in already_done]
    
    # Initialize client
    client = OpenAI(api_key=load_api_key(), base_url="https://us.api.openai.com/v1")
    
    # Generate features
    print("\nGenerating features with category-specific examples...")
    print("=" * 60)
    
    for i, word in enumerate(words_to_process):
        category = word_categories.get(word, 'OTHER')
        
        # Use general prompt for low performers or underperforming categories
        use_general = (args.use_general_for_low and word in low_performer_words) or \
                      (args.test_general_for_categories and category in underperforming_categories)
        prompt_type = "GENERAL" if use_general else category
        print(f"\n[{i+1}/{len(words_to_process)}] {word.upper()} (Prompt: {prompt_type})")
        
        if use_general:
            # Use general examples
            examples = [(w, mcrae_data[w]) for w in GENERAL_EXAMPLES if w in mcrae_data and w != word][:5]
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
                print(f"  Error for {word}: {e}")
                features = []
        else:
            features = generate_features(client, word, category, mcrae_data)
        
        results[word] = {
            "category": category,
            "features": features
        }
        
        print(f"  Generated {len(features)} features")
        if features:
            print(f"  Sample: {features[:3]}")
        
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, sort_keys=True)
        
        time.sleep(0.5)
    
    print(f"\n\nResults saved to: {output_file}")
    print(f"Total words processed: {len(results)}")
    
    # Category breakdown
    cat_counts = {}
    for data in results.values():
        cat = data.get('category', 'OTHER')
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    print("\nCategory breakdown:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
