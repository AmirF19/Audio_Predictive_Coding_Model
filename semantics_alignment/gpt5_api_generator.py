import os
import json
import time
from pathlib import Path
from openai import OpenAI

# Configuration - Load API key from file
try:
    with open(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\api_key.txt", 'r') as f:
        API_KEY = f.read().strip()
except FileNotFoundError:
    print("❌ Error: api_key.txt not found!")
    print("Please create the file with your OpenAI API key.")
    exit(1)
MODEL = "gpt-5.1"  # Using GPT-5.1 as requested
OUTPUT_DIR = Path("chat_gpt_output_gpt51")

# Test words
TEST_WORDS = ['rocker', 'mirror', 'level', 'football', 'garlic', 'menu', 'carpet', 'cigar', 'hamster', 'buckle']

def load_optimized_prompt():
    """Load the optimized prompt template."""
    with open('chatgpt_optimized_prompt.txt', 'r', encoding='utf-8') as f:
        return f.read()

def generate_features_for_word(client, prompt_template, word):
    """
    Generate semantic features for a single word using GPT-5.1.
    """
    # Replace placeholder with actual word
    prompt = prompt_template.replace('[WORD_TO_TEST]', word)

    try:
        print(f"Generating features for '{word}'...")

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Balanced creativity vs consistency
            max_completion_tokens=1000,  # Should be plenty for 25 features
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )

        # Extract the response content
        content = response.choices[0].message.content.strip()

        # Try to parse as JSON
        try:
            # Find JSON in the response (in case there's extra text)
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_content = content[json_start:json_end]
                parsed_response = json.loads(json_content)

                # Validate it has the expected structure
                if 'features' in parsed_response and isinstance(parsed_response['features'], list):
                    return parsed_response['features']
                else:
                    print(f"Warning: Unexpected JSON structure for '{word}'")
                    return None
            else:
                print(f"Warning: No JSON found in response for '{word}'")
                return None

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for '{word}': {e}")
            print(f"Raw response: {content[:500]}...")
            return None

    except Exception as e:
        print(f"Error generating features for '{word}': {e}")
        return None

def save_features_to_file(word, features):
    """Save features to individual JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    data = {"features": features}
    output_file = OUTPUT_DIR / f"{word}.txt"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(features)} features for '{word}' to {output_file}")

def main():
    # Check API key
    if API_KEY == "YOUR_OPENAI_API_KEY_HERE":
        print("❌ Please replace 'YOUR_OPENAI_API_KEY_HERE' with your actual OpenAI API key!")
        return

    # Initialize OpenAI client
    client = OpenAI(api_key=API_KEY)

    # Load prompt template
    try:
        prompt_template = load_optimized_prompt()
        print("✅ Loaded optimized prompt template")
    except FileNotFoundError:
        print("❌ Error: chatgpt_optimized_prompt.txt not found!")
        return

    # Generate features for each test word
    results = {}

    for i, word in enumerate(TEST_WORDS, 1):
        print(f"\n🔄 [{i}/{len(TEST_WORDS)}] Processing '{word}'...")

        features = generate_features_for_word(client, prompt_template, word)

        if features:
            save_features_to_file(word, features)
            results[word] = len(features)
            print(f"✅ Success: Generated {len(features)} features")
        else:
            print(f"❌ Failed to generate features for '{word}'")
            results[word] = 0

        # Rate limiting - be respectful to the API
        if i < len(TEST_WORDS):  # Don't wait after the last request
            print("⏳ Waiting 2 seconds for rate limiting...")
            time.sleep(2)

    # Summary
    print("\n🎉 GENERATION COMPLETE!")
    print("=" * 50)
    print(f"Processed {len(TEST_WORDS)} words:")
    total_features = 0
    for word, count in results.items():
        status = "✅" if count > 0 else "❌"
        print(f"  {status} {word}: {count} features")
        total_features += count

    print(f"\n📊 Total features generated: {total_features}")
    print(f"💾 Files saved to: {OUTPUT_DIR}/")

    # Cost estimate (rough)
    estimated_tokens = (len(prompt_template.split()) * len(TEST_WORDS) * 1.3) + (total_features * 8)
    cost_estimate = (estimated_tokens / 1_000_000) * 11.25  # GPT-5.1 total rate
    print(".4f")
if __name__ == '__main__':
    main()
