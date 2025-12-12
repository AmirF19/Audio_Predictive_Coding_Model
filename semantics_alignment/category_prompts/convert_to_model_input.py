"""
Convert all_words_features.json to model input format (word -> feature list).
Outputs: all_words_model_input.json
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
OUTPUT_DIR = PROJECT_ROOT / "semantics_alignment" / "category_prompts" / "output"
INPUT_FILE = OUTPUT_DIR / "all_words_features.json"
OUTPUT_FILE = OUTPUT_DIR / "all_words_model_input.json"
FEATURE_COUNT = 30  # truncate if more


def dedup_preserve_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    out = {}
    empty_words = []

    for word, payload in data.items():
        feats = payload.get("features", []) or []
        feats = [str(x).strip() for x in feats if str(x).strip()]
        feats = dedup_preserve_order(feats)
        if FEATURE_COUNT:
            feats = feats[:FEATURE_COUNT]
        if not feats:
            empty_words.append(word)
        out[word] = feats

    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)

    print(f"Converted: {len(out)} words")
    print(f"Empty feature lists: {len(empty_words)}")
    if empty_words:
        print(f"  e.g., {', '.join(empty_words[:10])} ...")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

