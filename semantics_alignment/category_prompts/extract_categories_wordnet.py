"""
Extract semantic categories using McRae features first, then WordNet.
McRae provides ground truth for words it contains.
"""

import json
import re
from pathlib import Path
from collections import defaultdict

try:
    from nltk.corpus import wordnet as wn
    import nltk
except ImportError:
    print("Installing NLTK...")
    import subprocess
    subprocess.run(['pip', 'install', 'nltk'], check=True)
    from nltk.corpus import wordnet as wn
    import nltk

# Download WordNet data if needed
try:
    wn.synsets('test')
except LookupError:
    print("Downloading WordNet data...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')

PROJECT_ROOT = Path(r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project")
WORDS_FILE = PROJECT_ROOT / "my_800_words.csv"
MCRAE_FILE = PROJECT_ROOT / "outputs" / "mcrae_gold_standard.json"
OUTPUT_DIR = PROJECT_ROOT / "semantics_alignment" / "category_prompts"

# McRae taxonomic feature patterns -> category
MCRAE_PATTERNS = {
    r'^a_bird$': 'BIRD',
    r'^a_fish$': 'FISH',
    r'^an_insect$': 'INSECT',
    r'^a_bug$': 'INSECT',
    r'^an_animal$': 'ANIMAL',
    r'^a_mammal$': 'ANIMAL',
    r'^a_reptile$': 'ANIMAL',
    r'^a_carnivore$': 'ANIMAL',
    r'^a_predator$': 'ANIMAL',
    r'^a_rodent$': 'ANIMAL',
    r'^a_fruit$': 'FRUIT',
    r'^a_vegetable$': 'VEGETABLE',
    r'^a_food$': 'FOOD',
    r'^a_tool$': 'TOOL',
    r'^a_utensil$': 'TOOL',
    r'^a_weapon$': 'WEAPON',
    r'^a_vehicle$': 'VEHICLE',
    r'^a_container$': 'CONTAINER',
    r'^furniture$': 'FURNITURE',
    r'^a_piece_of_furniture$': 'FURNITURE',
    r'^a_building$': 'BUILDING',
    r'^a_dwelling$': 'BUILDING',
    r'^clothing$': 'CLOTHING',
    r'^a_garment$': 'CLOTHING',
    r'^an_appliance$': 'APPLIANCE',
    r'^a_musical_instrument$': 'MUSICAL_INSTRUMENT',
    r'^a_plant$': 'PLANT',
    r'^a_tree$': 'PLANT',
    r'^a_flower$': 'PLANT',
    r'^a_body_part$': 'BODY_PART',
}

# Manual overrides for known problematic words (word sense disambiguation)
MANUAL_OVERRIDES = {
    # Wrong animal sense
    'permit': 'COMMUNICATION',
    'hobby': 'EVENT',
    'flicker': 'EVENT',
    'quarrel': 'COMMUNICATION',
    'beaver': 'ANIMAL',
    'tiger': 'ANIMAL',
    'bunny': 'ANIMAL',
    
    # Wrong insect sense  
    'soldier': 'PERSON',
    'worker': 'PERSON',
    
    # CLOTHING
    'blanket': 'CLOTHING',
    'bracelet': 'CLOTHING',
    'collar': 'CLOTHING',
    'helmet': 'CLOTHING',
    'jersey': 'CLOTHING',
    'necklace': 'CLOTHING',
    'outfit': 'CLOTHING',
    'sandal': 'CLOTHING',
    'slipper': 'CLOTHING',
    
    # VEGETABLE
    'carrot': 'VEGETABLE',
    'onion': 'VEGETABLE',
    
    # FRUIT
    'walnut': 'FRUIT',
    'cherry': 'FRUIT',
    'chestnut': 'FOOD',
    
    # WEAPON
    'cannon': 'WEAPON',
    'arrow': 'WEAPON',
    
    # MATERIAL
    'asphalt': 'MATERIAL',
    'cement': 'MATERIAL',
    'concrete': 'MATERIAL',
    'crystal': 'MATERIAL',
    'jewel': 'MATERIAL',
    'lotion': 'MATERIAL',
    'lumber': 'MATERIAL',
    'piping': 'MATERIAL',
    'poison': 'MATERIAL',
    'powder': 'MATERIAL',
    'ribbon': 'MATERIAL',
    'streamer': 'MATERIAL',
    'wire': 'MATERIAL',
    
    # FURNITURE
    'carpet': 'FURNITURE',
    'mattress': 'FURNITURE',
    'pillow': 'FURNITURE',
    'table': 'FURNITURE',
    
    # CONTAINER
    'barrel': 'CONTAINER',
    'package': 'CONTAINER',
    'platter': 'CONTAINER',
    
    # PLANT
    'blossom': 'PLANT',
    'bouquet': 'PLANT',
    'petal': 'PLANT',
    'pollen': 'PLANT',
    'seaweed': 'PLANT',
    
    # NATURAL_FEATURE
    'boulder': 'NATURAL_FEATURE',
    'drizzle': 'NATURAL_FEATURE',
    'forest': 'NATURAL_FEATURE',
    'moonlight': 'NATURAL_FEATURE',
    'pebble': 'NATURAL_FEATURE',
    'planet': 'NATURAL_FEATURE',
    'puddle': 'NATURAL_FEATURE',
    'sunlight': 'NATURAL_FEATURE',
    'sunshine': 'NATURAL_FEATURE',
    'weather': 'STATE',
    
    # TOOL
    'bandage': 'TOOL',
    'dial': 'TOOL',
    'ladder': 'TOOL',
    'mouthpiece': 'TOOL',
    'napkin': 'TOOL',
    'needle': 'TOOL',
    
    # APPLIANCE
    'burner': 'APPLIANCE',
    'mixer': 'APPLIANCE',
    'shower': 'APPLIANCE',
    
    # BUILDING (parts and structures)
    'ceiling': 'BUILDING',
    'chimney': 'BUILDING',
    'column': 'BUILDING',
    'doorway': 'BUILDING',
    'entrance': 'BUILDING',
    'exit': 'BUILDING',
    'gutter': 'BUILDING',
    'platform': 'BUILDING',
    'sewer': 'BUILDING',
    'college': 'BUILDING',
    'estate': 'BUILDING',
    'household': 'BUILDING',
    'station': 'BUILDING',
    'tunnel': 'BUILDING',
    
    # PLACE
    'alley': 'PLACE',
    'country': 'PLACE',
    'highway': 'PLACE',
    'nation': 'PLACE',
    'village': 'PLACE',
    
    # PERSON
    'army': 'PERSON',
    'audience': 'PERSON',
    'choir': 'PERSON',
    'council': 'PERSON',
    'devil': 'PERSON',
    'driver': 'PERSON',
    'jury': 'PERSON',
    'mister': 'PERSON',
    'navy': 'PERSON',
    'people': 'PERSON',
    'builder': 'PERSON',
    'toaster': 'APPLIANCE',
    'antique': 'OTHER',
    
    # VEHICLE
    'railroad': 'VEHICLE',
    'tire': 'VEHICLE',
    
    # QUANTITY
    'amount': 'QUANTITY',
    'distance': 'QUANTITY',
    'mileage': 'QUANTITY',
    'output': 'QUANTITY',
    'portion': 'QUANTITY',
    'profit': 'QUANTITY',
    'segment': 'QUANTITY',
    'series': 'QUANTITY',
    
    # BODY_PART
    'body': 'BODY_PART',
    'moustache': 'BODY_PART',
    
    # EVENT
    'circus': 'EVENT',
    'decay': 'EVENT',
    'exchange': 'EVENT',
    'function': 'EVENT',
    'import': 'EVENT',
    'labour': 'EVENT',
    'meeting': 'EVENT',
    'party': 'EVENT',
    'reflex': 'EVENT',
    'shiver': 'EVENT',
    
    # FIXES from manual review
    'chilly': 'STATE',           # temperature, not chili pepper
    'turtle': 'ANIMAL',          # reptile, not turtleneck
    'midnight': 'TIME',
    'nightfall': 'TIME',
    'sunset': 'TIME',
    'anchor': 'TOOL',
    'buckle': 'CLOTHING',
    'button': 'CLOTHING',
    'zipper': 'CLOTHING',
    'candle': 'TOOL',
    'saucer': 'CONTAINER',
    'vacuum': 'APPLIANCE',
    'maple': 'PLANT',
    'soda': 'FOOD',
    'lettuce': 'VEGETABLE',
    'whisker': 'BODY_PART',
    'pedal': 'TOOL',
    'trailer': 'VEHICLE',
    'maroon': 'OTHER',
    'kernel': 'FOOD',
    'lighter': 'TOOL',
    'picture': 'COMMUNICATION',
    'china': 'MATERIAL',
    'express': 'VEHICLE',
    'giant': 'PERSON',
    
    # Keep as OTHER
    'balloon': 'OTHER',
    'upright': 'OTHER',
}

# Target categories mapped to WordNet hypernym synsets
CATEGORY_HYPERNYMS = {
    # Living things
    'ANIMAL': ['animal.n.01', 'mammal.n.01', 'reptile.n.01', 'amphibian.n.01'],
    'BIRD': ['bird.n.01'],
    'FISH': ['fish.n.01'],
    'INSECT': ['insect.n.01'],
    'PLANT': ['vascular_plant.n.01', 'flower.n.01', 'tree.n.01', 'herb.n.01'],
    
    # Food
    'FRUIT': ['edible_fruit.n.01'],
    'VEGETABLE': ['vegetable.n.01', 'root_vegetable.n.01'],
    'FOOD': ['food.n.01', 'food.n.02', 'dish.n.02', 'beverage.n.01', 'nutriment.n.01', 'foodstuff.n.02'],
    
    # Objects
    'TOOL': ['tool.n.01', 'hand_tool.n.01', 'implement.n.01', 'utensil.n.01'],
    'WEAPON': ['weapon.n.01', 'projectile.n.01'],
    'VEHICLE': ['vehicle.n.01', 'wheeled_vehicle.n.01', 'craft.n.02', 'conveyance.n.03'],
    'CONTAINER': ['container.n.01', 'vessel.n.03', 'receptacle.n.01'],
    'FURNITURE': ['furniture.n.01', 'piece_of_furniture.n.01', 'seat.n.03'],
    'APPLIANCE': ['appliance.n.01', 'home_appliance.n.01', 'device.n.01'],
    'MUSICAL_INSTRUMENT': ['musical_instrument.n.01'],
    'CLOTHING': ['clothing.n.01', 'garment.n.01', 'footwear.n.01', 'headdress.n.01', 'accessory.n.01'],
    
    # Places/Structures
    'BUILDING': ['building.n.01', 'dwelling.n.01', 'house.n.01', 'structure.n.01'],
    'ROOM': ['room.n.01', 'enclosure.n.01'],
    'PLACE': ['location.n.01', 'region.n.01', 'area.n.01', 'tract.n.01', 'geological_formation.n.01'],
    'NATURAL_FEATURE': ['body_of_water.n.01', 'land.n.04', 'mountain.n.01', 'valley.n.01'],
    
    # People
    'PERSON': ['person.n.01', 'human.n.01', 'adult.n.01', 'worker.n.01', 'professional.n.01'],
    
    # Body
    'BODY_PART': ['body_part.n.01', 'organ.n.05', 'extremity.n.01'],
    
    # Materials/Substances
    'MATERIAL': ['material.n.01', 'substance.n.01', 'fabric.n.01', 'textile.n.01', 'metal.n.01'],
    
    # Abstract
    'EMOTION': ['feeling.n.01', 'emotion.n.01'],
    'COGNITION': ['cognition.n.01', 'knowledge.n.01', 'idea.n.01', 'concept.n.01'],
    'COMMUNICATION': ['communication.n.02', 'message.n.01', 'writing.n.01', 'document.n.01'],
    'EVENT': ['event.n.01', 'act.n.02', 'activity.n.01', 'social_event.n.01'],
    'STATE': ['state.n.02', 'condition.n.01', 'attribute.n.02'],
    'TIME': ['time_period.n.01', 'time.n.05'],
    'QUANTITY': ['measure.n.02', 'amount.n.03', 'quantity.n.01'],
}

# Priority order for categories (more specific first)
CATEGORY_PRIORITY = [
    # Specific living things first
    'BIRD', 'FISH', 'INSECT',
    'FRUIT', 'VEGETABLE',
    # Then general living
    'ANIMAL', 'PLANT', 'PERSON',
    # Concrete objects
    'FOOD', 'TOOL', 'WEAPON', 'MUSICAL_INSTRUMENT',
    'VEHICLE', 'CONTAINER', 'FURNITURE', 'APPLIANCE', 'CLOTHING',
    # Places
    'ROOM', 'BUILDING', 'NATURAL_FEATURE', 'PLACE',
    # Body/Material
    'BODY_PART', 'MATERIAL',
    # Abstract (last priority)
    'EMOTION', 'COGNITION', 'COMMUNICATION', 'EVENT', 'STATE', 'TIME', 'QUANTITY'
]


def get_hypernym_chain(synset, max_depth=10):
    """Get all hypernyms up to max_depth."""
    hypernyms = set()
    to_visit = [synset]
    depth = 0
    
    while to_visit and depth < max_depth:
        current = to_visit.pop(0)
        for hyper in current.hypernyms():
            if hyper.name() not in hypernyms:
                hypernyms.add(hyper.name())
                to_visit.append(hyper)
        depth += 1
    
    return hypernyms


def categorize_word(word):
    """Categorize a word using manual overrides, then WordNet."""
    
    # Check manual overrides first
    if word.lower() in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[word.lower()], f"MANUAL:{word}"
    
    # Get noun synsets for the word
    synsets = wn.synsets(word, pos=wn.NOUN)
    
    if not synsets:
        # Try without underscores
        word_clean = word.replace('_', ' ')
        synsets = wn.synsets(word_clean, pos=wn.NOUN)
    
    if not synsets:
        return 'OTHER', None
    
    # Check each synset's hypernym chain (only first synset for most common sense)
    for synset in synsets[:1]:  # Use only most common sense
        hypernyms = get_hypernym_chain(synset)
        hypernyms.add(synset.name())
        
        # Check categories in priority order
        for category in CATEGORY_PRIORITY:
            if category in CATEGORY_HYPERNYMS:
                target_hypernyms = CATEGORY_HYPERNYMS[category]
                for target in target_hypernyms:
                    if target in hypernyms:
                        return category, synset.name()
    
    return 'OTHER', synsets[0].name() if synsets else None


def main():
    import pandas as pd
    
    # Load words
    print("Loading words...")
    df = pd.read_csv(WORDS_FILE)
    all_words = df['word'].str.strip().str.lower().tolist()
    
    # Also load McRae words
    with open(MCRAE_FILE, 'r') as f:
        mcrae_words = list(json.load(f).keys())
    
    # Combine and deduplicate
    all_words = list(set(all_words + mcrae_words))
    print(f"Total unique words: {len(all_words)}")
    
    # Categorize each word
    word_categories = {}
    category_counts = defaultdict(list)
    synset_info = {}
    
    print("\nCategorizing words with WordNet...")
    for word in sorted(all_words):
        category, synset = categorize_word(word)
        word_categories[word] = category
        category_counts[category].append(word)
        if synset:
            synset_info[word] = synset
    
    # Save word -> category mapping
    output_file = OUTPUT_DIR / "word_categories.json"
    with open(output_file, 'w') as f:
        json.dump(word_categories, f, indent=2, sort_keys=True)
    print(f"\nSaved word categories to: {output_file}")
    
    # Save category -> words mapping
    category_words = {cat: sorted(words) for cat, words in category_counts.items()}
    output_file = OUTPUT_DIR / "category_words.json"
    with open(output_file, 'w') as f:
        json.dump(category_words, f, indent=2, sort_keys=True)
    print(f"Saved category words to: {output_file}")
    
    # Save synset info for debugging
    output_file = OUTPUT_DIR / "word_synsets.json"
    with open(output_file, 'w') as f:
        json.dump(synset_info, f, indent=2, sort_keys=True)
    print(f"Saved synset info to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CATEGORY DISTRIBUTION (WordNet)")
    print("=" * 60)
    
    sorted_cats = sorted(category_counts.items(), key=lambda x: len(x[1]), reverse=True)
    for cat, words in sorted_cats:
        print(f"\n{cat} ({len(words)} words):")
        print(f"  Examples: {', '.join(words[:8])}")
        if len(words) > 8:
            print(f"  ... and {len(words) - 8} more")
    
    print(f"\n\nTotal words categorized: {len(word_categories)}")
    print(f"Total categories: {len(category_counts)}")
    
    # Report uncategorized
    other_count = len(category_counts.get('OTHER', []))
    categorized = len(all_words) - other_count
    print(f"\nCategorized: {categorized} ({100*categorized/len(all_words):.1f}%)")
    print(f"Uncategorized (OTHER): {other_count} ({100*other_count/len(all_words):.1f}%)")


if __name__ == "__main__":
    main()
