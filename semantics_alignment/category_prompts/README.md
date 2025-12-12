# Category-Specific Semantic Feature Generation

## Overview

This module generates semantic features for words using GPT-5.1, optimized through category-specific prompt engineering. The goal is to produce features that align with human-generated McRae feature norms for use in the predictive coding model.

---

## Methodology

### Phase 1: Word Categorization (WordNet)

We categorized all 829 words in the lexicon using WordNet hypernym hierarchies.

**Script:** `extract_categories_wordnet.py`

**Process:**
1. For each word, retrieve the most common WordNet synset
2. Traverse hypernym tree to find matching category
3. Apply manual overrides for Word Sense Disambiguation errors

**Results:**
- 23 semantic categories identified
- 95.1% of words successfully categorized
- 41 words remain in "OTHER" category

**Categories:**
| Category | Count | Examples |
|----------|-------|----------|
| ANIMAL | 23 | tiger, lion, rabbit |
| BIRD | 6 | eagle, robin, sparrow |
| INSECT | 5 | beetle, butterfly, ant |
| FRUIT | 6 | apple, orange, lemon |
| VEGETABLE | 4 | carrot, cabbage, radish |
| FOOD | 36 | bread, cheese, chicken |
| TOOL | 17 | hammer, chisel, shovel |
| WEAPON | 9 | sword, rifle, dagger |
| VEHICLE | 10 | car, airplane, bicycle |
| CONTAINER | 16 | bucket, barrel, bottle |
| FURNITURE | 5 | sofa, table, dresser |
| BUILDING | 33 | church, barn, hotel |
| ROOM | 3 | bedroom, kitchen, bathroom |
| CLOTHING | 15 | jacket, pants, shirt |
| APPLIANCE | 5 | refrigerator, stove, oven |
| MUSICAL_INSTRUMENT | 7 | piano, guitar, trumpet |
| PLANT | 20 | oak, rose, daisy |
| BODY_PART | 20 | arm, leg, heart |
| MATERIAL | 1 | leather |
| PERSON | 2 | doctor, teacher |
| COMMUNICATION | 2 | letter, message |
| EVENT | 2 | party, concert |
| STATE | 1 | freedom |
| OTHER | 41 | Various uncategorized |

---

### Phase 2: Category-Specific Prompt Development

Each category uses McRae examples from the same semantic domain.

**Script:** `category_batch_generator.py`

**Prompt Structure:**
```
Persona: You are a computational research scientist...

Generate semantic attributes for a given word for the McRae et al. (2005) 
feature production task:
1. List specific properties of the concept
2. Include physical, functional, categorical, and behavioral properties
3. All words considered as nouns only
4. Use underscores between words
5. Respond with JSON: {"features": ["property1", "property2", ...]}

Examples (5 from same category):
Word: [example1]
Participant: {"features": [...]}
...

Word: [target_word]
Participant:
```

**Configuration:**
- Model: GPT-5.1
- Temperature: 0.7
- Features per word: 25

---

### Phase 3: Validation Against McRae Norms

**Script:** `validate_by_category.py`

**Method:**
1. Generate features for 99 McRae words using category-specific prompts
2. Compare against human-generated McRae features
3. Use SentenceTransformer embeddings for semantic similarity
4. Calculate Precision, Recall, and F1 per word

**Embedding Model:** `all-MiniLM-L6-v2`

**Category-Specific Results:**
| Category | N | Mean F1 | Status |
|----------|---|---------|--------|
| FRUIT | 5 | 0.8081 | OK |
| BIRD | 4 | 0.7975 | OK |
| ANIMAL | 11 | 0.7871 | OK |
| VEGETABLE | 4 | 0.7851 | OK |
| WEAPON | 6 | 0.7675 | OK |
| FOOD | 4 | 0.7551 | OK |
| MUSICAL_INSTRUMENT | 4 | 0.7405 | OK |
| BUILDING | 5 | 0.7167 | OK |
| FURNITURE | 5 | 0.7150 | OK |
| CLOTHING | 7 | 0.7021 | OK |
| INSECT | 1 | 0.6997 | Below threshold |
| APPLIANCE | 5 | 0.6969 | Below threshold |
| ROOM | 3 | 0.6964 | Below threshold |
| TOOL | 11 | 0.6848 | Below threshold |
| CONTAINER | 5 | 0.6757 | Below threshold |
| STATE | 1 | 0.6623 | Below threshold |
| EVENT | 2 | 0.6602 | Below threshold |
| PLANT | 5 | 0.6510 | Below threshold |
| COMMUNICATION | 2 | 0.6509 | Below threshold |
| VEHICLE | 4 | 0.6503 | Below threshold |
| PERSON | 2 | 0.6490 | Below threshold |
| MATERIAL | 1 | 0.6480 | Below threshold |
| OTHER | 1 | 0.6398 | Below threshold |

**Overall:** Mean F1 = 0.7136

---

### Phase 4: General Prompt A/B Testing

For categories below 0.7 F1 threshold, we tested a general prompt with diverse examples (tiger, hammer, sofa, apple, bottle).

**Script:** `strategy_comparison.py`

**Comparison Results:**
| Category | Cat-Specific | General | Winner |
|----------|--------------|---------|--------|
| APPLIANCE | 0.6969 | 0.7041 | **GENERAL** |
| COMMUNICATION | 0.6509 | 0.5880 | CATEGORY |
| CONTAINER | 0.6757 | 0.5642 | CATEGORY |
| EVENT | 0.6602 | 0.6675 | **GENERAL** |
| INSECT | 0.6997 | 0.7325 | **GENERAL** |
| MATERIAL | 0.6480 | 0.5142 | CATEGORY |
| OTHER | 0.6398 | 0.5893 | CATEGORY |
| PERSON | 0.6490 | 0.6034 | CATEGORY |
| PLANT | 0.6510 | 0.6421 | CATEGORY |
| ROOM | 0.6964 | 0.6871 | CATEGORY |
| STATE | 0.6623 | 0.6476 | CATEGORY |
| TOOL | 0.6848 | 0.6949 | **GENERAL** |
| VEHICLE | 0.6503 | 0.5919 | CATEGORY |

---

### Phase 5: Final Optimal Strategy

**Output:** `category_strategy.json`

**Decision:**
- **19 categories** use category-specific prompts
- **4 categories** use general prompt: APPLIANCE, EVENT, INSECT, TOOL

**Final Results:**
| Metric | Value |
|--------|-------|
| Words validated | 99 |
| Mean F1 | **0.7159** |
| Mean Precision | 0.7197 |
| Mean Recall | 0.7022 |
| Improvement over category-only | +0.0023 |

**Top Performing Words:**
| Word | Category | F1 |
|------|----------|-----|
| robin | BIRD | 0.8707 |
| rifle | WEAPON | 0.8610 |
| lemon | FRUIT | 0.8603 |
| turtle | ANIMAL | 0.8375 |
| hammer | APPLIANCE | 0.8319 |

**Bottom Performing Words:**
| Word | Category | F1 |
|------|----------|-----|
| pepper | PLANT | 0.4942 |
| basket | CONTAINER | 0.5184 |
| razor | TOOL | 0.5130 |
| bucket | CONTAINER | 0.5608 |
| buckle | CLOTHING | 0.5675 |

---

## File Structure

```
category_prompts/
├── README.md                      # This file
├── extract_categories_wordnet.py  # WordNet categorization
├── category_batch_generator.py    # Feature generation (category-specific)
├── validate_by_category.py        # Validation against McRae
├── validate_general.py            # Validation for general prompt
├── strategy_comparison.py         # A/B test comparison
├── generate_all_words.py          # Final 800-word generator
├── word_categories.json           # Word -> category mapping
├── category_strategy.json         # Optimal strategy per category
└── output/
    ├── category_features_raw.json      # Generated features (category)
    ├── category_validation_results.csv # Per-word validation (category)
    ├── general_features_raw.json       # Generated features (general)
    ├── general_validation_results.csv  # Per-word validation (general)
    ├── final_validation_results.csv    # Combined optimal results
    ├── final_summary.csv               # Per-category final summary
    └── comprehensive_report.txt        # Full report
```

---

## Usage

### Generate Features for All 800 Words

```bash
cd semantics_alignment/category_prompts
python generate_all_words.py --all
```

This script:
1. Loads the optimal strategy from `category_strategy.json`
2. For each word, selects the appropriate prompt type (category-specific or general)
3. Generates 25 semantic features per word
4. Saves results to `output/all_words_features.json`

### Convert to Model Input Format

After generation, convert to the format expected by the predictive coding model:

```bash
python convert_to_model_input.py  # (to be created)
```

---

## Next Steps

1. **Generate features for all 800 words** using optimal strategy
2. **Convert output** to model input format (`_model_input.json`)
3. **Run predictive coding simulation** with new semantic features
4. **Analyze N400 predictions** for clear vs. noisy speech conditions

---

## Notes

- All example words are dynamically excluded when generating features for that word
- Validation uses embedding-based semantic similarity, not exact string matching
- The 0.7 F1 threshold was chosen as a reasonable balance between quality and coverage
- Some individual words may underperform even in well-performing categories (e.g., pepper in PLANT)

---

## References

- McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. Behavior Research Methods, 37(4), 547-559.


