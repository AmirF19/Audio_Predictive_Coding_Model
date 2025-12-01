import numpy as np
import pandas as pd
import sys

# --- CONFIGURATION ---
CONSOLIDATED_FILE = "lex_lookup25.csv"
CONC_COL = 'brys_concreteness'  # Column name for the concreteness rating
HIGH_RICHNESS_COUNT = 18        # Feature count for words above the median
LOW_RICHNESS_COUNT = 9          # Feature count for words below the median

def load_data(filepath, index_col='word'):
    """
    Loads the consolidated lexicon CSV, ensuring the necessary columns are present 
    and cleaning the concreteness data.
    """
    print(f"Loading data from: {filepath}")
    try:
        df = pd.read_csv(filepath, index_col=index_col, low_memory=False)
        
        if df.empty:
            raise ValueError("Lexicon file is empty.")
            
        if CONC_COL not in df.columns:
            print(f"FATAL ERROR: Missing required column '{CONC_COL}'. Check your CSV header.")
            sys.exit(1)
            
        # Clean numeric data: convert column to numeric and drop rows where 
        # concreteness rating is missing (NaN).
        df[CONC_COL] = pd.to_numeric(df[CONC_COL], errors='coerce')
        df.dropna(subset=[CONC_COL], inplace=True)
        
        if df.empty:
            raise ValueError("All rows were dropped after cleaning the concreteness column.")
            
        return df

    except FileNotFoundError:
        print(f"FATAL ERROR: Consolidated file not found: {filepath}")
        print("Please ensure 'lex_lookup25.csv' is in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR reading {filepath}: {e}")
        sys.exit(1)

def process_concreteness_split():
    """
    Performs the median split on concreteness ratings to determine semantic richness 
    (target feature count of 9 or 18).
    """
    # 1. Load Data
    lexicon_df = load_data(CONSOLIDATED_FILE)
    
    print("\n--- Running Concreteness Processor and Richness Split ---")
    
    # 2. Calculate Median
    concreteness_ratings = lexicon_df[CONC_COL].values
    median_concreteness = np.median(concreteness_ratings)
    print(f"Total words analyzed: {len(lexicon_df)}")
    print(f"Calculated Median Concreteness: {median_concreteness:.2f}")

    # 3. Apply Median Split Logic
    # High Concreteness (>= median) is assigned High Richness (18 features)
    # Low Concreteness (< median) is assigned Low Richness (9 features)
    lexicon_df['target_feature_count'] = np.where(
        lexicon_df[CONC_COL] >= median_concreteness, 
        HIGH_RICHNESS_COUNT, 
        LOW_RICHNESS_COUNT
    )
    
    # 4. Verification and Output
    high_rich_count = len(lexicon_df[lexicon_df['target_feature_count'] == HIGH_RICHNESS_COUNT])
    low_rich_count = len(lexicon_df[lexicon_df['target_feature_count'] == LOW_RICHNESS_COUNT])
    
    print("\nRichness Split Results:")
    print(f"  Words >= Median Concreteness (Target {HIGH_RICHNESS_COUNT} features): {high_rich_count}")
    print(f"  Words < Median Concreteness (Target {LOW_RICHNESS_COUNT} features): {low_rich_count}")
    
    # Show example output
    print("\nExample Concreteness and Target Feature Count:")
    print(lexicon_df[[CONC_COL, 'target_feature_count']].head(10))

if __name__ == '__main__':
    # To run this script: ensure you have 'lex_lookup25.csv' populated 
    # with the 'brys_concreteness' column, and run: 
    # python concreteness_processor.py
    process_concreteness_split()