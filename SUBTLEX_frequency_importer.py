import zipfile
import pandas as pd
import os
import sys

# --- CONFIGURATION ---
# The exact path to the zip file you provided
ZIP_FILE_PATH = r"C:\Users\Muhammad\OneDrive\Desktop\comp_ling_project\SUBTLEXusfrequencyabove1.zip"
# Assumed name of the Excel file inside the zip (must match exactly)
INTERNAL_FILE_NAME = "SUBTLEXusfrequencyabove1.xlsx" 
OUTPUT_CSV = "subtlex_filtered_frequency.csv"

# --- USER INPUT CONFIGURATION ---
USER_WORDS_FILE = "my_800_words.csv" 
USER_INPUT_WORD_COL = 'Word' # The column name in the user's CSV

# --- SUBTLEX DB CONFIGURATION ---
DB_WORD_COL = 'Word' # The column name in the SUBTLEX DB
FREQ_COL = 'Lg10WF'

def load_user_words(filepath):
    """Loads the list of words the user is interested in from a CSV, assuming it might be tab-separated."""
    try:
        # Load the CSV, explicitly trying tab-separation due to the user's reported header format.
        df = pd.read_csv(filepath, sep='\t')
        
        if USER_INPUT_WORD_COL not in df.columns:
            # If tab-separated read failed, try reading as standard comma-separated CSV
            try:
                df = pd.read_csv(filepath)
                if USER_INPUT_WORD_COL not in df.columns:
                    print(f"FATAL ERROR: The CSV must contain a column named '{USER_INPUT_WORD_COL}'.")
                    print(f"Columns found: {df.columns.tolist()}")
                    sys.exit(1)
            except Exception as e:
                print(f"FATAL ERROR: Could not read CSV: {e}")
                sys.exit(1)

        # Normalize to lowercase and convert to a list
        words = df[USER_INPUT_WORD_COL].astype(str).str.strip().str.lower().tolist()
        
        if not words:
            print(f"Error: {filepath} is empty after filtering.")
            sys.exit(1)
            
        print(f"Loaded {len(words)} target words from {filepath}.")
        return words
    except FileNotFoundError:
        print(f"FATAL ERROR: User word list not found at '{filepath}'.")
        print("Please create a file named 'my_800_words.csv' with your words in a column titled 'Word'.")
        sys.exit(1)


def process_subtlex_zip():
    """Extracts, filters, and saves the required Lg10WF data."""
    
    # 1. Load the user's target word list
    target_words = load_user_words(USER_WORDS_FILE)

    print(f"\nProcessing SUBTLEX zip file at: {ZIP_FILE_PATH}")
    
    try:
        # 2. Open the zip archive
        with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as z:
            
            # Check if the expected file name exists in the archive
            if INTERNAL_FILE_NAME not in z.namelist():
                print(f"FATAL ERROR: '{INTERNAL_FILE_NAME}' not found inside the zip.")
                print("Available files in zip: ", z.namelist())
                print("Please check the correct Excel/TXT file name and update INTERNAL_FILE_NAME.")
                sys.exit(1)
            
            # 3. Read the file into a pandas DataFrame
            print(f"Reading internal file: {INTERNAL_FILE_NAME}...")
            
            # Use 'z.open' for in-memory reading, and detect file type
            with z.open(INTERNAL_FILE_NAME) as f:
                # Use engine='openpyxl' for robust Excel reading
                subtlex_df = pd.read_excel(f, engine='openpyxl')
                
    except FileNotFoundError:
        print(f"FATAL ERROR: Zip file not found at the specified path: {ZIP_FILE_PATH}")
        sys.exit(1)
    except zipfile.BadZipFile:
        print(f"FATAL ERROR: The file at {ZIP_FILE_PATH} is not a valid zip file.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during file processing: {e}")
        sys.exit(1)

    # 4. Data Cleaning and Filtering
    
    # Ensure the word column is in lowercase for matching
    subtlex_df[DB_WORD_COL] = subtlex_df[DB_WORD_COL].astype(str).str.lower()
    
    # Filter for the user's target words
    filtered_df = subtlex_df[subtlex_df[DB_WORD_COL].isin(target_words)]
    
    # Select only the required columns: word and Lg10WF
    result_df = filtered_df[[DB_WORD_COL, FREQ_COL]].copy()
    
    # Rename the frequency column for consistency with the main project naming convention
    result_df.rename(columns={FREQ_COL: 'brys_lg10freq'}, inplace=True)
    
    # 5. Save the result
    result_df.to_csv(OUTPUT_CSV, index=False)
    
    words_found = len(result_df)
    words_missing = len(target_words) - words_found
    
    print(f"\nSuccessfully found frequency data for {words_found} words. ({words_missing} words not found in SUBTLEX-US).")
    print(f"Filtered frequency data saved to {OUTPUT_CSV}")
    print("--- SUBTLEX Processing Complete ---")
    
if __name__ == '__main__':
    process_subtlex_zip()