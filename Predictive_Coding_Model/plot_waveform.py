import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from config import MAX_STEPS, PRECISION_CLEAR, PRECISION_NOISY
from data_loader import load_phoneme_input, load_lexicon, load_semantic_matrix
from pc_model import PCModel
from simulation import construct_pronunciation_matrix

def plot_single_word_trace(target_word="abuse"):
    """
    Runs one word and plots its N400 trajectory (Error over Time).
    """
    # 1. Setup
    lexicon = load_lexicon()
    if target_word not in lexicon:
        print(f"Word '{target_word}' not found in lexicon.")
        return

    V_SL, _ = load_semantic_matrix(lexicon)
    V_LP = construct_pronunciation_matrix(lexicon)
    model = PCModel(V_LP=V_LP, V_SL=V_SL)

    # 2. Run Clear Condition
    model.reset_state()
    input_clear = load_phoneme_input(target_word, 'clear')
    trace_clear = []
    for _ in range(MAX_STEPS):
        model.step(input_clear, PRECISION_CLEAR)
        trace_clear.append(model.get_n400_total())

    # 3. Run Noisy Condition
    model.reset_state()
    input_noisy = load_phoneme_input(target_word, 'noisy')
    trace_noisy = []
    for _ in range(MAX_STEPS):
        model.step(input_noisy, PRECISION_NOISY)
        trace_noisy.append(model.get_n400_total())

    # 4. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(trace_clear, label='Clear Input (High Precision)', color='blue', linewidth=2)
    plt.plot(trace_noisy, label='Noisy Input (Low Precision)', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"Simulated N400 Waveform: '{target_word}'")
    plt.xlabel("Time Steps (Iterations)")
    plt.ylabel("N400 Amplitude (Prediction Error)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = f"n400_trace_{target_word}.png"
    plt.savefig(save_path)
    print(f"Waveform plot saved to {save_path}")

if __name__ == "__main__":
    # You can change this to any word in your CSV
    plot_single_word_trace("abuse")