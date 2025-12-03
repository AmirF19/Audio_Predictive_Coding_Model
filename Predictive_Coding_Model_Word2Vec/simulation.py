"""
N400 Priming Simulation (Word2Vec version)

2x2 design: Semantic Similarity (Same/Similar/Dissimilar) x Clarity (Clear/Noisy)
N400 operationalized as semantic prediction error between prime and target.

This version uses Word2Vec embeddings for semantic similarity instead of
Llama-generated binary features.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from config import (
    INPUT_DIM,
    MAX_STEPS,
    PRIME_SETTLE_STEPS,
    PRECISION_CLEAR,
    PRECISION_NOISY,
    RESULTS_CSV_FILE,
    RESULTS_PLOT_FILE,
    DEVICE,
)
from data_loader import (
    load_lexicon,
    load_semantic_matrix,
    load_phoneme_input,
)
from pc_model import PCModel


def construct_V_LP(lexicon):
    """Build phoneme-to-lexical weight matrix from clear audio vectors."""
    V_LP = torch.zeros(INPUT_DIM, len(lexicon), dtype=torch.float32, device=DEVICE)
    
    count = 0
    for i, word in enumerate(tqdm(lexicon, desc="Building V_LP")):
        vec = load_phoneme_input(word, 'clear')
        if vec.abs().sum() > 0:
            V_LP[:, i] = vec
            count += 1
            
    print(f"Loaded {count}/{len(lexicon)} phoneme vectors")
    return V_LP


def compute_similarity_matrix(V_SL):
    """Pairwise semantic similarity between all words using Word2Vec embeddings."""
    norms = torch.norm(V_SL, dim=0, keepdim=True) + 1e-10
    V_norm = V_SL / norms
    return V_norm.T @ V_norm


def get_available_words(lexicon):
    """Words that have audio data."""
    available = []
    for word in lexicon:
        vec = load_phoneme_input(word, 'clear')
        if vec.abs().sum() > 0:
            available.append(word)
    return available


def select_pairs(lexicon, sim_matrix):
    """
    Select prime-target pairs for each semantic condition.
    Word2Vec typically gives more graded similarity, so thresholds adjusted.
    """
    available = get_available_words(lexicon)
    word_to_idx = {w: i for i, w in enumerate(lexicon)}
    
    pairs = []
    used = set()
    
    # Same pairs (repetition)
    for word in available:
        pairs.append({
            'prime': word,
            'target': word,
            'condition': 'same',
            'similarity': 1.0
        })
    
    # Similar pairs (0.4 < sim < 0.8 for Word2Vec - more graded)
    for prime in available:
        if prime in used:
            continue
        prime_idx = word_to_idx[prime]
        sims = sim_matrix[prime_idx].cpu().numpy()
        
        candidates = np.where((sims > 0.4) & (sims < 0.8))[0]
        for target_idx in candidates:
            target = lexicon[target_idx]
            if target in available and target not in used and target != prime:
                pairs.append({
                    'prime': prime,
                    'target': target,
                    'condition': 'similar',
                    'similarity': float(sims[target_idx])
                })
                used.add(target)
                break
    
    # Dissimilar pairs (sim < 0.3 for Word2Vec)
    used_dissim = set()
    for prime in available:
        if prime in used_dissim:
            continue
        prime_idx = word_to_idx[prime]
        sims = sim_matrix[prime_idx].cpu().numpy()
        
        candidates = np.where(sims < 0.3)[0]
        for target_idx in candidates:
            target = lexicon[target_idx]
            if target in available and target not in used_dissim and target != prime:
                pairs.append({
                    'prime': prime,
                    'target': target,
                    'condition': 'dissimilar',
                    'similarity': float(sims[target_idx])
                })
                used_dissim.add(target)
                break
    
    return pairs


def run_trial(model, prime, target, condition, lexicon):
    """Run single priming trial and return metrics."""
    word_to_idx = {w: i for i, w in enumerate(lexicon)}
    prime_idx = word_to_idx.get(prime, -1)
    target_idx = word_to_idx.get(target, -1)
    
    # Prime phase
    model.reset_state()
    input_prime = load_phoneme_input(prime, 'clear')
    if input_prime.abs().sum() == 0:
        return None
    
    for _ in range(PRIME_SETTLE_STEPS):
        model.step(input_prime, PRECISION_CLEAR)
    
    model.store_semantic_expectation()
    prime_winner, _ = model.get_winner_info()
    prime_predicted = lexicon[prime_winner] if 0 <= prime_winner < len(lexicon) else "UNKNOWN"
    
    # Target phase
    model.prepare_for_target()
    precision = PRECISION_CLEAR if condition == 'clear' else PRECISION_NOISY
    input_target = load_phoneme_input(target, condition)
    
    if input_target.abs().sum() == 0:
        return None
    
    n400_trace = []
    target_activation_trace = []
    winner_activation_trace = []
    for _ in range(MAX_STEPS):
        model.step(input_target, precision)
        n400_trace.append(model.get_semantic_prediction_error())
        # Track target word activation
        target_activation_trace.append(model.state_lexical[target_idx].item())
        # Track max activation (whatever word is winning)
        winner_activation_trace.append(model.state_lexical.max().item())
    
    metrics = model.get_metrics(target_idx)
    target_winner = metrics['winner_idx']
    target_predicted = lexicon[target_winner] if 0 <= target_winner < len(lexicon) else "UNKNOWN"
    
    # Max activations
    max_target_activation = max(target_activation_trace)
    max_winner_activation = max(winner_activation_trace)  # Intelligibility proxy (correct or not)
    
    return {
        'prime': prime,
        'target': target,
        'clarity': condition,
        'prime_correct': (prime_winner == prime_idx),
        'prime_predicted': prime_predicted,
        'target_correct': metrics['correct'],
        'target_predicted': target_predicted,
        'target_prob': metrics['target_prob'],
        'max_target_activation': max_target_activation,
        'max_winner_activation': max_winner_activation,
        'n400_mean': np.mean(n400_trace),
        'n400_peak': np.max(n400_trace),
        'n400_final': n400_trace[-1],
        'semantic_sim': metrics['semantic_similarity'],
        'n400_trace': n400_trace,
    }


def run_experiment():
    """Main experiment."""
    print("=" * 60)
    print("N400 Priming Simulation (Word2Vec)")
    print("=" * 60)
    
    lexicon = load_lexicon()
    if not lexicon:
        return
    
    V_SL, _ = load_semantic_matrix(lexicon)
    if V_SL.numel() == 0:
        print("Failed to load Word2Vec embeddings")
        return
        
    V_LP = construct_V_LP(lexicon)
    sim_matrix = compute_similarity_matrix(V_SL)
    
    pairs = select_pairs(lexicon, sim_matrix)
    print(f"\nTotal pairs: {len(pairs)}")
    print(f"  Same: {sum(1 for p in pairs if p['condition']=='same')}")
    print(f"  Similar: {sum(1 for p in pairs if p['condition']=='similar')}")
    print(f"  Dissimilar: {sum(1 for p in pairs if p['condition']=='dissimilar')}")
    
    model = PCModel(V_LP=V_LP, V_SL=V_SL)
    
    results = []
    for pair in tqdm(pairs, desc="Running trials"):
        for clarity in ['clear', 'noisy']:
            result = run_trial(model, pair['prime'], pair['target'], clarity, lexicon)
            if result:
                result['semantic_condition'] = pair['condition']
                result['true_similarity'] = pair['similarity']
                results.append(result)
    
    df = pd.DataFrame(results)
    df_save = df.drop(columns=['n400_trace'], errors='ignore')
    df_save.to_csv(RESULTS_CSV_FILE, index=False)
    
    print_summary(df)
    plot_results(df)


def print_summary(df):
    """Print results summary."""
    print("\n" + "=" * 60)
    print("RESULTS (Word2Vec)")
    print("=" * 60)
    
    print("\nN400 by Condition:")
    summary = df.groupby(['semantic_condition', 'clarity']).agg({
        'n400_mean': ['mean', 'std', 'count'],
        'target_correct': 'mean',
    }).round(4)
    print(summary)
    
    print("\nMain Effects:")
    print("\nSemantic Similarity:")
    for cond in ['same', 'similar', 'dissimilar']:
        subset = df[df['semantic_condition'] == cond]
        if len(subset) > 0:
            print(f"  {cond}: {subset['n400_mean'].mean():.4f}")
    
    print("\nClarity:")
    for cond in ['clear', 'noisy']:
        subset = df[df['clarity'] == cond]
        if len(subset) > 0:
            print(f"  {cond}: {subset['n400_mean'].mean():.4f}")
    
    print("\nInteraction (Semantic x Clarity):")
    for clarity in ['clear', 'noisy']:
        print(f"\n{clarity.upper()}:")
        for sem in ['same', 'similar', 'dissimilar']:
            subset = df[(df['semantic_condition'] == sem) & (df['clarity'] == clarity)]
            if len(subset) > 0:
                print(f"  {sem}: {subset['n400_mean'].mean():.4f}")
    
    print("\nRecognition Accuracy:")
    for cond in ['clear', 'noisy']:
        subset = df[df['clarity'] == cond]
        if len(subset) > 0:
            print(f"  {cond}: {subset['target_correct'].mean()*100:.1f}%")


def plot_results(df):
    """Generate result plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    conditions = ['same', 'similar', 'dissimilar']
    x = np.arange(len(conditions))
    width = 0.35
    
    # N400 bar plot
    ax = axes[0, 0]
    clear_means = [df[(df['semantic_condition']==c) & (df['clarity']=='clear')]['n400_mean'].mean() 
                   for c in conditions]
    noisy_means = [df[(df['semantic_condition']==c) & (df['clarity']=='noisy')]['n400_mean'].mean() 
                   for c in conditions]
    
    ax.bar(x - width/2, clear_means, width, label='Clear', color='#2ecc71')
    ax.bar(x + width/2, noisy_means, width, label='Noisy', color='#e74c3c')
    ax.set_ylabel('N400 (Semantic Prediction Error)')
    ax.set_title('N400 by Condition (Word2Vec)')
    ax.set_xticks(x)
    ax.set_xticklabels(['Same', 'Similar', 'Dissimilar'])
    ax.legend()
    
    # N400 timecourse
    ax = axes[0, 1]
    colors = {'same': '#3498db', 'dissimilar': '#e74c3c'}
    for sem in ['same', 'dissimilar']:
        for clarity, ls in [('clear', '-'), ('noisy', '--')]:
            subset = df[(df['semantic_condition']==sem) & (df['clarity']==clarity)]
            if len(subset) > 0 and 'n400_trace' in subset.columns:
                traces = np.array(subset['n400_trace'].tolist())
                mean_trace = traces.mean(axis=0)
                ax.plot(mean_trace, color=colors[sem], linestyle=ls, 
                       label=f'{sem}/{clarity}')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Semantic Prediction Error')
    ax.set_title('N400 Timecourse')
    ax.legend()
    
    # Recognition accuracy
    ax = axes[1, 0]
    clear_acc = [df[(df['semantic_condition']==c) & (df['clarity']=='clear')]['target_correct'].mean()*100 
                 for c in conditions]
    noisy_acc = [df[(df['semantic_condition']==c) & (df['clarity']=='noisy')]['target_correct'].mean()*100 
                 for c in conditions]
    
    ax.bar(x - width/2, clear_acc, width, label='Clear', color='#2ecc71')
    ax.bar(x + width/2, noisy_acc, width, label='Noisy', color='#e74c3c')
    ax.set_ylabel('Recognition Accuracy (%)')
    ax.set_title('Target Recognition')
    ax.set_xticks(x)
    ax.set_xticklabels(['Same', 'Similar', 'Dissimilar'])
    ax.legend()
    ax.set_ylim(0, 105)
    
    # N400 vs similarity scatter
    ax = axes[1, 1]
    for clarity, color in [('clear', '#2ecc71'), ('noisy', '#e74c3c')]:
        subset = df[df['clarity'] == clarity]
        if len(subset) > 0:
            ax.scatter(subset['true_similarity'], subset['n400_mean'],
                      alpha=0.4, label=clarity, color=color, s=20)
    ax.set_xlabel('Word2Vec Cosine Similarity')
    ax.set_ylabel('N400')
    ax.set_title('N400 vs Similarity')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(RESULTS_PLOT_FILE, dpi=150)
    print(f"\nPlot saved to {RESULTS_PLOT_FILE}")


if __name__ == "__main__":
    run_experiment()
