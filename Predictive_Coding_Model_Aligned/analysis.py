"""
Analysis and Visualization Module for Auditory N400 Simulation

This module handles all post-simulation analysis, including:
- Statistical summaries (N400 by condition, recognition accuracy)
- Visualization (timecourse plots, bar charts, accuracy plots)
- Results export (CSV, PNG)

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def print_summary(df):
    """
    Print summary statistics for N400 and recognition accuracy.
    
    Displays:
    - N400 mean, peak, and peak timing for each condition×clarity combination
    - Recognition accuracy by clarity (clean vs noisy)
    
    Args:
        df: DataFrame with simulation results (one row per trial)
    """
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    # N400 effects by condition and clarity
    print("\nN400 by Condition:")
    for condition in df['condition'].unique():
        for clarity in df['clarity'].unique():
            subset = df[(df['condition'] == condition) & (df['clarity'] == clarity)]
            if len(subset) > 0:
                print(f"  {condition}/{clarity}: "
                      f"mean={subset['n400_mean'].mean():.2f}, "
                      f"peak={subset['n400_peak'].mean():.2f}, "
                      f"peak_iter={subset['n400_peak_iter'].mean():.1f}")

    # Recognition accuracy by clarity
    print("\nRecognition Accuracy:")
    for clarity in df['clarity'].unique():
        subset = df[df['clarity'] == clarity]
        if len(subset) > 0:
            acc = subset['target_correct'].mean() * 100
            print(f"  {clarity}: {acc:.1f}%")


def plot_results(df, output_dir, num_iters=20, blanks_before=5, target_iters=20):
    """
    Generate separate visualization files for N400 results.
    
    Creates 3 separate figures:
    1. N400 timecourse showing all 4 conditions
    2. N400 mean bar plot with 2×2 grouped bars
    3. Recognition accuracy by clarity
    
    Args:
        df: DataFrame with simulation results
        output_dir: Path object for saving the figures
        num_iters: Number of prime iterations
        blanks_before: Number of blank iterations before target
        target_iters: Number of target iterations
    """

    # ========== FIGURE 1: N400 TIMECOURSE - ALL 4 CONDITIONS ==========
    fig1, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Define time windows for plotting
    target_start = num_iters + blanks_before
    target_end = target_start + target_iters
    plot_start = target_start - blanks_before  # Include pre-target baseline
    plot_end = target_end

    # Color/style scheme: blue=same, red=different; solid=clear, dashed=noisy
    condition_styles = {
        ('same', 'clear'): {'color': 'blue', 'linestyle': '-', 'label': 'Same/Clear'},
        ('same', 'noisy'): {'color': 'blue', 'linestyle': '--', 'label': 'Same/Noisy'},
        ('different', 'clear'): {'color': 'red', 'linestyle': '-', 'label': 'Different/Clear'},
        ('different', 'noisy'): {'color': 'red', 'linestyle': '--', 'label': 'Different/Noisy'}
    }

    max_peak = 0
    for (cond, clarity), style in condition_styles.items():
        subset = df[(df['condition'] == cond) & (df['clarity'] == clarity)]
        if len(subset) > 0:
            # Extract and average traces across trials
            # Handle both list format (from simulation) and string format (from CSV)
            first_trace = subset['trace_lexsem_err'].iloc[0]
            if isinstance(first_trace, str):
                # From CSV: comma-separated strings
                traces = np.array([list(map(float, t.split(','))) for t in subset['trace_lexsem_err']])
            else:
                # From simulation: already lists
                traces = np.array([t if isinstance(t, list) else list(t) for t in subset['trace_lexsem_err']])
            
            seg = traces[:, plot_start:plot_end]
            mean_trace = seg.mean(axis=0)
            
            # X-axis: iterations relative to target onset (0 = target starts)
            x_vals = np.arange(-blanks_before, seg.shape[1] - blanks_before)
            ax.plot(x_vals, mean_trace, color=style['color'], 
                   linestyle=style['linestyle'], linewidth=2, label=style['label'])
            max_peak = max(max_peak, mean_trace.max())

    # Mark target onset
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('Iterations from Target Onset', fontsize=12)
    ax.set_ylabel('Lexico-Semantic PE (N400)', fontsize=12)
    ax.set_title('N400 Timecourse: All Conditions', fontsize=13, fontweight='bold')
    ax.set_xlim(-blanks_before, target_iters)
    ax.set_ylim(0, max_peak * 1.1 if max_peak > 0 else 1)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    # Save timecourse figure (all 4 conditions)
    plt.tight_layout()
    timecourse_path = output_dir / "n400_timecourse_all.png"
    plt.savefig(timecourse_path, dpi=150)
    print(f"Timecourse (all conditions) saved to: {timecourse_path}")
    plt.close()
    
    # ========== FIGURE 1B: N400 TIMECOURSE - SAME CONDITION ONLY ==========
    fig1b, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    max_peak_same = 0
    for clarity in ['clear', 'noisy']:
        subset = df[(df['condition'] == 'same') & (df['clarity'] == clarity)]
        if len(subset) > 0:
            # Extract traces
            first_trace = subset['trace_lexsem_err'].iloc[0]
            if isinstance(first_trace, str):
                traces = np.array([list(map(float, t.split(','))) for t in subset['trace_lexsem_err']])
            else:
                traces = np.array([t if isinstance(t, list) else list(t) for t in subset['trace_lexsem_err']])
            
            seg = traces[:, plot_start:plot_end]
            mean_trace = seg.mean(axis=0)
            x_vals = np.arange(-blanks_before, seg.shape[1] - blanks_before)
            
            # Blue for same, solid=clear, dashed=noisy
            linestyle = '-' if clarity == 'clear' else '--'
            label = f'Same/{clarity.capitalize()}'
            ax.plot(x_vals, mean_trace, color='blue', linestyle=linestyle, 
                   linewidth=2.5, label=label)
            max_peak_same = max(max_peak_same, mean_trace.max())
    
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('Iterations from Target Onset', fontsize=12)
    ax.set_ylabel('Lexico-Semantic PE (N400)', fontsize=12)
    ax.set_title('N400 Timecourse: Same Condition', fontsize=13, fontweight='bold')
    ax.set_xlim(-blanks_before, target_iters)
    ax.set_ylim(0, max_peak_same * 1.1 if max_peak_same > 0 else 1)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    same_path = output_dir / "n400_timecourse_same.png"
    plt.savefig(same_path, dpi=150)
    print(f"Timecourse (same condition) saved to: {same_path}")
    plt.close()
    
    # ========== FIGURE 1C: N400 TIMECOURSE - DIFFERENT CONDITION ONLY ==========
    fig1c, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    max_peak_diff = 0
    for clarity in ['clear', 'noisy']:
        subset = df[(df['condition'] == 'different') & (df['clarity'] == clarity)]
        if len(subset) > 0:
            # Extract traces
            first_trace = subset['trace_lexsem_err'].iloc[0]
            if isinstance(first_trace, str):
                traces = np.array([list(map(float, t.split(','))) for t in subset['trace_lexsem_err']])
            else:
                traces = np.array([t if isinstance(t, list) else list(t) for t in subset['trace_lexsem_err']])
            
            seg = traces[:, plot_start:plot_end]
            mean_trace = seg.mean(axis=0)
            x_vals = np.arange(-blanks_before, seg.shape[1] - blanks_before)
            
            # Red for different, solid=clear, dashed=noisy
            linestyle = '-' if clarity == 'clear' else '--'
            label = f'Different/{clarity.capitalize()}'
            ax.plot(x_vals, mean_trace, color='red', linestyle=linestyle,
                   linewidth=2.5, label=label)
            max_peak_diff = max(max_peak_diff, mean_trace.max())
    
    ax.axvline(0, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    ax.set_xlabel('Iterations from Target Onset', fontsize=12)
    ax.set_ylabel('Lexico-Semantic PE (N400)', fontsize=12)
    ax.set_title('N400 Timecourse: Different Condition', fontsize=13, fontweight='bold')
    ax.set_xlim(-blanks_before, target_iters)
    ax.set_ylim(0, max_peak_diff * 1.1 if max_peak_diff > 0 else 1)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    diff_path = output_dir / "n400_timecourse_different.png"
    plt.savefig(diff_path, dpi=150)
    print(f"Timecourse (different condition) saved to: {diff_path}")
    plt.close()

    # ========== FIGURE 2: N400 MEAN - 2×2 GROUPED BARS ==========
    fig2, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Compute means and SDs for all 4 conditions
    cond_data = {}
    for cond in df['condition'].unique():
        for clarity in df['clarity'].unique():
            subset = df[(df['condition'] == cond) & (df['clarity'] == clarity)]
            if len(subset) > 0:
                cond_data[(cond, clarity)] = {
                    'mean': subset['n400_mean'].mean(),
                    'std': subset['n400_mean'].std()
                }

    # Bar positions and width
    x = np.arange(2)  # same, different
    width = 0.35
    
    # Extract data for plotting
    conditions_order = ['same', 'different']
    if 'clear' in df['clarity'].unique() and 'noisy' in df['clarity'].unique():
        clear_means = [cond_data.get((c, 'clear'), {'mean': 0})['mean'] for c in conditions_order]
        clear_stds = [cond_data.get((c, 'clear'), {'std': 0})['std'] for c in conditions_order]
        noisy_means = [cond_data.get((c, 'noisy'), {'mean': 0})['mean'] for c in conditions_order]
        noisy_stds = [cond_data.get((c, 'noisy'), {'std': 0})['std'] for c in conditions_order]
        
        ax.bar(x - width/2, clear_means, width, yerr=clear_stds, capsize=5, 
               label='Clear', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, noisy_means, width, yerr=noisy_stds, capsize=5,
               label='Noisy', color='coral', alpha=0.8)
    
    ax.set_xlabel('Prime-Target Condition', fontsize=12)
    ax.set_ylabel('Mean N400 (PE)', fontsize=12)
    ax.set_title('N400 by Condition × Clarity', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Same', 'Different'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Save bar plot figure
    plt.tight_layout()
    bars_path = output_dir / "n400_bars.png"
    plt.savefig(bars_path, dpi=150)
    print(f"Bar plot saved to: {bars_path}")
    plt.close()

    # ========== FIGURE 3: RECOGNITION ACCURACY ==========
    fig3, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Compute accuracy by clarity
    clarity_acc = {}
    for clarity in df['clarity'].unique():
        subset = df[df['clarity'] == clarity]
        if len(subset) > 0:
            clarity_acc[clarity] = subset['target_correct'].mean() * 100
    
    clarities = list(clarity_acc.keys())
    accuracies = list(clarity_acc.values())
    colors_acc = ['steelblue' if c == 'clear' else 'coral' for c in clarities]
    
    bars = ax.bar(clarities, accuracies, color=colors_acc, alpha=0.8, width=0.5)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=12)
    ax.set_title('Word Recognition by Clarity', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Save accuracy figure
    plt.tight_layout()
    accuracy_path = output_dir / "recognition_accuracy.png"
    plt.savefig(accuracy_path, dpi=150)
    print(f"Accuracy plot saved to: {accuracy_path}")
    plt.close()
    
    print(f"\n========================================")
    print(f"All 5 plots saved to: {output_dir}")
    print(f"========================================")


def save_results(df, output_dir):
    """
    Save simulation results to CSV.
    
    Converts list-valued trace columns to comma-separated strings for CSV compatibility.
    
    Args:
        df: DataFrame with simulation results
        output_dir: Path object for saving the CSV
    """
    # Convert trace lists to comma-separated strings
    df_export = df.copy()
    trace_columns = ['trace_lexsem_err', 'trace_lex_err', 'trace_sem_err', 'trace_max_activation']
    for col in trace_columns:
        if col in df_export.columns:
            df_export[col] = df_export[col].apply(lambda x: ','.join(map(str, x)))
    
    output_path = output_dir / "simulation_results.csv"
    df_export.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

