"""
Visualization for Quantum-Enhanced TDA

Clean, modular plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from matplotlib.figure import Figure

from config import ExperimentConfig
from experiments import ExperimentResults, TrialResult


# Color scheme for methods
METHOD_COLORS = {
    'classical': '#1f77b4',              # blue
    'swap_test': '#ff7f0e',              # orange
    'trace_distance_classical': '#2ca02c',  # green
    'trace_distance_quantum': '#d62728',    # red
}


def plot_single_trial(result: TrialResult, 
                      config: ExperimentConfig,
                      signal_data: np.ndarray = None) -> Figure:
    """
    Plot results from a single trial.
    
    Parameters
    ----------
    result : TrialResult
        Trial results
    config : ExperimentConfig
        Configuration
    signal_data : np.ndarray, optional
        Original signal for plotting
        
    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    method_name = config.method_display_names.get(result.method, result.method)
    color = METHOD_COLORS.get(result.method, 'blue')
    
    # 1. Distance matrix
    ax1 = axes[0, 0]
    im = ax1.imshow(result.distance_matrix, cmap='hot', aspect='auto')
    ax1.set_xlabel('Segment Index')
    ax1.set_ylabel('Segment Index')
    ax1.set_title(f'Distance Matrix ({method_name})')
    plt.colorbar(im, ax=ax1, label='Distance')
    
    # 2. Sequential distances
    ax2 = axes[0, 1]
    seq_dist = result.detection.sequential_distances
    times = result.segment_times[:-1]
    
    ax2.plot(times, seq_dist, color=color, linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=np.mean(seq_dist), color='gray', linestyle='--', alpha=0.5,
                label=f'Mean: {np.mean(seq_dist):.3f}')
    ax2.axhline(y=np.mean(seq_dist) + np.std(seq_dist), color='gray', 
                linestyle=':', alpha=0.5, label='+1σ threshold')
    ax2.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                linestyle='--', alpha=0.7, label='True transition')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Distance to Next Segment')
    ax2.set_title('Sequential Distances')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Cumulative drift
    ax3 = axes[1, 0]
    cum_drift = result.detection.cumulative_drift
    
    ax3.plot(result.segment_times, cum_drift, color=color, linewidth=2,
             marker='o', markersize=4)
    ax3.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                linestyle='--', alpha=0.7, label='True transition')
    
    # Mark detection metrics
    if result.detection.metrics_at_transition:
        m = result.detection.metrics_at_transition
        idx = m['closest_segment_idx']
        ax3.scatter([result.segment_times[idx]], [cum_drift[idx]], 
                    s=200, color='red', marker='*', zorder=5,
                    label=f"t={m['closest_segment_time']:.1f}s")
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Distance from Initial State')
    ax3.set_title('Cumulative Drift')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Mode counts
    ax4 = axes[1, 1]
    ax4.plot(result.segment_times, result.mode_counts, 'ko-', linewidth=2, markersize=6)
    ax4.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                linestyle='--', alpha=0.7)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Detected Modes')
    ax4.set_title('TDA Mode Count')
    ax4.set_ylim(bottom=0)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'{method_name} - Trial {result.trial_idx + 1}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_method_comparison(results: ExperimentResults) -> Figure:
    """
    Compare all methods across trials.
    
    Parameters
    ----------
    results : ExperimentResults
        Results from experiment
        
    Returns
    -------
    fig : Figure
    """
    config = results.config
    methods = list(results.results_by_method.keys())
    n_methods = len(methods)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Detection rates bar chart
    ax1 = axes[0, 0]
    rates = results.get_detection_rates()
    
    x = np.arange(n_methods)
    width = 0.35
    
    local_max_rates = [rates.get(m, {}).get('local_max_rate', 0) for m in methods]
    jump_rates = [rates.get(m, {}).get('jump_rate', 0) for m in methods]
    
    colors = [METHOD_COLORS.get(m, 'gray') for m in methods]
    
    bars1 = ax1.bar(x - width/2, local_max_rates, width, label='Local Maximum',
                    color=colors, alpha=0.7, edgecolor='black')
    bars2 = ax1.bar(x + width/2, jump_rates, width, label='Jump Detected',
                    color=colors, alpha=0.4, edgecolor='black', hatch='//')
    
    ax1.set_ylabel('Detection Rate (%)')
    ax1.set_title('Detection at Transition Time')
    ax1.set_xticks(x)
    ax1.set_xticklabels([config.method_display_names.get(m, m) for m in methods],
                        rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars1, local_max_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # 2. Drift percentile boxplot
    ax2 = axes[0, 1]
    percentile_data = []
    for m in methods:
        trials = results.results_by_method.get(m, [])
        percs = [t.detection.metrics_at_transition['drift_percentile'] 
                 for t in trials if t.detection.metrics_at_transition]
        percentile_data.append(percs)
    
    bp = ax2.boxplot(percentile_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50th percentile')
    ax2.set_ylabel('Drift Percentile at Transition (%)')
    ax2.set_title('How Close to Peak is Transition?')
    ax2.set_xticklabels([config.method_display_names.get(m, m) for m in methods],
                        rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Example cumulative drift curves (trial 0)
    ax3 = axes[1, 0]
    for method in methods:
        trials = results.results_by_method.get(method, [])
        if trials:
            t0 = trials[0]
            ax3.plot(t0.segment_times, t0.detection.cumulative_drift,
                     color=METHOD_COLORS.get(method, 'gray'),
                     linewidth=2, label=config.method_display_names.get(method, method))
    
    ax3.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                linestyle='--', alpha=0.7, label='True transition')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cumulative Drift')
    ax3.set_title('Example Drift Curves (Trial 1)')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Rate of change boxplot
    ax4 = axes[1, 1]
    roc_data = []
    for m in methods:
        trials = results.results_by_method.get(m, [])
        rocs = [t.detection.metrics_at_transition['rate_of_change'] 
                for t in trials if t.detection.metrics_at_transition]
        roc_data.append(rocs)
    
    bp2 = ax4.boxplot(roc_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax4.set_ylabel('Rate of Change at Transition')
    ax4.set_title('Drift Acceleration')
    ax4.set_xticklabels([config.method_display_names.get(m, m) for m in methods],
                        rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Title with drift info
    drifting_mode = config.signal.modes[config.signal.drifting_mode_index]
    plt.suptitle(f'Method Comparison - {drifting_mode.freq} Hz Mode Drifting',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_three_mode_study(all_results: Dict[str, ExperimentResults]) -> Figure:
    """
    Summary plot comparing all three drift scenarios.
    
    Parameters
    ----------
    all_results : Dict[str, ExperimentResults]
        Results from run_three_mode_drift_study()
        
    Returns
    -------
    fig : Figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    drift_types = ['strong', 'medium', 'weak']
    drift_labels = ['Strong (0.5 Hz)', 'Medium (1.2 Hz)', 'Weak (2.5 Hz)']
    
    for col, (drift_type, drift_label) in enumerate(zip(drift_types, drift_labels)):
        if drift_type not in all_results:
            continue
            
        results = all_results[drift_type]
        config = results.config
        methods = list(results.results_by_method.keys())
        rates = results.get_detection_rates()
        
        colors = [METHOD_COLORS.get(m, 'gray') for m in methods]
        
        # Top row: Detection rates
        ax_top = axes[0, col]
        x = np.arange(len(methods))
        
        local_max_rates = [rates.get(m, {}).get('local_max_rate', 0) for m in methods]
        jump_rates = [rates.get(m, {}).get('jump_rate', 0) for m in methods]
        
        width = 0.35
        ax_top.bar(x - width/2, local_max_rates, width, color=colors, alpha=0.7,
                   edgecolor='black', label='Local Max')
        ax_top.bar(x + width/2, jump_rates, width, color=colors, alpha=0.4,
                   edgecolor='black', hatch='//', label='Jump')
        
        ax_top.set_ylabel('Detection Rate (%)')
        ax_top.set_title(f'{drift_label} Drifting')
        ax_top.set_xticks(x)
        ax_top.set_xticklabels(['Euc', 'Swap', 'Tr-C', 'Tr-Q'], fontsize=9)
        ax_top.set_ylim(0, 100)
        ax_top.grid(True, alpha=0.3, axis='y')
        if col == 0:
            ax_top.legend(fontsize=8)
        
        # Bottom row: Drift percentiles
        ax_bot = axes[1, col]
        percentile_data = []
        for m in methods:
            trials = results.results_by_method.get(m, [])
            percs = [t.detection.metrics_at_transition['drift_percentile'] 
                     for t in trials if t.detection.metrics_at_transition]
            percentile_data.append(percs)
        
        bp = ax_bot.boxplot(percentile_data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax_bot.axhline(y=50, color='red', linestyle='--', alpha=0.5)
        ax_bot.set_ylabel('Drift Percentile (%)')
        ax_bot.set_xticklabels(['Euc', 'Swap', 'Tr-C', 'Tr-Q'], fontsize=9)
        ax_bot.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Three-Mode-Drift Study: Detection Performance vs Mode Strength',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_distance_correlation(results: ExperimentResults,
                              trial_idx: int = 0) -> Figure:
    """
    Plot correlation between distance methods.
    
    Parameters
    ----------
    results : ExperimentResults
    trial_idx : int
        Which trial to use
        
    Returns
    -------
    fig : Figure
    """
    config = results.config
    methods = list(results.results_by_method.keys())
    n_methods = len(methods)
    
    # Get distance matrices for specified trial
    matrices = {}
    for method in methods:
        trials = results.results_by_method.get(method, [])
        if trial_idx < len(trials):
            matrices[method] = trials[trial_idx].distance_matrix
    
    # Compute correlation matrix
    corr_matrix = np.zeros((n_methods, n_methods))
    
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            if m1 in matrices and m2 in matrices:
                # Flatten upper triangle
                d1 = matrices[m1][np.triu_indices_from(matrices[m1], k=1)]
                d2 = matrices[m2][np.triu_indices_from(matrices[m2], k=1)]
                corr_matrix[i, j] = np.corrcoef(d1, d2)[0, 1]
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    labels = [config.method_display_names.get(m, m) for m in methods]
    ax.set_xticks(range(n_methods))
    ax.set_yticks(range(n_methods))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    
    # Add correlation values
    for i in range(n_methods):
        for j in range(n_methods):
            ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
    
    plt.colorbar(im, ax=ax, label='Correlation')
    ax.set_title(f'Distance Matrix Correlation (Trial {trial_idx + 1})')
    plt.tight_layout()
    
    return fig
