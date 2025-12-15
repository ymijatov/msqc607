"""
Visualization

Clean, modular plotting functions.
Separates L1 and L2 metric families for proper scale comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from typing import Dict, List, Optional, Tuple
from matplotlib.figure import Figure

from config import ExperimentConfig
from experiments import ExperimentResults, TrialResult
from real_data import load_gesl_signal
from real_data_experiments import RealDataExperimentResults


# Color scheme for methods
METHOD_COLORS = {
    'classical': '#1f77b4',                  # blue
    'swap_test': '#ff7f0e',                  # orange
    'trace_distance_classical': '#2ca02c',   # green
    'trace_distance_quantum': '#d62728',     # red
}

# Consistent display names
METHOD_DISPLAY_NAMES = {
    'classical': 'Euclidean (L2)',
    'swap_test': 'Swap Test (L2)',
    'trace_distance_classical': 'Trace Classical (L1)',
    'trace_distance_quantum': 'Trace Quantum (L1)',
}

# Metric family groupings
L2_METHODS = ['classical', 'swap_test']
L1_METHODS = ['trace_distance_classical', 'trace_distance_quantum']


def _get_display_name(method: str, config: ExperimentConfig = None) -> str:
    """Get consistent display name for a method."""
    if config and hasattr(config, 'method_display_names'):
        return config.method_display_names.get(method, METHOD_DISPLAY_NAMES.get(method, method))
    return METHOD_DISPLAY_NAMES.get(method, method)


def _get_methods_by_family(methods: List[str]) -> Tuple[List[str], List[str]]:
    """Split methods into L2 and L1 families."""
    l2 = [m for m in methods if m in L2_METHODS]
    l1 = [m for m in methods if m in L1_METHODS]
    return l2, l1


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
    
    method_name = _get_display_name(result.method, config)
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
    
    Separates L1 and L2 metric families into different subplots for proper scaling.
    
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
    l2_methods, l1_methods = _get_methods_by_family(methods)
    
    # 3 rows x 2 cols layout:
    # Row 1: Detection rates | Drift percentile boxplot
    # Row 2: L2 cumulative drift | L1 cumulative drift  
    # Row 3: Rate of change boxplot | Correlation summary
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    
    # =========================================
    # 1. Detection rates bar chart (top-left)
    # =========================================
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
    ax1.set_xticklabels([_get_display_name(m, config) for m in methods],
                        rotation=45, ha='right', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars1, local_max_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                 f'{val:.0f}%', ha='center', va='bottom', fontsize=8)
    
    # =========================================
    # 2. Drift percentile boxplot (top-right)
    # =========================================
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
    ax2.set_xticklabels([_get_display_name(m, config) for m in methods],
                        rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # =========================================
    # 3. L2 Cumulative drift curves (middle-left)
    # =========================================
    ax3 = axes[1, 0]
    for method in l2_methods:
        trials = results.results_by_method.get(method, [])
        if trials:
            t0 = trials[0]
            ax3.plot(t0.segment_times, t0.detection.cumulative_drift,
                     color=METHOD_COLORS.get(method, 'gray'),
                     linewidth=2, label=_get_display_name(method, config))
    
    ax3.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                linestyle='--', alpha=0.7, label='True transition')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Cumulative Drift (L2 distance)')
    ax3.set_title('L2 Metric Family: Euclidean Distances')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # =========================================
    # 4. L1 Cumulative drift curves (middle-right)
    # =========================================
    ax4 = axes[1, 1]
    for method in l1_methods:
        trials = results.results_by_method.get(method, [])
        if trials:
            t0 = trials[0]
            ax4.plot(t0.segment_times, t0.detection.cumulative_drift,
                     color=METHOD_COLORS.get(method, 'gray'),
                     linewidth=2, label=_get_display_name(method, config))
    
    ax4.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                linestyle='--', alpha=0.7, label='True transition')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Cumulative Drift (L1 distance)')
    ax4.set_title('L1 Metric Family: Trace Distances (bounded [0,1])')
    ax4.set_ylim(0, 1.0)  # L1 trace distance is bounded
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # =========================================
    # 5. Rate of change boxplot (bottom-left)
    # =========================================
    ax5 = axes[2, 0]
    roc_data = []
    for m in methods:
        trials = results.results_by_method.get(m, [])
        rocs = [t.detection.metrics_at_transition['rate_of_change'] 
                for t in trials if t.detection.metrics_at_transition]
        roc_data.append(rocs)
    
    bp2 = ax5.boxplot(roc_data, patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax5.set_ylabel('Rate of Change at Transition')
    ax5.set_title('Drift Acceleration')
    ax5.set_xticklabels([_get_display_name(m, config) for m in methods],
                        rotation=45, ha='right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # =========================================
    # 6. Method family comparison summary (bottom-right)
    # =========================================
    ax6 = axes[2, 1]
    
    # Compare L2 vs L1 family performance
    l2_rates = [rates.get(m, {}).get('mean_percentile', 0) for m in l2_methods]
    l1_rates = [rates.get(m, {}).get('mean_percentile', 0) for m in l1_methods]
    
    x_fam = np.arange(2)
    l2_mean = np.mean(l2_rates) if l2_rates else 0
    l1_mean = np.mean(l1_rates) if l1_rates else 0
    
    bars = ax6.bar(x_fam, [l2_mean, l1_mean], color=['#3498db', '#27ae60'], 
                   edgecolor='black', alpha=0.8)
    ax6.set_xticks(x_fam)
    ax6.set_xticklabels(['L2 Family\n(Euclidean)', 'L1 Family\n(Trace Distance)'])
    ax6.set_ylabel('Mean Drift Percentile at Transition (%)')
    ax6.set_title('Metric Family Comparison')
    ax6.axhline(y=50, color='red', linestyle='--', alpha=0.5)
    ax6.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, [l2_mean, l1_mean]):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Title with drift info
    drifting_mode = config.signal.modes[config.signal.drifting_mode_index]
    plt.suptitle(f'Method Comparison - {drifting_mode.freq} Hz Mode Drifting',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig


def plot_three_mode_study(all_results: Dict[str, ExperimentResults]) -> Figure:
    """
    Summary plot comparing all three drift scenarios.
    
    Separates L1 and L2 families for proper comparison.
    
    Parameters
    ----------
    all_results : Dict[str, ExperimentResults]
        Results from run_three_mode_drift_study()
        
    Returns
    -------
    fig : Figure
    """
    # 3 rows x 3 cols:
    # Row 1: Detection rates for strong, medium, weak
    # Row 2: L2 drift curves for strong, medium, weak
    # Row 3: L1 drift curves for strong, medium, weak
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    
    drift_types = ['strong', 'medium', 'weak']
    drift_labels = ['Strong (0.5 Hz)', 'Medium (1.2 Hz)', 'Weak (2.5 Hz)']
    
    for col, (drift_type, drift_label) in enumerate(zip(drift_types, drift_labels)):
        if drift_type not in all_results:
            continue
            
        results = all_results[drift_type]
        config = results.config
        methods = list(results.results_by_method.keys())
        rates = results.get_detection_rates()
        l2_methods, l1_methods = _get_methods_by_family(methods)
        
        colors = [METHOD_COLORS.get(m, 'gray') for m in methods]
        
        # =========================================
        # Row 1: Detection rates
        # =========================================
        ax_top = axes[0, col]
        x = np.arange(len(methods))
        
        local_max_rates = [rates.get(m, {}).get('local_max_rate', 0) for m in methods]
        jump_rates = [rates.get(m, {}).get('jump_rate', 0) for m in methods]
        
        width = 0.35
        ax_top.bar(x - width/2, local_max_rates, width, color=colors, alpha=0.7,
                   edgecolor='black', label='Local Max')
        ax_top.bar(x + width/2, jump_rates, width, color=colors, alpha=0.4,
                   edgecolor='black', hatch='//', label='Jump')
        
        ax_top.set_ylabel('Detection Rate (%)' if col == 0 else '')
        ax_top.set_title(f'{drift_label} Drifting')
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([_get_display_name(m, config) for m in methods],
                              rotation=45, ha='right', fontsize=8)
        ax_top.set_ylim(0, 100)
        ax_top.grid(True, alpha=0.3, axis='y')
        if col == 0:
            ax_top.legend(fontsize=8, loc='upper right')
        
        # =========================================
        # Row 2: L2 cumulative drift
        # =========================================
        ax_l2 = axes[1, col]
        for method in l2_methods:
            trials = results.results_by_method.get(method, [])
            if trials:
                t0 = trials[0]
                ax_l2.plot(t0.segment_times, t0.detection.cumulative_drift,
                          color=METHOD_COLORS.get(method, 'gray'),
                          linewidth=2, label=_get_display_name(method, config))
        
        ax_l2.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                      linestyle='--', alpha=0.7)
        ax_l2.set_xlabel('')
        ax_l2.set_ylabel('L2 Drift' if col == 0 else '')
        if col == 0:
            ax_l2.legend(fontsize=7, loc='upper left')
        ax_l2.grid(True, alpha=0.3)
        if col == 1:
            ax_l2.set_title('L2 Family (Euclidean)', fontsize=10)
        
        # =========================================
        # Row 3: L1 cumulative drift
        # =========================================
        ax_l1 = axes[2, col]
        for method in l1_methods:
            trials = results.results_by_method.get(method, [])
            if trials:
                t0 = trials[0]
                ax_l1.plot(t0.segment_times, t0.detection.cumulative_drift,
                          color=METHOD_COLORS.get(method, 'gray'),
                          linewidth=2, label=_get_display_name(method, config))
        
        ax_l1.axvline(x=config.signal.transition_time, color='red', linewidth=2,
                      linestyle='--', alpha=0.7)
        ax_l1.set_xlabel('Time (s)')
        ax_l1.set_ylabel('L1 Drift' if col == 0 else '')
        ax_l1.set_ylim(0, 1.0)  # L1 bounded
        if col == 0:
            ax_l1.legend(fontsize=7, loc='upper left')
        ax_l1.grid(True, alpha=0.3)
        if col == 1:
            ax_l1.set_title('L1 Family (Trace Distance, bounded [0,1])', fontsize=10)
    
    plt.suptitle('Three-Mode-Drift Study: L1 vs L2 Metric Families',
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
    
    labels = [_get_display_name(m, config) for m in methods]
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


# =========================================
# Real Data Benchmark Visualization
# =========================================

def benchmark_visualization(
    signal_path: str,
    metadata_path: str,
    results: RealDataExperimentResults,
    pmu_id: str = 'P001',
    measurement: str = 'vp_m',
    known_frequencies: Optional[List[float]] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Create comprehensive benchmark visualization for real data.
    
    Separates L1 and L2 metric families into different subplots.
    """
    # Load raw signal (not normalized, for visualization)
    t, signal_raw, fs, info = load_gesl_signal(
        signal_path, metadata_path,
        pmu_id=pmu_id, measurement=measurement,
        detrend=True, normalize=False
    )
    
    if known_frequencies is None and info:
        known_frequencies = info.oscillation_frequencies
    
    # Get results by family
    l2_methods, l1_methods = _get_methods_by_family(list(results.results_by_method.keys()))
    
    # 5 rows now: signal, spectrogram, modes, L2 drift, L1 drift
    fig = plt.figure(figsize=(16, 18))
    
    # =========================================
    # 1. Raw signal with regime changes
    # =========================================
    ax1 = fig.add_subplot(5, 1, 1)
    ax1.plot(t, signal_raw, 'k-', linewidth=0.3, alpha=0.7)
    ax1.set_ylabel(f'{measurement}')
    ax1.set_title(f'Signal {info.signal_id if info else ""}: {pmu_id}.{measurement}')
    
    # Overlay regime change detections
    for method in results.results_by_method.keys():
        color = METHOD_COLORS.get(method, 'gray')
        transitions = results.get_detected_transitions(method)
        for i, trans in enumerate(transitions):
            label = _get_display_name(method) if i == 0 else None
            ax1.axvline(trans, color=color, linestyle='--', alpha=0.7, linewidth=1.5, label=label)
    
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(t[0], t[-1])
    
    # =========================================
    # 2. Spectrogram (ground truth time-frequency)
    # =========================================
    ax2 = fig.add_subplot(5, 1, 2)
    
    nperseg = int(15 * fs)
    noverlap = int(nperseg * 0.5)
    f_spec, t_spec, Sxx = sig.spectrogram(
        signal_raw, fs=fs, 
        nperseg=nperseg, noverlap=noverlap,
        window='hann'
    )
    
    freq_mask = f_spec <= 3.0
    
    im = ax2.pcolormesh(
        t_spec, f_spec[freq_mask], 
        10 * np.log10(Sxx[freq_mask, :] + 1e-12),
        shading='gouraud', cmap='viridis'
    )
    
    if known_frequencies:
        for freq in known_frequencies:
            ax2.axhline(freq, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Spectrogram — Red dashed = known oscillation frequencies')
    plt.colorbar(im, ax=ax2, label='Power (dB)')
    ax2.set_xlim(t[0], t[-1])
    
    # =========================================
    # 3. Detected modes per segment
    # =========================================
    ax3 = fig.add_subplot(5, 1, 3)
    
    trace_result = results.results_by_method.get('trace_distance_classical')
    if trace_result:
        segment_times = trace_result.segment_times
        modes_per_segment = trace_result.modes_per_segment
        
        for seg_t, modes in zip(segment_times, modes_per_segment):
            if len(modes) > 0:
                ax3.scatter([seg_t] * len(modes), modes, c='blue', s=30, alpha=0.7)
        
        if known_frequencies:
            for freq in known_frequencies:
                ax3.axhline(freq, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        
        ax3.set_ylabel('Detected Mode (Hz)')
        ax3.set_title('TDA Mode Detection — Blue dots = detected, Red dashed = known')
        ax3.set_ylim(0, 3.0)
        ax3.set_xlim(t[0], t[-1])
    
    # =========================================
    # 4. L2 Cumulative drift
    # =========================================
    ax4 = fig.add_subplot(5, 1, 4)
    
    for method in l2_methods:
        if method in results.results_by_method:
            result = results.results_by_method[method]
            ax4.plot(result.segment_times, result.cumulative_drift, 
                    color=METHOD_COLORS.get(method, 'gray'), 
                    linewidth=2, label=_get_display_name(method))
    
    ax4.set_ylabel('L2 Cumulative Drift')
    ax4.set_title('L2 Metric Family (Euclidean distances)')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.set_xlim(t[0], t[-1])
    ax4.grid(True, alpha=0.3)
    
    # =========================================
    # 5. L1 Cumulative drift
    # =========================================
    ax5 = fig.add_subplot(5, 1, 5)
    
    for method in l1_methods:
        if method in results.results_by_method:
            result = results.results_by_method[method]
            ax5.plot(result.segment_times, result.cumulative_drift, 
                    color=METHOD_COLORS.get(method, 'gray'), 
                    linewidth=2, label=_get_display_name(method))
    
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('L1 Cumulative Drift')
    ax5.set_title('L1 Metric Family (Trace distances, bounded [0,1])')
    ax5.set_ylim(0, 1.0)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.set_xlim(t[0], t[-1])
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def quick_benchmark(
    signal_path: str,
    metadata_path: str, 
    results: RealDataExperimentResults,
    save_path: Optional[str] = None,
) -> Figure:
    """One-liner benchmark."""
    return benchmark_visualization(
        signal_path, metadata_path, results,
        pmu_id=results.config.pmu_id,
        save_path=save_path,
    )