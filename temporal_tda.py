"""
Temporal TDA with Quantum Distance Analysis
Extension for time-varying dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
# import seaborn as sns
from tda import TDAAnalyzer, QuantumDistanceComputer
import time

class TemporalQuantumTDA(TDAAnalyzer):
    """
    Extends TDA with temporal segmentation and quantum distance analysis
    """
    
    def __init__(self, fs=30.0, method='classical', shots=512):
        super().__init__(fs, method)
        self.quantum_computer = QuantumDistanceComputer(method=method, shots=shots)
        print(f"\nTemporal Quantum TDA initialized")
        print(f"  Will track mode evolution using quantum distances")
    
    def segment_signal(self, signal_data, segment_length=60, overlap=0.5):
        """
        Divide signal into overlapping segments
        
        Parameters:
        -----------
        segment_length : float
            Length of each segment in seconds
        overlap : float
            Overlap fraction (0 to 1)
        
        Returns:
        --------
        segments : list of arrays
        segment_times : array of segment center times
        """
        n_samples_per_segment = int(segment_length * self.fs)
        hop = int(n_samples_per_segment * (1 - overlap))
        
        segments = []
        segment_times = []
        
        start = 0
        while start + n_samples_per_segment <= len(signal_data):
            segment = signal_data[start:start + n_samples_per_segment]
            center_time = (start + n_samples_per_segment/2) / self.fs
            
            segments.append(segment)
            segment_times.append(center_time)
            
            start += hop
        
        print(f"\n  Created {len(segments)} segments")
        print(f"  Segment length: {segment_length}s")
        print(f"  Overlap: {overlap*100:.0f}%")
        
        return segments, np.array(segment_times)
    
    def compute_psd_vector(self, segment, freq_range=(0.1, 5.0)):
        """
        Compute PSD for a segment and return as a vector
        """
        f, Pxx = self.compute_welch_psd(segment)
        
        # Filter to frequency range
        freq_mask = (f >= freq_range[0]) & (f <= freq_range[1])
        f_filtered = f[freq_mask]
        psd_filtered = Pxx[freq_mask]
        
        # Work in log scale and normalize
        log_psd = np.log10(psd_filtered + 1e-10)
        psd_normalized = (log_psd - np.mean(log_psd)) / (np.std(log_psd) + 1e-10)
        
        return f_filtered, psd_normalized
    
    def compute_quantum_distance_matrix(self, psd_vectors, method='quantum'):
        """
        Compute pairwise quantum distances between PSD vectors
        
        This is where quantum computation enters!
        Each PSD is a high-dimensional vector - perfect for quantum comparison.
        """
        n_segments = len(psd_vectors)
        distance_matrix = np.zeros((n_segments, n_segments))
        
        print(f"\n  Computing {method} distance matrix:")
        print(f"  Comparing {n_segments} PSD vectors")
        print(f"  PSD dimension: {len(psd_vectors[0])}")
        
        # Set up quantum computer for this dimension
        computer = QuantumDistanceComputer(method=method, shots=512)
        
        start_time = time.time()
        
        for i in range(n_segments):
            for j in range(i+1, n_segments):
                dist = computer.compute_distance(psd_vectors[i], psd_vectors[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
            
            if (i + 1) % max(1, n_segments // 5) == 0:
                elapsed = time.time() - start_time
                progress = 100 * (i + 1) / n_segments
                print(f"    Progress: {progress:.1f}% ({elapsed:.1f}s)")
        
        total_time = time.time() - start_time
        print(f"  ✓ Completed in {total_time:.2f}s")
        
        return distance_matrix, total_time
    
    def analyze_temporal_evolution(self, signal_data, segment_length=60, 
                                   overlap=0.5, freq_range=(0.1, 5.0),
                                   method='classical'):
        """
        Full temporal analysis pipeline
        """
        print("\n" + "="*70)
        print("TEMPORAL QUANTUM TDA ANALYSIS")
        print("="*70)
        
        results = {}
        
        # Step 1: Segment signal
        print("\n[1/5] Segmenting signal...")
        segments, segment_times = self.segment_signal(
            signal_data, segment_length, overlap
        )
        results['segments'] = segments
        results['segment_times'] = segment_times
        
        # Step 2: Compute PSD for each segment
        print("\n[2/5] Computing PSD for each segment...")
        psd_vectors = []
        frequencies = None
        
        for i, segment in enumerate(segments):
            f, psd = self.compute_psd_vector(segment, freq_range)
            if frequencies is None:
                frequencies = f
            psd_vectors.append(psd)
        
        results['frequencies'] = frequencies
        results['psd_vectors'] = np.array(psd_vectors)
        print(f"  ✓ {len(psd_vectors)} PSD vectors computed")
        
        # Step 3: Quantum distance matrix
        print(f"\n[3/5] Computing {method} distance matrix...")
        distance_matrix, dist_time = self.compute_quantum_distance_matrix(
            psd_vectors, method=method
        )
        results['distance_matrix'] = distance_matrix
        results['distance_time'] = dist_time
        results['method'] = method
        
        # Step 4: H0 TDA on each segment
        print("\n[4/5] Applying H0 TDA to each segment...")
        segment_modes = []
        
        for i, segment in enumerate(segments):
            # Get full PSD (not normalized) for TDA
            f_full, psd_full = self.compute_welch_psd(segment)
            
            # Apply H0 TDA
            h0_diagram, f_filt, psd_filt = self.compute_h0_persistence_1d(
                f_full, psd_full, freq_range
            )
            
            if len(h0_diagram) > 0:
                persistences = h0_diagram[:, 1] - h0_diagram[:, 0]
                threshold = np.percentile(persistences, 80)
                
                modes = self.extract_peak_frequencies_h0(
                    f_filt, psd_filt, h0_diagram, threshold
                )
            else:
                modes = np.array([])
            
            segment_modes.append(modes)
        
        results['segment_modes'] = segment_modes
        print(f"  ✓ Modes extracted for all segments")
        
        # Step 5: Detect regime changes
        print("\n[5/5] Detecting regime changes...")
        regime_changes = self.detect_regime_changes(distance_matrix, threshold=0.5)
        results['regime_changes'] = regime_changes
        
        if len(regime_changes) > 0:
            print(f"  ✓ Detected {len(regime_changes)} regime changes at:")
            for idx in regime_changes:
                print(f"    t = {segment_times[idx]:.1f}s")
        else:
            print(f"  ✓ No regime changes detected (stable dynamics)")
        
        print("\n" + "="*70)
        print("TEMPORAL ANALYSIS COMPLETE")
        print("="*70)
        
        return results
    
    def detect_regime_changes(self, distance_matrix, threshold=0.5):
        """
        Detect sudden changes in dynamics by looking at distance jumps
        """
        n = len(distance_matrix)
        
        # Look at distance from each segment to next
        sequential_distances = np.array([
            distance_matrix[i, i+1] for i in range(n-1)
        ])
        
        # Find jumps
        mean_dist = np.mean(sequential_distances)
        std_dist = np.std(sequential_distances)
        
        regime_changes = np.where(
            sequential_distances > mean_dist + threshold * std_dist
        )[0]
        
        return regime_changes
    
    def plot_temporal_analysis(self, signal_data, results, true_modes=None):
        """
        Comprehensive visualization of temporal analysis
        """
        fig = plt.figure(figsize=(18, 12))
        
        segment_times = results['segment_times']
        frequencies = results['frequencies']
        psd_vectors = results['psd_vectors']
        distance_matrix = results['distance_matrix']
        segment_modes = results['segment_modes']
        method = results['method']
        
        # 1. Original signal with segment markers
        ax1 = plt.subplot(4, 3, 1)
        t = np.arange(len(signal_data)) / self.fs
        ax1.plot(t, signal_data, 'b-', linewidth=0.5, alpha=0.7)
        
        for st in segment_times:
            ax1.axvline(x=st, color='r', alpha=0.2, linestyle='--')
        
        if 'regime_changes' in results and len(results['regime_changes']) > 0:
            for idx in results['regime_changes']:
                ax1.axvline(x=segment_times[idx], color='red', 
                           linewidth=2, label='Regime change')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Signal with Segmentation')
        ax1.grid(True, alpha=0.3)
        
        # 2. Spectrogram (PSD evolution)
        ax2 = plt.subplot(4, 3, 2)
        im = ax2.pcolormesh(segment_times, frequencies, psd_vectors.T,
                            shading='auto', cmap='viridis')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('PSD Evolution Over Time')
        plt.colorbar(im, ax=ax2, label='Normalized log(PSD)')
        
        # 3. Distance matrix heatmap
        ax3 = plt.subplot(4, 3, 3)
        im = ax3.imshow(distance_matrix, cmap='hot', aspect='auto')
        ax3.set_xlabel('Segment Index')
        ax3.set_ylabel('Segment Index')
        ax3.set_title(f'Distance Matrix ({method})')
        plt.colorbar(im, ax=ax3, label='Distance')
        
        # 4. Sequential distances (regime change detection)
        ax4 = plt.subplot(4, 3, 4)
        n = len(distance_matrix)
        sequential_dist = [distance_matrix[i, i+1] for i in range(n-1)]
        ax4.plot(segment_times[:-1], sequential_dist, 'b-', linewidth=2)
        
        mean_dist = np.mean(sequential_dist)
        ax4.axhline(y=mean_dist, color='g', linestyle='--', 
                   label=f'Mean: {mean_dist:.3f}')
        
        if 'regime_changes' in results:
            for idx in results['regime_changes']:
                ax4.axvline(x=segment_times[idx], color='red', 
                           linewidth=2, alpha=0.5)
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Distance to Next Segment')
        ax4.set_title('Temporal Distance Evolution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Mode tracking over time
        ax5 = plt.subplot(4, 3, 5)
        
        for i, modes in enumerate(segment_modes):
            if len(modes) > 0:
                time = segment_times[i]
                ax5.scatter([time]*len(modes), modes, c='blue', s=100, alpha=0.6)
        
        if true_modes is not None:
            for mode in true_modes:
                ax5.axhline(y=mode['freq'], color='red', linestyle='--', 
                           alpha=0.5, linewidth=2)
        
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Detected Mode Frequency (Hz)')
        ax5.set_title('Mode Evolution (H0 TDA per segment)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Distance distribution
        ax6 = plt.subplot(4, 3, 6)
        # Get upper triangle of distance matrix (unique distances)
        triu_indices = np.triu_indices_from(distance_matrix, k=1)
        all_distances = distance_matrix[triu_indices]
        
        ax6.hist(all_distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax6.axvline(x=np.mean(all_distances), color='red', linestyle='--',
                   linewidth=2, label=f'Mean: {np.mean(all_distances):.3f}')
        ax6.set_xlabel('Distance')
        ax6.set_ylabel('Count')
        ax6.set_title('Distribution of PSD Distances')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7-9: Individual segment PSDs for first, middle, last
        segment_indices = [0, len(segment_times)//2, len(segment_times)-1]
        for plot_idx, seg_idx in enumerate(segment_indices):
            ax = plt.subplot(4, 3, 7 + plot_idx)
            
            psd = psd_vectors[seg_idx]
            ax.plot(frequencies, psd, 'b-', linewidth=2)
            
            # Mark detected modes
            modes = segment_modes[seg_idx]
            if len(modes) > 0:
                for mode_freq in modes:
                    idx = np.argmin(np.abs(frequencies - mode_freq))
                    ax.plot(mode_freq, psd[idx], 'r*', markersize=15)
            
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Normalized log(PSD)')
            ax.set_title(f'Segment {seg_idx} (t={segment_times[seg_idx]:.1f}s)')
            ax.grid(True, alpha=0.3)
        
        # 10. Number of modes over time
        ax10 = plt.subplot(4, 3, 10)
        n_modes = [len(modes) for modes in segment_modes]
        ax10.plot(segment_times, n_modes, 'bo-', linewidth=2, markersize=8)
        ax10.set_xlabel('Time (s)')
        ax10.set_ylabel('Number of Detected Modes')
        ax10.set_title('Mode Count Evolution')
        ax10.grid(True, alpha=0.3)
        
        # 11. Distance to first segment (drift detection)
        ax11 = plt.subplot(4, 3, 11)
        dist_to_first = distance_matrix[0, :]
        ax11.plot(segment_times, dist_to_first, 'g-', linewidth=2)
        ax11.set_xlabel('Time (s)')
        ax11.set_ylabel('Distance to Initial State')
        ax11.set_title('Cumulative Drift from t=0')
        ax11.grid(True, alpha=0.3)
        
        # 12. Summary statistics
        ax12 = plt.subplot(4, 3, 12)
        ax12.axis('off')
        
        summary = f"""
        TEMPORAL ANALYSIS SUMMARY
        
        Signal Duration: {len(signal_data)/self.fs:.1f}s
        Number of Segments: {len(segment_times)}
        
        Distance Matrix: {method}
        Mean Distance: {np.mean(all_distances):.4f}
        Std Distance: {np.std(all_distances):.4f}
        
        Regime Changes: {len(results.get('regime_changes', []))}
        
        Modes Detected:
        """
        
        # Count unique modes across all segments
        all_modes = []
        for modes in segment_modes:
            all_modes.extend(modes)
        
        if len(all_modes) > 0:
            unique_modes = np.unique(np.round(all_modes, 1))
            summary += f"  {len(unique_modes)} unique: {unique_modes}\n"
        else:
            summary += "  None detected\n"
        
        ax12.text(0.1, 0.5, summary, fontsize=10, family='monospace',
                 verticalalignment='center')
        
        plt.suptitle(f'Temporal Quantum TDA Analysis ({method})',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig


def generate_evolving_signal(t, modes, transition_time=60):
    """
    Generate signal where one mode changes frequency over time
    (simulates a destabilizing mode)
    """
    y = np.zeros_like(t)
    
    for i, mode in enumerate(modes):
        freq = mode['freq']
        zeta = mode['damping']
        A = mode['amplitude']
        
        # Make the last mode drift in frequency
        if i == len(modes) - 1:
            # Frequency drift after transition_time
            freq_drift = np.where(
                t < transition_time,
                freq,
                freq + 0.5 * (t - transition_time) / (t[-1] - transition_time)
            )
            omega_d = 2 * np.pi * freq_drift
        else:
            omega_d = 2 * np.pi * freq
        
        decay = np.exp(-zeta * omega_d * t)
        oscillation = A * decay * np.sin(omega_d * t)
        y += oscillation
    
    noise = 0.02 * np.random.randn(len(t))
    y += noise
    
    return y


def test_temporal_quantum_tda():
    """
    Test temporal analysis with evolving dynamics
    """
    print("\n" + "="*70)
    print("TESTING TEMPORAL QUANTUM TDA")
    print("="*70)
    
    # Generate signal with evolving mode
    modes = [
        {'freq': 0.5, 'damping': 0.03, 'amplitude': 0.25},
        {'freq': 1.2, 'damping': 0.03, 'amplitude': 0.25},
        {'freq': 2.5, 'damping': 0.03, 'amplitude': 0.20},  # This one will drift
    ]
    
    duration = 180.0  # 3 minutes
    fs = 30.0
    t = np.arange(0, duration, 1/fs)
    
    signal_data = generate_evolving_signal(t, modes, transition_time=90)
    
    print(f"\nGenerated signal: {duration}s")
    print(f"  Mode at 2.5 Hz drifts upward after t=90s")
    
    # Analyze classically
    c_analyzer = TemporalQuantumTDA(fs=fs, method='classical')
    # Record the start time
    c_start_time = time.time()
    c_results = c_analyzer.analyze_temporal_evolution(
        signal_data,
        segment_length=30,  # 30s segments
        overlap=0.5,
        method='classical'  # Start with classical
    )
    # Record the start time
    c_end_time = time.time()
    # Calculate the elapsed time
    c_elapsed_time = c_end_time - c_start_time
    c_results['elapsed_time'] = c_elapsed_time

    # Print the result
    print(f"Execution time: {c_elapsed_time:.4f} seconds")
    
    # Plot
    fig = c_analyzer.plot_temporal_analysis(signal_data, c_results, true_modes=modes)
    plt.savefig('temporal_classical_tda.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: temporal_classical_tda.png")
    
    plt.show()

    # Analyze with Q
    q_analyzer = TemporalQuantumTDA(fs=fs, method='swap_test')
    
    # Record the start time
    q_start_time = time.time()
    q_results = c_analyzer.analyze_temporal_evolution(
        signal_data,
        segment_length=30,  # 30s segments
        overlap=0.5,
        method='swap_test'  # Start with classical
    )
    # Record the start time
    q_end_time = time.time()
    # Calculate the elapsed time
    q_elapsed_time = q_end_time - q_start_time
    q_results['elapsed_time'] = q_elapsed_time

    # Print the result
    print(f"Execution time: {q_elapsed_time:.4f} seconds")
    
    # Plot
    fig = c_analyzer.plot_temporal_analysis(signal_data, q_results, true_modes=modes)
    plt.savefig('temporal_quantum_tda.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: temporal_quantum_tda.png")
    
    plt.show()
    
    return c_results, q_results

def compare_classical_vs_quantum():
    """
    Run temporal analysis with BOTH classical and quantum distances
    Direct comparison
    """
    print("\n" + "="*70)
    print("COMPARING CLASSICAL VS QUANTUM DISTANCE COMPUTATION")
    print("="*70)
    
    # Generate signal with evolving mode
    modes = [
        {'freq': 0.5, 'damping': 0.03, 'amplitude': 0.25},
        {'freq': 1.2, 'damping': 0.03, 'amplitude': 0.25},
        {'freq': 2.5, 'damping': 0.03, 'amplitude': 0.20},  # Drifts
    ]
    
    duration = 180.0
    fs = 30.0
    t = np.arange(0, duration, 1/fs)
    
    signal_data = generate_evolving_signal(t, modes, transition_time=90)
    
    print(f"\nGenerated signal: {duration}s")
    print(f"  Mode at 2.5 Hz drifts upward after t=90s")
    
    # Run CLASSICAL
    print("\n" + "="*70)
    print("RUNNING CLASSICAL ANALYSIS")
    print("="*70)
    
    analyzer_classical = TemporalQuantumTDA(fs=fs, method='classical')
    
    results_classical = analyzer_classical.analyze_temporal_evolution(
        signal_data,
        segment_length=30,
        overlap=0.5,
        method='classical'
    )
    
    # Run QUANTUM
    print("\n" + "="*70)
    print("RUNNING QUANTUM ANALYSIS")
    print("="*70)
    
    analyzer_quantum = TemporalQuantumTDA(fs=fs, method='swap_test')
    
    results_quantum = analyzer_quantum.analyze_temporal_evolution(
        signal_data,
        segment_length=30,
        overlap=0.5,
        method='swap_test'
    )
    
    # Compare results
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    
    print(f"\nDistance Matrix Computation Time:")
    print(f"  Classical: {results_classical['distance_time']:.3f}s")
    print(f"  Quantum:   {results_quantum['distance_time']:.3f}s")
    print(f"  Speedup:   {results_quantum['distance_time']/results_classical['distance_time']:.1f}x slower")
    
    print(f"\nDistance Matrix Statistics:")
    dm_c = results_classical['distance_matrix']
    dm_q = results_quantum['distance_matrix']
    
    # Get upper triangle (unique distances)
    triu = np.triu_indices_from(dm_c, k=1)
    dist_c = dm_c[triu]
    dist_q = dm_q[triu]
    
    print(f"  Classical: mean={np.mean(dist_c):.4f}, std={np.std(dist_c):.4f}")
    print(f"  Quantum:   mean={np.mean(dist_q):.4f}, std={np.std(dist_q):.4f}")
    
    # Correlation
    corr = np.corrcoef(dist_c, dist_q)[0, 1]
    print(f"\nCorrelation between distance matrices: {corr:.4f}")
    
    # Compare regime changes
    print(f"\nRegime Changes Detected:")
    rc_c = results_classical.get('regime_changes', [])
    rc_q = results_quantum.get('regime_changes', [])
    print(f"  Classical: {len(rc_c)} changes at {[results_classical['segment_times'][i] for i in rc_c]}")
    print(f"  Quantum:   {len(rc_q)} changes at {[results_quantum['segment_times'][i] for i in rc_q]}")
    
    # Plot comparison
    fig = plot_classical_vs_quantum_comparison(
        signal_data, results_classical, results_quantum, modes, fs
    )
    
    plt.savefig('classical_vs_quantum_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: classical_vs_quantum_comparison.png")
    
    plt.show()
    
    return results_classical, results_quantum


def plot_classical_vs_quantum_comparison(signal_data, results_c, results_q, 
                                         true_modes, fs):
    """
    Side-by-side comparison visualization
    """
    fig = plt.figure(figsize=(20, 12))
    
    segment_times = results_c['segment_times']
    t = np.arange(len(signal_data)) / fs
    
    # 1. Original signal (shared)
    ax1 = plt.subplot(4, 3, 1)
    ax1.plot(t, signal_data, 'b-', linewidth=0.5, alpha=0.7)
    ax1.axvline(x=90, color='green', linewidth=3, alpha=0.5, 
                label='True transition (t=90s)')
    for st in segment_times:
        ax1.axvline(x=st, color='gray', alpha=0.2, linestyle='--')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Signal (Mode drifts at t=90s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Classical Distance Matrix
    ax2 = plt.subplot(4, 3, 2)
    im = ax2.imshow(results_c['distance_matrix'], cmap='hot', aspect='auto')
    ax2.set_xlabel('Segment Index')
    ax2.set_ylabel('Segment Index')
    ax2.set_title('CLASSICAL Distance Matrix')
    plt.colorbar(im, ax=ax2, label='Distance')
    
    # 3. Quantum Distance Matrix
    ax3 = plt.subplot(4, 3, 3)
    im = ax3.imshow(results_q['distance_matrix'], cmap='hot', aspect='auto')
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Segment Index')
    ax3.set_title('QUANTUM Distance Matrix')
    plt.colorbar(im, ax=ax3, label='Distance')
    
    # 4. Difference Matrix
    ax4 = plt.subplot(4, 3, 4)
    diff_matrix = np.abs(results_q['distance_matrix'] - results_c['distance_matrix'])
    im = ax4.imshow(diff_matrix, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Segment Index')
    ax4.set_ylabel('Segment Index')
    ax4.set_title('DIFFERENCE |Quantum - Classical|')
    plt.colorbar(im, ax=ax4, label='Absolute Difference')
    
    # 5. Sequential distances comparison
    ax5 = plt.subplot(4, 3, 5)
    n = len(results_c['distance_matrix'])
    seq_dist_c = [results_c['distance_matrix'][i, i+1] for i in range(n-1)]
    seq_dist_q = [results_q['distance_matrix'][i, i+1] for i in range(n-1)]
    
    ax5.plot(segment_times[:-1], seq_dist_c, 'b-o', linewidth=2, 
             markersize=8, label='Classical')
    ax5.plot(segment_times[:-1], seq_dist_q, 'r-s', linewidth=2, 
             markersize=8, label='Quantum', alpha=0.7)
    ax5.axvline(x=90, color='green', linewidth=2, alpha=0.3, 
                linestyle='--', label='True transition')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Distance to Next Segment')
    ax5.set_title('Temporal Evolution (Classical vs Quantum)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative drift comparison
    ax6 = plt.subplot(4, 3, 6)
    drift_c = results_c['distance_matrix'][0, :]
    drift_q = results_q['distance_matrix'][0, :]
    
    ax6.plot(segment_times, drift_c, 'b-o', linewidth=2, label='Classical')
    ax6.plot(segment_times, drift_q, 'r-s', linewidth=2, label='Quantum', alpha=0.7)
    ax6.axvline(x=90, color='green', linewidth=2, alpha=0.3, linestyle='--')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Distance from t=0')
    ax6.set_title('Cumulative Drift Detection')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Distance distributions
    ax7 = plt.subplot(4, 3, 7)
    triu = np.triu_indices_from(results_c['distance_matrix'], k=1)
    dist_c = results_c['distance_matrix'][triu]
    dist_q = results_q['distance_matrix'][triu]
    
    ax7.hist(dist_c, bins=20, alpha=0.5, color='blue', label='Classical', 
             edgecolor='black')
    ax7.hist(dist_q, bins=20, alpha=0.5, color='red', label='Quantum', 
             edgecolor='black')
    ax7.axvline(x=np.mean(dist_c), color='blue', linestyle='--', linewidth=2)
    ax7.axvline(x=np.mean(dist_q), color='red', linestyle='--', linewidth=2)
    ax7.set_xlabel('Distance')
    ax7.set_ylabel('Count')
    ax7.set_title('Distance Distribution Comparison')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Scatter: Classical vs Quantum distances
    ax8 = plt.subplot(4, 3, 8)
    ax8.scatter(dist_c, dist_q, alpha=0.6, s=50)
    
    # Perfect correlation line
    min_val = min(np.min(dist_c), np.min(dist_q))
    max_val = max(np.max(dist_c), np.max(dist_q))
    ax8.plot([min_val, max_val], [min_val, max_val], 'k--', 
             linewidth=2, label='Perfect correlation')
    
    # Correlation coefficient
    corr = np.corrcoef(dist_c, dist_q)[0, 1]
    ax8.text(0.05, 0.95, f'Correlation: {corr:.4f}', 
             transform=ax8.transAxes, fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax8.set_xlabel('Classical Distance')
    ax8.set_ylabel('Quantum Distance')
    ax8.set_title('Classical vs Quantum Correlation')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Regime change detection comparison
    ax9 = plt.subplot(4, 3, 9)
    
    rc_c = results_c.get('regime_changes', [])
    rc_q = results_q.get('regime_changes', [])
    
    # Plot both regime change detections
    y_classical = np.ones(len(rc_c)) * 0.3
    y_quantum = np.ones(len(rc_q)) * 0.7
    
    if len(rc_c) > 0:
        ax9.scatter([segment_times[i] for i in rc_c], 
                   y_classical, s=200, c='blue', marker='|', 
                   linewidths=4, label='Classical')
    
    if len(rc_q) > 0:
        ax9.scatter([segment_times[i] for i in rc_q], 
                   y_quantum, s=200, c='red', marker='|', 
                   linewidths=4, label='Quantum')
    
    ax9.axvline(x=90, color='green', linewidth=3, alpha=0.5, 
                label='True transition')
    
    ax9.set_xlabel('Time (s)')
    ax9.set_yticks([0.3, 0.7])
    ax9.set_yticklabels(['Classical', 'Quantum'])
    ax9.set_ylim([0, 1])
    ax9.set_title('Regime Change Detection')
    ax9.legend(loc='upper right')
    ax9.grid(True, alpha=0.3, axis='x')
    
    # 10. Relative error per segment pair
    ax10 = plt.subplot(4, 3, 10)
    relative_error = np.abs(results_q['distance_matrix'] - 
                           results_c['distance_matrix']) / (results_c['distance_matrix'] + 1e-10)
    
    im = ax10.imshow(relative_error, cmap='plasma', aspect='auto')
    ax10.set_xlabel('Segment Index')
    ax10.set_ylabel('Segment Index')
    ax10.set_title('Relative Error: |Q-C|/C')
    plt.colorbar(im, ax=ax10, label='Relative Error')
    
    # 11. Summary statistics table
    ax11 = plt.subplot(4, 3, 11)
    ax11.axis('off')
    
    summary = f"""
    COMPARISON SUMMARY
    
    Computation Time:
      Classical: {results_c['distance_time']:.3f}s
      Quantum:   {results_q['distance_time']:.3f}s
      Ratio:     {results_q['distance_time']/results_c['distance_time']:.0f}x
    
    Distance Statistics:
      Classical mean: {np.mean(dist_c):.4f}
      Quantum mean:   {np.mean(dist_q):.4f}
      
      Correlation: {corr:.4f}
      Mean abs diff: {np.mean(np.abs(dist_q - dist_c)):.4f}
      Max abs diff:  {np.max(np.abs(dist_q - dist_c)):.4f}
    
    Regime Changes:
      Classical: {len(rc_c)}
      Quantum:   {len(rc_q)}
    """
    
    ax11.text(0.1, 0.5, summary, fontsize=9, family='monospace',
             verticalalignment='center')
    
    # 12. Mode detection consistency
    ax12 = plt.subplot(4, 3, 12)
    
    modes_c = results_c['segment_modes']
    modes_q = results_q['segment_modes']
    
    # Count modes per segment
    n_modes_c = [len(m) for m in modes_c]
    n_modes_q = [len(m) for m in modes_q]
    
    ax12.plot(segment_times, n_modes_c, 'b-o', linewidth=2, 
             markersize=8, label='Classical')
    ax12.plot(segment_times, n_modes_q, 'r-s', linewidth=2, 
             markersize=8, label='Quantum', alpha=0.7)
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Number of Modes')
    ax12.set_title('Mode Count (H0 TDA on segments)')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.suptitle('Classical vs Quantum Distance Computation Comparison',
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    # Run temporal test
    c_results, q_results = test_temporal_quantum_tda()

    results_classical, results_quantum = compare_classical_vs_quantum()