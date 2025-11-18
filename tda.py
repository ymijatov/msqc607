"""
Quantum-Enhanced TDA for Synchrophasor Data

Following Mishra & Vanfretti (2025):
- Use H0 persistence on 1D PSD function
- Find maxima (peaks) directly as persistent H0 components
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

# Check gudhi
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    print("Warning: Gudhi not installed. Install with: pip install gudhi")
    GUDHI_AVAILABLE = False

print(f"✓ Using Qiskit with AerSimulator")
print(f"✓ Gudhi available: {GUDHI_AVAILABLE}")

class QuantumDistanceComputer:
    """
    Quantum Distance Matrix Computation
    NOTE: For H0 on 1D functions, we don't actually need this!
    But keeping it for the quantum learning exercise.
    """
    
    def __init__(self, method='swap_test', shots=512):
        self.method = method
        self.shots = shots
        self.simulator = AerSimulator()
        print(f"\nQuantum Distance Computer initialized:")
        print(f"  Method: {method}")
        print(f"  Shots: {shots}")
    
    def normalize_vector(self, vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
    
    def create_swap_test_circuit(self, state1, state2):
        n_qubits = int(np.ceil(np.log2(max(len(state1), len(state2)))))
        n_qubits = max(n_qubits, 1)
        padded_size = 2 ** n_qubits
        
        state1_padded = np.pad(state1, (0, padded_size - len(state1)))
        state2_padded = np.pad(state2, (0, padded_size - len(state2)))
        
        state1_padded = state1_padded / np.linalg.norm(state1_padded)
        state2_padded = state2_padded / np.linalg.norm(state2_padded)
        
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_state1 = QuantumRegister(n_qubits, 'state1')
        qr_state2 = QuantumRegister(n_qubits, 'state2')
        cr = ClassicalRegister(1, 'result')
        
        qc = QuantumCircuit(qr_ancilla, qr_state1, qr_state2, cr)
        
        qc.initialize(state1_padded, qr_state1)
        qc.initialize(state2_padded, qr_state2)
        
        qc.h(qr_ancilla[0])
        
        for i in range(n_qubits):
            qc.cswap(qr_ancilla[0], qr_state1[i], qr_state2[i])
        
        qc.h(qr_ancilla[0])
        qc.measure(qr_ancilla[0], cr[0])
        
        return qc
    
    def compute_distance(self, vec1, vec2):
        """Compute distance - keeping for quantum learning"""
        if self.method == 'classical':
            return np.linalg.norm(vec1 - vec2)
        elif self.method == 'swap_test':
            # Quantum distance computation
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                state1 = vec1 / norm1
                state2 = vec2 / norm2
                
                qc = self.create_swap_test_circuit(state1, state2)
                job = self.simulator.run(qc, shots=self.shots)
                result = job.result()
                counts = result.get_counts()
                
                prob_0 = counts.get('0', 0) / self.shots
                inner_product_squared = max(0, 2 * prob_0 - 1)
                inner_product = np.sqrt(inner_product_squared)
                
                distance_squared = norm1**2 + norm2**2 - 2*norm1*norm2*inner_product
                distance = np.sqrt(max(0, distance_squared))
                
                return distance
            else:
                return 0.0


class TDAAnalyzer:
    """
    Uses H0 persistence on 1D PSD function
    """
    
    def __init__(self, fs=30.0, method='classical'):
        self.fs = fs
        self.dt = 1.0 / fs
        self.method = method
        
        # We don't actually need quantum for H0 on 1D function!
        # But keeping option for learning
        if method != 'classical':
            self.quantum_computer = QuantumDistanceComputer(method=method)
        
        print(f"\nQuantum TDA Analyzer initialized (fs={fs} Hz)")
        print(f"  Using H0 persistence on 1D PSD function (correct method)")
    
    def generate_synchrophasor_signal(self, t, modes, noise_level=0.02):
        """Generate synthetic synchrophasor signal"""
        y = np.zeros_like(t)
        
        for mode in modes:
            freq = mode['freq']
            zeta = mode['damping']
            A = mode['amplitude']
            
            omega_d = 2 * np.pi * freq
            decay = np.exp(-zeta * omega_d * t)
            oscillation = A * decay * np.sin(omega_d * t)
            y += oscillation
        
        noise = noise_level * np.random.randn(len(t))
        y += noise
        
        return y
    
    def compute_welch_psd(self, signal_data, nperseg=None):
        """Compute power spectral density using Welch's method"""
        if nperseg is None:
            nperseg = min(len(signal_data) // 8, 1024)
        
        f, Pxx = signal.welch(
            signal_data,
            fs=self.fs,
            window='hann',
            nperseg=nperseg,
            noverlap=nperseg // 2,
            scaling='density'
        )
        
        return f, Pxx
    
    def compute_h0_persistence_1d(self, frequencies, psd_values, freq_range=(0.1, 5.0)):
        """
        Compute H0 persistence on 1D PSD function.
        """
        if not GUDHI_AVAILABLE:
            return np.array([]), np.array([])
        
        # Filter to frequency range
        freq_mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
        f_filtered = frequencies[freq_mask]
        psd_filtered = psd_values[freq_mask]
        
        # Work with log PSD
        log_psd = np.log10(psd_filtered + 1e-10)
        
        # For finding MAXIMA: use superlevel sets by negating
        negated_function = -log_psd
        
        # Create 1D cubical complex
        cubical_complex = gudhi.CubicalComplex(
            dimensions=[len(negated_function)],
            top_dimensional_cells=negated_function
        )
        
        # Compute persistence
        persistence = cubical_complex.persistence()
        
        # Extract H0 features
        h0_pairs = []
        for dim, (birth, death) in persistence:
            if dim == 0 and death != float('inf'):
                h0_pairs.append([birth, death])
        
        h0_diagram = np.array(h0_pairs) if len(h0_pairs) > 0 else np.array([]).reshape(0, 2)
        
        return h0_diagram, f_filtered, psd_filtered
    
    def extract_peak_frequencies_h0(self, frequencies, psd_values, h0_diagram, threshold):
        """
        Extract peak frequencies from H0 persistence.
        
        Alternative approach: Find local maxima in the function directly,
        then match them to significant persistence features.
        """
        if len(h0_diagram) == 0:
            return np.array([])
        
        # Calculate persistences
        persistences = h0_diagram[:, 1] - h0_diagram[:, 0]
        significant_mask = persistences > threshold
        
        if not np.any(significant_mask):
            return np.array([])
        
        # Work with log PSD
        log_psd = np.log10(psd_values + 1e-10)
        
        # Find ALL local maxima in the function
        from scipy.signal import argrelextrema
        local_max_indices = argrelextrema(log_psd, np.greater)[0]
        
        if len(local_max_indices) == 0:
            return np.array([])
        
        # Get values at local maxima
        local_max_values = log_psd[local_max_indices]
        
        # Sort by prominence (value)
        sorted_indices = np.argsort(local_max_values)[::-1]  # Descending
        
        # Match top N significant features to top N maxima
        n_significant = np.sum(significant_mask)
        
        # Take the top n_significant maxima by value
        peak_indices = local_max_indices[sorted_indices[:n_significant]]
        peak_frequencies = frequencies[peak_indices]
        
        return np.sort(peak_frequencies)
    
    def find_spectral_peaks_classical(self, f, Pxx, height_percentile=85, n_top_peaks=5):
        """Classical peak detection for comparison"""
        from scipy.signal import find_peaks
        
        threshold = np.percentile(Pxx, height_percentile)
        peaks, properties = find_peaks(Pxx, height=threshold, distance=3)
        
        if len(peaks) == 0:
            return np.array([]), np.array([])
        
        peak_freqs = f[peaks]
        peak_powers = Pxx[peaks]
        
        if len(peaks) > n_top_peaks:
            top_indices = np.argsort(peak_powers)[-n_top_peaks:]
            peak_freqs = peak_freqs[top_indices]
            peak_powers = peak_powers[top_indices]
        
        return peak_freqs, peak_powers
    
    def analyze_signal(self, signal_data, freq_range=(0.1, 5.0), 
                  n_expected_modes=3, persistence_percentile=80):
        """
        Complete TDA pipeline using H0 method
        """
        print("\n" + "="*70)
        print("TDA PIPELINE (H0)")
        print("="*70)
        
        results = {}
        
        # Step 1: Welch PSD
        print("\n[1/4] Computing Welch PSD...")
        start = time.time()
        f, Pxx = self.compute_welch_psd(signal_data)
        results['spectral_time'] = time.time() - start
        print(f"  ✓ {results['spectral_time']:.3f}s")
        
        # Step 2: Classical peak detection (for comparison)
        print("\n[2/4] Classical peak detection...")
        peak_freqs, peak_powers = self.find_spectral_peaks_classical(
            f, Pxx, n_top_peaks=n_expected_modes+2
        )
        results['peak_freqs'] = peak_freqs
        results['peak_powers'] = peak_powers
        print(f"  ✓ Found {len(peak_freqs)} peaks: {peak_freqs}")
        
        # Step 3: H0 Persistence on 1D PSD
        print("\n[3/4] Computing H0 persistence on 1D PSD...")
        start = time.time()
        h0_diagram, f_filtered, psd_filtered = self.compute_h0_persistence_1d(
            f, Pxx, freq_range
        )
        results['tda_time'] = time.time() - start
        
        if len(h0_diagram) > 0:
            # Calculate threshold
            persistences = h0_diagram[:, 1] - h0_diagram[:, 0]
            threshold = np.percentile(persistences, persistence_percentile)
            n_significant = np.sum(persistences > threshold)
            
            print(f"  ✓ {len(h0_diagram)} H0 features, {n_significant} significant")
            print(f"  ✓ Threshold: {threshold:.4f}")
            
            # Step 4: Extract peak frequencies
            print("\n[4/4] Extracting peak frequencies from H0...")
            tda_peak_freqs = self.extract_peak_frequencies_h0(
                f_filtered, psd_filtered, h0_diagram, threshold
            )
            
            print(f"  ✓ TDA detected peaks at: {tda_peak_freqs} Hz")
        else:
            threshold = 0
            n_significant = 0
            tda_peak_freqs = np.array([])
            print("  ✗ No H0 features found")
        
        # Package results
        results.update({
            'frequency': f,
            'psd': Pxx,
            'h0_diagram': h0_diagram,
            'threshold': threshold,
            'n_significant': n_significant,
            'tda_peak_freqs': tda_peak_freqs,
            'total_time': results['spectral_time'] + results['tda_time']
        })
        
        print(f"\n{'='*70}")
        print("RESULTS SUMMARY")
        print(f"{'='*70}")
        print(f"Classical peaks: {results['peak_freqs']}")
        print(f"TDA (H0) peaks:  {results['tda_peak_freqs']}")
        print(f"Total time: {results['total_time']:.3f}s")
        
        return results
    
    # def plot_results(self, signal_data, results, true_modes=None):
    #     """Visualize H0 TDA results"""
    #     fig = plt.figure(figsize=(15, 8))
        
    #     # 1. Time series
    #     ax1 = plt.subplot(2, 3, 1)
    #     t = np.arange(len(signal_data)) / self.fs
    #     ax1.plot(t, signal_data, 'b-', linewidth=0.5, alpha=0.7)
    #     ax1.set_xlabel('Time (s)')
    #     ax1.set_ylabel('Amplitude')
    #     ax1.set_title('Synchrophasor Signal')
    #     ax1.grid(True, alpha=0.3)
        
    #     # 2. PSD with peaks
    #     ax2 = plt.subplot(2, 3, 2)
    #     f = results['frequency']
    #     Pxx = results['psd']
        
    #     ax2.semilogy(f, Pxx, 'b-', linewidth=1.5, label='PSD')
        
    #     if 'peak_freqs' in results and len(results['peak_freqs']) > 0:
    #         ax2.plot(results['peak_freqs'], results['peak_powers'], 
    #                 'ro', markersize=10, label='Classical peaks')
        
    #     if 'tda_peak_freqs' in results and len(results['tda_peak_freqs']) > 0:
    #         # Get PSD values at TDA peaks
    #         tda_powers = []
    #         for freq in results['tda_peak_freqs']:
    #             idx = np.argmin(np.abs(f - freq))
    #             tda_powers.append(Pxx[idx])
    #         ax2.plot(results['tda_peak_freqs'], tda_powers, 
    #                 'g*', markersize=15, label='TDA (H0) peaks')
        
    #     if true_modes is not None:
    #         true_freqs = [m['freq'] for m in true_modes]
    #         ax2.axvline(x=true_freqs[0], color='k', linestyle='--', alpha=0.3)
    #         for freq in true_freqs[1:]:
    #             ax2.axvline(x=freq, color='k', linestyle='--', alpha=0.3)
        
    #     ax2.set_xlabel('Frequency (Hz)')
    #     ax2.set_ylabel('PSD')
    #     ax2.set_title('Power Spectral Density')
    #     ax2.set_xlim([0, 5])
    #     ax2.grid(True, alpha=0.3)
    #     ax2.legend(fontsize=8)
        
    #     # 3. Log PSD (what we actually analyze)
    #     ax3 = plt.subplot(2, 3, 3)
    #     log_psd = np.log10(Pxx + 1e-10)
    #     ax3.plot(f, log_psd, 'b-', linewidth=1.5)
    #     ax3.set_xlabel('Frequency (Hz)')
    #     ax3.set_ylabel('log₁₀(PSD)')
    #     ax3.set_title('Log PSD (1D Function for H0)')
    #     ax3.set_xlim([0, 5])
    #     ax3.grid(True, alpha=0.3)
        
    #     # 4. H0 Persistence Diagram
    #     ax4 = plt.subplot(2, 3, 4)
    #     if len(results['h0_diagram']) > 0:
    #         h0 = results['h0_diagram']
    #         persistences = h0[:, 1] - h0[:, 0]
    #         threshold = results['threshold']
    #         significant_mask = persistences > threshold
            
    #         ax4.scatter(h0[~significant_mask, 0], h0[~significant_mask, 1],
    #                    c='blue', s=50, alpha=0.4, label='Noise')
            
    #         if np.any(significant_mask):
    #             ax4.scatter(h0[significant_mask, 0], h0[significant_mask, 1],
    #                        c='red', s=150, marker='*',
    #                        label=f'Significant ({np.sum(significant_mask)})')
            
    #         max_val = np.max(h0)
    #         ax4.plot([np.min(h0), max_val], [np.min(h0), max_val], 'k--', alpha=0.3)
    #         ax4.legend(loc='lower right')
    #         ax4.set_xlabel('Birth')
    #         ax4.set_ylabel('Death')
    #         ax4.set_title(f'H₀ Persistence Diagram ({len(h0)} features)')
    #     else:
    #         ax4.text(0.5, 0.5, 'No H0 features',
    #                 ha='center', va='center', transform=ax4.transAxes)
    #     ax4.grid(True, alpha=0.3)
        
    #     # 5. Comparison bar chart
    #     ax5 = plt.subplot(2, 3, 5)
    #     if true_modes is not None:
    #         true_freqs = np.array([m['freq'] for m in true_modes])
    #         classical_freqs = results.get('peak_freqs', np.array([]))
    #         tda_freqs = results.get('tda_peak_freqs', np.array([]))
            
    #         x = np.arange(len(true_freqs))
    #         width = 0.25
            
    #         ax5.bar(x - width, true_freqs, width, label='Ground Truth', color='black', alpha=0.5)
            
    #         if len(classical_freqs) >= len(true_freqs):
    #             ax5.bar(x, classical_freqs[:len(true_freqs)], width, label='Classical', color='red', alpha=0.7)
            
    #         if len(tda_freqs) >= len(true_freqs):
    #             ax5.bar(x + width, tda_freqs[:len(true_freqs)], width, label='TDA (H0)', color='green', alpha=0.7)
            
    #         ax5.set_ylabel('Frequency (Hz)')
    #         ax5.set_title('Mode Detection Comparison')
    #         ax5.set_xticks(x)
    #         ax5.set_xticklabels([f'Mode {i+1}' for i in range(len(true_freqs))])
    #         ax5.legend()
    #         ax5.grid(True, alpha=0.3, axis='y')
        
    #     # 6. Timing
    #     ax6 = plt.subplot(2, 3, 6)
    #     stages = ['Spectral\n(Welch)', 'TDA\n(H0)']
    #     times = [results.get('spectral_time', 0), results.get('tda_time', 0)]
    #     ax6.bar(stages, times, color=['blue', 'green'], alpha=0.7)
    #     ax6.set_ylabel('Time (seconds)')
    #     ax6.set_title('Pipeline Timing')
    #     ax6.grid(True, alpha=0.3, axis='y')
        
    #     plt.suptitle('TDA Analysis using H0 Persistence (Correct Method)', 
    #                 fontsize=14, fontweight='bold')
    #     plt.tight_layout()
        
    #     return fig


# def test_h0_method():
#     """Test the CORRECT H0 method"""
#     print("\n" + "="*70)
#     print("TESTING H0 METHOD (What Mishra & Vanfretti Actually Do)")
#     print("="*70)
    
#     # Generate test signal
#     modes = [
#         {'freq': 0.5, 'damping': 0.03, 'amplitude': 0.25},
#         {'freq': 1.2, 'damping': 0.03, 'amplitude': 0.25},
#         {'freq': 2.8, 'damping': 0.03, 'amplitude': 0.20},
#     ]
    
#     duration = 120.0
#     fs = 30.0
#     t = np.arange(0, duration, 1/fs)
    
#     print(f"\nTest signal: {duration}s, {len(modes)} modes")
#     for i, mode in enumerate(modes, 1):
#         print(f"  Mode {i}: {mode['freq']:.1f} Hz, damping={mode['damping']:.0%}")
    
#     # Create analyzer
#     analyzer = TDAAnalyzer(method='classical')
#     signal_data = analyzer.generate_synchrophasor_signal(t, modes, noise_level=0.02)
    
#     # Analyze
#     results = analyzer.analyze_signal(signal_data, n_expected_modes=len(modes))
    
#     # Plot
#     fig = analyzer.plot_results(signal_data, results, true_modes=modes)
#     plt.savefig('h0_tda_correct_method.png', dpi=150, bbox_inches='tight')
#     print("\n✓ Saved: h0_tda_correct_method.png")
    
#     plt.show()
    
#     return results


# if __name__ == "__main__":
#     results = test_h0_method()
#     print("\n" + "="*70)
#     print("H0 TDA METHOD TEST COMPLETE!")
#     print("="*70)