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
    
    def normalize_to_probability(self, vec):
        """Normalize vector to probability distribution"""
        vec_positive = np.abs(vec)  # Ensure non-negative
        total = np.sum(vec_positive)
        return vec_positive / total if total > 0 else vec_positive

    def compute_trace_distance_classical(self, vec1, vec2):
        """
        Classical trace distance for diagonal density matrices.
        D(ρ₁, ρ₂) = ½ Σᵢ |p₁ᵢ - p₂ᵢ|
        """
        # Normalize to probability distributions
        p1 = self.normalize_to_probability(vec1)
        p2 = self.normalize_to_probability(vec2)
        
        # Pad to same length
        max_len = max(len(p1), len(p2))
        p1_padded = np.pad(p1, (0, max_len - len(p1)))
        p2_padded = np.pad(p2, (0, max_len - len(p2)))
        
        # L1 distance
        trace_dist = 0.5 * np.sum(np.abs(p1_padded - p2_padded))
        
        return trace_dist

    def create_trace_distance_circuit(self, prob_dist1, prob_dist2):
        """
        Quantum circuit to estimate trace distance.
        Uses amplitude encoding and measurement statistics.
        """
        n_qubits = int(np.ceil(np.log2(len(prob_dist1))))
        n_qubits = max(n_qubits, 1)
        padded_size = 2 ** n_qubits
        
        # Pad probability distributions
        p1 = np.pad(prob_dist1, (0, padded_size - len(prob_dist1)))
        p2 = np.pad(prob_dist2, (0, padded_size - len(prob_dist2)))
        
        # Create amplitude-encoded states
        # |ψ₁⟩ = Σᵢ √pᵢ|i⟩
        state1 = np.sqrt(p1)
        state2 = np.sqrt(p2)
        
        # Normalize
        state1 = state1 / np.linalg.norm(state1)
        state2 = state2 / np.linalg.norm(state2)
        
        # Create circuit with two registers
        qr1 = QuantumRegister(n_qubits, 'state1')
        qr2 = QuantumRegister(n_qubits, 'state2')
        cr = ClassicalRegister(n_qubits, 'result')
        
        qc = QuantumCircuit(qr1, qr2, cr)
        
        # Initialize states
        qc.initialize(state1, qr1)
        qc.initialize(state2, qr2)
        
        # Measure in computational basis
        qc.measure(qr1, cr)
        
        return qc, p1, p2

    def compute_trace_distance_quantum(self, vec1, vec2):
        """
        Quantum estimation of trace distance using amplitude encoding.
        Estimates via sampling from probability distributions.
        """
        # Normalize to probabilities
        p1 = self.normalize_to_probability(vec1)
        p2 = self.normalize_to_probability(vec2)
        
        # Pad to same length and power of 2
        max_len = max(len(p1), len(p2))
        padded_size = 2 ** int(np.ceil(np.log2(max_len)))
        p1_padded = np.pad(p1, (0, padded_size - len(p1)))
        p2_padded = np.pad(p2, (0, padded_size - len(p2)))
        
        # Create and run circuit
        qc, p1_final, p2_final = self.create_trace_distance_circuit(p1_padded, p2_padded)
        
        job = self.simulator.run(qc, shots=self.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Estimate probability distributions from measurements
        p1_estimated = np.zeros(padded_size)
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            p1_estimated[idx] = count / self.shots
        
        # For trace distance, we need both distributions
        # Run second circuit for p2
        qr = QuantumRegister(int(np.log2(padded_size)), 'state2')
        cr = ClassicalRegister(int(np.log2(padded_size)), 'result')
        qc2 = QuantumCircuit(qr, cr)
        state2 = np.sqrt(p2_final) / np.linalg.norm(np.sqrt(p2_final))
        qc2.initialize(state2, qr)
        qc2.measure(qr, cr)
        
        job2 = self.simulator.run(qc2, shots=self.shots)
        result2 = job2.result()
        counts2 = result2.get_counts()
        
        p2_estimated = np.zeros(padded_size)
        for bitstring, count in counts2.items():
            idx = int(bitstring, 2)
            p2_estimated[idx] = count / self.shots
        
        # Compute trace distance from estimated distributions
        trace_dist = 0.5 * np.sum(np.abs(p1_estimated - p2_estimated))
        
        return trace_dist

    def compute_distance(self, vec1, vec2):
        """Enhanced compute_distance with trace distance options"""
        if self.method == 'classical':
            return np.linalg.norm(vec1 - vec2)
        
        elif self.method == 'trace_distance_classical':
            return self.compute_trace_distance_classical(vec1, vec2)
        
        elif self.method == 'trace_distance_quantum':
            return self.compute_trace_distance_quantum(vec1, vec2)
        
        elif self.method == 'swap_test':
            # Original swap test code
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