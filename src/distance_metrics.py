"""
Distance Metrics for Quantum-Enhanced TDA

Four distance computation methods:
1. Euclidean (L2) - classical
2. Swap Test (L2) - quantum  
3. Trace Distance Classical (L1) - classical
4. Trace Distance Quantum (L1 + sampling noise) - quantum-inspired

Note on "Quantum Trace Distance":
This is NOT a quantum algorithm for trace distance. It encodes distributions
as quantum states, samples from them (adding quantum measurement noise), then
computes classical trace distance on the noisy estimates. This explores the
effect of quantum sampling noise, not quantum speedup.
"""

import numpy as np
from typing import Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator

from config import QuantumConfig


class DistanceComputer:
    """
    Compute pairwise distances between PSD vectors using various methods.
    """
    
    def __init__(self, method: str = 'classical', config: QuantumConfig = None):
        """
        Parameters
        ----------
        method : str
            One of: 'classical', 'swap_test', 'trace_distance_classical', 
                    'trace_distance_quantum'
        config : QuantumConfig
            Quantum configuration (shots, etc.)
        """
        self.method = method
        self.config = config or QuantumConfig()
        self.simulator = AerSimulator()
        
    def compute_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute distance between two vectors."""
        if self.method == 'classical':
            return self._euclidean_distance(vec1, vec2)
        elif self.method == 'swap_test':
            return self._swap_test_distance(vec1, vec2)
        elif self.method == 'trace_distance_classical':
            return self._trace_distance_classical(vec1, vec2)
        elif self.method == 'trace_distance_quantum':
            return self._trace_distance_quantum(vec1, vec2)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def compute_distance_matrix(self, vectors: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Parameters
        ----------
        vectors : np.ndarray
            Shape (n_vectors, dim)
            
        Returns
        -------
        distances : np.ndarray
            Shape (n_vectors, n_vectors), symmetric
        """
        n = len(vectors)
        D = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                d = self.compute_distance(vectors[i], vectors[j])
                D[i, j] = d
                D[j, i] = d
        
        return D
    
    # --- Distance implementations ---
    
    def _euclidean_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Standard L2 distance."""
        return np.linalg.norm(vec1 - vec2)
    
    def _trace_distance_classical(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Classical trace distance for probability distributions.
        D(p, q) = (1/2) Σ |p_i - q_i|
        
        Bounded in [0, 1].
        """
        p1 = self._to_probability(vec1)
        p2 = self._to_probability(vec2)
        
        # Pad to same length
        max_len = max(len(p1), len(p2))
        p1 = np.pad(p1, (0, max_len - len(p1)))
        p2 = np.pad(p2, (0, max_len - len(p2)))
        
        return 0.5 * np.sum(np.abs(p1 - p2))
    
    def _trace_distance_quantum(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        "Quantum" trace distance: encode as quantum states, sample, then compute
        classical trace distance on empirical distributions.
        
        This adds quantum measurement noise to the estimate.
        """
        p1 = self._to_probability(vec1)
        p2 = self._to_probability(vec2)
        
        # Pad to power of 2
        max_len = max(len(p1), len(p2))
        n_qubits = int(np.ceil(np.log2(max(max_len, 2))))
        padded_size = 2 ** n_qubits
        
        p1 = np.pad(p1, (0, padded_size - len(p1)))
        p2 = np.pad(p2, (0, padded_size - len(p2)))
        
        # Renormalize after padding
        p1 = p1 / np.sum(p1)
        p2 = p2 / np.sum(p2)
        
        # Sample from quantum-encoded distributions
        p1_est = self._quantum_sample(p1, n_qubits)
        p2_est = self._quantum_sample(p2, n_qubits)
        
        # Classical trace distance on estimates
        trace_dist = 0.5 * np.sum(np.abs(p1_est - p2_est))
        return np.clip(trace_dist, 0.0, 1.0)
    
    def _swap_test_distance(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Quantum swap test to estimate |⟨ψ₁|ψ₂⟩|².
        
        Returns Euclidean-like distance: √(||v1||² + ||v2||² - 2||v1||||v2||⟨ψ₁|ψ₂⟩)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        state1 = vec1 / norm1
        state2 = vec2 / norm2
        
        # Build and run swap test circuit
        qc = self._build_swap_test_circuit(state1, state2)
        job = self.simulator.run(qc, shots=self.config.shots)
        result = job.result()
        counts = result.get_counts()
        
        # P(0) = (1 + |⟨ψ₁|ψ₂⟩|²) / 2
        prob_0 = counts.get('0', 0) / self.config.shots
        inner_product_squared = max(0, 2 * prob_0 - 1)
        inner_product = np.sqrt(inner_product_squared)
        
        # Euclidean distance formula
        dist_squared = norm1**2 + norm2**2 - 2 * norm1 * norm2 * inner_product
        return np.sqrt(max(0, dist_squared))
    
    # --- Helper methods ---
    
    def _to_probability(self, vec: np.ndarray) -> np.ndarray:
        """Normalize vector to probability distribution."""
        vec_pos = np.abs(vec)
        total = np.sum(vec_pos)
        if total == 0:
            return vec_pos
        return vec_pos / total
    
    def _quantum_sample(self, prob_dist: np.ndarray, n_qubits: int) -> np.ndarray:
        """
        Encode probability distribution as quantum state and sample.
        
        |ψ⟩ = Σ √p_k |k⟩
        Measurement yields outcome k with probability p_k.
        """
        # Amplitude encoding
        amplitudes = np.sqrt(prob_dist)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)  # Normalize
        
        # Build circuit
        qr = QuantumRegister(n_qubits, 'q')
        cr = ClassicalRegister(n_qubits, 'c')
        qc = QuantumCircuit(qr, cr)
        
        qc.initialize(amplitudes, qr)
        qc.measure(qr, cr)
        
        # Run and collect statistics
        job = self.simulator.run(qc, shots=self.config.shots)
        result = job.result()
        counts = result.get_counts()
        
        # Reconstruct empirical distribution
        p_estimated = np.zeros(len(prob_dist))
        for bitstring, count in counts.items():
            idx = int(bitstring, 2)
            if idx < len(p_estimated):
                p_estimated[idx] = count / self.config.shots
        
        return p_estimated
    
    def _build_swap_test_circuit(self, state1: np.ndarray, 
                                  state2: np.ndarray) -> QuantumCircuit:
        """Build swap test circuit for two states."""
        n_qubits = int(np.ceil(np.log2(max(len(state1), len(state2)))))
        n_qubits = max(n_qubits, 1)
        padded_size = 2 ** n_qubits
        
        # Pad and normalize
        s1 = np.pad(state1, (0, padded_size - len(state1)))
        s2 = np.pad(state2, (0, padded_size - len(state2)))
        s1 = s1 / np.linalg.norm(s1)
        s2 = s2 / np.linalg.norm(s2)
        
        # Create circuit
        qr_ancilla = QuantumRegister(1, 'ancilla')
        qr_state1 = QuantumRegister(n_qubits, 'state1')
        qr_state2 = QuantumRegister(n_qubits, 'state2')
        cr = ClassicalRegister(1, 'result')
        
        qc = QuantumCircuit(qr_ancilla, qr_state1, qr_state2, cr)
        
        # Initialize states
        qc.initialize(s1, qr_state1)
        qc.initialize(s2, qr_state2)
        
        # Swap test
        qc.h(qr_ancilla[0])
        for i in range(n_qubits):
            qc.cswap(qr_ancilla[0], qr_state1[i], qr_state2[i])
        qc.h(qr_ancilla[0])
        
        qc.measure(qr_ancilla[0], cr[0])
        
        return qc
