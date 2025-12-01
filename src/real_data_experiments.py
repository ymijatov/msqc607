"""
Real Data Experiments for Quantum-Enhanced TDA

Adapter to run the existing pipeline on real GESL synchrophasor data.
No ground truth transition time - we're doing pure detection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from real_data import load_gesl_signal, GESLSignalInfo
from config import PSDConfig, SegmentationConfig, QuantumConfig, TDAConfig
from psd_analysis import segment_signal, compute_psd_matrix, normalize_psd_for_distance
from distance_metrics import DistanceComputer
from tda_analysis import count_modes_per_segment, get_modes_per_segment
from regime_detection import (
    compute_sequential_distances, 
    compute_cumulative_drift,
    detect_regime_changes
)


@dataclass
class RealDataConfig:
    """Configuration for real data experiments."""
    # File paths
    signal_path: str
    metadata_path: Optional[str] = None
    
    # PMU selection
    pmu_id: str = 'P001'
    measurement: str = 'vp_m'
    
    # Processing
    detrend: bool = True
    normalize: bool = True
    
    # PSD and segmentation (reuse existing configs)
    psd: PSDConfig = field(default_factory=PSDConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    
    # Distance methods to compare
    methods: List[str] = field(default_factory=lambda: [
        'classical',
        'swap_test', 
        'trace_distance_classical',
        'trace_distance_quantum',
    ])
    
    method_display_names: Dict[str, str] = field(default_factory=lambda: {
        'classical': 'Euclidean (L2)',
        'swap_test': 'Swap Test (L2)',
        'trace_distance_classical': 'Trace Classical (L1)',
        'trace_distance_quantum': 'Trace Quantum (L1+noise)',
    })


@dataclass
class RealDataResult:
    """Results from analyzing real data with one method."""
    method: str
    
    # Signal info
    signal_info: Optional[GESLSignalInfo]
    pmu_id: str
    
    # Time/frequency info
    segment_times: np.ndarray
    frequencies: np.ndarray
    
    # Distance analysis
    distance_matrix: np.ndarray
    sequential_distances: np.ndarray
    cumulative_drift: np.ndarray
    regime_change_indices: np.ndarray
    
    # TDA mode detection
    mode_counts: np.ndarray
    modes_per_segment: List[np.ndarray]
    
    # Timing
    elapsed_time: float


@dataclass
class RealDataExperimentResults:
    """Results from full real data experiment (all methods)."""
    config: RealDataConfig
    signal_info: Optional[GESLSignalInfo]
    
    # Raw signal info
    signal_length: int
    signal_duration: float
    fs: float
    n_segments: int
    
    # Results by method
    results_by_method: Dict[str, RealDataResult] = field(default_factory=dict)
    
    def get_detected_transitions(self, method: str) -> np.ndarray:
        """Get transition times (in seconds) for a method."""
        result = self.results_by_method.get(method)
        if result is None:
            return np.array([])
        return result.segment_times[result.regime_change_indices]


def run_real_data_experiment(config: RealDataConfig, verbose: bool = True) -> RealDataExperimentResults:
    """
    Run the full pipeline on real GESL data.
    
    Parameters
    ----------
    config : RealDataConfig
        Configuration specifying file paths and parameters
    verbose : bool
        Print progress
        
    Returns
    -------
    results : RealDataExperimentResults
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"REAL DATA EXPERIMENT")
        print(f"{'='*60}")
        print(f"Signal: {config.signal_path}")
        print(f"PMU: {config.pmu_id}, Measurement: {config.measurement}")
    
    # Load signal
    t, signal, fs, info = load_gesl_signal(
        config.signal_path,
        config.metadata_path,
        pmu_id=config.pmu_id,
        measurement=config.measurement,
        detrend=config.detrend,
        normalize=config.normalize,
    )
    
    if verbose:
        print(f"Loaded: {len(signal)} samples, {t[-1]:.1f}s duration, {fs:.1f} Hz")
        if info and info.oscillation_frequencies:
            print(f"Known oscillation frequencies: {info.oscillation_frequencies}")
    
    # Segment signal
    segments, segment_times = segment_signal(signal, fs, config.segmentation)
    
    if verbose:
        print(f"Segmented into {len(segments)} segments of {config.segmentation.segment_duration}s each")
    
    # Compute PSDs
    frequencies, psd_matrix = compute_psd_matrix(segments, fs, config.psd)
    
    if verbose:
        print(f"PSD matrix shape: {psd_matrix.shape} (segments × frequency bins)")
        print(f"Frequency range: {frequencies[0]:.2f} - {frequencies[-1]:.2f} Hz")
    
    # Initialize results container
    results = RealDataExperimentResults(
        config=config,
        signal_info=info,
        signal_length=len(signal),
        signal_duration=t[-1],
        fs=fs,
        n_segments=len(segments),
    )
    
    # Run each distance method
    for method in config.methods:
        if verbose:
            print(f"\n  Processing: {method}...", end=' ', flush=True)
        
        start_time = time.time()
        
        # Normalize PSDs appropriately for distance type
        if method in ['trace_distance_classical', 'trace_distance_quantum']:
            psd_normalized = normalize_psd_for_distance(psd_matrix, method='probability')
        else:
            psd_normalized = normalize_psd_for_distance(psd_matrix, method='log_zscore')
        
        # Compute distance matrix
        computer = DistanceComputer(method=method, config=config.quantum)
        distance_matrix = computer.compute_distance_matrix(psd_normalized)
        
        # Regime detection (no ground truth)
        seq_dist = compute_sequential_distances(distance_matrix)
        cum_drift = compute_cumulative_drift(distance_matrix)
        regime_changes = detect_regime_changes(distance_matrix, threshold_sigma=1.0)
        
        # TDA mode detection
        mode_counts = count_modes_per_segment(psd_matrix, frequencies, config.tda)
        modes_per_segment = get_modes_per_segment(psd_matrix, frequencies, config.tda)
        
        elapsed = time.time() - start_time
        
        result = RealDataResult(
            method=method,
            signal_info=info,
            pmu_id=config.pmu_id,
            segment_times=segment_times,
            frequencies=frequencies,
            distance_matrix=distance_matrix,
            sequential_distances=seq_dist,
            cumulative_drift=cum_drift,
            regime_change_indices=regime_changes,
            mode_counts=mode_counts,
            modes_per_segment=modes_per_segment,
            elapsed_time=elapsed,
        )
        
        results.results_by_method[method] = result
        
        if verbose:
            n_changes = len(regime_changes)
            print(f"done ({elapsed:.2f}s). Detected {n_changes} regime changes.")
    
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        for method in config.methods:
            result = results.results_by_method[method]
            display_name = config.method_display_names.get(method, method)
            transitions = results.get_detected_transitions(method)
            
            print(f"\n{display_name}:")
            print(f"  Regime changes detected: {len(result.regime_change_indices)}")
            if len(transitions) > 0:
                print(f"  At times: {', '.join(f'{t:.1f}s' for t in transitions)}")
            
            # Mode detection summary
            mean_modes = np.mean(result.mode_counts)
            print(f"  Mean modes per segment: {mean_modes:.2f}")
    
    return results


def compare_with_known_frequencies(
    results: RealDataExperimentResults,
    method: str = 'trace_distance_classical',
) -> None:
    """
    Compare detected modes with known oscillation frequencies from metadata.
    """
    if results.signal_info is None:
        print("No metadata available for comparison.")
        return
    
    known_freqs = results.signal_info.oscillation_frequencies
    if not known_freqs:
        print("No known oscillation frequencies in metadata.")
        return
    
    result = results.results_by_method.get(method)
    if result is None:
        print(f"No results for method: {method}")
        return
    
    print(f"\n{'='*60}")
    print(f"MODE DETECTION vs KNOWN FREQUENCIES")
    print(f"{'='*60}")
    print(f"Known frequencies: {known_freqs}")
    print(f"\nDetected modes per segment (method: {method}):")
    
    for i, (t, modes) in enumerate(zip(result.segment_times, result.modes_per_segment)):
        if len(modes) > 0:
            modes_str = ', '.join(f'{m:.3f}' for m in modes)
        else:
            modes_str = 'none'
        print(f"  t={t:6.1f}s: {modes_str} Hz")
    
    # Check how well detected modes match known frequencies
    all_detected = np.concatenate(result.modes_per_segment) if any(len(m) > 0 for m in result.modes_per_segment) else np.array([])
    
    if len(all_detected) > 0:
        print(f"\nAll detected modes: {sorted(set(np.round(all_detected, 2)))}")
        
        # For each known frequency, find closest detected
        print("\nMatching:")
        for known in known_freqs:
            if len(all_detected) > 0:
                closest = all_detected[np.argmin(np.abs(all_detected - known))]
                error = abs(closest - known)
                print(f"  Known {known:.2f} Hz → Closest detected: {closest:.3f} Hz (error: {error:.3f} Hz)")


def quick_analysis(
    signal_path: str,
    metadata_path: str,
    pmu_id: str = 'P001',
    segment_duration: float = 15.0,
) -> RealDataExperimentResults:
    """
    Quick one-liner to analyze a GESL signal.
    
    Example:
        results = quick_analysis('sigId-1032.csv', 'sigId-1032-Metadata.csv')
    """
    config = RealDataConfig(
        signal_path=signal_path,
        metadata_path=metadata_path,
        pmu_id=pmu_id,
    )
    config.segmentation.segment_duration = segment_duration
    
    return run_real_data_experiment(config)


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) >= 3:
        results = quick_analysis(sys.argv[1], sys.argv[2])
        compare_with_known_frequencies(results)
    else:
        print("Usage: python real_data_experiments.py <signal.csv> <metadata.csv>")