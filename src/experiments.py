"""
Experiments for Quantum-Enhanced TDA

Systematic comparison of distance metrics across different drift scenarios.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

from config import (ExperimentConfig, SignalConfig, 
                    make_strong_mode_drift_config,
                    make_medium_mode_drift_config, 
                    make_weak_mode_drift_config)
from signal_generation import generate_signal, get_mode_info
from psd_analysis import segment_signal, compute_psd_matrix, normalize_psd_for_distance
from distance_metrics import DistanceComputer
from tda_analysis import count_modes_per_segment
from regime_detection import full_detection_analysis, DetectionResult


@dataclass
class TrialResult:
    """Results from a single trial."""
    trial_idx: int
    method: str
    seed: int
    
    # Core results
    distance_matrix: np.ndarray
    segment_times: np.ndarray
    detection: DetectionResult
    
    # TDA results
    mode_counts: np.ndarray
    
    # Timing
    elapsed_time: float


@dataclass 
class ExperimentResults:
    """Results from a complete experiment (multiple trials, multiple methods)."""
    config: ExperimentConfig
    
    # Results organized by method
    results_by_method: Dict[str, List[TrialResult]] = field(default_factory=dict)
    
    def add_result(self, result: TrialResult):
        if result.method not in self.results_by_method:
            self.results_by_method[result.method] = []
        self.results_by_method[result.method].append(result)
    
    def get_detection_rates(self) -> Dict[str, Dict[str, float]]:
        """
        Compute detection rates for each method.
        
        Returns dict of method -> {local_max_rate, jump_rate, mean_percentile}
        """
        rates = {}
        for method, trials in self.results_by_method.items():
            metrics = [t.detection.metrics_at_transition for t in trials 
                       if t.detection.metrics_at_transition is not None]
            
            if not metrics:
                continue
                
            rates[method] = {
                'local_max_rate': np.mean([m['is_local_max'] for m in metrics]) * 100,
                'jump_rate': np.mean([m['jump_detected'] for m in metrics]) * 100,
                'mean_percentile': np.mean([m['drift_percentile'] for m in metrics]),
                'mean_rate_of_change': np.mean([m['rate_of_change'] for m in metrics]),
            }
        
        return rates


def run_single_trial(config: ExperimentConfig, 
                     method: str,
                     trial_idx: int,
                     seed: int) -> TrialResult:
    """
    Run a single trial with one method.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    method : str
        Distance method to use
    trial_idx : int
        Trial index
    seed : int
        Random seed for this trial
        
    Returns
    -------
    result : TrialResult
    """
    start_time = time.time()
    
    # Generate signal
    signal_data = generate_signal(config.signal, seed=seed)
    
    # Segment
    segments, segment_times = segment_signal(
        signal_data, 
        config.signal.fs,
        config.segmentation
    )
    
    # Compute PSDs
    frequencies, psd_matrix = compute_psd_matrix(
        segments, 
        config.signal.fs,
        config.psd
    )
    
    # Normalize for distance computation
    if method in ['trace_distance_classical', 'trace_distance_quantum']:
        psd_normalized = normalize_psd_for_distance(psd_matrix, method='probability')
    else:
        psd_normalized = normalize_psd_for_distance(psd_matrix, method='log_zscore')
    
    # Compute distances
    computer = DistanceComputer(method=method, config=config.quantum)
    distance_matrix = computer.compute_distance_matrix(psd_normalized)
    
    # Regime detection
    detection = full_detection_analysis(
        distance_matrix,
        segment_times,
        true_transition_time=config.signal.transition_time
    )
    
    # TDA mode counting
    mode_counts = count_modes_per_segment(psd_matrix, frequencies, config.tda)
    
    elapsed = time.time() - start_time
    
    return TrialResult(
        trial_idx=trial_idx,
        method=method,
        seed=seed,
        distance_matrix=distance_matrix,
        segment_times=segment_times,
        detection=detection,
        mode_counts=mode_counts,
        elapsed_time=elapsed
    )


def run_experiment(config: ExperimentConfig, 
                   verbose: bool = True) -> ExperimentResults:
    """
    Run complete experiment with all methods and trials.
    
    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    verbose : bool
        Print progress
        
    Returns
    -------
    results : ExperimentResults
    """
    results = ExperimentResults(config=config)
    
    total_runs = config.n_trials * len(config.methods)
    run_count = 0
    
    if verbose:
        drifting_mode = config.signal.modes[config.signal.drifting_mode_index]
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {drifting_mode.freq} Hz mode drifting")
        print(f"{'='*60}")
        print(f"Trials: {config.n_trials}, Methods: {len(config.methods)}")
        print(f"Transition at t={config.signal.transition_time}s")
    
    for trial_idx in range(config.n_trials):
        seed = config.seed_base + trial_idx
        
        for method in config.methods:
            run_count += 1
            if verbose:
                print(f"\r  [{run_count}/{total_runs}] Trial {trial_idx+1}, {method}...", 
                      end='', flush=True)
            
            result = run_single_trial(config, method, trial_idx, seed)
            results.add_result(result)
    
    if verbose:
        print("\n  Done!")
    
    return results


def run_three_mode_drift_study(n_trials: int = 20,
                               seed_base: int = 42,
                               verbose: bool = True) -> Dict[str, ExperimentResults]:
    """
    Run the complete three-mode-drift study.
    
    Compares detection performance when:
    1. Strong mode (0.5 Hz) drifts
    2. Medium mode (1.2 Hz) drifts  
    3. Weak mode (2.5 Hz) drifts
    
    Returns
    -------
    all_results : Dict[str, ExperimentResults]
        Keys: 'strong', 'medium', 'weak'
    """
    all_results = {}
    
    configs = {
        'strong': make_strong_mode_drift_config(),
        'medium': make_medium_mode_drift_config(),
        'weak': make_weak_mode_drift_config(),
    }
    
    for name, config in configs.items():
        config.n_trials = n_trials
        config.seed_base = seed_base
        
        if verbose:
            print(f"\n{'#'*60}")
            print(f"  MODE DRIFT STUDY: {name.upper()}")
            print(f"{'#'*60}")
        
        all_results[name] = run_experiment(config, verbose=verbose)
    
    return all_results


def summarize_study(all_results: Dict[str, ExperimentResults]) -> None:
    """Print summary of three-mode-drift study."""
    print("\n" + "="*70)
    print("THREE-MODE-DRIFT STUDY SUMMARY")
    print("="*70)
    
    for drift_type, results in all_results.items():
        config = results.config
        drifting_mode = config.signal.modes[config.signal.drifting_mode_index]
        
        print(f"\n{drift_type.upper()} MODE DRIFT ({drifting_mode.freq} Hz, amplitude={drifting_mode.amplitude}):")
        print("-" * 50)
        
        rates = results.get_detection_rates()
        
        for method in config.methods:
            display_name = config.method_display_names.get(method, method)
            if method in rates:
                r = rates[method]
                print(f"  {display_name:30s}")
                print(f"    Local max at transition: {r['local_max_rate']:5.1f}%")
                print(f"    Jump detected:           {r['jump_rate']:5.1f}%")
                print(f"    Mean drift percentile:   {r['mean_percentile']:5.1f}%")
