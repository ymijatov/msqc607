"""
Regime Detection for Quantum-Enhanced TDA

Detect regime changes (transitions) from distance matrices.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class DetectionResult:
    """Results from regime detection analysis."""
    # Sequential distances (segment i to segment i+1)
    sequential_distances: np.ndarray
    
    # Cumulative drift from initial state
    cumulative_drift: np.ndarray
    
    # Detected regime change indices
    regime_change_indices: np.ndarray
    
    # Detection metrics at specific time
    metrics_at_transition: Dict = None


def compute_sequential_distances(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute segment-to-segment distances (consecutive pairs).
    
    Returns array of length n-1 where n = number of segments.
    """
    n = len(distance_matrix)
    return np.array([distance_matrix[i, i+1] for i in range(n-1)])


def compute_cumulative_drift(distance_matrix: np.ndarray) -> np.ndarray:
    """
    Compute distance of each segment from the initial segment.
    
    This captures how much the system has drifted from its starting state.
    """
    return distance_matrix[0, :]


def detect_regime_changes(distance_matrix: np.ndarray,
                          threshold_sigma: float = 1.0) -> np.ndarray:
    """
    Detect regime changes based on sequential distance jumps.
    
    A regime change is detected when the segment-to-segment distance
    exceeds mean + threshold_sigma * std.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distance matrix (n x n)
    threshold_sigma : float
        Number of standard deviations above mean for detection
        
    Returns
    -------
    change_indices : np.ndarray
        Indices where regime changes detected
    """
    seq_dist = compute_sequential_distances(distance_matrix)
    
    mean_dist = np.mean(seq_dist)
    std_dist = np.std(seq_dist)
    
    threshold = mean_dist + threshold_sigma * std_dist
    change_indices = np.where(seq_dist > threshold)[0]
    
    return change_indices


def analyze_detection_at_time(distance_matrix: np.ndarray,
                              segment_times: np.ndarray,
                              target_time: float) -> Dict:
    """
    Analyze detection metrics at a specific target time.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distances
    segment_times : np.ndarray
        Time (seconds) of each segment center
    target_time : float
        Time point to analyze (e.g., true transition time)
        
    Returns
    -------
    metrics : Dict
        Detection metrics at target time
    """
    # Find segment index closest to target time
    idx = np.argmin(np.abs(segment_times - target_time))
    
    seq_dist = compute_sequential_distances(distance_matrix)
    cum_drift = compute_cumulative_drift(distance_matrix)
    
    metrics = {
        'target_time': target_time,
        'closest_segment_idx': idx,
        'closest_segment_time': segment_times[idx],
    }
    
    # Cumulative drift at target
    metrics['cumulative_drift'] = cum_drift[idx]
    metrics['drift_percentile'] = (np.sum(cum_drift <= cum_drift[idx]) / len(cum_drift)) * 100
    
    # Is this a local maximum in drift?
    if 0 < idx < len(cum_drift) - 1:
        is_local_max = (cum_drift[idx] > cum_drift[idx-1] and 
                        cum_drift[idx] > cum_drift[idx+1])
    else:
        is_local_max = False
    metrics['is_local_max'] = is_local_max
    
    # Rate of change (derivative) at target
    if 0 < idx < len(cum_drift) - 1:
        dt = segment_times[idx+1] - segment_times[idx-1]
        rate = (cum_drift[idx+1] - cum_drift[idx-1]) / dt
    else:
        rate = 0.0
    metrics['rate_of_change'] = rate
    
    # Sequential distance jump at target
    if idx < len(seq_dist):
        metrics['sequential_distance'] = seq_dist[idx]
        
        mean_seq = np.mean(seq_dist)
        std_seq = np.std(seq_dist)
        metrics['jump_detected'] = seq_dist[idx] > (mean_seq + std_seq)
    else:
        metrics['sequential_distance'] = 0.0
        metrics['jump_detected'] = False
    
    return metrics


def full_detection_analysis(distance_matrix: np.ndarray,
                            segment_times: np.ndarray,
                            true_transition_time: float = None) -> DetectionResult:
    """
    Complete regime detection analysis.
    
    Parameters
    ----------
    distance_matrix : np.ndarray
        Pairwise distances
    segment_times : np.ndarray
        Segment center times
    true_transition_time : float, optional
        Ground truth transition time for evaluation
        
    Returns
    -------
    result : DetectionResult
        Complete detection results
    """
    seq_dist = compute_sequential_distances(distance_matrix)
    cum_drift = compute_cumulative_drift(distance_matrix)
    regime_changes = detect_regime_changes(distance_matrix)
    
    metrics = None
    if true_transition_time is not None:
        metrics = analyze_detection_at_time(
            distance_matrix, segment_times, true_transition_time
        )
    
    return DetectionResult(
        sequential_distances=seq_dist,
        cumulative_drift=cum_drift,
        regime_change_indices=regime_changes,
        metrics_at_transition=metrics
    )
