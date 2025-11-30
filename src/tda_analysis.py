"""
TDA Analysis for Quantum-Enhanced Pipeline

H0 persistence on 1D PSD functions for mode detection.
Following Mishra & Vanfretti (2025) methodology.
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.signal import argrelextrema

from config import TDAConfig, PSDConfig

# Check for gudhi
try:
    import gudhi
    GUDHI_AVAILABLE = True
except ImportError:
    GUDHI_AVAILABLE = False
    print("Warning: gudhi not installed. TDA features disabled.")


def compute_h0_persistence(frequencies: np.ndarray, 
                           psd_values: np.ndarray) -> np.ndarray:
    """
    Compute H0 persistence diagram for 1D PSD function.
    
    Uses superlevel set filtration (via negation) to find peaks as
    persistent connected components.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency bins
    psd_values : np.ndarray
        PSD values (will use log scale)
        
    Returns
    -------
    h0_diagram : np.ndarray
        Shape (n_features, 2) with (birth, death) pairs
        Empty array if no features found
    """
    if not GUDHI_AVAILABLE:
        return np.array([]).reshape(0, 2)
    
    # Work in log scale
    log_psd = np.log10(psd_values + 1e-10)
    
    # For finding MAXIMA: negate (superlevel sets become sublevel sets)
    negated = -log_psd
    
    # 1D cubical complex
    cubical = gudhi.CubicalComplex(
        dimensions=[len(negated)],
        top_dimensional_cells=negated
    )
    
    persistence = cubical.persistence()
    
    # Extract H0 features (exclude infinite death)
    h0_pairs = []
    for dim, (birth, death) in persistence:
        if dim == 0 and death != float('inf'):
            h0_pairs.append([birth, death])
    
    if len(h0_pairs) == 0:
        return np.array([]).reshape(0, 2)
    
    return np.array(h0_pairs)


def extract_significant_modes(frequencies: np.ndarray,
                              psd_values: np.ndarray,
                              h0_diagram: np.ndarray,
                              config: TDAConfig) -> np.ndarray:
    """
    Extract significant mode frequencies from H0 persistence.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency bins
    psd_values : np.ndarray  
        PSD values
    h0_diagram : np.ndarray
        H0 persistence diagram
    config : TDAConfig
        Configuration with threshold percentile
        
    Returns
    -------
    mode_frequencies : np.ndarray
        Detected mode frequencies (Hz), sorted
    """
    if len(h0_diagram) == 0:
        return np.array([])
    
    # Calculate persistences
    persistences = h0_diagram[:, 1] - h0_diagram[:, 0]
    threshold = np.percentile(persistences, config.persistence_threshold_percentile)
    
    n_significant = np.sum(persistences > threshold)
    if n_significant == 0:
        return np.array([])
    
    # Find local maxima in log PSD
    log_psd = np.log10(psd_values + 1e-10)
    local_max_indices = argrelextrema(log_psd, np.greater)[0]
    
    if len(local_max_indices) == 0:
        return np.array([])
    
    # Sort by value (most prominent first)
    local_max_values = log_psd[local_max_indices]
    sorted_indices = np.argsort(local_max_values)[::-1]
    
    # Take top n_significant maxima
    peak_indices = local_max_indices[sorted_indices[:n_significant]]
    mode_frequencies = frequencies[peak_indices]
    
    return np.sort(mode_frequencies)


def count_modes_per_segment(psd_matrix: np.ndarray,
                            frequencies: np.ndarray,
                            config: TDAConfig) -> np.ndarray:
    """
    Count detected modes for each segment.
    
    Parameters
    ----------
    psd_matrix : np.ndarray
        Shape (n_segments, n_freq_bins)
    frequencies : np.ndarray
        Frequency bins
    config : TDAConfig
        TDA configuration
        
    Returns
    -------
    mode_counts : np.ndarray
        Number of modes detected in each segment
    """
    n_segments = len(psd_matrix)
    mode_counts = np.zeros(n_segments, dtype=int)
    
    for i, psd in enumerate(psd_matrix):
        h0 = compute_h0_persistence(frequencies, psd)
        modes = extract_significant_modes(frequencies, psd, h0, config)
        mode_counts[i] = len(modes)
    
    return mode_counts


def get_modes_per_segment(psd_matrix: np.ndarray,
                          frequencies: np.ndarray,
                          config: TDAConfig) -> List[np.ndarray]:
    """
    Get detected mode frequencies for each segment.
    
    Returns
    -------
    modes_list : List[np.ndarray]
        List of mode frequency arrays, one per segment
    """
    modes_list = []
    
    for psd in psd_matrix:
        h0 = compute_h0_persistence(frequencies, psd)
        modes = extract_significant_modes(frequencies, psd, h0, config)
        modes_list.append(modes)
    
    return modes_list
