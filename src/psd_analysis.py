"""
PSD Analysis for Quantum-Enhanced TDA

Welch PSD computation and temporal segmentation.
"""

import numpy as np
from scipy import signal
from typing import Tuple, List
from config import SignalConfig, PSDConfig, SegmentationConfig


def compute_welch_psd(signal_data: np.ndarray, fs: float, 
                      config: PSDConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute power spectral density using Welch's method.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    config : PSDConfig
        PSD configuration
        
    Returns
    -------
    frequencies : np.ndarray
        Frequency bins (Hz)
    psd : np.ndarray
        Power spectral density
    """
    f, Pxx = signal.welch(
        signal_data,
        fs=fs,
        window='hann',
        nperseg=config.nperseg,
        noverlap=config.noverlap,
        scaling='density'
    )
    return f, Pxx


def segment_signal(signal_data: np.ndarray, fs: float,
                   config: SegmentationConfig) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Divide signal into (possibly overlapping) segments.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Input signal
    fs : float
        Sampling frequency
    config : SegmentationConfig
        Segmentation configuration
        
    Returns
    -------
    segments : List[np.ndarray]
        List of signal segments
    segment_times : np.ndarray
        Center time of each segment (seconds)
    """
    segment_samples = int(config.segment_duration * fs)
    hop = int(segment_samples * (1 - config.overlap_ratio))
    
    segments = []
    segment_times = []
    
    start = 0
    while start + segment_samples <= len(signal_data):
        segment = signal_data[start:start + segment_samples]
        center_time = (start + segment_samples / 2) / fs
        
        segments.append(segment)
        segment_times.append(center_time)
        
        start += hop
    
    return segments, np.array(segment_times)


def compute_psd_matrix(segments: List[np.ndarray], fs: float,
                       psd_config: PSDConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute PSD for each segment, returning matrix of PSD vectors.
    
    Parameters
    ----------
    segments : List[np.ndarray]
        Signal segments
    fs : float
        Sampling frequency
    psd_config : PSDConfig
        PSD configuration
        
    Returns
    -------
    frequencies : np.ndarray
        Frequency bins (within freq_range)
    psd_matrix : np.ndarray
        Shape (n_segments, n_freq_bins), each row is a PSD vector
    """
    psd_vectors = []
    frequencies = None
    
    for segment in segments:
        f, Pxx = compute_welch_psd(segment, fs, psd_config)
        
        # Filter to frequency range
        freq_mask = (f >= psd_config.freq_range[0]) & (f <= psd_config.freq_range[1])
        f_filtered = f[freq_mask]
        psd_filtered = Pxx[freq_mask]
        
        if frequencies is None:
            frequencies = f_filtered
        
        psd_vectors.append(psd_filtered)
    
    return frequencies, np.array(psd_vectors)


def normalize_psd_for_distance(psd_vectors: np.ndarray, 
                                method: str = 'log_zscore') -> np.ndarray:
    """
    Normalize PSD vectors for distance computation.
    
    Parameters
    ----------
    psd_vectors : np.ndarray
        Shape (n_segments, n_freq_bins)
    method : str
        'log_zscore' : log10 then z-score (for Euclidean/swap test)
        'probability' : normalize to sum to 1 (for trace distance)
        
    Returns
    -------
    normalized : np.ndarray
        Normalized PSD vectors
    """
    if method == 'log_zscore':
        log_psd = np.log10(psd_vectors + 1e-10)
        mean = np.mean(log_psd, axis=1, keepdims=True)
        std = np.std(log_psd, axis=1, keepdims=True) + 1e-10
        return (log_psd - mean) / std
    
    elif method == 'probability':
        # For trace distance: normalize to probability distribution
        psd_positive = np.abs(psd_vectors)
        row_sums = np.sum(psd_positive, axis=1, keepdims=True)
        return psd_positive / (row_sums + 1e-10)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
