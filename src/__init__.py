"""
Quantum-Enhanced TDA for Power Grid Oscillation Detection

A modular pipeline for comparing distance metrics in regime detection.
"""

from .config import (
    ExperimentConfig, 
    SignalConfig, 
    PSDConfig, 
    SegmentationConfig,
    QuantumConfig,
    TDAConfig,
    ModeConfig,
    make_strong_mode_drift_config,
    make_medium_mode_drift_config,
    make_weak_mode_drift_config,
)

from .signal_generation import generate_signal, get_mode_info
from .psd_analysis import compute_welch_psd, segment_signal, compute_psd_matrix
from .distance_metrics import DistanceComputer
from .tda_analysis import compute_h0_persistence, extract_significant_modes
from .regime_detection import full_detection_analysis, DetectionResult
from .experiments import run_experiment, run_three_mode_drift_study, ExperimentResults

__version__ = '0.1.0'
