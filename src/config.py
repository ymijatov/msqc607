"""
Configuration for Quantum-Enhanced TDA Experiments

All experiment parameters in one place.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np


@dataclass
class ModeConfig:
    """Configuration for a single oscillation mode."""
    freq: float          # Hz
    damping: float       # damping ratio (zeta)
    amplitude: float     # relative amplitude
    
    def to_dict(self) -> Dict:
        return {'freq': self.freq, 'damping': self.damping, 'amplitude': self.amplitude}


@dataclass 
class SignalConfig:
    """Configuration for synthetic signal generation."""
    duration: float = 180.0      # seconds
    fs: float = 30.0             # sampling frequency (Hz)
    noise_level: float = 0.02    # additive noise std
    
    # Default three-mode structure
    modes: List[ModeConfig] = field(default_factory=lambda: [
        ModeConfig(freq=0.5, damping=0.03, amplitude=0.25),   # Strong mode
        ModeConfig(freq=1.2, damping=0.03, amplitude=0.25),   # Medium mode  
        ModeConfig(freq=2.5, damping=0.03, amplitude=0.20),   # Weak mode
    ])
    
    # Drift parameters
    transition_time: float = 90.0    # seconds - when drift begins
    drift_amount: float = 0.5        # Hz - total frequency drift
    drifting_mode_index: int = 2     # which mode drifts (0, 1, or 2)
    
    @property
    def t(self) -> np.ndarray:
        """Time vector."""
        return np.arange(0, self.duration, 1/self.fs)
    
    @property
    def n_samples(self) -> int:
        return int(self.duration * self.fs)


@dataclass
class PSDConfig:
    """Configuration for PSD computation."""
    nperseg: int = 256           # Welch segment length
    noverlap: int = 128          # Welch overlap
    freq_range: tuple = (0.1, 5.0)  # Hz - frequency range of interest
    
    @property
    def freq_resolution(self) -> float:
        """Frequency resolution in Hz (assuming fs=30)."""
        return 30.0 / self.nperseg  # Δf = fs / nperseg


@dataclass
class SegmentationConfig:
    """Configuration for temporal segmentation."""
    segment_duration: float = 15.0   # seconds per segment
    overlap_ratio: float = 0.0       # overlap between segments
    
    def n_segments(self, signal_duration: float, fs: float) -> int:
        """Calculate number of segments."""
        segment_samples = int(self.segment_duration * fs)
        hop = int(segment_samples * (1 - self.overlap_ratio))
        total_samples = int(signal_duration * fs)
        return (total_samples - segment_samples) // hop + 1


@dataclass
class QuantumConfig:
    """Configuration for quantum computations."""
    shots: int = 1024            # measurement shots for sampling
    
    
@dataclass
class TDAConfig:
    """Configuration for TDA (H0 persistence)."""
    persistence_threshold_percentile: float = 80.0  # percentile for significance


@dataclass
class ExperimentConfig:
    """Master configuration for an experiment."""
    signal: SignalConfig = field(default_factory=SignalConfig)
    psd: PSDConfig = field(default_factory=PSDConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    quantum: QuantumConfig = field(default_factory=QuantumConfig)
    tda: TDAConfig = field(default_factory=TDAConfig)
    
    # Experiment parameters
    n_trials: int = 20
    seed_base: int = 42
    
    # Which distance methods to compare
    methods: List[str] = field(default_factory=lambda: [
        'classical',              # Euclidean (L2)
        'swap_test',              # Quantum swap test (L2)
        'trace_distance_classical',  # Classical trace distance (L1)
        'trace_distance_quantum',    # Quantum-sampled trace distance (L1 + noise)
    ])
    
    method_display_names: Dict[str, str] = field(default_factory=lambda: {
        'classical': 'Euclidean (L2)',
        'swap_test': 'Swap Test (L2)',
        'trace_distance_classical': 'Trace Classical (L1)',
        'trace_distance_quantum': 'Trace Quantum (L1+noise)',
    })


# Pre-configured experiments for the three-mode-drift study
def make_strong_mode_drift_config() -> ExperimentConfig:
    """Experiment where the STRONG mode (0.5 Hz) drifts."""
    config = ExperimentConfig()
    config.signal.drifting_mode_index = 0
    return config


def make_medium_mode_drift_config() -> ExperimentConfig:
    """Experiment where the MEDIUM mode (1.2 Hz) drifts."""
    config = ExperimentConfig()
    config.signal.drifting_mode_index = 1
    return config


def make_weak_mode_drift_config() -> ExperimentConfig:
    """Experiment where the WEAK mode (2.5 Hz) drifts."""
    config = ExperimentConfig()
    config.signal.drifting_mode_index = 2
    return config
