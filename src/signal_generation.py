"""
Signal Generation for Quantum-Enhanced TDA

Synthetic synchrophasor signals with configurable mode drift.
"""

import numpy as np
from typing import List, Optional
from config import SignalConfig, ModeConfig


def generate_signal(config: SignalConfig, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate synthetic synchrophasor signal with optional mode drift.
    
    Parameters
    ----------
    config : SignalConfig
        Signal configuration including modes, drift parameters
    seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    signal : np.ndarray
        Time series signal
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = config.t
    y = np.zeros_like(t)
    
    for i, mode in enumerate(config.modes):
        freq = mode.freq
        zeta = mode.damping
        A = mode.amplitude
        
        # Apply drift to specified mode
        if i == config.drifting_mode_index:
            freq_array = _apply_drift(
                t, 
                base_freq=freq,
                transition_time=config.transition_time,
                drift_amount=config.drift_amount
            )
            omega = 2 * np.pi * freq_array
        else:
            omega = 2 * np.pi * freq
        
        # Damped oscillation
        decay = np.exp(-zeta * omega * t)
        oscillation = A * decay * np.sin(omega * t)
        y += oscillation
    
    # Add noise
    noise = config.noise_level * np.random.randn(len(t))
    y += noise
    
    return y


def _apply_drift(t: np.ndarray, base_freq: float, 
                 transition_time: float, drift_amount: float) -> np.ndarray:
    """
    Apply linear frequency drift after transition time.
    
    Before transition: freq = base_freq
    After transition: freq increases linearly to base_freq + drift_amount
    """
    freq_array = np.where(
        t < transition_time,
        base_freq,
        base_freq + drift_amount * (t - transition_time) / (t[-1] - transition_time)
    )
    return freq_array


def generate_stationary_signal(config: SignalConfig, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate signal with NO drift (control case).
    """
    # Temporarily disable drift
    original_drift = config.drift_amount
    config.drift_amount = 0.0
    
    signal = generate_signal(config, seed)
    
    # Restore
    config.drift_amount = original_drift
    
    return signal


def get_mode_info(config: SignalConfig) -> str:
    """Get human-readable description of signal configuration."""
    drifting = config.modes[config.drifting_mode_index]
    lines = [
        f"Signal Configuration:",
        f"  Duration: {config.duration}s, fs={config.fs} Hz",
        f"  Modes:",
    ]
    for i, mode in enumerate(config.modes):
        drift_marker = " ← DRIFTS" if i == config.drifting_mode_index else ""
        lines.append(f"    [{i}] {mode.freq} Hz, amplitude={mode.amplitude}{drift_marker}")
    
    lines.append(f"  Transition at t={config.transition_time}s, drift={config.drift_amount} Hz")
    
    return "\n".join(lines)
