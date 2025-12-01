"""
Real Data Loader for GESL PMU Signals

Loads synchrophasor data from GESL (Grid Event Signature Library) CSV files.
Extracts single-PMU voltage magnitude streams for oscillation analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass


@dataclass
class GESLSignalInfo:
    """Metadata about a loaded GESL signal."""
    signal_id: int
    num_pmus: int
    fs: float  # sampling rate (Hz)
    duration: float  # seconds
    event_date: str
    oscillation_frequencies: List[float]  # known frequencies from metadata, if any
    pmu_ids: List[str]  # available PMU IDs (P001, P002, etc.)
    

def parse_metadata(metadata_path: str) -> Dict:
    """
    Parse GESL metadata CSV.
    
    Returns dict with: signal_id, num_pmus, fs, duration, event_date, 
                       oscillation_frequencies (if noted)
    """
    df = pd.read_csv(metadata_path)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    row = df.iloc[0]
    
    info = {
        'signal_id': int(row['ID']),
        'fs': float(row['Rate (Hz)']),
        'duration': float(row['Duration (s)']),
        'event_date': str(row['Event Date']).strip(),
    }
    
    # Parse description for PMU count and oscillation frequencies
    desc = str(row['Description'])
    
    # Extract number of PMUs
    if 'Number of measuring PMUs:' in desc:
        import re
        match = re.search(r'Number of measuring PMUs:\s*(\d+)', desc)
        if match:
            info['num_pmus'] = int(match.group(1))
    
    # Extract oscillation frequencies if mentioned
    osc_freqs = []
    if 'Oscillation frequency:' in desc:
        import re
        matches = re.findall(r'(\d+\.?\d*)\s*Hz', desc)
        osc_freqs = [float(f) for f in matches]
    info['oscillation_frequencies'] = osc_freqs
    
    return info

def load_gesl_signal(
    signal_path: str,
    metadata_path: Optional[str] = None,
    pmu_id: str = 'P001',
    measurement: str = 'vp_m',
    detrend: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[GESLSignalInfo]]:
    """
    Load a GESL signal CSV and extract single PMU measurement.
    
    Parameters
    ----------
    signal_path : str
        Path to signal CSV file (e.g., sigId-1032_copy.csv)
    metadata_path : str, optional
        Path to metadata CSV file (e.g., sigId-1032-Metadata.csv)
    pmu_id : str
        PMU to extract (e.g., 'P001', 'P002')
    measurement : str
        Measurement type: 'vp_m' (voltage magnitude), 'f' (frequency), etc.
    detrend : bool
        Remove mean from signal (center around zero)
    normalize : bool
        Normalize to unit variance
        
    Returns
    -------
    t : np.ndarray
        Time vector (seconds)
    signal : np.ndarray
        Extracted signal values
    fs : float
        Sampling frequency (Hz)
    info : GESLSignalInfo, optional
        Metadata if metadata_path provided
    """
    # Load signal data
    df = pd.read_csv(signal_path)
    
    # Extract time
    t = df['Time'].values
    
    # Infer sampling rate from time vector
    dt = np.median(np.diff(t))
    fs = 1.0 / dt
    
    # Build column name
    col_name = f'{pmu_id}.{measurement}'
    
    if col_name not in df.columns:
        available = [c for c in df.columns if measurement in c]
        raise ValueError(f"Column {col_name} not found. Available {measurement} columns: {available}")
    
    signal = df[col_name].values.astype(float)
    
    # Handle any NaN or inf values
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Preprocessing
    if detrend:
        signal = signal - np.mean(signal)
    
    if normalize:
        std = np.std(signal)
        if std > 0:
            signal = signal / std
    
    # Load metadata if provided
    info = None
    if metadata_path:
        meta = parse_metadata(metadata_path)
        
        # Get available PMU IDs from signal file
        pmu_ids = sorted(list(set(
            c.split('.')[0] for c in df.columns if c.startswith('P') and '.' in c
        )))
        
        info = GESLSignalInfo(
            signal_id=meta['signal_id'],
            num_pmus=meta.get('num_pmus', len(pmu_ids)),
            fs=meta['fs'],
            duration=meta['duration'],
            event_date=meta['event_date'],
            oscillation_frequencies=meta['oscillation_frequencies'],
            pmu_ids=pmu_ids,
        )
    
    return t, signal, fs, info


def list_available_pmus(signal_path: str) -> List[str]:
    """List all PMU IDs available in a signal file."""
    df = pd.read_csv(signal_path, nrows=1)
    pmu_ids = sorted(list(set(
        c.split('.')[0] for c in df.columns if c.startswith('P') and '.' in c
    )))
    return pmu_ids


def load_multiple_pmus(
    signal_path: str,
    metadata_path: Optional[str] = None,
    pmu_ids: Optional[List[str]] = None,
    measurement: str = 'vp_m',
    detrend: bool = True,
    normalize: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], float, Optional[GESLSignalInfo]]:
    """
    Load multiple PMU signals from a GESL file.
    
    Returns
    -------
    t : np.ndarray
        Time vector
    signals : Dict[str, np.ndarray]
        Dict mapping PMU ID to signal array
    fs : float
        Sampling frequency
    info : GESLSignalInfo, optional
    """
    # Get available PMUs if not specified
    if pmu_ids is None:
        pmu_ids = list_available_pmus(signal_path)
    
    signals = {}
    t = None
    fs = None
    info = None
    
    for pmu_id in pmu_ids:
        t_pmu, sig, fs_pmu, info_pmu = load_gesl_signal(
            signal_path, 
            metadata_path if info is None else None,  # only parse metadata once
            pmu_id=pmu_id,
            measurement=measurement,
            detrend=detrend,
            normalize=normalize,
        )
        
        if t is None:
            t = t_pmu
            fs = fs_pmu
            info = info_pmu
        
        signals[pmu_id] = sig
    
    return t, signals, fs, info


# Quick diagnostic function
def describe_gesl_file(signal_path: str, metadata_path: str) -> str:
    """Generate a text description of a GESL signal file."""
    pmu_ids = list_available_pmus(signal_path)
    meta = parse_metadata(metadata_path)
    
    lines = [
        f"Signal ID: {meta['signal_id']}",
        f"Event Date: {meta['event_date']}",
        f"Duration: {meta['duration']:.1f} seconds ({meta['duration']/60:.1f} minutes)",
        f"Sampling Rate: {meta['fs']} Hz",
        f"PMUs: {len(pmu_ids)} ({', '.join(pmu_ids)})",
    ]
    
    if meta['oscillation_frequencies']:
        freqs = ', '.join(f"{f} Hz" for f in meta['oscillation_frequencies'])
        lines.append(f"Known Oscillation Frequencies: {freqs}")
    else:
        lines.append("Known Oscillation Frequencies: Not specified")
    
    return '\n'.join(lines)


if __name__ == '__main__':
    # Quick test
    import sys
    if len(sys.argv) >= 3:
        print(describe_gesl_file(sys.argv[1], sys.argv[2]))