# Distance Metric Geometry for Regime Detection in Synchrophasor Data

**Comparing Classical and Quantum-Inspired Approaches**

Yekaterina Mijatovic • University of Maryland • MSQC 607

---

## Overview

This codebase implements a pipeline for detecting regime changes and oscillation modes in power grid synchrophasor data using distance metrics and topological data analysis (TDA). The central finding is that **metric geometry (L1 vs L2) matters more than quantum vs classical computation**.

### Key Features

- **Four distance metrics**: Euclidean (L2), Swap Test (quantum L2), Trace Distance (L1), Quantum-sampled Trace Distance
- **TDA mode detection**: H0 persistent homology on PSD functions identifies oscillation modes without manual tuning
- **Synthetic experiments**: Controlled three-mode drift studies for method comparison
- **Real data analysis**: GESL (Grid Event Signature Library) signal benchmarking

### Main Findings

| What We Tested | What We Found |
|----------------|---------------|
| L1 vs L2 | Detects different event types  -  complementary, not competing |
| Quantum vs Classical | Same distances, added noise  -  no accuracy benefit |
| TDA mode detection | 0.036 Hz mean error against known frequencies |

---

## Installation

### Requirements

```bash
Python 3.10+
numpy
scipy
pandas
matplotlib
qiskit >= 1.0
gudhi
```

### Install

```bash
pip install numpy scipy pandas matplotlib qiskit gudhi
```

---

## Quick Start

### Run synthetic experiments

```bash
# Full three-mode drift study (20 trials each)
python run_pipeline.py

# Quick test (5 trials)
python run_pipeline.py --quick

# Single drift type
python run_pipeline.py --weak-only
python run_pipeline.py --medium-only
python run_pipeline.py --strong-only

# Custom trial count
python run_pipeline.py --trials 50
```

### Run real data analysis (GESL)

```bash
# Analyze a GESL signal
python real_data_experiments.py data/sigId-1015.csv data/sigId-1015-Metadata.csv

# Or in Python:
from real_data_experiments import quick_analysis, compare_with_known_frequencies

results = quick_analysis('data/sigId-1015.csv', 'data/sigId-1015-Metadata.csv')
compare_with_known_frequencies(results)
```

---

## Project Structure

```
├── run_pipeline.py          # Main entry point for synthetic experiments
├── real_data_experiments.py # Entry point for GESL analysis
├── config.py                # All configuration parameters
├── signal_generation.py     # Synthetic signal generation
├── psd_analysis.py          # PSD computation (Welch's method)
├── distance_metrics.py      # L1, L2, quantum distance implementations
├── tda_analysis.py          # H0 persistent homology for mode detection
├── regime_detection.py      # Threshold-based regime change detection
├── visualization.py         # Plotting functions
├── real_data.py             # GESL data loading utilities
├── test_real.py             # Test script for real data
├── figures/                  # Output figures
│   ├── three_mode_study_summary.png
│   ├── sigId-*-benchmark.png
│   └── ...
└── data/                     # GESL signal files (not included)
    ├── sigId-1015.csv
    ├── sigId-1015-Metadata.csv
    └── ...
```

---

## Distance Metrics

### L2 Family (Euclidean Geometry)

**What it detects**: "Did it get loud?"  -  amplitude excitation events, sudden power bursts

| Method | Description |
|--------|-------------|
| `classical` | Standard Euclidean distance between PSD vectors |
| `swap_test` | Quantum circuit estimates inner product → derives L2 + shot noise |

### L1 Family (Trace Distance Geometry)

**What it detects**: "Did the shape change?"  -  frequency redistribution, mode onset

| Method | Description |
|--------|-------------|
| `trace_distance_classical` | Normalize PSDs to probabilities, compute total variation |
| `trace_distance_quantum` | Amplitude encode, sample, compute trace distance + shot noise |

---

## Configuration

All parameters are controlled via `config.py`:

```python
from config import ExperimentConfig, make_weak_mode_drift_config

# Default configuration
config = ExperimentConfig()

# Or use pre-built configurations
config = make_strong_mode_drift_config()  # 0.5 Hz mode drifts
config = make_medium_mode_drift_config()  # 1.2 Hz mode drifts
config = make_weak_mode_drift_config()    # 2.5 Hz mode drifts

# Customize
config.n_trials = 50
config.signal.duration = 300.0
config.quantum.shots = 2048
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `signal.duration` | 180s | Signal length |
| `signal.fs` | 30 Hz | Sampling frequency |
| `signal.transition_time` | 90s | When mode drift begins |
| `segmentation.segment_duration` | 15s | Segment length for PSD |
| `quantum.shots` | 1024 | Measurement shots |
| `tda.persistence_threshold_percentile` | 80 | Mode significance threshold |

---

## GESL Data

This pipeline is validated on real PMU recordings from [PNNL's Grid Event Signature Library](https://www.pnnl.gov/projects/gesl).

### Signals Used

| Signal ID | Duration | Known Frequencies | Notes |
|-----------|----------|-------------------|-------|
| 1015 | 5 min | 0.55 Hz | Inter-area mode |
| 1032 | 7 min | 0.20, 0.55, 0.75 Hz | Multiple modes |
| 1058 | 10 min | 0.75, 1.50, 0.10 Hz | Sustained oscillation |
| 1084 | 13 min | 0.60, 0.75 Hz | Burst + sustained |
| 1085 | 14 min | 0.10 Hz | Low-frequency mode |
| 1232 | 12 min | 0.75 Hz | Amplitude variation |

### Data Format

GESL signals should be CSV files with columns for each PMU measurement. Metadata files contain event information and known oscillation frequencies.

---

## Output Figures

### Synthetic Experiments

- `three_mode_study_summary.png`  -  Comparison across strong/medium/weak drift scenarios
- `{drift_type}_mode_comparison.png`  -  Detailed view per scenario
- `{drift_type}_mode_correlation.png`  -  Method correlation analysis

### Real Data Benchmarks

- `sigId-{id}-benchmark.png`  -  Five-panel analysis:
  1. Raw signal with regime change markers
  2. Spectrogram with known oscillation frequencies
  3. TDA mode detection over time
  4. L2 cumulative drift (Euclidean vs Swap Test)
  5. L1 cumulative drift (Trace Classical vs Trace Quantum)

---

## API Reference

### Signal Generation

```python
from signal_generation import generate_signal, get_mode_info
from config import SignalConfig

config = SignalConfig()
t, signal = generate_signal(config, seed=42)
print(get_mode_info(config))
```

### Distance Computation

```python
from distance_metrics import DistanceComputer
from config import QuantumConfig

computer = DistanceComputer(method='swap_test', config=QuantumConfig())
D = computer.compute_distance_matrix(psd_matrix)
```

### TDA Mode Detection

```python
from tda_analysis import get_modes_per_segment, count_modes_per_segment
from config import TDAConfig

modes = get_modes_per_segment(psd_matrix, frequencies, TDAConfig())
counts = count_modes_per_segment(psd_matrix, frequencies, TDAConfig())
```

### Regime Detection

```python
from regime_detection import detect_regime_changes, compute_cumulative_drift

changes = detect_regime_changes(distance_matrix, threshold_sigma=1.0)
drift = compute_cumulative_drift(distance_matrix)
```

---

## Citation

If you use this code, please cite:

```
Mijatovic, Y. (2025). Distance Metric Geometry for Regime Detection in 
Synchrophasor Data: Comparing Classical and Quantum-Inspired Approaches.
University of Maryland, MSQC 607.
```

### Related Work

- Mishra, S., & Vanfretti, L. (2025). Automatically Discerning Power System Dynamics in Synchrophasor Measurements Data Spectra. *IJEPES*, 170.

---

## License

MIT License

---

## Acknowledgments

- PNNL for the Grid Event Signature Library
- University of Maryland MSQC program and Dr. P. Aaron Lott's guidance throughout the project